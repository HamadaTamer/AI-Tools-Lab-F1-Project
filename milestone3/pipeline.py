"""One chat turn from user message to logged answer.

Pipeline: classify the intent (rules, then follow-up check, then embeddings) -> extract entities
-> resolve or ask a clarification -> merge with the session cache -> refuse / hand over to the
prediction view / retrieve from Neo4j -> ask every selected LLM with the same prompt and rows
-> log one JSON line. `handle_turn` returns a `Turn` holding everything the UI shows and the log keeps.

Follow-up rule (spec 1.b): if the rule classifier finds an intent, that intent is used. Otherwise
a message that starts with "and / what about / how about ...", or is short (<= 6 words) and
names an entity, keeps the cached intent; anything else goes to the embedding classifier.
Self-check without Neo4j or Groq: python -m milestone3.tests.check_pipeline
"""
import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime

from neo4j.exceptions import DriverError, Neo4jError

from milestone3 import config
from milestone3.llm.client import LLMResult, complete
from milestone3.llm.fallback import NO_DATA, read_rows
from milestone3.llm.prompts import answer_messages
from milestone3.nlu.entities import Extraction
from milestone3.nlu.intents import INTENTS
from milestone3.nlu.rule_classifier import classify_rule
from milestone3.retrieval.graph import missing_slots
from milestone3.retrieval.templates import REQUIRED
from milestone3.session import Session

REFUSAL = "I don't have the knowledge to answer that. I can only answer Formula 1 questions the knowledge graph covers."
REPHRASE = "I could not tell what you are asking. Could you rephrase the question?"
GRAPH_DOWN = "The knowledge graph (Neo4j) is not reachable. Start it and try again."
FOLLOW_UP = re.compile(r"^(and|what about|how about|also|then|same for)\b", re.I)
SHORT_WORDS = 6
SLOT_NAMES = {"driver": "driver", "driver2": "second driver", "constructor": "team", "circuit": "circuit",
              "race_name": "Grand Prix", "year": "season"}

# Feature values a user may state in a prediction question: feature -> patterns (first number found).
_POS = r"\s*(?:p|pos(?:ition)?\.?\s*)?(\d{1,2})(?:st|nd|rd|th)?\b"
VALUE_PATTERNS = {
    "grid_pos": [r"\b(?:start(?:s|ed|ing)?|grid|lin(?:es|ed|ing) up)\s*(?:from|in|at|on)?" + _POS,
                 r"\bp(\d{1,2}) on the grid"],
    "quali_pos": [r"\bqualif(?:y|ies|ied|ying)\s*(?:in|at|on)?" + _POS,
                  r"(\d{1,2})(?:st|nd|rd|th) in qualifying"],
    "drv_standing": [r"(\d{1,2})(?:st|nd|rd|th) in the (?:drivers'?s? )?(?:championship|standings)"],
    "con_standing": [r"(\d{1,2})(?:st|nd|rd|th) in the constructors'?s? (?:championship|standings)"],
}
RACE_AT_CIRCUIT = """
MATCH (ra:Race {year: $year})-[:AT_CIRCUIT]->(:Circuit {circuitId: $circuitId})
RETURN ra.name AS race_name ORDER BY ra.date LIMIT 1
"""
POLE = re.compile(r"\b(?:from|on) pole\b", re.I)
PIT_LANE = re.compile(r"pit[- ]?lane start|from the pit[- ]?lane", re.I)


@dataclass
class Answer:
    """One model's column: its text (or the deterministic fallback) and the API-error note."""
    model: str
    text: str
    note: str | None        # config.API_ERROR when the call failed, else None
    result: LLMResult


@dataclass
class Turn:
    text: str
    kind: str = "answer"    # answer | no_data | clarify | refuse | rephrase | predict | error
    message: str = ""       # the assistant's own words for every kind except "answer"
    intent: str | None = None
    how: str | None = None  # which classifier decided: rule | follow-up | embedding | clarification
    found: dict = field(default_factory=dict)      # entities named in this message
    entities: dict = field(default_factory=dict)   # slots actually sent to retrieval (cache merged)
    retrieval: object = None
    prompt: list | None = None
    answers: list = field(default_factory=list)
    prefill: dict | None = None    # driver / race / year / feature values for the prediction view


# ---------------------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------------------
def feature_values(text: str) -> dict:
    """Pre-race feature values stated in the message ("starting P3" -> grid_pos 3)."""
    values = {}
    for name, patterns in VALUE_PATTERNS.items():
        for pattern in patterns:
            if match := re.search(pattern, text, re.I):
                values[name] = int(match.group(1))
                break
    if POLE.search(text):
        values.setdefault("grid_pos", 1)
    if PIT_LANE.search(text):
        values["pit_lane_start"] = 1
    return values


def choose_intent(text, session, extraction, classifier):
    """(intent, how) for a new message: rules, then the follow-up rule, then embeddings."""
    if intent := classify_rule(text):
        return intent, "rule"
    names_entity = extraction.entities or extraction.ambiguous
    if session.intent and (FOLLOW_UP.match(text) or (names_entity and len(text.split()) <= SHORT_WORDS)):
        return session.intent, "follow-up"
    return classifier.classify(text).intent, "embedding"


def clarification(slot, candidates) -> str:
    """'Which Schumacher do you mean: Michael, Ralf or Mick?' (full names when surnames differ)."""
    surnames = {name.split()[-1] for name in candidates}
    if slot.startswith("driver") and len(surnames) == 1:
        label, options = surnames.pop(), [name.split()[0] for name in candidates]
    else:
        label, options = SLOT_NAMES.get(slot, slot), candidates
    return f"Which {label} do you mean: {', '.join(options[:-1])} or {options[-1]}?"


def resolve_pending(text, session, extractor):
    """Apply the user's reply to the open clarification; None if it names none of the candidates."""
    pending = session.pending
    slot, candidates = next(iter(pending["ambiguous"].items()))
    words = set(re.findall(r"\w+", text.lower()))
    matches = [c for c in candidates if words & set(re.findall(r"\w+", c.lower())) and
               words <= set(re.findall(r"\w+", c.lower())) | {"the", "one", "i", "mean"}]
    if len(matches) != 1:
        return None
    chosen = next(iter(extractor.extract(matches[0]).entities.values()), None)
    if not isinstance(chosen, dict):
        return None
    rest = {s: c for s, c in pending["ambiguous"].items() if s != slot}
    extraction = Extraction({**pending["extraction"].entities, slot: chosen}, rest)
    return pending["intent"], extraction


def usable_slots(intent, cache, found, intent_changed) -> dict:
    """Slots sent to retrieval: everything the message named, plus cached slots the intent can use.
    After an explicit change of intent only the *required* slots are carried over, so a stale
    optional filter (e.g. a driver on a pit-stop question) does not silently narrow the answer."""
    required = {slot for options in REQUIRED.get(intent, []) for slot in options}
    carry = required if intent_changed else required | set(INTENTS[intent].optional)
    return {slot: value for slot, value in cache.items() if slot in found or slot in carry}


def log_turn(turn: Turn):
    """Append one JSON line describing the turn (clarification and refusal turns have nulls)."""
    r = turn.retrieval
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "user_message": turn.text, "kind": turn.kind, "intent": turn.intent, "classifier": turn.how,
        "entities": turn.entities, "template": r and r.template, "cypher": r and r.cypher,
        "params": r and r.params, "rows": r and r.rows, "prompt": turn.prompt,
        "models": [{"model": a.model, "answer": a.text, "latency_s": round(a.result.latency_s, 2),
                    "input_tokens": a.result.input_tokens, "output_tokens": a.result.output_tokens,
                    "error": a.result.error} for a in turn.answers],
    }
    write_log(record)


def log_prediction(values, prediction, facts, messages, llm: LLMResult):
    """Append one JSON line for a prediction explanation (the prediction view's LLM call)."""
    write_log({
        "timestamp": datetime.now().isoformat(timespec="seconds"), "kind": "prediction",
        "features": values, "probability": prediction["probability"], "threshold": prediction["threshold"],
        "contributions": prediction["contributions"], "graph_facts": facts, "prompt": messages,
        "models": [{"model": llm.model, "answer": llm.text, "latency_s": round(llm.latency_s, 2),
                    "input_tokens": llm.input_tokens, "output_tokens": llm.output_tokens, "error": llm.error}],
    })


def write_log(record: dict):
    config.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(config.LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


# ---------------------------------------------------------------------------------------
# The turn
# ---------------------------------------------------------------------------------------
def answer_from_graph(turn, graph, models):
    """Retrieve, then ask every selected model with the same few-shot prompt and the same rows."""
    turn.retrieval = graph.retrieve(turn.intent, turn.entities)
    if not turn.retrieval.rows:
        turn.kind, turn.message = "no_data", NO_DATA
        return
    turn.prompt = answer_messages(turn.text, turn.intent, turn.entities, turn.retrieval,
                                  few_shot=True, variant=config.ANSWER_VARIANT)
    with ThreadPoolExecutor() as pool:   # both models are called at the same time
        results = list(pool.map(lambda m: complete(m, turn.prompt), models))
    for result in results:
        failed = result.error is not None
        text = read_rows(turn.intent, turn.retrieval) if failed else result.text
        turn.answers.append(Answer(result.model, text, config.API_ERROR if failed else None, result))


def run_turn(turn, session, graph, extractor, classifier, models):
    """Fill `turn` and update the session; raises only on graph connection problems."""
    # 1. a reply to an open clarification, otherwise a fresh message
    resolved = resolve_pending(turn.text, session, extractor) if session.pending else None
    if resolved:
        (turn.intent, extraction), turn.how = resolved, "clarification"
    else:
        extraction = extractor.extract(turn.text)
        turn.intent, turn.how = choose_intent(turn.text, session, extraction, classifier)
    session.pending = None

    # 2. no intent -> rephrase; out of scope -> fixed refusal (the cache is left untouched)
    if turn.intent is None:
        turn.kind, turn.message = "rephrase", REPHRASE
        return
    if turn.intent == "out_of_scope":
        turn.kind, turn.message = "refuse", REFUSAL
        return

    # 3. an ambiguous name is asked about first; the rest of the message waits in the session
    if extraction.ambiguous:
        slot, candidates = next(iter(extraction.ambiguous.items()))
        session.pending = {"intent": turn.intent, "extraction": extraction, "ambiguous": extraction.ambiguous}
        turn.kind, turn.message = "clarify", clarification(slot, candidates)
        return

    # 4. merge into the cache: only the slots named now are replaced. In a head-to-head follow-up
    #    ("And against Rosberg?") a single new driver is the new opponent, so it replaces driver2.
    found = extraction.entities
    if turn.intent == session.intent == "head_to_head" and "driver" in found and "driver2" not in found:
        found["driver2"] = found.pop("driver")
    changed = turn.intent != session.intent
    session.merge(extraction.entities)
    session.intent = turn.intent
    turn.found = extraction.entities

    # 5. a prediction question goes to the prediction view with whatever is known
    if turn.intent == "podium_prediction":
        cache = session.entities
        turn.entities = {s: cache[s] for s in ("driver", "race_name", "year") if s in cache}
        # "at Monza in 2022": the prediction view needs the Grand Prix name, so find it from circuit + year
        if "circuit" in cache and "year" in cache and ("race_name" not in turn.entities or "circuit" in turn.found):
            rows = graph.run(RACE_AT_CIRCUIT, {"circuitId": cache["circuit"]["circuitId"], "year": cache["year"]})
            if rows:
                turn.entities["race_name"] = rows[0]["race_name"]
        turn.prefill = {**turn.entities, "values": feature_values(turn.text)}
        turn.kind, turn.message = "predict", "Opening the podium prediction view with the details I found."
        return

    # 6. required slots still missing -> ask; otherwise retrieve and answer
    turn.entities = usable_slots(turn.intent, session.entities, turn.found, changed)
    if missing := missing_slots(turn.intent, turn.entities):
        names = [" or ".join(SLOT_NAMES[s] for s in m.split(" or ")) for m in missing]
        turn.kind, turn.message = "clarify", f"Which {' and which '.join(names)} do you mean?"
        return
    answer_from_graph(turn, graph, models)


def handle_turn(text, session: Session, graph, extractor, classifier, models: list[str]) -> Turn:
    """Process one user message, log it, and return what the UI should show."""
    turn = Turn(text=text)
    try:
        run_turn(turn, session, graph, extractor, classifier, models)
    except (DriverError, Neo4jError, OSError):
        turn.kind, turn.message = "error", GRAPH_DOWN
    session.history.append(turn)
    log_turn(turn)
    return turn
