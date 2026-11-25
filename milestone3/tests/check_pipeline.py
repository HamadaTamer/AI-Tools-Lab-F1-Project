"""Self-check of the conversation logic in `pipeline.py` with a fake graph and a fake LLM.

Needs neither Neo4j nor Groq. It covers the session merge rules (follow-ups, range vs year,
pronouns), refusals, the clarification flow, missing slots, empty rows, the API-error
fallback, the prediction hand-over and the turn log.
Run from the repository root: python -m milestone3.tests.check_pipeline
"""
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from milestone3 import config, pipeline
from milestone3.llm.client import LLMResult
from milestone3.llm.fallback import NO_DATA
from milestone3.nlu.entities import EntityExtractor
from milestone3.pipeline import feature_values, handle_turn
from milestone3.retrieval.graph import Retrieval
from milestone3.session import Session


def main():
    # fake look-up tables: Schumacher is ambiguous on purpose (three drivers)
    def driver(i, first, last, a, b):
        return {"driverId": i, "forename": first, "surname": last, "code": None, "driverRef": last.lower(),
                "first_year": a, "last_year": b}
    tables = {"drivers": [driver(1, "Lewis", "Hamilton", 2007, 2024), driver(2, "Lando", "Norris", 2019, 2024),
                          driver(3, "Michael", "Schumacher", 1991, 2012), driver(4, "Ralf", "Schumacher", 1997, 2007),
                          driver(5, "Mick", "Schumacher", 2021, 2022)],
              "constructors": [], "races": ["Monaco Grand Prix", "Italian Grand Prix"],
              "circuits": [{"circuitId": 14, "name": "Autodromo Nazionale di Monza", "location": "Monza",
                            "country": "Italy", "circuitRef": "monza"}]}
    extractor = EntityExtractor(tables)
    classifier = SimpleNamespace(classify=lambda text: SimpleNamespace(intent=None))
    calls = []
    empty = {"on": False}

    # fake graph and LLM: they record calls and can be switched to 'no rows' / 'API error'
    class FakeGraph:
        def retrieve(self, intent, entities):
            calls.append((intent, dict(entities)))
            rows = [] if empty["on"] else [{"wins": 3}]
            return Retrieval("fake", "MATCH ...", {}, rows, None, "scope")

    broken = {"on": False}

    def fake_complete(model, messages):
        err = config.API_ERROR if broken["on"] else None
        return LLMResult(model, None if err else "ok", err, 0.1, 5, 5)

    pipeline.complete = fake_complete   # every LLM call in the pipeline now goes to the fake
    config.LOG_FILE = Path(tempfile.mkdtemp()) / "turns.jsonl"

    def say(session, text, models=("m1", "m2")):
        return handle_turn(text, session, FakeGraph(), extractor, classifier, list(models))

    def slots(turn):
        return {k: (v["name"] if isinstance(v, dict) else v) for k, v in turn.entities.items()}

    s = Session()
    t = say(s, "How many wins did Hamilton have in 2019?")
    assert (t.intent, slots(t)) == ("driver_season_stats", {"driver": "Lewis Hamilton", "year": 2019}), t
    assert [a.model for a in t.answers] == ["m1", "m2"] and t.answers[0].note is None

    t = say(s, "And in 2020?")                     # keeps intent and driver, replaces the season
    assert (t.intent, t.how, slots(t)) == ("driver_season_stats", "follow-up", {"driver": "Lewis Hamilton", "year": 2020})
    t = say(s, "What about Norris?")               # keeps the season, replaces the driver
    assert slots(t) == {"driver": "Lando Norris", "year": 2020}, slots(t)
    t = say(s, "How many points did he score in 2019?")   # pronoun reuses the cached driver
    assert slots(t) == {"driver": "Lando Norris", "year": 2019}
    t = say(s, "2021?")
    assert slots(t) == {"driver": "Lando Norris", "year": 2021}
    t = say(s, "since 2015?")                      # a range replaces the single year
    assert slots(t) == {"driver": "Lando Norris", "year_from": 2015}, slots(t)
    t = say(s, "2019?")                            # and a single year replaces the range
    assert slots(t) == {"driver": "Lando Norris", "year": 2019}

    n = len(calls)
    t = say(s, "What is the capital of France?")   # refusal: no retrieval, cache kept
    assert t.kind == "refuse" and t.retrieval is None and len(calls) == n
    assert s.intent == "driver_season_stats" and s.entities["driver"]["name"] == "Lando Norris"
    t = say(s, "And in 2018?")
    assert slots(t) == {"driver": "Lando Norris", "year": 2018}

    t = say(s, "Show the fastest pit stops at the Monaco Grand Prix 2019")   # new intent: stale driver not carried
    assert t.intent == "pit_stops" and "driver" not in t.entities, slots(t)

    # clarification flow: an ambiguous surname is asked about, then resolved by the reply
    s = Session()
    t = say(s, "How many wins did Schumacher have?")
    assert t.kind == "clarify" and t.message == "Which Schumacher do you mean: Michael, Ralf or Mick?", t.message
    t = say(s, "Ralf")
    assert t.kind == "answer" and slots(t) == {"driver": "Ralf Schumacher"}, slots(t)
    s = Session()
    say(s, "How many wins did Schumacher have?")
    t = say(s, "Hamilton podiums 2019")             # an unrelated reply drops the question
    assert t.kind == "answer" and s.pending is None and slots(t)["driver"] == "Lewis Hamilton"

    # missing slot, answered by the next message
    s = Session()
    t = say(s, "How many podiums did he get?")
    assert t.kind == "clarify" and "driver" in t.message, t.message
    t = say(s, "Norris")
    assert t.kind == "answer" and slots(t) == {"driver": "Lando Norris"}, slots(t)

    # empty rows: no LLM call; API error: fallback text plus the note
    empty["on"] = True
    t = say(s, "And in 2019?")
    assert t.kind == "no_data" and t.answers == [] and t.message == NO_DATA
    empty["on"], broken["on"] = False, True
    t = say(s, "And in 2020?")
    assert [a.note for a in t.answers] == [config.API_ERROR] * 2 and t.answers[0].text.startswith("Result:")
    broken["on"] = False

    # prediction hand-over with stated feature values
    s = Session()
    t = say(s, "Will Hamilton get a podium at the Italian Grand Prix 2021 starting P3, qualified 2nd?")
    assert t.kind == "predict" and t.prefill["values"] == {"grid_pos": 3, "quali_pos": 2}, t.prefill
    assert t.prefill["race_name"] == "Italian Grand Prix" and t.prefill["year"] == 2021
    assert feature_values("he starts from pole after a pit lane start") == {"grid_pos": 1, "pit_lane_start": 1}

    log = [json.loads(line) for line in config.LOG_FILE.read_text(encoding="utf-8").splitlines()]
    assert all("timestamp" in r and "user_message" in r for r in log) and log[0]["models"][0]["model"] == "m1"
    print(f"all pipeline checks passed ({len(log)} turns logged)")


if __name__ == "__main__":
    main()
