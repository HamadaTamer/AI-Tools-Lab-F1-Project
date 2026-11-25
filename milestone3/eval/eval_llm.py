"""LLM comparison: every model x {zero-shot, few-shot} on the same prompts and the same gold rows.

Usage (from the repo root):
    python -m milestone3.eval.eval_llm [--variant base|strict] [--models M ...] [--limit N]

Gold rows come from running the correct template on the gold entities of qa_testset.json at
evaluation time. Every call is stored in results/llm_runs.jsonl (labelled by prompt variant,
so the before-and-after runs sit side by side); results/llm_comparison.csv and
results/llm_summary.md are rebuilt from that file. Calls already stored are not repeated.

Metrics (all automatic, per answer, then averaged):
- correctness: share of the gold numbers and names in the rows that appear in the answer.
- faithfulness: an answer is faithful when it contains no number and no name-like capitalised
  word that is missing from the context (question, entities, scope, rows).
- refusal: share of no-data items (the graph returned no rows, or the question is out of
  scope) whose answer says the data is insufficient. In the app, out-of-scope questions are
  refused by the pipeline before any LLM call; feeding them to the LLM with empty rows is the
  stricter test of the same behaviour, so it is measured here at the LLM level.
- explanation fidelity: share of the three strongest local contributions (by |SHAP|) named
  in the explanation, and whether any feature outside the context is named.
- latency (seconds) and input / output tokens, averaged over the successful calls.
"""
import argparse
import csv
import json
import re
import time
from collections import defaultdict
from statistics import mean

from milestone3 import config
from milestone3.llm.client import complete
from milestone3.llm.prompts import answer_messages, explanation_messages

RESULTS = config.RESULTS_DIR
RUNS_FILE = RESULTS / "llm_runs.jsonl"
PAUSE_S = 1.0                      # gap between calls, to stay inside the free tier
STYLES = {"zero-shot": False, "few-shot": True}
NO_DATA_WORDS = re.compile(r"insufficient|no data|not enough|no rows|does not contain|"
                           r"doesn't contain|cannot|can't|unable|no information", re.I)

# ---- metric helpers ---------------------------------------------------------------------

def numbers_in(text: str) -> set[str]:
    """Numbers of a text in a canonical form, so that 12.0, 12 and '12' all become '12'."""
    found = re.findall(r"\d[\d,]*(?:\.\d+)?", text)
    return {f"{float(n.replace(',', '')):g}" for n in found}


def words_in(text: str) -> set[str]:
    return {w.lower() for w in re.findall(r"[^\W\d_]+", text)}


NAME_COLUMNS = {"driver", "driver1", "driver2"}   # the names a question can ask for ("Who won ...?")
# question words that ask for a column under another name ("victories" asks for the wins column)
SYNONYMS = {"victories": "wins", "score": "points", "scored": "points",
            "reliable": "reliability", "gained": "places", "fastest": "seconds", "quickest": "seconds"}


def stems(words) -> set[str]:
    return {w[:3] for w in words if len(w) >= 3}       # 3 letters: "win" matches "wins"


def gold_facts(question: str, rows: list[dict]) -> tuple[set[str], set[str]]:
    """The facts the question asks for, taken from the TOP row (rows 2-5 of a top-5 list are context).
    - numbers: the columns whose name matches a question word ("How many wins" -> wins); if none is
      named, every numeric column except years/dates (they only echo the period asked about), unless
      the question is a "who" question answered by a name;
    - names: the driver of the top row when the question does not already name them, unless the
      question asked for a specific number and is not a "who" question. Compared by surname."""
    top = rows[0]
    asked = stems(SYNONYMS.get(w, w) for w in words_in(question))
    numeric = {k: v for k, v in top.items() if isinstance(v, (int, float)) and not isinstance(v, bool)
               and not k.lower().endswith("id") and "year" not in k}
    names = {str(top[k]).split()[-1].lower() for k in NAME_COLUMNS & top.keys() if top[k]} - words_in(question)
    named = {k: v for k, v in numeric.items() if stems(k.lower().split("_")) & asked}
    who = "who" in words_in(question)
    if named and not who:
        names = set()
    elif not named and not (names and who):
        named = numeric
    return {f"{float(v):g}" for v in named.values()}, names


def correctness(answer: str, question: str, rows: list[dict]) -> float | None:
    """Share of the asked-for gold numbers and names that the answer mentions (None if there are none)."""
    nums, names = gold_facts(question, rows)
    if not nums and not names:
        return None
    hit = len(nums & numbers_in(answer)) + len(names & words_in(answer))
    return hit / (len(nums) + len(names))


def capitalised_names(answer: str) -> list[str]:
    """Capitalised words that are probably names: every capitalised word inside a sentence,
    plus a sentence's first word only when the next word is capitalised too (a full name).
    (A lone name opening a sentence is not checked; the first word is usually 'The', 'In'...)"""
    names = []
    for sentence in re.split(r"(?<=[.!?])\s+", answer):
        words = re.findall(r"[^\W\d_]+", sentence)
        caps = [w[0].isupper() for w in words]
        names += [w for w, c in zip(words[1:], caps[1:]) if c and len(w) > 1]   # skip the pronoun "I"
        if len(words) > 1 and caps[0] and caps[1]:
            names.append(words[0])
    return names


def is_faithful(answer: str, context: str) -> bool:
    """True when the answer has no number or name-like word that the context lacks."""
    if numbers_in(answer) - numbers_in(context):
        return False
    context_words = words_in(context)
    return all(w.lower() in context_words for w in capitalised_names(answer))


def explanation_scores(answer: str, top3: list[str], shown: list[str], vocabulary: dict) -> tuple:
    """(share of the top-3 features named, True if no feature outside the context is named).
    `vocabulary` maps every known feature to the phrases that name it."""
    def named(feature):
        return any(re.search(rf"\b{re.escape(p)}\b", answer, re.I) for p in vocabulary[feature])
    recall = sum(named(f) for f in top3) / len(top3)
    return recall, not any(named(f) for f in vocabulary if f not in shown)


# ---- building the inputs ----------------------------------------------------------------

def qa_items(limit):
    """Test items with their gold rows: the gold template is run on the gold entities."""
    from milestone3.retrieval.graph import Graph, Retrieval
    graph = Graph()
    items = json.load(open(config.EVAL_DIR / "qa_testset.json", encoding="utf-8"))
    out = []
    for item in items:
        # prediction and ambiguous items never reach the answer prompt (prediction view / clarifying question)
        if item["intent"] == "podium_prediction" or item.get("ambiguous"):
            continue
        if item["intent"] == "out_of_scope":      # nothing to retrieve: empty rows
            retrieval = Retrieval("none", "", {}, [], None, "No data was retrieved.")
        else:
            retrieval = graph.retrieve(item["intent"], item["entities"])
        out.append((item, retrieval))
    return out[:limit] if limit else out


def vocabulary_of_features(names: list[str]) -> dict:
    """Phrases that name each feature: its code name, the name with spaces, and its label."""
    labels = {}
    try:
        from milestone3.prediction.features import FEATURES
        labels = {f.name: f.label for f in FEATURES}
        names = list(dict.fromkeys(names + list(labels)))
    except ImportError:
        pass
    return {n: [n, n.replace("_", " ")] + ([labels[n]] if n in labels else []) for n in names}


def explanation_items():
    """Predictions for the explanation cases, with their pre-race graph facts."""
    from milestone3.prediction.explain import pre_race_facts
    from milestone3.prediction.features import prefill
    from milestone3.prediction.predictor import PodiumPredictor
    from milestone3.retrieval.graph import Graph
    graph, predictor = Graph(), PodiumPredictor()
    out = []
    for case in json.load(open(config.EVAL_DIR / "explanation_cases.json", encoding="utf-8")):
        values, _, race = prefill(case["driverId"], case["race_name"], case["year"])
        prediction = predictor.predict(values)
        facts = pre_race_facts(graph, case["driverId"], race.get("constructorId"),
                               race.get("circuitId"), race.get("date"))
        out.append((case, prediction, facts))
    return out


# ---- running the models -----------------------------------------------------------------

def load_runs() -> list[dict]:
    if not RUNS_FILE.exists():
        return []
    return [json.loads(line) for line in open(RUNS_FILE, encoding="utf-8")]


def append_run(run):
    """Append one call to the log. Appending (not rewriting) is safe to resume and avoids
    OneDrive locking a file that is rewritten after every call. Failed calls stay in the file
    but are dropped by run_all on the next start, so they are retried."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RUNS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(run, ensure_ascii=False, default=str) + "\n")


def call(model, messages, base: dict) -> dict:
    """One LLM call, returned as a run record (the base fields plus the call's outcome)."""
    result = complete(model, messages)
    time.sleep(PAUSE_S)
    return {**base, "model": model, "prompt": messages, "answer": result.text,
            "error": result.error, "latency_s": round(result.latency_s, 3),
            "input_tokens": result.input_tokens, "output_tokens": result.output_tokens}


def run_all(variant, models, limit, skip_explain):
    runs = load_runs()
    done = {(r["variant"], r["model"], r["style"], r["id"]) for r in runs if r["answer"]}
    runs = [r for r in runs if r["answer"]]                # failed calls are retried
    jobs = []
    for item, retrieval in qa_items(limit):
        for style, few_shot in STYLES.items():
            messages = answer_messages(item["question"], item["intent"], item["entities"],
                                       retrieval, few_shot, variant)
            base = {"variant": variant, "kind": "qa", "style": style, "id": item["id"],
                    "intent": item["intent"], "category": item["category"],
                    "question": item["question"], "rows": retrieval.rows}
            jobs.append((base, messages))
    if not skip_explain:
        for i, (case, prediction, facts) in enumerate(explanation_items(), 1):
            contribs = sorted(prediction["contributions"], key=lambda c: -abs(c[2]))
            base = {"variant": variant, "kind": "explain", "style": "explain", "id": f"e{i}",
                    "intent": "podium_prediction", "category": "explanation",
                    "question": f"{case['driver']} at the {case['year']} {case['race_name']}",
                    "top3": [c[0] for c in contribs[:3]],
                    "shown": [c[0] for c in contribs if c[2] > 0][:3]
                             + [c[0] for c in contribs if c[2] < 0][:3]
                             + [g["feature"] for g in prediction["global_importance"][:8]]}
            jobs.append((base, explanation_messages(prediction, facts, variant)))
    for base, messages in jobs:
        for model in models:
            if (variant, model, base["style"], base["id"]) not in done:
                runs.append(call(model, messages, base))
                print(f"{variant} {model} {base['style']} {base['id']} "
                      f"{'ok' if runs[-1]['answer'] else 'FAILED'}")
                append_run(runs[-1])                       # keep progress if interrupted
    return runs


# ---- scoring and reporting --------------------------------------------------------------

def score(run: dict, vocabulary: dict) -> dict:
    """Metric values of one successful run."""
    answer = run["answer"]
    if run["kind"] == "explain":
        recall, clean = explanation_scores(answer, run["top3"], run["shown"], vocabulary)
        return {"expl_recall": recall, "expl_no_foreign": clean,
                "expl_fidelity": recall if clean else 0.0}
    context = run["prompt"][-1]["content"]
    out = {"faithful": is_faithful(answer, context)}
    if run["rows"]:
        out["correct"] = correctness(answer, run["question"], run["rows"])
    else:
        out["refusal"] = bool(NO_DATA_WORDS.search(answer))
    return out


def summarise(runs):
    """One row per variant x model x style: mean of every metric, latency and tokens."""
    explained = [r for r in runs if r["kind"] == "explain"]
    vocabulary = vocabulary_of_features(sorted({f for r in explained for f in r["top3"] + r["shown"]}))
    groups = defaultdict(list)
    for run in runs:
        groups[(run["variant"], run["model"], run["style"])].append(run)
    table = []
    for (variant, model, style), group in sorted(groups.items()):
        ok = [r for r in group if r["answer"]]
        scores = [score(r, vocabulary) for r in ok]
        row = {"variant": variant, "model": model, "style": style, "n": len(group),
               "failed": len(group) - len(ok)}
        for metric in ["correct", "faithful", "refusal", "expl_recall", "expl_no_foreign",
                       "expl_fidelity"]:
            values = [float(s[metric]) for s in scores if s.get(metric) is not None]
            row[metric] = round(mean(values), 3) if values else ""
        for field in ["latency_s", "input_tokens", "output_tokens"]:
            row[field] = round(mean(r[field] for r in ok), 2) if ok else ""
        table.append(row)
    return table


def write_reports(table):
    cols = list(table[0])
    with open(RESULTS / "llm_comparison.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, cols)
        writer.writeheader()
        writer.writerows(table)
    lines = ["# LLM comparison", "", "Metrics are defined in `eval/eval_llm.py`. Blank = not applicable.", ""]
    for variant in dict.fromkeys(r["variant"] for r in table):
        rows = [r for r in table if r["variant"] == variant]
        lines += [f"## Prompt variant: {variant}", "", "| " + " | ".join(cols[1:]) + " |",
                  "|" + " --- |" * (len(cols) - 1)]
        lines += ["| " + " | ".join(str(r[c]) for c in cols[1:]) + " |" for r in rows]
        lines += ["", f"### Few-shot minus zero-shot ({variant})", "",
                  "| model | correct | faithful | refusal | latency_s | input_tokens |", "| --- | --- | --- | --- | --- | --- |"]
        for model in dict.fromkeys(r["model"] for r in rows):
            zero = next((r for r in rows if r["model"] == model and r["style"] == "zero-shot"), None)
            few = next((r for r in rows if r["model"] == model and r["style"] == "few-shot"), None)
            if zero and few:
                diff = lambda c: (f"{few[c] - zero[c]:+.3f}" if zero[c] != "" and few[c] != "" else "")
                lines.append(f"| {model} | " + " | ".join(diff(c) for c in
                             ["correct", "faithful", "refusal", "latency_s", "input_tokens"]) + " |")
        lines.append("")
    (RESULTS / "llm_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="base", choices=["base", "strict"])
    parser.add_argument("--models", nargs="+", default=config.LLM_MODELS)
    parser.add_argument("--limit", type=int, help="only the first N qa items (quick test)")
    parser.add_argument("--skip-explain", action="store_true")
    args = parser.parse_args()
    runs = run_all(args.variant, args.models, args.limit, args.skip_explain)
    write_reports(summarise(runs))
    print((RESULTS / "llm_summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
