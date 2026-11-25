"""Qualitative rating sheet: 4 questions per retrieval intent, answered by every compared model.

Reads the answers logged by eval_llm.py (results/llm_runs.jsonl, few-shot style of one prompt
variant), prints the items under the 1-5 fluency rubric and writes
results/qualitative_sheet.csv. Raters fill in the last three columns by hand.

Usage (from the repo root): python -m milestone3.eval.qualitative_sheet [--variant base]
"""
import argparse
import csv
import json
from collections import defaultdict

from milestone3 import config

RUBRIC = """Fluency rubric (1-5), rated independently by two raters:
  1 = unreadable or broken sentences
  2 = understandable but awkward, repetitive or confusing
  3 = acceptable, a few clumsy phrases
  4 = clear and natural, minor flaws
  5 = fluent, natural and precise, like a human analyst
causal_language: mark x if the answer states a cause ("because", "led to", "caused by")
that the data does not show."""
PER_INTENT = 4
COLUMNS = ["item", "intent", "question", "model", "answer",
           "fluency_rater1", "fluency_rater2", "causal_language"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="base")
    parser.add_argument("--style", default="few-shot")
    args = parser.parse_args()

    runs = [json.loads(line) for line in open(config.RESULTS_DIR / "llm_runs.jsonl", encoding="utf-8")]
    runs = [r for r in runs if r["kind"] == "qa" and r["variant"] == args.variant
            and r["style"] == args.style and r["intent"] != "out_of_scope"]
    # the log is append-only, so a retried call appears twice: keep the last record per model and item
    runs = list({(r["model"], r["id"]): r for r in runs}.values())

    # The first 4 items of each intent (in test-set order) that have an answer.
    by_intent = defaultdict(list)
    for r in runs:
        if r["id"] not in by_intent[r["intent"]]:
            by_intent[r["intent"]].append(r["id"])
    chosen = {i for ids in by_intent.values() for i in ids[:PER_INTENT]}

    rows = [{"item": r["id"], "intent": r["intent"], "question": r["question"],
             "model": r["model"], "answer": r["answer"] or config.API_ERROR,
             "fluency_rater1": "", "fluency_rater2": "", "causal_language": ""}
            for r in runs if r["id"] in chosen]
    rows.sort(key=lambda r: (r["intent"], r["item"], r["model"]))

    with open(config.RESULTS_DIR / "qualitative_sheet.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(RUBRIC, f"\n\n{len(rows)} answers to rate ({len(chosen)} questions x models)\n")
    for r in rows:
        print(f"[{r['item']}] ({r['intent']}) {r['question']}\n  {r['model']}: {r['answer']}\n")


if __name__ == "__main__":
    main()
