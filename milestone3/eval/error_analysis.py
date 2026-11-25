"""Error analysis by pipeline stage (Milestone 3, spec section 6).

Every failure is assigned to the stage where it happened, using the results the other evaluation
scripts already wrote (nothing is re-run except the cheap metric functions of eval_llm.py):
  intent        - hybrid_rule_first predictions vs gold (intent_predictions.csv) and pipeline items
  entity        - pipeline items whose entities (or clarification) were wrong
  query         - pipeline items with a template error or unexpected empty / non-empty rows
  misread rows  - LLM answers on items that have rows but mention < 100% of the gold facts
  hallucination - LLM answers with a number or name that is not in the context (is_faithful False)
The faithfulness heuristic has false positives (a capitalised word from the scope sentence, a
computed total), so hallucination examples list the unsupported tokens for a human to judge.

Run from the repo root:  python -m milestone3.eval.error_analysis   (after eval_pipeline / eval_llm)
Writes results/error_analysis.md.
"""
from collections import Counter

import pandas as pd

from milestone3 import config
from milestone3.eval.eval_llm import (capitalised_names, correctness, is_faithful, load_runs,
                                      numbers_in, words_in)

RESULTS = config.RESULTS_DIR
EXAMPLES = 3
SNIPPET = 300


def clip(text, n=SNIPPET):
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n] + "..."


def intent_failures():
    """Wrong intents: the deployed classifier on the intent test set, and the pipeline items."""
    preds = pd.read_csv(RESULTS / "intent_predictions.csv")
    wrong = preds[preds.hybrid_rule_first != preds.intent]
    pipe = pd.read_csv(RESULTS / "pipeline_items.csv")
    pipe = pipe[pipe.failing_stage == "intent"]
    return ([f"`{r.question}` - predicted `{r.hybrid_rule_first}`, gold `{r.intent}`" for r in wrong.itertuples()],
            len(preds),
            [f"`{r.question}` ({r.id}) - predicted `{r.pred_intent}`, gold `{r.gold_intent}`" for r in pipe.itertuples()])


def pipeline_failures(stage):
    items = pd.read_csv(RESULTS / "pipeline_items.csv")
    out = []
    for r in items[items.failing_stage == stage].itertuples():
        detail = (f"expected `{r.expected_kind}`, got `{r.turn_kind}` with {r.n_rows} rows; {r.error}"
                  if stage == "query" else f"gold {r.gold_entities}, sent {r.sent_entities} (turn `{r.turn_kind}`)")
        out.append(f"`{r.question}` ({r.id}) - {detail}")
    return out, len(items)


def llm_failures():
    """(variant, model) -> {'misread': [...], 'hallucination': [...], 'n': answers}, from llm_runs.jsonl."""
    groups = {}
    for run in load_runs():
        if run["kind"] != "qa" or not run["answer"]:
            continue
        g = groups.setdefault((run["variant"], run["model"]), {"misread": [], "hallucination": [], "n": 0})
        g["n"] += 1
        context, answer = run["prompt"][-1]["content"], run["answer"]
        where = f"{run['style']}, {run['id']}"
        if run["rows"] and (c := correctness(answer, run["question"], run["rows"])) is not None and c < 1:
            g["misread"].append(f"`{run['question']}` ({where}, correctness {c:.2f}) - rows `{clip(run['rows'][0], 150)}` "
                                f"- answer: \"{clip(answer)}\"")
        if not is_faithful(answer, context):
            nums = sorted(numbers_in(answer) - numbers_in(context))
            names = sorted({w for w in capitalised_names(answer) if w.lower() not in words_in(context)})
            g["hallucination"].append(f"`{run['question']}` ({where}) - unsupported numbers {nums}, names {names} "
                                      f"- answer: \"{clip(answer)}\"")
    return groups


def section(title, failures, total, unit="items"):
    lines = [f"### {title}", "", f"{len(failures)} of {total} {unit}.", ""]
    return lines + [f"- {f}" for f in failures[:EXAMPLES]] + [""]


def main():
    intent_test, n_test, intent_pipe = intent_failures()
    entity, n_items = pipeline_failures("entity")
    query, _ = pipeline_failures("query")
    llm = llm_failures()

    lines = ["# Error analysis", "", "Failures grouped by the stage where they occur. Counts first, then examples "
             "(the first few of each list; full lists are in `pipeline_items.csv`, `intent_predictions.csv`, "
             "`llm_runs.jsonl`).", "", "## Counts per stage", "",
             "| stage | source | failures | out of |", "| --- | --- | --- | --- |",
             f"| intent | intent test set (hybrid_rule_first) | {len(intent_test)} | {n_test} |",
             f"| intent | pipeline items | {len(intent_pipe)} | {n_items} |",
             f"| entity | pipeline items | {len(entity)} | {n_items} |",
             f"| query | pipeline items | {len(query)} | {n_items} |"]
    for (variant, model), g in sorted(llm.items()):
        lines.append(f"| model misread the rows | {model} ({variant}) | {len(g['misread'])} | {g['n']} answers |")
    for (variant, model), g in sorted(llm.items()):
        lines.append(f"| hallucination | {model} ({variant}) | {len(g['hallucination'])} | {g['n']} answers |")

    lines += ["", "## Examples", ""]
    lines += section("Intent (intent test set)", intent_test, n_test, "questions")
    lines += section("Intent (pipeline items)", intent_pipe, n_items)
    lines += section("Entity", entity, n_items)
    lines += section("Query", query, n_items)
    for kind, title in [("misread", "Model misread the rows"), ("hallucination", "Hallucination (check the unsupported tokens: "
                        "scope-sentence words and computed totals are false positives)")]:
        for (variant, model), g in sorted(llm.items()):
            lines += section(f"{title} - {model} ({variant})", g[kind], g["n"], "answers")
    (RESULTS / "error_analysis.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
