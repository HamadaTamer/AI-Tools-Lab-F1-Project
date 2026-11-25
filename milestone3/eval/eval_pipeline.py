"""End-to-end pipeline evaluation (Milestone 3, spec section 6): every QA item through `handle_turn`.

The real NLU (rules + embeddings + entity extraction), session and Neo4j retrieval run; only the
LLM call is replaced by a fixed answer, because eval_llm.py evaluates the LLM stage. Each item is
therefore judged on the three stages before it: intent -> entities -> query.
Follow-up items are replayed in a deep copy of the session left by the item they follow.

Expected behaviour per category:
  normal / follow_up : gold intent and gold entities, then an answer (podium_prediction: "predict")
  ambiguous          : a clarifying question (items without an "ambiguous" list must resolve like normal)
  out_of_scope       : refusal and no retrieval
  empty              : gold intent and entities, the query returns no rows ("no_data")

Run from the repo root:  python -m milestone3.eval.eval_pipeline
Writes results/pipeline_items.csv and results/pipeline_summary.md.
"""
import copy
import json
import tempfile
from pathlib import Path

import pandas as pd

from milestone3 import config, pipeline
from milestone3.llm.client import LLMResult
from milestone3.nlu.embed_classifier import EmbeddingClassifier
from milestone3.nlu.entities import EntityExtractor
from milestone3.retrieval.graph import Graph
from milestone3.session import Session

RESULTS = config.RESULTS_DIR
MODELS = ["stub-model"]


def stub_complete(model, messages, temperature=0):
    """Stands in for the Groq call: a fixed answer, no network."""
    return LLMResult(model, "stub answer", None, 0.0, 0, 0)


def entity_ids(entities: dict) -> dict:
    """Compare entities by id: driverId / constructorId / circuitId for named things, else the raw value."""
    ids = {}
    for slot, value in entities.items():
        if isinstance(value, dict):
            value = next(v for k, v in value.items() if k.endswith("Id"))
        ids[slot] = value
    return ids


def expected_kind(item) -> str:
    """The turn kind the category asks for."""
    if item["category"] == "out_of_scope":
        return "refuse"
    if item.get("ambiguous"):
        return "clarify"
    if item["category"] == "empty":
        return "no_data"
    return "predict" if item["intent"] == "podium_prediction" else "answer"


def judge(item, turn, kind) -> str:
    """Failing stage of one item: intent | entity | query | ok."""
    if kind == "refuse":
        return "ok" if turn.kind == "refuse" and turn.retrieval is None else "intent"
    if kind == "clarify":
        return "ok" if turn.kind == "clarify" else "entity"
    if turn.intent != item["intent"]:
        return "intent"
    if entity_ids(turn.entities) != entity_ids(item["entities"]):
        return "entity"
    return "ok" if turn.kind == kind else "query"


def run_item(item, session, graph, extractor, classifier):
    """Run one question; a crash inside the pipeline (e.g. a template error) is kept as the error text."""
    try:
        return pipeline.handle_turn(item["question"], session, graph, extractor, classifier, MODELS), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def main():
    pipeline.complete = stub_complete                                   # no LLM call
    config.LOG_FILE = Path(tempfile.gettempdir()) / "eval_pipeline_turns.jsonl"   # not the app's log
    config.LOG_FILE.unlink(missing_ok=True)
    graph = Graph()
    extractor, classifier = EntityExtractor(graph.lookup_tables()), EmbeddingClassifier()
    items = json.load(open(config.EVAL_DIR / "qa_testset.json", encoding="utf-8"))
    followed = {i["follows"] for i in items if i["follows"]}
    sessions, rows = {}, []
    for item in items:
        session = copy.deepcopy(sessions[item["follows"]]) if item["follows"] else Session()
        turn, error = run_item(item, session, graph, extractor, classifier)
        kind = expected_kind(item)
        stage = "query" if error else judge(item, turn, kind)
        if item["id"] in followed:
            sessions[item["id"]] = copy.deepcopy(session)               # state for later follow-ups
        rows.append({
            "id": item["id"], "category": item["category"], "question": item["question"],
            "gold_intent": item["intent"], "pred_intent": turn and turn.intent,
            "expected_kind": kind, "turn_kind": turn and turn.kind,
            "gold_entities": entity_ids(item["entities"]), "sent_entities": turn and entity_ids(turn.entities),
            "n_rows": len(turn.retrieval.rows) if turn and turn.retrieval else None,
            "passed": stage == "ok", "failing_stage": stage, "error": error or ""})
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "pipeline_items.csv", index=False, encoding="utf-8")
    write_summary(table)
    print((RESULTS / "pipeline_summary.md").read_text(encoding="utf-8"))


def write_summary(table):
    """Pass rates overall and per category, the two special rates and the failures by stage."""
    rate = lambda df: f"{df.passed.sum()}/{len(df)} = {df.passed.mean():.1%}"
    oos, empty = table[table.category == "out_of_scope"], table[table.category == "empty"]
    lines = ["# Pipeline evaluation (LLM stubbed)", "",
             f"Overall pass rate: **{rate(table)}**", "", "| category | passed |", "| --- | --- |"]
    lines += [f"| {c} | {rate(g)} |" for c, g in table.groupby("category")]
    lines += ["", f"Out-of-scope refusal rate (refused, nothing retrieved): **{rate(oos)}**",
              f"Empty-result handling rate (graph returned no rows, user told so): **{rate(empty)}**", "",
              "Failures by stage:", "", "| stage | items |", "| --- | --- |"]
    lines += [f"| {s} | {', '.join(g.id)} |" for s, g in table[~table.passed].groupby("failing_stage")]
    (RESULTS / "pipeline_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
