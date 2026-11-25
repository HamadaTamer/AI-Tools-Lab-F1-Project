"""Prompt builders for the two LLM jobs: grounded answers and podium-prediction explanations.

Both prompts have three parts: PERSONA (who is speaking), CONTEXT (the only facts the model
may use) and TASK (the rules). The model never sees the graph or Cypher, only these facts.
`variant="strict"` is the improved task text used in the before-and-after experiment.
"""
import json

from milestone3.prediction import features

PERSONA = (
    "You are a meticulous Formula 1 statistician working on the pit wall. You only report "
    "what the timing and results data in front of you shows, and you never fill a gap from "
    "memory."
)

ANSWER_TASK = (
    "Task: answer the question using only the context above.\n"
    "- Quote numbers and names exactly as they appear in the rows.\n"
    "- If the rows do not contain the answer, say the data is insufficient; do not guess.\n"
    "- State the scope of the figures (what they cover and exclude) in one short clause.\n"
    "- Never add a count, year or name that is not in the context.\n"
    "- Be conversational but precise: one to three sentences."
)

# Improved task text: it targets the failure seen on the first run (extra numbers and
# names that are not in the rows, e.g. totals the model computed itself).
STRICT_EXTRA = (
    "\n- Do not calculate anything: no sums, differences or percentages. Use only numbers "
    "that are written in the rows or in the question.\n"
    "- Mention only names and numbers from the rows; leave out background knowledge such as "
    "teams, nationalities or dates that the rows do not show."
)

# Worked examples with made-up rows, clearly marked so they are never read as real data.
EXAMPLES = [
    {"question": "How many races did Driver A win in 2001?", "intent": "driver_season_stats",
     "rows": [{"driver": "Anna Example", "year": 2001, "wins": 4, "points": 88.0}],
     "scope": "Race results of the 2001 season only.",
     "answer": "Anna Example won 4 races in 2001 and scored 88 points (race results of the "
               "2001 season only)."},
    {"question": "Who were the top two in the 1990 Example Grand Prix?", "intent": "race_result",
     "rows": [{"position": 1, "driver": "Ben Sample", "points": 9.0},
              {"position": 2, "driver": "Carl Demo", "points": 6.0}],
     "scope": "Final classification of that one race.",
     "answer": "Ben Sample won the 1990 Example Grand Prix, followed by Carl Demo in second "
               "(final classification of that race only)."},
    {"question": "What was Dana Test's best lap at the 1985 Sample Grand Prix?", "intent": "pit_stops",
     "rows": [], "scope": "Pit stop data only.",
     "answer": "The retrieved data is insufficient to answer that: it contains no rows for "
               "this question, so I cannot give a figure."},
]


def _table(rows: list[dict]) -> str:
    """Rows as compact JSON lines, whole floats written as integers (12.0 -> 12)."""
    clean = [{k: (int(v) if isinstance(v, float) and v.is_integer() else v)
              for k, v in row.items() if v is not None} for row in rows]
    return "\n".join(json.dumps(r, ensure_ascii=False) for r in clean) or "(no rows)"


def _context(question, intent, entities, rows, scope) -> str:
    return (f"Question: {question}\nDetected intent: {intent}\n"
            f"Extracted entities: {json.dumps(entities, ensure_ascii=False)}\n"
            f"Scope of the figures: {scope}\nRetrieved rows:\n{_table(rows)}")


def answer_messages(question, intent, entities, retrieval, few_shot=True, variant="base") -> list[dict]:
    """Chat messages for a grounded answer: persona (system), then context + task (user)."""
    task = ANSWER_TASK + (STRICT_EXTRA if variant == "strict" else "")
    messages = [{"role": "system", "content": PERSONA}]
    # Few-shot: each worked example is a user/assistant pair with the same layout as the real turn.
    if few_shot:
        for ex in EXAMPLES:
            ctx = _context(ex["question"], ex["intent"], {}, ex["rows"], ex["scope"])
            messages += [{"role": "user", "content": f"EXAMPLE (made-up data)\n{ctx}\n\n{task}"},
                         {"role": "assistant", "content": ex["answer"]}]
    ctx = _context(question, intent, entities, retrieval.rows, retrieval.scope)
    messages.append({"role": "user", "content": f"{ctx}\n\n{task}"})
    return messages


EXPLAIN_TASK = (
    "Task: explain this podium prediction to a Formula 1 fan with no statistics background, "
    "in 4 to 6 sentences.\n"
    "- Say the predicted probability and whether the model predicts a podium, and name the "
    "features that pushed the probability up and those that pushed it down, with roughly how "
    "much (use the numbers given; call them small, moderate or strong pushes).\n"
    "- Contributions are in log-odds units; do not explain the maths, just compare sizes.\n"
    "- Never use causal language: a feature does not cause or lead to a podium. Say the model "
    "'relies on' or 'gives weight to' a feature, and that a push is an association in the "
    "model, not a reason.\n"
    "- Use only the features, values and facts in the context. Do not invent any other "
    "feature, value or fact. Mention the pre-race facts only if they are listed."
)


# Improvement after the first evaluation run (explanation fidelity was the weakest metric): the
# base prompt shows raw feature codes such as drv_finish_last, which models paraphrased or misread.
# The "strict" variant shows each feature's readable label and meaning, and asks for those labels.
LABEL_RULE = ("\n- Refer to each feature by its label exactly as written in the context, and name every "
              "feature listed under the pushes up and down.")


def explanation_messages(prediction: dict, graph_facts: list[str], variant: str = "base") -> list[dict]:
    """Chat messages explaining one prediction from its SHAP contributions and graph facts."""
    contribs = sorted(prediction["contributions"], key=lambda c: -abs(c[2]))
    up = [c for c in contribs if c[2] > 0][:3]
    down = [c for c in contribs if c[2] < 0][:3]

    # base: raw feature codes; strict: readable label plus a one-line meaning per feature
    specs = features.SPECS
    if variant == "strict":
        name = lambda f: specs[f].label
        line = lambda c: f"- {name(c[0])} = {round(c[1], 2):g}: {c[2]:+.2f}  ({specs[c[0]].help})"
    else:
        name = lambda f: f
        line = lambda c: f"- {c[0]} = {c[1]}: {c[2]:+.2f}"

    importance = ", ".join(f"{name(g['feature'])} {g['mean absolute SHAP']:.2f}"
                           for g in prediction["global_importance"][:8])
    context = "\n".join([
        f"Podium probability: {prediction['probability']:.1%} (threshold {prediction['threshold']:.2f}); "
        f"predicted class: {'podium' if prediction['predicted_podium'] else 'no podium'}",
        "Strongest local pushes up (SHAP, log-odds):", "\n".join(map(line, up)) or "- none",
        "Strongest local pushes down (SHAP, log-odds):", "\n".join(map(line, down)) or "- none",
        f"Global feature importance from Milestone 1 (mean |SHAP| over 200 test rows): {importance}",
        "Pre-race facts from the knowledge graph (races strictly before this one):",
        "\n".join(f"- {f}" for f in graph_facts) or "- none available",
    ])
    task = EXPLAIN_TASK + (LABEL_RULE if variant == "strict" else "")
    return [{"role": "system", "content": PERSONA},
            {"role": "user", "content": f"{context}\n\n{task}"}]
