"""Evaluate the intent classifiers on the labelled test set (Milestone 3, part 1a).

Compares the rule-based classifier, the embedding classifier (three Hugging Face models,
thresholds chosen on the classifier's own examples) and two rule/embedding hybrids.
A "rephrase" answer (classifier unsure) counts as its own label and as wrong.
Writes confusion matrices (PNG + CSV), per-question predictions, the model comparison,
the threshold sweep and a short summary into eval/results/intent_*.
Run from the repo root:  python -m milestone3.eval.eval_intents
"""
import itertools

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from milestone3 import config
from milestone3.nlu.embed_classifier import EmbeddingClassifier
from milestone3.nlu.intents import INTENTS
from milestone3.nlu.rule_classifier import classify_rule

MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",   # small, fast, strong general-purpose baseline
    "sentence-transformers/all-mpnet-base-v2",  # larger, best quality of the sentence-transformers family
    "BAAI/bge-small-en-v1.5",                   # small retrieval-tuned model, different training recipe
]
SIMILARITY_GRID = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
MARGIN_GRID = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08]
MAX_REPHRASE = 0.10   # budget: at most one question in ten may trigger "please rephrase"
LABELS = list(INTENTS) + ["rephrase"]
OUT = config.RESULTS_DIR


def leave_one_out_sweep(clf: EmbeddingClassifier) -> pd.DataFrame:
    """Score every (min_similarity, min_margin) pair on the classifier's own examples.

    Each example is classified against all the *other* examples, so the thresholds are
    chosen without touching the test set."""
    texts = [e for i in INTENTS.values() for e in i.examples]
    similarities = clf.example_vectors @ clf.example_vectors.T
    np.fill_diagonal(similarities, -1.0)   # an example may not match itself
    scores = [clf.intent_scores(row) for row in similarities]
    rows = []
    for min_similarity, min_margin in itertools.product(SIMILARITY_GRID, MARGIN_GRID):
        clf.min_similarity, clf.min_margin = min_similarity, min_margin
        predicted = [clf.decide(s, t).intent for s, t in zip(scores, texts)]
        truth = clf.example_intents
        correct = sum(p == t for p, t in zip(predicted, truth))
        rephrase = sum(p is None for p in predicted)
        rows.append({"min_similarity": min_similarity, "min_margin": min_margin,
                     "loo_accuracy": correct / len(texts), "loo_rephrase_rate": rephrase / len(texts),
                     "loo_precision": correct / max(len(texts) - rephrase, 1)})
    return pd.DataFrame(rows)


def best_thresholds(sweep: pd.DataFrame) -> pd.Series:
    """Most precise answers while asking to rephrase at most MAX_REPHRASE of the time;
    on a tie prefer the lower (more permissive) thresholds."""
    allowed = sweep[sweep.loo_rephrase_rate <= MAX_REPHRASE]
    ranked = allowed.assign(permissive=-(allowed.min_similarity + allowed.min_margin))
    return ranked.sort_values(["loo_precision", "permissive"], ascending=False).iloc[0]


def predict_all(clf: EmbeddingClassifier, questions: list[str]) -> pd.DataFrame:
    predictions = [clf.classify(q) for q in questions]
    return pd.DataFrame({"embedding": [p.intent or "rephrase" for p in predictions],
                         "emb_score": [p.score for p in predictions],
                         "emb_margin": [p.margin for p in predictions]})


def confusion(truth: pd.Series, predicted: pd.Series, name: str, title: str):
    """Save the confusion matrix as CSV and as a labelled heat-map PNG."""
    table = pd.crosstab(truth, predicted).reindex(index=list(INTENTS), columns=LABELS, fill_value=0)
    table.to_csv(OUT / f"intent_confusion_{name}.csv")
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(table.values, cmap="Blues")
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=60, ha="right")
    ax.set_yticks(range(len(INTENTS)), list(INTENTS))
    for (r, c), value in np.ndenumerate(table.values):
        if value:
            ax.text(c, r, value, ha="center", va="center", color="white" if value > 3 else "black")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(OUT / f"intent_confusion_{name}.png", dpi=150)
    plt.close(fig)


def markdown_table(df: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(df.columns) + " |", "|" + " --- |" * len(df.columns)]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in row) + " |")
    return "\n".join(lines)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    test = pd.read_csv(config.EVAL_DIR / "intent_testset.csv")
    questions = test["question"].tolist()
    accuracy = lambda predicted: float((predicted == test["intent"]).mean())

    # Approach 1: rules.
    rule = pd.Series([classify_rule(q) or "rephrase" for q in questions])

    # Approach 2: embeddings, one run per model with thresholds picked by leave-one-out.
    comparison, runs = [], {}
    for model in MODELS:
        clf = EmbeddingClassifier(model)
        sweep = leave_one_out_sweep(clf)
        chosen = best_thresholds(sweep)
        clf.min_similarity, clf.min_margin = chosen.min_similarity, chosen.min_margin
        predictions = predict_all(clf, questions)
        runs[model] = (predictions, sweep, chosen)
        comparison.append({"model": model, "min_similarity": chosen.min_similarity,
                           "min_margin": chosen.min_margin, "loo_precision": chosen.loo_precision,
                           "test_accuracy": accuracy(predictions["embedding"]),
                           "test_rephrase_rate": float((predictions["embedding"] == "rephrase").mean())})
    comparison = pd.DataFrame(comparison)
    # the embedding model is chosen on the classifier examples (leave-one-out), never on the test set
    winner = comparison.sort_values("loo_precision", ascending=False).iloc[0]
    embedding, sweep, _ = runs[winner.model]

    # Hybrids: one classifier decides first, the other answers when the first has no answer.
    rule_first = rule.where(rule != "rephrase", embedding["embedding"])
    embed_first = embedding["embedding"].where(embedding["embedding"] != "rephrase", rule)
    approaches = {"rule": rule, "embedding": embedding["embedding"],
                  "hybrid_rule_first": rule_first, "hybrid_embedding_first": embed_first}
    results = pd.DataFrame({name: [accuracy(p)] for name, p in approaches.items()}).T
    results.columns = ["test_accuracy"]
    results["rephrase_rate"] = [float((p == "rephrase").mean()) for p in approaches.values()]
    results = results.reset_index(names="approach")

    # Save everything.
    per_question = test.assign(rule=rule, embedding=embedding["embedding"], hybrid_rule_first=rule_first,
                               hybrid_embedding_first=embed_first, emb_score=embedding["emb_score"].round(3),
                               emb_margin=embedding["emb_margin"].round(3))
    per_question.to_csv(OUT / "intent_predictions.csv", index=False)
    comparison.to_csv(OUT / "intent_model_comparison.csv", index=False)
    sweep.to_csv(OUT / "intent_threshold_sweep.csv", index=False)
    for name, title in [("rule", "Rule-based classifier"), ("embedding", f"Embedding classifier ({winner.model})"),
                        ("hybrid_rule_first", "Hybrid: rules first, embeddings as fallback")]:
        confusion(test["intent"], approaches[name], name.replace("hybrid_rule_first", "hybrid"), title)

    best = results.sort_values("test_accuracy", ascending=False).iloc[0]
    wrong = per_question[per_question[best.approach] != per_question["intent"]]
    summary = f"""# Intent classification results

Test set: {len(test)} questions, {test['intent'].value_counts().min()}-{test['intent'].value_counts().max()} per intent
(worded differently from the classifier examples). "rephrase" counts as wrong.

## Approaches on the test set

{markdown_table(results)}

## Embedding models (thresholds chosen by leave-one-out on the classifier examples)

Thresholds: the most precise pair on leave-one-out over the classifier examples whose rephrase
rate stays at or below {MAX_REPHRASE:.0%}. Nothing was tuned on the test set.

{markdown_table(comparison)}

Winner (highest leave-one-out precision on the examples): `{winner.model}` with MIN_SIMILARITY={winner.min_similarity}, MIN_MARGIN={winner.min_margin}
(test accuracy {winner.test_accuracy:.3f}).

## Recommendation

Best test accuracy: **{best.approach}** ({best.test_accuracy:.3f}, rephrase rate {best.rephrase_rate:.3f}).
Misclassified by it: {len(wrong)} of {len(test)}; see `intent_predictions.csv`.
Confusion matrices: `intent_confusion_*.png/csv`. Threshold sweep: `intent_threshold_sweep.csv`.
"""
    (OUT / "intent_summary.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
