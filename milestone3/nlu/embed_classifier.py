"""Embedding-similarity intent classifier (approach 2 of the intent classification step).

Every example phrasing from `intents.py` is embedded once at start-up. A new question is
embedded and compared (cosine similarity) with all examples; each intent scores as the
best similarity among its examples. If the winner is too weak or too close to the
runner-up the classifier returns intent=None and the app asks the user to rephrase.
"""
import re
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

from milestone3 import config
from milestone3.nlu.intents import INTENTS
from milestone3.nlu.rule_classifier import classify_rule

# Words that signal an F1 question; used only to double-check an out_of_scope guess.
_F1_WORDS = re.compile(
    r"\b(f1|formula|grand prix|gp|driver|drivers|race|races|lap|laps|team|constructor|qualif\w*|"
    r"pole|podium|circuit|season|championship|pit|grid|standings?|points|wins?)\b", re.I)


@dataclass
class IntentPrediction:
    intent: str | None   # None = ask the user to rephrase
    score: float         # similarity of the best intent
    runner_up: str | None
    margin: float        # best score minus runner-up score


class EmbeddingClassifier:
    """Nearest-example intent classifier with similarity, margin and out_of_scope checks."""

    def __init__(self, model_name=config.EMBEDDING_MODEL,
                 min_similarity=config.MIN_SIMILARITY, min_margin=config.MIN_MARGIN):
        self.model = SentenceTransformer(model_name)
        self.min_similarity = min_similarity
        self.min_margin = min_margin
        # One row per example phrasing, L2-normalised so a dot product is the cosine.
        self.example_intents = [i.name for i in INTENTS.values() for _ in i.examples]
        examples = [e for i in INTENTS.values() for e in i.examples]
        self.example_vectors = self.embed(examples)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Unit-length sentence vectors, one row per text."""
        return self.model.encode(texts, normalize_embeddings=True)

    def intent_scores(self, similarities: np.ndarray) -> dict[str, float]:
        """Best similarity per intent. Max (not mean) because each intent has varied
        phrasings: one close paraphrase should be enough, and averaging would punish
        intents whose examples are intentionally diverse."""
        scores = {name: -1.0 for name in INTENTS}
        for name, sim in zip(self.example_intents, similarities):
            scores[name] = max(scores[name], float(sim))
        return scores

    def decide(self, scores: dict[str, float], text: str) -> IntentPrediction:
        """Apply the minimum-score, margin and out_of_scope rules to the intent scores."""
        (best, best_score), (second, second_score) = sorted(scores.items(), key=lambda kv: -kv[1])[:2]
        prediction = IntentPrediction(best, best_score, second, best_score - second_score)
        if best_score < self.min_similarity or prediction.margin < self.min_margin:
            prediction.intent = None
        # Spec rule: never refuse a question on similarity alone. out_of_scope is accepted
        # only if the text also has no F1 vocabulary or an explicit unsupported topic
        # (weather, salary, tyres, ...) matches the rule classifier.
        elif best == "out_of_scope" and _F1_WORDS.search(text) and classify_rule(text) != "out_of_scope":
            prediction.intent = None
        return prediction

    def classify(self, text: str) -> IntentPrediction:
        """Intent of `text` (None = ask to rephrase) with its score and margin."""
        similarities = self.example_vectors @ self.embed([text])[0]
        return self.decide(self.intent_scores(similarities), text)
