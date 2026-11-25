# Intent classification results

Test set: 94 questions, 7-10 per intent
(worded differently from the classifier examples). "rephrase" counts as wrong.

## Approaches on the test set

| approach | test_accuracy | rephrase_rate |
| --- | --- | --- |
| rule | 0.915 | 0.043 |
| embedding | 0.787 | 0.085 |
| hybrid_rule_first | 0.936 | 0.011 |
| hybrid_embedding_first | 0.862 | 0.011 |

## Embedding models (thresholds chosen by leave-one-out on the classifier examples)

Thresholds: the most precise pair on leave-one-out over the classifier examples whose rephrase
rate stays at or below 10%. Nothing was tuned on the test set.

| model | min_similarity | min_margin | loo_precision | test_accuracy | test_rephrase_rate |
| --- | --- | --- | --- | --- | --- |
| sentence-transformers/all-MiniLM-L6-v2 | 0.300 | 0.005 | 0.355 | 0.532 | 0.085 |
| sentence-transformers/all-mpnet-base-v2 | 0.450 | 0.005 | 0.518 | 0.787 | 0.085 |
| BAAI/bge-small-en-v1.5 | 0.300 | 0.000 | 0.270 | 0.532 | 0.011 |

Winner: `sentence-transformers/all-mpnet-base-v2` with MIN_SIMILARITY=0.45, MIN_MARGIN=0.005
(test accuracy 0.787).

## Recommendation

Best test accuracy: **hybrid_rule_first** (0.936, rephrase rate 0.011).
Misclassified by it: 6 of 94; see `intent_predictions.csv`.
Confusion matrices: `intent_confusion_*.png/csv`. Threshold sweep: `intent_threshold_sweep.csv`.
