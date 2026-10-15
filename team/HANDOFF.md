# Hand-off: data → modeling

Sections 1–10 of `F1_podium_prediction.ipynb` are done: audit, cleaning, EDA, the three data-engineering questions, feature engineering, split, feature selection and preprocessing. Sections 11–16 are empty headings in the same notebook, with the brief's requirements written under each one; section 17 has the data assumptions and limitations. Everyone works in this one notebook. This file says what you get, how to work in it without breaking each other's work, and what each remaining section must contain.

## 1. What you get

After Run All reaches the end of section 10 (about 75 seconds), these names exist:

| Name | What it is |
| --- | --- |
| `model_table` | 24,724 rows, one per driver–race (starters only, no Indianapolis 500): ids, `year`, `split`, target `podium`, the 9 features |
| `SELECTED` | the 9 features, in order |
| `FEATURE_GROUPS` | `grid`, `qualifying`, `standings`, `driver form` → the units of the ablation |
| `X_train, y_train` | 1950–2019, 22,589 rows, 13.4% podiums |
| `X_val, y_val` | 2020–2021, 779 rows, 15.0% podiums |
| `X_test, y_test` | 2022–2024, 1,356 rows, 15.0% podiums |
| `make_preprocessor()` | median imputation + missing indicators + standard scaling (sklearn pipeline, pandas output) |
| `evaluate(y, proba, threshold)` | dict with ROC-AUC, PR-AUC, F1, precision, recall |
| `best_threshold(y_val, proba_val)` | the F1-optimal threshold, to be chosen on validation |
| `make_lr()` | the logistic regression used for every data decision (a ready baseline) |
| `score_windows(data, features, model)` | the three-window validation used in sections 8–9, if you want the same robustness check |
| `models_to_compare` | empty dict; sections 11 and 12 add their final fitted models, section 13 scores all of them the same way |

The `X_*` matrices contain raw values with `NaN`. Linear models and the neural network need `make_preprocessor()`; tree models do not need scaling and `HistGradientBoostingClassifier` takes `NaN` directly. Keep the same rows and the same 9 features for every model so the scores are comparable (the brief requires it).

Anything you put in `models_to_compare` must have `predict_proba(X)` on the raw `X_*` (so put `make_preprocessor()` and the model in one `Pipeline`). For a Keras or PyTorch network, wrap it in a small class with a `predict_proba` method that applies the fitted preprocessor first.

The table is also written to `outputs/model_table.csv` and the feature list and groups to `outputs/feature_spec.json`, if you want to explore it outside the notebook.

## 2. Working in one notebook

A notebook is one JSON file, and plain git merges it line by line. We tested this: two branches that each added cells to their own section and re-ran the notebook produced 87 conflict markers with plain git. With **nbdime** (a notebook-aware merge tool) the same merge succeeded, kept both sections and gave a valid notebook. Stripping outputs before committing was not enough on its own: the empty section headings sit next to each other, so a line-based merge still conflicted.

### One-time setup, per clone

```bash
pip install -r requirements.txt     # includes nbdime
git-nbmergedriver config --enable   # run inside the repo; the committed .gitattributes points .ipynb merges to it
```

Merge with the venv activated: the merge driver is a command inside the venv. If git cannot find it, the merge stops with a conflict and nothing is lost; activate the venv and merge again.

The data downloads itself on the first Run All (into `data/`, git-ignored).

### Who owns what

Fill in the names.

| Section | Owner |
| --- | --- |
| 11. Baseline and statistical models | |
| 12. Shallow neural network | |
| 13. Model comparison | |
| 14. Feature-group ablation | |
| 15. Explainability (XAI) | |
| 16. Inference function | |
| 17 + report sections 8–10 | |

### Workflow

1. `git switch -c <section>` (e.g. `s11-models`), then Run All once so everything up to section 10 exists.
2. Add cells **only under your section heading**. Never edit cells in sections 1–10 or in someone else's section. nbdime merges different cells automatically, but two people editing the same cell is still a conflict.
3. Never reassign the names in the table above (`X_train`, `SELECTED`, ...). Create your own (`X_train_scaled`, `rf_search`, ...). Later sections rely on the originals.
4. Before you commit: Restart Kernel and Run All, no errors.
5. Merge into `main` (pull request or `git merge`). After a merge, whoever merged runs Run All on `main` once and commits, so the saved outputs match the code.
6. Do not edit the notebook together on Kaggle: two people saving the same Kaggle notebook overwrite each other. Kaggle is only for the final upload.

Sections 11 and 12 can run in parallel; 13–16 need the chosen model from 11–12.

## 3. Rules that keep the results valid

- Fit everything (preprocessing, model) on train only.
- Choose hyperparameters and the classification threshold on validation only.
- Score the test set once, at the end, in section 13. Never go back and change a model because of a test score.
- Report the podium base rate next to PR-AUC (15% in val/test): a PR-AUC is only meaningful compared with it.
- Expect train scores to be *lower* than validation scores. The training seasons include the 1950s–1990s, when bigger fields and poor reliability made podiums much harder to predict (the logistic regression gets PR-AUC 0.57 on train and 0.73 on validation). This is not underfitting by itself; compare models on validation.
- Differences in the third decimal are noise. Section 7.1 measured it: dropping 1% of the training rows moves the mean PR-AUC by up to ±0.005 (logistic regression) and ±0.011 (gradient boosting).
- Keep every attempt you compare in the notebook (the brief asks for the trials to stay visible), each in its own cell, with a short markdown conclusion under it.

## 4. What each section must contain (from the brief)

### 11. Baseline and statistical models

- A baseline (the brief accepts a statistical model or a shallow FFNN; `make_lr()` is ready).
- At least two different statistical models (e.g. logistic regression, random forest, gradient boosting).
- Hyperparameter tuning on validation, shown as a table of the values tried.
- How each model works and its limitations, in a few lines each.
- Train and validation performance.
- Add the final models to `models_to_compare`.

### 12. Shallow neural network

- 1–2 hidden layers, Keras or PyTorch, inputs from `make_preprocessor()`.
- Training curves (loss and validation metric per epoch), early stopping on validation.
- At least one architecture or hyperparameter variant, compared with the baseline.
- Add the final network to `models_to_compare` (wrapped, see section 1).

### 13. Model comparison

- One table: each model in `models_to_compare` × train / validation / test × ROC-AUC, PR-AUC, F1, precision, recall, all at the threshold chosen on validation.
- The brief says "all five metrics" but its table lists three; we report ROC-AUC, PR-AUC, F1 plus precision and recall. Say so in the notebook.
- Primary and secondary metric, justified. Recommendation: PR-AUC primary (podiums are the rare class), ROC-AUC secondary, F1 at the validation threshold.
- ROC and PR curves; at least three distinct modeling attempts in total.

### 14. Feature-group ablation

- Retrain the chosen model once per group in `FEATURE_GROUPS`, each time without that group; report the drop in validation metrics per group.
- Section 8.3 is the forward version made before training; the ablation is the after-training measurement.

### 15. Explainability (XAI)

- Global: permutation importance and SHAP (summary plot).
- Local: at least one prediction explained with LIME and/or a SHAP waterfall.
- Interpret every plot in words, and state explicitly that importance shows what the model relies on, not what causes podiums.

### 16. Inference function

- `predict_podium(record)`: takes a dict with the 9 feature values, returns the probability and the 0/1 prediction at the chosen threshold.
- Put `make_preprocessor()` and the model in one sklearn `Pipeline`, so the function can take raw values and the imputer handles missing ones.
- Demonstrate it on a complete record and on a record with missing values (e.g. no qualifying data: `quali_pos` and `q1_gap_pct` missing).

### 17. Assumptions and limitations

- The data assumptions and limitations are already written; add the modeling ones (threshold sensitivity, era drift on the test set, what the model cannot know).

## 5. Report

`report/report.md` has sections 1–7 written, with every figure the notebook saves in `report/figures/`. The brief wants every plot in both the notebook and the report: save each new figure with `save_fig('name')` before `plt.show()` and add it to the report. Every Run All overwrites the figures, so the report always shows the latest run; numbers quoted in the text do not update themselves, so re-check them against the notebook after a change. Fill sections 8–10 and export the final version to PDF for the submission.

## 6. Dependencies

Add what your section uses to `requirements.txt` (for example `shap`, `lime`, and `tensorflow` or `torch`). On Kaggle these are usually preinstalled.

## 7. Style

The notebook is read and graded by people. Short markdown that says what we did, why, and what the evidence shows; numbers taken from the output above it; simple plots with titles and axis labels; no decoration.
