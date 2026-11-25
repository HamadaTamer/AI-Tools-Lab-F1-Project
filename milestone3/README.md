# Milestone 3: Graph-RAG Formula 1 assistant

AI Tools Lab, Milestone 3, Team 97. Team members and the Milestone 1 work are listed in the [top-level README](../README.md).

The assistant has two capabilities. (1) Grounded question answering over the Milestone 2 Neo4j graph: a question is classified into one of 12 intents, its entities are extracted, a fixed parameterised Cypher template is run, and only the returned rows go to a language model, which phrases the answer. Two Groq models answer the same prompt side by side. (2) A podium-prediction explainer: the saved Milestone 1 shallow neural network predicts the podium probability of a driver in a race from nine pre-race features, live SHAP values show what drove the number, and a language model explains the SHAP values and a few pre-race facts from the graph in plain language, without calling anything a cause.

- **Report:** [`report/report.pdf`](report/report.pdf) (source `report/report.tex`).
- **Code walkthrough (for the oral exam):** [`STUDY.md`](STUDY.md). Interface contracts: [`CONTRACTS.md`](CONTRACTS.md).

## 1. Setup

All commands run from the repository root. Windows paths are shown; on Linux/macOS use `.venv/bin/python`.

**Python environment.**

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` covers all three milestones. `torch` is only used for inference here, so the CPU build is enough: install it first with `pip install torch --index-url https://download.pytorch.org/whl/cpu` to avoid the large CUDA download. The first run downloads the sentence-embedding model `all-mpnet-base-v2` (about 400 MB) from Hugging Face.

**Neo4j with the Milestone 2 graph.** Start Neo4j, put the connection settings in `milestone2/config.txt` (`URI`, `USERNAME`, `PASSWORD`, optional `DATABASE`; the same file the Milestone 2 builder reads), keep the Milestone 2 CSVs next to the script (`milestone2/*.csv`) and load the graph once:

```
python milestone2/Create_kg.py
```

**Groq key.** Create a file named `.env` at the repository root with one line, `GROQ_API_KEY=...`. It is git-ignored and must never be committed. All three language models are served by Groq's free tier.

**Model artifacts and the model table (both come from the Milestone 1 notebook).**

- `milestone3/models/` (`nn.pt`, `preprocessor.joblib`, `shap_background.npy`, `global_shap.csv`, `meta.json`) is written by the export cell in section 18 of `F1_podium_prediction.ipynb`. It holds the trained network, the fitted preprocessing, the decision threshold and the SHAP background, so the assistant predicts without retraining.
- `milestone3/models/model_table.csv` is the Milestone 1 modeling table (built in the notebook, section 8, also written to the git-ignored `outputs/`); the export cell saves a copy next to the model so a fresh clone works. The prediction view reads it to prefill the features of a real driver-race, and `prediction/features.py` reads it for the feature ranges.
- Both are needed. To regenerate them, download the dataset as described in the top-level README and run `F1_podium_prediction.ipynb` top to bottom (section 18 is the last section), so that both come from the same run.

## 2. Run the application

```
python -m streamlit run milestone3/app.py
```

The sidebar switches between the two views and always shows the cached intent and entity slots of the conversation, with a button to reset it.

**Chat view.** Type a Formula 1 question. Two models, chosen in the view from the three listed in section 7, answer side by side from the same retrieved rows. Under each in-domain answer a retrieval trace shows the intent, the entities, the Cypher query with its parameters and the returned rows; graphical answers also draw the retrieved subgraph with node, edge and relationship tables. The session caches the intent and entities, so follow-ups work ("What about Norris?", "And at Monza?").

**Podium prediction view.** Pick a driver, a race and a year to prefill the nine pre-race features from the Milestone 1 model table, or type all values manually. Values are validated against the ranges observed in the training data before the model is called, and invalid or missing values are rejected with a message. The view then shows the podium probability and the class at the Milestone 1 threshold, a SHAP waterfall plot for this driver-race, and a plain-language explanation written by a selected model from the SHAP values and from pre-race graph facts (taken strictly before the race date).

**"API connection error".** If a Groq call fails (no key, no network, rate limit), that model's column shows the text `API connection error` and, below it, a deterministic reading of the retrieved rows, so the data is still visible; the other model's column is unaffected. If Neo4j is not reachable the app says so instead of answering. Every chat turn is logged to `milestone3/logs/turns.jsonl` (git-ignored).

## 3. Run the evaluations

From the repository root, in this order. On Windows set `PYTHONIOENCODING=utf-8` first (PowerShell: `$env:PYTHONIOENCODING = "utf-8"`; Git Bash: `export PYTHONIOENCODING=utf-8`) so the scripts can print non-ASCII characters.

```
python -m milestone3.eval.eval_intents                  # rules vs 3 embedding models vs hybrids, thresholds, confusion matrices
python -m milestone3.eval.eval_pipeline                 # end-to-end behaviour on 57 items, LLM stubbed
python -m milestone3.eval.eval_llm --variant base       # 3 models x zero-/few-shot, answers and explanations (Groq)
python -m milestone3.eval.eval_llm --variant strict     # same with the improved prompts (before/after)
python -m milestone3.eval.error_analysis                # failures grouped by pipeline stage
python -m milestone3.eval.qualitative_sheet             # 40-item rating sheet for the fluency rating
```

`eval_llm` options: `--models M ...`, `--limit N` (quick test), `--skip-explain`. It resumes: calls already stored in `llm_runs.jsonl` are skipped and failed ones are retried. Only `eval_llm` needs the Groq key. The Groq free tier has a daily token cap per model (200k tokens per day for qwen); if it is reached, some calls fail, and running `python -m milestone3.eval.eval_llm --variant base` and then `--variant strict` again the next day re-runs only the missing or failed calls. Running the evaluations on demo day can exhaust a model's quota, after which the app shows "API connection error" plus the deterministic reading of the rows (by design).

Self-checks:

```
python -m milestone3.retrieval.graph          # one example per intent against Neo4j
python -m milestone3.tests.check_pipeline     # conversation logic with a fake graph and a fake LLM, no network
python -m milestone3.prediction.predictor     # predicts every explanation case and saves a waterfall plot
```

All outputs go to `milestone3/eval/results/`: `intent_*` (summary, per-question predictions, threshold sweep, confusion matrices as csv and png), `pipeline_summary.md` and `pipeline_items.csv`, `llm_runs.jsonl` (every call), `llm_comparison.csv` and `llm_summary.md`, `error_analysis.md`, `qualitative_sheet.csv`, `feature_ranges.csv` and `waterfall_example.png`. `python -m milestone3.prediction.features` rewrites `feature_ranges.csv`.

## 4. The report

[`report/report.pdf`](report/report.pdf) is built from `report/report.tex`. To rebuild it, run `pdflatex report.tex` twice (the second pass resolves references) inside `milestone3/report/`.

## 5. Required statements

**(a) How the pre-race features are computed.** The assistant does not compute them from the graph. The nine features of the Milestone 1 model (`grid_pos`, `pit_lane_start`, `quali_pos`, `q1_gap_pct`, `drv_standing`, `con_standing`, `drv_podium_last`, `drv_finish_last`, `drv_career_podium`) are taken from the Milestone 1 modeling table (`milestone3/models/model_table.csv`, a copy of `outputs/model_table.csv` saved by the export cell), which the notebook built with its leakage rules: a feature may only use what is known after qualifying and before the start. Standings are the championship positions after the previous round of the same season (none before round 1). Form features (`drv_podium_last`, `drv_finish_last`) use the driver's previous 3 races only, and `drv_career_podium` is the podium rate over all earlier races; rows are ordered by date, shifted by one race and then averaged, so the race itself is never included. The notebook checks this automatically (section 6.3). For a race that is not in the data (a hypothetical race, or a missing qualifying record), the values come from what the user types, with the training-season medians as defaults for anything left out, which is also what the Milestone 1 imputer fills.

**(b) The graph schema is unchanged.** No label, relationship type or property was added, renamed or removed relative to `milestone2/Create_kg.py`. All Cypher in the assistant is read-only and parameterised, and the language model never writes Cypher and never sees the graph, only the returned rows.

**(c) Questions that are refused or not answered directly.**

- Non-Formula 1 questions, and Formula 1 questions the graph cannot answer (weather, tyres, salaries, telemetry, car specifications, news, opinions and similar), get "I don't have the knowledge to answer that. I can only answer Formula 1 questions the knowledge graph covers." Nothing is retrieved and no retrieval trace is shown.
- An in-scope question whose query returns no rows gets "The knowledge graph has no data for that question."
- An ambiguous name (for example two drivers with the same surname) or a missing required entity gets a clarifying question ("Which Schumacher do you mean: ...?"); the reply is merged with the cached slots.
- A message the classifier cannot place gets a request to rephrase ("I could not tell what you are asking. Could you rephrase the question?").

## 6. Repository layout of `milestone3/`

```
app.py                 Streamlit UI: chat view, prediction view, sidebar entity cache, retrieval trace
pipeline.py            handle_turn: one chat message from input to logged answer
session.py             cached intent, entities, pending clarification, history
config.py              paths, Groq key (from .env), model ids, thresholds, prompt variants
nlu/                   12 intents, rule classifier, embedding classifier, entity extraction
retrieval/             Cypher templates and read-only Neo4j access
llm/                   Groq client, prompts, deterministic fallback for failed calls
prediction/            feature specs and validation, saved-model predictor with SHAP, graph facts
models/                artifacts exported from the notebook (section 18)
eval/                  test sets, evaluation scripts, results/
tests/                 check_pipeline.py (no Neo4j, no Groq)
report/                report.tex and report.pdf
logs/                  turns.jsonl (git-ignored)
STUDY.md, CONTRACTS.md walkthrough and interface contracts
```

`STUDY.md` has the file-by-file map and the walkthrough of one turn.

## 7. Models used and deployed choices

- **Intent classification:** ordered regular-expression rules first, with the sentence-embedding classifier `sentence-transformers/all-mpnet-base-v2` as the fallback when no rule matches (minimum similarity 0.45 and minimum margin 0.005, below which the user is asked to rephrase). The embedding model and thresholds were chosen on the classifier examples by leave-one-out, not on the test set.
- **Language models (Groq):** `allam-2-7b`, `qwen/qwen3.8-27b`, `openai/gpt-oss-20b`. The chat uses the few-shot prompt.
- **Prompt variants** (`config.py`): answers use `base` (`ANSWER_VARIANT`), explanations use `strict` (`EXPLAIN_VARIANT`). The strict explanation prompt raised explanation fidelity clearly; the strict answer rules gave mixed results, so answers keep `base`.
- **Prediction model:** the Milestone 1 shallow neural network (hidden layers 64 and 32, dropout 0.2), decision threshold 0.69, SHAP `DeepExplainer`, no retraining.

## 8. Key results

Numbers are copied from the files in [`eval/results/`](eval/results/); open them for the full tables.

- **Intent classification** ([`intent_summary.md`](eval/results/intent_summary.md)), 94 test questions: rules 0.915, embedding (all-mpnet-base-v2) 0.787, rules then embedding (deployed) 0.936 with a 1.1% rephrase rate. Six of the 94 are misclassified by the deployed hybrid.
- **End-to-end pipeline, LLM stubbed** ([`pipeline_summary.md`](eval/results/pipeline_summary.md)): 57 of 57 items pass, including out-of-scope refusals 3 of 3 and empty-result handling 7 of 7.
- **LLM comparison** ([`llm_summary.md`](eval/results/llm_summary.md)), 52 questions per prompt style and 9 explanation cases. With the base prompts, few-shot faithfulness is 0.615 (allam-2-7b), 0.962 (gpt-oss-20b) and 1.0 (qwen3.8-27b), and few-shot correctness is 0.577, 0.782 and 0.724 (qwen few-shot: 51 answers; in the strict variant three qwen calls failed on the Groq daily token cap). Explanation fidelity rises from 0.667, 0.333 and 0.296 (base) to 0.741, 0.963 and 0.963 (strict) for allam-2-7b, gpt-oss-20b and qwen3.8-27b.
- **Error analysis** ([`error_analysis.md`](eval/results/error_analysis.md)): no intent, entity or query failures on the 57 pipeline items; the remaining errors are model misreadings of the rows, and hallucinated content is concentrated in allam-2-7b.
