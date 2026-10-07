# CLAUDE.md: F1 Podium Prediction (AI Tools Lab, Milestone 1)

Handover from the previous Claude session. Read this first.

## The task
Predict whether a driver finishes on the podium (`positionOrder <= 3`) in an F1 race, using **only information available after qualifying and before lights out**. Unit = one driver entering one race. Data: Kaggle *Formula 1 World Championship (1950–2024)* (Ergast), 14 CSVs in `data/` (git-ignored).

The brief and the F1 primer are in `material/` (`MS1 description.pdf`, `intro to f1.pdf`). Re-read the brief before modeling. It requires:
- EDA before *and* after cleaning
- the 3 data-engineering questions
- ≥2 preprocessing-order experiments
- 2 statistical/traditional models + 1 shallow neural net
- feature-group ablation
- XAI (permutation importance, SHAP, local SHAP/LIME)
- metrics ROC-AUC, PR-AUC, F1
- model/threshold selection on train+val only; test untouched until the end

## Environment
- Windows laptop, CPU. Python venv at `.venv/` (`pip install -r requirements.txt`). Notebook run in VS Code.
- **VS Code autosave is on**: if the notebook is open and you edit the .ipynb on disk, VS Code overwrites your edit. Ask Ahmed to close the tab (or disable autosave / "File: Revert File") before editing the notebook file, then verify the change is on disk.
- `find_data_dir()` looks in `/kaggle/input/...`, `data/`, `../data`, `.`. Outputs go to `outputs/` (git-ignored).
- Don't commit/push unless asked.

## The notebook: `F1_podium_prediction.ipynb` (single Run-All notebook, ~156 cells)
Detective-style narrative: each EDA "clue" ends with a 🔎 Lead to the next. `note(clue, finding, decision)` logs every finding into `CASE_LOG`, printed at the end as a findings → decisions table (report material). Cells marked 🗃 came from Ahmed's first-pass EDA.

**Part I: EDA, Clues 0–15**
| Clue | Topic |
|---|---|
| 0 | Table coverage over time |
| 1 | `\N` sentinels + semantic typing on copies (`raw` stays untouched, typed copies in `df`) |
| 2 | PK/FK checks and cross-table (race, driver) checks |
| 3 | Shared drives → `collapse_shared_drives()` |
| 4 | Target/base rate, Indy 500 |
| 5 | Non-starters, `grid=0`, status mapping `classify_status()` |
| 6 | Qualifying coverage, grid≠quali, session reached |
| 7 | Grid AUC by era |
| 8 | Leakage audit of every column of every table + AUC screen + same-race vs before-race standings |
| 9 | Form features + alibi test |
| 10 | Reliability |
| 11 | Home (nationality→country map `NAT2COUNTRY`) |
| 12 | Driver turnover / cold start |
| 13 | Sprint (legal, but absent before 2021 → excluded) |
| 14 | Split audit |
| 15 | Before/after |

**Part II: Cleaning.** `build_base(df)` → `base`: one row per (raceId, driverId), starters only. Steps, each logged in `LEDGER`, with invariant asserts:
1. Collapse shared drives: features from the started car; label = podium in **any** car.
2. Drop the Indianapolis 500.
3. Drop non-starters (`status_cat == "non_start"`).
4. `grid=0` for starters → `grid_filled` = back of field, plus a `pit_lane_start` flag.

**Part III: DE questions** (each ends with a "✅ Answer" markdown cell generated from the numbers)
- **Q1:** dumbbell chart per circuit. ● = front-row *start* (grid 1–2), ◆ = front-row *qualifier* (quali 1–2), grey = pre-2014, red = 2014+. Circuits are ranked **within each period** (average of the two period rates). Never pool all seasons: conversion rose after 2014, so pooling favoured newer circuits (that bug made Yas Marina look best; Ahmed saw Suzuka is #1 in both periods).
- **Q2:** home effect tested with CMH over grid-band strata and a logit (`podium ~ home + C(grid_band) + C(era)`, errors clustered by driver). Plain-language explanation of odds/OR/CI/p, per-band table and forest plot.
- **Q3:** mechanical retirement rate per constructor with Wilson CIs (≥20 starts), plus a sensitivity check that counts ambiguous statuses as mechanical.

**Part IV: Features, selection, split, preprocessing**
- `build_features(base, df, history=None)` → `FX`. Every rolling feature uses `shift(1)` before rolling; standings come from `merge_asof(..., allow_exact_matches=False)` (the table *before* this race); constructor features are computed at constructor-race grain. Two alibi tests assert there is no leakage.
- `GROUPS`, which double as the ablation groups:
  - grid
  - qualifying
  - standings (points *share* of leader, not raw points)
  - driver_form (finish *percentile*)
  - constructor_form (incl. previous mechanical rate)
  - time_experience (driver_age, years_in_f1, races_with_team, days_since_last_race, season_progress)
  - context (home)
- Raw date/year deliberately **not** a feature (the time split makes test years out of range).
- `BANNED` = post-race columns; an assert checks none slipped in.
- Split: train ≤2019, val 2020–21, test ≥2022 (by race).
- Selection evidence on **train only**: univariate AUC (drop if <0.52 unless a support feature), redundancy |ρ|>0.9 flagged not dropped, then groups added cumulatively to a logistic regression scored on val → `FEATURES`, `FEATURE_GROUPS`.
- `F1Prep` (sklearn transformer, fit on train only): domain fills (missing `quali_pos` → grid; `grid_minus_quali` → 0) → missing-indicator columns (`INDICATOR_COLS`) → train median → standard scaling.
- Era/coverage fixes in `build_features`:
  - `session_reached` is NaN before 2006 (`KNOCKOUT_START`), because Q2/Q3 didn't exist; its `_missing` indicator is the era flag.
  - `q1_gap_pct` is **deliberately kept** before 2006: blanking it was tried and cost ~0.01 val PR-AUC, because the pre-2006 gap is the same signal.
  - Missing `prev_standing_pos` / `con_prev_standing_pos` after round 1 → behind the last listed entry, because old standings tables omit zero-point entries.
- Experiments, decided by **paired race-level bootstrap** of Δ val PR-AUC (`paired_bootstrap`). If the CI contains 0 it's a tie, settled by a tie-breaker stated in advance:
  - E1: prep fit on train vs all data → always train (the other leaks).
  - E2: filter→features vs features→filter → filter first.
  - E3: full history vs modern window (seasons with qualifying data) → modern on a tie.
  - E3b: modern vs knockout window (2006+) → modern on a tie.
  - Current result: all tie, so `best_window` = modern (1994).
- Hand-off files: `outputs/modeling_table.csv` (ids, `split`, `podium`, `FEATURES`) and `outputs/feature_spec.json` (features, groups, train_window_start, split, target).

## Status / what's next
Milestone 1 data work is done.
- **Run on real data:** full Run-All verified on real data (2026-10-07).
- **Loose ends closed (2026-10-07):**
  - status review: `other` bucket is empty, decisions in the Clue 5 markdown;
  - whitespace strip at typing fixed the unmapped `'Argentinian '`;
  - the 18 post-1957 duplicates are 3 double non-starts + 6 car swaps (1958–64), documented in Clue 3, along with the limitation that the "started car" tie-break uses laps.
- Brief says "all five metrics from Section 4", but only 3 are listed. Ask the TA (precision/recall?).

**Next: modeling**, appended to the same notebook as Part V+ (or a second notebook that loads `outputs/modeling_table.csv` + `feature_spec.json`):
1. Baseline: grid-only logistic regression.
2. Two statistical models (e.g. regularised logistic regression; gradient boosting such as HistGradientBoosting/LightGBM). Tune on val, class imbalance via weights, threshold chosen on train/val.
3. Shallow NN (MLP, 1–2 hidden layers), with architecture/hyperparameter experiments.
4. Compare on ROC-AUC, PR-AUC (primary, rare class ~14%) and F1. Test only at the end.
5. Ablation: remove one `FEATURE_GROUPS` group at a time.
6. XAI: permutation importance, SHAP (global + local), LIME. Importance ≠ causation.
7. Inference function for a new race.

## How Ahmed likes to work
- Casual, direct, wants the real-world return on effort. Answer his questions plainly and give explicit written conclusions ("✅ Answer" cells), not just charts.
- **When he asks for a small change, make a small change.** He was annoyed when a requested tweak to the Q1 chart turned into a full redesign.
- Explain stats in plain words (he asked what CMH/OR/p meant, and how to decide when numbers look tied).
- Check your own outputs on the real data before saying something works.
