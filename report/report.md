# Formula 1 Podium Prediction — Milestone 1 Report

*Draft. Sections 1–7 cover data cleaning, the data-engineering questions, feature engineering, the split and preprocessing. Sections 8–10 are completed with the modeling work. Every number comes from `F1_podium_prediction.ipynb`; section numbers in brackets (§) point to the notebook.*

## 1. Task and data

For every driver in every race we predict whether the driver finishes on the podium (`positionOrder <= 3`), using only information available after qualifying and before the start. One row of the modeling table is one driver in one race, so `(raceId, driverId)` must be unique.

The data is the Kaggle / Ergast Formula 1 dataset, 1950–2024: 14 relational tables, 26,759 result rows in 1,125 races. Not every table covers every season: qualifying starts in 1994, lap times in 1996, pit stops in 2011 and sprint results in 2021 (§2).

![Seasons covered by each table](figures/table_coverage.png)

## 2. Data cleaning

Before cleaning (§2.3), the result rows per race peak at about 39 in the late 1970s–80s, the podium share dips exactly there, and `grid = 0` is a spike of 1,638 rows that mixes drivers who never started with pit-lane starts.

![Raw rows per race, podium share and grid values](figures/raw_target_grid.png)

| Issue (how it was found) | Decision | Evidence |
| --- | --- | --- |
| Missing values stored as the text `\N` in 30 columns, and as empty strings in `q2`/`q3` for 2023–2024 | Read both as missing; nothing else is treated as missing | sentinel count per column (§2.1) |
| Dates, lap times, qualifying times and pit-stop durations stored as text, in three formats (`23.456`, `1:26.572`, `2:05:05.152`) | One parser to seconds; dates to `datetime` | parsed lap times match the `milliseconds` column on all 589,081 laps (§3.1) |
| One impossible qualifying time (`16:42.640`, 1995 Japanese GP, pole ≈ 98 s) | Times above 150% of pole set to missing (only this one) | gap-to-pole distribution: 99.9% of times are within 19.4% of pole (§3.1) |
| Primary and foreign keys | All unique, complete and reconciled; no change needed | key checks on all 14 tables (§3.2) |
| 91 extra rows on `(raceId, driverId)`: 85 shared drives, 1950–1978 | Keep the row of the car the driver started in; label it a podium if the driver finished on the podium in any car (rule D) | four rules compared: keeping the best result takes the grid slot of the wrong car in 68 of 85 pairs; keeping only the own car's result loses 17 credited podiums (§3.3) |
| `grid = 0` mixes 1,563 entries that never started with 72 pit-lane starts | Keep a row if the driver had a grid slot or completed a lap; pit-lane starts placed at the back of the grid with a flag | none of the dropped rows is a podium; dropping by status letter or by `grid = 0` alone keeps non-starters or loses pit-lane starts (§3.4) |
| Indianapolis 500 (1950–1960) | Dropped | 103 of its 107 drivers never raced another championship race (§3.4) |

Result: 26,759 result rows → 24,724 driver–race rows (podium share 13.6%).

After cleaning, the podium share in every season follows `3 / starters per race`; the dip before 1995 came from the non-starters (§4.1).

![Effect of cleaning](figures/cleaning_effect.png)

Other findings that shape the later steps:

- **The grid is the strongest signal, and it drifts.** A pole position converted to a podium 53% of the time in 1950–69 and 81% in 2014–24 (§4.2).
- **Qualifying coverage is patchy.** There is no qualifying data before 1994, it is complete in 1994–95, covers only 6–59% of starters in 1996–2002, and is complete from 2003 on (§4.3).
- **Reliability changed.** About half of the starters finished a race until the 1990s, against 86% in the 2020s (§4.4).

![Podium rate by grid position, per era](figures/podium_by_grid.png)

![Qualifying coverage](figures/qualifying_coverage.png)

![Share of starters who reached the finish](figures/finish_rate.png)

## 3. Data-engineering questions

### Q1. Which circuits convert a front-row start into a podium most often, and does the ranking change after 2014?

**Grain:** one driver starting from the front row (grid 1–2) at one race. **Joins:** results → races → circuits, plus qualifying. **Missingness:** grid is complete; the qualifying definition only exists from 1994 and is analysed separately. Circuits need ≥ 10 front-row starts in a period to be ranked.

- Since 2014 the best converters are Suzuka (89%), Yas Marina (86%), Monaco (85%), Barcelona and Spa (82%). Before 2014 the same circuits were led by Suzuka (70%), Hungaroring (68%) and Bahrain (67%).
- The ranking changed completely: the rank correlation between the periods is −0.03 (−0.07 when "before" is limited to 1994–2013). Hungaroring fell from 2nd to 16th; Monaco rose from 15th to 3rd.
- Conversion rose almost everywhere (56% → 76% overall), in line with better reliability.
- **Qualifying vs grid:** 36 of 986 front-row qualifiers since 1994 did not start from the front row (penalties), and only 31% of them reached the podium. The circuit ranking hardly depends on the definition (rank correlation 0.90).
- With 14–26 front-row starts per circuit since 2014, one podium moves a rate by 4–7 points, so neighbouring ranks are not meaningfully different.

![Q1](figures/q1_front_row_conversion.png)

### Q2. Do drivers podium more often at home, after controlling for starting position?

**Grain:** one starter at one race. **Joins:** results → drivers (nationality) and results → races → circuits (country), linked by a hand-made nationality → country map. **Method:** compare home and away drivers within the same grid band, pooled with the Cochran–Mantel–Haenszel test.

- Without control, home drivers podium less often (11.8% vs 13.7%, 1,944 home starts).
- Controlling for the grid: pooled odds ratio 0.95 (95% CI 0.80–1.13, p = 0.58). **No evidence of a home advantage.** The result is the same with exact grid positions (0.96) and with the Indianapolis 500 included (0.96).
- In 1994–2024 alone the odds ratio is 0.70 (CI 0.51–0.95, p = 0.025), i.e. slightly *worse* at home; as one of four variants this is weak evidence.

![Q2](figures/q2_home_by_grid_band.png)

### Q3. Which constructors have the highest mechanical-retirement rate in 2014–2021 vs 2022–2024?

**Grain:** one car that started one race. **Joins:** results → status → constructors, results → races. **Mechanical** = a component failed; the 139 status values were grouped by hand, with an *unclear* group (tyre, puncture, wing, wheel, damage, the bare "Retired") checked separately. Teams with fewer than 40 starts in a period are not ranked.

- **2014–2021:** Lotus F1 17.3% of starts (13/75), Toro Rosso 16.1% (39/242), McLaren 13.5% (43/318), Renault 13.5% (27/200). Mercedes was the most reliable of the teams present in all eight seasons (3.8%).
- **2022–2024:** Alfa Romeo 12.5% (11/88), Alpine 11.0% (15/136), AlphaTauri 8.0% (7/88); most teams are between 2% and 6%.
- Mechanical retirements fell from 9.1% to 5.8% of starts. Counting the unclear statuses as mechanical leaves the leaders unchanged.
- Renamed teams count separately (e.g. Alfa Romeo 2022–23 and Sauber 2024 are the same team: 8.8% together).

![Q3](figures/q3_mechanical_rate.png)

## 4. Features

Every candidate column was screened against the cutoff (§6.1): race results, points, status, laps, pit stops, lap times and same-race standings are excluded. Standings are taken after the previous round; form features use only earlier races. Three automatic checks confirm this (§6.3): a driver's first start has no form, round 1 has no standings, and the form of 300 random rows is recomputed by hand from earlier races.

Selection used the training seasons only: the single-feature relationship with the podium (§8.1), redundancy (§8.2), and trials in which feature groups are added one at a time (§8.3).

| Group | Feature | Why it is valid at the cutoff | Evidence |
| --- | --- | --- | --- |
| grid | `grid_pos`, `pit_lane_start` | the starting grid is fixed before the start | strongest signal in every era |
| qualifying | `quali_pos`, `q1_gap_pct` (gap to the fastest Q1 time) | qualifying ends before the race | +0.038 PR-AUC on top of the grid |
| standings | `drv_standing`, `con_standing` | championship positions after the previous round | +0.011 (logistic regression), +0.049 (gradient boosting) |
| driver form | podium rate and finishing percentile over the last 3 starts, career podium rate | computed from earlier races only | +0.022 (gradient boosting) |

![Podium rate across each feature](figures/feature_effects.png)

Not selected: raw championship points (the points system changed; −0.026), constructor form, driver experience, field size and the home flag (gains within the noise level), `grid − qualifying` (a linear combination of two kept features), sprint results (no sprint before 2021, so no training rows have one).

## 5. Data split

By season: **train 1950–2019 (22,589 rows), validation 2020–2021 (779), test 2022–2024 (1,356).** The model must predict future races from past ones, every race stays in a single split, and the test set is the new 2022 regulation era. The podium rate is 13.4% in train and 15.0% in validation and test.

Data decisions were evaluated without touching the test set, on three two-season validation windows (2016–17, 2018–19, 2020–21), with PR-AUC as the main metric. A change was accepted only if it beat the reference by more than the measured noise level (the spread of scores when 1% of the training rows are dropped at random: 0.005 for logistic regression, 0.011 for gradient boosting) and in at least 2 of the 3 windows. Training on all seasons beat every shorter training window (§8.3).

## 6. Preprocessing

Median imputation with missing-value indicators, then standard scaling, both fitted on the training rows only. Median + indicators beat median only, mean only and filling qualifying with the grid slot (§9.1). Clipping or log-transforming the qualifying gap and the choice of pit-lane encoding made no difference beyond the noise level.

![Effect of preprocessing](figures/preprocessing_effect.png)

**Preprocessing-order experiments (§9.3)**

| Experiment | Orders compared | Result | Decision |
| --- | --- | --- | --- |
| E1 | fit imputer/scaler after the split (train only) vs before it (all rows) | +0.0003 for the leaky order, far below the noise level | split first: the leaky order uses statistics from the evaluation seasons |
| E2 | impute → scale vs scale → impute | tie (−0.0004); scale → impute leaves imputed features with a standard deviation of 0.61 instead of 1 | impute → scale |
| E3 | build form features after vs before removing non-starters | tie (+0.0001), but 55% of rows get different form values | filter first, so form describes races the driver started |

## 7. Modeling table

`outputs/model_table.csv`: 24,724 rows × 9 features, target `podium`, split column; `outputs/feature_spec.json` lists the features and the 4 groups used by the ablation.

## 8. Models and comparison

*To be completed: baseline, at least two statistical models, a shallow neural network, hyperparameters and threshold chosen on validation, all metrics on train/validation/test, primary metric (recommended: PR-AUC) and secondary metric (ROC-AUC).*

## 9. Ablation and explainability

*To be completed: feature-group ablation (grid, qualifying, standings, driver form); permutation importance and SHAP; one local explanation (LIME or SHAP waterfall). Importance shows what the model relies on; it is not a causal effect.*

## 10. Limitations

The assumptions behind the data decisions are listed, with their evidence, in notebook §17.

- **Qualifying coverage drift:** 63% of the training rows have imputed qualifying features (none before 1994, partial in 1996–2002).
- **Era sensitivity:** the grid-to-podium relationship and reliability changed strongly over 75 years; the test seasons are a new regulation era.
- **Renamed teams** start with no history (standings and form follow `constructorId`).
- **Manual mappings** (status groups, nationality → country) are judgement calls; Q3 includes a sensitivity check.
- **Small validation windows** (about 40 races each): differences below the noise level are treated as ties and decided on domain grounds.
- *Modeling limitations to be added.*
