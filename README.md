# Formula 1 Podium Prediction

AI Tools Lab, Milestone 1, Team 97: Ahmed Tamer Mohamed Hany Saleh Elmeazawy (58-14045), Youssef Mohamed Mohamed Elkhodary (58-18247), Omar Gamaleldin Mohamed Abdelbaky (58-13583), Omar Hisham Abdelrahman Mahrous Abdelrahman (58-25441).

For every driver in every Formula 1 race we predict whether the driver finishes on the podium (`positionOrder <= 3`), using only what is known after qualifying and before the start. The project covers the whole pipeline: auditing and cleaning the 14 relational tables of the Kaggle/Ergast dataset, three data-engineering questions, a leakage-safe driver–race modeling table, feature selection by experiment, preprocessing, three models (logistic regression, random forest, shallow neural network), a feature-group ablation, explainability (permutation importance, SHAP) and an inference function.

- **Notebook:** [`F1_podium_prediction.ipynb`](F1_podium_prediction.ipynb), one notebook that runs top to bottom.
- **Report:** [`report/report.pdf`](report/report.pdf) (source: `report/report.tex`).

## Results

Final models on the test seasons 2022–2024 (68 races, never used for any choice). Thresholds and hyperparameters were chosen on the validation seasons 2020–2021.

| Model | ROC-AUC | PR-AUC | F1 | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Shallow neural network (2 hidden layers) | **0.941** | **0.714** | **0.698** | **0.638** | 0.770 |
| Logistic regression (baseline) | 0.933 | 0.701 | 0.690 | 0.602 | 0.809 |
| Random forest | 0.935 | 0.696 | 0.678 | 0.580 | **0.814** |

PR-AUC is the primary metric (podiums are 13–15% of rows), ROC-AUC the secondary one. A random ranking scores about 0.15 PR-AUC on these seasons.

Data-engineering questions, in short:

1. **Front-row conversion.** Since 2014 Suzuka (89%), Yas Marina (86%) and Monaco (85%) turn a front-row start into a podium most often. The circuit ranking before and after 2014 is unrelated (rank correlation −0.03), and front-row starts convert far more often now (56% → 76%).
2. **Home advantage.** None once the starting position is controlled for: pooled odds ratio 0.95 (95% CI 0.80–1.13).
3. **Mechanical retirements.** Highest rates in 2014–2021: Caterham, Lotus F1, Toro Rosso; in 2022–2024: Alfa Romeo, Alpine, AlphaTauri. Overall the rate fell from 9.5% to 5.8% of starts.

## The notebook

| Section | Content |
| --- | --- |
| 1 | setup: locates or downloads the data, fixes the random seed |
| 2 | audit of the raw tables before any change: column profiles, missing-value markers, categories, duration formats |
| 3 | cleaning: typing, keys and validity rules, points reconciliation, shared drives (one row per driver and race), who counts as a starter, missing-values map |
| 4 | EDA after cleaning |
| 5 | the three data-engineering questions, each with its figure and robustness checks |
| 6 | feature engineering: leakage screen, 28 candidate features, automatic leakage checks |
| 7 | split by season and how data decisions are evaluated (three validation windows, measured noise level) |
| 8 | feature selection by experiment |
| 9 | preprocessing trials, the final pipeline and its effect, preprocessing-order experiments |
| 10 | the modeling table and evaluation helpers |
| 11–13 | logistic regression, random forest, shallow neural network, comparison on train, validation and test |
| 14 | feature-group ablation |
| 15 | explainability: permutation importance, SHAP (global and local) |
| 16 | inference function, shown on a complete and a partially missing record |
| 17 | assumptions and limitations |

Every run writes the report figures to `report/figures/` and the modeling table and feature list to `outputs/`.

## Running it

### On Kaggle

1. Create a new notebook and import `F1_podium_prediction.ipynb` (File → Import Notebook).
2. Add the dataset *Formula 1 World Championship (1950 - 2024)* by rohanrao (Add Input), or turn Internet on in the notebook settings; the notebook then downloads the data itself.
3. Run All. It needs nothing beyond Kaggle's standard image (pandas, scikit-learn, statsmodels, PyTorch, SHAP) and takes about 4 minutes on a 4-core CPU session.

The committed notebook already contains the outputs the report quotes. A rerun with other library versions gives the same data sections and the same logistic regression, but the random forest, the gradient-boosting trials and the neural network can move by about 0.01 PR-AUC (the notebook says so in sections 7.1, 11.2 and 12); every conclusion stays the same. To publish exactly the outputs of the report, import the executed notebook and save the version without rerunning it (Save Version → Quick Save).

### Locally

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter lab                         # or open the notebook in VS Code with the .venv kernel
```

Run All takes about 5 minutes on a laptop CPU. The first run downloads the dataset into `data/` (git-ignored). To get it by hand instead, download the zip from the [dataset page](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020) and extract the CSVs into `data/`, or:

```bash
curl -L -o f1.zip "https://www.kaggle.com/api/v1/datasets/download/rohanrao/formula-1-world-championship-1950-2020"
unzip f1.zip -d data
```

The notebook has been run top to bottom with Python 3.14 (pandas 3.0, scikit-learn 1.9, PyTorch 2.14, SHAP 0.54) and, from an empty folder that downloads the data, with Python 3.11 (pandas 2.2, scikit-learn 1.2, PyTorch 2.5, SHAP 0.44).

### The report

```bash
cd report
pdflatex report.tex && pdflatex report.tex
```

The report reads its figures from `report/figures/`, which every Run All refreshes. The numbers in its text are written by hand, so check them against the notebook after any change.

## Repository layout

| Path | Content |
| --- | --- |
| `F1_podium_prediction.ipynb` | the notebook |
| `report/` | the report (LaTeX source and PDF) and the figures the notebook saves |
| `material/` | the milestone description and the F1 introduction from the course |
| `requirements.txt` | Python packages |
| `data/` | the 14 CSVs, downloaded on the first run (git-ignored) |
| `outputs/` | modeling table and feature list written by the notebook (git-ignored) |

## Data

[Formula 1 World Championship (1950–2024)](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020) on Kaggle, built from the Ergast API: 14 tables, 1,125 races, 26,759 result rows. The notebook uses the dataset's final version (29 January 2025).
