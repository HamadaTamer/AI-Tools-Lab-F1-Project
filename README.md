# AI Tools Lab — Formula 1 Podium Prediction

This repository contains our **AI Tools Lab project** for predicting Formula 1 podium finishes using historical race-weekend data.

The goal is to build a complete data-engineering and machine-learning pipeline that predicts whether a driver will finish on the podium using only information available **after qualifying and before the race starts**.

## Project Overview

The project uses the Formula 1 World Championship dataset from Kaggle / Ergast, covering historical Formula 1 data from 1950–2024.

The prediction unit is:

**One driver entering one race**

The target is:

**Podium finish = finishing position ≤ 3**

A key requirement is avoiding **data leakage**. Information only known after the race, such as current-race results, points, race status, laps, and pit stops, cannot be used as prediction features.

## Main Project Areas

### 1. Data Cleaning & Analysis
- Exploratory Data Analysis
- Missing-value and sentinel handling
- Data type conversion
- Duplicate driver-race handling
- Primary and foreign key validation
- Required data-engineering questions
- Visualizations and interpretation

### 2. Feature Engineering & Selection
- Construction of the driver-race modeling table
- Leakage-safe historical features
- Driver and constructor form features
- Qualifying and grid features
- Feature selection
- Preprocessing
- Train / validation / test splitting

### 3. Statistical Models
- Statistical and traditional machine-learning models
- Hyperparameter tuning
- Model comparison
- Feature-group ablation
- Performance evaluation

### 4. Neural Networks
- Shallow feed-forward neural network
- Architecture and hyperparameter experiments
- Training and validation
- Comparison against statistical models

### 5. Explainable AI (XAI)
- Permutation importance
- SHAP
- Local prediction explanations using SHAP and/or LIME
- Interpretation of model behavior

## Evaluation

Models will be evaluated using:

- ROC-AUC
- PR-AUC
- F1 Score

Model selection, hyperparameter tuning, and threshold selection will use training and validation data only. The test set will remain unseen until final evaluation.

## Dataset

**Formula 1 World Championship (1950–2024)**  
Source: Kaggle / Ergast

The dataset contains multiple relational tables covering drivers, constructors, races, circuits, qualifying, results, standings, lap times, pit stops, and other Formula 1 data.

## Repository layout

| Path | What it is |
| --- | --- |
| `F1_podium_prediction.ipynb` | the single run-all notebook the whole team works in (sections 1–10 done, 11–16 to do, 17 started) |
| `team/HANDOFF.md` | what the data sections provide, how to work in the shared notebook, the rules, and what each remaining section must contain |
| `report/report.md`, `report/figures/` | report draft (sections 1–7) and the figures the notebook saves on every run |
| `material/` | the milestone description and the F1 introduction |
| `data/` | the 14 CSVs, downloaded by the notebook on its first run (git-ignored) |
| `outputs/` | modeling table and feature spec written by the notebook (git-ignored) |
| `.gitattributes` | tells git to merge `.ipynb` files with nbdime (see below) |

## Development

The notebook `F1_podium_prediction.ipynb` runs top to bottom (Run All, about 75 seconds on a laptop CPU) either **locally** or on **Kaggle**. It looks for the data in `data/`, then under `/kaggle/input/`, and otherwise downloads the latest version of the Kaggle dataset into `data/` (29 January 2025, the author's final update: 1950–2024). Every run overwrites the figures in `report/figures/` and the files in `outputs/`. It has been run with Python 3.14 / pandas 3.0 / scikit-learn 1.9 and with Python 3.11 / pandas 2.2 / scikit-learn 1.5.

### Run locally (any OS, venv)

From the repo root:

```powershell
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The `.venv/` folder is git-ignored (virtual environments aren't portable), so each person creates their own with the commands above. In VS Code, pick the `.venv` interpreter as the notebook kernel (Select Kernel → Python Environments → `.venv`).

<details>
<summary>Alternative: conda</summary>

```powershell
conda create -n f1 python=3.11 -y
conda activate f1
pip install -r requirements.txt
```
</details>

The data needs no manual step: the first Run All downloads it into `data/` (git-ignored, so the CSVs are never committed). To get it by hand instead, any of:

- **Direct download, no Kaggle account needed:**
  ```bash
  curl -L -o f1.zip "https://www.kaggle.com/api/v1/datasets/download/rohanrao/formula-1-world-championship-1950-2020"
  unzip f1.zip -d data
  ```
  ```powershell
  Invoke-WebRequest "https://www.kaggle.com/api/v1/datasets/download/rohanrao/formula-1-world-championship-1950-2020" -OutFile f1.zip
  Expand-Archive f1.zip data
  ```
- **Browser:** download the zip from https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020 and extract the CSVs into `data/` (so you have `data/results.csv`, `data/races.csv`, ...).
- **Kaggle API** (needs `kaggle.json` from Kaggle → Settings → API):
  ```powershell
  kaggle datasets download -d rohanrao/formula-1-world-championship-1950-2020 -p data --unzip
  ```

Then open the notebook in VS Code (select the `.venv` kernel) or run `jupyter lab`, and **Run All**. The notebook writes the modeling table to `outputs/` and the report figures to `report/figures/`.

### Run on Kaggle

Upload the notebook and attach the dataset *Formula 1 World Championship (1950 - 2024)* (or turn on Internet in the notebook settings and it downloads the data itself). Either *Save & Run All*, or upload the notebook already executed locally and use *Save Version → Quick Save* to keep its outputs exactly as they are. The gradient-boosting numbers can differ slightly between scikit-learn versions (the notebook explains this in section 7.1); every decision stays the same. On Kaggle the figures and the modeling table go to the notebook's Output tab, not to this repository.

### Working on the remaining sections

The whole team works in the one notebook, each person in their own section, on their own branch. Once per clone, after installing the requirements:

```bash
git-nbmergedriver config --enable
```

This makes git merge notebooks cell by cell with nbdime (the committed `.gitattributes` points `.ipynb` merges to it; merge with the venv activated). Without it, two branches that both ran the notebook conflict on almost every cell. The full workflow, the section owners and what each section must contain are in `team/HANDOFF.md`.

## Status

Milestone 1, deadline 18 October 2026.

- Done: data audit and cleaning, EDA before and after cleaning, the three data-engineering questions, feature engineering and selection, split, preprocessing and preprocessing-order experiments, report sections 1–7.
- To do: models and baseline, shallow neural network, model comparison, feature-group ablation, XAI, inference function, report sections 8–10 and PDF, Kaggle and GitHub links.
