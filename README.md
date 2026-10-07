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

## Development

The notebook `F1_podium_prediction.ipynb` runs top to bottom (Run All) either **locally on CPU** or on **Kaggle**. It finds the data automatically in `data/` or under `/kaggle/input/`. GitHub is used for version control.

### Run locally (any OS, venv)

Needs Python 3.11–3.13. From the repo root:

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

Get the data into `data/` (the folder is git-ignored, so the CSVs are never committed). Either:

- **Browser:** download the zip from https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020 and extract the CSVs into `data/` (so you have `data/results.csv`, `data/races.csv`, ...), or
- **Kaggle API** (needs `kaggle.json` from Kaggle → Settings → API, placed in `%USERPROFILE%\.kaggle\`):
  ```powershell
  kaggle datasets download -d rohanrao/formula-1-world-championship-1950-2020 -p data --unzip
  ```

Then open the notebook in VS Code (select the `.venv` kernel) or run `jupyter lab`, and **Run All**. Outputs (tables, CSVs, the modeling table) are written to `outputs/`.

### Run on Kaggle

Upload the notebook, attach the dataset *Formula 1 World Championship (1950 - 2024)*, add `statsmodels` if it is missing, and Run All.

## Status

Milestone 1 — In Progress
