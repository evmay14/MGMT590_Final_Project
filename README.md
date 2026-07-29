# MGMT 590 — LendingClub Loan Default Risk Capstone (Indiana Borrowers)

Graduate Business Analytics capstone (MGMT 59000, Summer 2026, Section
DY2, Purdue University). Predicts loan default risk for Indiana
LendingClub borrowers and supports lending decisions through a Streamlit
decision-support application.

This repository is built incrementally across six phases. **This is
Phase 1: Project Architecture & Data Foundation.**

## Phase 1 deliverables

- GitHub-ready project structure
- `requirements.txt`
- `src/config.py` — all paths, constants, and column definitions
- `src/utils.py` — reusable ingestion, validation, cleaning, preprocessing,
  splitting, and serialization functions
- `src/train_models.py` — Phase 1 orchestration (`run_phase1_pipeline`)
  plus fixed-signature Phase 3 stubs (model training not yet implemented)
- `notebooks/MGMT590_LendingClub_Analysis.ipynb` — executed walkthrough of
  the full Phase 1 pipeline
- Leakage-safe train/validation/test split
- Serialized (`joblib`) `ColumnTransformer` preprocessing pipeline

## Phase 2 deliverables (this commit)

- `src/eda_utils.py` — reusable, additive module (does not modify Phase 1
  files) providing:
  - Dataset overview / variable-description helpers
  - Extended descriptive statistics (mean, median, std, variance, min,
    max, quartiles, skewness, kurtosis) and categorical frequency tables
  - A consistent, business-styled plotting library (distributions,
    boxplots, violin plots, scatter/hexbin, correlation heatmap,
    pairplot, missing-value heatmap/bar, outlier grids)
  - Default-rate-by-group analysis with quartile/band binning helpers
  - Statistical testing: Pearson/Spearman correlation, chi-square,
    Welch's t-test, one-way ANOVA, Wilson-score confidence intervals —
    each returned as a standardized `TestResult` with null/alternative
    hypotheses, statistic, p-value, effect size, and a plain-language
    business interpretation
  - Multicollinearity diagnostics: high-correlation-pair detection and
    Variance Inflation Factors (VIF)
- `notebooks/MGMT590_LendingClub_EDA_Phase2.ipynb` — executed, executive-
  quality EDA notebook covering dataset overview, descriptive statistics,
  30+ visualizations, default-rate analysis, all seven research
  questions, statistical testing, feature-relationship/multicollinearity
  assessment, and a Phase 3 hand-off summary. **No modeling is performed
  in this notebook** — see `src/train_models.py`'s Phase 3 stubs for
  where that logic will live in Phase 3.
- `tests/test_eda_utils.py` — unit tests for every statistical/table
  function in `eda_utils.py` (plotting functions are smoke-tested)

## Project structure

```
mgmt590_capstone/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── config.py            # paths, constants, column groups
│   ├── utils.py             # ingestion / validation / cleaning / pipeline / split / serialization
│   └── train_models.py      # Phase 1 orchestration + Phase 3 stubs
├── notebooks/
│   └── MGMT590_LendingClub_Analysis.ipynb
├── data/
│   ├── raw/                 # place the real LendingClub Indiana extract here
│   ├── processed/           # cleaned dataset (generated)
│   └── splits/              # X/y train/val/test CSVs (generated)
├── models/                  # serialized trained models (Phase 3+)
├── pipelines/               # serialized fitted preprocessing pipeline
├── logs/                    # pipeline run logs
├── app/                     # Streamlit application (Phase 5/6)
└── tests/
    ├── generate_synthetic_fixture.py  # synthetic data for local pipeline testing only
    └── build_notebook.py               # regenerates the analysis notebook
```

## Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Data

Place the real Indiana LendingClub extract (~37,515 rows, ~27.5 MB) at:

```
data/raw/lendingclub_indiana_raw.csv
```

If you don't have the file yet and want to verify the pipeline runs,
generate a small **synthetic** fixture (NOT real data, for pipeline
testing only):

```bash
python tests/generate_synthetic_fixture.py
```

## Running the Phase 1 pipeline

```bash
python -m src.train_models
```

This ingests, validates, cleans, splits, and fits/serializes the
preprocessing pipeline, logging every step to `logs/pipeline.log` and
printing a summary. Equivalently, open and run
`notebooks/MGMT590_LendingClub_Analysis.ipynb`.

Programmatic usage:

```python
from src.train_models import run_phase1_pipeline

artifacts = run_phase1_pipeline()
artifacts.X_train, artifacts.y_train   # ready for Phase 3 model training
artifacts.preprocessor                  # fitted ColumnTransformer
```

## Target variable

`default_flag` (binary):
- `1` — loan status was `Charged Off` or `Default`
- `0` — loan status was `Fully Paid`
- All other statuses (e.g. `Current`, `Late (31-120 days)`,
  `In Grace Period`) are **excluded** — those loans have not reached a
  final resolution.

## Roadmap

| Phase | Scope |
|---|---|
| 1 | Architecture, ingestion, validation, cleaning, preprocessing pipeline, leakage-safe split |
| 2 (this commit) | Exploratory data analysis, descriptive statistics, default-rate analysis, research-question analysis, statistical testing, feature-relationship assessment |
| 3 | Model training: Logistic Regression → Random Forest → XGBoost |
| 4 | Model evaluation, comparison, and selection |
| 5 | Streamlit decision-support application |
| 6 | Deployment, documentation, final report |

## Notes for future phases

- Import shared logic from `src.config` and `src.utils` rather than
  re-implementing it.
- The preprocessing pipeline in `pipelines/preprocessing_pipeline.joblib`
  is fit on `X_train` only — always `.transform()` (never re-`.fit()`) it
  on validation/test/live data.
- `src/train_models.py` already declares the Phase 3 function signatures
  (`train_logistic_regression`, `train_random_forest`, `train_xgboost`,
  `evaluate_model`) — implement their bodies in Phase 3 rather than
  restructuring the module.
