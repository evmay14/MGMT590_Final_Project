"""
config.py
=========
Central configuration module for the MGMT 590 LendingClub Loan Default
Risk capstone project.

This module is the SINGLE SOURCE OF TRUTH for:
    - File / directory paths
    - Column name conventions
    - Data cleaning parameters (mappings, valid categories, thresholds)
    - Train / validation / test split parameters
    - Modeling constants (random seed, target column name)
    - Logging configuration

Design intent
-------------
Every later phase (EDA, model training, evaluation, Streamlit deployment)
should import shared constants from this module instead of hard-coding
values. This guarantees that a change made in one place (e.g. moving the
raw data file, changing the random seed, or adding a new categorical
column) propagates consistently through the entire pipeline.

Nothing in this file performs I/O or computation — it is pure
configuration and should have no side effects on import.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

# ---------------------------------------------------------------------------
# 1. PROJECT ROOT & DIRECTORY STRUCTURE
# ---------------------------------------------------------------------------
# PROJECT_ROOT resolves to the top-level project folder regardless of the
# current working directory from which a script/notebook is launched, as
# long as this file stays at <PROJECT_ROOT>/src/config.py.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
SPLITS_DIR: Path = DATA_DIR / "splits"

MODELS_DIR: Path = PROJECT_ROOT / "models"
PIPELINES_DIR: Path = PROJECT_ROOT / "pipelines"
LOGS_DIR: Path = PROJECT_ROOT / "logs"
NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"
APP_DIR: Path = PROJECT_ROOT / "app"

# Directories that must exist before any pipeline step runs. Created lazily
# (idempotently) by `ensure_directories()` in utils.py rather than at
# import time, so importing config.py never touches the filesystem.
REQUIRED_DIRS: List[Path] = [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    SPLITS_DIR,
    MODELS_DIR,
    PIPELINES_DIR,
    LOGS_DIR,
    NOTEBOOKS_DIR,
    APP_DIR,
]

# ---------------------------------------------------------------------------
# 2. FILE PATHS
# ---------------------------------------------------------------------------
# Raw input file. The project brief specifies the Indiana-only LendingClub
# extract (~37,515 rows, ~27.5 MB). Replace this file with the real export
# before running the pipeline end-to-end.
RAW_DATA_FILENAME: str = "lendingclub_indiana_raw.csv"
RAW_DATA_PATH: Path = RAW_DATA_DIR / RAW_DATA_FILENAME

# Cleaned / feature-ready dataset (post validation + cleaning, pre-split).
CLEANED_DATA_PATH: Path = PROCESSED_DATA_DIR / "lendingclub_indiana_cleaned.csv"

# Train / validation / test split outputs (features and target saved
# separately to make downstream loading explicit and leakage-safe).
X_TRAIN_PATH: Path = SPLITS_DIR / "X_train.csv"
X_VAL_PATH: Path = SPLITS_DIR / "X_val.csv"
X_TEST_PATH: Path = SPLITS_DIR / "X_test.csv"
Y_TRAIN_PATH: Path = SPLITS_DIR / "y_train.csv"
Y_VAL_PATH: Path = SPLITS_DIR / "y_val.csv"
Y_TEST_PATH: Path = SPLITS_DIR / "y_test.csv"

# Serialized fitted preprocessing pipeline (joblib). Fit ONLY on training
# data; reused (via .transform) on validation/test/live inference data to
# prevent data leakage.
PREPROCESSOR_PATH: Path = PIPELINES_DIR / "preprocessing_pipeline.joblib"

# Serialized trained models (populated in Phase 2 — Logistic Regression,
# Random Forest, XGBoost). Filenames fixed now so later phases and the
# Streamlit app can reference stable paths.
LOGISTIC_REGRESSION_MODEL_PATH: Path = MODELS_DIR / "logistic_regression_model.joblib"
RANDOM_FOREST_MODEL_PATH: Path = MODELS_DIR / "random_forest_model.joblib"
XGBOOST_MODEL_PATH: Path = MODELS_DIR / "xgboost_model.joblib"

# Log file for the full pipeline run.
PIPELINE_LOG_PATH: Path = LOGS_DIR / "pipeline.log"

# ---------------------------------------------------------------------------
# 3. REPRODUCIBILITY
# ---------------------------------------------------------------------------
RANDOM_STATE: int = 42

# ---------------------------------------------------------------------------
# 4. TARGET / FILTERING DEFINITIONS
# ---------------------------------------------------------------------------
# Only Indiana borrowers are in scope for this project.
TARGET_STATE: str = "IN"
STATE_COLUMN: str = "addr_state"

# Raw loan_status values and how they map onto the binary target.
# Any loan_status value NOT present in this mapping is dropped (e.g.
# "Current", "In Grace Period", "Late (16-30 days)", "Late (31-120 days)")
# because those loans have not reached a final resolution and including
# them would leak future information / mislabel censored outcomes.
LOAN_STATUS_COLUMN: str = "loan_status"
TARGET_COLUMN: str = "default_flag"

LOAN_STATUS_TARGET_MAP: Dict[str, int] = {
    "Fully Paid": 0,
    "Charged Off": 1,
    "Default": 1,
}

# ---------------------------------------------------------------------------
# 5. COLUMN GROUPS
# ---------------------------------------------------------------------------
# These lists drive both the cleaning functions in utils.py and the
# ColumnTransformer built in build_preprocessing_pipeline(). Keeping the
# groups here (rather than inferring dtypes dynamically at runtime) makes
# the pipeline deterministic and easy to audit.

# Columns that arrive as "12.5%"-style strings and must be converted to
# numeric floats (e.g. 12.5) before modeling.
PERCENTAGE_COLUMNS: List[str] = [
    "int_rate",
    "revol_util",
]

# Raw employment-length column (e.g. "10+ years", "< 1 year", "3 years")
# and the numeric column it is parsed into.
EMP_LENGTH_RAW_COLUMN: str = "emp_length"
EMP_LENGTH_NUMERIC_COLUMN: str = "emp_length_years"

# Mapping used by parse_emp_length(); values not found default to NaN and
# are imputed later using the median strategy defined in NUMERIC_FEATURES.
EMP_LENGTH_MAP: Dict[str, float] = {
    "< 1 year": 0.0,
    "1 year": 1.0,
    "2 years": 2.0,
    "3 years": 3.0,
    "4 years": 4.0,
    "5 years": 5.0,
    "6 years": 6.0,
    "7 years": 7.0,
    "8 years": 8.0,
    "9 years": 9.0,
    "10+ years": 10.0,
}

# Numeric predictor columns fed into the pipeline's numeric branch
# (imputation + scaling). emp_length_years and the parsed percentage
# columns are included here since they are numeric AFTER cleaning.
NUMERIC_FEATURES: List[str] = [
    "loan_amnt",
    "int_rate",
    "installment",
    "annual_inc",
    "dti",
    "delinq_2yrs",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "mort_acc",
    "pub_rec_bankruptcies",
    "emp_length_years",
]

# Low/medium-cardinality categorical columns -> one-hot encoded.
ONEHOT_CATEGORICAL_FEATURES: List[str] = [
    "term",
    "home_ownership",
    "verification_status",
    "purpose",
    "initial_list_status",
    "application_type",
]

# Ordinal categorical columns -> ordinal encoded (natural risk ordering).
ORDINAL_CATEGORICAL_FEATURES: List[str] = [
    "grade",
]
ORDINAL_CATEGORY_ORDER: List[List[str]] = [
    ["A", "B", "C", "D", "E", "F", "G"],
]

# Columns intentionally EXCLUDED from modeling (identifiers, free text,
# leakage-prone or post-origination fields, and the raw pre-cleaning
# versions of engineered columns). Kept here explicitly (rather than
# silently dropped) so the rationale is auditable in one place.
EXCLUDED_COLUMNS: List[str] = [
    "id",
    "member_id",
    "emp_title",
    "url",
    "desc",
    "title",
    "zip_code",
    "addr_state",
    "issue_d",
    "earliest_cr_line",
    "sub_grade",
    "loan_status",  # replaced by TARGET_COLUMN
    "emp_length",  # replaced by EMP_LENGTH_NUMERIC_COLUMN
]

# All columns the raw ingestion step expects to find. Used by
# validate_dataset() to flag schema drift early.
EXPECTED_RAW_COLUMNS: List[str] = sorted(
    set(
        NUMERIC_FEATURES
        + ONEHOT_CATEGORICAL_FEATURES
        + ORDINAL_CATEGORICAL_FEATURES
        + EXCLUDED_COLUMNS
        + [LOAN_STATUS_COLUMN, STATE_COLUMN, EMP_LENGTH_RAW_COLUMN]
    )
    - {EMP_LENGTH_NUMERIC_COLUMN}  # engineered, not present in raw file
)

# ---------------------------------------------------------------------------
# 6. TRAIN / VALIDATION / TEST SPLIT
# ---------------------------------------------------------------------------
# Proportions must sum to 1.0. Test is held out first, then validation is
# carved out of the remaining training pool, so the test set is never used
# to inform any preprocessing or modeling decision (leakage prevention).
TEST_SIZE: float = 0.15
VALIDATION_SIZE: float = 0.15  # fraction of the ORIGINAL full dataset
STRATIFY_COLUMN: str = TARGET_COLUMN

# ---------------------------------------------------------------------------
# 7. LOGGING
# ---------------------------------------------------------------------------
LOG_LEVEL: int = logging.INFO
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
