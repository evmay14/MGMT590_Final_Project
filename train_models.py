"""
train_models.py
================
Orchestration entry point for the MGMT 590 LendingClub Loan Default Risk
capstone project.

PHASE 1 SCOPE (this file, as delivered now)
--------------------------------------------
This phase implements and exercises the full DATA pipeline:
    raw ingestion -> validation -> cleaning -> preprocessing pipeline
    construction -> leakage-safe train/val/test split -> serialization.

Running this module end-to-end (``python -m src.train_models``) will:
    1. Ensure the project directory structure exists.
    2. Load the raw Indiana LendingClub extract.
    3. Validate it and log a data-quality report.
    4. Clean it (dedupe, percentage parsing, employment-length parsing,
       binary target construction, column pruning).
    5. Split it into train/validation/test sets (stratified, leak-safe).
    6. Fit the preprocessing ColumnTransformer on the TRAINING split only.
    7. Serialize the fitted preprocessor and save all split artifacts.
    8. Save the full cleaned (pre-split) dataset for EDA/reporting use.

PHASE 2 SCOPE (NOT implemented here — see NotImplementedError stubs below)
----------------------------------------------------------------------------
Model training (Logistic Regression -> Random Forest -> XGBoost),
cross-validation, hyperparameter tuning, and evaluation-metric reporting
will be implemented in Phase 2 inside the functions already stubbed out
below (``train_logistic_regression``, ``train_random_forest``,
``train_xgboost``, ``evaluate_model``). They intentionally raise
``NotImplementedError`` rather than returning fake/placeholder results,
per project requirements. Their signatures are fixed now so Phase 2 can
implement them without changing how ``main()`` or any other module calls
them.

This module is designed to be run either as a script or imported:

    from src.train_models import run_phase1_pipeline
    artifacts = run_phase1_pipeline()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import pandas as pd
from sklearn.compose import ColumnTransformer

from src import config, utils

logger = utils.get_logger(__name__)


@dataclass
class Phase1Artifacts:
    """
    Container for every artifact produced by the Phase 1 data pipeline,
    returned by ``run_phase1_pipeline()`` so callers (notebook, Phase 2
    script, tests) can access in-memory results without re-reading from
    disk if they don't want to.

    Attributes
    ----------
    raw_df : pd.DataFrame
        Unmodified raw dataset as loaded from disk.
    validation_report : dict
        Output of ``utils.validate_dataset`` run on ``raw_df``.
    cleaned_df : pd.DataFrame
        Output of ``utils.clean_dataset`` (post cleaning, pre-split).
    X_train, X_val, X_test : pd.DataFrame
        Feature splits.
    y_train, y_val, y_test : pd.Series
        Target splits.
    preprocessor : ColumnTransformer
        Preprocessing pipeline FIT on X_train only.
    """

    raw_df: pd.DataFrame
    validation_report: Dict[str, Any]
    cleaned_df: pd.DataFrame
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    preprocessor: ColumnTransformer


def run_phase1_pipeline(persist: bool = True) -> Phase1Artifacts:
    """
    Execute the complete Phase 1 data pipeline end-to-end.

    Parameters
    ----------
    persist : bool
        If True (default), write the cleaned dataset, split artifacts,
        and fitted preprocessor to disk (config.PROCESSED_DATA_DIR,
        config.SPLITS_DIR, config.PIPELINES_DIR). Set False for a
        dry-run / unit-test invocation that keeps everything in memory
        only.

    Returns
    -------
    Phase1Artifacts
        Every intermediate and final artifact from the pipeline.
    """
    logger.info("=" * 70)
    logger.info("PHASE 1 PIPELINE START")
    logger.info("=" * 70)

    utils.ensure_directories()

    # 1. Ingest
    raw_df = utils.load_raw_data()

    # 2. Validate
    validation_report = utils.validate_dataset(raw_df)

    # 3. Clean
    cleaned_df = utils.clean_dataset(raw_df)
    if persist:
        utils.save_dataframe(cleaned_df, config.CLEANED_DATA_PATH)

    # 4. Split (leakage-safe: test/val carved out before any fitting)
    X_train, X_val, X_test, y_train, y_val, y_test = utils.split_data(cleaned_df)
    if persist:
        utils.save_splits(X_train, X_val, X_test, y_train, y_val, y_test)

    # 5. Build + fit preprocessing pipeline on TRAINING data only
    preprocessor = utils.build_preprocessing_pipeline()
    preprocessor.fit(X_train)
    logger.info(
        "Preprocessor fit complete. Output feature count: %d",
        len(utils.get_output_feature_names(preprocessor)),
    )
    if persist:
        utils.save_object(preprocessor, config.PREPROCESSOR_PATH)

    logger.info("=" * 70)
    logger.info("PHASE 1 PIPELINE COMPLETE")
    logger.info("=" * 70)

    return Phase1Artifacts(
        raw_df=raw_df,
        validation_report=validation_report,
        cleaned_df=cleaned_df,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        preprocessor=preprocessor,
    )


# ---------------------------------------------------------------------------
# PHASE 2 STUBS — signatures fixed now, implementation deferred.
# ---------------------------------------------------------------------------
# These are intentionally NOT implemented with placeholder/fake models.
# Calling them in Phase 1 will raise NotImplementedError so it is
# impossible to mistake a stub for a real, evaluated model. Phase 2 will
# fill in the bodies without needing to change any calling code, since the
# preprocessor + splits they depend on are already produced and persisted
# by run_phase1_pipeline() above.
# ---------------------------------------------------------------------------


def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series) -> Any:
    """
    (Phase 2) Fit a Logistic Regression baseline classifier on the
    preprocessed training features.

    Not implemented in Phase 1 — raises NotImplementedError by design so
    no placeholder/fake model results are produced.
    """
    raise NotImplementedError(
        "train_logistic_regression will be implemented in Phase 2."
    )


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> Any:
    """
    (Phase 2) Fit a Random Forest classifier, including hyperparameter
    tuning, on the preprocessed training features.

    Not implemented in Phase 1 — raises NotImplementedError by design.
    """
    raise NotImplementedError("train_random_forest will be implemented in Phase 2.")


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series) -> Any:
    """
    (Phase 2) Fit an XGBoost classifier, including hyperparameter tuning,
    on the preprocessed training features.

    Not implemented in Phase 1 — raises NotImplementedError by design.
    """
    raise NotImplementedError("train_xgboost will be implemented in Phase 2.")


def evaluate_model(model: Any, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    """
    (Phase 2) Compute the standard evaluation-metric suite (accuracy,
    precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix) for a
    fitted model on a given feature/target set.

    Not implemented in Phase 1 — raises NotImplementedError by design.
    """
    raise NotImplementedError("evaluate_model will be implemented in Phase 2.")


def main() -> None:
    """
    Script entry point: run the Phase 1 pipeline and print a concise
    summary. Intended usage: ``python -m src.train_models``.
    """
    artifacts = run_phase1_pipeline(persist=True)

    print("\n--- PHASE 1 SUMMARY ---")
    print(f"Raw rows loaded:        {len(artifacts.raw_df):,}")
    print(f"Rows after cleaning:    {len(artifacts.cleaned_df):,}")
    print(f"Train / Val / Test:     {len(artifacts.X_train):,} / "
          f"{len(artifacts.X_val):,} / {len(artifacts.X_test):,}")
    print(f"Default rate (train):   {artifacts.y_train.mean():.3%}")
    print(f"Preprocessed feature count: "
          f"{len(utils.get_output_feature_names(artifacts.preprocessor)):,}")
    print(f"Cleaned dataset saved to:   {config.CLEANED_DATA_PATH}")
    print(f"Preprocessor saved to:      {config.PREPROCESSOR_PATH}")
    print("Phase 2 (model training) is intentionally not yet implemented.")


if __name__ == "__main__":
    main()
