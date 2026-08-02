# MGMT 590 — LendingClub Loan Default Risk Capstone (Indiana Borrowers)

Graduate Business Analytics capstone (MGMT 59000, Summer 2026, Section
DY2, Purdue University). Predicts loan default risk for Indiana
LendingClub borrowers and supports lending decisions through a Streamlit
decision-support application.

This repository is built incrementally across seven phases (the roadmap
was revised from six to seven phases starting Phase 4B, splitting what
was "Phase 4" into 4A/4B and adding a dedicated Phase 7 for final review/
documentation). **Phases 1-3 (data pipeline, exploratory analysis, and
supervised modeling), Phase 4A (explainability + risk scoring), Phase 4B
(borrower segmentation), Phase 5 (the Streamlit dashboard), and Phase 6
(integration/edge-case testing, performance optimization, and deployment
preparation) are complete.**

## Phase 1 deliverables

- GitHub-ready project structure
- `requirements.txt`
- `src/config.py` — all paths, constants, and column definitions
- `src/utils.py` — reusable ingestion, validation, cleaning, preprocessing,
  splitting, and serialization functions
- `src/train_models.py` — Phase 1 orchestration (`run_phase1_pipeline`);
  Phase 3 model training (`run_phase3_pipeline`) implemented as of Phase 3
- `notebooks/MGMT590_LendingClub_Analysis.ipynb` — executed walkthrough of
  the full Phase 1 pipeline
- Leakage-safe train/validation/test split
- Serialized (`joblib`) `ColumnTransformer` preprocessing pipeline

## Phase 2 deliverables

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
  assessment, and a Phase 3 hand-off summary. No modeling is performed
  in this notebook.
- `tests/test_eda_utils.py` — unit tests for every statistical/table
  function in `eda_utils.py` (plotting functions are smoke-tested)

## Phase 3 deliverables (this commit)

**Files Modified:**
- `src/config.py` — appended a Phase 3 section (new `REPORTS_DIR`,
  model/report artifact paths, CV settings, hyperparameter search
  spaces, threshold/cost constants); Phases 1-2 sections untouched.
- `src/train_models.py` — the four Phase 3 stubs
  (`train_logistic_regression`, `train_random_forest`, `train_xgboost`,
  `evaluate_model`) are now implemented (they delegate to
  `model_utils.py`); added `Phase3Artifacts` + `run_phase3_pipeline()`;
  `main()` now runs Phase 1 then Phase 3. `run_phase1_pipeline()` and
  everything else from Phase 1 is unchanged.
- `requirements.txt` — added `scipy`/`statsmodels` (already required by
  Phase 2's `eda_utils.py`; formalized here).

**Files Created:**
- `src/model_utils.py` — reusable Phase 3 ML framework: pipeline
  builders (preprocessing + classifier, leakage-safe), GridSearchCV /
  RandomizedSearchCV wrappers, Stratified K-Fold CV reporting, the full
  classification-metric suite (incl. calibration error), threshold
  optimization (cost-based), feature importance (coefficients/odds
  ratios, impurity, permutation, XGBoost gain/weight/cover), plotting
  functions (confusion matrix, ROC, PR, calibration, learning curve,
  validation curve, importance/coefficient plots, probability
  distribution, threshold analysis), and the model-comparison table
  builder.
- `notebooks/MGMT590_LendingClub_Modeling_Phase3.ipynb` — executed,
  executive-quality notebook: algorithm rationale for all three models,
  cross-validated hyperparameter search, full metric suite, diagnostic
  visualizations, feature importance (3 methods per applicable model),
  threshold optimization, executive model-comparison table with a
  production-model recommendation, robustness assessment, cross-model
  business interpretation, and a Phase 4 hand-off summary.
- `tests/test_model_utils.py` — 22 unit tests covering metrics,
  calibration error, threshold optimization, feature importance, and
  hyperparameter search.
- `tests/build_notebook_phase3.py` — regenerates the Phase 3 notebook.

**Files Unchanged:** `src/utils.py`, `src/eda_utils.py`,
`notebooks/MGMT590_LendingClub_Analysis.ipynb`,
`notebooks/MGMT590_LendingClub_EDA_Phase2.ipynb`,
`tests/test_utils.py`, `tests/test_eda_utils.py`,
`tests/generate_synthetic_fixture.py`, `.gitignore`.

**Not implemented (explicitly deferred to Phase 4):** SHAP explanations,
borrower clustering. `src/train_models.py`'s Phase 3 functions and
`model_utils.py` are stable, importable interfaces for Phase 4 to build
on without modification.

## Phase 4A deliverables (this commit)

**Files Modified:**
- `src/config.py` — appended a Phase 4A section (new `EXPLAINABILITY_DIR`,
  `PRODUCTION_MODEL_KEY`, SHAP sample-size settings, explainability
  artifact paths); Phases 1-3 sections untouched.
- `requirements.txt` — added `shap`.

**Files Created:**
- `src/configurable_thresholds.py` — `RiskThresholdConfig` dataclass
  (risk tiers, lending actions, interest-rate adjustments, loan-grade
  bands) with a JSON load/save API so a lending-operations stakeholder
  can edit `reports/risk_threshold_config.json` directly — no code
  change or redeploy required. Self-bootstraps built-in defaults to
  disk on first run; `validate()` sanity-checks tier coverage/overlap.
- `src/interpretation_utils.py` — feature-name humanization (technical
  → business-friendly labels), research-question linkage, executive
  business-summary text generation (borrower-level and model-level),
  fairness reporting (reuses `model_utils.compute_classification_metrics`
  per subgroup), and `ExportableReport` (Markdown/JSON, Streamlit-
  download-button-ready).
- `src/risk_scoring.py` — `RiskScoringEngine`: probability → 0-100 risk
  score → risk tier → confidence score → recommended action / interest
  rate / loan grade, entirely driven by `RiskThresholdConfig` (no
  hard-coded thresholds). Also adds `expand_threshold_analysis` /
  `plot_expanded_threshold_analysis`, extending Phase 3's threshold
  table with approval rate and false-positive/false-negative rate.
- `src/explainability.py` — `ExplainabilityEngine`: SHAP-based global
  and local explanations (`TreeExplainer` for Random Forest/XGBoost,
  `LinearExplainer` for Logistic Regression), every method from the
  Phase 4A brief (`explain_prediction`, `explain_global_model`,
  `generate_shap_summary` [beeswarm + bar], `generate_waterfall_plot`,
  `generate_force_plot`, `generate_dependence_plot`,
  `generate_decision_plot`, `summarize_feature_importance`,
  `generate_business_summary`), plus feature-interaction analysis (the
  5 required pairs), Partial Dependence/ICE plots, exportable reports,
  and `persist_explainability_artifacts()`.
- `notebooks/MGMT590_LendingClub_Explainability_Phase4A.ipynb` —
  executed, executive-quality notebook demonstrating all four modules
  end-to-end against the Phase 3 production model.
- `tests/test_configurable_thresholds.py`, `tests/test_interpretation_utils.py`,
  `tests/test_risk_scoring.py`, `tests/test_explainability.py` — 83 new
  unit tests.
- `tests/build_notebook_phase4a.py` — regenerates the Phase 4A notebook.

**Files Unchanged:** `src/utils.py`, `src/eda_utils.py`,
`src/model_utils.py`, `src/train_models.py`, all Phase 1-3 notebooks and
tests, `.gitignore`.

**Not implemented (explicitly deferred):** the Streamlit dashboard
(Phase 5) and borrower clustering (Phase 4B). `ExplainabilityEngine` and
`RiskScoringEngine` are stable, Streamlit-ready interfaces — every
plotting method returns a `matplotlib.figure.Figure` and every summary
is a plain dataclass/DataFrame/string, ready for direct use in a future
`st.pyplot(...)` / `st.dataframe(...)` / `st.download_button(...)` call
without modification.

## Phase 4B deliverables (this commit)

**Files Modified:**
- `src/config.py` — appended a Phase 4B section only (new
  `SEGMENTATION_DIR`, `CLUSTERING_MODEL_PATH`, clustering feature lists,
  outlier/optimal-k/algorithm defaults, segmentation artifact paths).
  Phases 1-4A sections untouched.
- `requirements.txt` — added `umap-learn`.

**Files Created:**
- `src/cluster_analysis.py` — data preparation (IQR outlier clipping,
  a clustering-specific preprocessor scoped to numeric + ordinal
  features only), dimensionality reduction (PCA/t-SNE/UMAP), four
  clustering algorithms (K-Means, Agglomerative, Gaussian Mixture,
  DBSCAN — each with a full advantages/disadvantages/business-
  applicability/computational-considerations docstring), and optimal-k
  evaluation (elbow, silhouette, Calinski-Harabasz, Davies-Bouldin,
  combined into a transparent rank-based recommendation).
- `src/cluster_visualization.py` — dimensionality-reduction scatter
  plots (single + side-by-side method comparison), the four-panel
  optimal-k chart, cluster heatmap, parallel-coordinates plot, radar
  chart, cluster size distribution, and generic feature-by-cluster
  bar/boxplot builders (used for the required income/interest-rate/
  DTI/default-rate-by-cluster charts).
- `src/segment_profiles.py` — comprehensive per-cluster profiling,
  DATA-DRIVEN business naming (`assign_segment_names` — priority logic
  based on relative income/DTI/rate/default-rate z-scores, not fixed
  assumptions), business-action recommendations keyed on measured risk
  tier, segment comparison tables, and exportable segment reports.
- `src/segmentation_engine.py` — `SegmentationEngine`: every method
  from the Phase 4B brief (`fit`, `predict_cluster`, `assign_segment`,
  `describe_segment`, `generate_cluster_profile`, `visualize_clusters`,
  `compare_segments`, `recommend_business_actions`,
  `export_segment_summary`), plus `compare_with_supervised_models()`
  cross-referencing cluster membership against the Phase 3/4A
  production model's predicted probabilities, and
  `persist_segmentation_artifacts()`.
- `notebooks/MGMT590_LendingClub_Segmentation_Phase4B.ipynb` — 55
  executed cells, 14 real visualizations: data preparation rationale,
  dimensionality-reduction comparison, optimal-k analysis, algorithm
  comparison, full engine fit, all required visualizations, cluster
  profiles, business naming, business recommendations, segment
  comparison, relationship-to-ML cross-check, research-question
  support, exportable reports, and a Phase 5 hand-off.
- `tests/test_cluster_analysis.py`, `tests/test_cluster_visualization.py`,
  `tests/test_segment_profiles.py`, `tests/test_segmentation_engine.py`
  — 65 new unit tests (204 total across the project, all passing).
- `tests/build_notebook_phase4b.py` — regenerates the Phase 4B notebook.

**Files Unchanged:** `src/utils.py`, `src/eda_utils.py`,
`src/model_utils.py`, `src/train_models.py`, `src/explainability.py`,
`src/risk_scoring.py`, `src/interpretation_utils.py`,
`src/configurable_thresholds.py`, all Phase 1-4A notebooks and tests,
`.gitignore`.

**Not implemented (explicitly deferred):** the Streamlit dashboard
(Phase 5), integration testing/performance optimization/deployment
(Phase 6), and final documentation/presentation assets (Phase 7).
`SegmentationEngine` is a stable, Streamlit-ready interface — every
plotting method returns a `matplotlib.figure.Figure`, every summary is a
plain dataclass/DataFrame/string, and `export_segment_summary()` returns
an `interpretation_utils.ExportableReport` ready for
`st.download_button(...)`, without modification.

## Phase 5 deliverables (this commit)

**Files Modified:**
- `requirements.txt` — updated the `streamlit` dependency's comment to
  reflect it now backs an actual deployed dashboard rather than a
  placeholder; no version-range changes.

**Files Created:**
- `app/app.py` — entry point. Wires up `st.navigation`/`st.Page`
  multipage routing (grouped into Overview / Analysis / Decision Tools /
  Project Info) and renders the shared sidebar. Contains no ML logic.
- `app/common.py` — every shared UI utility: cached loaders for the
  cleaned dataset, Phase 1 splits, Phase 3 reports, and the
  `RiskScoringEngine`/`ExplainabilityEngine`/`SegmentationEngine`
  instances (`st.cache_resource`/`st.cache_data`, so models are loaded
  and the segmentation model is fit exactly once per session); sidebar
  controls (model selector, borrower filters, display-density toggle,
  download options, about blurb); KPI/page-header/executive-summary-box
  styling; CSV/Markdown/JSON/PNG download helpers. This module is the
  ONLY place caching and engine-loading logic lives — every page imports
  from it rather than re-implementing any of it.
- `app/app_pages/` — all eight required pages, each purely an
  orchestration layer over the Phase 3/4A/4B engines:
  - `executive_dashboard.py` — portfolio KPIs, grade distribution,
    default rate by grade, top risk factors (from `ExplainabilityEngine`),
    executive takeaways.
  - `exploratory_analysis.py` — interactive EDA re-rendering
    `eda_utils.py`'s plotting functions against the sidebar-filtered
    dataset.
  - `model_comparison.py` — Phase 3's comparison table, ROC/PR curves,
    confusion matrices, calibration curves, and on-demand learning
    curves for all three models side-by-side.
  - `borrower_risk_prediction.py` — the interactive prediction form;
    calls `RiskScoringEngine.generate_prediction_summary()` and
    `ExplainabilityEngine.explain_prediction()`/`generate_waterfall_plot()`/
    `generate_force_plot()` and displays their output (probability gauge,
    risk meter, top risk/protective factors, executive summary,
    exportable reports) — no scoring or SHAP logic implemented here.
  - `borrower_segmentation.py` — segment overview, per-segment detail
    and recommendations, PCA/t-SNE/radar/heatmap visualizations, and the
    supervised-model cross-check, entirely via `SegmentationEngine`.
  - `business_insights.py` — all seven research questions in
    question/finding/evidence/visualization/recommendation/decision-impact
    format.
  - `model_explainability.py` — global SHAP summary/dependence/decision/
    waterfall plots and business interpretation via `ExplainabilityEngine`.
  - `about_project.py` — business problem, research questions, PDID
    framework, methodology, technology stack, workflow, limitations,
    future improvements.
- `.streamlit/config.toml` — professional navy/slate theme.
- `tests/test_app.py` — 18 tests using Streamlit's `AppTest` headless
  testing API: every page loads without exception, sidebar controls
  work, the full prediction form submits end-to-end, Business Insights
  covers all seven research questions.

**Files Unchanged:** every `src/*.py` module from Phases 1-4B, all
notebooks and their tests, `.gitignore`.

**Not implemented (explicitly deferred):** integration testing,
performance optimization, and deployment configuration (Phase 6), and
final documentation/presentation assets (Phase 7).

## Phase 6 deliverables (this commit)

**Files Modified:**
- `src/risk_scoring.py` — fixed a genuine edge-case bug found during
  Phase 6 testing (`predict_probability` crashed with a low-level
  sklearn error on a zero-row DataFrame instead of returning an empty
  array); promoted a local `matplotlib.pyplot` import to module level,
  fixing a `pyflakes`-flagged undefined-name-in-annotation issue.
- `src/config.py` — added `MGMT590_PROJECT_ROOT` and `MGMT590_LOG_LEVEL`
  environment-variable overrides (Phase 6 configuration-management
  requirement); no existing paths or defaults changed.
- `src/model_utils.py`, `src/eda_utils.py`, `src/explainability.py`,
  `src/segmentation_engine.py`, `src/interpretation_utils.py`,
  `src/train_models.py`, `src/segment_profiles.py` — removed unused
  imports/dead code found by a full-codebase `pyflakes` audit; no
  behavior changes.
- `app/common.py` — added three `st.cache_data`-wrapped functions
  (`get_cluster_visualization`, `get_learning_curve_figure`,
  `get_global_explanation`) for expensive computations that were
  previously recomputed on every page rerun.
- `app/app_pages/borrower_risk_prediction.py`,
  `app/app_pages/borrower_segmentation.py` — wrapped engine calls in
  try/except with logged tracebacks and friendly `st.error` messages;
  wired in the new cached wrappers.
- `app/app_pages/executive_dashboard.py`, `app/app_pages/model_comparison.py`,
  `app/app_pages/model_explainability.py` — wired in the new cached
  wrappers; removed unused imports.
- `app/app.py` — added session-scoped startup logging.
- `requirements.txt` — removed three confirmed-unused dependencies
  (`imbalanced-learn`, `plotly`, `python-dateutil`), found by grepping
  the entire codebase for their imports and finding none.

**Files Created:**
- `tests/test_integration.py` — 18 tests verifying every cross-component
  seam (preprocessing pipeline, all serialized models, the clustering
  model, all three engines, configuration files, logging) works
  end-to-end against real artifacts on disk, not mocks.
- `tests/test_edge_cases.py` — 25 tests covering missing values, empty
  datasets, invalid loan grades, negative income, extreme loan
  amounts/DTI, unexpected categories, corrupted/missing serialized
  files, and malformed user input. This suite caught the
  `predict_probability` empty-DataFrame bug fixed above.
- `requirements-app.txt` — a lean, deployment-only dependency set
  (excludes Jupyter/notebook/pytest) for a smaller/faster deployment
  install.
- `runtime.txt` — pins the Python version for deployment platforms that
  read it (Render, Heroku-style buildpacks).
- `PERFORMANCE_REPORT.md` — measured startup time, prediction/SHAP/
  clustering latency, memory usage, ranked bottlenecks, optimizations
  made, and recommendations for future scaling.
- `QA_CHECKLIST.md` — a comprehensive, test-file-referenced QA checklist
  covering data loading through documentation references.
- `DEPLOYMENT.md` — environment setup, launch instructions, a folder/
  dependency verification checklist, and platform-specific guidance for
  Streamlit Community Cloud, Render, Railway, and Hugging Face Spaces
  (no actual deployment performed).

**Files Unchanged:** every other `src/*.py` module's public interface
and behavior, all notebooks and their tests, every other `app/app_pages/*.py`
page's structure, `.streamlit/config.toml`, `.gitignore`.

**Not implemented (explicitly deferred):** final documentation and
presentation assets (Phase 7). No new analytics or ML models were added
in this phase, per the Phase 6 brief.

## Project structure

```
mgmt590_capstone/
├── README.md
├── requirements.txt
├── requirements-app.txt      # Phase 6: lean deployment-only dependency set
├── runtime.txt                # Phase 6: pinned Python version for deployment platforms
├── PERFORMANCE_REPORT.md      # Phase 6: measured startup/latency/memory + bottlenecks
├── QA_CHECKLIST.md            # Phase 6: comprehensive, test-referenced QA checklist
├── DEPLOYMENT.md              # Phase 6: environment setup + platform deployment guidance
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── config.py                  # paths, constants, column groups, CV/search/explainability/segmentation settings
│   ├── utils.py                    # ingestion / validation / cleaning / pipeline / split / serialization
│   ├── eda_utils.py                # Phase 2: descriptive stats, plotting, statistical tests
│   ├── model_utils.py              # Phase 3: ML pipelines, hyperparameter search, metrics, importance
│   ├── train_models.py            # Phase 1 + Phase 3 orchestration (run_phase1_pipeline, run_phase3_pipeline)
│   ├── configurable_thresholds.py # Phase 4A: risk-tier/action/rate/grade business-policy config
│   ├── interpretation_utils.py    # Phase 4A: feature humanization, business text, fairness, exports
│   ├── risk_scoring.py            # Phase 4A: RiskScoringEngine
│   ├── explainability.py          # Phase 4A: ExplainabilityEngine (SHAP)
│   ├── cluster_analysis.py        # Phase 4B: data prep, dimensionality reduction, clustering algorithms, optimal-k
│   ├── cluster_visualization.py   # Phase 4B: cluster scatter/heatmap/parallel-coords/radar/bar plotting
│   ├── segment_profiles.py        # Phase 4B: profiling, data-driven naming, recommendations, exports
│   └── segmentation_engine.py     # Phase 4B: SegmentationEngine
├── notebooks/
│   ├── MGMT590_LendingClub_Analysis.ipynb                # Phase 1
│   ├── MGMT590_LendingClub_EDA_Phase2.ipynb              # Phase 2
│   ├── MGMT590_LendingClub_Modeling_Phase3.ipynb         # Phase 3
│   ├── MGMT590_LendingClub_Explainability_Phase4A.ipynb  # Phase 4A
│   └── MGMT590_LendingClub_Segmentation_Phase4B.ipynb    # Phase 4B
├── data/
│   ├── raw/                 # place the real LendingClub Indiana extract here
│   ├── processed/           # cleaned dataset (generated)
│   └── splits/              # X/y train/val/test CSVs (generated)
├── models/                  # serialized trained models (logistic_regression/random_forest/xgboost/clustering_model .joblib)
├── pipelines/               # serialized fitted preprocessing pipeline (Phase 1/2 use)
├── reports/                 # Phase 3 evaluation artifacts + Phase 4A risk_threshold_config.json
│   ├── explainability/      # Phase 4A: SHAP importance, business summaries, metadata, fairness report
│   └── segmentation/        # Phase 4B: cluster centroids, segment definitions, metadata, profiles, optimal-k table
├── logs/                    # pipeline run logs
├── .streamlit/
│   └── config.toml          # Phase 5: dashboard theme (navy/slate palette)
├── app/                     # Phase 5: Streamlit dashboard
│   ├── app.py                # entry point — navigation + shared sidebar (no ML logic)
│   ├── common.py             # cached engine/data loaders, sidebar controls, styling, download helpers
│   └── app_pages/
│       ├── executive_dashboard.py       # Page 1: portfolio KPIs
│       ├── exploratory_analysis.py      # Page 2: interactive EDA
│       ├── model_comparison.py          # Page 3: ROC/PR/confusion/calibration/learning curves
│       ├── borrower_risk_prediction.py  # Page 4: live single-borrower scoring + SHAP
│       ├── borrower_segmentation.py     # Page 5: segment lookup + visualizations
│       ├── business_insights.py         # Page 6: research questions as an executive report
│       ├── model_explainability.py      # Page 7: global SHAP explanations
│       └── about_project.py             # Page 8: methodology & documentation
└── tests/
    ├── generate_synthetic_fixture.py     # synthetic data for local pipeline testing only
    ├── build_notebook.py                  # regenerates the Phase 1 notebook
    ├── build_notebook_phase2.py           # regenerates the Phase 2 notebook
    ├── build_notebook_phase3.py           # regenerates the Phase 3 notebook
    ├── build_notebook_phase4a.py          # regenerates the Phase 4A notebook
    ├── build_notebook_phase4b.py          # regenerates the Phase 4B notebook
    ├── test_utils.py
    ├── test_eda_utils.py
    ├── test_model_utils.py
    ├── test_configurable_thresholds.py
    ├── test_interpretation_utils.py
    ├── test_risk_scoring.py
    ├── test_explainability.py
    ├── test_cluster_analysis.py
    ├── test_cluster_visualization.py
    ├── test_segment_profiles.py
    ├── test_segmentation_engine.py
    ├── test_app.py             # Phase 5: Streamlit AppTest-based headless UI tests
    ├── test_integration.py     # Phase 6: cross-component integration tests
    └── test_edge_cases.py      # Phase 6: missing/invalid/corrupted-input edge cases
```

## Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

For a deployment-only environment (no Jupyter/pytest), use
`requirements-app.txt` instead — see `DEPLOYMENT.md` for full guidance.

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

## Running the pipeline (Phase 1 + Phase 3)

```bash
python -m src.train_models
```

This runs Phase 1 (ingest, validate, clean, split, fit/serialize the
preprocessing pipeline) followed by Phase 3 (train + tune Logistic
Regression, Random Forest, and XGBoost; serialize models and evaluation
artifacts), logging every step to `logs/pipeline.log` and printing a
summary of both phases. Equivalently, open and run
`notebooks/MGMT590_LendingClub_Analysis.ipynb` (Phase 1),
`notebooks/MGMT590_LendingClub_EDA_Phase2.ipynb` (Phase 2 EDA — read-only
analysis, no artifacts required beyond Phase 1's), and
`notebooks/MGMT590_LendingClub_Modeling_Phase3.ipynb` (Phase 3 modeling).

Programmatic usage:

```python
from src.train_models import run_phase1_pipeline, run_phase3_pipeline

phase1 = run_phase1_pipeline()
phase1.X_train, phase1.y_train   # leakage-safe training split
phase1.preprocessor               # fitted ColumnTransformer (Phase 1/2 use)

phase3 = run_phase3_pipeline(phase1)
phase3.results["xgboost"].best_estimator   # tuned Pipeline (preprocessor + classifier)
phase3.comparison_table                     # executive model-comparison table
```

## Running the Dashboard (Phase 5)

The dashboard reads already-serialized artifacts — run the pipeline
above (plus persist the Phase 4A/4B artifacts, shown below) at least
once first:

```bash
python -m src.train_models

python -c "
from src.explainability import ExplainabilityEngine
ExplainabilityEngine().persist_explainability_artifacts()
"

python -c "
from src import utils
from src.segmentation_engine import SegmentationEngine
X_train, _, _, y_train, _, _ = utils.load_splits()
engine = SegmentationEngine()
engine.fit(X_train, default_flags=y_train)
engine.persist_segmentation_artifacts()
"
```

Then, from the project root:

```bash
streamlit run app/app.py
```

The first load fits the segmentation model and computes a SHAP
background sample (a few seconds); every model and preprocessing
pipeline is loaded from disk, never retrained. Pages that need an
artifact that hasn't been generated yet show a friendly notice with the
exact command to run, instead of crashing.

Test the dashboard headlessly (no browser required) via Streamlit's
`AppTest` API, alongside the Phase 6 integration and edge-case suites:

```bash
pytest tests/test_app.py tests/test_integration.py tests/test_edge_cases.py -v

# Or the full project suite (265 tests as of Phase 6):
pytest tests/ -q
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
| 2 | Exploratory data analysis, descriptive statistics, default-rate analysis, research-question analysis, statistical testing, feature-relationship assessment |
| 3 | Supervised model training: Logistic Regression → Random Forest → XGBoost, hyperparameter tuning, evaluation, threshold optimization, feature importance |
| 4A | Explainable AI layer: `ExplainabilityEngine` (SHAP), `RiskScoringEngine`, configurable business thresholds, fairness assessment |
| 4B | Borrower segmentation: `SegmentationEngine`, clustering algorithm/optimal-k comparison, data-driven segment naming, business recommendations |
| 5 | Streamlit dashboard: 8-page multipage app orchestrating all four engines, professional theme, caching, exportable reports |
| 6 (this commit) | Integration testing (18 tests), edge-case testing (25 tests), performance optimization (caching), code cleanup, logging, deployment prep |
| 7 | Final code review, documentation & presentation assets |

## Notes for future phases

- Import shared logic from `src.config`, `src.utils`, `src.model_utils`,
  and (for anything explainability/risk-scoring/segmentation related)
  `src.explainability`, `src.risk_scoring`, `src.interpretation_utils`,
  `src.configurable_thresholds`, `src.segmentation_engine`,
  `src.cluster_analysis`, `src.cluster_visualization`,
  `src.segment_profiles` rather than re-implementing it.
- The preprocessing pipeline in `pipelines/preprocessing_pipeline.joblib`
  is fit on `X_train` only — always `.transform()` (never re-`.fit()`) it
  on validation/test/live data. Phase 3's tuned model pipelines
  (`models/*.joblib`) each already bundle their own preprocessor
  internally (see the design-decision note in
  `model_utils.build_model_pipeline`) — no separate transform step is
  needed to use them. `SegmentationEngine` similarly owns its own
  clustering-specific preprocessor internally (`config.CLUSTERING_PREPROCESSOR_PATH`)
  — a deliberately NARROWER feature space (numeric + ordinal `grade`
  only) than the supervised models use; see `config.py`'s Phase 4B
  section for why.
- Phase 5 (Streamlit) is now built exactly this way — see `app/common.py`
  for the actual `st.cache_resource`-wrapped constructors. Every plot
  method returns a `matplotlib.figure.Figure` for `st.pyplot(...)`; every
  summary is a plain dataclass/DataFrame/string; every exportable report
  is an `interpretation_utils.ExportableReport` whose `.to_markdown()`/
  `.to_json()` output is wired to `st.download_button(...)`. Business
  thresholds are editable via `reports/risk_threshold_config.json`
  without touching any engine's or page's code.
- Phase 6 (integration testing / performance / deployment) is now
  complete: `tests/test_integration.py` and `tests/test_edge_cases.py`
  join `tests/test_app.py` for headless, `AppTest`/pytest-based
  verification — extend these rather than writing a parallel test
  harness. For deployment, the app expects `streamlit run app/app.py`
  from the project root (so relative paths in `.streamlit/config.toml`
  resolve); `app/common.py`'s `st.cache_resource`/`st.cache_data` calls
  mean cold-start cost (model loading, SHAP background sampling,
  segmentation fit, t-SNE/learning-curve computation) is paid once per
  server process or per distinct cache key, not on every rerun. See
  `PERFORMANCE_REPORT.md` for measured numbers, `QA_CHECKLIST.md` for
  what's been verified and how, and `DEPLOYMENT.md` for environment
  setup and platform-specific launch guidance.
- Phase 7 (final documentation/presentation): `app/app_pages/about_project.py`
  already contains the methodology/PDID/limitations narrative in a
  form suitable for reuse in a written report or slide deck;
  `PERFORMANCE_REPORT.md` and `QA_CHECKLIST.md` are similarly reusable
  as appendices for a final report.
