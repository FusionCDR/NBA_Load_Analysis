# NBA Schedule Impact Analysis

Data science project that attempts to show how schedule load (rest, back-to-backs, travel) affects NBA win probabilities. It involves fitting a logistic GLM to historical games, computes counterfactual “neutral” win chances, and exports per-game schedule impact. Includes a forward-prediction script for testing.

## Features
- End-to-end pipeline: fetch + cache game logs from `nba_api`, feature engineering, model fitting, reporting, and plots.
- Travel & load features: rest days, back-to-backs, stretch density, travel distance/timezone shifts, rolling travel accumulation, composite scores, and z-scored differentials vs opponent.
- Team strength: rolling 10-game net rating with a 1-game lag to avoid leakage, then team vs opponent strength differential.
- Models: GLM variants (basic, interaction, threshold, b2b_travel, etc.) defined in `src/model.py`; Optuna tuning for feature weights. Basic model comparison using AIC and Brier determined b2b_travel to be marginally more accurate than other variants.
- Forward predictions: run on an in-progress season after N games, using cached params/fit to score the remaining schedule with interactive plots.

## Getting Started
- Requirements: Python 3.7+ recommended; install deps with `pip install -r requirements.txt`.
- Optional: create a venv  
  ```bash
  python3 -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  ```
- Data source: free `nba_api` (no key needed). Game logs are cached under `data/raw/`.

## Structure
- `scripts/run_pipeline.py` – end-to-end CLI (data load, model comparison/optimization, analysis, plots).
- `scripts/run_predictions.py` – forward predictions for a partial season with interactive scatter plot.
- `src/data_loader.py` – fetch/process game logs, compute per-game net rating.
- `src/features.py` – feature engineering (strength, schedule density, travel, composites, z-scores).
- `src/model.py` – GLM variants, fitting, counterfactuals, evaluation.
- `src/optimization.py` – Optuna tuner and optimal param persistence.
- `src/analysis.py` – analysis helpers, summary export, plotting.

## Workflows
- Full pipeline (defaults in `config.yaml`):  
  ```bash
  python3 scripts/run_pipeline.py --mode full
  ```
- Compare model variants only:  
  ```bash
  python3 scripts/run_pipeline.py --mode compare --seasons 2019 2020 2021 2022 2023
  ```
- Run Optuna quickly:  
  ```bash
  python3 scripts/run_pipeline.py --mode optimize --quick --n-trials 50
  ```
- Forward predictions on an in-progress season (uses saved optimal params):  
  ```bash
  python3 scripts/run_predictions.py \
    --train-seasons 2019 2020 2021 2022 2023 \
    --predict-season 2024 \
    --completed-games 200 \
    --predict-games 10 \
    --model-variant b2b_travel
  ```
  Hover over points in the resulting scatter plot to see team/opponent, schedule impact, and win probability.

## Findings (latest cached results)
- Best model: `b2b_travel` edged out other GLM variants in log loss (0.651 vs ~0.6512) with similar AUC (~0.663) and Brier (~0.230). Travel main effect was not significant in the full fit.
- Top schedule winners (2019–2024 sample): NOP, DEN, DET, PHI, MIA gained roughly +0.54 to +0.81 wins from schedule context.
- Biggest schedule losers: LAC, ATL, OKC, BOS, GSW lost roughly -0.44 to -1.22 wins due to schedule context.
- Key coefficients (b2b_travel fit): home advantage positive and significant; team strength differential strongly positive; load differential negative and significant; travel differential not significant. Most notably, travel, as defined by distance and timezone changes in the optimized span of time, was not found to have a statistically significant impact on team performance at any point in the pipeline.
- Optimal feature weights from Optuna (quick study): `w_rest`≈1.85, `w_stretch`≈2.00, `w_density`≈2.99, `w_distance`≈1.06, `w_recent_travel`≈1.44; model variant `b2b_travel`.
- Calibrated risky threshold: schedule impact below about -0.075 flags games as risky in exports.


## Key Outputs
- `data/results/predictions*.csv` – per-game probabilities, schedule impacts, load/travel scores.
- Plots: feature importance, team schedule impact, interactive scatter/bar chart for forward predictions.
- `data/results/optimal_params.json` – tuned feature weights and chosen model variant.

## Data Flow & Caching
- Raw data
  - `scripts/run_pipeline.py` and `scripts/run_predictions.py` call `DataLoader` (`src/data_loader.py`).
  - For each season, `nba_api` `LeagueGameLog` is queried once and cached as `data/raw/games_<season>.parquet`.
  - Subsequent runs reuse the parquet unless `force_refresh` is requested.
- Model-ready games
  - `prepare_model_data` takes the raw logs, joins opponent stats, flags home/away, computes win flag, and estimates possessions.
  - It then computes `NET_RATING_GAME` (using standard formula), sorts chronologically, and adds team abbreviations.
- Feature engineering
  - `FeatureEngineering.compute_features` (`src/features.py`) takes the processed games plus team location data and builds:
    - `STRENGTH_PRE`: lagged rolling 10-game net rating per team/season.
    - Schedule features: rest days, back-to-backs, stretch density, rest penalties.
    - Travel features: haversine distances, timezone changes, rolling travel load.
    - Composite scores (`LOAD_SCORE`, `TRAVEL_SCORE`) and differentials vs opponent.
  - Results are cached as `data/cache/features_<hash>.parquet`, where the hash is a function of:
    - Number of rows, unique `GAME_ID`s, seasons present.
    - All feature parameters (e.g., window size, lookback windows, weights).
  - If an identical call is made later, features are loaded from cache instead of recomputed.
- Model fitting and params
  - `src/model.py` defines GLM variants and handles fitting, evaluation, and counterfactuals (`P_FACTUAL`, `P_NEUTRAL`, `SCHEDULE_IMPACT`).
  - `src/optimization.py` uses Optuna to search over feature parameters; the best config is stored in `data/results/optimal_params.json` along with a CSV trial history.
  - `ScheduleAnalyzer` (`src/analysis.py`) pulls optimal params (or defaults), recomputes features (using feature cache), fits the chosen GLM, and materializes full-season results for plotting and CSV export.
- Forward prediction path
  - `scripts/run_predictions.py`:
    - Loads historical seasons and fits a `ScheduleAnalyzer` model using cached features and `optimal_params.json`.
    - Loads a prediction separately, computes features for that season (reusing the feature cache when possible), and splits games into “completed” vs “future” based on CLI arguments.
    - The team strength approximation is based on a rolling window of the past 10 games played for that team. The input block is not sorted by team, but at 300 games, all teams will have approximately played 10 games each. This is thus the minimum recommended number of games treated as completed; the strength is a strong predictor of performance, so games involving no data for either team are dropped from the prediction output entirely.
    - Calls `ScheduleAnalyzer.predict_from_features` on the future portion only, then writes predictions to a dedicated CSV and opens an interactive Matplotlib scatter/bar plot.

## Notes
- Features are strictly time-aware: team strength uses lagged rolling net rating; schedule density can look ahead because the schedule is known.
- Forward prediction runs include only completed games for strength; travel/load derive from published schedule.
- Dependencies include `mplcursors` for interactive hovering; everything else is mainstream DS stack (pandas, numpy, statsmodels, seaborn, optuna, sklearn).
