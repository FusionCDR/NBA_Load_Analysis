import copy
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


DEFAULT_CONFIG: Dict[str, Any] = {
    "project": {
        "default_seasons": ["2019", "2020", "2021", "2022", "2023", "2024"],
        "default_model_variant": "basic",
    },
    "paths": {
        "raw_data_dir": "data/raw",
        "feature_cache_dir": "data/cache",
        "results_dir": "data/results",
        "locations_file": "data/locations.json",
        "optuna_storage": "data/cache/optuna_study.db",
        "predictions_file": "data/results/predictions.csv",
        "summary_file": "data/results/analysis_summary.json",
        "optimal_params_file": "data/results/optimal_params.json",
        "model_comparison_file": "data/results/model_comparison.csv",
        "optimization_history_file": "data/results/optimization_history.csv",
        "team_impacts_plot": "data/results/team_schedule_impacts.png",
        "optimization_history_plot": "data/results/optimization_history.png",
        "feature_importance_plot": "data/results/feature_importance.png",
    },
    "feature_params": {
        "window_N": 10,
        "games_lookback": 5,
        "games_lookahead": 1,
        "travel_lookback": 5,
        "tz_lookback": 5,
        "w_rest": 1.0,
        "w_stretch": 1.0,
        "w_density": 2.0,
        "w_distance": 1.0,
        "w_recent_travel": 1.0,
        "w_tz": 0.5,
        "w_tz_rolling": 0.5,
        "stretch_configs": {
            "n": [2, 3, 4, 4],
            "m": [2, 4, 5, 6],
        },
        "use_log_distance": False,
        "use_polynomial": False,
        "use_interactions": True,
    },
    "analysis": {
        "risky_threshold": -0.10,
        "risk_bins": [-1.0, -0.20, -0.15, -0.10, 0.0],
        "calibration_bins": {"start": -0.3, "stop": 0.3, "step": 0.05},
        "calibration_min_games": 100,
        "calibration_min_divergence": 0.05,
        "top_n_teams": 10,
        "default_prediction_season": "2024",
    },
    "optimization": {
        "study_name": "nba_schedule_load",
        "storage_file": "data/cache/optuna_study.db",
        "n_splits": 3,
        "val_size": 0.15,
        "test_size": 0.15,
    },
}


def deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


@lru_cache(maxsize=1)
def load_from_disk(config_path: str) -> Dict[str, Any]:
    """Load config.yaml once and merge onto defaults."""
    path = Path(config_path)
    merged = copy.deepcopy(DEFAULT_CONFIG)
    if path.exists():
        with path.open("r") as f:
            user_config = yaml.safe_load(f) or {}
        merged = deep_merge(merged, user_config)
    return merged


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    config_path = path or "config.yaml"
    return copy.deepcopy(load_from_disk(config_path))


def get_config_section(section: str, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    config = load_config()
    value = config.get(section)
    if value is None:
        return copy.deepcopy(default) if default is not None else {}
    return copy.deepcopy(value)


def get_feature_defaults() -> Dict[str, Any]:
    return get_config_section("feature_params")


def get_path(name: str, fallback: Optional[str] = None) -> Optional[str]:
    paths = get_config_section("paths")
    return paths.get(name, fallback)


def get_project_defaults() -> Dict[str, Any]:
    return get_config_section("project")


def get_analysis_defaults() -> Dict[str, Any]:
    return get_config_section("analysis")


def get_optimization_defaults() -> Dict[str, Any]:
    return get_config_section("optimization")
