import optuna
import pandas as pd
import numpy as np
from typing import Dict, Optional, Callable
import logging
import json
from pathlib import Path
from datetime import datetime

from src.config import (
    get_feature_defaults,
    get_optimization_defaults,
    get_path,
    get_project_defaults,
)
from src.features import FeatureEngineering
from src.model import (
    ScheduleGLM,
    time_series_split,
    get_model_variant_formula,
    get_variant_required_columns,
    is_variant_supported
)

logger = logging.getLogger(__name__)


class Optimizer:

    def __init__(self, 
                 games: pd.DataFrame,
                 location_df: pd.DataFrame,
                 study_name: Optional[str] = None,
                 storage_path: Optional[str] = None,
                 model_variant: Optional[str] = None):
        
        self.games = games
        self.location_df = location_df
        opt_cfg = get_optimization_defaults()
        project_cfg = get_project_defaults()
        self.study_name = study_name or opt_cfg.get('study_name', 'nba_schedule_load')
        storage_file = storage_path or opt_cfg.get('storage_file') or get_path('optuna_storage', 'data/cache/optuna_study.db')
        storage_file = str(storage_file)
        self.storage_url = f"sqlite:///{storage_file}"
        resolved_variant = model_variant or project_cfg.get('default_model_variant', 'basic')
        self.model_variant = resolved_variant
        self.model_formula = get_model_variant_formula(resolved_variant)
        logger.info("Optuna optimizer targeting model variant: %s", self.model_variant)
        self.splits = time_series_split(
            games,
            n_splits=opt_cfg.get('n_splits', 3),
            val_size=opt_cfg.get('val_size', 0.15),
            test_size=opt_cfg.get('test_size', 0.15)
        )
        self.best_params = None
        self.best_score = float('inf')
        
    def create_search_space(self, trial: optuna.Trial) -> Dict:

        params = {
            'window_N': trial.suggest_int('window_N', 5, 15),  # used 10 in simple
            'games_lookback': trial.suggest_int('games_lookback', 3, 7),  # 5
            'games_lookahead': trial.suggest_int('games_lookahead', 0, 2),  # 1
            'travel_lookback': trial.suggest_int('travel_lookback', 3, 7),  # 5
            'tz_lookback': trial.suggest_int('tz_lookback', 3, 7),  # 5
            'w_rest': trial.suggest_float('w_rest', 0.5, 2.0),  # 1.0
            'w_stretch': trial.suggest_float('w_stretch', 0.5, 2.0),  # 1.0
            'w_density': trial.suggest_float('w_density', 1.0, 3.0),  # 2.0
            'w_distance': trial.suggest_float('w_distance', 0.1, 2.0, log=True),
            'w_recent_travel': trial.suggest_float('w_recent_travel', 0.1, 2.0, log=True),
            'w_tz': trial.suggest_float('w_tz', 0.1, 1.5),
            'w_tz_rolling': trial.suggest_float('w_tz_rolling', 0.1, 1.0),
            'use_log_distance': trial.suggest_categorical('use_log_distance', [True, False]),
            'use_polynomial': trial.suggest_categorical('use_polynomial', [True, False]),
            'use_interactions': trial.suggest_categorical('use_interactions', [True, False])
        }
        
        params['stretch_configs'] = {
            'n': [2, 3, 4, 4],
            'm': [2, 4, 5, 6]
        }
        
        return params
    
    def objective(self, trial: optuna.Trial) -> float:

        params = self.create_search_space(trial)
        fe = FeatureEngineering()
        
        val_scores = []
        
        for i, split in enumerate(self.splits):
            train_data = self.games.loc[split['train']]
            val_data = self.games.loc[split['val']]
            combined = pd.concat([train_data, val_data])
            features = fe.compute_features(combined, self.location_df, params)
            
            if not is_variant_supported(self.model_variant, features.columns):
                required_cols = get_variant_required_columns(self.model_variant)
                missing_cols = sorted(required_cols - set(features.columns))
                logger.warning(
                    "Trial %s fold %s missing required columns %s for variant '%s'. "
                    "Returning inf.",
                    trial.number,
                    i,
                    missing_cols,
                    self.model_variant
                )
                return float('inf')
            
            train_features = features.iloc[:len(split['train'])]
            val_features = features.iloc[len(split['train']):len(split['train'])+len(split['val'])]
            model = ScheduleGLM(self.model_formula)
            
            try:
                model.fit(train_features)
                print(f"Train NaNs: {train_features.isna().sum().sum()}")
                print(f"Val NaNs: {val_features.isna().sum().sum()}")
                val_features = val_features.dropna(subset=['WIN', 'HOME', 'STRENGTH_DIFF', 'LOAD_DIFF_Z', 'TRAVEL_DIFF_Z'])
                metrics = model.evaluate(val_features)
                val_scores.append(metrics['log_loss'])
                trial.report(metrics['log_loss'], i)
                if trial.should_prune():
                    raise optuna.TrialPruned()
                    
            except Exception as e:
                logger.warning(f"Trial {trial.number} fold {i} failed: {e}")
                raise optuna.TrialPruned()
        
        avg_loss = np.mean(val_scores)
        
        if avg_loss < self.best_score:
            self.best_score = avg_loss
            best_params = dict(params)
            best_params['model_variant'] = self.model_variant
            self.best_params = best_params
            logger.info(f"New best score: {avg_loss:.4f}")
        
        return avg_loss
    
    def run_optimization(self, 
                        n_trials: int = 100,
                        timeout: Optional[int] = None,
                        n_jobs: int = 1) -> optuna.Study:

        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage_url,
            load_if_exists=True,
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=10,
                n_warmup_steps=1
            )
        )
        
        logger.info(
            "Starting optimization for variant '%s': %s trials, timeout=%ss",
            self.model_variant,
            n_trials,
            timeout
        )
        
        study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            show_progress_bar=True
        )
        
        self.save_best_params(study)
        
        return study
    
    def save_best_params(self, study: optuna.Study):

        results_dir = Path(get_path('results_dir', 'data/results'))
        results_dir.mkdir(parents=True, exist_ok=True)
        
        best_trial = study.best_trial
        best_params = dict(best_trial.params)
        best_params['model_variant'] = self.model_variant
        
        results = {
            'best_params': best_params,
            'best_value': best_trial.value,
            'n_trials': len(study.trials),
            'optimization_date': datetime.now().isoformat(),
            'study_name': self.study_name
        }
        
        default_optimal = results_dir / "optimal_params.json"
        output_file = Path(get_path('optimal_params_file', str(default_optimal)))
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open('w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved best parameters to {output_file}")
        
        # Also save detailed trial history
        history = study.trials_dataframe()
        history.columns = [col.upper() for col in history.columns]
        history_file = Path(get_path('optimization_history_file', str(results_dir / "optimization_history.csv")))
        history_file.parent.mkdir(parents=True, exist_ok=True)
        history.to_csv(history_file, index=False)
    
    def analyze_importance(self, study: optuna.Study) -> pd.DataFrame:
        
        importance = optuna.importance.get_param_importances(study)
        df_importance = pd.DataFrame([
            {'PARAMETER': k, 'IMPORTANCE': v} for k, v in importance.items()
        ]).sort_values('IMPORTANCE', ascending=False)
        
        return df_importance


def load_optimal_params(filepath: Optional[str] = None) -> Dict:

    project_cfg = get_project_defaults()
    variant = project_cfg.get('default_model_variant', 'basic')
    param_path = Path(filepath or get_path('optimal_params_file', 'data/results/optimal_params.json'))
    if not param_path.exists():
        logger.warning("Optimal params file %s not found, using defaults from config", param_path)
        params = get_feature_defaults()
        params.setdefault('model_variant', variant)
        return params

    with param_path.open('r') as f:
        results = json.load(f)
    params = dict(results['best_params'])
    params.setdefault('model_variant', variant)
    return params

# config'd quick run
def quick_optimization(games: pd.DataFrame, 
                       location_df: pd.DataFrame,
                       n_trials: int = 50,
                       model_variant: Optional[str] = None) -> Dict:

    project_cfg = get_project_defaults()
    resolved_variant = model_variant or project_cfg.get('default_model_variant', 'basic')

    optimizer = Optimizer(
        games, 
        location_df, 
        study_name="nba_quick_test",
        model_variant=resolved_variant
    )
    
    # simple search space
    def simple_search_space(trial):
        return {
            'window_N': 10,  # Fix most parameters
            'games_lookback': 5,
            'games_lookahead': 1,
            'travel_lookback': 5,
            'tz_lookback': 5,
            'stretch_configs': {'n': [2, 3, 4, 4], 'm': [2, 4, 5, 6]},
            
            'w_rest': trial.suggest_float('w_rest', 0.5, 2.0),
            'w_stretch': trial.suggest_float('w_stretch', 0.5, 2.0),
            'w_density': trial.suggest_float('w_density', 1.0, 3.0),
            'w_distance': trial.suggest_float('w_distance', 0.5, 2.0),
            'w_recent_travel': trial.suggest_float('w_recent_travel', 0.5, 2.0),
            'w_tz': 0.5,
            'w_tz_rolling': 0.5,
            
            'use_log_distance': True,
            'use_polynomial': False,
            'use_interactions': True
        }
    
    optimizer.create_search_space = simple_search_space
    study = optimizer.run_optimization(n_trials=n_trials)
    return optimizer.best_params or {
        **study.best_params,
        'model_variant': resolved_variant
    }
