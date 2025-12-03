import copy
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import optuna

from src.config import (
    get_analysis_defaults,
    get_feature_defaults,
    get_optimization_defaults,
    get_path,
)
from src.features import FeatureEngineering
from src.model import ScheduleGLM, get_model_variant_formula

logger = logging.getLogger(__name__)

class ScheduleAnalyzer:
    
    def __init__(self, 
                 games: pd.DataFrame,
                 location_df: pd.DataFrame,
                 params: Optional[Dict] = None,
                 model_variant: Optional[str] = None):
        
        self.games = games
        self.location_df = location_df
        self.analysis_cfg = get_analysis_defaults()
        if params is None:
            params = self._load_optimal_params()
        self.params = copy.deepcopy(params)
        self.model_variant = model_variant or self.params.get('model_variant')
        fe = FeatureEngineering()
        self.features = fe.compute_features(games, location_df, self.params)
        model_formula = None
        if self.model_variant:
            try:
                model_formula = get_model_variant_formula(self.model_variant)
            except ValueError:
                logger.warning("Unknown model variant '%s', falling back to default", self.model_variant)
                model_formula = None
        self.model = ScheduleGLM(model_formula)
        self.model.fit(self.features)
        self.results = self.model.compute_counterfactuals(self.features)

    def _load_optimal_params(self) -> Dict:

        param_file = Path(get_path("optimal_params_file", "data/results/optimal_params.json"))
        if param_file.exists():
            with param_file.open('r') as f:
                return json.load(f)['best_params']
        else:
            logger.warning("No optimal params found, using defaults from config")
            return get_feature_defaults()
    
    def aggregate_team_effects(self) -> pd.DataFrame:

        team_effects = (
            self.results
            .groupby('TEAM_ID')
            .agg({
                'SCHEDULE_IMPACT': 'sum',
                'GAME_ID': 'count'
            })
            .rename(columns={
                'SCHEDULE_IMPACT': 'WINS_DUE_TO_SCHEDULE',
                'GAME_ID': 'N_GAMES'
            })
        )
        
        loader = pd.DataFrame.from_dict(
            {1610612737: 'ATL', 1610612738: 'BOS', 1610612751: 'BKN', 
             1610612766: 'CHA', 1610612741: 'CHI', 1610612739: 'CLE',
             1610612742: 'DAL', 1610612743: 'DEN', 1610612765: 'DET',
             1610612744: 'GSW', 1610612745: 'HOU', 1610612754: 'IND',
             1610612746: 'LAC', 1610612747: 'LAL', 1610612763: 'MEM',
             1610612748: 'MIA', 1610612749: 'MIL', 1610612750: 'MIN',
             1610612740: 'NOP', 1610612752: 'NYK', 1610612760: 'OKC',
             1610612753: 'ORL', 1610612755: 'PHI', 1610612756: 'PHX',
             1610612757: 'POR', 1610612758: 'SAC', 1610612759: 'SAS',
             1610612761: 'TOR', 1610612762: 'UTA', 1610612764: 'WAS'},
            orient='index', columns=['TEAM']
        )
        
        team_effects = team_effects.merge(loader, left_index=True, right_index=True, how='left')
        team_effects = team_effects.sort_values('WINS_DUE_TO_SCHEDULE', ascending=False)
        
        return team_effects
    
    def find_risky_games(self, 
                        team_id: Optional[int] = None,
                        threshold: Optional[float] = None) -> pd.DataFrame:

        risk_threshold = threshold if threshold is not None else self.analysis_cfg.get('risky_threshold', -0.10)
        risky = self.results[self.results['SCHEDULE_IMPACT'] < risk_threshold].copy()
        
        if team_id is not None:
            risky = risky[risky['TEAM_ID'] == team_id]
        
        bins = self.analysis_cfg.get('risk_bins', [-1, -0.20, -0.15, -0.10, 0])
        risky['RISK_LEVEL'] = pd.cut(
            risky['SCHEDULE_IMPACT'],
            bins=bins,
            labels=['Extreme', 'High', 'Moderate', 'Low'],
            include_lowest=True
        )

        risky = risky.sort_values('SCHEDULE_IMPACT')
        output_cols = [
            'GAME_DATE', 'TEAM_ABBREV', 'OPP_ABBREV', 'HOME',
            'P_FACTUAL', 'P_NEUTRAL', 'SCHEDULE_IMPACT',
            'RISK_LEVEL', 'LOAD_SCORE', 'TRAVEL_SCORE',
            'IS_BACK_TO_BACK', 'STRETCH_TYPE', 'DISTANCE_MILES'
        ]
        
        return risky[output_cols]
    
    def compute_calibration_threshold(self) -> float:

        bins_cfg = self.analysis_cfg.get('calibration_bins', {})
        bin_start = bins_cfg.get('start', -0.3)
        bin_stop = bins_cfg.get('stop', 0.3)
        bin_step = bins_cfg.get('step', 0.05)
        bins = np.arange(bin_start, bin_stop, bin_step)
        labels = bins[:-1] + (bin_step / 2)

        self.results['IMPACT_BIN'] = pd.cut(
            self.results['SCHEDULE_IMPACT'],
            bins=bins,
            labels=labels
        )
        
        calibration = (
            self.results
            .groupby('IMPACT_BIN')
            .agg({
                'WIN': 'mean',  # Actual win rate
                'P_NEUTRAL': 'mean',  # Expected if neutral
                'GAME_ID': 'count'  # Sample size
            })
        )
        
        calibration['divergence'] = calibration['P_NEUTRAL'] - calibration['WIN']
        min_games = self.analysis_cfg.get('calibration_min_games', 100)
        min_divergence = self.analysis_cfg.get('calibration_min_divergence', 0.05)
        significant = calibration[
            (calibration['GAME_ID'] > min_games) & 
            (calibration['divergence'] > min_divergence)
        ]
        
        if len(significant) > 0:
            threshold = significant.index[0]
            logger.info(f"Calibrated threshold: {threshold}")
            return float(threshold)
        else:
            fallback = self.analysis_cfg.get('risky_threshold', -0.10)
            logger.warning("Could not calibrate threshold, using default %s", fallback)
            return fallback
    
    def sensitivity_analysis(self, param_name: str, values: List[float]) -> pd.DataFrame:

        results = []
        
        fe = FeatureEngineering()
        
        for value in values:
            test_params = self.params.copy()
            test_params[param_name] = value
            features = fe.compute_features(self.games, self.location_df, test_params)
            model = ScheduleGLM()
            model.fit(features)
            metrics = model.evaluate(features)
            metrics[param_name] = value
            coeffs = model.get_coefficients()
            if 'TRAVEL_DIFF_Z' in coeffs.index:
                metrics['travel_coef'] = coeffs.loc['TRAVEL_DIFF_Z', 'Coef.']
                metrics['travel_p'] = coeffs.loc['TRAVEL_DIFF_Z', 'P>|z|']
            results.append(metrics)
        
        return pd.DataFrame(results)
    
    def plot_team_impacts(self, top_n: Optional[int] = None):

        team_effects = self.aggregate_team_effects()
        n = top_n if top_n is not None else self.analysis_cfg.get('top_n_teams', 10)

        top_teams = pd.concat([
            team_effects.head(n),
            team_effects.tail(n)
        ])

        colors = ['green' if x > 0 else 'red' for x in top_teams['WINS_DUE_TO_SCHEDULE']]
        plt.figure(figsize=(12, 6))
        plt.barh(range(len(top_teams)), top_teams['WINS_DUE_TO_SCHEDULE'], color=colors)
        plt.yticks(range(len(top_teams)), top_teams['TEAM'])
        plt.xlabel('Wins Gained/Lost Due to Schedule')
        plt.title('Schedule Impact by Team (2019-2024)')
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        plt.tight_layout()
        plot_path = Path(get_path('team_impacts_plot', 'data/results/team_schedule_impacts.png'))
        plt.savefig(plot_path, dpi=150)
        plt.show()
        
        return top_teams
    
    def plot_optimization_history(self, study_name: Optional[str] = None):
        
        opt_cfg = get_optimization_defaults()
        study_to_use = study_name or opt_cfg.get('study_name', 'nba_schedule')
        storage_path = opt_cfg.get('storage_file') or get_path('optuna_storage', 'data/cache/optuna_study.db')
        storage_url = f"sqlite:///{storage_path}"
        study = optuna.load_study(study_name=study_to_use, storage=storage_url)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1 = axes[0]
        trials = [t.value for t in study.trials if t.value is not None]
        ax1.plot(trials, alpha=0.6, label='Trial values')
        ax1.axhline(y=study.best_value, color='red', linestyle='--', label='Best value')
        ax1.set_xlabel('Trial')
        ax1.set_ylabel('Validation Loss')
        ax1.set_title('Optimization History')
        ax1.legend()
        
        ax2 = axes[1]
        importance = optuna.importance.get_param_importances(study)
        params = list(importance.keys())[:10]
        values = [importance[p] for p in params]
        
        ax2.barh(range(len(params)), values)
        ax2.set_yticks(range(len(params)))
        ax2.set_yticklabels(params)
        ax2.set_xlabel('Importance')
        ax2.set_title('Parameter Importance')
        
        plt.tight_layout()
        plot_path = Path(get_path('optimization_history_plot', 'data/results/optimization_history.png'))
        plt.savefig(plot_path, dpi=150)
        plt.show()
    
    def plot_feature_importance(self):
        
        coeffs = self.model.get_coefficients()
        main_effects = coeffs[~coeffs.index.str.contains('SEASON')]
        main_effects = main_effects.sort_values('Coef.')
        
        plt.figure(figsize=(10, 6))
        colors = ['red' if p < 0.05 else 'gray' for p in main_effects['P>|z|']]
        plt.barh(range(len(main_effects)), main_effects['Coef.'], color=colors)
        plt.yticks(range(len(main_effects)), main_effects.index)
        plt.xlabel('Coefficient')
        plt.title('Feature Importance (Red = Significant)')
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        plt.tight_layout()
        plot_path = Path(get_path('feature_importance_plot', 'data/results/feature_importance.png'))
        plt.savefig(plot_path, dpi=150)
        plt.show()
        
        return main_effects
    
    def export_predictions(self, 
                          season: Optional[str] = None,
                          output_file: Optional[str] = None):
        season_to_use = season or self.analysis_cfg.get('default_prediction_season', '2024')
        season_games = self.results[self.results['SEASON'] == season_to_use].copy()
        threshold = self.compute_calibration_threshold()
        season_games['IS_RISKY'] = season_games['SCHEDULE_IMPACT'] < threshold
        output = season_games[[
            'GAME_DATE', 'TEAM_ID', 'TEAM_ABBREV', 'OPP_ABBREV',
            'HOME', 'P_FACTUAL', 'P_NEUTRAL', 'SCHEDULE_IMPACT',
            'IS_RISKY', 'LOAD_SCORE', 'TRAVEL_SCORE'
        ]]
        output_path = Path(output_file or get_path('predictions_file', 'data/results/predictions.csv'))
        output.to_csv(output_path, index=False)
        logger.info(f"Exported {len(output)} predictions to {output_path}")
        
        return output

    def predict_from_features(self, features: pd.DataFrame) -> pd.DataFrame:

        if self.model is None or self.model.fit_result is None:
            raise ValueError("Model must be trained before running predictions")
        
        features = features.copy()
        predictions = self.model.compute_counterfactuals(features)
        return predictions


def create_summary_report(analyzer: ScheduleAnalyzer) -> Dict:

    team_effects = analyzer.aggregate_team_effects()
    coeffs = analyzer.model.get_coefficients()
    extreme_games = analyzer.find_risky_games(threshold=-0.15)
    
    summary = {
        'most_helped_teams': team_effects.head(5).to_dict('records'),
        'most_hurt_teams': team_effects.tail(5).to_dict('records'),
        'model_performance': {
            'aic': analyzer.model.fit_result.aic,
            'bic': analyzer.model.fit_result.bic,
            'log_likelihood': analyzer.model.fit_result.llf
        },
        'significant_factors': coeffs[coeffs['P>|z|'] < 0.05].to_dict('index'),
        'travel_impact': {
            'coefficient': coeffs.loc['TRAVEL_DIFF_Z', 'Coef.'] if 'TRAVEL_DIFF_Z' in coeffs.index else None,
            'p_value': coeffs.loc['TRAVEL_DIFF_Z', 'P>|z|'] if 'TRAVEL_DIFF_Z' in coeffs.index else None,
            'is_significant': coeffs.loc['TRAVEL_DIFF_Z', 'P>|z|'] < 0.05 if 'TRAVEL_DIFF_Z' in coeffs.index else False
        },
        'extreme_schedule_games': len(extreme_games),
        'calibrated_threshold': analyzer.compute_calibration_threshold()
    }
    
    output_file = Path(get_path('summary_file', 'data/results/analysis_summary.json'))
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Saved summary to {output_file}")
    
    return summary
