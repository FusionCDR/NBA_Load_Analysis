#!/usr/bin/env python

import argparse
import logging
import pandas as pd

from pathlib import Path
from typing import Optional

from src.data_loader import DataLoader, prepare_model_data, load_locations
from src.features import FeatureEngineering
from src.model import compare_models, MODEL_VARIANT_SPECS
from src.optimization import Optimizer, quick_optimization, load_optimal_params
from src.analysis import ScheduleAnalyzer, create_summary_report
from src.config import get_path, get_project_defaults

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_data_pipeline(seasons: list = None) -> tuple:

    if seasons is None:
        project_defaults = get_project_defaults()
        seasons = project_defaults.get('default_seasons', ['2019', '2020', '2021', '2022', '2023', '2024'])
    
    logger.info(f"Loading data for seasons: {seasons}")
    
    loader = DataLoader()
    games = loader.load_games(seasons)
    games = prepare_model_data(games)
    location_df = load_locations()

    logger.info(f"Loaded {len(games)} games from {len(seasons)} seasons")
    
    return games, location_df

def run_optimization(games: pd.DataFrame, 
                    location_df: pd.DataFrame,
                    n_trials: int = 100,
                    timeout: int = None,
                    quick: bool = False,
                    model_variant: Optional[str] = None) -> dict:

    logger.info("Starting optimization")
    project_defaults = get_project_defaults()
    resolved_variant = model_variant or project_defaults.get('default_model_variant', 'basic')
    
    if quick:
        logger.info("Running quick version")
        params = quick_optimization(
            games, 
            location_df, 
            n_trials=min(n_trials, 50),
            model_variant=resolved_variant
        )
    else:
        optimizer = Optimizer(
            games, 
            location_df, 
            model_variant=resolved_variant
        )
        study = optimizer.run_optimization(n_trials=n_trials, timeout=timeout)
        
        params = optimizer.best_params or {
            **study.best_params,
            'model_variant': resolved_variant
        }
        
        importance = optimizer.analyze_importance(study)
        logger.info(f"Top 3 important parameters:\n{importance.head(3)}")
    
    results_dir = get_path('results_dir', 'data/results')
    logger.info(f"Optimization complete. Best params saved to {results_dir}/")
    
    return params


def run_analysis(games: pd.DataFrame, 
                location_df: pd.DataFrame,
                params: dict = None) -> dict:

    logger.info("Running analysis")
    
    analyzer = ScheduleAnalyzer(games, location_df, params)
    team_effects = analyzer.aggregate_team_effects()
    logger.info(f"Most helped team: {team_effects.iloc[0]['TEAM']} "
                f"(+{team_effects.iloc[0]['WINS_DUE_TO_SCHEDULE']:.2f} wins)")
    logger.info(f"Most hurt team: {team_effects.iloc[-1]['TEAM']} "
                f"({team_effects.iloc[-1]['WINS_DUE_TO_SCHEDULE']:.2f} wins)")
    
    logger.info("Creating visualizations")
    analyzer.plot_team_impacts()
    analyzer.plot_feature_importance()
    
    try:
        analyzer.plot_optimization_history()
    except:
        logger.info("No optimization history to plot")
    
    analyzer.export_predictions()
    summary = create_summary_report(analyzer)
    
    return summary


def run_model_comparison(games: pd.DataFrame, location_df: pd.DataFrame) -> pd.DataFrame:

    logger.info("Comparing model variants")
    
    fe = FeatureEngineering()
    params = load_optimal_params()
    features = fe.compute_features(games, location_df, params)
    comparison = compare_models(features)
    comparison_path = Path(get_path('model_comparison_file', 'data/results/model_comparison.csv'))
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(comparison_path, index=False)

    if comparison.empty:
        logger.error("Model comparison produced no valid results. Skipping downstream steps.")
        return comparison
    
    logger.info(f"Model comparison:\n{comparison}")
    
    return comparison


def main():

    parser = argparse.ArgumentParser(description='NBA Schedule Impact Analysis')
    project_defaults = get_project_defaults()
    default_seasons = project_defaults.get('default_seasons', ['2019', '2020', '2021', '2022', '2023'])
    default_variant = project_defaults.get('default_model_variant', 'basic')
    
    parser.add_argument('--mode', choices=['full', 'optimize', 'analyze', 'compare'],
                       default='full', help='Execution mode')
    parser.add_argument('--seasons', nargs='+', 
                       default=default_seasons,
                       help='Seasons to analyze')
    parser.add_argument('--n-trials', type=int, default=100,
                       help='Number of Optuna trials')
    parser.add_argument('--timeout', type=int, default=None,
                       help='Optimization timeout in seconds')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode with fewer trials')
    parser.add_argument('--model-variant', choices=sorted(MODEL_VARIANT_SPECS.keys()),
                       default=default_variant, help='Model variant defined in src.model')
    
    args = parser.parse_args()
    
    for key in ('raw_data_dir', 'feature_cache_dir', 'results_dir'):
        dir_path = get_path(key)
        if dir_path:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    games, location_df = run_data_pipeline(args.seasons)
    
    if args.mode == 'optimize':
        params = run_optimization(
            games, location_df, 
            n_trials=args.n_trials,
            timeout=args.timeout,
            quick=args.quick,
            model_variant=args.model_variant
        )
        
    elif args.mode == 'analyze':
        summary = run_analysis(games, location_df)
        
    elif args.mode == 'compare':
        comparison = run_model_comparison(games, location_df)
        if comparison.empty:
            logger.error("No model comparison results available, exiting")
            return
        
    else:
        logger.info("Running full pipeline")
        comparison = run_model_comparison(games, location_df)
        if comparison.empty:
            logger.error("No model comparison results available, aborting")
            return
        
        best_row = comparison.iloc[0]
        travel_significant = best_row.get('travel_significant')
        
        if travel_significant is False:
            logger.warning("Travel not significant with default params, running optimization")
            
            params = run_optimization(
                games, location_df,
                n_trials=args.n_trials,
                timeout=args.timeout,
                quick=args.quick,
                model_variant=args.model_variant
            )
            
            summary = run_analysis(games, location_df, params)
        else:
            if travel_significant is None:
                logger.warning("Could not determine travel significance from comparison results, proceeding without optimization")
            summary = run_analysis(games, location_df)
    
    logger.info("Pipeline complete, outputs in %s/", get_path('results_dir', 'data/results'))


if __name__ == "__main__":
    main()
