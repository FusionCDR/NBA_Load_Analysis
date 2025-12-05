import argparse
import logging
from pathlib import Path
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd

from src.analysis import ScheduleAnalyzer
from src.config import get_path, get_project_defaults
from src.data_loader import DataLoader, prepare_model_data, load_locations
from src.features import FeatureEngineering
from src.model import MODEL_VARIANT_SPECS
from src.optimization import load_optimal_params

OUTPUT_IMAGE = 'schedule_impact_analysis.png'

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_games_for_seasons(seasons: List[str]) -> pd.DataFrame:
    
    loader = DataLoader()
    games = loader.load_games(seasons)
    games = prepare_model_data(games)
    return games


def resolve_completed_games(
    season_games: pd.DataFrame,
    requested_completed: Optional[int],
    predict_games: Optional[int],
) -> Tuple[List[str], int]:

    unique_games = (
        season_games[["GAME_ID", "GAME_DATE"]]
        .drop_duplicates()
        .sort_values(["GAME_DATE", "GAME_ID"])
        .reset_index(drop=True)
    )

    if requested_completed is None:
        today = pd.Timestamp.today().normalize()
        requested_completed = int((unique_games["GAME_DATE"] <= today).sum())

    total_games = len(unique_games)
    completed_games = max(0, min(int(requested_completed), total_games))

    if completed_games >= total_games:
        return [], completed_games

    if predict_games is None:
        end_idx = total_games
    else:
        end_idx = min(total_games, completed_games + max(0, int(predict_games)))

    target_ids = unique_games.iloc[completed_games:end_idx]["GAME_ID"].tolist()

    return target_ids, completed_games


def build_output_path(season: str, completed_games: int, output_arg: Optional[str]) -> Path:

    if output_arg:
        output_path = Path(output_arg)
    else:
        results_dir = Path(get_path("results_dir", "data/results"))
        results_dir.mkdir(parents=True, exist_ok=True)
        filename = f"predictions_{season}_from_{completed_games}_games.csv"
        output_path = results_dir / filename

    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path

def plot_analysis(df):

    sns.set_theme(style="whitegrid")
    plt.rcParams['figure.figsize'] = (14, 10)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, height_ratios=[1.5, 1])
    fig.suptitle('NBA Schedule Impact Analysis', fontsize=16, weight='bold')
    df = df.reset_index(drop=True)
    df['Result'] = df['WL'].map({'W': 'Win', 'L': 'Loss'})
    
    colors_map = {'Win': '#2ecc71', 'Loss': '#e74c3c'}
    marker_map = {0: 'o', 1: 'X'}
    scatter_artists = []

    for b2b_val, marker in marker_map.items():
        subset = df[df['IS_BACK_TO_BACK'] == b2b_val]
        if subset.empty:
            continue
        artist = ax1.scatter(
            subset['SCHEDULE_IMPACT'],
            subset['P_FACTUAL'],
            c=subset['Result'].map(colors_map),
            marker=marker,
            s=200,
            alpha=0.9,
            edgecolor='black',
            linewidth=0.5,
        )
        artist._df_indices = subset.index.to_numpy()
        scatter_artists.append(artist)

    ax1.set_title('Win Probability vs. Schedule Impact', fontsize=14)
    ax1.set_xlabel('Schedule Impact (Negative = Fatigue/Travel Disadvantage)', fontsize=12)
    ax1.set_ylabel('Model Win Probability', fontsize=12)
    ax1.axvline(0, color='gray', linestyle='--', alpha=0.5)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Win', markerfacecolor=colors_map['Win'], markeredgecolor='black', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Loss', markerfacecolor=colors_map['Loss'], markeredgecolor='black', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Not B2B', markerfacecolor='gray', markeredgecolor='black', markersize=8),
        Line2D([0], [0], marker='X', color='w', label='B2B', markerfacecolor='gray', markeredgecolor='black', markersize=8),
    ]
    ax1.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1), title="Legend")

    import mplcursors

    if scatter_artists:
        cursor = mplcursors.cursor(scatter_artists, hover=True)

        @cursor.connect("add")
        def on_add(sel):
            artist = sel.artist
            df_indices = getattr(artist, "_df_indices", None)
            idx = df_indices[sel.index] if df_indices is not None else sel.index
            row = df.iloc[idx]
            sel.annotation.set(text=(
                f"{row['TEAM_ABBREV']} vs {row['OPP_ABBREV']}\n"
                f"Impact: {row['SCHEDULE_IMPACT']:.3f}\n"
                f"P(win): {row['P_FACTUAL']:.3f}"
            ))
            sel.annotation.get_bbox_patch().set(fc="white", ec="gray", alpha=0.95)

    fatigued_teams = df.sort_values('SCHEDULE_IMPACT').head(8)
    
    sns.barplot(
        data=fatigued_teams,
        x='SCHEDULE_IMPACT',
        y='TEAM_ABBREV',
        hue='Result',
        palette={'Win': '#2ecc71', 'Loss': '#e74c3c'},
        dodge=False,
        ax=ax2
    )
    
    ax2.set_title('Teams with Highest Schedule Disadvantage (Green=Won, Red=Lost)', fontsize=14)
    ax2.set_xlabel('Schedule Impact Score', fontsize=12)
    ax2.set_ylabel('Team', fontsize=12)
    ax2.axvline(0, color='gray', linestyle='-', alpha=0.5)
    ax2.legend(loc='upper right', title="Game Outcome")

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE)
    print(f"Analysis saved to {OUTPUT_IMAGE}")
    plt.show()

def main():

    parser = argparse.ArgumentParser(description="run forward predictions for a partial season")
    project_defaults = get_project_defaults()
    default_train = project_defaults.get("default_seasons", ["2019", "2020", "2021", "2022", "2023"])
    default_predict = default_train[-1]

    parser.add_argument(
        "--train-seasons",
        nargs="+",
        default=default_train[:-1],
        help="Seasons used for fitting the model",
    )
    parser.add_argument(
        "--predict-season",
        default=default_predict,
        help="Season to generate forward predictions for",
    )
    parser.add_argument(
        "--completed-games",
        type=int,
        default=None,
        help="Number of games in the prediction season treated as completed",
    )
    parser.add_argument(
        "--predict-games",
        type=int,
        default=None,
        help="Limit the number of future games to predict (default: all remaining).",
    )
    parser.add_argument(
        "--params-file",
        type=str,
        default=get_path("optimal_params_file", "data/results/optimal_params.json"),
        help="Path to the optimal parameter file",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Optional custom output CSV location.",
    )
    parser.add_argument(
        "--model-variant",
        choices=sorted(MODEL_VARIANT_SPECS.keys()),
        default=None,
        help="Override the model variant used for prediction (defaults to params file value).",
    )

    args = parser.parse_args()

    logger.info(
        "Training on seasons: %s; predicting %s",
        ", ".join(args.train_seasons),
        args.predict_season,
    )

    for key in ("raw_data_dir", "feature_cache_dir", "results_dir"):
        dir_path = get_path(key)
        if dir_path:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    games_train = load_games_for_seasons(args.train_seasons)
    games_predict = load_games_for_seasons([args.predict_season])
    location_df = load_locations()

    params = load_optimal_params(args.params_file)
    analyzer = ScheduleAnalyzer(
        games_train,
        location_df,
        params,
        model_variant=args.model_variant,
    )

    fe = FeatureEngineering()
    prediction_features = fe.compute_features(games_predict, location_df, analyzer.params)

    target_ids, completed_games = resolve_completed_games(
        games_predict, args.completed_games, args.predict_games
    )

    if not target_ids:
        logger.warning(
            "No games remain to predict after %s completed games for season %s",
            completed_games,
            args.predict_season,
        )
        return

    logger.info(
        "Treating first %s games as completed; generating predictions for %s future games",
        completed_games,
        len(target_ids),
    )

    predict_rows = prediction_features[prediction_features["GAME_ID"].isin(target_ids)].copy()
    predict_rows = predict_rows.sort_values(["GAME_DATE", "GAME_ID", "TEAM_ID"])
    predict_rows = predict_rows[predict_rows['STRENGTH_DIFF'] != 0]
    predictions = analyzer.predict_from_features(predict_rows)

    output_path = build_output_path(args.predict_season, completed_games, args.output_file)
    predictions.to_csv(output_path, index=False)

    plot_analysis(predictions)
    logger.info("Saved %s prediction rows to %s", len(predictions), output_path)


if __name__ == "__main__":
    main()
