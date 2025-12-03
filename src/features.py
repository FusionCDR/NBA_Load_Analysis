# computes schedule density, travel, load scores, and team strength features

import hashlib
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from haversine import Unit, haversine_vector

from src.config import get_path

TIMEZONE_OFFSETS = {
    'Eastern': -5,
    'Central': -6,
    'Mountain': -7,
    'Pacific': -8
}

logger = logging.getLogger(__name__)

class FeatureEngineering:
    
    def __init__(self, cache_dir: Optional[str] = None):
        default_cache = get_path("feature_cache_dir", "data/cache")
        self.cache_dir = Path(cache_dir or default_cache)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def compute_features(self, games: pd.DataFrame, location_df: pd.DataFrame, params: Dict) -> pd.DataFrame:

        cache_key = self.get_cache_key(games, params)
        cached_file = self.cache_dir / f"features_{cache_key}.parquet"
        if cached_file.exists():
            logger.info(f"Loading cached features from {cache_key}")
            return pd.read_parquet(cached_file)
        
        logger.info("Computing features from scratch")
        df = games.copy()
        
        df = compute_team_strength(df, window=params.get('window_N', 10))  
        df = compute_schedule_features(
            df, 
            games_lookback=params.get('games_lookback', 5),
            games_lookahead=params.get('games_lookahead', 1),
            stretch_configs=params.get('stretch_configs', {
                'n': [2, 3, 4, 4], 
                'm': [2, 4, 5, 6]
            })
        )
        df = compute_travel_features(
            df,
            location_df,
            travel_lookback=params.get('travel_lookback', 5),
            tz_lookback=params.get('tz_lookback', 5)
        )
        
        df = self.create_composite_scores(df, params)
        df = apply_transforms(df, params)
        df = create_differentials(df)
        df.to_parquet(cached_file, compression='snappy')
        
        return df
    
    def get_cache_key(self, games: pd.DataFrame, params: Dict) -> str:

        data_str = f"{len(games)}_{games['GAME_ID'].nunique()}_{games['SEASON'].unique().tolist()}"
        param_str = json.dumps(params, sort_keys=True)
        combined = f"{data_str}_{param_str}"
        return hashlib.md5(combined.encode()).hexdigest()[:16]
    
    def create_composite_scores(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:

        df['LOAD_SCORE'] = (
            df['REST_PENALTY'] * params.get('w_rest', 1.0) +
            df['STRETCH_PENALTY'] * params.get('w_stretch', 1.0) +
            df['GAME_DENSITY'] * params.get('w_density', 2.0)
        )

        df['TRAVEL_SCORE'] = (
            df['DISTANCE_IMPACT'] * params.get('w_distance', 1.0) +
            df['RECENT_TRAVEL'] * params.get('w_recent_travel', 1.0) +
            df['TZ_CHANGE'] * params.get('w_tz', 0.5) +
            df['TZ_CROSSINGS_ROLLING'] * params.get('w_tz_rolling', 0.5)
        )
        
        return df


def compute_team_strength(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    
    df = df.copy()
    df['STRENGTH_PRE'] = (
        df.groupby(['TEAM_ID', 'SEASON'])['NET_RATING_GAME']
        .transform(lambda x: x.rolling(window=window, min_periods=window).mean())
        .shift(1)  # lag by 1 to avoid leakage
        .fillna(0)
    )
    
    return df


def compute_schedule_features(df: pd.DataFrame, 
                              games_lookback: int = 5,
                              games_lookahead: int = 1,
                              stretch_configs: Dict = None) -> pd.DataFrame:
    
    df = df.copy()
    df = df.sort_values(['TEAM_ID', 'GAME_DATE'])
    
    df['PREV_GAME_DATE'] = df.groupby('TEAM_ID')['GAME_DATE'].shift(1)
    df['DAYS_REST'] = (df['GAME_DATE'] - df['PREV_GAME_DATE']).dt.days - 1
    df['DAYS_REST'] = df['DAYS_REST'].fillna(3).clip(lower=0, upper=3)
    
    df['IS_BACK_TO_BACK'] = (df['DAYS_REST'] == 0).astype(int) #b2b flag
    
    # Games density (games per day in window)
    # Note: games_lookahead is OK since schedule is predetermined
    window_size = games_lookback + games_lookahead + 1
    df['GAME_DENSITY'] = (
        df.groupby('TEAM_ID')['GAME_ID']
        .transform(lambda x: x.rolling(window=window_size, center=True, min_periods=1).count() / window_size)
    )
    
    # Consecutive road games
    df['CONSECUTIVE_ROAD'] = (
        df.groupby('TEAM_ID')['HOME']
        .transform(lambda x: (x == 0).groupby((x != 0).cumsum()).cumsum())
    )
    
    # Rest penalties (matching R)
    df['REST_PENALTY'] = np.select(
        [df['IS_BACK_TO_BACK'] == 1,
         df['DAYS_REST'] == 1,
         df['DAYS_REST'] == 2],
        [3, 2, 1],
        default=0
    )
    
    # Identify stretches (4-in-5, 4-in-6, etc.)
    df = identify_stretches(df, stretch_configs or {'n': [2, 3, 4, 4], 'm': [2, 4, 5, 6]})
    
    return df


def identify_stretches(df: pd.DataFrame, stretch_configs: Dict) -> pd.DataFrame:

    df = df.copy()
    df['STRETCH_TYPE'] = 'none'
    df['STRETCH_PENALTY'] = 0
    
    for n, m in zip(stretch_configs['n'], stretch_configs['m']):
        for team_id in df['TEAM_ID'].unique():
            team_df = df[df['TEAM_ID'] == team_id].copy()
            team_df = team_df.sort_values('GAME_DATE')
            for idx in team_df.index:
                window_start = df.loc[idx, 'GAME_DATE']
                window_end = window_start + timedelta(days=m-1)
                games_in_window = team_df[
                    (team_df['GAME_DATE'] >= window_start) & 
                    (team_df['GAME_DATE'] <= window_end)
                ]
                
                if len(games_in_window) >= n:
                    stretch_label = f"{n}-in-{m}"
                    df.loc[games_in_window.index, 'STRETCH_TYPE'] = stretch_label
                    
                    if n == 4 and m == 5:
                        df.loc[games_in_window.index, 'STRETCH_PENALTY'] = 4
                    elif (n == 4 and m == 6) or (n == 3 and m == 4):
                        df.loc[games_in_window.index, 'STRETCH_PENALTY'] = 2
                    elif n == 2 and m == 2:
                        df.loc[games_in_window.index, 'STRETCH_PENALTY'] = 1
    
    return df


def compute_travel_features(df: pd.DataFrame, 
                           location_df: pd.DataFrame,
                           travel_lookback: int = 5,
                           tz_lookback: int = 5) -> pd.DataFrame:

    df = df.copy()
    
    team_locations = location_df.rename(columns={
        'TEAM': 'TEAM_ABBREV',
        'LATITUDE': 'TEAM_LAT',
        'LONGITUDE': 'TEAM_LON',
        'TIMEZONE': 'TEAM_TZ'
    })
    df = df.merge(
        team_locations[['TEAM_ABBREV', 'TEAM_LAT', 'TEAM_LON', 'TEAM_TZ']],
        on='TEAM_ABBREV',
        how='left'
    )

    opp_locations = location_df.rename(columns={
        'TEAM': 'OPP_ABBREV',
        'LATITUDE': 'OPP_LAT',
        'LONGITUDE': 'OPP_LON',
        'TIMEZONE': 'OPP_TZ'
    })

    df = df.merge(
        opp_locations[['OPP_ABBREV', 'OPP_LAT', 'OPP_LON', 'OPP_TZ']],
        on='OPP_ABBREV',
        how='left'
    )

    # Normalize tz vals
    for tz_col in ['TEAM_TZ', 'OPP_TZ']:
        if tz_col in df.columns:
            mapped = df[tz_col].map(TIMEZONE_OFFSETS)
            missing_mask = df[tz_col].notna() & mapped.isna()
            if missing_mask.any():
                unknown_labels = sorted(df.loc[missing_mask, tz_col].unique())
                raise ValueError(f"Unknown timezone labels encountered: {unknown_labels}")
            df[tz_col] = mapped
    
    df['GAME_LAT'] = np.where(df['HOME'] == 1, df['TEAM_LAT'], df['OPP_LAT'])
    df['GAME_LON'] = np.where(df['HOME'] == 1, df['TEAM_LON'], df['OPP_LON'])
    df['GAME_TZ'] = np.where(df['HOME'] == 1, df['TEAM_TZ'], df['OPP_TZ'])

    df = df.sort_values(['TEAM_ID', 'GAME_DATE'])
    df['PREV_LAT'] = df.groupby('TEAM_ID')['GAME_LAT'].shift(1)
    df['PREV_LON'] = df.groupby('TEAM_ID')['GAME_LON'].shift(1)
    df['PREV_TZ'] = df.groupby('TEAM_ID')['GAME_TZ'].shift(1)
    
    valid_coords = df[['PREV_LAT', 'PREV_LON', 'GAME_LAT', 'GAME_LON']].notna().all(axis=1)
    df['DISTANCE_MILES'] = 0.0
    if valid_coords.any():
        df.loc[valid_coords, 'DISTANCE_MILES'] = haversine_vector(
            df.loc[valid_coords, ['PREV_LAT', 'PREV_LON']].to_numpy(),
            df.loc[valid_coords, ['GAME_LAT', 'GAME_LON']].to_numpy(),
            Unit.MILES
        )
    
    # Non-linear distance impact (diminishing returns)
    df['DISTANCE_IMPACT'] = np.where(
        df['DISTANCE_MILES'] < 500, 
        df['DISTANCE_MILES'] / 500,
        np.where(
            df['DISTANCE_MILES'] < 2000,
            1 + np.log1p(df['DISTANCE_MILES'].clip(lower=500) - 500) / np.log1p(1500),
            2  # Plateau for very long trips
        )
    )
    
    # Timezone changes
    df['TZ_CHANGE'] = np.abs(df['GAME_TZ'] - df['PREV_TZ']).fillna(0)
    
    # Rolling travel accumulation (past games only)
    df['RECENT_TRAVEL'] = (
        df.groupby('TEAM_ID')['DISTANCE_MILES']
        .transform(lambda x: x.rolling(window=travel_lookback, min_periods=1).sum())
        .shift(1) / 1000  # Shift to avoid including current game, scale by 1000
    ).fillna(0)
    
    # Rolling timezone crossings
    df['TZ_CROSSINGS_ROLLING'] = (
        df.groupby('TEAM_ID')['TZ_CHANGE']
        .transform(lambda x: x.rolling(window=tz_lookback, min_periods=1).sum())
    )
    
    return df


def apply_transforms(df: pd.DataFrame, params: Dict) -> pd.DataFrame:

    df = df.copy()
    
    # log transformations for distances
    if params.get('use_log_distance', False):
        distance_for_log = df['DISTANCE_MILES'].clip(lower=0).fillna(0)
        with np.errstate(invalid='ignore'):
            df['LOG_DISTANCE'] = np.log1p(distance_for_log)
    
    # polynomial terms for extreme values
    if params.get('use_polynomial', False):
        df['LOAD_SCORE_SQ'] = df['LOAD_SCORE'] ** 2
        df['TRAVEL_SCORE_SQ'] = df['TRAVEL_SCORE'] ** 2
    
    # travel matters more when tired
    if params.get('use_interactions', True):
        df['LOAD_TRAVEL_INTERACTION'] = df['LOAD_SCORE'] * df['TRAVEL_SCORE']
        df['B2B_TRAVEL_INTERACTION'] = df['IS_BACK_TO_BACK'] * df['DISTANCE_IMPACT']
    
    # threshold indicators
    df['EXTREME_TRAVEL'] = (df['DISTANCE_MILES'] > 2000).astype(int)
    df['EXTREME_LOAD'] = (df['LOAD_SCORE'] > df['LOAD_SCORE'].quantile(0.9)).astype(int)
    
    return df


def create_differentials(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    
    opp_cols = ['GAME_ID', 'TEAM_ID', 'STRENGTH_PRE', 'LOAD_SCORE', 'TRAVEL_SCORE']
    df_opp = df[opp_cols].copy()
    df_opp.columns = ['GAME_ID', 'OPP_ID', 'STRENGTH_PRE_OPP', 'LOAD_SCORE_OPP', 'TRAVEL_SCORE_OPP']
    
    df = df.merge(df_opp, on=['GAME_ID', 'OPP_ID'], how='left')
    
    df['STRENGTH_DIFF'] = df['STRENGTH_PRE'] - df['STRENGTH_PRE_OPP']
    df['LOAD_DIFF'] = df['LOAD_SCORE'] - df['LOAD_SCORE_OPP']
    df['TRAVEL_DIFF'] = df['TRAVEL_SCORE'] - df['TRAVEL_SCORE_OPP']
    
    for col in ['LOAD_SCORE', 'TRAVEL_SCORE', 'LOAD_DIFF', 'TRAVEL_DIFF']:
        df[f'{col}_Z'] = df.groupby('SEASON')[col].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-8)
        )
    
    return df


def safe_zscore(x: pd.Series) -> pd.Series:

    if x.isna().all():
        return pd.Series(np.nan, index=x.index)
    
    mean_val = x.mean()
    std_val = x.std()
    
    if pd.isna(std_val) or std_val == 0:
        return pd.Series(0, index=x.index)
    
    return (x - mean_val) / std_val
