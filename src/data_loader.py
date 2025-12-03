import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from nba_api.stats.endpoints import leaguegamelog

from src.config import get_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    
    def __init__(self, cache_dir: Optional[str] = None):
        default_cache = get_path("raw_data_dir", "data/raw")
        self.cache_dir = Path(cache_dir or default_cache)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.team_mapping = {
            1610612737: 'ATL', 1610612738: 'BOS', 1610612751: 'BKN', 
            1610612766: 'CHA', 1610612741: 'CHI', 1610612739: 'CLE',
            1610612742: 'DAL', 1610612743: 'DEN', 1610612765: 'DET',
            1610612744: 'GSW', 1610612745: 'HOU', 1610612754: 'IND',
            1610612746: 'LAC', 1610612747: 'LAL', 1610612763: 'MEM',
            1610612748: 'MIA', 1610612749: 'MIL', 1610612750: 'MIN',
            1610612740: 'NOP', 1610612752: 'NYK', 1610612760: 'OKC',
            1610612753: 'ORL', 1610612755: 'PHI', 1610612756: 'PHX',
            1610612757: 'POR', 1610612758: 'SAC', 1610612759: 'SAS',
            1610612761: 'TOR', 1610612762: 'UTA', 1610612764: 'WAS'
        }
        
    def fetch_season_data(self, season: str, force_refresh: bool = False) -> pd.DataFrame:
        # check cache
        cache_file = self.cache_dir / f"games_{season}.parquet"
        
        if cache_file.exists() and not force_refresh:
            logger.info(f"Loading cached data for {season}")
            return pd.read_parquet(cache_file)
        
        logger.info(f"Fetching data from API for {season}")
        try:
            # format: "2019-20" for api
            season_str = f"{season}-{str(int(season) + 1)[2:]}"
            gamelog = leaguegamelog.LeagueGameLog(
                season=season_str,
                season_type_all_star="Regular Season"
            )
            df = gamelog.get_data_frames()[0]
            df.to_parquet(cache_file, compression='snappy')
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch {season}: {e}")
            raise
    
    def load_games(self, seasons: List[str]) -> pd.DataFrame:

        all_games = []
        
        for season in seasons:
            season_df = self.fetch_season_data(season)
            season_df['SEASON'] = season
            all_games.append(season_df)
        
        games = pd.concat(all_games, ignore_index=True)
        return self._process_raw_games(games)
    
    def _process_raw_games(self, df: pd.DataFrame) -> pd.DataFrame:
        
        games = df.copy()
        games['HOME'] = games['MATCHUP'].str.contains('vs.').astype(int)
        games['WIN'] = (games['WL'] == 'W').astype(int)
        games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])
        
        opp_stats_cols = ['GAME_ID', 'TEAM_ID', 'PTS', 'FGA', 'OREB', 'TOV', 'FTA']
        games_opp = games[opp_stats_cols].copy()
        games_opp.columns = ['GAME_ID', 'OPP_ID', 'OPP_PTS', 'OPP_FGA', 'OPP_OREB', 'OPP_TOV', 'OPP_FTA']
        games = games.merge(
            games_opp, 
            on='GAME_ID',
            how='inner'
        )
        games = games[games['TEAM_ID'] != games['OPP_ID']]
        
        return games


def prepare_model_data(games: pd.DataFrame) -> pd.DataFrame:

    df = games.copy()

    def estimate_possessions(fga, oreb, tov, fta):
        return fga - oreb + tov + 0.44 * fta

    team_possessions = estimate_possessions(df['FGA'], df['OREB'], df['TOV'], df['FTA'])
    opp_possessions = estimate_possessions(df['OPP_FGA'], df['OPP_OREB'], df['OPP_TOV'], df['OPP_FTA'])

    team_possessions = team_possessions.mask(team_possessions == 0)
    opp_possessions = opp_possessions.mask(opp_possessions == 0)

    off_rating = (df['PTS'] / team_possessions) * 100
    def_rating = (df['OPP_PTS'] / opp_possessions) * 100
    df['NET_RATING_GAME'] = (off_rating - def_rating).fillna(0)

    df = df.sort_values(['TEAM_ID', 'SEASON', 'GAME_DATE', 'GAME_ID'])
    
    loader = DataLoader()
    df['TEAM_ABBREV'] = df['TEAM_ID'].map(loader.team_mapping)
    df['OPP_ABBREV'] = df['OPP_ID'].map(loader.team_mapping)
    
    return df


def load_locations(filepath: Optional[str] = None) -> pd.DataFrame:
    loc_path = Path(filepath or get_path("locations_file", "data/locations.json"))
    with loc_path.open('r') as f:
        locations = json.load(f)
        location_df = pd.DataFrame(locations)
        column_map = {'team': 'TEAM'}
        location_df = location_df.rename(columns=lambda c: column_map.get(c, c.upper()))

    return location_df
