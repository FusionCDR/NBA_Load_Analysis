import pandas as pd

from src.features import FeatureEngineering, compute_travel_features

def _sample_games() -> pd.DataFrame:
    data = [
        {
            'TEAM_ID': 100,
            'OPP_ID': 200,
            'SEASON': '2023',
            'GAME_DATE': '2023-10-25',
            'GAME_ID': 'G1',
            'HOME': 1,
            'TEAM_ABBREV': 'ATL',
            'OPP_ABBREV': 'LAL',
            'NET_RATING_GAME': 8.0,
            'PTS': 118,
            'OPP_PTS': 110,
            'WIN': 1
        },
        {
            'TEAM_ID': 200,
            'OPP_ID': 100,
            'SEASON': '2023',
            'GAME_DATE': '2023-10-25',
            'GAME_ID': 'G1',
            'HOME': 0,
            'TEAM_ABBREV': 'LAL',
            'OPP_ABBREV': 'ATL',
            'NET_RATING_GAME': -8.0,
            'PTS': 110,
            'OPP_PTS': 118,
            'WIN': 0
        },
        {
            'TEAM_ID': 100,
            'OPP_ID': 200,
            'SEASON': '2023',
            'GAME_DATE': '2023-10-27',
            'GAME_ID': 'G2',
            'HOME': 0,
            'TEAM_ABBREV': 'ATL',
            'OPP_ABBREV': 'LAL',
            'NET_RATING_GAME': -6.0,
            'PTS': 102,
            'OPP_PTS': 108,
            'WIN': 0
        },
        {
            'TEAM_ID': 200,
            'OPP_ID': 100,
            'SEASON': '2023',
            'GAME_DATE': '2023-10-27',
            'GAME_ID': 'G2',
            'HOME': 1,
            'TEAM_ABBREV': 'LAL',
            'OPP_ABBREV': 'ATL',
            'NET_RATING_GAME': 6.0,
            'PTS': 108,
            'OPP_PTS': 102,
            'WIN': 1
        }
    ]
    df = pd.DataFrame(data)
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    return df


def _sample_locations() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {'TEAM': 'ATL', 'LATITUDE': 33.753746, 'LONGITUDE': -84.38633, 'TIMEZONE': 'Eastern'},
            {'TEAM': 'LAL', 'LATITUDE': 34.0430175, 'LONGITUDE': -118.267254, 'TIMEZONE': 'Pacific'}
        ]
    )


def test_compute_travel_features_timezone_handling():
    games = _sample_games()
    locations = _sample_locations()

    result = compute_travel_features(games, locations, travel_lookback=2, tz_lookback=2)
    atl_tz_change = (
        result[result['TEAM_ID'] == 100]
        .sort_values('GAME_DATE')['TZ_CHANGE']
        .tolist()
    )

    assert atl_tz_change == [0.0, 3.0]
    assert result['DISTANCE_MILES'].max() > 0


def test_feature_engineering_pipeline_creates_scores(tmp_path):
    games = _sample_games()
    locations = _sample_locations()
    params = {
        'window_N': 2,
        'games_lookback': 1,
        'games_lookahead': 0,
        'travel_lookback': 2,
        'tz_lookback': 2
    }

    fe = FeatureEngineering(cache_dir=tmp_path)
    result = fe.compute_features(games, locations, params)
    required_cols = {'LOAD_SCORE', 'TRAVEL_SCORE', 'STRENGTH_DIFF', 'LOAD_DIFF', 'TRAVEL_DIFF'}
    assert required_cols.issubset(result.columns)
    assert not result[['LOAD_SCORE', 'TRAVEL_SCORE']].isna().any().any()
