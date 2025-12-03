import math
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from typing import Dict, Optional, Iterable
import logging

logger = logging.getLogger(__name__)


MODEL_VARIANT_SPECS = {
    'basic': {
        'formula': (
            'WIN ~ HOME + STRENGTH_DIFF + '
            'LOAD_DIFF_Z + TRAVEL_DIFF_Z'
        ),
        'required_columns': set()
    },
    'interaction': {
        'formula': (
            'WIN ~ HOME + STRENGTH_DIFF + '
            'LOAD_DIFF_Z * TRAVEL_DIFF_Z'
        ),
        'required_columns': set()
    },
    'home_opponent_travel': {
        'formula': (
            'WIN ~ HOME + STRENGTH_DIFF + LOAD_DIFF_Z + '
            'TRAVEL_DIFF_Z + HOME:I(TRAVEL_DIFF_Z > 0)'
        ),
        'required_columns': set()
    },
    'threshold': {
        'formula': (
            'WIN ~ HOME + STRENGTH_DIFF + LOAD_DIFF_Z + '
            'TRAVEL_DIFF_Z + EXTREME_TRAVEL + EXTREME_LOAD'
        ),
        'required_columns': {'EXTREME_TRAVEL', 'EXTREME_LOAD'}
    },
    'b2b_travel': {
        'formula': (
            'WIN ~ HOME + STRENGTH_DIFF + LOAD_DIFF_Z + '
            'TRAVEL_DIFF_Z + B2B_TRAVEL_INTERACTION'
        ),
        'required_columns': {'B2B_TRAVEL_INTERACTION'}
    }
}

class ScheduleGLM:

    def __init__(self, formula: Optional[str] = None):

        # Initialize with model formula
        if formula is None:
            self.formula = (
                'WIN ~ HOME + STRENGTH_DIFF + C(SEASON) + '
                'LOAD_DIFF_Z + TRAVEL_DIFF_Z'
            )
        else:
            self.formula = formula
            
        self.model = None
        self.fit_result = None
        
    def fit(self, df: pd.DataFrame) -> 'ScheduleGLM':
        logger.info(f"Fitting GLM with formula: {self.formula}")
        
        model_cols = ['WIN', 'HOME', 'STRENGTH_DIFF', 'SEASON', 'LOAD_DIFF_Z', 'TRAVEL_DIFF_Z']
        df_clean = df.dropna(subset=model_cols)
    
        logger.info(f"Training on {len(df_clean)} games")

        self.model = smf.glm(
            formula=self.formula,
            data=df_clean,
            family=sm.families.Binomial()
        )
        self.fit_result = self.model.fit()
        
        logger.info(f"Model AIC: {self.fit_result.aic:.2f}")
        self._log_significant_coefficients()
        
        return self
    
    def _log_significant_coefficients(self):

        summary = self.fit_result.summary2().tables[1]
        significant = summary[summary['P>|z|'] < 0.05]
        
        for idx, row in significant.iterrows():
            logger.info(f"{idx}: {row['Coef.']:.4f} (p={row['P>|z|']:.4f})")
    
    def predict_probability(self, df: pd.DataFrame) -> np.ndarray:
        if self.fit_result is None:
            raise ValueError("Model must be fitted before predicting")
        return self.fit_result.predict(df)
    
    def compute_counterfactuals(self, df: pd.DataFrame) -> pd.DataFrame:

        if self.fit_result is None:
            raise ValueError("Model must be fitted first")
        
        df = df.copy()
        
        # Factual predictions (actual schedule)
        df['P_FACTUAL'] = self.predict_probability(df)
        
        # neutralize schedule effects
        df_neutral = df.copy()
        df_neutral['LOAD_DIFF_Z'] = 0
        df_neutral['TRAVEL_DIFF_Z'] = 0
        
        if 'LOAD_TRAVEL_INTERACTION' in df.columns:
            df_neutral['LOAD_TRAVEL_INTERACTION'] = 0
        if 'B2B_TRAVEL_INTERACTION' in df.columns:
            df_neutral['B2B_TRAVEL_INTERACTION'] = 0
        
        df['P_NEUTRAL'] = self.predict_probability(df_neutral)
        df['SCHEDULE_IMPACT'] = df['P_FACTUAL'] - df['P_NEUTRAL']
        
        return df
    
    def evaluate(self, df: pd.DataFrame, y_true_col: str = 'WIN') -> Dict:
 
        from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss, accuracy_score
        
        y_true = df[y_true_col]
        y_pred = self.predict_probability(df)
        
        metrics = {
            'log_loss': log_loss(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_pred),
            'brier_score': brier_score_loss(y_true, y_pred),
            'accuracy': accuracy_score(y_true, (y_pred > 0.5).astype(int)),
            'aic': self.fit_result.aic,
            'bic': self.fit_result.bic
        }
        
        return metrics
    
    def get_coefficients(self) -> pd.DataFrame:
        if self.fit_result is None:
            raise ValueError("Model must be fitted first")
        
        summary = self.fit_result.summary2().tables[1]
        summary['significant'] = summary['P>|z|'] < 0.05
        
        return summary


def get_model_variant_formula(name: str) -> str:
    spec = MODEL_VARIANT_SPECS.get(name)
    if spec is None:
        available = ", ".join(sorted(MODEL_VARIANT_SPECS.keys()))
        raise ValueError(f"Unknown model variant '{name}'. Available: {available}")
    return spec['formula']


def get_variant_required_columns(name: str) -> set:
    spec = MODEL_VARIANT_SPECS.get(name)
    if spec is None:
        available = ", ".join(sorted(MODEL_VARIANT_SPECS.keys()))
        raise ValueError(f"Unknown model variant '{name}'. Available: {available}")
    return set(spec.get('required_columns', set()))


def is_variant_supported(name: str, columns: Iterable[str]) -> bool:
    spec = MODEL_VARIANT_SPECS.get(name)
    if spec is None:
        return False
    required = spec.get('required_columns', set())
    return required.issubset(set(columns))


def build_model_variants(df: pd.DataFrame) -> Dict[str, ScheduleGLM]:

    variants = {}
    available_columns = set(df.columns)
    
    for name, spec in MODEL_VARIANT_SPECS.items():
        required_cols = spec.get('required_columns', set())
        if required_cols.issubset(available_columns):
            variants[name] = ScheduleGLM(spec['formula'])
        else:
            missing = ", ".join(sorted(required_cols - available_columns))
            logger.debug(
                "Skipping model variant %s because missing columns: %s",
                name,
                missing
            )
    
    return variants



def time_series_split(df: pd.DataFrame, 
                      n_splits: int = 3,
                      val_size: float = 0.15,
                      test_size: float = 0.15) -> list:

    df = df.sort_values('GAME_DATE')
    n = len(df)
    if n == 0:
        return []
    season_order = (
        df[['SEASON', 'GAME_DATE']]
        .drop_duplicates('SEASON')
        .sort_values('GAME_DATE')['SEASON']
        .tolist()
    )
    n_seasons = len(season_order)
    season_lengths = (
        df.groupby('SEASON')
          .size()
          .reindex(season_order)
          .astype(int)
    )
    cumulative_lengths = season_lengths.cumsum().tolist()

    def seasons_for_rows(target_rows: int) -> int:
        if target_rows <= 0:
            return 0
        for idx, cumulative in enumerate(cumulative_lengths, start=1):
            if cumulative >= target_rows:
                return idx
        return n_seasons

    def interpret_window_size(size: float, label: str) -> int:
        if size <= 0:
            raise ValueError(f"{label} must be positive")
        if size < 1:
            target_rows = max(1, int(math.floor(n * size)))
            return max(1, seasons_for_rows(target_rows))
        return min(n_seasons, int(math.ceil(size)))

    val_season_count = interpret_window_size(val_size, "val_size")
    test_season_count = interpret_window_size(test_size, "test_size")

    if val_season_count + test_season_count >= n_seasons:
        raise ValueError(
            "Not enough seasons to create non-overlapping validation and test windows"
        )

    splits = []
    max_train_end = n_seasons - (val_season_count + test_season_count)
    if max_train_end <= 0:
        logger.warning("Unable to allocate train window given val/test sizes; returning no splits")
        return splits

    last_train_end = 0

    for i in range(n_splits):
        train_ratio = min(0.95, 0.4 + i * 0.15)
        train_target_rows = max(1, int(math.floor(n * train_ratio)))
        train_season_count = seasons_for_rows(train_target_rows)
        train_season_count = min(max_train_end, max(1, train_season_count))

        if train_season_count <= last_train_end:
            if last_train_end >= max_train_end:
                break
            train_season_count = last_train_end + 1

        val_start = train_season_count
        val_end = val_start + val_season_count
        test_end = val_end + test_season_count

        if test_end > n_seasons:
            break

        train_seasons = season_order[:train_season_count]
        val_seasons = season_order[val_start:val_end]
        test_seasons = season_order[val_end:test_end]

        train_idx = df[df['SEASON'].isin(train_seasons)].index
        val_idx = df[df['SEASON'].isin(val_seasons)].index
        test_idx = df[df['SEASON'].isin(test_seasons)].index

        if len(val_idx) == 0 or len(test_idx) == 0:
            break

        splits.append({
            'train': train_idx,
            'val': val_idx,
            'test': test_idx,
            'train_seasons': train_seasons,
            'val_seasons': val_seasons,
            'test_seasons': test_seasons
        })

        logger.info(
            "Split %s: Train %s (%s) -> Val %s (%s) -> Test %s (%s)",
            len(splits) - 1,
            train_seasons, len(train_idx),
            val_seasons, len(val_idx),
            test_seasons, len(test_idx)
        )

        last_train_end = train_season_count

    if len(splits) < n_splits:
        logger.info(
            "Generated %s splits (requested %s) based on available seasons",
            len(splits), n_splits
        )

    return splits



def compare_models(df: pd.DataFrame, n_splits: int = 4, val_size: float = 0.15, 
                                                        test_size: float = 0.15, 
                                                        val_set: str = 'val') -> pd.DataFrame:

    assert val_set in {'val', 'test'}
    df = df.copy()
    if 'SEASON' in df.columns:
        season_categories = sorted(df['SEASON'].unique())
        if not pd.api.types.is_categorical_dtype(df['SEASON']):
            df['SEASON'] = pd.Categorical(df['SEASON'], categories=season_categories)
        else:
            missing = set(df['SEASON'].unique()) - set(df['SEASON'].cat.categories)
            if missing:
                df['SEASON'] = df['SEASON'].cat.add_categories(sorted(missing))

    variants = build_model_variants(df)
    splits = time_series_split(df, n_splits=n_splits, val_size=val_size, test_size=test_size)

    rows = []
    for name, model in variants.items():
        log_losses = []
        aucs = []
        briers = []
        accs = []
        aics = []

        for split in splits:
            train_idx = split['train']
            val_idx = split[val_set]

            # Skip degenerate windows
            if len(val_idx) < 5 or len(train_idx) < 20:
                continue
            
            train_df = df.loc[split['train']]
            eval_df = df.loc[split[val_set]]

            try:
                model.fit(train_df)
                metrics = model.evaluate(eval_df)
                log_losses.append(metrics['log_loss'])
                aucs.append(metrics['roc_auc'])
                briers.append(metrics['brier_score'])
                accs.append(metrics['accuracy'])
                aics.append(metrics['aic'])
            except Exception as e:
                logger.warning(f"Time-series val failed for {name}: {e}")
                continue

        if log_losses:
            travel_significant = None
            try:
                if 'TRAVEL_DIFF_Z' in df.columns:
                    model.fit(df)
                    coeffs = model.get_coefficients()
                    if 'TRAVEL_DIFF_Z' in coeffs.index:
                        travel_significant = bool(coeffs.loc['TRAVEL_DIFF_Z', 'significant'])
            except Exception as e:
                logger.warning(f"Failed to assess travel significance for {name}: {e}")

            rows.append({
                'model': name,
                'splits_used': len(log_losses),
                'log_loss_mean': float(np.mean(log_losses)),
                'log_loss_std': float(np.std(log_losses)),
                'roc_auc_mean': float(np.mean(aucs)) if aucs else None,
                'brier_mean': float(np.mean(briers)) if briers else None,
                'accuracy_mean': float(np.mean(accs)) if accs else None,
                'aic_mean': float(np.mean(aics)) if aics else None,
                'travel_significant': travel_significant
            })

    result = pd.DataFrame(rows).sort_values('log_loss_mean') if rows else pd.DataFrame()
    return result
