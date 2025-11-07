import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

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


def build_model_variants(df: pd.DataFrame) -> Dict[str, ScheduleGLM]:

    variants = {}
    
    # basic model
    variants['basic'] = ScheduleGLM(
        'WIN ~ HOME + STRENGTH_DIFF + C(SEASON) + LOAD_DIFF_Z + TRAVEL_DIFF_Z'
    )
    
    # with itneraction
    variants['interaction'] = ScheduleGLM(
        'WIN ~ HOME + STRENGTH_DIFF + C(SEASON) + LOAD_DIFF_Z * TRAVEL_DIFF_Z'
    )
    
    # travel components instead of composite
    if 'DISTANCE_IMPACT' in df.columns:
        variants['components'] = ScheduleGLM(
            'WIN ~ HOME + STRENGTH_DIFF + C(SEASON) + LOAD_DIFF_Z + '
            'DISTANCE_IMPACT + TZ_CHANGE + RECENT_TRAVEL'
        )
    
    # threshold effects
    if 'EXTREME_TRAVEL' in df.columns:
        variants['threshold'] = ScheduleGLM(
            'WIN ~ HOME + STRENGTH_DIFF + C(SEASON) + LOAD_DIFF_Z + '
            'TRAVEL_DIFF_Z + EXTREME_TRAVEL + EXTREME_LOAD'
        )
    
    # w/ b2b-travel interaction
    if 'B2B_TRAVEL_INTERACTION' in df.columns:
        variants['b2b_travel'] = ScheduleGLM(
            'WIN ~ HOME + STRENGTH_DIFF + C(SEASON) + LOAD_DIFF_Z + '
            'TRAVEL_DIFF_Z + B2B_TRAVEL_INTERACTION'
        )
    
    return variants



def time_series_split(df: pd.DataFrame, 
                      n_splits: int = 3,
                      val_size: float = 0.15,
                      test_size: float = 0.15) -> list:

    df = df.sort_values('GAME_DATE')
    n = len(df)
    
    splits = []
    
    for i in range(n_splits):

        train_end = int(n * (0.4 + i * 0.15)) # progressive icnreasing
        val_end = train_end + int(n * val_size)
        test_end = min(val_end + int(n * test_size), n)
        
        train_idx = df.index[:train_end]
        val_idx = df.index[train_end:val_end]
        test_idx = df.index[val_end:test_end]
        
        splits.append({
            'train': train_idx,
            'val': val_idx,
            'test': test_idx,
            'train_seasons': df.loc[train_idx, 'SEASON'].unique(),
            'val_seasons': df.loc[val_idx, 'SEASON'].unique(),
            'test_seasons': df.loc[test_idx, 'SEASON'].unique()
        })
    
    return splits


def compare_models(df: pd.DataFrame, n_splits: int = 4, val_size: float = 0.15, 
                                                        test_size: float = 0.15, 
                                                        val_set: str = 'val') -> pd.DataFrame:

    assert val_set in {'val', 'test'}

    variants = build_model_variants(df)
    splits = time_series_split(df, n_splits=n_splits, val_size=val_size, test_size=test_size)

    rows = []
    for name, model in variants.items():
        log_losses = []
        aucs = []
        briers = []
        accs = []

        for split in splits:
            train_idx = split['train']
            val_idx = split[val_set]

            # Skip degenerate windows
            if len(val_idx) < 5 or len(train_idx) < 20:
                continue

            train_df = df.loc[train_idx]
            eval_df = df.loc[val_idx]

            try:
                model.fit(train_df)
                metrics = model.evaluate(eval_df)
                log_losses.append(metrics['log_loss'])
                aucs.append(metrics['roc_auc'])
                briers.append(metrics['brier_score'])
                accs.append(metrics['accuracy'])
            except Exception as e:
                logger.warning(f"Time-series val failed for {name}: {e}")
                continue

        if log_losses:
            rows.append({
                'model': name,
                'splits_used': len(log_losses),
                'log_loss_mean': float(np.mean(log_losses)),
                'log_loss_std': float(np.std(log_losses)),
                'roc_auc_mean': float(np.mean(aucs)) if aucs else None,
                'brier_mean': float(np.mean(briers)) if briers else None,
                'accuracy_mean': float(np.mean(accs)) if accs else None,
            })

    result = pd.DataFrame(rows).sort_values('log_loss_mean') if rows else pd.DataFrame()
    return result

