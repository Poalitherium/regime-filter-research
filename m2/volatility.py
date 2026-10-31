"""M2 volatility-targeting overlay. Outputs target exposure, not futures lots.
All values at index t are computed using information available by close t.
Execution must occur later, according to the M1 backtest convention.
"""
import numpy as np
import pandas as pd


def volatility_target(signals: pd.DataFrame, returns: pd.Series,
                      window: int = 60, target_vol: float = 0.15,
                      annualization: int = 252, max_scale: float = 2.0,
                      vol_floor: float = 0.01) -> pd.DataFrame:
    """Calculate annualized realized vol and volatility-scaled signal exposures.

    returns must be a valid, *tradable* daily return series from M1. For futures,
    do not blindly use pct_change of back-adjusted continuous prices.
    Undefined vol and missing signals remain NaN; do not silently trade them.
    """
    if not isinstance(signals, pd.DataFrame) or signals.empty:
        raise TypeError('signals must be a nonempty DataFrame')
    if not isinstance(returns, pd.Series):
        raise TypeError('returns must be a pandas Series')
    if not signals.index.equals(returns.index):
        raise ValueError('signals and returns must have identical indices')
    if not isinstance(signals.index, pd.DatetimeIndex) or signals.index.has_duplicates or not signals.index.is_monotonic_increasing:
        raise ValueError('index must be sorted unique trading dates')
    if not isinstance(window, int) or window < 2 or not isinstance(annualization, int) or annualization < 1:
        raise ValueError('invalid rolling window or annualization')
    if not (np.isfinite(target_vol) and target_vol > 0 and np.isfinite(max_scale) and max_scale > 0 and np.isfinite(vol_floor) and vol_floor > 0):
        raise ValueError('target_vol, max_scale, and vol_floor must be positive finite')
    if not np.isfinite(returns.dropna().to_numpy(dtype=float)).all():
        raise ValueError('returns contain infinity')
    if not np.isfinite(signals.dropna().to_numpy(dtype=float)).all():
        raise ValueError('signals contain infinity')
    if not signals.isin([-1, 0, 1]).where(signals.notna(), True).all().all():
        raise ValueError('signals must be -1, 0, +1, or NaN')
    vol = returns.astype(float).rolling(window, min_periods=window).std(ddof=1) * np.sqrt(annualization)
    scale = (target_vol / vol.clip(lower=vol_floor)).clip(upper=max_scale)
    scale = scale.where(vol.notna())
    result = pd.DataFrame({'realized_vol': vol, 'vol_scale': scale}, index=signals.index)
    for col in signals.columns:
        result[f'{col}_raw'] = signals[col].astype(float)
        result[f'{col}_vol_target'] = signals[col].astype(float) * scale
    return result
