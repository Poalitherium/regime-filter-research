"""M2 daily trend-following signal library. Signals observed after close t."""
import numpy as np
import pandas as pd


def _validate(close):
    if not isinstance(close, pd.Series):
        raise TypeError('close must be a pandas Series indexed by trading date')
    if not isinstance(close.index, pd.DatetimeIndex):
        raise TypeError('close index must be DatetimeIndex')
    if not close.index.is_monotonic_increasing or close.index.has_duplicates:
        raise ValueError('dates must be strictly increasing and unique')
    if not pd.api.types.is_numeric_dtype(close):
        raise TypeError('close must be numeric')
    if not np.isfinite(close.dropna().to_numpy()).all():
        raise ValueError('close contains infinite values')
    return close.astype(float)


def tsmom(close, lookback=60):
    """Directional signal from L-day price change; NaN until full history."""
    close = _validate(close)
    if not isinstance(lookback, int) or lookback < 1:
        raise ValueError('lookback must be positive integer')
    delta = close - close.shift(lookback)
    return np.sign(delta).where(delta.notna()).rename('tsmom')


def ma_crossover(close, short=20, long=100):
    """Short vs long simple moving averages; NaN during warm-up."""
    close = _validate(close)
    if not all(isinstance(x, int) for x in (short, long)) or not 1 <= short < long:
        raise ValueError('require integer windows 1 <= short < long')
    short_ma = close.rolling(short, min_periods=short).mean()
    long_ma = close.rolling(long, min_periods=long).mean()
    spread = short_ma - long_ma
    return np.sign(spread).where(spread.notna()).rename('ma_crossover')


def donchian_breakout(close, lookback=55):
    """Break prior L closes; otherwise carry prior signal, initially flat."""
    close = _validate(close)
    if not isinstance(lookback, int) or lookback < 1:
        raise ValueError('lookback must be positive integer')
    high = close.shift(1).rolling(lookback, min_periods=lookback).max()
    low = close.shift(1).rolling(lookback, min_periods=lookback).min()
    valid = close.notna() & high.notna() & low.notna()
    up = valid & (close > high)
    down = valid & (close < low)
    out = pd.Series(np.nan, index=close.index, name='donchian')
    state = 0.0
    for i in range(len(close)):
        if not valid.iloc[i]:
            state = 0.0
            continue
        if up.iloc[i]:
            state = 1.0
        elif down.iloc[i]:
            state = -1.0
        out.iloc[i] = state
    return out


def generate_signals(close):
    """Generate all three close-of-day signals for one contract."""
    close = _validate(close)
    return pd.concat([tsmom(close), ma_crossover(close), donchian_breakout(close)], axis=1)
