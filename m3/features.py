"""HMM observation features for one contract, on that contract's own calendar.

Every value at date t uses information available by the close of t only.

In-market (from the contract's `ret`):
    logvolN : log annualised realised volatility over N sessions
    erN     : logit of the efficiency ratio |sum r| / sum |r| over N sessions
              (1 = straight-line trend, 0 = pure back-and-forth)

Off-market (shared by all contracts), written "x:<column>":
    a column of an exogenous table whose row dated d holds what was known on d
    (publication lags already applied, e.g. data/processed/offmarket/offmarket_daily.csv).
    It is carried forward to the contract's dates, never backward.
"""

import re

import numpy as np
import pandas as pd

DEFAULT_FEATURES = ("logvol20", "er20", "er60")
OFFMARKET_FEATURES = ("x:vix", "x:term", "x:epu", "x:gpr")
ER_CLIP = 1e-3
EXOG_FFILL_LIMIT = 5  # business days a stale exogenous value may be carried


def _in_market(ret: pd.Series, name: str) -> pd.Series:
    m = re.fullmatch(r"(logvol|er)(\d+)", name)
    if not m:
        raise ValueError(f"unknown feature {name!r}")
    kind, n = m.group(1), int(m.group(2))
    if kind == "logvol":
        vol = ret.rolling(n, min_periods=n).std(ddof=1) * np.sqrt(252)
        return np.log(vol.where(vol > 0)).rename(name)
    er = ret.rolling(n, min_periods=n).sum().abs() / ret.abs().rolling(n, min_periods=n).sum()
    er = er.clip(ER_CLIP, 1 - ER_CLIP)
    return np.log(er / (1 - er)).rename(name)


def _off_market(index: pd.DatetimeIndex, exog: pd.DataFrame | None, name: str) -> pd.Series:
    col = name[2:]
    if exog is None or col not in exog.columns:
        raise ValueError(f"feature {name!r} needs column {col!r} in the exogenous table")
    s = exog[col].sort_index()
    s = s.reindex(s.index.union(index)).ffill(limit=EXOG_FFILL_LIMIT).reindex(index)
    return s.rename(name)


def feature(ret: pd.Series, name: str, exog: pd.DataFrame | None = None) -> pd.Series:
    if name.startswith("x:"):
        return _off_market(ret.index, exog, name)
    return _in_market(ret, name)


def build_features(ret: pd.Series, names=DEFAULT_FEATURES, exog: pd.DataFrame | None = None) -> pd.DataFrame:
    if not isinstance(ret.index, pd.DatetimeIndex) or not ret.index.is_monotonic_increasing:
        raise ValueError("ret must have a sorted DatetimeIndex")
    return pd.concat([feature(ret.astype(float), n, exog) for n in names], axis=1)


def load_exog(path) -> pd.DataFrame:
    return pd.read_csv(path, index_col="date", parse_dates=True).sort_index()
