"""Provisional backtest engine (stand-in until M1's framework lands).

Conventions
-----------
- A target exposure w_t is known after the close of day t (M2/M3 convention).
- `delay` = closes between the signal close and the execution close.
  delay=1 (default): trade at the close of t+1, so w_t first earns ret_{t+2}.
  delay=0: trade at the signal close itself, w_t earns ret_{t+1}.
- Each contract runs on its own trading calendar. On days a market is closed its
  P&L is 0 and the position is carried.
- A missing (NaN) target means flat.
- Exposure units: fraction of capital in notional per contract. P&L = position x ret,
  where `ret` is data/final's held-contract daily return (roll jumps removed).
- Costs (provisional, replace with M1's model): `cost_bps` per side on |change in
  position|, plus a full round trip on the held position on each roll day.
"""

import numpy as np
import pandas as pd

# per-side cost in bps of notional, by asset class (placeholder until M1's cost model)
COST_BPS = {"equity": 1.0, "rates": 0.5, "fx": 0.5, "energy": 2.0, "metal": 1.5, "agri": 2.0}


def contract_pnl(target: pd.Series, ret: pd.Series, roll: pd.Series | None = None,
                 cost_bps: float = 0.0, delay: int = 1) -> pd.DataFrame:
    """Daily P&L of one contract on its own calendar (the index of `ret`)."""
    if delay < 0:
        raise ValueError("delay must be >= 0")
    if not ret.index.is_monotonic_increasing or ret.index.has_duplicates:
        raise ValueError("ret index must be sorted unique dates")
    pos = target.reindex(ret.index).shift(1 + delay).fillna(0.0)
    gross = pos * ret.fillna(0.0)
    turnover = pos.diff().abs().fillna(pos.abs())
    roll_trade = 2 * pos.abs() * (roll.reindex(ret.index).fillna(False).astype(bool)
                                   if roll is not None else 0.0)
    cost = (turnover + roll_trade) * cost_bps / 1e4
    return pd.DataFrame({"position": pos, "gross": gross, "cost": cost,
                         "net": gross - cost, "turnover": turnover})


def portfolio(pnl: dict[str, pd.DataFrame], universe: pd.DataFrame,
              min_contracts: int = 1) -> pd.DataFrame:
    """Equal-weight portfolio across contracts on the union calendar.

    universe: bool DataFrame (dates x contracts), True from the day a contract is in
    the book. On each day the weight is 1 / (number of contracts in the book), so a
    market holiday does not change the other contracts' weights.
    """
    dates = universe.index
    cols = list(universe.columns)
    inb = universe.reindex(columns=cols).astype(bool)
    n = inb.sum(axis=1)
    out = {}
    for field in ("gross", "cost", "net", "turnover"):
        wide = pd.DataFrame({c: pnl[c][field] for c in cols}).reindex(dates).fillna(0.0)
        out[field] = wide.where(inb, 0.0).sum(axis=1) / n.where(n > 0)
    gross_exp = pd.DataFrame({c: pnl[c]["position"].abs() for c in cols}).reindex(dates).ffill()
    out["gross_exposure"] = gross_exp.where(inb, 0.0).sum(axis=1) / n.where(n > 0)
    out["n_contracts"] = n
    res = pd.DataFrame(out)
    return res[res.n_contracts >= min_contracts]


def in_book(valid: pd.DataFrame) -> pd.DataFrame:
    """A contract joins the book on its first valid target and stays in (holidays included)."""
    return valid.fillna(False).astype(bool).cummax()


__all__ = ["COST_BPS", "contract_pnl", "portfolio", "in_book"]
