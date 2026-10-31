"""Shared inputs for the M5 calendar trials: per-contract A2 combo targets and P&L.

Everything is on the contract's own calendar. P&L follows the provisional backtest:
a target known after close t is traded at close t+1 and first earns ret on t+2.
"""

from pathlib import Path

import pandas as pd

from backtest.engine import COST_BPS, contract_pnl
from m2.pipeline import build_m2
from m3.features import build_features

FACTORS = ["tsmom", "ma_crossover", "donchian"]
DEV_END = "2014-12-31"
DELAY = 1
LEAD = DELAY + 1  # weekdays from a decision close to the first P&L day it affects


def a2_combo(d: pd.DataFrame, code: str) -> pd.Series:
    m2 = build_m2(pd.DataFrame({"date": d.index, "contract": code,
                                "close": d.close_adj.to_numpy(), "return": d.ret.to_numpy()}))
    a2 = m2.pivot(index="date", columns="factor", values="approach2_target_close_t")
    return a2[FACTORS].mean(axis=1, skipna=False).reindex(d.index)


def load_contract(path: Path, code: str, asset: str) -> dict:
    d = pd.read_csv(path, index_col="date", parse_dates=True)
    target = a2_combo(d, code)
    net = contract_pnl(target, d.ret, d.roll if "roll" in d else None, COST_BPS.get(asset, 0.0), DELAY)
    gross = contract_pnl(target, d.ret, None, 0.0, DELAY)
    live = target.shift(1 + DELAY).notna()  # P&L days on which a position exists
    return {"data": d, "asset": asset, "target": target, "net": net.net, "gross": gross.gross,
            "position": gross.position, "live": live,
            "logvol20": build_features(d.ret, ("logvol20",)).logvol20}


def load_all(data_dir: Path, universe: pd.DataFrame | None = None) -> dict:
    uni = pd.read_csv(data_dir / "universe.csv") if universe is None else universe
    out = {}
    for r in uni.itertuples():
        dd = getattr(r, "data_dir", None)
        d_dir = Path(dd) if isinstance(dd, str) and dd else data_dir
        out[r.code] = load_contract(d_dir / f"{r.code}.csv", r.code, r.asset_class)
    return out
