"""Run approaches 1/2/3 for every M2 factor through the provisional engine.

    python -m backtest.run                                  # development window only
    python -m backtest.run --m3 hmm_k3_quarter_top --delay 0
    python -m backtest.run --filter-file m5_outputs/some_filter.csv   # any filter file
    python -m backtest.run --include-test                   # also report the hold-out

The hold-out window (after --dev-end) is reported only with --include-test. Freeze
M2/M3 settings before using it.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from m2.pipeline import build_m2

from .engine import COST_BPS, contract_pnl, in_book, portfolio
from .metrics import paired, perf, vol_matched

FACTORS = ("tsmom", "ma_crossover", "donchian", "combo")


def load_targets(data_dir: Path, m3_file: Path, universe: pd.DataFrame | None = None) -> tuple[dict, pd.DataFrame]:
    """Per contract: daily data + targets for A1/A2/A3 of every factor (own calendar).
    `universe` (code, asset_class) replaces data_dir/universe.csv, e.g. to run a subset."""
    uni = pd.read_csv(data_dir / "universe.csv") if universe is None else universe
    filt = pd.read_csv(m3_file, parse_dates=["date"]).set_index(["contract", "date"])["filter_close_t"]
    contracts = {}
    for code, asset in zip(uni.code, uni.asset_class):
        d = pd.read_csv(data_dir / f"{code}.csv", index_col="date", parse_dates=True)
        m2 = build_m2(pd.DataFrame({"date": d.index, "contract": code,
                                    "close": d.close_adj.to_numpy(), "return": d.ret.to_numpy()}))
        a1 = m2.pivot(index="date", columns="factor", values="approach1_target_close_t")
        a2 = m2.pivot(index="date", columns="factor", values="approach2_target_close_t")
        a1["combo"] = a1[list(FACTORS[:3])].mean(axis=1, skipna=False)
        a2["combo"] = a2[list(FACTORS[:3])].mean(axis=1, skipna=False)
        f = filt.loc[code].reindex(d.index)
        a3 = a2.mul(f, axis=0)
        contracts[code] = {"data": d, "asset": asset, "A1": a1, "A2": a2, "A3": a3}
    return contracts, uni


def run(data_dir: Path, m3_file: Path, delay: int, costs: bool, dev_end: str,
        include_test: bool, min_contracts: int, out_dir: Path | None):
    contracts, uni = load_targets(data_dir, m3_file)
    union = sorted(set().union(*(c["data"].index for c in contracts.values())))
    # same book for every approach: a contract enters once its A3 (combo) target exists
    valid = pd.DataFrame({k: c["A3"]["combo"].notna() for k, c in contracts.items()}).reindex(union)
    book = in_book(valid)

    windows = {"dev": (None, dev_end)}
    if include_test:
        windows["test"] = (pd.Timestamp(dev_end) + pd.Timedelta(days=1), None)

    rows, pairs, daily = [], [], {}
    for factor in FACTORS:
        port = {}
        for ap in ("A1", "A2", "A3"):
            pnl = {k: contract_pnl(c[ap][factor], c["data"].ret, c["data"].roll,
                                   COST_BPS[c["asset"]] if costs else 0.0, delay)
                   for k, c in contracts.items()}
            port[ap] = portfolio(pnl, book, min_contracts)
            daily[f"{factor}_{ap}"] = port[ap].net
        for wname, (lo, hi) in windows.items():
            sl = {ap: p.loc[lo:hi] for ap, p in port.items()}
            for ap, p in sl.items():
                s = perf(p.net)
                rows.append({"window": wname, "factor": factor, "approach": ap, **s,
                             "sharpe_gross": perf(p.gross)["sharpe"],
                             "turnover_pa": p.turnover.mean() * 252, "cost_pa": p.cost.mean() * 252,
                             "gross_exp": p.gross_exposure.mean(),
                             "n_min": int(p.n_contracts.min()), "n_max": int(p.n_contracts.max())})
            a3m = vol_matched(sl["A3"].net, sl["A2"].net)
            pairs.append({"window": wname, "factor": factor, **paired(sl["A3"].net, sl["A2"].net),
                          "A3_volmatched_max_dd": perf(a3m)["max_dd"],
                          "A3_volmatched_calmar": perf(a3m)["calmar"]})

    res, prs = pd.DataFrame(rows), pd.DataFrame(pairs)
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        for wname in windows:
            r = res[res.window == wname]
            print(f"\n=== {wname} window {r.start.min()} .. {r.end.max()}  "
                  f"(contracts in book {r.n_min.min()}-{r.n_max.max()}, delay={delay}, costs={'on' if costs else 'off'})")
            print(r[["factor", "approach", "ann_ret", "ann_vol", "sharpe", "sharpe_gross", "max_dd",
                     "calmar", "turnover_pa", "cost_pa", "gross_exp"]].round(3).to_string(index=False))
            print(f"\n--- A3 vs A2 ({wname}): paired daily difference")
            print(prs[prs.window == wname].drop(columns="window").round(3).to_string(index=False))
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        res.to_csv(out_dir / "summary.csv", index=False)
        prs.to_csv(out_dir / "a3_vs_a2.csv", index=False)
        dev = pd.DataFrame(daily).loc[:dev_end] if not include_test else pd.DataFrame(daily)
        dev.rename_axis("date").to_csv(out_dir / "daily_net_returns.csv")
    return res, prs


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data/final"))
    p.add_argument("--m3", default="hmm_k3_quarter_top", help="stem of an m3_outputs/*_filter.csv")
    p.add_argument("--filter-file", type=Path, default=None,
                   help="path to any filter CSV (date, contract, filter_close_t); overrides --m3")
    p.add_argument("--delay", type=int, default=1)
    p.add_argument("--no-costs", action="store_true")
    p.add_argument("--dev-end", default="2014-12-31")
    p.add_argument("--include-test", action="store_true")
    p.add_argument("--min-contracts", type=int, default=10)
    p.add_argument("--out-dir", type=Path, default=Path("backtest_outputs"))
    a = p.parse_args()
    filter_file = a.filter_file if a.filter_file is not None else Path("m3_outputs") / f"{a.m3}_filter.csv"
    run(a.data_dir, filter_file, a.delay, not a.no_costs,
        a.dev_end, a.include_test, a.min_contracts, a.out_dir)
