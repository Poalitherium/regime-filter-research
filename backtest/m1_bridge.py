"""Run the project's comparisons through M1's backtest engine (m1/backtest.py).

    python -m backtest.m1_bridge --baseline --out-dir backtest_outputs/m1_baseline  # M2: A1/A2 per factor
    python -m backtest.m1_bridge --all --out-dir backtest_outputs/m1_rerun          # all 8 M3 trials
    python -m backtest.m1_bridge --m5 --out-dir backtest_outputs/m1_m5              # the 4 M5 trials
    python -m backtest.m1_bridge --m3 bocpd_h250 --out-dir backtest_outputs/m1_bocpd_h250

Every trial is a treatment vs a control on the same contracts: A3 vs A2 (combo factor)
for the filter trials, the dynamic signal vs static 12-month TSMOM for trial 5, and the
pure HMM strategy vs static 12-month TSMOM for trial 8. Each pair runs through M1's
`contract_pnl` and fixed-membership `portfolio` (M1's accounting: execution at the next
close, a sample that starts flat, costs on the execution day, roll turnover
|before| + |after|, fixed 1/13 weights) under two cost models:

- research: M1's per-side bps allowance by asset class (equity 1, rates 0.5, fx 0.5,
  energy 2, metal 1.5, agri 2), exactly what `m1.run_backtest` charges;
- official: the exact per-contract fees of m1/config/fee_schedule.json converted to bps
  of each day's contract notional (backtest/official_costs.py), zero slippage.

Two conventions are bridged so the comparison means what it meant in M3:
- same book: targets are blanked (flat) before a contract enters the trial's book;
- window: it starts on the first day with at least `min_contracts` contracts in that
  book and ends at the development end. There is no option to go past 2014-12-31.

The paired tests are backtest.metrics.paired (circular block bootstrap of the Sharpe
difference, 20-day blocks, 5,000 resamples, seed 0), as in M3.
"""

import argparse
from pathlib import Path

import pandas as pd

from m1.backtest import COST_BPS as M1_RESEARCH_BPS
from m1.backtest import contract_pnl as m1_contract_pnl
from m1.backtest import portfolio as m1_portfolio

from m2.pipeline import build_m2

from .engine import in_book
from .metrics import paired, perf
from .official_costs import fee_bps
from .run import load_targets

DEV_END = "2014-12-31"
COST_MODELS = ("research", "official")
FACTORS = ("tsmom", "ma_crossover", "donchian", "combo")
M5_TRIALS = {  # trial: (name, m5_outputs filter stem); M5-3 and M5-4 failed their gates (A3 = A2)
    "M5-1": ("Class-pooled HMM", "pool_class_k3_quarter_top"),
    "M5-2": ("Global-pooled HMM", "pool_global_k3_quarter_top"),
}
M5_IDENTITY = {"M5-3": "Seasonality (gate failed, A3 = A2)", "M5-4": "Contract expiry window (gate failed, A3 = A2)"}
FILTER_TRIALS = {  # trial: (name, m3_outputs filter stem)
    1: ("HMM (in-market features)", "hmm_k3_quarter_top"),
    2: ("HMM + macro/news", "hmm_k3_quarter_top_offmkt"),
    3: ("BOCPD changepoints", "bocpd_h250"),
    4: ("Portfolio covariance vol target", "portvol"),
    6: ("Meta-labeling (GBM)", "meta_hand_gbm"),
    7: ("Signature meta-labeling", "meta_sig_gbm"),
}


# ---------------------------------------------------------------- trial inputs

def filter_trial(data_dir: Path, filter_file: Path, factor: str = "combo", universe: pd.DataFrame | None = None):
    """A2 (control) and A3 (treatment) targets, with the book of backtest/run.py.
    `universe` (code, asset_class) replaces data_dir/universe.csv, e.g. to run a subset."""
    contracts, uni = load_targets(data_dir, filter_file, universe)
    union = sorted(set().union(*(c["data"].index for c in contracts.values())))
    book = in_book(pd.DataFrame({k: c["A3"]["combo"].notna() for k, c in contracts.items()}).reindex(union))
    data = {k: c["data"] for k, c in contracts.items()}
    asset = {k: c["asset"] for k, c in contracts.items()}
    targets = {"control": {k: c["A2"][factor] for k, c in contracts.items()},
               "treatment": {k: c["A3"][factor] for k, c in contracts.items()}}
    return data, asset, targets, book


def strategy_trial(module: str, data_dir: Path):
    """Trial 5 (dynamic vs static 12m TSMOM) or trial 8 (pure HMM vs static 12m TSMOM)."""
    if module == "turning_points":
        from m3.turning_points import build_targets
        uni, data, _, _, tg, book = build_targets(data_dir)
        targets = {"control": tg["static_slow"], "treatment": tg["dynamic"]}
    elif module == "regime_strategy":
        from m3.regime_strategy import build_targets
        uni, data, tg, _, book = build_targets(data_dir)
        targets = {"control": tg["tsmom_252"], "treatment": tg["hmm"]}
    else:
        raise ValueError(module)
    return data, uni.asset_class.to_dict(), targets, book


def window_start(book: pd.DataFrame, min_contracts: int = 10) -> pd.Timestamp:
    n = book.sum(axis=1)
    if not (n >= min_contracts).any():
        raise ValueError(f"never {min_contracts} contracts in the book")
    return n.index[n >= min_contracts][0]


# ---------------------------------------------------------------- M1 engine

def m1_book(targets: dict, data: dict, asset: dict, book: pd.DataFrame, start, end=DEV_END,
            costs: str = "official", delay: int = 1, cost_multiplier: float = 1.0) -> pd.DataFrame:
    """M1's per-product ledgers and fixed equal-weight portfolio for one set of targets.

    Like m1.run_backtest.run on [start, end]: every product's market data and targets are cut
    to the window, so the sample starts flat. Targets outside the book are blanked first.
    `cost_multiplier` scales the research bps (M1's --fee-multiplier); it is ignored otherwise."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    ledgers, union = {}, set()
    for c, d in data.items():
        d = d.loc[(d.index >= start) & (d.index <= end)]
        union |= set(d.index)
        inb = book[c].reindex(d.index).fillna(False).astype(bool)
        t = targets[c].reindex(d.index).where(inb)
        if costs == "research":
            ledgers[c] = m1_contract_pnl(t, d.ret, d.roll.astype(bool), M1_RESEARCH_BPS[asset[c]] * cost_multiplier, delay)
        elif costs == "official":
            led = m1_contract_pnl(t, d.ret, d.roll.astype(bool), 0.0, delay)
            fee = led.turnover * fee_bps(d.close, c) / 1e4  # turnover is charged on the execution day
            led["fee_cost"], led["cost"], led["net"] = fee, fee + led.slippage_cost, led.gross - fee - led.slippage_cost
            ledgers[c] = led
        elif costs == "none":
            ledgers[c] = m1_contract_pnl(t, d.ret, d.roll.astype(bool), 0.0, delay)
        else:
            raise ValueError(f"unknown cost model {costs!r}")
    membership = pd.DataFrame(True, index=pd.DatetimeIndex(sorted(union)), columns=list(data))
    return m1_portfolio(ledgers, membership, min_contracts=len(data))


def evaluate(ports: dict) -> dict:
    t, c = ports["treatment"], ports["control"]
    pr = paired(t.net, c.net)
    out = {"start": perf(c.net)["start"], "end": perf(c.net)["end"], "days": len(c),
           "sharpe_control": perf(c.net)["sharpe"], "sharpe_treatment": perf(t.net)["sharpe"],
           "cost_pa_control": c.cost.mean() * 252, "cost_pa_treatment": t.cost.mean() * 252,
           "turnover_pa_control": c.turnover.mean() * 252, "turnover_pa_treatment": t.turnover.mean() * 252,
           "gross_exp_treatment": t.gross_exposure.mean()}
    out.update({k: pr[k] for k in ("d_sharpe", "ci_low", "ci_high", "se_boot", "min_gap_5pct", "p_boot", "corr")})
    return out


def run_trial(data, asset, targets, book, min_contracts: int = 10, end: str = DEV_END):
    start = window_start(book, min_contracts)
    rows, daily = [], {}
    for costs in COST_MODELS:
        ports = {k: m1_book(tg, data, asset, book, start, end, costs) for k, tg in targets.items()}
        rows.append({"cost_model": costs, **evaluate(ports)})
        daily.update({f"{costs}_{k}": p.net for k, p in ports.items()})
    return pd.DataFrame(rows), pd.DataFrame(daily)


def run_all(data_dir: Path, out_dir: Path, m3_dir: Path = Path("m3_outputs"),
            provisional: Path = Path("m3/report/tables/trials.csv"), min_contracts: int = 10):
    """All 8 M3 trials on M1's engine under both cost models, next to M3's original results."""
    jobs = [(tid, name, lambda f=f: filter_trial(data_dir, m3_dir / f"{f}_filter.csv"))
            for tid, (name, f) in FILTER_TRIALS.items()]
    jobs += [(5, "Momentum turning points", lambda: strategy_trial("turning_points", data_dir)),
             (8, "Pure HMM regime strategy", lambda: strategy_trial("regime_strategy", data_dir))]
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    orig = pd.read_csv(provisional).set_index("trial") if provisional.is_file() else None
    for tid, name, load in sorted(jobs, key=lambda j: j[0]):
        print(f"trial {tid}: {name}", flush=True)
        res, daily = run_trial(*load(), min_contracts=min_contracts)
        daily.rename_axis("date").to_csv(out_dir / f"trial{tid}_daily_net.csv")
        if orig is not None and tid in orig.index:
            o = orig.loc[tid]
            rows.append({"trial": tid, "name": name, "engine": "provisional (M3 report)", "cost_model": "provisional bps",
                         "start": o.start, "end": o.end, "sharpe_control": o.sharpe_control,
                         "sharpe_treatment": o.sharpe_treat, "d_sharpe": o.d_sharpe, "ci_low": o.ci_low,
                         "ci_high": o.ci_high, "min_gap_5pct": o.min_gap_5pct, "p_boot": o.p_boot, "corr": o["corr"]})
        for r in res.to_dict("records"):
            rows.append({"trial": tid, "name": name, "engine": "m1", **r})
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "trials.csv", index=False)
    with pd.option_context("display.width", 240, "display.max_columns", 30):
        print(table[["trial", "engine", "cost_model", "start", "sharpe_control", "sharpe_treatment", "d_sharpe",
                     "ci_low", "ci_high", "p_boot", "cost_pa_control"]].round(4).to_string(index=False))
    return table


def m2_baseline(data_dir: Path, out_dir: Path, min_contracts: int = 10, end: str = DEV_END):
    """M2's approaches 1 and 2 for every factor on M1's engine (no regime filter).

    Book: a contract enters once its A2 combo target exists (all three factors warmed up), the
    same book for every factor and approach. Reports A2 - A1 as well, but the two differ in risk
    (A1 is unscaled), so that difference is descriptive, not a test of anything."""
    uni = pd.read_csv(data_dir / "universe.csv")
    data, asset = {}, {}
    targets = {f"{ap}_{f}": {} for ap in ("A1", "A2") for f in FACTORS}
    for code, ac in zip(uni.code, uni.asset_class):
        d = pd.read_csv(data_dir / f"{code}.csv", index_col="date", parse_dates=True)
        m2 = build_m2(pd.DataFrame({"date": d.index, "contract": code,
                                    "close": d.close_adj.to_numpy(), "return": d.ret.to_numpy()}))
        for ap, col in (("A1", "approach1_target_close_t"), ("A2", "approach2_target_close_t")):
            w = m2.pivot(index="date", columns="factor", values=col)
            w["combo"] = w[list(FACTORS[:3])].mean(axis=1, skipna=False)
            for f in FACTORS:
                targets[f"{ap}_{f}"][code] = w[f].reindex(d.index)
        data[code], asset[code] = d, ac
    union = sorted(set().union(*(d.index for d in data.values())))
    book = in_book(pd.DataFrame({c: targets["A2_combo"][c].notna() for c in data}).reindex(union))
    start = window_start(book, min_contracts)
    rows, pairs = [], []
    for costs in COST_MODELS:
        ports = {k: m1_book(tg, data, asset, book, start, end, costs) for k, tg in targets.items()}
        for f in FACTORS:
            for ap in ("A1", "A2"):
                p = ports[f"{ap}_{f}"]
                s = perf(p.net)
                rows.append({"cost_model": costs, "factor": f, "approach": ap, "start": s["start"], "end": s["end"],
                             "ann_ret": s["ann_ret"], "ann_vol": s["ann_vol"], "sharpe": s["sharpe"],
                             "sharpe_gross": perf(p.gross)["sharpe"], "max_dd": s["max_dd"], "calmar": s["calmar"],
                             "turnover_pa": p.turnover.mean() * 252, "cost_pa": p.cost.mean() * 252,
                             "gross_exp": p.gross_exposure.mean()})
            pr = paired(ports[f"A2_{f}"].net, ports[f"A1_{f}"].net)
            pairs.append({"cost_model": costs, "factor": f,
                          **{k: pr[k] for k in ("d_sharpe", "ci_low", "ci_high", "p_boot", "corr")}})
    res, prs = pd.DataFrame(rows), pd.DataFrame(pairs)
    out_dir.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_dir / "baseline.csv", index=False)
    prs.to_csv(out_dir / "a2_vs_a1.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(res.round(4).to_string(index=False))
        print(prs.round(4).to_string(index=False))
    return res, prs


def run_m5(data_dir: Path, out_dir: Path, m5_dir: Path = Path("m5_outputs"),
           provisional: Path = Path("m5/report/tables/trials.csv"), min_contracts: int = 10):
    """M5's four pre-registered trials on M1's engine, with Holm across the four per cost model.
    M5-3 and M5-4 failed their gate tests, so their method is the identity filter: A3 = A2 exactly,
    a difference of 0 and a primary p-value of 1 (M5 PREREG section 3)."""
    from m3.predictability import holm
    orig = pd.read_csv(provisional).set_index("trial") if provisional.is_file() else None
    rows = []
    for tid, (name, stem) in M5_TRIALS.items():
        print(f"{tid}: {name}", flush=True)
        res, daily = run_trial(*filter_trial(data_dir, m5_dir / f"{stem}_filter.csv"), min_contracts=min_contracts)
        out_dir.mkdir(parents=True, exist_ok=True)
        daily.rename_axis("date").to_csv(out_dir / f"{tid}_daily_net.csv")
        if orig is not None and tid in orig.index:
            o = orig.loc[tid]
            rows.append({"trial": tid, "name": name, "engine": "provisional (M5 report)", "cost_model": "provisional bps",
                         "start": o.start, "end": o.end, "sharpe_control": o.sharpe_A2, "sharpe_treatment": o.sharpe_A3,
                         "d_sharpe": o.d_sharpe, "ci_low": o.ci_low, "ci_high": o.ci_high,
                         "min_gap_5pct": o.min_gap_5pct, "p_boot": o.p_boot, "corr": o["corr"]})
        rows += [{"trial": tid, "name": name, "engine": "m1", **r} for r in res.to_dict("records")]
    for tid, name in M5_IDENTITY.items():
        for engine, costs in [("provisional (M5 report)", "provisional bps")] + [("m1", c) for c in COST_MODELS]:
            rows.append({"trial": tid, "name": name, "engine": engine, "cost_model": costs, "d_sharpe": 0.0, "p_boot": 1.0})
    table = pd.DataFrame(rows)
    table["p_holm"] = table.groupby("cost_model").p_boot.transform(lambda p: holm(p.to_numpy()))
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "trials.csv", index=False)
    with pd.option_context("display.width", 240, "display.max_columns", 30):
        print(table[["trial", "engine", "cost_model", "start", "sharpe_control", "sharpe_treatment", "d_sharpe",
                     "ci_low", "ci_high", "p_boot", "p_holm"]].round(4).to_string(index=False))
    return table


# ---------------------------------------------------------------- M1's own runner (CSV interface)

def export_targets(data_dir: Path, filter_file: Path, factor: str = "combo",
                   min_contracts: int = 10, dev_end: str = DEV_END) -> tuple[pd.DataFrame, pd.Timestamp]:
    """M1's long target table (A1/A2/A3 of one factor) for the development window, and its start."""
    contracts, _ = load_targets(data_dir, filter_file)
    union = sorted(set().union(*(c["data"].index for c in contracts.values())))
    book = in_book(pd.DataFrame({k: c["A3"]["combo"].notna() for k, c in contracts.items()}).reindex(union))
    start, end = window_start(book, min_contracts), pd.Timestamp(dev_end)
    rows = []
    for code, c in contracts.items():
        idx = c["data"].index
        sel = (idx >= start) & (idx <= end)
        inb = book[code].reindex(idx).fillna(False).astype(bool)
        for ap in ("A1", "A2", "A3"):
            t = c[ap][factor].where(inb)[sel]
            rows.append(pd.DataFrame({"date": t.index.strftime("%Y-%m-%d"), "contract": code,
                                      "strategy": ap, "target": t.to_numpy()}))
    return pd.concat(rows, ignore_index=True), start


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--all", action="store_true", help="all 8 M3 trials")
    p.add_argument("--baseline", action="store_true", help="M2's A1/A2 for every factor")
    p.add_argument("--m5", action="store_true", help="the 4 M5 trials (needs m5_outputs/ pool filters)")
    p.add_argument("--data-dir", type=Path, default=Path("data/final"))
    p.add_argument("--m3", default="hmm_k3_quarter_top", help="stem of an m3_outputs/*_filter.csv (single trial)")
    p.add_argument("--filter-file", type=Path, default=None, help="any filter CSV; overrides --m3")
    p.add_argument("--min-contracts", type=int, default=10)
    p.add_argument("--out-dir", type=Path, required=True)
    a = p.parse_args()
    if a.all:
        run_all(a.data_dir, a.out_dir, min_contracts=a.min_contracts)
    elif a.baseline:
        m2_baseline(a.data_dir, a.out_dir, a.min_contracts)
    elif a.m5:
        run_m5(a.data_dir, a.out_dir, min_contracts=a.min_contracts)
    else:
        ff = a.filter_file if a.filter_file is not None else Path("m3_outputs") / f"{a.m3}_filter.csv"
        res, daily = run_trial(*filter_trial(a.data_dir, ff), min_contracts=a.min_contracts)
        a.out_dir.mkdir(parents=True, exist_ok=True)
        res.to_csv(a.out_dir / "a3_vs_a2.csv", index=False)
        daily.rename_axis("date").to_csv(a.out_dir / "daily_net.csv")
        with pd.option_context("display.width", 220, "display.max_columns", 30):
            print(res.round(4).to_string(index=False))
