"""Trial 5: momentum turning points (Garg, Goulding, Harvey & Mazzoleni 2021).

slow = sign(P_t - P_{t-252}), fast = sign(P_t - P_{t-21}) on close_adj. States:
bull (+,+), bear (-,-), correction (slow +, fast -), rebound (slow -, fast +).

Dynamic signal: the common sign in bull/bear. In correction/rebound, fast if, pooled
across all contracts, fast's mean vol-standardised return in that state (signal at t-1 x
z_t, through day t-1) beats slow's, else slow; 0.5*slow + 0.5*fast until the state has
250 contract-days of history. All signals are sized like A2 (60-day vol, 15% target, cap 2).

    python -m m3.turning_points
"""

from pathlib import Path

import numpy as np
import pandas as pd

from backtest.engine import COST_BPS, contract_pnl, in_book, portfolio
from backtest.metrics import paired, perf

from .bocpd import standardise
from .portfolio_vol import a2_targets
from .run_real_data import CODES

SLOW, FAST, MIN_HIST = 252, 21, 250
DEV_END = "2014-12-31"


def vol_scale(ret: pd.Series) -> pd.Series:
    vol = ret.rolling(60, min_periods=60).std(ddof=1) * np.sqrt(252)
    return (0.15 / vol.clip(lower=0.01)).clip(upper=2.0)


def signals(d: pd.DataFrame) -> pd.DataFrame:
    p = d.close_adj
    slow = np.sign(p - p.shift(SLOW)).where(p.shift(SLOW).notna())
    fast = np.sign(p - p.shift(FAST)).where(p.shift(FAST).notna())
    state = pd.Series(np.select([(slow > 0) & (fast > 0), (slow < 0) & (fast < 0), (slow > 0) & (fast < 0),
                                 (slow < 0) & (fast > 0)], ["bull", "bear", "correction", "rebound"], "none"),
                      index=d.index).where(slow.notna() & fast.notna())
    return pd.DataFrame({"slow": slow, "fast": fast, "state": state, "z": standardise(d.ret)})


def dynamic_choice(sig: dict, union: pd.DatetimeIndex) -> pd.DataFrame:
    """Per date and disagreement state: 1 = use fast, 0 = use slow, 0.5 = not enough history.
    Uses signal returns (signal at t-1 x z_t) pooled over contracts, through t-1."""
    rec = []
    for code, s in sig.items():
        st = s.state.shift(1)
        for name in ("slow", "fast"):
            r = s[name].shift(1) * s.z
            for state in ("correction", "rebound"):
                m = (st == state) & r.notna()
                rec.append(pd.DataFrame({"date": s.index[m], "state": state, "sig": name, "r": r[m].values}))
    df = pd.concat(rec)
    out = {}
    for state in ("correction", "rebound"):
        g = df[df.state == state]
        daily = g.groupby(["date", "sig"]).r.agg(["sum", "count"]).unstack("sig").reindex(union).fillna(0.0)
        cum_sum, cum_n = daily["sum"].cumsum(), daily["count"].cumsum()
        mean = cum_sum / cum_n.where(cum_n > 0)
        choice = (mean["fast"] > mean["slow"]).astype(float)
        choice[cum_n["fast"] < MIN_HIST] = 0.5
        out[state] = choice.shift(1)  # decide at t's close with history through t-1
    return pd.DataFrame(out)


def build_targets(data_dir: Path = Path("data/final")):
    """Per-contract targets of every strategy, the shared book (a contract enters once both the
    dynamic signal and M2's A2 exist), and the inputs they came from."""
    uni = pd.read_csv(data_dir / "universe.csv").set_index("code")
    data = {c: pd.read_csv(data_dir / f"{c}.csv", index_col="date", parse_dates=True) for c in CODES}
    sig = {c: signals(d) for c, d in data.items()}
    union = pd.DatetimeIndex(sorted(set().union(*(d.index for d in data.values()))))
    choice = dynamic_choice(sig, union)

    targets = {"dynamic": {}, "static_slow": {}, "static_5050": {}, "m2_A2": {}}
    for c, d in data.items():
        s, scale = sig[c], vol_scale(d.ret)
        ch = choice.reindex(d.index)
        a = pd.Series(np.nan, index=d.index)
        a[s.state == "correction"] = ch.correction[s.state == "correction"]
        a[s.state == "rebound"] = ch.rebound[s.state == "rebound"]
        dyn = np.where(s.state.isin(["bull", "bear"]), s.slow, (1 - a) * s.slow + a * s.fast)
        dyn = pd.Series(dyn, index=d.index).where(s.state.notna())
        targets["dynamic"][c] = dyn * scale
        targets["static_slow"][c] = s.slow * scale
        targets["static_5050"][c] = 0.5 * (s.slow + s.fast) * scale
        targets["m2_A2"][c] = a2_targets(d, c)["combo"]

    valid = pd.DataFrame({c: targets["dynamic"][c].notna() & targets["m2_A2"][c].notna() for c in CODES}).reindex(union)
    return uni, data, sig, choice, targets, in_book(valid)


def run(data_dir: Path = Path("data/final"), out_dir: Path = Path("m3_outputs")):
    uni, data, sig, choice, targets, book = build_targets(data_dir)
    port = {}
    for name, tg in targets.items():
        pnl = {c: contract_pnl(tg[c], data[c].ret, data[c].roll, COST_BPS[uni.asset_class[c]], 1) for c in CODES}
        port[name] = portfolio(pnl, book, 10).loc[:DEV_END]
    rows = []
    for name, p in port.items():
        s = perf(p.net)
        rows.append({"strategy": name, **{k: s[k] for k in ("start", "end", "ann_ret", "ann_vol", "sharpe", "max_dd", "calmar")},
                     "sharpe_gross": perf(p.gross)["sharpe"], "turnover_pa": p.turnover.mean() * 252,
                     "gross_exp": p.gross_exposure.mean()})
    pairs = []
    for ctrl in ("static_slow", "static_5050", "m2_A2"):
        r = paired(port["dynamic"].net, port[ctrl].net)
        pairs.append({"treatment": "dynamic", "control": ctrl, **{k: r[k] for k in
                      ("d_sharpe", "ci_low", "ci_high", "min_gap_5pct", "p_boot", "corr")}})
    # how often each state occurs and what the dynamic rule chose (dev window)
    states = pd.concat([s.state.loc[:DEV_END] for s in sig.values()]).value_counts(normalize=True)
    ch_dev = choice.loc[:DEV_END].dropna()
    summary, prs = pd.DataFrame(rows), pd.DataFrame(pairs)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "turning_points_summary.csv", index=False)
    prs.to_csv(out_dir / "turning_points_pairs.csv", index=False)
    pd.DataFrame({k: v.net for k, v in port.items()}).rename_axis("date").to_csv(out_dir / "turning_points_daily.csv")
    with pd.option_context("display.width", 200):
        print(summary.round(3).to_string(index=False))
        print(prs.round(3).to_string(index=False))
        print("state shares (dev):", states.round(3).to_dict())
        print("dynamic rule, share of dev days choosing fast:",
              {k: round(float((ch_dev[k] == 1).mean()), 3) for k in ch_dev}, " undecided:",
              {k: round(float((ch_dev[k] == 0.5).mean()), 3) for k in ch_dev})
    return summary, prs


if __name__ == "__main__":
    run()
