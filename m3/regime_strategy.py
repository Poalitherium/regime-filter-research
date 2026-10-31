"""Trial 8: a strategy built from the regime model alone (no M2 signal underneath).

Per contract, a 2-state Gaussian HMM on vol-standardised returns z_t, refitted quarterly
on data before each quarter. The position is the sign of the model's one-step-ahead mean,

    E[z_{t+1} | data <= t] = sum_k (p_t A)_k mu_k     (p_t = filtered state probabilities)

sized like A2 (60-day vol, 15% target, cap 2). Compared with static TSMOM at several
horizons, sized the same way, and with M2's combo A2.

    python -m m3.regime_strategy
"""

from pathlib import Path

import numpy as np
import pandas as pd

from backtest.engine import COST_BPS, contract_pnl, in_book, portfolio
from backtest.metrics import paired, perf

from .bocpd import standardise
from .hmm import GaussianHMM
from .portfolio_vol import a2_targets
from .run_real_data import CODES
from .turning_points import vol_scale

DEV_END = "2014-12-31"
HORIZONS = (21, 63, 126, 252)
MIN_TRAIN, N_INIT = 750, 4


def hmm_direction(ret: pd.Series, seed: int = 0):
    """Daily one-step-ahead expected z and the fitted state profile per quarterly refit."""
    z = standardise(ret)
    first = z.first_valid_index()
    z = z.loc[first:]
    X = z.to_numpy().reshape(-1, 1)
    out = pd.DataFrame(np.nan, index=ret.index, columns=["exp_z", "p_high"])
    fits, prev = [], None
    for period in z.index.to_period("Q").unique():
        start, end = period.start_time, (period + 1).start_time
        n_train = int((z.index < start).sum())
        if z.iloc[:n_train].notna().sum() < MIN_TRAIN:
            continue
        n_end = int((z.index < end).sum())
        model = GaussianHMM(n_states=2, n_init=N_INIT, seed=seed + period.ordinal)
        model.fit(X[:n_train], warm_start=prev)
        model.reorder(np.argsort(model.means[:, 0]))  # state 1 = higher mean
        prev = model.params()
        alpha = model.filter(X[:n_end])[n_train:]
        pred = alpha @ model.A
        days = z.index[n_train:n_end]
        ok = z.iloc[n_train:n_end].notna().to_numpy()
        exp = np.where(ok, pred @ model.means[:, 0], np.nan)
        out.loc[days, "exp_z"] = exp
        out.loc[days, "p_high"] = np.where(ok, alpha[:, 1], np.nan)
        fits.append({"vintage": start.date(), "mu_low": model.means[0, 0], "mu_high": model.means[1, 0],
                     "sd_low": float(np.sqrt(model.covs[0, 0, 0])), "sd_high": float(np.sqrt(model.covs[1, 0, 0])),
                     "dur_low": 1 / max(1 - model.A[0, 0], 1e-12), "dur_high": 1 / max(1 - model.A[1, 1], 1e-12)})
    return out, pd.DataFrame(fits)


def build_targets(data_dir: Path = Path("data/final")):
    """Per-contract targets of every strategy, the shared book (a contract enters once the HMM
    direction, 12-month TSMOM and M2's A2 all exist), the inputs and the fitted state profiles."""
    uni = pd.read_csv(data_dir / "universe.csv").set_index("code")
    data = {c: pd.read_csv(data_dir / f"{c}.csv", index_col="date", parse_dates=True) for c in CODES}
    targets = {k: {} for k in ["hmm"] + [f"tsmom_{h}" for h in HORIZONS] + ["m2_A2"]}
    profiles = []
    for c, d in data.items():
        scale = vol_scale(d.ret)
        hd, fits = hmm_direction(d.ret)
        targets["hmm"][c] = np.sign(hd.exp_z) * scale
        p = d.close_adj
        for h in HORIZONS:
            targets[f"tsmom_{h}"][c] = np.sign(p - p.shift(h)).where(p.shift(h).notna()) * scale
        targets["m2_A2"][c] = a2_targets(d, c)["combo"]
        f = fits[pd.to_datetime(fits.vintage) <= pd.Timestamp(DEV_END)]
        profiles.append({"contract": c, "asset_class": uni.asset_class[c], "refits": len(f),
                         "sharpe_low": f.mu_low.median() * np.sqrt(252), "sharpe_high": f.mu_high.median() * np.sqrt(252),
                         "sd_low": f.sd_low.median(), "sd_high": f.sd_high.median(),
                         "dur_low": f.dur_low.median(), "dur_high": f.dur_high.median(),
                         "share_long": float((np.sign(hd.exp_z.loc[:DEV_END]) > 0).mean())})
        print(f"{c:4s} refits {len(fits):3d}  long share {profiles[-1]['share_long']:.2f}", flush=True)

    union = pd.DatetimeIndex(sorted(set().union(*(d.index for d in data.values()))))
    valid = pd.DataFrame({c: targets["hmm"][c].notna() & targets["tsmom_252"][c].notna() & targets["m2_A2"][c].notna()
                          for c in CODES}).reindex(union)
    return uni, data, targets, profiles, in_book(valid)


def run(data_dir: Path = Path("data/final"), out_dir: Path = Path("m3_outputs")):
    uni, data, targets, profiles, book = build_targets(data_dir)
    port = {}
    for name, tg in targets.items():
        pnl = {c: contract_pnl(tg[c], data[c].ret, data[c].roll, COST_BPS[uni.asset_class[c]], 1) for c in CODES}
        port[name] = portfolio(pnl, book, 10).loc[:DEV_END]
    rows = []
    for name, p in port.items():
        s = perf(p.net)
        rows.append({"strategy": name, "start": s["start"], "end": s["end"], "ann_ret": s["ann_ret"], "ann_vol": s["ann_vol"],
                     "sharpe": s["sharpe"], "sharpe_gross": perf(p.gross)["sharpe"], "max_dd": s["max_dd"],
                     "calmar": s["calmar"], "turnover_pa": p.turnover.mean() * 252, "gross_exp": p.gross_exposure.mean(),
                     "ret_corr_with_hmm": float(p.net.corr(port["hmm"].net))})
    pairs = []
    for ctrl in ["tsmom_252", "m2_A2"] + [f"tsmom_{h}" for h in HORIZONS if h != 252]:
        r = paired(port["hmm"].net, port[ctrl].net)
        pairs.append({"treatment": "hmm", "control": ctrl, **{k: r[k] for k in
                      ("d_sharpe", "ci_low", "ci_high", "min_gap_5pct", "p_boot", "corr")}})
    # pooled correlation of daily position signs (in book, dev window)
    sign_corr = []
    for h in HORIZONS:
        a, b = [], []
        for c in CODES:
            days = book.index[book[c].reindex(book.index).fillna(False).to_numpy()]
            days = days[days <= pd.Timestamp(DEV_END)]
            x = np.sign(targets["hmm"][c].reindex(days))
            y = np.sign(targets[f"tsmom_{h}"][c].reindex(days))
            m = x.notna() & y.notna()
            a.append(x[m].to_numpy()); b.append(y[m].to_numpy())
        a, b = np.concatenate(a), np.concatenate(b)
        sign_corr.append({"horizon": h, "sign_corr": float(np.corrcoef(a, b)[0, 1]), "agree": float((a == b).mean()),
                          "ret_corr": float(port["hmm"].net.corr(port[f"tsmom_{h}"].net))})
    summary, prs, sc, prof = pd.DataFrame(rows), pd.DataFrame(pairs), pd.DataFrame(sign_corr), pd.DataFrame(profiles)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "regime_strategy_summary.csv", index=False)
    prs.to_csv(out_dir / "regime_strategy_pairs.csv", index=False)
    sc.to_csv(out_dir / "regime_strategy_signcorr.csv", index=False)
    prof.to_csv(out_dir / "regime_strategy_states.csv", index=False)
    pd.DataFrame({k: v.net for k, v in port.items()}).rename_axis("date").to_csv(out_dir / "regime_strategy_daily.csv")
    with pd.option_context("display.width", 220):
        print(summary.round(3).to_string(index=False))
        print(prs.round(3).to_string(index=False))
        print(sc.round(3).to_string(index=False))
        print(prof.round(3).to_string(index=False))
    return summary, prs, sc, prof


if __name__ == "__main__":
    run()
