"""Is what remains after volatility just noise? Three pre-registered evidence tests (dev window).

    python -m m3.evidence noise   # B: Ljung-Box / BDS before and after GARCH standardisation
    python -m m3.evidence bound   # A: upper bound on predictability -> maximum Sharpe gain
    python -m m3.evidence power   # C: plant regimes in real data, measure detection power
"""

import argparse
import json
from pathlib import Path

import numba
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import bds

from .run_real_data import CODES

DEV_END = "2014-12-31"
OUT = Path("m3_outputs")


# ---------------------------------------------------------------- B. noise after volatility

@numba.njit(cache=True)
def _garch_nll(params, r):
    omega, alpha, beta = params
    s2 = np.var(r)
    nll = 0.0
    for t in range(len(r)):
        if t > 0:
            s2 = omega + alpha * r[t - 1] ** 2 + beta * s2
        nll += 0.5 * (np.log(s2) + r[t] ** 2 / s2)
    return nll


@numba.njit(cache=True)
def _garch_sigma(params, r):
    omega, alpha, beta = params
    out = np.empty(len(r))
    s2 = np.var(r)
    for t in range(len(r)):
        if t > 0:
            s2 = omega + alpha * r[t - 1] ** 2 + beta * s2
        out[t] = np.sqrt(s2)
    return out


def fit_garch(r: np.ndarray):
    """Gaussian quasi-MLE GARCH(1,1); returns (omega, alpha, beta), conditional sd, converged.

    Fitted on r / sd(r) so the parameters are O(1) for the optimiser; sigma is scaled back."""
    scale = np.std(r)
    x = r / scale
    cons = [{"type": "ineq", "fun": lambda p: 0.9999 - p[1] - p[2]}]
    best = None
    for a0, b0 in [(0.05, 0.90), (0.10, 0.85), (0.03, 0.95)]:
        res = minimize(lambda p: _garch_nll(p, x), x0=np.array([1 - a0 - b0, a0, b0]), method="SLSQP",
                       bounds=[(1e-8, None), (0.0, 1.0), (0.0, 1.0)], constraints=cons,
                       options={"maxiter": 500, "ftol": 1e-10})
        if best is None or res.fun < best.fun:
            best = res
    return best.x, _garch_sigma(best.x, x) * scale, bool(best.success)


def robust_autocorr_test(y: np.ndarray, lags: int = 6):
    """Joint test that lags 1..L do not predict y, with White (heteroskedasticity-robust) errors."""
    from .predictability import chi2_sf, ols
    Y = y[lags:]
    X = np.column_stack([np.ones(len(Y))] + [y[lags - k:-k] for k in range(1, lags + 1)])
    b, V, _ = ols(Y, X, nw_lags=0)
    idx = list(range(1, lags + 1))
    W = float(b[idx] @ np.linalg.solve(V[np.ix_(idx, idx)], b[idx]))
    return W, chi2_sf(W, lags)


def noise(data_dir: Path = Path("data/final")):
    rows = []
    for c in CODES:
        ret = pd.read_csv(data_dir / f"{c}.csv", index_col="date", parse_dates=True).ret.loc[:DEV_END].dropna()
        r = (ret - ret.mean()).to_numpy()
        params, sig, ok = fit_garch(r)
        z = r / sig
        lb = lambda x: float(acorr_ljungbox(x, lags=[10]).lb_pvalue.iloc[0])
        b_r, b_z = bds(r, max_dim=3, distance=1.0)[1], bds(z, max_dim=3, distance=1.0)[1]
        rows.append({"contract": c, "n": len(r), "converged": ok, "alpha": params[1], "beta": params[2],
                     "lb_r": lb(r), "lb_r2": lb(r ** 2), "lb_z": lb(z), "lb_z2": lb(z ** 2),
                     "bds_r_m2": b_r[0], "bds_r_m3": b_r[1], "bds_z_m2": b_z[0], "bds_z_m3": b_z[1]})
    t = pd.DataFrame(rows)
    pcols = [c for c in t.columns if c.startswith(("lb_", "bds_"))]
    share = (t[pcols] < 0.05).mean().rename("share_rejecting_5pct")

    from .predictability import build
    _, _, port, _ = build(data_dir, Path("data/processed/offmarket/offmarket_daily.csv"))
    monthly = port.net.groupby(port.index.to_period("M")).sum()
    lb_m = acorr_ljungbox(monthly.to_numpy(), lags=[6], return_df=True)
    acf1 = float(monthly.autocorr(1))
    acfs = {k: round(float(monthly.autocorr(k)), 3) for k in range(1, 7)}
    W, p_rob = robust_autocorr_test(monthly.to_numpy(), 6)
    OUT.mkdir(exist_ok=True)
    t.round(6).to_csv(OUT / "evidence_noise.csv", index=False)
    summary = {"share_rejecting": share.round(3).to_dict(), "monthly_A2_lb6_p": float(lb_m.lb_pvalue.iloc[0]),
               "monthly_A2_acf1": acf1, "monthly_A2_acf_1to6": acfs,
               "monthly_A2_robust_wald6": W, "monthly_A2_robust_p": p_rob, "months": len(monthly),
               "garch_converged_all": bool(t.converged.all())}
    (OUT / "evidence_noise_summary.json").write_text(json.dumps(summary, indent=2))
    with pd.option_context("display.width", 200):
        print(t.round(3).to_string(index=False))
    print("\nshare of 13 contracts rejecting at 5%:")
    print(share.round(3).to_string())
    print(f"\nmonthly A2 combo portfolio: Ljung-Box Q(6) p = {summary['monthly_A2_lb6_p']:.3f}, "
          f"lag-1 autocorrelation {acf1:+.3f}, {len(monthly)} months")
    print(f"autocorrelations lags 1-6: {acfs}")
    print(f"heteroskedasticity-robust joint test of lags 1-6: W = {W:.2f}, p = {p_rob:.3f}")
    print("GARCH converged for all contracts:", summary["garch_converged_all"])
    return t, summary


# ---------------------------------------------------------------- A. predictability bound

def _oos(panel: pd.DataFrame, cols: list, kind: str, group: str = "month"):
    """Walk-forward yearly refits from 2006 (purge + embargo); returns y and OOS forecasts."""
    from pandas.tseries.offsets import BDay
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    out = []
    for y in range(2006, 2015):
        refit = pd.Timestamp(y, 1, 1)
        tr = panel[(panel.label_end < refit - BDay(5)) & panel.fwd.notna()]
        te = panel[(panel.week >= refit) & (panel.week < pd.Timestamp(y + 1, 1, 1)) & panel.fwd.notna()]
        if len(tr) < 100 or te.empty:
            continue
        model = (HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=100,
                                               l2_regularization=1.0, random_state=0) if kind == "gbm" else
                 make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0)))
        model.fit(tr[cols], tr.fwd)
        out.append(te[[group, "fwd"]].assign(pred=model.predict(te[cols])))
    return pd.concat(out)


def _r2_with_bound(o: pd.DataFrame, group: str, reps: int = 5000, seed: int = 0):
    """OOS R^2 vs a zero forecast, and its 95% upper bound from a bootstrap over months."""
    r2 = 1 - ((o.fwd - o.pred) ** 2).sum() / (o.fwd ** 2).sum()
    g = o.assign(e2=(o.fwd - o.pred) ** 2, y2=o.fwd ** 2).groupby(group)[["e2", "y2"]].sum()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), (reps, len(g)))
    e2, y2 = g.e2.to_numpy()[idx].sum(1), g.y2.to_numpy()[idx].sum(1)
    boot = 1 - e2 / y2
    return float(r2), float(np.quantile(boot, 0.95))


def max_sharpe_gain(sr0_month: float, r2: float) -> float:
    """Campbell-Thompson: annualised Sharpe gain of optimal timing with predictive R^2 (monthly)."""
    r2 = max(r2, 0.0)
    sr_star = np.sqrt((sr0_month ** 2 + r2) / (1 - r2))
    return float((sr_star - abs(sr0_month)) * np.sqrt(12))


def insample_linear(monthly: pd.DataFrame, cols: list):
    """Infeasible full-sample linear fit (contract fixed effects, month-clustered errors).
    In-sample R^2 can only overstate linear predictability, so it bounds it from above."""
    from .predictability import chi2_sf
    d = monthly.dropna(subset=["fwd"]).copy()
    X = d[cols].astype(float)
    X = X.fillna(X.median())
    y = (d.fwd - d.groupby("contract").fwd.transform("mean")).to_numpy()
    X = X - X.groupby(d.contract).transform("mean")  # contract fixed effects
    X = X.loc[:, X.std(ddof=1) > 1e-12]
    X = ((X - X.mean()) / X.std(ddof=1)).to_numpy()
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    u = y - X @ b
    XtX_inv = np.linalg.pinv(X.T @ X)
    groups = d.month.astype(str).to_numpy()
    S = np.zeros((X.shape[1], X.shape[1]))
    for gid in np.unique(groups):
        s = (X[groups == gid] * u[groups == gid, None]).sum(axis=0)
        S += np.outer(s, s)
    G = len(np.unique(groups))
    V = XtX_inv @ S @ XtX_inv * G / (G - 1)
    r2 = 1 - (u ** 2).sum() / (y ** 2).sum()
    n, k = len(y), X.shape[1] + d.contract.nunique()
    adj = 1 - (1 - r2) * (n - 1) / (n - k)
    W = float(b @ np.linalg.pinv(V) @ b)
    return {"n": n, "features": int(X.shape[1]), "r2_insample": float(r2), "adj_r2": float(adj),
            "wald": W, "p": chi2_sf(W, X.shape[1]), "rank": int(np.linalg.matrix_rank(X))}


def bound(data_dir: Path = Path("data/final")):
    from .metalabel import HAND, SIG, build_panel
    panel = build_panel(data_dir, Path("data/processed/offmarket/offmarket_daily.csv"),
                        OUT / "bocpd_h250_filter.csv")
    panel["month"] = panel.week.dt.to_period("M")
    last = panel.groupby(["contract", "month"]).week.transform("max")
    monthly = panel[panel.week == last].copy()  # one sample per contract-month, ~21-day horizon
    cols = HAND + [c for c in SIG if c != "logvol20"]
    rows = []
    # contract level (panel)
    sr_c = monthly.groupby("contract").fwd.apply(lambda s: s.mean() / s.std()).median()
    for kind in ("ridge", "gbm"):
        o = _oos(monthly, cols, kind)
        r2, ub = _r2_with_bound(o, "month")
        rows.append({"level": "contract", "model": kind, "n": len(o), "oos_r2": r2, "r2_upper95": ub,
                     "sr0_month": sr_c, "max_gain_at_upper": max_sharpe_gain(sr_c, ub)})
    # portfolio level: average target and features across contracts each month
    pm = monthly.groupby("month")[cols + ["fwd"]].mean()
    pm = pm.assign(week=monthly.groupby("month").week.max(), label_end=monthly.groupby("month").label_end.max()).reset_index()
    sr_p = pm.fwd.mean() / pm.fwd.std()
    for kind in ("ridge", "gbm"):
        o = _oos(pm, cols, kind)
        r2, ub = _r2_with_bound(o, "month")
        rows.append({"level": "portfolio", "model": kind, "n": len(o), "oos_r2": r2, "r2_upper95": ub,
                     "sr0_month": sr_p, "max_gain_at_upper": max_sharpe_gain(sr_p, ub)})
    # reference: what a given R^2 would be worth (contract-level SR0)
    ref = [{"r2": r, "max_gain_contract": max_sharpe_gain(sr_c, r), "max_gain_portfolio": max_sharpe_gain(sr_p, r)}
           for r in (0.001, 0.0025, 0.005, 0.01, 0.02, 0.05)]
    t, rt = pd.DataFrame(rows), pd.DataFrame(ref)
    ins = insample_linear(monthly[monthly.week <= DEV_END], cols)
    ins["max_gain_at_insample_r2"] = max_sharpe_gain(sr_c, ins["r2_insample"])
    ins["max_gain_at_adj_r2"] = max_sharpe_gain(sr_c, ins["adj_r2"])
    t.round(6).to_csv(OUT / "evidence_bound.csv", index=False)
    rt.round(6).to_csv(OUT / "evidence_bound_reference.csv", index=False)
    (OUT / "evidence_bound_insample.json").write_text(json.dumps(ins, indent=2, default=float))
    print(t.round(4).to_string(index=False))
    print("\nin-sample (infeasible) linear fit, contract level, all months 2001-2014:",
          {k: round(v, 4) if isinstance(v, float) else v for k, v in ins.items()})
    print("\nwhat a monthly predictive R^2 is worth (Campbell-Thompson, annualised Sharpe gain):")
    print(rt.round(4).to_string(index=False))
    return t, rt


# ---------------------------------------------------------------- C. injection / power

MUS = (0.02, 0.04, 0.06, 0.10)
SEEDS = (0, 1, 2, 3, 4)
DURATION = 60


@numba.njit(cache=True)
def _inject(r, sigma, g, mu, lookback):
    """r~_t = r_t + mu * sigma_{t-1} * s_t * g_t, s_t = sign of the trailing synthetic return."""
    out = r.copy()
    run = 0.0
    for t in range(len(r)):
        if t >= lookback:
            s = 1.0 if run > 0 else (-1.0 if run < 0 else 0.0)
            if not np.isnan(sigma[t]):
                out[t] = r[t] + mu * sigma[t] * s * g[t]
        run += out[t]
        if t >= lookback:
            run -= out[t - lookback]
    return out


def markov_states(n: int, rng, duration: int = DURATION) -> np.ndarray:
    g = np.empty(n)
    g[0] = 1.0 if rng.random() < 0.5 else -1.0
    switch = rng.random(n) < 1.0 / duration
    for t in range(1, n):
        g[t] = -g[t - 1] if switch[t] else g[t - 1]
    return g


def make_world(base: Path, dest: Path, mu: float, seed: int, duration: int = DURATION) -> dict:
    """Synthetic data/final-like folder on the dev window; returns the hidden states per contract."""
    import shutil
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy(base / "universe.csv", dest / "universe.csv")
    rng = np.random.default_rng(1000 + seed)
    states = {}
    for c in CODES:
        d = pd.read_csv(base / f"{c}.csv", index_col="date", parse_dates=True).loc[:DEV_END]
        r = d.ret.fillna(0.0).to_numpy()
        sigma = d.ret.rolling(60, min_periods=60).std(ddof=1).shift(1).to_numpy()
        g = markov_states(len(r), rng, duration)
        rt = _inject(r, sigma, g, mu, 60)
        px = 100 * np.cumprod(1 + rt)
        pd.DataFrame({"close": px, "close_adj": px, "ret": rt, "volume": np.nan, "roll": d.roll.values,
                      "kind": "synthetic"}, index=d.index).rename_axis("date").to_csv(dest / f"{c}.csv")
        states[c] = pd.Series(g, index=d.index)
    return states


def _paired_one_sided(a3, a2, reps=2000):
    from backtest.metrics import block_bootstrap_sharpe_diff
    res = block_bootstrap_sharpe_diff(a3, a2, reps=reps, seed=0, return_boot=True)
    boot = res.pop("boot")
    return res["d_sharpe"], float(np.mean(boot - boot.mean() >= res["d_sharpe"] - 1e-12))


def _combo_returns(world: Path, filter_file: Path, start: str = "2006-01-01"):
    from backtest.engine import COST_BPS, contract_pnl, in_book, portfolio
    from backtest.run import load_targets
    contracts, _ = load_targets(world, filter_file)
    union = sorted(set().union(*(c["data"].index for c in contracts.values())))
    book = in_book(pd.DataFrame({k: c["A3"]["combo"].notna() for k, c in contracts.items()}).reindex(union))
    out = {}
    for ap in ("A2", "A3"):
        pnl = {k: contract_pnl(c[ap]["combo"], c["data"].ret, c["data"].roll, COST_BPS[c["asset"]], 1)
               for k, c in contracts.items()}
        out[ap] = portfolio(pnl, book, 10).loc[start:DEV_END].net
    return out["A2"], out["A3"]


def power(base: Path = Path("data/final"), work: Path = Path("m3_outputs/power_work"),
          mus=MUS, seeds=SEEDS, duration: int = DURATION, tag: str = ""):
    from sklearn.metrics import roc_auc_score
    from .bocpd import BOCPDConfig
    from .metalabel import HAND, build_panel, walk_forward, write_filter
    from .run_bocpd import run as run_bocpd
    rows = []
    for mu in mus:
        for seed in seeds:
            w = work / f"d{duration}_mu{mu}_s{seed}"
            states = make_world(base, w / "data", mu, seed, duration)
            run_bocpd(w / "data", w, BOCPDConfig())
            panel = build_panel(w / "data", Path("data/processed/offmarket/offmarket_daily.csv"), w / "bocpd_h250_filter.csv")
            preds, aucs = walk_forward(panel, HAND, "gbm")
            write_filter(preds, w / "data", w / "meta_filter.csv")
            lab = preds.y.notna()
            auc = roc_auc_score(preds.y[lab], preds.p[lab])
            a2, a6 = _combo_returns(w / "data", w / "meta_filter.csv")
            d6, p6 = _paired_one_sided(a6, a2)
            orows = [pd.DataFrame({"date": s.index, "contract": c, "filter_close_t": (s > 0).astype(float).values})
                     for c, s in states.items()]
            pd.concat(orows).to_csv(w / "oracle_filter.csv", index=False)
            a2o, ao = _combo_returns(w / "data", w / "oracle_filter.csv")
            d_or, _ = _paired_one_sided(ao, a2o, reps=200)
            rows.append({"duration": duration, "mu": mu, "seed": seed, "auc": auc, "d_sharpe_meta": d6,
                         "p_meta_one_sided": p6, "detected": p6 < 0.05, "d_sharpe_oracle": d_or})
            print(rows[-1], flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / f"evidence_power{tag}.csv", index=False)
    s = t.groupby(["duration", "mu"]).agg(power=("detected", "mean"), auc=("auc", "mean"),
                                          d_meta=("d_sharpe_meta", "mean"), d_oracle=("d_sharpe_oracle", "mean"))
    s.to_csv(OUT / f"evidence_power{tag}_summary.csv")
    print(s.round(3).to_string())
    return t, s


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("test", choices=["noise", "bound", "power", "control"])
    p = a.parse_args()
    if p.test == "noise":
        noise()
    elif p.test == "bound":
        bound()
    elif p.test == "power":
        power()
    else:  # positive control: identifiable (slow) regimes
        power(mus=(0.06, 0.10), seeds=(0, 1, 2), duration=250, tag="_control")
