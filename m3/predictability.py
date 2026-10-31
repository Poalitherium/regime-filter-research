"""Can anything known today predict next-period trend-signal returns, beyond volatility?

Pre-registered checks before building further regime models (see the M3 notes).
Development window only (to 2014-12-31).

    python -m m3.predictability --round 1   # off-market variables, next week
    python -m m3.predictability --round 2   # in-market candidates, next month

Target: next period's net return of A2 (vol-scaled average of the three M2 factors),
for the equal-weight portfolio (time series) and for each contract (panel), on
non-overlapping periods. Predictors are known at the period's last close.

Round 2 candidates:
    agree : 20-day mean of |mean of the three M2 signals| (1 = all agree)
    mom   : trailing 63-day net return of A2 (own contract / portfolio)
    corr  : mean pairwise correlation of in-book contracts' daily returns, last 60 days
    disp  : cross-sectional std of 60-day trend strength sum(r) / (sd(r) * sqrt(60))
"""

import argparse
from math import erf, exp, lgamma, log, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.engine import COST_BPS, contract_pnl, in_book, portfolio
from m2.pipeline import build_m2

from .features import build_features, load_exog

MACRO = ("vix", "term", "epu", "gpr")
CHANGE_DAYS = 20
DEV_END = "2014-12-31"
OOS_START = "2008-01-01"


# ---------------------------------------------------------------- statistics (no scipy)

def norm_sf(z):
    return 0.5 * (1 - erf(z / sqrt(2)))


def chi2_sf(x, k):
    """Upper tail of chi-square(k): regularised upper incomplete gamma Q(k/2, x/2)."""
    a, x = k / 2.0, x / 2.0
    if x <= 0:
        return 1.0
    if x < a + 1:  # series for P, then Q = 1 - P
        term = s = 1.0 / a
        n = a
        for _ in range(500):
            n += 1
            term *= x / n
            s += term
            if abs(term) < abs(s) * 1e-15:
                break
        return max(0.0, 1.0 - s * exp(-x + a * log(x) - lgamma(a)))
    b, c, d = x + 1 - a, 1e300, 1 / (x + 1 - a)  # continued fraction for Q
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1 / d
        h *= d * c
        if abs(d * c - 1) < 1e-15:
            break
    return exp(-x + a * log(x) - lgamma(a)) * h


def ols(y, X, groups=None, nw_lags=4):
    """OLS with Newey-West (time series) or cluster-robust (by `groups`) covariance."""
    y, X = np.asarray(y, float), np.asarray(X, float)
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    u = y - X @ b
    if groups is None:
        g = X * u[:, None]
        S = g.T @ g
        for L in range(1, nw_lags + 1):
            w = 1 - L / (nw_lags + 1)
            G = g[L:].T @ g[:-L]
            S += w * (G + G.T)
        V = XtX_inv @ S @ XtX_inv * n / (n - k)
    else:
        groups = np.asarray(groups)
        S = np.zeros((k, k))
        for gid in np.unique(groups):
            s = (X[groups == gid] * u[groups == gid, None]).sum(axis=0)
            S += np.outer(s, s)
        G = len(np.unique(groups))
        V = XtX_inv @ S @ XtX_inv * (G / (G - 1)) * ((n - 1) / (n - k))
    return b, V, u


def wald(b, V, idx):
    bq, Vq = b[idx], V[np.ix_(idx, idx)]
    W = float(bq @ np.linalg.solve(Vq, bq))
    return W, chi2_sf(W, len(idx))


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


# ---------------------------------------------------------------- data

FACTORS = ["tsmom", "ma_crossover", "donchian"]


def m2_outputs(d: pd.DataFrame, code: str) -> tuple[pd.Series, pd.DataFrame]:
    """A2 combo target (vol-scaled average of the three factors) and the three raw signals."""
    m2 = build_m2(pd.DataFrame({"date": d.index, "contract": code,
                                "close": d.close_adj.to_numpy(), "return": d.ret.to_numpy()}))
    a2 = m2.pivot(index="date", columns="factor", values="approach2_target_close_t")
    sig = m2.pivot(index="date", columns="factor", values="signal_close_t")
    return a2[FACTORS].mean(axis=1, skipna=False), sig[FACTORS]


def macro_table(exog: pd.DataFrame) -> pd.DataFrame:
    """Levels and 20-business-day changes, on the off-market table's own (lagged) dates."""
    lv = exog[list(MACRO)]
    ch = (lv - lv.shift(CHANGE_DAYS)).add_suffix(f"_chg{CHANGE_DAYS}")
    return pd.concat([lv, ch], axis=1)


def asof(df: pd.DataFrame, dates: pd.DatetimeIndex, limit: int = 5) -> pd.DataFrame:
    return df.reindex(df.index.union(dates)).ffill(limit=limit).reindex(dates)


def build(data_dir: Path, exog_file: Path, bocpd_file: Path | None = None, end: str = DEV_END):
    uni = pd.read_csv(data_dir / "universe.csv")
    bocpd = (pd.read_csv(bocpd_file, parse_dates=["date"]).set_index(["contract", "date"])
             if bocpd_file is not None else None)
    contracts = {}
    for code, asset in zip(uni.code, uni.asset_class):
        d = pd.read_csv(data_dir / f"{code}.csv", index_col="date", parse_dates=True)
        target, signals = m2_outputs(d, code)
        feat = build_features(d.ret, ("logvol20", "er60"))
        if bocpd is not None:  # trial-3 BOCPD outputs, already on this contract's calendar
            b = bocpd.loc[code].reindex(d.index)
            feat = feat.assign(cp20=b["cp20_close_t"], align=b["align_close_t"])
        contracts[code] = {
            "pnl": contract_pnl(target, d.ret, d.roll, COST_BPS[asset], delay=1),
            "valid": target.notna(),
            "feat": feat,
            "signals": signals,
            "ret": d.ret,
        }
    union = sorted(set().union(*(c["pnl"].index for c in contracts.values())))
    book = in_book(pd.DataFrame({k: c["valid"] for k, c in contracts.items()}).reindex(union))
    port = portfolio({k: c["pnl"] for k, c in contracts.items()}, book, min_contracts=10).loc[:end]
    return contracts, book, port, macro_table(load_exog(exog_file))


def market_wide(rets: pd.DataFrame, book: pd.DataFrame, ends: pd.DatetimeIndex, window: int = 60):
    """At each period end: mean pairwise correlation and dispersion of trend strength, in-book contracts."""
    out = []
    for e in ends:
        cols = [c for c in rets.columns if book.at[e, c]]
        R = rets.loc[:e, cols].tail(window)
        if len(R) < window or len(cols) < 3:
            out.append((np.nan, np.nan))
            continue
        C = R.corr().to_numpy()
        corr = np.nanmean(C[~np.eye(len(cols), dtype=bool)])
        strength = R.sum() / (R.std(ddof=1) * np.sqrt(window))
        out.append((corr, strength.std(ddof=1)))
    return pd.DataFrame(out, index=ends, columns=["corr", "disp"])


def sample(port: pd.DataFrame, contracts: dict, book: pd.DataFrame, macro: pd.DataFrame, freq: str):
    """Period-end predictors and next-period target, for the portfolio and for each contract.

    freq "W-FRI" = weeks ending Friday (round 1), "M" = calendar months (round 2).
    """
    days = port.index
    per = days.to_period(freq)
    ends = pd.DatetimeIndex(pd.Series(days, index=days).groupby(per).max().values)  # last trading day
    ret_p = port.net.groupby(per).sum()

    def in_book_mean(series_by_contract):
        wide = pd.DataFrame(series_by_contract).reindex(days).ffill(limit=5)
        return wide.where(book.reindex(days)).mean(axis=1)

    agree_c = {k: c["signals"].mean(axis=1, skipna=False).abs().rolling(20, min_periods=20).mean()
               for k, c in contracts.items()}
    mom_c = {k: c["pnl"].net.rolling(63, min_periods=63).sum() for k, c in contracts.items()}
    rets = pd.DataFrame({k: c["ret"] for k, c in contracts.items()}).reindex(book.index).fillna(0.0)
    mkt = market_wide(rets, book, ends)

    ts = pd.DataFrame({
        "y": ret_p.shift(-1).values,
        "logvol": np.log(port.net.rolling(20, min_periods=20).std() * np.sqrt(252)).reindex(ends).values,
        "er60": in_book_mean({k: c["feat"].er60 for k, c in contracts.items()}).reindex(ends).values,
        "agree": in_book_mean(agree_c).reindex(ends).values,
        "mom": port.net.rolling(63, min_periods=63).sum().reindex(ends).values,
    }, index=ends).join(mkt).join(asof(macro, ends))
    extra_feats = [c for c in ("cp20", "align") if all(c in v["feat"] for v in contracts.values())]
    for col in extra_feats:
        ts[col] = in_book_mean({k: c["feat"][col] for k, c in contracts.items()}).reindex(ends).values

    panel = []
    for code, c in contracts.items():
        pnl = c["pnl"].net.reindex(days).fillna(0.0).where(book[code].reindex(days), np.nan)
        y = pnl.groupby(per).sum(min_count=1).shift(-1)
        inb = book[code].reindex(ends).to_numpy()
        f = asof(c["feat"], ends)
        p = pd.DataFrame({"contract": code, "period": ends, "y": y.values,
                          "logvol": f.logvol20.values, "er60": f.er60.values,
                          "agree": asof(agree_c[code].to_frame("a"), ends)["a"].values,
                          "mom": asof(mom_c[code].to_frame("m"), ends)["m"].values,
                          **{col: f[col].values for col in extra_feats}},
                         index=ends)[inb]
        panel.append(p.join(mkt).join(asof(macro, p.index)))
    return ts, pd.concat(panel)


# ---------------------------------------------------------------- tests

MACRO_COLS = list(MACRO) + [f"{m}_chg{CHANGE_DAYS}" for m in MACRO]
ROUNDS = {
    1: {"freq": "W-FRI", "label": "week", "ann": 52, "nw_lags": 4, "family": MACRO_COLS, "extra": ["er60"]},
    2: {"freq": "M", "label": "month", "ann": 12, "nw_lags": 3, "family": ["agree", "mom", "corr", "disp"],
        "extra": ["er60"], "secondary_joint": MACRO_COLS},
}


def run_tests(df: pd.DataFrame, panel: bool, family: list, extra: list, ann: int, nw_lags: int):
    """One candidate at a time (+ vol control), then the family jointly; Holm across the family."""
    used = ["y", "logvol"] + family + extra
    d = df.dropna(subset=used).copy()
    z = lambda s: (s - s.mean()) / s.std(ddof=1)
    for c in family + extra + ["logvol"]:
        d[c] = z(d[c])
    if panel:  # contract fixed effects: demean within contract
        d[used] = d[used] - d.groupby("contract")[used].transform("mean")
    groups = d["period"].to_numpy() if panel else None
    const = [] if panel else [np.ones(len(d))]

    rows = []
    for x in family + extra:
        X = np.column_stack(const + [d.logvol, d[x]])
        b, V, _ = ols(d.y, X, groups, nw_lags)
        t = b[-1] / np.sqrt(V[-1, -1])
        rows.append({"predictor": x, "ann_ret_per_sd": b[-1] * ann, "t": t, "p": 2 * norm_sf(abs(t))})
    uni = pd.DataFrame(rows).set_index("predictor")
    uni.loc[family, "p_holm"] = holm(uni.loc[family, "p"])

    X = np.column_stack(const + [d.logvol] + [d[c] for c in family])
    b, V, _ = ols(d.y, X, groups, nw_lags)
    off = len(const) + 1
    W, p = wald(b, V, list(range(off, off + len(family))))
    return uni, {"wald": W, "df": len(family), "p": p, "n": len(d)}


def oos_r2(ts: pd.DataFrame, family: list, nw_lags: int):
    """Expanding-window forecasts from OOS_START: control + family vs control-only model."""
    ts = ts.dropna(subset=["y", "logvol"] + family)
    yhat_f, yhat_b, ys = [], [], []
    for i, w in enumerate(ts.index):
        if w < pd.Timestamp(OOS_START):
            continue
        tr = ts.iloc[:i]
        Xf = np.column_stack([np.ones(len(tr)), tr.logvol] + [tr[c] for c in family])
        Xb = np.column_stack([np.ones(len(tr)), tr.logvol])
        bf = np.linalg.lstsq(Xf, tr.y, rcond=None)[0]
        bb = np.linalg.lstsq(Xb, tr.y, rcond=None)[0]
        row = ts.iloc[i]
        yhat_f.append(np.r_[1, row.logvol, row[family].to_numpy(float)] @ bf)
        yhat_b.append(np.r_[1, row.logvol] @ bb)
        ys.append(row.y)
    y, f, bch = map(np.array, (ys, yhat_f, yhat_b))
    r2 = 1 - ((y - f) ** 2).sum() / ((y - bch) ** 2).sum()
    cw = (y - bch) ** 2 - ((y - f) ** 2 - (bch - f) ** 2)  # Clark-West adjusted MSPE gain
    b, V, _ = ols(cw, np.ones((len(cw), 1)), nw_lags=nw_lags)
    t = b[0] / np.sqrt(V[0, 0])
    return {"oos_r2": r2, "clark_west_t": t, "p_one_sided": norm_sf(t), "periods": len(y)}


def main(data_dir: Path, exog_file: Path, out_dir: Path, rnd: int = 1):
    cfg = ROUNDS[rnd]
    contracts, book, port, macro = build(data_dir, exog_file)
    ts, panel = sample(port, contracts, book, macro, cfg["freq"])
    fam = cfg["family"]
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, df, is_panel in [("portfolio", ts, False), ("panel", panel, True)]:
        uni, joint = run_tests(df, is_panel, fam, cfg["extra"], cfg["ann"], cfg["nw_lags"])
        results[name] = (uni, joint)
        n_periods = df.dropna(subset=["y", "logvol"] + fam).index.nunique()
        print(f"\n=== round {rnd}, {name}: next-{cfg['label']} A2 return on predictor (+ vol control); "
              f"per +1 SD, annualised  ({joint['n']} rows, {n_periods} {cfg['label']}s)")
        print(uni.round(4).to_string())
        print(f"joint Wald, decision family ({', '.join(fam)}): W={joint['wald']:.2f}, df={joint['df']}, "
              f"p={joint['p']:.3f}")
        if "secondary_joint" in cfg:
            _, sj = run_tests(df, is_panel, cfg["secondary_joint"], [], cfg["ann"], cfg["nw_lags"])
            print(f"secondary: joint Wald, 8 off-market predictors at this horizon: W={sj['wald']:.2f}, "
                  f"p={sj['p']:.3f}")
        uni.to_csv(out_dir / f"predictability_r{rnd}_{name}.csv")
    o = oos_r2(ts, fam, cfg["nw_lags"])
    print(f"\n=== round {rnd}, out-of-sample (portfolio, expanding from {OOS_START}, vs vol-only model): "
          f"R2_OS={o['oos_r2']:+.4f}, Clark-West t={o['clark_west_t']:+.2f} "
          f"(one-sided p={o['p_one_sided']:.3f}), {cfg['label']}s={o['periods']}")
    return results, o


ROUND3_HORIZONS = [("W-FRI", "week", 52, 4), ("M", "month", 12, 3)]
ROUND3_FAMILY = ["cp20", "align"]


def main_round3(data_dir: Path, exog_file: Path, out_dir: Path, bocpd_file: Path):
    """BOCPD outputs: {cp20, align} x {next week, next month}; Holm across the 4 at each level."""
    contracts, book, port, macro = build(data_dir, exog_file, bocpd_file)
    rows, oos = [], {}
    for freq, label, ann, lags in ROUND3_HORIZONS:
        ts, panel = sample(port, contracts, book, macro, freq)
        for level, df, is_panel in [("portfolio", ts, False), ("panel", panel, True)]:
            uni, joint = run_tests(df, is_panel, ROUND3_FAMILY, [], ann, lags)
            for x, r in uni.iterrows():
                rows.append({"level": level, "horizon": label, "predictor": x, "ann_ret_per_sd": r.ann_ret_per_sd,
                             "t": r.t, "p": r.p, "joint_p": joint["p"], "n": joint["n"]})
        oos[label] = oos_r2(ts, ROUND3_FAMILY, lags)
    res = pd.DataFrame(rows)
    res["p_holm"] = res.groupby("level")["p"].transform(lambda s: pd.Series(holm(s.to_numpy()), index=s.index))
    out_dir.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_dir / "predictability_r3.csv", index=False)
    print("=== round 3 (BOCPD): next-period A2 return on predictor (+ vol control); per +1 SD, annualised")
    print(res.round(4).to_string(index=False))
    for label, o in oos.items():
        print(f"out-of-sample, {label} (portfolio, from {OOS_START}): R2_OS={o['oos_r2']:+.4f}, "
              f"Clark-West t={o['clark_west_t']:+.2f} (one-sided p={o['p_one_sided']:.3f}), n={o['periods']}")
    passed = [(r.horizon, r.predictor, r.level) for r in res.itertuples()
              if r.p_holm < 0.05 and oos[r.horizon]["oos_r2"] > 0]
    print("decision:", "informative: " + str(passed) if passed else "not informative (pre-registered rule)")
    pd.DataFrame([{"horizon": k, **v} for k, v in oos.items()]).to_csv(out_dir / "predictability_r3_oos.csv", index=False)
    return res, oos


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--round", type=int, choices=[1, 2, 3], default=1)
    p.add_argument("--bocpd", type=Path, default=Path("m3_outputs/bocpd_h250_filter.csv"))
    p.add_argument("--data-dir", type=Path, default=Path("data/final"))
    p.add_argument("--exog", type=Path, default=Path("data/processed/offmarket/offmarket_daily.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("m3_outputs"))
    a = p.parse_args()
    if a.round == 3:
        main_round3(a.data_dir, a.exog, a.out_dir, a.bocpd)
    else:
        main(a.data_dir, a.exog, a.out_dir, a.round)
