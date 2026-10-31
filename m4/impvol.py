"""M4-4 (implied / realised vol sizing) and M4-5 (VIX term-structure inversion x risk-on position).

    python -m m4.impvol            # after python -m m4.gates and python -m m4.crowding

Pre-registration: PREREG_M4-4_5.md. Writes to m4_outputs/:
    impvol_filter.csv              M4-4 filter (no gate; deterministic rule)
    inversion_gate.csv, inversion_filter.csv (+ _log)   M4-5
    impvol_describe.csv            forecasting check, per-contract filter stats, M4-5 descriptives
and adds M4-4, M4-5 to gates.json.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.engine import COST_BPS, contract_pnl
from m3.features import load_exog
from m3.predictability import ols

from .common import (DEV_END, LEAD, M4_DIR, OFFMARKET_FILE, RISK_DIR, asof, fe_test, holm, load_all,
                     norm_sf, panel_oos_r2, proportional_filter)

IV_FILE = M4_DIR / "data" / "implied_vol_2014.csv"
IV_FOR = {"ES": "vix", "CL": "ovx", "GC": "gvz", "ZN": "move"}
RV_DAYS, MIN_OBS, FUT_DAYS = 60, 250, 20
LAG_BDAYS, FFILL = 1, 5


def load_iv(exog: pd.DataFrame, path: Path = IV_FILE) -> pd.DataFrame:
    """Every index on the date it becomes usable: one business day after its close.
    `vix` comes from the off-market table, which is already lagged (and in logs)."""
    raw = pd.read_csv(path, index_col="date", parse_dates=True).loc[:DEV_END]
    lagged = raw.copy()
    lagged.index = raw.index + pd.offsets.BDay(LAG_BDAYS)
    lagged = lagged.groupby(level=0).last()  # a weekend print lands on the same Monday as Friday's; keep the later
    vix = np.exp(exog["vix"].dropna()).loc[:DEV_END].rename("vix")
    return lagged.join(vix, how="outer")


def on_dates(s: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    return asof(s.dropna().to_frame("x"), index, limit=FFILL)["x"]


def realised(ret: pd.Series, n: int = RV_DAYS) -> pd.Series:
    return ret.rolling(n, min_periods=n).std(ddof=1) * np.sqrt(252)


# ---------------------------------------------------------------- M4-4

def impvol_filter(contracts: dict, iv: pd.DataFrame):
    rows, stats = [], []
    for code, c in contracts.items():
        idx = c["data"].index
        if code not in IV_FOR:
            rows.append(pd.DataFrame({"date": idx, "contract": code, "filter_close_t": 1.0}))
            continue
        q = realised(c["data"].ret) / on_dates(iv[IV_FOR[code]], idx)
        med = q.expanding(min_periods=MIN_OBS).median()
        f = (q / med).clip(upper=1.0).where(med.notna()).fillna(1.0)
        rows.append(pd.DataFrame({"date": idx, "contract": code, "filter_close_t": f.to_numpy()}))
        live = c["live"].loc[:DEV_END]
        ff = f.reindex(live.index[live])
        on = med.reindex(ff.index).notna()
        stats.append({"analysis": f"M4-4 filter, {code} <- {IV_FOR[code]}", "first_active": ff[on].index.min().date(),
                      "mean_filter": float(ff[on].mean()), "share_below_1": float((ff[on] < 1).mean()),
                      "share_below_0.7": float((ff[on] < 0.7).mean())})
    return pd.concat(rows, ignore_index=True).sort_values(["contract", "date"]), pd.DataFrame(stats)


def forecast_check(contracts: dict, iv: pd.DataFrame) -> pd.DataFrame:
    """Does log IV predict next-20-day log RV beyond log RV60? (descriptive, no strategy returns)."""
    out = []
    for code, col in IV_FOR.items():
        r = contracts[code]["data"].ret.loc[:DEV_END]
        fut = realised(r, FUT_DAYS).shift(-FUT_DAYS)  # RV over t+1 .. t+20
        d = pd.DataFrame({"y": np.log(fut), "rv": np.log(realised(r)), "iv": np.log(on_dates(iv[col], r.index))}).dropna()
        X = np.column_stack([np.ones(len(d)), d.rv, d.iv])
        b, V, _ = ols(d.y.to_numpy(), X, nw_lags=FUT_DAYS)
        t = b[2] / np.sqrt(V[2, 2])
        r2_rv = 1 - np.var(d.y - np.poly1d(np.polyfit(d.rv, d.y, 1))(d.rv)) / np.var(d.y)
        r2_both = 1 - np.var(d.y - X @ b) / np.var(d.y)
        out.append({"analysis": f"forecast next-20d log RV, {code} <- {col}", "n": len(d), "coef_iv": float(b[2]),
                    "t": float(t), "p": 2 * norm_sf(abs(t)), "coef_rv60": float(b[1]),
                    "r2_rv60_only": float(r2_rv), "r2_rv60_and_iv": float(r2_both)})
    return pd.DataFrame(out)


def per_contract_sharpe(contracts: dict, filt: pd.DataFrame) -> pd.DataFrame:
    """Single-contract A3 vs A2 Sharpe on the provisional engine (research bps); descriptive."""
    f = filt.set_index(["contract", "date"]).filter_close_t
    out = []
    for code in IV_FOR:
        c = contracts[code]
        d = c["data"].loc[:DEV_END]
        t2 = c["target"].loc[:DEV_END]
        t3 = t2 * f.loc[code].reindex(d.index)
        first = f.loc[code][f.loc[code] < 1].index.min()
        res = {}
        for k, t in (("A2", t2), ("A3", t3)):
            p = contract_pnl(t, d.ret, d.roll, COST_BPS[c["asset"]], 1).net.loc[first:]
            res[k] = p.mean() / p.std() * np.sqrt(252)
        out.append({"analysis": f"single-contract Sharpe from first active filter, {code}", "from": first.date(),
                    "sharpe_A2": res["A2"], "sharpe_A3": res["A3"], "d_sharpe": res["A3"] - res["A2"]})
    return pd.DataFrame(out)


# ---------------------------------------------------------------- M4-5

def inversion_states(contracts: dict, iv: pd.DataFrame, k: float = 1.0) -> dict:
    out = {}
    for code, rdir in RISK_DIR.items():
        c = contracts[code]
        idx = c["data"].index
        vix, v3 = on_dates(iv["vix"], idx), on_dates(iv["vix3m"], idx)
        inv = (vix >= k * v3).astype(float).where(vix.notna() & v3.notna())
        R = ((np.sign(c["target"]) * rdir) == 1).astype(float).where(c["target"].notna(), 0.0)
        out[code] = pd.DataFrame({"I": inv, "R": R, "W": inv * R}, index=idx)
    return out


def inversion_panel(contracts: dict, st: dict) -> pd.DataFrame:
    rows = []
    for code, s in st.items():
        c = contracts[code]
        live = c["live"].loc[:DEV_END]
        idx = live.index[live]
        lag = s.shift(LEAD).reindex(idx)
        rows.append(pd.DataFrame({
            "contract": code, "date": idx, "period": idx.to_period("M").to_timestamp(),
            "y": c["gross"].reindex(idx).to_numpy(), "logvol": c["logvol20"].shift(LEAD).reindex(idx).to_numpy(),
            "I": lag.I.to_numpy(), "R": lag.R.to_numpy(), "W": lag.W.to_numpy()}))
    return pd.concat(rows, ignore_index=True).dropna(subset=["y", "logvol", "I", "R", "W"])


def main(data_dir: Path, exog_file: Path, out_dir: Path):
    contracts = load_all(data_dir)
    iv = load_iv(load_exog(exog_file))

    # M4-4: deterministic filter, no gate
    f4, s4 = impvol_filter(contracts, iv)
    f4.to_csv(out_dir / "impvol_filter.csv", index=False)
    desc = [forecast_check(contracts, iv), s4, per_contract_sharpe(contracts, f4)]

    # M4-5: gate
    st = inversion_states(contracts, iv)
    df = inversion_panel(contracts, st)
    r = fe_test(df, ["W", "I", "R"])
    o = panel_oos_r2(df, ["logvol", "I", "R"], ["W"])
    prior = pd.read_csv(out_dir / "gates.csv").set_index("trial").p
    p3 = pd.read_csv(out_dir / "crowding_gate.csv").p.iloc[0]
    ps = holm([prior["M4-1"], prior["M4-2"], p3, r["W"]["p"]])
    w = df.W.astype(bool)
    gate = pd.DataFrame([{
        "trial": "M4-5", "name": "VIX term-structure inversion x risk-on position", "term": "W",
        "contracts": df.contract.nunique(), "days": r["n"], "months": r["months"], "state_share": float(w.mean()),
        "inversion_share": float(df.I.mean()), "start": df.date.min().date(), "end": df.date.max().date(),
        "coef_ann": r["W"]["coef"] * 252, "t": r["W"]["t"], "p": r["W"]["p"],
        "I_coef_ann": r["I"]["coef"] * 252, "I_p": r["I"]["p"], "R_coef_ann": r["R"]["coef"] * 252, "R_p": r["R"]["p"],
        "ann_in": df.y[w].mean() * 252, "ann_out": df.y[~w].mean() * 252,
        "oos_r2": o["oos_r2"], "oos_days": o["days"], "p_holm_4": ps[3]}])
    gate["passed"] = (gate.p_holm_4 < 0.05) & (gate.oos_r2 > 0)
    gate.to_csv(out_dir / "inversion_gate.csv", index=False)

    strict = fe_test(inversion_panel(contracts, inversion_states(contracts, iv, 1.05)), ["W", "I", "R"])
    runs = (df.groupby("contract").I.apply(lambda s: s.ne(s.shift()).cumsum()[s == 1].value_counts().mean()))
    desc.append(pd.DataFrame([
        {"analysis": "M4-5 stricter inversion (VIX >= 1.05 x VIX3M)", "coef_ann": strict["W"]["coef"] * 252,
         "t": strict["W"]["t"], "p": strict["W"]["p"], "n": strict["n"]},
        {"analysis": "M4-5 inversion: share of days / mean spell length (days)", "share": float(df.I.mean()),
         "mean_spell": float(runs.mean())}]))
    pd.concat(desc, ignore_index=True).to_csv(out_dir / "impvol_describe.csv", index=False)

    f5, l5 = proportional_filter(
        contracts, list(RISK_DIR),
        pnl_state={k: s.W.shift(LEAD).dropna().astype(bool) for k, s in st.items()},
        filter_state={k: s.W.fillna(0).astype(bool) for k, s in st.items()}, dev_end=DEV_END)
    f5.to_csv(out_dir / "inversion_filter.csv", index=False)
    l5.to_csv(out_dir / "inversion_filter_log.csv", index=False)

    g = gate.iloc[0]
    gj = json.loads((out_dir / "gates.json").read_text())
    gj["M4-4"] = {"passed": True, "p": None, "p_holm": None, "oos_r2": None, "gate": "none (deterministic rule)",
                  "filter_file": str(out_dir / "impvol_filter.csv"), "role": "primary"}
    gj["M4-5"] = {"passed": bool(g.passed), "p": float(g.p), "p_holm": float(g.p_holm_4), "oos_r2": float(g.oos_r2),
                  "filter_file": str(out_dir / "inversion_filter.csv"),
                  "role": "primary" if g.passed else "exploratory"}
    gj["rule_M4-5"] = "Holm over the 4 M4 gates p < 0.05 and panel OOS R2 > 0 (PREREG_M4-4_5 section 4)"
    (out_dir / "gates.json").write_text(json.dumps(gj, indent=2, default=float))
    with pd.option_context("display.width", 240, "display.max_columns", 40):
        print("=== M4-5 gate"); print(gate.round(4).T.to_string())
        print("\ndescriptive:"); print(pd.concat(desc, ignore_index=True).round(4).to_string(index=False))
    return gate


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data/final"))
    p.add_argument("--exog", type=Path, default=OFFMARKET_FILE)
    p.add_argument("--out-dir", type=Path, default=Path("m4_outputs"))
    a = p.parse_args()
    main(a.data_dir, a.exog, a.out_dir)
