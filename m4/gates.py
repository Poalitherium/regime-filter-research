"""M4 gate tests, descriptive analyses and filters (PREREG sections 3-5).

    python -m m4.gates

Development window only (to 2014-12-31). Writes to m4_outputs/:
    gates.csv / gates.json         the two pre-registered gate tests and the decision
    describe.csv                   descriptive analyses (never gates)
    offmarket_audit.csv            coverage of the off-market table
    fomc_filter.csv, riskoff_filter.csv (+ *_log.csv)
        built in every case: the primary filter if the gate passed, otherwise only for the
        exploratory run (PREREG section 3)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from m3.features import load_exog

from .common import (CAL_START, DEV_END, LEAD, OFFMARKET_FILE, RISK_DIR, event_state, fe_test, fomc_dates,
                     holm, load_all, on_calendar, panel_oos_r2, proportional_filter, vix_stress)

ALL = ["ES", "FESX", "NK", "CL", "NG", "GC", "SI", "ZN", "FGBL", "6E", "6J", "ZC", "ZS"]


# ---------------------------------------------------------------- panels

def live_days(c: dict, dev_end: str = DEV_END) -> pd.DatetimeIndex:
    live = c["live"].loc[:dev_end]
    return live.index[live]


def event_panel(contracts: dict, codes=ALL, dev_end: str = DEV_END, timing: str = "prereg") -> pd.DataFrame:
    """Only days the FOMC file covers (from CAL_START): earlier meetings are not in it."""
    rows = []
    for code in codes:
        c = contracts[code]
        idx = live_days(c, dev_end)
        idx = idx[idx >= CAL_START]
        rows.append(pd.DataFrame({
            "contract": code, "asset": c["asset"], "date": idx,
            "period": idx.to_period("M").to_timestamp(),
            "y": c["gross"].reindex(idx).to_numpy(),
            "logvol": c["logvol20"].shift(LEAD).reindex(idx).to_numpy(),
            "E": event_state(idx, code, timing=timing).astype(float).to_numpy(),
            "E_pre": event_state(idx, code, lead=1, timing=timing).astype(float).to_numpy(),  # day before
            "abs_ret": c["data"].ret.abs().reindex(idx).to_numpy(),
        }))
    return pd.concat(rows, ignore_index=True).dropna(subset=["y", "logvol"])


def riskoff_states(contracts: dict, exog: pd.DataFrame, q: float) -> dict:
    """Decision-day S, R, W per participating contract (PREREG 4.2)."""
    stress = vix_stress(exog, q)
    out = {}
    for code, rdir in RISK_DIR.items():
        c = contracts[code]
        idx = c["data"].index
        S = on_calendar(stress, idx)
        R = ((np.sign(c["target"]) * rdir) == 1).astype(float).where(c["target"].notna(), 0.0)
        out[code] = pd.DataFrame({"S": S, "R": R, "W": S * R}, index=idx)
    return out


def riskoff_panel(contracts: dict, states: dict, dev_end: str = DEV_END) -> pd.DataFrame:
    rows = []
    for code, st in states.items():
        c = contracts[code]
        idx = live_days(c, dev_end)
        lag = st.shift(LEAD).reindex(idx)  # decision-day states for the P&L of day tau
        rows.append(pd.DataFrame({
            "contract": code, "asset": c["asset"], "date": idx,
            "period": idx.to_period("M").to_timestamp(),
            "y": c["gross"].reindex(idx).to_numpy(),
            "logvol": c["logvol20"].shift(LEAD).reindex(idx).to_numpy(),
            "S": lag.S.to_numpy(), "R": lag.R.to_numpy(), "W": lag.W.to_numpy()}))
    return pd.concat(rows, ignore_index=True).dropna(subset=["y", "logvol", "S", "R", "W"])


# ---------------------------------------------------------------- audit

def offmarket_audit(exog: pd.DataFrame, start: str = "2001-01-01", end: str = DEV_END) -> pd.DataFrame:
    meta_path = OFFMARKET_FILE.parent / "metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    w = exog.loc[start:end]
    rows = []
    for col in ("vix", "term", "epu", "gpr"):
        s = w[col]
        runs = s.isna().astype(int).groupby(s.notna().cumsum()).sum()
        rows.append({"column": col, "lag_bdays": meta.get(col, {}).get("lag_bdays"),
                     "first_valid": exog[col].first_valid_index().date(), "rows_2001_2014": len(s),
                     "missing_2001_2014": int(s.isna().sum()), "longest_missing_run": int(runs.max()),
                     "used_by_m4": col == "vix"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- main

def describe(ev: pd.DataFrame, contracts: dict, exog: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add(trial, label, df, x, controls=("logvol",), y="y"):
        r = fe_test(df, x, controls, y)
        for name in x:
            scale = 252 if y == "y" else 1
            rows.append({"trial": trial, "analysis": label, "term": name, "outcome": y, "n": r["n"],
                         "coef_ann" if y == "y" else "coef": r[name]["coef"] * scale,
                         "t": r[name]["t"], "p": r[name]["p"]})

    for asset, sub in ev.groupby("asset", sort=False):
        add("M4-1", f"class {asset}", sub, ["E"])
    add("M4-1", "day before the event day", ev, ["E_pre"])
    add("M4-1", "|ret| on event days (no control)", ev, ["E"], controls=(), y="abs_ret")
    base = ev[ev.E == 0].abs_ret.mean()
    rows[-1]["fe_ratio_in_out"] = 1 + rows[-1]["coef"] / base

    # post hoc (REPORT deviation D2): settlement-time mapping for GC/SI (always) and ZC/ZS (before 2013)
    ev2 = event_panel(contracts, timing="settle")
    add("M4-1 post hoc", "gate regression, settlement-time mapping", ev2, ["E"])
    rows[-1]["oos_r2"] = panel_oos_r2(ev2, ["logvol"], ["E"])["oos_r2"]
    for timing, panel in (("prereg", ev), ("settle", ev2)):
        g = panel.groupby(["contract", "E"]).abs_ret.mean().unstack()
        for code, ratio in (g[1.0] / g[0.0]).items():
            rows.append({"trial": "M4-1", "analysis": f"|ret| event/other, {timing} timing", "term": code,
                         "outcome": "abs_ret", "n": int(panel[panel.contract == code].E.sum()),
                         "fe_ratio_in_out": float(ratio)})

    for q in (0.70, 0.90):
        add("M4-2", f"stress quantile {q:.2f}", riskoff_panel(contracts, riskoff_states(contracts, exog, q)),
            ["W", "S", "R"])
    ro = riskoff_panel(contracts, riskoff_states(contracts, exog, 0.80))
    for code, sub in ro.groupby("contract", sort=False):
        add("M4-2", f"contract {code}", sub, ["W", "S", "R"])
    return pd.DataFrame(rows)


def main(data_dir: Path, exog_file: Path, out_dir: Path):
    contracts = load_all(data_dir)
    exog = load_exog(exog_file)
    out_dir.mkdir(parents=True, exist_ok=True)
    offmarket_audit(exog).to_csv(out_dir / "offmarket_audit.csv", index=False)

    # M4-1 gate
    ev = event_panel(contracts)
    r1 = fe_test(ev, ["E"])
    o1 = panel_oos_r2(ev, ["logvol"], ["E"])
    e = ev.E.astype(bool)
    g1 = {"trial": "M4-1", "name": "FOMC scheduled statement days", "term": "E", "contracts": ev.contract.nunique(),
          "days": r1["n"], "months": r1["months"], "state_share": float(e.mean()),
          "events": int(len(fomc_dates()[(fomc_dates() >= ev.date.min()) & (fomc_dates() <= ev.date.max())])), "start": ev.date.min().date(), "end": ev.date.max().date(),
          "coef_ann": r1["E"]["coef"] * 252, "t": r1["E"]["t"], "p": r1["E"]["p"],
          "ann_in": ev.y[e].mean() * 252, "ann_out": ev.y[~e].mean() * 252, "oos_r2": o1["oos_r2"],
          "oos_days": o1["days"]}

    # M4-2 gate
    states = riskoff_states(contracts, exog, 0.80)
    ro = riskoff_panel(contracts, states)
    r2 = fe_test(ro, ["W", "S", "R"])
    o2 = panel_oos_r2(ro, ["logvol", "S", "R"], ["W"])
    w = ro.W.astype(bool)
    g2 = {"trial": "M4-2", "name": "VIX stress x risk-on position", "term": "W", "contracts": ro.contract.nunique(),
          "days": r2["n"], "months": r2["months"], "state_share": float(w.mean()), "events": np.nan,
          "start": ro.date.min().date(), "end": ro.date.max().date(),
          "coef_ann": r2["W"]["coef"] * 252, "t": r2["W"]["t"], "p": r2["W"]["p"],
          "ann_in": ro.y[w].mean() * 252, "ann_out": ro.y[~w].mean() * 252, "oos_r2": o2["oos_r2"],
          "oos_days": o2["days"], "S_coef_ann": r2["S"]["coef"] * 252, "S_p": r2["S"]["p"],
          "R_coef_ann": r2["R"]["coef"] * 252, "R_p": r2["R"]["p"], "stress_share": float(ro.S.mean())}

    gates = pd.DataFrame([g1, g2])
    gates["p_holm"] = holm(gates.p.to_numpy())
    gates["passed"] = (gates.p_holm < 0.05) & (gates.oos_r2 > 0)
    gates.to_csv(out_dir / "gates.csv", index=False)

    desc = describe(ev, contracts, exog)
    desc.to_csv(out_dir / "describe.csv", index=False)

    # filters (always built; primary only if the gate passed)
    f1, l1 = proportional_filter(
        contracts, ALL,
        pnl_state={k: event_state(contracts[k]["data"].index, k).loc[CAL_START:] for k in ALL},
        filter_state={k: event_state(contracts[k]["data"].index, k, lead=LEAD) for k in ALL}, dev_end=DEV_END)
    f2, l2 = proportional_filter(
        contracts, list(RISK_DIR),
        pnl_state={k: st.W.shift(LEAD).dropna().astype(bool) for k, st in states.items()},
        filter_state={k: st.W.astype(bool) for k, st in states.items()}, dev_end=DEV_END)
    for name, f, lg in (("fomc", f1, l1), ("riskoff", f2, l2)):
        f.to_csv(out_dir / f"{name}_filter.csv", index=False)
        lg.to_csv(out_dir / f"{name}_filter_log.csv", index=False)

    decision = {r.trial: {"passed": bool(r.passed), "p": r.p, "p_holm": r.p_holm, "oos_r2": r.oos_r2,
                          "filter_file": str(out_dir / f"{'fomc' if r.trial == 'M4-1' else 'riskoff'}_filter.csv"),
                          "role": "primary" if r.passed else "exploratory"}
                for r in gates.itertuples()}
    decision["rule"] = "Holm(2) p < 0.05 and panel OOS R2 > 0 (PREREG section 3)"
    (out_dir / "gates.json").write_text(json.dumps(decision, indent=2, default=float))
    with pd.option_context("display.width", 240, "display.max_columns", 40):
        print("=== M4 gates (daily gross A2 combo P&L, contract FE, SEs clustered by month)")
        print(gates.round(4).to_string(index=False))
        print("\ndescriptive (not gates):")
        print(desc.round(4).to_string(index=False))
        print("\nfilter estimates, last vintages:")
        print(l1.tail(3).round(4).to_string(index=False))
        print(l2.tail(3).round(4).to_string(index=False))
    return gates, desc


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data/final"))
    p.add_argument("--exog", type=Path, default=OFFMARKET_FILE)
    p.add_argument("--out-dir", type=Path, default=Path("m4_outputs"))
    a = p.parse_args()
    main(a.data_dir, a.exog, a.out_dir)
