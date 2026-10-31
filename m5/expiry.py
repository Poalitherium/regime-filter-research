"""Trial M5-4: contract expiry (PREREG 4.5). Gate test first; a filter only if it passes.

    python -m m5.expiry

Calendar: m5/calendar.py (exchange last-trading-day rules; the data's `roll` column is
never an input). FESX and FGBL are stand-in series with no contract switch: excluded,
filter 1.

Gate (development window only):
    sample   : 11 futures, daily, from the first A2 combo position to 2014-12-31
    target   : gross A2 combo P&L on day tau (no costs, so the engine's roll-day cost
               cannot create a window effect by construction)
    predictor: W = 1[tau within +-3 weekdays of a scheduled switch day];
               control: logvol20 at the decision close (2 trading days earlier)
    test     : contract fixed effects, SEs clustered by calendar month, two-sided
    pass     : p < 0.05 AND out-of-sample R2 > 0 (monthly expanding re-estimation from 2008-01)

Descriptive only (never a gate): per-class effects, pre/post halves of the window,
signal behaviour (|change in target|, sign flips), |ret| in vs out of the window, the
vendor's monthly cycle for GC/SI/6E/6J, and the calendar check against detected rolls.

Writes m5_outputs/expiry_gate.csv, expiry_gate.json, expiry_describe.csv,
expiry_calendar_qa.csv and, if the gate passes, expiry_filter.csv.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from m3.predictability import OOS_START, norm_sf, ols

from . import calendar as cal
from . import calfilter
from .common import DEV_END, LEAD, load_all

HALF_WIDTH = 3
FUTURES = ["ES", "NK", "CL", "NG", "GC", "SI", "ZN", "6E", "6J", "ZC", "ZS"]
VENDOR_MONTHLY = ("GC", "SI", "6E", "6J")  # descriptive alternative cycle


def daily_panel(contracts: dict, codes=FUTURES, schedules: dict | None = None, dev_end: str = DEV_END):
    rows = []
    for code in codes:
        c = contracts[code]
        live = c["live"].loc[:dev_end]
        idx = live.index[live]
        d = cal.days_to_roll(idx, code, (schedules or {}).get(code))  # alternative: descriptive only
        dist = np.minimum(d.d_next, d.d_last)
        target = c["target"]
        dw = target.diff().abs()
        flip = (np.sign(target) != np.sign(target.shift(1))) & target.notna() & target.shift(1).notna()
        rows.append(pd.DataFrame({
            "contract": code, "asset": c["asset"], "date": idx,
            "period": idx.to_period("M").to_timestamp(),
            "y": c["gross"].reindex(idx).to_numpy(),
            "logvol": c["logvol20"].shift(LEAD).reindex(idx).to_numpy(),  # decision-close vol
            "W": (dist <= HALF_WIDTH).astype(float).to_numpy(),
            "W_pre": ((d.d_next >= 1) & (d.d_next <= HALF_WIDTH)).astype(float).to_numpy(),
            "W_post": (d.d_last <= HALF_WIDTH).astype(float).to_numpy(),
            # signal behaviour on the same calendar days (decision dates, window at lead 0)
            "abs_dw": dw.reindex(idx).to_numpy(), "flip": flip.reindex(idx).astype(float).to_numpy(),
            "abs_ret": c["data"].ret.abs().reindex(idx).to_numpy(),
        }))
    return pd.concat(rows, ignore_index=True).dropna(subset=["y", "logvol"])


def fe_test(df: pd.DataFrame, x: list, y: str = "y", control: bool = True) -> dict:
    """Within-contract OLS of y on x (+ logvol), SEs clustered by calendar month.
    Coefficients are raw: for a dummy, the in-minus-out daily mean."""
    cols = [y] + (["logvol"] if control else []) + x
    d = df.dropna(subset=cols).copy()
    d[cols] = d[cols] - d.groupby("contract")[cols].transform("mean")
    X = d[cols[1:]].to_numpy()
    b, V, _ = ols(d[y].to_numpy(), X, groups=d["period"].to_numpy())
    out = {"n": len(d), "months": d.period.nunique()}
    for j, name in enumerate(cols[1:]):
        if name in x:
            t = b[j] / np.sqrt(V[j, j])
            out[name] = {"coef": float(b[j]), "t": float(t), "p": 2 * norm_sf(abs(t))}
    return out


def panel_oos_r2(df: pd.DataFrame, x: list, start: str = OOS_START) -> dict:
    """Monthly expanding re-estimation: FE + logvol + x vs FE + logvol, from `start`."""
    codes = sorted(df.contract.unique())
    D = lambda d: np.column_stack([(d.contract == k).to_numpy(float) for k in codes])  # noqa: E731
    ys, ff, bb = [], [], []
    for m in sorted(df.period[df.period >= pd.Timestamp(start)].unique()):
        tr, te = df[df.period < m], df[df.period == m]
        Xb, Tb = np.column_stack([D(tr), tr.logvol]), np.column_stack([D(te), te.logvol])
        Xf, Tf = np.column_stack([Xb, tr[x]]), np.column_stack([Tb, te[x]])
        ff += list(Tf @ np.linalg.lstsq(Xf, tr.y, rcond=None)[0])
        bb += list(Tb @ np.linalg.lstsq(Xb, tr.y, rcond=None)[0])
        ys += list(te.y)
    y, f, b = map(np.array, (ys, ff, bb))
    return {"oos_r2": float(1 - ((y - f) ** 2).sum() / ((y - b) ** 2).sum()), "days": len(y)}


def calendar_qa(contracts: dict) -> pd.DataFrame:
    """Dates only: how close the data's detected rolls are to the scheduled switch days."""
    rows = []
    for code in FUTURES:
        d = contracts[code]["data"]
        det = d.index[d.roll.astype(bool)]
        for label, sched in [("exchange", None), ("monthly", (cal.ALL_MONTHS, cal.SCHEDULES[code][1]))]:
            if label == "monthly" and code not in VENDOR_MONTHLY:
                continue
            dr = cal.days_to_roll(det, code, sched)
            dist = np.minimum(dr.d_next, dr.d_last)
            sw = cal.roll_days(code, d.index[0], d.index[-1], sched)
            rows.append({"contract": code, "cycle": label, "detected": len(det), "scheduled": len(sw),
                         "within_0": float((dist == 0).mean()), "within_1": float((dist <= 1).mean()),
                         "within_3": float((dist <= 3).mean())})
    return pd.DataFrame(rows)


def describe(df: pd.DataFrame, df_monthly: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add(label, sub, x, y="y", control=True, data=None):
        r = fe_test(sub if data is None else data, x, y, control)
        for name in x:
            rows.append({"analysis": label, "term": name, "outcome": y, "n": r["n"],
                         "coef_ann" if y == "y" else "coef": r[name]["coef"] * (252 if y == "y" else 1),
                         "t": r[name]["t"], "p": r[name]["p"]})

    add("all futures, pre/post halves", df, ["W_pre", "W_post"])
    for asset, sub in df.groupby("asset", sort=False):
        add(f"class {asset}", sub, ["W"])
    add("vendor monthly cycle for GC/SI/6E/6J", None, ["W"], data=df_monthly)
    for y in ("abs_dw", "flip", "abs_ret"):
        add(f"signal behaviour: {y}", df, ["W"], y=y, control=False)
    out = pd.DataFrame(rows)
    # within-contract ratio implied by the fixed-effects coefficient (a raw pooled ratio would
    # mix contracts: monthly-cycle CL/NG are both more volatile and more often in the window)
    base = {y: df[df.W == 0][y].mean() for y in ("abs_dw", "flip", "abs_ret")}
    sig = out.outcome.isin(list(base))
    out.loc[sig, "fe_ratio_in_out"] = 1 + out.loc[sig, "coef"] / out.loc[sig, "outcome"].map(base)
    return out


def main(data_dir: Path, out_dir: Path):
    contracts = load_all(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    qa = calendar_qa(contracts)
    qa.to_csv(out_dir / "expiry_calendar_qa.csv", index=False)

    df = daily_panel(contracts)
    r = fe_test(df, ["W"])
    oos = panel_oos_r2(df, ["W"])
    w = df.W.astype(bool)
    gate = pd.DataFrame([{
        "contracts": len(FUTURES), "days": r["n"], "months": r["months"], "window_share": float(w.mean()),
        "start": df.date.min().date(), "end": df.date.max().date(),
        "coef_ann": r["W"]["coef"] * 252, "t": r["W"]["t"], "p": r["W"]["p"],
        "ann_in": df.y[w].mean() * 252, "ann_out": df.y[~w].mean() * 252,
        "oos_r2": oos["oos_r2"], "oos_days": oos["days"]}])
    gate["passed"] = (gate.p < 0.05) & (gate.oos_r2 > 0)
    gate.to_csv(out_dir / "expiry_gate.csv", index=False)

    alt = {c: (cal.ALL_MONTHS, cal.SCHEDULES[c][1]) for c in VENDOR_MONTHLY}
    desc = describe(df, daily_panel(contracts, schedules=alt))
    desc.to_csv(out_dir / "expiry_describe.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print("=== calendar check (dates only)")
        print(qa.round(3).to_string(index=False))
        print("\n=== M5-4 gate: daily gross A2 combo P&L on roll-window dummy (+ logvol), FE, clustered by month")
        print(gate.round(4).to_string(index=False))
        print("\ndescriptive (not a gate):")
        print(desc.round(4).to_string(index=False))

    decision = {"passed": bool(gate.passed.iloc[0]), "filter_built": bool(gate.passed.iloc[0]),
                "rule": "p < 0.05 and OOS R2 > 0 (PREREG 4.5)"}
    if decision["passed"]:
        groups = {}
        for code in FUTURES:
            groups.setdefault(contracts[code]["asset"], []).append(code)
        window = lambda idx, code, lead: cal.roll_window(idx, code, HALF_WIDTH, lead)  # noqa: E731
        filt, log = calfilter.build(contracts, groups, window, lead=LEAD)
        filt.to_csv(out_dir / "expiry_filter.csv", index=False)
        log.to_csv(out_dir / "expiry_filter_log.csv", index=False)
        decision["filter_file"] = str(out_dir / "expiry_filter.csv")
        print("OUTPUT", out_dir / "expiry_filter.csv")
    else:
        print("decision: gate not passed; no filter built (method = identity filter, A3 = A2)")
    (out_dir / "expiry_gate.json").write_text(json.dumps(decision, indent=2))
    return gate, decision


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data/final"))
    p.add_argument("--out-dir", type=Path, default=Path("m5_outputs"))
    a = p.parse_args()
    main(a.data_dir, a.out_dir)
