"""Trial M5-3: seasonality (PREREG 4.4). Gate test first; a filter only if it passes.

    python -m m5.seasonality

Gate, in the style of m3/predictability.py (development window only):
    sample   : each contract's calendar months, from its first A2 combo target to 2014-12-31
    target   : next month's A2 combo net return (delay 1, class costs)
    predictor: S = 1[next month is in season] (known in advance), control: logvol20 at month end
    tests    : agri  = ZC + ZS panel, contract fixed effects, SEs clustered by month
               energy = NG time series, Newey-West (3 lags)
    pass     : Holm-adjusted p < 0.05 across the two tests AND out-of-sample R2 > 0
               (expanding window from 2008-01 against a fixed-effects + vol benchmark)

Writes m5_outputs/season_gate.csv, season_gate.json, season_describe.csv and, for a
group that passes, season_filter.csv (+ season_filter_log.csv).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from m3.predictability import OOS_START, holm, oos_r2, run_tests

from . import calfilter
from .calendar import SEASONS, in_season
from .common import DEV_END, LEAD, load_all

GROUPS = {"agri": ["ZC", "ZS"], "energy": ["NG"]}
HORIZON = {"ann": 12, "nw_lags": 3}


def monthly_sample(c: dict, code: str, dev_end: str = DEV_END) -> pd.DataFrame:
    """Month-end rows: next month's A2 net return, logvol20 at the month end, season flag."""
    first = c["target"].first_valid_index()
    net = c["net"].loc[:dev_end]
    net = net[net.index.to_period("M") > first.to_period("M")]  # whole months with a position
    per = net.index.to_period("M")
    ends = pd.Series(net.index, index=net.index).groupby(per).max()
    y = net.groupby(per).sum().shift(-1)
    nxt = (ends.index + 1).month
    df = pd.DataFrame({"contract": code, "period": ends.index.to_timestamp(), "y": y.to_numpy(),
                       "logvol": c["logvol20"].reindex(ends.to_numpy()).to_numpy(),
                       "season": np.isin(nxt, SEASONS[code]).astype(float)},
                      index=pd.DatetimeIndex(ends.to_numpy()))
    return df.dropna(subset=["y", "logvol"])


def panel_oos_r2(df: pd.DataFrame, family: list, start: str = OOS_START) -> dict:
    """Expanding-window forecasts for a contract panel: FE + logvol (+ family) vs FE + logvol."""
    codes = sorted(df.contract.unique())
    dummies = lambda d: np.column_stack([(d.contract == k).to_numpy(float) for k in codes])  # noqa: E731
    ys, ff, bb = [], [], []
    for e in sorted(df.period[df.period >= pd.Timestamp(start)].unique()):
        tr, te = df[df.period < e], df[df.period == e]
        Xb = np.column_stack([dummies(tr), tr.logvol])
        Xf = np.column_stack([Xb] + [tr[c] for c in family])
        bf = np.linalg.lstsq(Xf, tr.y, rcond=None)[0]
        bb_ = np.linalg.lstsq(Xb, tr.y, rcond=None)[0]
        Tb = np.column_stack([dummies(te), te.logvol])
        Tf = np.column_stack([Tb] + [te[c] for c in family])
        ys += list(te.y)
        ff += list(Tf @ bf)
        bb += list(Tb @ bb_)
    y, f, b = map(np.array, (ys, ff, bb))
    return {"oos_r2": float(1 - ((y - f) ** 2).sum() / ((y - b) ** 2).sum()), "periods": len(y)}


def describe(samples: dict) -> pd.DataFrame:
    """Raw in-season minus out-of-season means (annualised), and the vol ratio (descriptive)."""
    rows = []
    for code, df in samples.items():
        s = df.season.astype(bool)
        rows.append({"contract": code, "months_in": int(s.sum()), "months_out": int((~s).sum()),
                     "ann_ret_in": df.y[s].mean() * 12, "ann_ret_out": df.y[~s].mean() * 12,
                     "diff_ann": (df.y[s].mean() - df.y[~s].mean()) * 12,
                     "vol_ratio_in_out": float(np.exp(df.logvol[s].mean() - df.logvol[~s].mean())),
                     "seasons_in_sample": int(len(set(df.index[s].year)))})
    return pd.DataFrame(rows)


def gate(contracts: dict):
    samples = {code: monthly_sample(contracts[code], code) for codes in GROUPS.values() for code in codes}
    rows = []
    for gname, codes in GROUPS.items():
        df = pd.concat([samples[c] for c in codes])
        panel = len(codes) > 1
        uni, joint = run_tests(df, panel, ["season"], [], HORIZON["ann"], HORIZON["nw_lags"])
        r = uni.loc["season"]
        oos = panel_oos_r2(df, ["season"]) if panel else oos_r2(df, ["season"], HORIZON["nw_lags"])
        s = df.season.astype(bool)
        rows.append({"group": gname, "contracts": "+".join(codes), "months": joint["n"],
                     "start": df.index.min().date(), "end": df.index.max().date(),
                     "t": r.t, "p": r.p, "ann_ret_per_sd": r.ann_ret_per_sd,
                     "raw_diff_ann": (df.y[s].mean() - df.y[~s].mean()) * 12,
                     "oos_r2": oos["oos_r2"], "oos_periods": oos["periods"]})
    res = pd.DataFrame(rows)
    res["p_holm"] = holm(res.p.to_numpy())
    res["passed"] = (res.p_holm < 0.05) & (res.oos_r2 > 0)
    return res, samples


def main(data_dir: Path, out_dir: Path):
    contracts = load_all(data_dir)
    res, samples = gate(contracts)
    out_dir.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_dir / "season_gate.csv", index=False)
    desc = describe(samples)
    desc.to_csv(out_dir / "season_describe.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print("=== M5-3 gate: next-month A2 combo net return on season dummy (+ logvol control), dev window")
        print(res.round(4).to_string(index=False))
        print("\ndescriptive (not a test):")
        print(desc.round(4).to_string(index=False))
    passed = {g: GROUPS[g] for g in res.group[res.passed]}
    decision = {"passed_groups": list(passed), "filter_built": bool(passed),
                "rule": "Holm p < 0.05 and OOS R2 > 0 (PREREG 4.4)"}
    if passed:
        filt, log = calfilter.build(contracts, passed, in_season, lead=LEAD)
        filt.to_csv(out_dir / "season_filter.csv", index=False)
        log.to_csv(out_dir / "season_filter_log.csv", index=False)
        decision["filter_file"] = str(out_dir / "season_filter.csv")
        print("OUTPUT", out_dir / "season_filter.csv")
    else:
        print("decision: no group passed the gate; no filter built (method = identity filter, A3 = A2)")
    (out_dir / "season_gate.json").write_text(json.dumps(decision, indent=2))
    return res, decision


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data/final"))
    p.add_argument("--out-dir", type=Path, default=Path("m5_outputs"))
    a = p.parse_args()
    main(a.data_dir, a.out_dir)
