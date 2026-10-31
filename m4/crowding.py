"""M4-3: speculative crowding from CFTC COT (PREREG_M4-3.md). Gate, descriptives, filter.

    python -m m4.crowding          # after python -m m4.gates (reads its gate p-values for Holm)

Writes m4_outputs/crowding_gate.csv, crowding_describe.csv, crowding_filter.csv (+ _log),
crowding_weekly.csv (usable dates and percentiles), and adds M4-3 to gates.json.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from . import cot
from .common import DEV_END, LEAD, fe_test, holm, load_all, panel_oos_r2, proportional_filter

CODES = list(cot.CFTC_CODES)


def states(contracts: dict, feat: pd.DataFrame, hi: float = cot.HI, lo: float = cot.LO) -> dict:
    """Decision-day crowding state C_t per contract (NaN where the percentile is missing)."""
    out = {}
    for code in CODES:
        c = contracts[code]
        pct = cot.daily_pct(feat, code, c["data"].index)
        out[code] = pd.DataFrame({"pct": pct, "C": cot.crowded(c["target"], pct, hi, lo)})
    return out


def panel(contracts: dict, st: dict, dev_end: str = DEV_END) -> pd.DataFrame:
    rows = []
    for code, s in st.items():
        c = contracts[code]
        live = c["live"].loc[:dev_end]
        idx = live.index[live]
        lag = s.shift(LEAD).reindex(idx)
        rows.append(pd.DataFrame({
            "contract": code, "asset": c["asset"], "date": idx, "period": idx.to_period("M").to_timestamp(),
            "y": c["gross"].reindex(idx).to_numpy(), "logvol": c["logvol20"].shift(LEAD).reindex(idx).to_numpy(),
            "C": lag.C.to_numpy()}))
    return pd.concat(rows, ignore_index=True).dropna(subset=["y", "logvol", "C"])


def main(data_dir: Path, out_dir: Path):
    contracts = load_all(data_dir)
    raw = cot.load()
    feat = cot.weekly_features(raw, "nc")
    feat.to_csv(out_dir / "crowding_weekly.csv", index=False)
    st = states(contracts, feat)
    df = panel(contracts, st)
    r = fe_test(df, ["C"])
    o = panel_oos_r2(df, ["logvol"], ["C"])
    c = df.C.astype(bool)
    prior = pd.read_csv(out_dir / "gates.csv").set_index("trial").p  # M4-1, M4-2 (PREREG_M4-3 section 4)
    p_all = holm([prior["M4-1"], prior["M4-2"], r["C"]["p"]])
    gate = pd.DataFrame([{
        "trial": "M4-3", "name": "COT speculative crowding", "term": "C", "contracts": df.contract.nunique(),
        "days": r["n"], "months": r["months"], "state_share": float(c.mean()),
        "start": df.date.min().date(), "end": df.date.max().date(),
        "coef_ann": r["C"]["coef"] * 252, "t": r["C"]["t"], "p": r["C"]["p"],
        "ann_in": df.y[c].mean() * 252, "ann_out": df.y[~c].mean() * 252,
        "oos_r2": o["oos_r2"], "oos_days": o["days"], "p_holm_3": p_all[2]}])
    gate["passed"] = (gate.p_holm_3 < 0.05) & (gate.oos_r2 > 0)
    gate.to_csv(out_dir / "crowding_gate.csv", index=False)

    # descriptive (not gates)
    rows = []

    def add(label, d, x="C"):
        rr = fe_test(d, [x])
        rows.append({"analysis": label, "n": rr["n"], "state_share": float(d[x].mean()),
                     "coef_ann": rr[x]["coef"] * 252, "t": rr[x]["t"], "p": rr[x]["p"]})

    add("thresholds 0.80 / 0.20", panel(contracts, states(contracts, feat, 0.80, 0.20)))
    for asset, sub in df.groupby("asset", sort=False):
        add(f"class {asset}", sub)
    add("hedgers (commercial) net instead of speculators", panel(contracts, states(contracts, cot.weekly_features(raw, "comm"))))
    desc = pd.DataFrame(rows)
    desc.to_csv(out_dir / "crowding_describe.csv", index=False)

    filt, log = proportional_filter(
        contracts, CODES,
        pnl_state={k: s.C.shift(LEAD).dropna().astype(bool) for k, s in st.items()},
        filter_state={k: s.C.fillna(0).astype(bool) for k, s in st.items()}, dev_end=DEV_END)
    filt.to_csv(out_dir / "crowding_filter.csv", index=False)
    log.to_csv(out_dir / "crowding_filter_log.csv", index=False)

    g = gate.iloc[0]
    gj = json.loads((out_dir / "gates.json").read_text())
    gj["M4-3"] = {"passed": bool(g.passed), "p": float(g.p), "p_holm": float(g.p_holm_3), "oos_r2": float(g.oos_r2),
                  "filter_file": str(out_dir / "crowding_filter.csv"), "role": "primary" if g.passed else "exploratory"}
    gj["rule_M4-3"] = "Holm over the 3 M4 gates p < 0.05 and panel OOS R2 > 0 (PREREG_M4-3 section 4)"
    (out_dir / "gates.json").write_text(json.dumps(gj, indent=2, default=float))
    with pd.option_context("display.width", 240, "display.max_columns", 40):
        print("=== M4-3 gate"); print(gate.round(4).to_string(index=False))
        print("\ndescriptive:"); print(desc.round(4).to_string(index=False))
        print("\nfilter, every 8th vintage:"); print(log.iloc[::8].round(3).to_string(index=False))
    return gate, desc


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data/final"))
    p.add_argument("--out-dir", type=Path, default=Path("m4_outputs"))
    a = p.parse_args()
    main(a.data_dir, a.out_dir)
