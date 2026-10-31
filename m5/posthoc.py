"""Post hoc, exploratory analyses (decided after seeing the M5-4 descriptive results).

    python -m m5.posthoc

Not pre-registered, not a trial, never a gate. Logged in the report's trial log so
that M6 can count them.

Roll event profile: for each weekday offset k = -5..+5 from the scheduled switch day
(exchange cycle, 11 futures, development window), the mean daily gross A2 combo P&L and
mean |ret| relative to days outside +-5, with SEs clustered by calendar month. Purpose:
see whether the descriptive post-window effect of M5-4 sits on the switch day itself,
which would point to the estimated roll gaps (a data question for M1) rather than to a
market effect.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from m3.predictability import norm_sf, ols

from . import calendar as cal
from .common import DEV_END, load_all
from .expiry import FUTURES

OFFSETS = range(-5, 6)


def event_profile(contracts: dict, dev_end: str = DEV_END) -> pd.DataFrame:
    rows = []
    for code in FUTURES:
        c = contracts[code]
        live = c["live"].loc[:dev_end]
        idx = live.index[live]
        d = cal.days_to_roll(idx, code)
        off = np.where(d.d_next <= d.d_last, -d.d_next, d.d_last)  # signed distance to the nearest switch
        rows.append(pd.DataFrame({"contract": code, "period": idx.to_period("M").to_timestamp(), "offset": off,
                                  "y": c["gross"].reindex(idx).to_numpy(),
                                  "abs_ret": c["data"].ret.abs().reindex(idx).to_numpy()}))
    df = pd.concat(rows, ignore_index=True).dropna()
    out = []
    for outcome, scale in (("y", 252), ("abs_ret", 1)):
        d = df.copy()
        dummies = [f"k{k}" for k in OFFSETS]
        for k in OFFSETS:
            d[f"k{k}"] = (d.offset == k).astype(float)
        cols = [outcome] + dummies
        d[cols] = d[cols] - d.groupby("contract")[cols].transform("mean")
        b, V, _ = ols(d[outcome].to_numpy(), d[dummies].to_numpy(), groups=d["period"].to_numpy())
        base = df[df.offset.abs() > 5][outcome].mean()
        for j, k in enumerate(OFFSETS):
            t = b[j] / np.sqrt(V[j, j])
            out.append({"outcome": outcome, "offset": k, "n": int((df.offset == k).sum()),
                        "diff": b[j] * scale, "se": np.sqrt(V[j, j]) * scale, "t": t, "p": 2 * norm_sf(abs(t)),
                        "ratio_to_outside": (base + b[j]) / base if outcome == "abs_ret" else np.nan})
    return pd.DataFrame(out)


def main(data_dir: Path, out_dir: Path):
    prof = event_profile(load_all(data_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    prof.to_csv(out_dir / "posthoc_roll_event_profile.csv", index=False)
    with pd.option_context("display.width", 200):
        print("POST HOC (exploratory): offsets from the scheduled switch day vs days outside +-5")
        print(prof.round(4).to_string(index=False))
    return prof


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data/final"))
    p.add_argument("--out-dir", type=Path, default=Path("m5_outputs"))
    a = p.parse_args()
    main(a.data_dir, a.out_dir)
