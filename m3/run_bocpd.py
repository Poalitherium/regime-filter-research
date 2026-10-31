"""Run BOCPD on data/final/{CODE}.csv and write the pre-registered trial-3 filter.

    python -m m3.run_bocpd                 # hazard 1/250 (pre-registered)
    python -m m3.run_bocpd --run-length 120  # sensitivity only
    python -m m3.run_bocpd --data-dir data/eras --codes SPX,NKY   # another set of {CODE}.csv files

`--codes` defaults to the 13 contracts of data/final; the default output is unchanged.

Filter: f_t = 1 - cp20_t * 1[align_t < 0], with align_t = sign(A2 combo position_t) x seg_mu_t.
It only cuts exposure right after a likely break whose new drift points against the position.

Writes m3_outputs/bocpd_h{L}_filter.csv (date, contract, filter_close_t, cp20_close_t,
seg_mu_close_t, align_close_t) in the same layout as the HMM filter files.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from m2.pipeline import build_m2

from .bocpd import BOCPDConfig, contract_features
from .run_real_data import CODES

FACTORS = ["tsmom", "ma_crossover", "donchian"]


def a2_combo(d: pd.DataFrame, code: str) -> pd.Series:
    m2 = build_m2(pd.DataFrame({"date": d.index, "contract": code,
                                "close": d.close_adj.to_numpy(), "return": d.ret.to_numpy()}))
    a2 = m2.pivot(index="date", columns="factor", values="approach2_target_close_t")
    return a2[FACTORS].mean(axis=1, skipna=False)


def contract_filter(d: pd.DataFrame, code: str, cfg: BOCPDConfig) -> pd.DataFrame:
    f = contract_features(d.ret, cfg)
    pos = a2_combo(d, code).reindex(d.index)
    align = np.sign(pos) * f.seg_mu
    cp = f[f"cp{cfg.k_recent}"]
    filt = (1 - cp * (align < 0)).where(cp.notna() & align.notna())
    return pd.DataFrame({"filter_close_t": filt, f"cp{cfg.k_recent}_close_t": cp,
                         "seg_mu_close_t": f.seg_mu, "align_close_t": align}, index=d.index)


def run(data_dir: Path, out_dir: Path, cfg: BOCPDConfig, codes: tuple[str, ...] = CODES):
    rows = []
    for code in codes:
        t0 = time.perf_counter()
        d = pd.read_csv(data_dir / f"{code}.csv", index_col="date", parse_dates=True)
        out = contract_filter(d, code, cfg)
        rows.append(out.assign(contract=code).rename_axis("date").reset_index())
        f = out.filter_close_t.dropna()
        print(f"{code:4s} first {f.index[0].date()}  mean filter {f.mean():.3f}  "
              f"days cut {(f < 1).mean():.1%}  {time.perf_counter() - t0:4.1f}s")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"bocpd_h{round(1 / cfg.hazard)}_filter.csv"
    res = pd.concat(rows, ignore_index=True)
    res[["date", "contract"] + [c for c in res.columns if c not in ("date", "contract")]].to_csv(path, index=False)
    print("OUTPUT", path)
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data/final"))
    p.add_argument("--out-dir", type=Path, default=Path("m3_outputs"))
    p.add_argument("--run-length", type=float, default=250, help="expected segment length = 1 / hazard")
    p.add_argument("--codes", default=None, help="comma-separated contract codes (default: the 13 of data/final)")
    a = p.parse_args()
    codes = tuple(a.codes.split(",")) if a.codes else CODES
    run(a.data_dir, a.out_dir, BOCPDConfig(hazard=1 / a.run_length), codes)
