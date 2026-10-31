"""Run the HMM regime filter on data/final/{CODE}.csv for all 13 contracts.

    python -m m3.run_real_data --n-states 3 --refit quarter --mapping top
    python -m m3.run_real_data --n-states 3 --refit quarter --offmarket     # + macro/news
    python -m m3.run_real_data --n-states 3 --refit quarter --data-dir data/eras --codes SPX,NKY

`--codes` (default: the 13 contracts of data/final) runs another set of {CODE}.csv files
with the same settings; the default output is unchanged.

Writes m3_outputs/hmm_k{K}_{refit}_{mapping}[_offmkt]_filter.csv (one row per date x
contract) and ..._fits.csv (one row per contract x refit).
"""

import argparse
import time
from pathlib import Path

import pandas as pd

from .adaptor import AdaptorConfig, run_contract
from .features import DEFAULT_FEATURES, OFFMARKET_FEATURES, load_exog

CODES = ("ES", "FESX", "NK", "CL", "NG", "GC", "SI", "ZN", "FGBL", "6E", "6J", "ZC", "ZS")
EXOG_FILE = Path("data/processed/offmarket/offmarket_daily.csv")


def run(data_dir: Path, out_dir: Path, cfg: AdaptorConfig, exog_file: Path = EXOG_FILE, tag: str = "",
        codes: tuple[str, ...] = CODES):
    exog = load_exog(exog_file) if any(f.startswith("x:") for f in cfg.features) else None
    filters, fits = [], []
    for code in codes:
        t0 = time.perf_counter()
        d = pd.read_csv(data_dir / f"{code}.csv", index_col="date", parse_dates=True)
        out, recs = run_contract(d["ret"], cfg, exog)
        out = out.rename(columns=lambda c: c if c == "vintage" else f"{c}_close_t")
        filters.append(out.assign(contract=code).reset_index())
        fits += [{"contract": code, **r} for r in recs]
        first = out["filter_close_t"].first_valid_index()
        print(f"{code:4s} first filter {first.date() if first is not None else '-'}  "
              f"refits={len(recs):2d}  {time.perf_counter() - t0:5.1f}s")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"hmm_k{cfg.n_states}_{cfg.refit}_{cfg.mapping}" + (f"_{tag}" if tag else "")
    filt = pd.concat(filters, ignore_index=True)
    filt = filt[["date", "contract"] + [c for c in filt.columns if c not in ("date", "contract")]]
    filt.to_csv(out_dir / f"{stem}_filter.csv", index=False)
    pd.DataFrame(fits).to_csv(out_dir / f"{stem}_fits.csv", index=False)
    print("OUTPUT", out_dir / f"{stem}_filter.csv")
    return filt, pd.DataFrame(fits)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data/final"))
    p.add_argument("--out-dir", type=Path, default=Path("m3_outputs"))
    p.add_argument("--n-states", type=int, default=2)
    p.add_argument("--refit", choices=["year", "quarter", "month"], default="year")
    p.add_argument("--mapping", choices=["top", "linear"], default="top")
    p.add_argument("--offmarket", action="store_true", help="add the macro/news features " + ", ".join(OFFMARKET_FEATURES))
    p.add_argument("--exog", type=Path, default=EXOG_FILE)
    p.add_argument("--codes", default=None, help="comma-separated contract codes (default: the 13 of data/final)")
    a = p.parse_args()
    feats = DEFAULT_FEATURES + (OFFMARKET_FEATURES if a.offmarket else ())
    cfg = AdaptorConfig(n_states=a.n_states, refit=a.refit, mapping=a.mapping, features=feats)
    codes = tuple(a.codes.split(",")) if a.codes else CODES
    run(a.data_dir, a.out_dir, cfg, a.exog, "offmkt" if a.offmarket else "", codes)
