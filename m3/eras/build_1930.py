"""Build the 1930-2014 US industry dataset (PREREG_1930.md section 2) into data/eras1930/.

    python -m m3.eras.build_1930

BUILD_LOG.md keeps the command the committed data was built with (python -m eras.build_1930).

Ken French's data library, daily:
- 10_Industry_Portfolios_daily_CSV.zip, section "Average Value Weighted Returns -- Daily";
- F-F_Research_Data_Factors_daily_CSV.zip, column RF.
Returns are in percent; the series used is the excess return r - RF (decimals). Rows after
2014-12-31 are dropped as the files are parsed. Missing values (-99.99, -999) are dropped.
No bad tick is removed (PREREG_1930 2.2): days the PREREG rule would flag are only logged.
"""

import hashlib
import io
import json
import re
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .build_data import bad_ticks, price_returns

BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
INDUSTRY_ZIP, FACTORS_ZIP = "10_Industry_Portfolios_daily_CSV.zip", "F-F_Research_Data_Factors_daily_CSV.zip"
VW_TITLE = "Average Value Weighted Returns -- Daily"
START, END = pd.Timestamp("1926-07-01"), pd.Timestamp("2014-12-31")
MISSING = (-99.99, -999.0)
NAMES = {"NoDur": "Consumer nondurables", "Durbl": "Consumer durables", "Manuf": "Manufacturing",
         "Enrgy": "Energy", "HiTec": "Business equipment and technology", "Telcm": "Telecommunications",
         "Shops": "Wholesale and retail", "Hlth": "Healthcare", "Utils": "Utilities", "Other": "Other"}


def _rows(lines: list[str], start: int) -> pd.DataFrame:
    """Header line at `start` (',col1,col2,...'), then YYYYMMDD rows until the first other line."""
    cols = [c.strip() for c in lines[start].split(",")[1:]]
    data = []
    for line in lines[start + 1:]:
        if not re.match(r"\s*\d{8}\s*,", line):
            break
        parts = [p.strip() for p in line.split(",")]
        data.append([pd.Timestamp(parts[0])] + [float(v) for v in parts[1:len(cols) + 1]])
    df = pd.DataFrame(data, columns=["date"] + cols).set_index("date")
    return df.mask(df.isin(MISSING))


def parse_industries(text: str) -> pd.DataFrame:
    lines = text.splitlines()
    i = next(k for k, line in enumerate(lines) if VW_TITLE in line)
    head = next(k for k in range(i + 1, len(lines)) if lines[k].strip().startswith(","))
    return _rows(lines, head)


def parse_factors(text: str) -> pd.DataFrame:
    lines = text.splitlines()
    head = next(k for k, line in enumerate(lines) if line.strip().startswith(",") and "RF" in line)
    return _rows(lines, head)


def excess_returns(ind: pd.DataFrame, rf: pd.Series) -> pd.DataFrame:
    """Decimal excess returns on [START, END]; a day without RF has no excess return."""
    ex = ind.sub(rf.reindex(ind.index), axis=0) / 100
    return ex.loc[START:END]


def fetch(name: str) -> tuple[str, dict]:
    r = requests.get(BASE + name, timeout=300, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    text = z.read(z.namelist()[0]).decode("latin-1")
    return text, {"file": name, "url": BASE + name, "downloaded": str(date.today()),
                  "sha256_zip": hashlib.sha256(r.content).hexdigest()}


def main(out_dir: Path = Path("data/eras1930"), raw_dir: Path = Path("data/raw/eras1930")):
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    ind_text, ind_src = fetch(INDUSTRY_ZIP)
    fac_text, fac_src = fetch(FACTORS_ZIP)
    ind, fac = parse_industries(ind_text), parse_factors(fac_text)
    ind.loc[:END].rename_axis("date").to_csv(raw_dir / "industries_vw_daily_percent.csv")
    fac[["RF"]].loc[:END].rename_axis("date").to_csv(raw_dir / "rf_daily_percent.csv")
    (raw_dir / "source_files.json").write_text(json.dumps([ind_src, fac_src], indent=2))
    ex = excess_returns(ind, fac["RF"])
    infos, flagged = [], []
    for col in ind.columns:
        code = col.upper()
        r = ex[col].dropna()
        level = 100 * (1 + r).cumprod()
        for d in bad_ticks(level, price_returns):  # logged only, never removed
            t = r.index.get_loc(d)
            flagged.append({"code": code, "date": d.date(), "ret": r.iloc[t],
                            "next_ret": r.iloc[t + 1] if t + 1 < len(r) else np.nan})
        ret = r.copy()
        ret.iloc[0] = np.nan
        close = 100 * (1 + ret.fillna(0)).cumprod()
        pd.DataFrame({"close": close, "close_adj": close, "ret": ret, "volume": np.nan, "roll": False,
                      "kind": "proxy"}, index=r.index.rename("date")).to_csv(out_dir / f"{code}.csv")
        infos.append({"code": code, "name": NAMES.get(col, col), "asset_class": "equity", "pool": "industry",
                      "source": "Ken French data library", "series": f"{INDUSTRY_ZIP} [{VW_TITLE}] {col} minus RF",
                      "first": r.index[0].date(), "last": r.index[-1].date(), "rows": len(r),
                      "flagged_not_removed": sum(f["code"] == code for f in flagged),
                      "max_abs_ret": float(r.abs().max())})
        print(f"{code:6s} {r.index[0].date()} .. {r.index[-1].date()}  rows {len(r):6d}  "
              f"flagged (kept) {infos[-1]['flagged_not_removed']}  max |ret| {infos[-1]['max_abs_ret']:.3f}")
    uni = pd.DataFrame(infos)
    uni.to_csv(out_dir / "universe.csv", index=False)
    fl = pd.DataFrame(flagged, columns=["code", "date", "ret", "next_ret"])
    fl.to_csv(out_dir / "flagged_days_kept.csv", index=False)
    days = ex.index.to_series()
    per_year = pd.DataFrame({"trading_days": days.groupby(days.dt.year).size(),
                             "saturdays": days[days.dt.dayofweek == 5].groupby(days[days.dt.dayofweek == 5].dt.year).size()}
                            ).fillna(0).astype(int).rename_axis("year")
    per_year.to_csv(out_dir / "trading_days_per_year.csv")
    write_log(out_dir, uni, fl, per_year, [ind_src, fac_src])
    return uni, fl, per_year


def write_log(out_dir, uni, fl, per_year, sources):
    lines = ["# 1930 study data: build log", "",
             "Built by `python -m eras.build_1930` (PREREG_1930.md section 2). Excess returns of Ken French's",
             "value-weighted daily industry portfolios over the daily RF, 1926-07-01 to 2014-12-31.", "",
             "| Code | Name | First | Last | Rows | Flagged by the bad-tick rule (kept) | Max abs return |",
             "|---|---|---|---|---|---|---|"]
    for r in uni.itertuples():
        lines.append(f"| {r.code} | {r.name} | {r.first} | {r.last} | {r.rows} | {r.flagged_not_removed} | {r.max_abs_ret:.4f} |")
    lines += ["", f"Days the PREREG bad-tick rule would flag: {len(fl)}, all kept (`flagged_days_kept.csv`).", "",
              "Trading days per year (`trading_days_per_year.csv`): "
              f"{per_year.loc[1927:1951, 'trading_days'].min()}-{per_year.loc[1927:1951, 'trading_days'].max()} in 1927-1951 "
              f"(with {per_year.loc[1927:1951, 'saturdays'].min()}-{per_year.loc[1927:1951, 'saturdays'].max()} Saturdays), "
              f"{per_year.loc[1953:2014, 'trading_days'].min()}-{per_year.loc[1953:2014, 'trading_days'].max()} in 1953-2014.", "",
              "Source files (the library is updated monthly; these hashes identify the version used):", ""]
    lines += [f"- `{s['file']}` downloaded {s['downloaded']}, zip SHA-256 `{s['sha256_zip']}`" for s in sources]
    (out_dir / "BUILD_LOG.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
