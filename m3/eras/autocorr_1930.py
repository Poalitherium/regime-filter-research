"""Post-hoc, descriptive: first-order autocorrelation of the 10 industries' daily excess
returns by era (REPORT_1930.md section 6). Not in PREREG_1930.md; no strategy is run.

    python -m m3.eras.autocorr_1930

Writes m3/eras/results_1930/autocorr_posthoc.csv (era, code, days, rho1; code "mean" is the
average over the 10 industries).
"""

from pathlib import Path

import pandas as pd

DATA, OUT = Path("data/eras1930"), Path("m3/eras/results_1930/autocorr_posthoc.csv")
ERAS = {"E0": ("1930-01-01", "1959-12-31"), "E1": ("1960-01-01", "1976-12-31"), "I": ("1977-01-01", "1989-12-31"),
        "II": ("1990-01-01", "1999-12-31"), "III": ("2000-01-01", "2014-12-31")}


def main():
    codes = pd.read_csv(DATA / "universe.csv").code
    rets = {c: pd.read_csv(DATA / f"{c}.csv", index_col="date", parse_dates=True)["ret"] for c in codes}
    rows = []
    for era, (lo, hi) in ERAS.items():
        era_rows = [{"era": era, "code": c, "days": int(r.loc[lo:hi].count()), "rho1": r.loc[lo:hi].autocorr(1)}
                    for c, r in rets.items()]
        rows += era_rows
        rows.append({"era": era, "code": "mean", "days": era_rows[0]["days"],
                     "rho1": sum(x["rho1"] for x in era_rows) / len(era_rows)})
    t = pd.DataFrame(rows)
    t.to_csv(OUT, index=False)
    print(t.pivot(index="code", columns="era", values="rho1")[list(ERAS)].round(3).to_string())
    print("wrote", OUT)


if __name__ == "__main__":
    main()
