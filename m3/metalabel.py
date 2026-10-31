"""Trials 6-7: meta-labeling with walk-forward gradient boosting (Lopez de Prado).

Each sample is a contract x week (Friday close). Label: the contract's next-21-day A2
combo net P&L > 0, i.e. "will the trend position make money?". A classifier trained
only on past samples gives p; the filter is f = 1 if p >= 0.5 else 2p, held for the week.

Walk-forward: refit each January from 2006 on samples whose label window ended at
least 5 business days before the refit date (purge + embargo). The original CLI
still ends at 2014-12-31. Explicit end/refit arguments support the separately
recorded post-hoc 80/20 rerun without changing that historical default.

    python -m m3.metalabel --features hand --model gbm    # trial 6 (primary)
    python -m m3.metalabel --features hand --model logit  # trial 6 secondary
    python -m m3.metalabel --features sig --model gbm     # trial 7
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import signature
from .bocpd import standardise
from .features import feature
from .predictability import MACRO_COLS, asof, build, market_wide

DEV_END = pd.Timestamp("2014-12-31")
HORIZON, EMBARGO, FIRST_REFIT, LAST_REFIT = 21, 5, 2006, 2014
HAND = ["logvol20", "er20", "er60", "agree", "mom63", "cp20", "align", "corr60", "disp60"] + MACRO_COLS
SIG = ["logvol20"] + [f"sig_{n}" for n in signature.NAMES if n not in signature.CONSTANT]


def build_panel(data_dir: Path, exog_file: Path, bocpd_file: Path, end=DEV_END) -> pd.DataFrame:
    contracts, book, port, macro = build(data_dir, exog_file, bocpd_file, end=str(end))
    days = port.index
    ends = pd.DatetimeIndex(pd.Series(days, index=days).groupby(days.to_period("W-FRI")).max().values)
    rets = pd.DataFrame({k: c["ret"] for k, c in contracts.items()}).reindex(book.index).fillna(0.0)
    mkt = market_wide(rets, book, ends).rename(columns={"corr": "corr60", "disp": "disp60"})
    mac = asof(macro, ends)
    rows = []
    for code, c in contracts.items():
        ret, net = c["ret"], c["pnl"].net
        f = pd.DataFrame({
            "logvol20": c["feat"].logvol20, "er60": c["feat"].er60, "er20": feature(ret, "er20"),
            "agree": c["signals"].mean(axis=1, skipna=False).abs().rolling(20, min_periods=20).mean(),
            "mom63": net.rolling(63, min_periods=63).sum(),
            "cp20": c["feat"]["cp20"], "align": c["feat"]["align"],
        }).join(signature.features(standardise(ret), 60))
        # label: sum of the next HORIZON trading days' net P&L on the contract's own calendar
        fwd = net[::-1].rolling(HORIZON, min_periods=HORIZON).sum()[::-1].shift(-1)
        label_end = pd.Series(net.index, index=net.index).shift(-HORIZON)
        inb = book[code].reindex(ends).to_numpy()
        e = ends[inb]
        p = asof(f, e).assign(contract=code, week=e,
                              fwd=asof(fwd.to_frame("v"), e)["v"].values,
                              label_end=asof(label_end.to_frame("v"), e)["v"].values)
        rows.append(p.join(mkt).join(mac))
    panel = pd.concat(rows)
    panel.loc[panel.label_end > pd.Timestamp(end), ["fwd", "label_end"]] = np.nan
    panel["y"] = (panel.fwd > 0).astype(float).where(panel.fwd.notna())
    return panel


def make_model(kind: str):
    if kind == "gbm":
        return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                              min_samples_leaf=100, l2_regularization=1.0, random_state=0)
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))


def walk_forward(panel: pd.DataFrame, cols: list, kind: str, first_refit=FIRST_REFIT, last_refit=LAST_REFIT):
    preds, aucs = [], []
    for y in range(first_refit, last_refit + 1):
        refit = pd.Timestamp(y, 1, 1)
        tr = panel[(panel.label_end < refit - BDay(EMBARGO)) & panel.y.notna()]
        te = panel[(panel.week >= refit) & (panel.week < pd.Timestamp(y + 1, 1, 1))]
        if len(tr) < 500 or te.empty:
            continue
        model = make_model(kind).fit(tr[cols], tr.y)
        p = model.predict_proba(te[cols])[:, 1]
        preds.append(te[["contract", "week"]].assign(p=p, y=te.y.values))
        lab = te.y.notna().to_numpy()
        if lab.sum() > 50 and len(np.unique(te.y[lab])) == 2:
            aucs.append({"year": y, "n_train": len(tr), "n_test": int(lab.sum()), "auc": roc_auc_score(te.y[lab], p[lab]),
                         "mean_p": float(p.mean()), "share_cut": float((p < 0.5).mean())})
    return pd.concat(preds), pd.DataFrame(aucs)


def write_filter(preds: pd.DataFrame, data_dir: Path, out: Path):
    preds = preds.assign(f=np.where(preds.p >= 0.5, 1.0, 2 * preds.p))
    rows = []
    for code, g in preds.groupby("contract"):
        d = pd.read_csv(data_dir / f"{code}.csv", index_col="date", parse_dates=True)
        f = g.set_index("week").f
        daily = f.reindex(f.index.union(d.index)).ffill(limit=10).reindex(d.index)
        rows.append(pd.DataFrame({"date": d.index, "contract": code, "filter_close_t": daily.values}))
    pd.concat(rows, ignore_index=True).to_csv(out, index=False)


def main(features: str, kind: str, data_dir: Path, exog: Path, bocpd: Path, out_dir: Path):
    panel = build_panel(data_dir, exog, bocpd)
    cols = HAND if features == "hand" else SIG
    preds, aucs = walk_forward(panel, cols, kind)
    tag = f"meta_{features}_{kind}"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_filter(preds, data_dir, out_dir / f"{tag}_filter.csv")
    aucs.to_csv(out_dir / f"{tag}_auc.csv", index=False)
    lab = preds.y.notna()
    print(f"{tag}: panel {len(panel)} rows, labelled {int(panel.y.notna().sum())}, base rate {panel.y.mean():.3f}")
    print(aucs.round(3).to_string(index=False))
    print(f"pooled walk-forward AUC {roc_auc_score(preds.y[lab], preds.p[lab]):.3f}  "
          f"(n={int(lab.sum())}); share of weeks cut {float((preds.p < 0.5).mean()):.1%}")
    print("OUTPUT", out_dir / f"{tag}_filter.csv")
    return panel, preds, aucs


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--features", choices=["hand", "sig"], default="hand")
    a.add_argument("--model", choices=["gbm", "logit"], default="gbm")
    a.add_argument("--data-dir", type=Path, default=Path("data/final"))
    a.add_argument("--exog", type=Path, default=Path("data/processed/offmarket/offmarket_daily.csv"))
    a.add_argument("--bocpd", type=Path, default=Path("m3_outputs/bocpd_h250_filter.csv"))
    a.add_argument("--out-dir", type=Path, default=Path("m3_outputs"))
    p = a.parse_args()
    main(p.features, p.model, p.data_dir, p.exog, p.bocpd, p.out_dir)
