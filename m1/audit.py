"""Validate final data and export a provenance manifest; never rewrite inputs."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd


def audit(data_dir, out_dir):
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    uni = pd.read_csv(data_dir/'universe.csv')
    if uni.code.duplicated().any():
        raise ValueError('duplicate universe codes')
    panels = {f:pd.read_csv(data_dir/f'{f}.csv', index_col='date', parse_dates=True) for f in ['ret','close_adj']}
    rows, inputs, hashes = [], [], []
    for code in uni.code:
        p = data_dir/f'{code}.csv'
        d = pd.read_csv(p, index_col='date', parse_dates=True)
        if d.index.has_duplicates or not d.index.is_monotonic_increasing or d.index.isna().any():
            raise ValueError(f'{code}: invalid dates')
        if not {'close','close_adj','ret','volume','roll','kind'}.issubset(d):
            raise ValueError(f'{code}: required fields missing')
        if not np.isfinite(d[['close','close_adj']].to_numpy()).all():
            raise ValueError(f'{code}: missing/nonfinite prices')
        if not np.isfinite(d.ret.iloc[1:]).all():
            raise ValueError(f'{code}: missing/nonfinite returns after initial row')
        if not d.roll.isin([True,False]).all():
            raise ValueError(f'{code}: invalid roll flags')
        row = dict(code=code, rows=len(d), start=str(d.index.min().date()), end=str(d.index.max().date()),
            missing_ret=int(d.ret.isna().sum()), nonpositive_adjusted=int(d.close_adj.le(0).sum()),
            estimated_rolls=int(d.roll.sum()), max_abs_return=float(d.ret.abs().max()),
            max_calendar_gap=int(d.index.to_series().diff().dt.days.max()))
        for f, panel in panels.items():
            if panel.index.has_duplicates or code not in panel:
                raise ValueError(f'{f}: invalid panel')
            actual = panel[code].reindex(d.index)
            mismatch = ~np.isclose(actual,d[f],rtol=1e-9,atol=1e-12,equal_nan=True)
            extra = panel[code].dropna().index.difference(d.index)
            row[f'{f}_panel_mismatches'] = int(mismatch.sum()+len(extra))
            if row[f'{f}_panel_mismatches']:
                raise ValueError(f'{code}: {f} panel differs')
        rows.append(row)
        inputs.append(pd.DataFrame({'date':d.index,'contract':code,'close':d.close_adj.to_numpy(),'return':d.ret.to_numpy()}))
    for p in sorted(data_dir.glob('*.csv')):
        hashes.append(dict(file=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    result = pd.DataFrame(rows).merge(uni[['code','asset_class','kind','reliability','source','note']],on='code')
    result.to_csv(out_dir/'data_audit.csv',index=False)
    pd.DataFrame(hashes).to_csv(out_dir/'input_sha256.csv',index=False)
    pd.concat(inputs).sort_values(['date','contract']).to_csv(out_dir/'standardized_daily.csv',index=False)
    return result
