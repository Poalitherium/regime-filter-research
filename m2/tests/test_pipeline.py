import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from pipeline import build_m2

rng = np.random.default_rng(20261015)
dates = pd.bdate_range('2020-01-01', periods=240)
rows = []
for contract, drift in [('ES', .0005), ('CL', -.0002)]:
    returns = rng.normal(drift, .012, len(dates))
    close = 100 * np.cumprod(1 + returns)
    rows.append(pd.DataFrame({'date': dates, 'contract': contract,
                              'close': close, 'return': returns}))
data = pd.concat(rows, ignore_index=True)
result = build_m2(data)
assert len(result) == 240 * 2 * 3
assert set(result['factor']) == {'tsmom', 'ma_crossover', 'donchian'}
assert result.groupby(['date', 'contract', 'factor']).size().eq(1).all()
assert result.loc[result['date'] == dates[0], 'approach2_target_close_t'].isna().all()
valid = result.dropna(subset=['approach2_target_close_t'])
assert np.allclose(valid['approach2_target_close_t'],
                   valid['approach1_target_close_t'] * valid['vol_scale_close_t'])
assert valid['vol_scale_close_t'].le(2).all()
# Causality: changing all future data must not alter earlier output.
changed = data.copy()
mask = changed['date'] > dates[175]
changed.loc[mask, 'close'] *= 1.4
changed.loc[mask, 'return'] = .03
old = result[result['date'] <= dates[175]].reset_index(drop=True)
new = build_m2(changed)
new = new[new['date'] <= dates[175]].reset_index(drop=True)
pd.testing.assert_frame_equal(old, new)
# Input order independence
pd.testing.assert_frame_equal(result, build_m2(data.sample(frac=1, random_state=11)))
# Duplicates are rejected
try:
    build_m2(pd.concat([data, data.iloc[[0]]]))
    raise AssertionError('duplicate accepted')
except ValueError:
    pass
print('PASS: integration, 2 contracts x 3 factors, no future leakage, order, duplicates')
print('Rows:', len(result), 'valid approach2:', len(valid))
