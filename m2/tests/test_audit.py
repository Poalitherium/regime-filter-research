import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
"""Additional deterministic boundary/contract checks. No real market evidence."""
import numpy as np
import pandas as pd
from factors import generate_signals, donchian_breakout
from volatility import volatility_target
from pipeline import build_m2

idx = pd.bdate_range('2024-01-01', periods=240)
prices = pd.Series(np.arange(1., 241.), index=idx)
# Missing close resets Donchian state. Following valid day cannot reuse stale position.
gap = prices.copy(); gap.iloc[170] = np.nan
sig = donchian_breakout(gap)
assert np.isnan(sig.iloc[170])
assert sig.iloc[171:226].isna().all(), 'full 55 prior closes required after gap'
assert sig.iloc[226] == 1, 'new breakout requires fresh full lookback'
# No forward fill for rolling MA and momentum on a gap.
all_s = generate_signals(gap)
assert all_s.loc[idx[170]].isna().all()
assert all_s.loc[idx[171], 'ma_crossover'] != all_s.loc[idx[171], 'ma_crossover']
# Extreme but finite return shock: no overflow; cap honored.
ret = pd.Series(np.sin(np.arange(len(idx))) * 0.005, index=idx)
ret.iloc[200] = 4.0
vol = volatility_target(generate_signals(prices), ret)
assert vol['vol_scale'].dropna().between(0, 2).all()
assert np.isfinite(vol['realized_vol'].dropna()).all()
# Flat returns produce zero realized vol, floor prevents divide by zero.
zero = volatility_target(generate_signals(prices), pd.Series(0.0, index=idx))
assert zero['vol_scale'].dropna().eq(2).all()
# Missing returns trigger unavailable risk scale until a complete window returns.
r_missing = ret.copy(); r_missing.iloc[170] = np.nan
v_missing = volatility_target(generate_signals(prices), r_missing)
assert v_missing['vol_scale'].iloc[170:230].isna().all()
# Multiple contracts with non-overlapping calendars are processed independently.
rows = []
for name, sl in [('ES', slice(None)), ('CL', slice(5, None, 2))]:
    ix = idx[sl]
    rows.append(pd.DataFrame({'date':ix, 'contract':name,
                              'close':np.arange(1.,len(ix)+1),
                              'return':np.sin(np.arange(len(ix))) * .01}))
df = pd.concat(rows, ignore_index=True)
out = build_m2(df)
assert len(out) == 3 * len(df)
assert out.groupby(['contract', 'factor']).size().loc[('ES','tsmom')] == 240
# Reject invalid input and contract IDs.
for bad in [pd.concat([df,df.iloc[[0]]]), df.assign(contract=' ')]:
    try:
        build_m2(bad)
    except ValueError:
        pass
    else:
        raise AssertionError('invalid input accepted')
# Future perturbation does not affect prior outputs, including gap behavior.
changed = gap.copy(); changed.iloc[200:] *= 100
pd.testing.assert_frame_equal(generate_signals(gap).iloc[:200], generate_signals(changed).iloc[:200])
print('PASS: gap reset, volatility extremes, calendar independence, invalid inputs, causality')
