import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from factors import tsmom, ma_crossover, donchian_breakout, generate_signals

dates = pd.bdate_range('2025-01-01', periods=160)
up = pd.Series(np.arange(1., 161.), index=dates)
down = pd.Series(np.arange(160., 0., -1.), index=dates)
flat = pd.Series(np.ones(160)*100, index=dates)
for fn in (tsmom, ma_crossover, donchian_breakout):
    assert fn(up).dropna().iloc[-1] == 1, fn.__name__
    assert fn(down).dropna().iloc[-1] == -1, fn.__name__
    assert fn(flat).dropna().iloc[-1] == 0, fn.__name__
assert tsmom(up).first_valid_index() == dates[60]
assert ma_crossover(up).first_valid_index() == dates[99]
assert donchian_breakout(up).first_valid_index() == dates[55]
# Causality: perturb future values; earlier signals must be unchanged.
base = generate_signals(up)
changed = up.copy()
changed.iloc[120:] *= -3
new = generate_signals(changed)
pd.testing.assert_frame_equal(base.iloc[:120], new.iloc[:120])
# Missing close should never create a fresh signal that day.
missing = up.copy(); missing.iloc[130] = np.nan
assert generate_signals(missing).iloc[130].isna().all()
print('PASS: trend direction, warm-up, causality, missing-date tests')
print('Last signals (rising series):', base.iloc[-1].to_dict())
