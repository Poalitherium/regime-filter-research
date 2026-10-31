import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from factors import generate_signals
from volatility import volatility_target

idx = pd.bdate_range('2020-01-01', periods=260)
price = pd.Series(100 + np.arange(260) * .15, index=idx)
signals = generate_signals(price)
returns = pd.Series(np.sin(np.arange(260) * 0.7) * .01, index=idx)
out = volatility_target(signals, returns)
assert out['realized_vol'].iloc[:59].isna().all()
assert out['vol_scale'].iloc[59:].le(2).all()
assert out['tsmom_vol_target'].iloc[:60].isna().all()
assert np.allclose(out['tsmom_vol_target'].iloc[110:], out['tsmom_raw'].iloc[110:] * out['vol_scale'].iloc[110:])
# Causality: changing future returns must not change earlier target exposures.
modified = returns.copy()
modified.iloc[180:] *= 100
out2 = volatility_target(signals, modified)
pd.testing.assert_frame_equal(out.iloc[:180], out2.iloc[:180])
# No hidden imputation of missing returns.
missing = returns.copy()
missing.iloc[100] = np.nan
out3 = volatility_target(signals, missing)
assert out3['realized_vol'].iloc[100:160].isna().all()
print('PASS: volatility warm-up, cap, exposure, causality, missing-return checks')
