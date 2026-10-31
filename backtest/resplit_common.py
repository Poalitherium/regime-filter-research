"""Fixed design for the user-requested post-hoc 80/20 comparison."""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'backtest_outputs/resplit_80_20'
START='2007-07-02'
DEV_END='2022-12-05'
TEST_START='2022-12-06'
END='2026-10-15'
REUSED={
 'M3-1':('HMM','m3_outputs/hmm_k3_quarter_top_filter.csv'),
 'M3-2':('HMM + macro/news','m3_outputs/hmm_k3_quarter_top_offmkt_filter.csv'),
 'M3-3':('BOCPD','m3_outputs/bocpd_h250_filter.csv'),
 'M3-4':('Portfolio volatility targeting','m3_outputs/portvol_filter.csv'),
 'M5-1':('Class-pooled HMM','m5_outputs/pool_class_k3_quarter_top_filter.csv'),
 'M5-2':('Global-pooled HMM','m5_outputs/pool_global_k3_quarter_top_filter.csv'),
}
NAMES={**{k:v[0] for k,v in REUSED.items()},'M3-6':'Meta-labeling','M3-7':'Signature meta-labeling',
 'M4-1':'FOMC calendar','M4-2':'VIX stress × risk-on','M4-3':'COT crowding',
 'M4-4':'Implied / realised volatility','M4-5':'VIX curve inversion',
 'M5-3':'Seasonality','M5-4':'Contract expiry'}

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write_spec():
 OUT.mkdir(parents=True,exist_ok=True)
 if (OUT/'analysis_spec.json').exists():return
 files=list((ROOT/'data/final').glob('*.csv'))+[ROOT/p for _,p in REUSED.values()]
 files += [ROOT/'data/processed/offmarket/offmarket_daily.csv',ROOT/'m1/config/fee_schedule.json']
 spec=dict(created_utc=datetime.now(timezone.utc).isoformat(),status='post-hoc 80/20 sensitivity analysis; previously viewed test years',
  split=dict(start=START,development_end=DEV_END,test_start=TEST_START,end=END,development_days=3984,test_days=997,
   rule='floor(0.8*N) of the union of trading dates after every included learned filter and all 13 contracts are warmed up'),
  universe='same 13 contracts and fixed 1/13 weights for every method and leg',
  approaches={'A2':'M2 combo × volatility target scale','R':'M2 combo × same filter','A3':'M2 combo × volatility target scale × same filter'},
  parameters='existing hyperparameters and causal annual/quarterly/online update schedules unchanged; no optimization',
  fitting='past data only at each refit, including past test observations after they become available; no future labels',
  gates='recompute original gates using pre-test data only; freeze gate decisions before test; no failed-gate reactivation',
  development='descriptive, because full development data determine gates; earlier data retained for training and warm-up',
  costs=['research','official'],cost_caveat='2026 fee snapshot scenario excludes spread/slippage; fixed bps are assumptions',
  inference=dict(block=20,reps=5000,seed=0,sided='two',comparisons=['A3-A2','R-A2','R-A3'],
   ci='unadjusted 95% percentile',multiplicity='Holm across 45 overlay comparisons per cost and 90 across both costs, separately by period; 14 regime rules and one portfolio-volatility benchmark'),
  extra_strategies='M3-5 and M3-8 change signal direction, so rerun separately with their original controls and M2 A2; not mislabeled multiplicative filters',
  bookkeeping='each evaluation window starts flat, delay 1, M1 accounting, no ex-post volatility normalization',
  prior_evidence='Original 2014 split and BOCPD confirmation retained separately; this split does not restore untouched data',
  input_sha256={str(p.relative_to(ROOT)).replace('\\','/'):digest(p) for p in files})
 (OUT/'analysis_spec.json').write_text(json.dumps(spec,indent=2),encoding='utf-8')

if __name__=='__main__':write_spec()
