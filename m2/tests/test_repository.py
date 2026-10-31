"""Repository integration checks: run with python -m unittest discover -s m2/tests."""
import unittest
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from m2.factors import donchian_breakout
from m2.pipeline import build_m2
from m2.run_real_data import run, CODES

class IntegrationTests(unittest.TestCase):
    def test_donchian_missing_resets_state(self):
        dates=pd.date_range('2020-01-01',periods=12)
        close=pd.Series([1,2,3,4,np.nan,3,3,3,3,3,3,3],index=dates)
        sig=donchian_breakout(close,lookback=2)
        self.assertTrue(np.isnan(sig.iloc[4]))
        self.assertEqual(sig.iloc[7],0)
    def test_non_numeric_price_rejected(self):
        df=pd.DataFrame({'date':['2020-01-01'],'contract':['ES'],'close':['bad'],'return':[0.0]})
        with self.assertRaises(TypeError):build_m2(df)
    def test_real_data_all_contracts(self):
        from tempfile import TemporaryDirectory
        data=ROOT/'data'/'final'
        if not data.is_dir():self.skipTest('data/final not available')
        with TemporaryDirectory() as d:
            result, summary=run(data,Path(d)/'signals.csv')
        self.assertEqual(set(summary.contract),set(CODES))
        self.assertEqual(len(result),3*int(summary.days.sum()))
        self.assertTrue(np.isfinite(result['approach2_target_close_t'].dropna()).all())
        self.assertTrue((result['approach2_target_close_t'].abs().dropna()<=2.0+1e-12).all())
    def test_future_data_causality(self):
        dates=pd.date_range('2020-01-01',periods=150)
        df=pd.DataFrame({'date':dates,'contract':'ES','close':100+np.arange(150)*.1,'return':np.sin(np.arange(150))*.01})
        first=build_m2(df)
        df.loc[130:,'close']+=200
        df.loc[130:,'return']+=.1
        second=build_m2(df)
        cols=['signal_close_t','realized_vol_close_t','vol_scale_close_t','approach1_target_close_t','approach2_target_close_t']
        pd.testing.assert_frame_equal(first.loc[first.date<dates[130],cols],second.loc[second.date<dates[130],cols])

if __name__=='__main__':unittest.main()
