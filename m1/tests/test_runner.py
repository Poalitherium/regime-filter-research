import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
import pandas as pd
from m1.run_backtest import run

class RunnerTests(unittest.TestCase):
    def fixture(self,p):
        days=pd.date_range('2020-01-01',periods=5)
        pd.DataFrame({'date':days,'contract':'X','return':[0,.01,.01,.01,.01],'roll':False}).to_csv(p/'market.csv',index=False)
        pd.DataFrame(dict(date=days,contract='X',target=[1,1,-1,-1,0])).to_csv(p/'targets.csv',index=False)
    def call(self,p,**kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return run(p/'targets.csv',p/'out',market_file=p/'market.csv',fee_bps=10,**kwargs)
    def test_independent_run_and_scenario(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);self.fixture(p);self.call(p)
            d=pd.read_csv(p/'out/product_ledger.csv')
            self.assertEqual(d.fee_cost.iloc[1],.001)
            self.assertEqual(d.gross.iloc[1],0)
            self.assertEqual(d.gross.iloc[2],.01)
            self.assertEqual(json.loads((p/'out/run_config.json').read_text())['slippage_scenario'],'ZERO_SLIPPAGE_ASSUMPTION')
    def test_same_close_initial_fee_included(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);self.fixture(p);self.call(p,delay=0)
            d=pd.read_csv(p/'out/daily_portfolio.csv')
            self.assertEqual(d.net.iloc[0],-.001)
    def test_multiple_external_strategies(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);self.fixture(p)
            t=pd.read_csv(p/'targets.csv')
            pd.concat([t.assign(strategy='one'),t.assign(strategy='two')]).to_csv(p/'targets.csv',index=False)
            self.call(p)
            d=pd.read_csv(p/'out/daily_portfolio.csv')
            self.assertEqual(d.loc[d.strategy=='one','net'].tolist(),d.loc[d.strategy=='two','net'].tolist())
    def test_bad_date_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);self.fixture(p)
            t=pd.read_csv(p/'targets.csv');t.loc[0,'date']='2019-12-31';t.to_csv(p/'targets.csv',index=False)
            with self.assertRaises(ValueError):self.call(p)
    def test_duplicate_target_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);self.fixture(p)
            t=pd.read_csv(p/'targets.csv');pd.concat([t,t.iloc[:1]]).to_csv(p/'targets.csv',index=False)
            with self.assertRaises(ValueError):self.call(p)
    def test_start_resets_to_flat_and_end_truncates(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);self.fixture(p);self.call(p,start='2020-01-03',end='2020-01-04')
            d=pd.read_csv(p/'out/product_ledger.csv')
            self.assertEqual(len(d),2)
            self.assertEqual(d.position.iloc[0],0)
            self.assertEqual(d.fee_cost.iloc[1],.001)
    def test_final_data_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);self.fixture(p)
            market=pd.read_csv(p/'market.csv')
            market.rename(columns={'return':'ret'}).to_csv(p/'X.csv',index=False)
            pd.DataFrame(dict(code=['X'],asset_class=['equity'])).to_csv(p/'universe.csv',index=False)
            with contextlib.redirect_stdout(io.StringIO()):run(p/'targets.csv',p/'out',data_dir=p)
            self.assertTrue((p/'out/summary.csv').is_file())
