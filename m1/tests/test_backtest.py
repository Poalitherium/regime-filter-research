import unittest
import numpy as np
import pandas as pd
from m1.backtest import contract_pnl,portfolio,performance

class BacktestTests(unittest.TestCase):
    def setUp(self): self.idx=pd.date_range('2020-01-01',periods=6)
    def s(self,x): return pd.Series(x,index=self.idx,dtype=float)
    def test_execution_cost_precedes_return(self):
        p=contract_pnl(self.s([1]*6),self.s([0,.1,.1,.1,.1,.1]),cost_bps=10,delay=1)
        self.assertEqual(p.cost.iloc[0],0)
        self.assertEqual(p.cost.iloc[1],.001)
        self.assertEqual(p.gross.iloc[1],0)
        self.assertEqual(p.gross.iloc[2],.1)
    def test_future_target_does_not_change_past(self):
        a=self.s([1]*6);b=a.copy();b.iloc[3:]=-1
        x=contract_pnl(a,self.s([.01]*6),cost_bps=10)
        y=contract_pnl(b,self.s([.01]*6),cost_bps=10)
        pd.testing.assert_frame_equal(x.iloc[:4],y.iloc[:4])
    def test_reverse_and_roll_counts_two(self):
        flags=pd.Series([False,False,True,False,False,False],index=self.idx)
        p=contract_pnl(self.s([1,-1,-1,-1,-1,-1]),self.s([0]*6),flags,10,1)
        self.assertEqual(p.turnover.iloc[2],2)
        self.assertEqual(p.cost.iloc[2],.002)
    def test_zero_slippage_and_positive_slippage(self):
        args=(self.s([1]*6),self.s([0]*6))
        a=contract_pnl(*args,cost_bps=10)
        b=contract_pnl(*args,cost_bps=10,slippage_bps=2)
        self.assertEqual(a.slippage_cost.sum(),0)
        self.assertAlmostEqual(b.net.sum()-a.net.sum(),-.0002)
    def test_missing_target_policy(self):
        t=self.s([1,np.nan,np.nan,1,1,1])
        a=contract_pnl(t,self.s([0]*6),missing_target='flat')
        b=contract_pnl(t,self.s([0]*6),missing_target='hold')
        self.assertEqual(a.position_close.iloc[2],0)
        self.assertEqual(b.position_close.iloc[2],1)
    def test_invalid_return_and_slippage(self):
        with self.assertRaises(ValueError): contract_pnl(self.s([1]*6),self.s([0,0,np.nan,0,0,0]))
        with self.assertRaises(ValueError): contract_pnl(self.s([1]*6),self.s([0]*6),slippage_bps=-1)
    def test_first_loss_drawdown(self):
        self.assertAlmostEqual(performance(self.s([-.1,0,0,0,0,0]))['max_drawdown'],-.1)
    def test_closed_market_fixed_denominator(self):
        days=self.idx.delete(3)
        a=contract_pnl(self.s([1]*6),self.s([0,.1,.1,.1,.1,.1]))
        b=contract_pnl(pd.Series(1.,index=days),pd.Series(0.,index=days))
        p=portfolio({'A':a,'B':b},pd.DataFrame(True,index=self.idx,columns=['A','B']),2)
        self.assertEqual(p.loc[self.idx[3],'net'],.05)
        self.assertEqual(p.loc[self.idx[3],'gross_exposure'],1.)
    def test_roll_on_liquidation(self):
        r=pd.Series(False,index=self.idx);r.iloc[2]=True
        p=contract_pnl(self.s([1,0,0,0,0,0]),self.s([0]*6),r,10)
        self.assertEqual(p.turnover.iloc[2],1)
    def test_pending_last_target_not_executed(self):
        p=contract_pnl(self.s([0,0,0,0,0,1]),self.s([0]*6),cost_bps=10)
        self.assertEqual(p.cost.sum(),0)
