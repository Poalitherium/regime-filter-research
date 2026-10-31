import unittest
import numpy as np
import pandas as pd
from m1.exposure_cost_model import transaction_costs

class CostTests(unittest.TestCase):
    def cost(self,b,a,roll=False,bps=10):
        return transaction_costs(pd.Series([b]),pd.Series([a]),pd.Series([roll]),bps).iloc[0]
    def test_open(self):self.assertAlmostEqual(self.cost(0,1).total_cost,.001)
    def test_reversal(self):self.assertAlmostEqual(self.cost(1,-1).total_cost,.002)
    def test_unchanged_roll(self):
        x=self.cost(1,1,True);self.assertEqual(x.extra_roll_turnover,2)
    def test_reversal_roll_not_double_counted(self):
        x=self.cost(1,-1,True);self.assertEqual(x.total_turnover,2);self.assertEqual(x.extra_roll_turnover,0)
    def test_close_on_roll(self):self.assertEqual(self.cost(1,0,True).total_turnover,1)
    def test_flat(self):self.assertEqual(self.cost(0,0,True).total_cost,0)
    def test_cost_scaling(self):self.assertAlmostEqual(self.cost(0,1,bps=20).total_cost,.002)
    def test_invalid_inputs(self):
        for value in [np.nan,np.inf]:
            with self.assertRaises(ValueError):self.cost(value,1)
        for bps in [-1,np.inf]:
            with self.assertRaises(ValueError):self.cost(0,1,bps=bps)
        with self.assertRaises(ValueError):self.cost(0,1,roll='False')
        with self.assertRaises(ValueError):transaction_costs(pd.Series([0],index=[0]),pd.Series([1],index=[1]))
