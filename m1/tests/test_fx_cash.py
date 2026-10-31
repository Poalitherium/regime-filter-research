import unittest
import pandas as pd
from m1.fx import FXRates,convert_amount
from m1.cash_ledger import price_ledger

class FXCashTests(unittest.TestCase):
    def setUp(self):
        self.fx=FXRates(pd.DataFrame({'date':['2020-01-03','2020-01-06'],'usdeur':[.8,.9]}))
        self.ledger=pd.DataFrame([dict(trade_date='2020-01-03',code='FESX',contract_id='FESXH0',side='buy',quantity=1,order_type='outright')])
    def test_quote_and_inverse(self):
        self.assertEqual(convert_amount(10,'USD','EUR','2020-01-03',self.fx)['amount_base'],8)
        self.assertEqual(convert_amount(8,'EUR','USD','2020-01-03',self.fx)['amount_base'],10)
    def test_same_currency_without_file(self):
        self.assertEqual(convert_amount(2,'USD','USD','2020-01-03')['amount_base'],2)
    def test_exact_gap_and_missing_fx_fail(self):
        with self.assertRaises(ValueError):self.fx.lookup('2020-01-04')
        with self.assertRaises(ValueError):convert_amount(1,'EUR','USD','2020-01-03')
    def test_backward_only_and_stale(self):
        self.assertEqual(self.fx.lookup('2020-01-05','backward',2),(.8,'2020-01-03'))
        with self.assertRaises(ValueError):self.fx.lookup('2020-01-02','backward')
        with self.assertRaises(ValueError):self.fx.lookup('2020-01-10','backward',3)
    def test_invalid_table(self):
        for d in [dict(date=['2020-01-03']*2,usdeur=[.8,.9]),dict(date=['2020-01-03'],usdeur=[0])]:
            with self.assertRaises(ValueError):FXRates(pd.DataFrame(d))
    def test_proxy_partial_total_survives_fx(self):
        r=price_ledger(self.ledger,fx=self.fx,base_currency='USD',allow_proxy=True).iloc[0]
        self.assertAlmostEqual(r.known_fee_subtotal_base,.525)
        self.assertTrue(pd.isna(r.modeled_total_base))
        self.assertEqual(r.slippage_assumption,'ZERO_SLIPPAGE_ASSUMPTION')
    def test_us_default_zero(self):
        l=self.ledger.assign(code='ES',contract_id='ESH0',quantity=2)
        r=price_ledger(l,base_currency='USD').iloc[0]
        self.assertAlmostEqual(r.modeled_total_base,4.494)
    def test_cash_override(self):
        l=self.ledger.assign(code='ES',contract_id='ESH0',quantity=2,slippage_per_contract=1.)
        r=price_ledger(l).iloc[0]
        self.assertAlmostEqual(r.modeled_total,6.494)
    def test_fractional_quantity_rejected(self):
        with self.assertRaises(ValueError):price_ledger(self.ledger.assign(quantity=1.5))
