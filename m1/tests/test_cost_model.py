import unittest
from m1.cost_model import execution_cost,position_legs,convert_cash,cash_to_bps,load_schedule

class CostTests(unittest.TestCase):
    def fee(self,code='ES',quantity=1,**kw):
        return execution_cost(code,quantity,'2010-01-04',mode='snapshot_scenario',**kw)
    def test_es_components(self):
        r=self.fee()
        self.assertEqual((r['exchange_fee'],r['broker_fee'],r['regulatory_fee']),(1.386,.85,.011))
        self.assertEqual(r['known_fee_subtotal'],2.247)
        self.assertIsNone(r['modeled_total'])
    def test_reversal(self):
        legs=position_legs({'ESZ6':1},{'ESZ6':-1})
        self.assertEqual(legs,[dict(contract_id='ESZ6',side='sell',quantity=2)])
        self.assertEqual(self.fee(quantity=2)['known_fee_subtotal'],4.494)
    def test_roll_and_reversal_no_double_charge(self):
        for after in [1,-1]:
            legs=position_legs({'ESZ6':1},{'ESH7':after})
            self.assertAlmostEqual(sum(self.fee(quantity=l['quantity'])['known_fee_subtotal'] for l in legs),4.494)
    def test_unchanged(self):
        self.assertEqual(position_legs({'ESZ6':1},{'ESZ6':1}),[])
    def test_bad_quantities(self):
        for q in [0,-1,.5,True]:
            with self.assertRaises(ValueError): self.fee(quantity=q)
    def test_proxy_gate(self):
        with self.assertRaises(ValueError): self.fee('FESX')
        r=self.fee('FESX',allow_proxy=True)
        self.assertEqual(r['known_fee_subtotal'],.42)
        self.assertIsNone(r['broker_fee'])
        self.assertFalse(r['fee_components_complete'])
        self.assertIsNone(r['modeled_total'])
    def test_no_historical_claim(self):
        with self.assertRaises(ValueError): execution_cost('ES',1,'2010-01-04',mode='historical')
    def test_order_type(self):
        with self.assertRaises(ValueError): self.fee(order_type='spread')
    def test_unknown_symbol(self):
        with self.assertRaises(ValueError): self.fee('UNKNOWN')
    def test_slippage_separate(self):
        self.assertEqual(self.fee(slippage_per_contract=12.5)['modeled_total'],14.747)
        with self.assertRaises(ValueError): self.fee(slippage_per_contract=float('nan'))
    def test_fx(self):
        with self.assertRaises(ValueError): convert_cash(.42,'EUR','USD')
        self.assertAlmostEqual(convert_cash(.42,'EUR','USD',{'EUR':1.1}),.462)
    def test_bps(self):
        self.assertAlmostEqual(cash_to_bps(2.24,100000),.224)
        with self.assertRaises(ValueError): cash_to_bps(2.24,0)
    def test_universe(self):
        self.assertEqual(len(load_schedule()),13)
        self.assertEqual(load_schedule()['NK']['fee_symbol'],'NKD')

if __name__=='__main__': unittest.main()
