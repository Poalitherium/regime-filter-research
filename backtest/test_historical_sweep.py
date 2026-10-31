"""Numerical and timing checks for the historical survey."""
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from backtest.historical_evaluate import bootstrap
from backtest.metrics import block_bootstrap_sharpe_diff


class HistoricalChecks(unittest.TestCase):
    def test_fast_bootstrap_matches_explicit_resampling(self):
        rng=np.random.default_rng(911)
        for n,block in [(400,20),(417,20),(371,60),(251,10)]:
            c=rng.normal(.0004,.009,n)
            a=.8*c+rng.normal(.00015,.002,n)
            old=block_bootstrap_sharpe_diff(pd.Series(a),pd.Series(c),block=block,reps=1100,seed=41,return_boot=True)
            new=bootstrap(a,c,block=block,reps=1100,seed=41,return_boot=True)
            np.testing.assert_allclose(new['boot'],old['boot'],atol=1e-10,rtol=1e-10)
            self.assertAlmostEqual(new['delta'],old['d_sharpe'],places=12)
            self.assertAlmostEqual(new['ci_low'],old['ci_low'],places=10)
            count=round(old['p_boot']*1100)
            self.assertEqual(new['exceedances'],count)

    def test_identity_is_exactly_zero(self):
        x=np.random.default_rng(5).normal(0,.01,301)
        r=bootstrap(x,x,reps=1100)
        self.assertEqual((r['delta'],r['ci_low'],r['ci_high'],r['p']),(0.,0.,0.,1.))

    @unittest.skipUnless(all(Path(p).is_file() for p in (
        'data/final/universe.csv', 'data/processed/offmarket/offmarket_daily.csv',
        'm3_outputs/bocpd_h250_filter.csv')), 'Market inputs and BOCPD filter not generated')
    def test_signature_adapter_matches_original_features_and_labels(self):
        from backtest.historical_sweep import ROOT,signature_panel
        from m3 import metalabel as ml
        with threadpool_limits(limits=1):
            original=ml.build_panel(ROOT/'data/final',ROOT/'data/processed/offmarket/offmarket_daily.csv',
                     ROOT/'m3_outputs/bocpd_h250_filter.csv',end='2026-10-15')
            adapted=signature_panel('futures')
        cols=ml.SIG+['fwd','y','label_end']
        o=original.reset_index(drop=True).set_index(['contract','week'])[cols].sort_index()
        a=adapted.reset_index(drop=True).set_index(['contract','week'])[cols].sort_index()
        pd.testing.assert_frame_equal(a,o,check_exact=False,rtol=1e-12,atol=1e-12)


if __name__=='__main__':unittest.main()
