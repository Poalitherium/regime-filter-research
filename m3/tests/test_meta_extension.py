"""Extending the evaluation horizon must preserve earlier predictions."""
import unittest
import numpy as np
import pandas as pd
from m3.metalabel import walk_forward


class MetaExtensionTests(unittest.TestCase):
    def test_extension_and_future_label_perturbation_preserve_prefix(self):
        rng = np.random.default_rng(31)
        weeks = pd.date_range('2002-01-04', '2017-12-29', freq='W-FRI')
        rows = []
        for code in ['A', 'B', 'C']:
            x = rng.normal(size=len(weeks))
            rows.append(pd.DataFrame(dict(contract=code, week=weeks, x=x,
                label_end=weeks + pd.Timedelta(days=30),
                y=(x + rng.normal(size=len(weeks)) > 0).astype(float))))
        panel = pd.concat(rows, ignore_index=True)
        old, _ = walk_forward(panel, ['x'], 'logit')
        extended, _ = walk_forward(panel, ['x'], 'logit', last_refit=2017)
        self.assertEqual(old.week.max().year, 2014)
        self.assertEqual(extended.week.max().year, 2017)
        pd.testing.assert_frame_equal(old, extended[extended.week.dt.year <= 2014])
        altered = panel.copy()
        later = altered.label_end >= pd.Timestamp('2016-01-01')
        altered.loc[later, 'y'] = 1 - altered.loc[later, 'y']
        changed, _ = walk_forward(altered, ['x'], 'logit', last_refit=2017)
        prefix = extended.week < pd.Timestamp('2016-01-01')
        np.testing.assert_array_equal(extended.loc[prefix, 'p'], changed.loc[prefix, 'p'])


if __name__ == '__main__':
    unittest.main()
