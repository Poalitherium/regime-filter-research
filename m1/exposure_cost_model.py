"""Transaction costs for externally supplied executed exposure changes.

This module does not generate targets, lag signals, fit models, compute market
P&L or compare strategies. The caller supplies before/after exposures for each
execution event. All costs are fractions of the same capital basis as exposure.
"""
import numpy as np
import pandas as pd

BASE_COST_BPS = {'equity':1., 'rates':.5, 'fx':.5, 'energy':2., 'metal':1.5, 'agri':2.}


def transaction_costs(before, after, roll=None, cost_bps=0.):
    """Count trade legs and estimate per-side bps costs.

    On a normal execution, volume = abs(after-before).
    On a roll, volume = abs(before)+abs(after), replacing normal volume.
    No timing or position decision is made inside this function.
    """
    if not isinstance(before,pd.Series) or not isinstance(after,pd.Series):
        raise TypeError('before and after must be Series')
    if not before.index.equals(after.index) or before.index.has_duplicates:
        raise ValueError('unique matching event indices required')
    if not np.isfinite(before.to_numpy(dtype=float)).all() or not np.isfinite(after.to_numpy(dtype=float)).all():
        raise ValueError('finite complete exposures required')
    if not np.isscalar(cost_bps) or not np.isfinite(cost_bps) or cost_bps<0:
        raise ValueError('cost_bps must be a finite nonnegative scalar')
    if roll is None:
        flags=pd.Series(False,index=before.index)
    else:
        if not isinstance(roll,pd.Series) or not roll.index.equals(before.index):
            raise ValueError('roll flags must have matching event indices')
        if roll.isna().any() or not roll.isin([True,False,0,1]).all():
            raise ValueError('complete boolean roll flags required')
        flags=roll.astype(bool)
    ordinary=(after-before).abs()
    total=ordinary.where(~flags,before.abs()+after.abs())
    extra=total-ordinary
    return pd.DataFrame(dict(exposure_before=before,exposure_after=after,roll=flags,
        rebalance_turnover=ordinary,extra_roll_turnover=extra,total_turnover=total,
        rebalance_cost=ordinary*cost_bps/10000,extra_roll_cost=extra*cost_bps/10000,
        total_cost=total*cost_bps/10000))
