"""Daily exposure accounting on each product's observed calendar."""
import numpy as np
import pandas as pd
from .exposure_cost_model import BASE_COST_BPS, transaction_costs

COST_BPS = BASE_COST_BPS.copy()


def contract_pnl(target, ret, roll=None, cost_bps=0., delay=1, *, slippage_bps=0., missing_target='flat'):
    """Execute close-t targets at close t+delay, earn returns thereafter.

    delay=0 is an optimistic same-close sensitivity, not the default.
    Costs are booked on the execution/roll date, not next day's P&L date.
    Position is continuous notional exposure, not integer futures lots.
    """
    if isinstance(delay,bool) or not isinstance(delay,int) or delay<0:
        raise ValueError('delay must be a nonnegative integer')
    if missing_target not in {'flat','hold'}:
        raise ValueError('missing_target must be flat or hold')
    if not isinstance(ret.index,pd.DatetimeIndex) or ret.index.has_duplicates or not ret.index.is_monotonic_increasing or ret.index.hasnans:
        raise ValueError('returns require unique increasing dates')
    if target.index.has_duplicates or len(target.index.difference(ret.index)):
        raise ValueError('targets must use the product calendar without duplicate dates')
    if not np.isfinite(ret.iloc[1:].to_numpy(dtype=float)).all() or np.isinf(ret.to_numpy(dtype=float)).any():
        raise ValueError('only the initial return may be missing')
    t=target.reindex(ret.index).astype(float)
    if np.isinf(t.to_numpy()).any():
        raise ValueError('infinite target')
    t=t.fillna(0.) if missing_target=='flat' else t.ffill().fillna(0.)
    after=t.shift(delay).fillna(0.)
    before=after.shift(1).fillna(0.)
    if (ret.isna() & before.ne(0)).any():
        raise ValueError('missing return while holding a position')
    if roll is None:
        flags=pd.Series(False,index=ret.index)
    else:
        if not roll.index.equals(ret.index) or roll.isna().any() or not roll.isin([True,False]).all():
            raise ValueError('complete Boolean roll flags on the return calendar required')
        flags=roll.astype(bool)
    fees=transaction_costs(before,after,flags,cost_bps)
    slip=transaction_costs(before,after,flags,slippage_bps).total_cost
    gross=before*ret.fillna(0.)
    return pd.DataFrame({'position':before,'position_close':after,'gross':gross,
                         'fee_cost':fees.total_cost,'slippage_cost':slip,
                         'cost':fees.total_cost+slip,'net':gross-fees.total_cost-slip,
                         'turnover':fees.total_turnover,'rebalance_turnover':fees.rebalance_turnover,
                         'extra_roll_turnover':fees.extra_roll_turnover},index=ret.index)


def portfolio(pnl, universe, min_contracts=1):
    """Equal-weight return index, cash on absent product dates.

    Fixed membership is used by the generic runner. This is a normalized
    return index; capital transfers and integer-lot allocation are not modeled.
    """
    if set(pnl)!=set(universe.columns) or universe.index.has_duplicates:
        raise ValueError('portfolio membership and product keys must match')
    if universe.isna().any().any() or not universe.isin([True,False]).all().all():
        raise ValueError('Boolean membership required')
    if universe.nunique().gt(1).any():
        raise ValueError('portfolio membership must be fixed over the sample')
    eligible=universe.astype(bool)
    n=eligible.sum(axis=1)
    out={}
    for field in ['gross','fee_cost','slippage_cost','cost','net','turnover']:
        wide=pd.DataFrame({c:pnl[c][field] for c in universe.columns}).reindex(universe.index).fillna(0.)
        out[field]=wide.where(eligible,0.).sum(axis=1)/n.replace(0,np.nan)
    exposure={}
    for c in universe.columns:
        # At closed-market dates, carry the last observed close position.
        carried=pnl[c].position_close.reindex(universe.index).ffill().shift(1).fillna(0.)
        exposure[c]=carried.abs()
    out['gross_exposure']=pd.DataFrame(exposure).where(eligible,0.).sum(axis=1)/n.replace(0,np.nan)
    out['n_contracts']=n
    return pd.DataFrame(out).loc[n>=min_contracts]


def performance(r, annualization=252):
    r=r.dropna().astype(float)
    if r.empty or not np.isfinite(r).all():
        raise ValueError('nonempty finite returns required')
    if (r<=-1).any():
        raise ValueError('return <= -100%; compounded wealth is not valid')
    wealth=(1+r).cumprod()
    peak=wealth.cummax().clip(lower=1.)
    dd=wealth/peak-1
    vol=float(r.std(ddof=1)*np.sqrt(annualization))
    ar=float(r.mean()*annualization)
    cagr=float(wealth.iloc[-1]**(annualization/len(r))-1)
    return dict(start=str(r.index[0].date()),end=str(r.index[-1].date()),days=len(r),
                ann_mean_return=ar,cagr=cagr,ann_vol=vol,sharpe=ar/vol if vol>0 else np.nan,
                max_drawdown=float(dd.min()),calmar=cagr/abs(dd.min()) if dd.min()<0 else np.nan)
