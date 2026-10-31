"""Explicit USD/EUR cash conversion; usdeur means EUR per one USD."""
from pathlib import Path
import numpy as np
import pandas as pd


class FXRates:
    def __init__(self, data):
        if not {'date','usdeur'}.issubset(data):
            raise ValueError('FX CSV must contain date,usdeur (EUR per USD)')
        d=data[['date','usdeur']].copy()
        d['date']=pd.to_datetime(d.date,errors='raise')
        if d.date.isna().any() or d.date.duplicated().any() or d.date.dt.tz is not None or not d.date.eq(d.date.dt.normalize()).all():
            raise ValueError('unique timezone-free calendar dates required')
        d['usdeur']=pd.to_numeric(d.usdeur,errors='raise')
        if d.empty or not np.isfinite(d.usdeur).all() or (d.usdeur<=0).any():
            raise ValueError('FX rates must be nonempty, positive and finite')
        self.rates=d.set_index('date').usdeur.sort_index()

    @classmethod
    def from_csv(cls,path):
        return cls(pd.read_csv(Path(path)))

    def lookup(self,when,policy='exact',max_age_days=3):
        day=pd.Timestamp(when)
        if pd.isna(day) or day.tzinfo is not None or day!=day.normalize():
            raise ValueError('execution date must be a timezone-free calendar date')
        if policy not in {'exact','backward'}:
            raise ValueError('FX policy must be exact or backward')
        if isinstance(max_age_days,bool) or not isinstance(max_age_days,int) or max_age_days<0:
            raise ValueError('max_age_days must be nonnegative integer')
        if policy=='exact':
            if day not in self.rates.index:
                raise ValueError(f'no exact FX rate for {day.date()}')
            used=day
        else:
            i=self.rates.index.searchsorted(day,side='right')-1
            if i<0:
                raise ValueError('no past FX rate; future filling is forbidden')
            used=self.rates.index[i]
            if (day-used).days>max_age_days:
                raise ValueError('past FX rate is too stale')
        return float(self.rates.loc[used]),str(used.date())


def convert_amount(amount, source, base, when, fx=None, policy='exact',max_age_days=3):
    if source not in {'USD','EUR'} or base not in {'USD','EUR'}:
        raise ValueError('only USD and EUR are supported')
    if not np.isfinite(amount):
        raise ValueError('finite cash amount required')
    if source==base:
        return dict(amount_base=float(amount),conversion_factor=1.,fx_date=None)
    if fx is None:
        raise ValueError('USD/EUR FX file required for cross-currency conversion')
    rate,used=fx.lookup(when,policy,max_age_days)
    factor=rate if source=='USD' else 1/rate
    return dict(amount_base=float(amount)*factor,conversion_factor=factor,fx_date=used)
