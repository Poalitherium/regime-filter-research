"""Past-only annual selection for the historical sweep's external/calendar filters."""
from contextlib import redirect_stdout
import io
from unittest.mock import patch
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .historical_sweep import ROOT, OUT, prepare


def run():
    from m3.features import load_exog
    from m3.predictability import holm
    from m5.common import load_all
    from m4 import common as mc, gates as mg, crowding as cr, impvol as ivm
    from m5 import seasonality as se, expiry as ex, calfilter as cf, calendar as cal
    from . import resplit_gates as old
    prepare()
    dest=OUT/'models/futures';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'selection_complete.json').exists(): return
    contracts=load_all(ROOT/'data/final')
    exog=load_exog(ROOT/'data/processed/offmarket/offmarket_daily.csv')
    oldout=ROOT/'backtest_outputs/resplit_80_20'
    dates=pd.DatetimeIndex(pd.read_csv(oldout/'inputs/fomc_scheduled.csv',parse_dates=['statement_date']).statement_date)
    cot=pd.read_csv(oldout/'cot_features.csv',parse_dates=['report_date','available'])
    iv=old.iv_inputs(exog)
    raw,panels={},{}
    with patch('m4.common.fomc_dates',return_value=dates),patch('m4.impvol.DEV_END','2026-10-15'):
        st2=mg.riskoff_states(contracts,exog,.8)
        st3=cr.states(contracts,cot)
        st5=ivm.inversion_states(contracts,iv)
        panels['M4-1']=mg.event_panel(contracts,dev_end='2026-10-15')
        panels['M4-2']=mg.riskoff_panel(contracts,st2,'2026-10-15')
        panels['M4-3']=cr.panel(contracts,st3,'2026-10-15')
        panels['M4-5']=ivm.inversion_panel(contracts,st5)
        states={
          'M4-1':(list(contracts),{k:mc.event_state(c['data'].index,k) for k,c in contracts.items()},
                    {k:mc.event_state(c['data'].index,k,lead=2) for k,c in contracts.items()}),
          'M4-2':(list(st2),{k:s.W.shift(2).dropna().astype(bool) for k,s in st2.items()},
                    {k:s.W.fillna(0).astype(bool) for k,s in st2.items()}),
          'M4-3':(list(st3),{k:s.C.shift(2).dropna().astype(bool) for k,s in st3.items()},
                    {k:s.C.fillna(0).astype(bool) for k,s in st3.items()}),
          'M4-5':(list(st5),{k:s.W.shift(2).dropna().astype(bool) for k,s in st5.items()},
                    {k:s.W.fillna(0).astype(bool) for k,s in st5.items()}),
        }
        for tid,(members,ps,fs) in states.items():
            f,log=mc.proportional_filter(contracts,members,ps,fs)
            raw[tid]=f
            log.to_csv(dest/f'{tid}_parameter_refits.csv',index=False)
        raw['M4-4'],_=ivm.impvol_filter(contracts,iv)
    for group,codes in se.GROUPS.items():
        f,log=cf.build(contracts,{group:codes},cal.in_season,lead=2)
        raw['season_'+group]=f
        log.to_csv(dest/f'season_{group}_parameter_refits.csv',index=False)
    groups={}
    for code in ex.FUTURES:groups.setdefault(contracts[code]['asset'],[]).append(code)
    fn=lambda idx,code,lead:cal.roll_window(idx,code,ex.HALF_WIDTH,lead)
    raw['M5-4'],log=cf.build(contracts,groups,fn,lead=2)
    log.to_csv(dest/'M5-4_parameter_refits.csv',index=False)
    panels['M5-4']=ex.daily_panel(contracts,dev_end='2026-10-15')
    print('Causal states and quarterly parameters prepared',flush=True)
    settings={
      'M4-1':('E',['E'],['logvol'],['E']),
      'M4-2':('W',['W','S','R'],['logvol','S','R'],['W']),
      'M4-3':('C',['C'],['logvol'],['C']),
      'M4-5':('W',['W','I','R'],['logvol','I','R'],['W']),
    }
    records=[];decisions={}
    original_sample=se.monthly_sample
    for year in range(2007,2027):
        cut=pd.Timestamp(year,1,1)-pd.Timedelta(days=1)
        rec=[]
        for tid,(term,terms,controls,added) in settings.items():
            df=panels[tid];df=df[df.date<=cut]
            try:
                if len(df)<100 or df.period.nunique()<12 or df[term].nunique()<2:
                    raise ValueError('insufficient observations or state variation')
                r=mc.fe_test(df,terms)
                o=mc.panel_oos_r2(df,controls,added)
                row=dict(trial=tid,year=year,train_end=str(cut.date()),last_observation=str(df.date.max().date()),
                         n=r['n'],p=r[term]['p'],oos_r2=o['oos_r2'],oos_n=o['days'],status='estimated')
            except (ValueError,np.linalg.LinAlgError) as exc:
                row=dict(trial=tid,year=year,train_end=str(cut.date()),p=1.,oos_r2=np.nan,
                         n=len(df),oos_n=0,status=str(exc))
            rec.append(row)
        ps=[r['p'] for r in rec]
        adj=[holm(ps[:2])[0],holm(ps[:2])[1],holm(ps[:3])[2],holm(ps)[3]]
        decisions[year]={}
        for r,p in zip(rec,adj):
            r['p_adjusted']=float(p);r['active']=bool(p<.05 and r['oos_r2']>0)
            decisions[year][r['trial']]=r['active']
        records+=rec
        # The seasonal function truncates net P&L before constructing next-month labels.
        if year <= 2008:
            # The original OOS check starts in 2008; no completed validation
            # history exists at these refits, so selection cannot yet succeed.
            season=pd.DataFrame([dict(group=g,passed=False,p=np.nan,oos_r2=np.nan,
                                     status='No completed original OOS validation history') for g in se.GROUPS])
        else:
            with patch('m5.seasonality.monthly_sample',side_effect=lambda c,k:original_sample(c,k,dev_end=str(cut.date()))):
                season,_=se.gate(contracts)
        for row in season.to_dict('records'):
            records.append(dict(trial='M5-3',year=year,train_end=str(cut.date()),**row,active=bool(row['passed'])))
            decisions[year]['season_'+row['group']]=bool(row['passed'])
        df=panels['M5-4'];df=df[df.date<=cut]
        r=ex.fe_test(df,['W']);o=ex.panel_oos_r2(df,['W'])
        active=bool(r['W']['p']<.05 and o['oos_r2']>0)
        records.append(dict(trial='M5-4',year=year,train_end=str(cut.date()),p=r['W']['p'],oos_r2=o['oos_r2'],
                            oos_n=o['days'],n=r['n'],active=active))
        decisions[year]['M5-4']=active
        print(year,{k:v for k,v in decisions[year].items() if v},flush=True)
        pd.DataFrame(records).to_csv(dest/'annual_selection.csv',index=False)
    for tid in ['M4-1','M4-2','M4-3','M4-4','M4-5','M5-4']:
        f=raw[tid].copy();f['date']=pd.to_datetime(f.date)
        if tid!='M4-4':
            active=f.date.dt.year.map(lambda y:decisions.get(y,{}).get(tid,False))
            f.loc[~active,'filter_close_t']=1.
        f.to_csv(dest/f'{tid}_filter.csv',index=False)
    f=old.identity(contracts);f['date']=pd.to_datetime(f.date)
    f=f.set_index(['contract','date'])
    for group,codes in se.GROUPS.items():
        r=raw['season_'+group].copy();r['date']=pd.to_datetime(r.date)
        r=r[r.contract.isin(codes)]
        active=r.date.dt.year.map(lambda y:decisions.get(y,{}).get('season_'+group,False))
        r.loc[~active,'filter_close_t']=1.
        r=r.set_index(['contract','date'])
        f.loc[r.index,'filter_close_t']=r.filter_close_t
    f.reset_index().to_csv(dest/'M5-3_filter.csv',index=False)
    (dest/'selection_complete.json').write_text(json.dumps(decisions,indent=2),encoding='utf-8')
    print('Annual selection complete',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):run()
