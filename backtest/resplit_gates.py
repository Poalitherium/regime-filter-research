"""Original gate rules, re-estimated before the new test period; causal filters."""
from unittest.mock import patch
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .resplit_common import ROOT,OUT,DEV_END,END,write_spec
from m3.features import load_exog
from m3.predictability import holm
from m5.common import load_all
from m4 import common as mc, gates as mg, cot, crowding as cr, impvol as ivm
from m5 import seasonality as se, expiry as ex, calfilter as cf, calendar as cal

def identity(contracts):
    return pd.concat([pd.DataFrame({'date':c['data'].index,'contract':code,'filter_close_t':1.0})
                      for code,c in contracts.items()],ignore_index=True)

def cot_features():
    raw=cot.load(OUT/'inputs/cot_legacy.csv')
    f=cot.weekly_features(raw)
    # Conservative release handling: unavailable backlog becomes usable only after catch-up.
    # Sources: CFTC 7864-19, Historical Special Announcements (2023), 9147-25.
    delays=[('2018-12-24','2019-03-05','2019-03-11'),
            ('2023-01-31','2023-03-14','2023-03-22'),
            ('2025-09-30','2025-12-23','2025-12-30')]
    for start,end,available in delays:
        mask=f.report_date.between(start,end)
        f.loc[mask,'available']=pd.Timestamp(available)
    f=f.sort_values(['code','report_date']).drop_duplicates(['code','available'],keep='last')
    f.to_csv(OUT/'cot_features.csv',index=False)
    (OUT/'cot_release_policy.json').write_text(json.dumps({'policy':'Original seven-day lag, but defer backlogs until full catch-up; conservative availability, not actual individual release dates',
      'delays':delays,'sources':['https://www.cftc.gov/PressRoom/PressReleases/7864-19',
      'https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalSpecialAnnouncements/index.htm',
      'https://www.cftc.gov/PressRoom/PressReleases/9147-25']},indent=2))
    return f

def iv_inputs(exog):
    raw=pd.read_csv(OUT/'inputs/implied_vol.csv',index_col='date',parse_dates=True)
    raw.index=raw.index+pd.offsets.BDay(1)
    raw=raw.groupby(level=0).last()
    return raw.join(np.exp(exog.vix.dropna()).rename('vix'),how='outer')

def evaluate_panel(tid,df,term,controls,added,terms=None):
    assert pd.to_datetime(df.date).max()<=pd.Timestamp(DEV_END)
    r=mc.fe_test(df,terms or [term])
    o=mc.panel_oos_r2(df,controls,added)
    return dict(trial=tid,gate_start=str(df.date.min().date()),gate_end=str(df.date.max().date()),
      n=r['n'],p=r[term]['p'],oos_r2=o['oos_r2'],coef_ann=r[term]['coef']*252)

def run():
    write_spec();dest=OUT/'filters';dest.mkdir(parents=True,exist_ok=True)
    contracts=load_all(ROOT/'data/final')
    exog=load_exog(ROOT/'data/processed/offmarket/offmarket_daily.csv')
    dates=pd.DatetimeIndex(pd.read_csv(OUT/'inputs/fomc_scheduled.csv',parse_dates=['statement_date']).statement_date)
    records=[]
    with patch('m4.common.fomc_dates',return_value=dates):
        ev=mg.event_panel(contracts,dev_end=DEV_END)
        records.append(evaluate_panel('M4-1',ev,'E',['logvol'],['E']))
        st2=mg.riskoff_states(contracts,exog,0.8)
        records.append(evaluate_panel('M4-2',mg.riskoff_panel(contracts,st2,DEV_END),'W',['logvol','S','R'],['W'],['W','S','R']))
        st3=cr.states(contracts,cot_features())
        records.append(evaluate_panel('M4-3',cr.panel(contracts,st3,DEV_END),'C',['logvol'],['C']))
        iv=iv_inputs(exog)
        st5=ivm.inversion_states(contracts,iv)
        with patch('m4.impvol.DEV_END',DEV_END):
            records.append(evaluate_panel('M4-5',ivm.inversion_panel(contracts,st5),'W',['logvol','I','R'],['W'],['W','I','R']))
        p=[r['p'] for r in records]
        adjustments=[holm(p[:2])[0],holm(p[:2])[1],holm(p[:3])[2],holm(p)[3]]
        for r,adj in zip(records,adjustments):
            r['gate_p_adjusted']=float(adj);r['passed']=bool(adj<.05 and r['oos_r2']>0)
            print(r,flush=True)
        pd.DataFrame(records).to_csv(OUT/'m4_gates.csv',index=False)
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
        for r in records:
            tid=r['trial']
            if r['passed']:
                members,pnl_state,filter_state=states[tid]
                filt,log=mc.proportional_filter(contracts,members,pnl_state,filter_state)
                log.to_csv(OUT/f'{tid}_refits.csv',index=False)
            else:filt=identity(contracts)
            filt.to_csv(dest/f'{tid}.csv',index=False)
    # Deterministic implied-volatility rule has no gate. Fit statistics are descriptive only.
    with patch('m4.impvol.DEV_END',DEV_END):
        f4,stats=ivm.impvol_filter(contracts,iv)
    f4.to_csv(dest/'M4-4.csv',index=False)
    records.append(dict(trial='M4-4',passed=True,gate='none'))
    # The seasonal gate must use complete calendar-month labels before the cut.
    monthly_end=str((pd.Timestamp(DEV_END).to_period('M').start_time-pd.Timedelta(days=1)).date())
    original_sample=se.monthly_sample
    with patch('m5.seasonality.monthly_sample',side_effect=lambda c,k:original_sample(c,k,dev_end=monthly_end)):
        season,samples=se.gate(contracts)
    season.to_csv(OUT/'season_gates.csv',index=False)
    passed={g:se.GROUPS[g] for g in season.group[season.passed]}
    if passed:
        f,log=cf.build(contracts,passed,cal.in_season,lead=2);log.to_csv(OUT/'M5-3_refits.csv',index=False)
    else:f=identity(contracts)
    f.to_csv(dest/'M5-3.csv',index=False)
    records.append(dict(trial='M5-3',passed=bool(passed),passed_groups=list(passed),gate_end=monthly_end))
    df=ex.daily_panel(contracts,dev_end=DEV_END)
    r=ex.fe_test(df,['W']);o=ex.panel_oos_r2(df,['W'])
    epass=bool(r['W']['p']<.05 and o['oos_r2']>0)
    rec=dict(trial='M5-4',passed=epass,p=r['W']['p'],oos_r2=o['oos_r2'],gate_end=DEV_END)
    records.append(rec)
    if epass:
        groups={}
        for code in ex.FUTURES:groups.setdefault(contracts[code]['asset'],[]).append(code)
        fn=lambda idx,code,lead:cal.roll_window(idx,code,ex.HALF_WIDTH,lead)
        f,log=cf.build(contracts,groups,fn,lead=2);log.to_csv(OUT/'M5-4_refits.csv',index=False)
    else:f=identity(contracts)
    f.to_csv(dest/'M5-4.csv',index=False)
    (OUT/'gate_decisions.json').write_text(json.dumps(records,indent=2,default=str))
    print('M5-3',season.to_string(index=False),flush=True);print(rec,flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=4):run()
