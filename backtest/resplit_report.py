"""Audit the saved 80/20 run and produce its report and slide evidence."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .resplit_common import ROOT, OUT, START, DEV_END, TEST_START, END, NAMES, digest
from .resplit_evaluate import load_base
from .m1_bridge import m1_book

BG='#1B365D'; FG='#DCE6F2'; MUTED='#BACBE0'; ORANGE='#E8A06A'; BLUE='#8FBEE8'
SHORT={'M3-1':'HMM','M3-2':'HMM + macro/news','M3-3':'BOCPD','M3-4':'Portfolio VT',
 'M3-6':'Meta-labeling','M3-7':'Signature meta','M4-4':'Implied volatility',
 'M5-1':'Class-pooled HMM','M5-2':'Global-pooled HMM','M5-3':'Seasonality'}


def md_table(df):
    values=df.astype(str).values.tolist()
    return '\n'.join(['| '+' | '.join(df.columns)+' |','| '+' | '.join(['---']*len(df.columns))+' |']+
                     ['| '+' | '.join(r)+' |' for r in values])


def audit(sm, pairs):
    spec=json.loads((OUT/'analysis_spec.json').read_text())
    for path, sha in spec['input_sha256'].items():assert digest(ROOT/path)==sha,path
    manifest=json.loads((OUT/'evaluated_filter_manifest.json').read_text())
    for row in manifest.values():assert digest(ROOT/row['file'])==row['sha256']
    assert len(sm)==180 and len(pairs)==180
    assert len(sm.trial.unique())==15
    for (window,cost),g in sm.groupby(['window','cost_model']):
        assert g[g.approach.eq('A2')].sharpe.nunique()==1
        assert g.days.nunique()==1 and g.start.nunique()==1 and g.end.nunique()==1
        assert g.days.iloc[0]==(3984 if window=='development' else 997)
    coverage=pd.read_csv(OUT/'filter_coverage.csv')
    assert len(coverage)==15*13 and coverage.missing.sum()==0
    identities=pd.read_csv(OUT/'identity_checks.csv')
    assert len(identities)==40 and identities.max_abs_error.max()==0
    train=pd.read_csv(OUT/'meta_training_audit.csv')
    for r in train.itertuples():
        assert pd.Timestamp(r.last_label)<pd.Timestamp(r.refit)-pd.offsets.BDay(r.embargo_bdays)
    gates=json.loads((OUT/'gate_decisions.json').read_text())
    for g in gates:
        if 'gate_end' in g:assert pd.Timestamp(g['gate_end'])<=pd.Timestamp(DEV_END)
    for tid in ['M3-6','M3-7']:
        assert json.loads((OUT/f'{tid}_reproduction.json').read_text())['max_abs_filter_difference']==0
    # Verify the MOVE source extension has the same units on the overlapping dates.
    raw=json.loads((OUT/'raw/move_daily.json').read_text())['chart']['result'][0]
    y=pd.Series(raw['indicators']['quote'][0]['close'],index=pd.to_datetime(raw['timestamp'],unit='s',utc=True).tz_convert('America/New_York').normalize().tz_localize(None))
    old=pd.read_csv(OUT/'inputs/implied_vol.csv',index_col='date',parse_dates=True).move.loc[:'2014-12-31']
    d=pd.concat([old.rename('old'),y.rename('yahoo')],axis=1).dropna()
    corr=float(d.old.corr(d.yahoo)); ratio=float((d.yahoo/d.old).median())
    assert corr>.98 and .98<ratio<1.02,(corr,ratio)
    result=dict(status='pass',overlay_rows=len(sm),comparison_rows=len(pairs),
      same_A2_all_15=True,common_days=[3984,997],filter_missing=0,
      identity_checks=len(identities),identity_max_error=0,meta_completed_prefix_error=0,
      meta_purge_embargo=True,gates_pre_test=True,input_hashes_unchanged=True,
      move_overlap=dict(n=len(d),correlation=corr,median_ratio=ratio),
      scope='Saved-run consistency and training-date audits. Existing causal unit tests run separately.')
    (OUT/'audit.json').write_text(json.dumps(result,indent=2))
    code=list((ROOT/'backtest').glob('resplit_*.py'))+[ROOT/'m3/metalabel.py',ROOT/'m3/predictability.py']
    (OUT/'code_manifest.json').write_text(json.dumps({str(f.relative_to(ROOT)):digest(f) for f in code},indent=2))
    return result


def style():
    plt.rcParams.update({'font.family':'Arial','font.size':12,'text.color':FG,'axes.labelcolor':FG,
      'xtick.color':MUTED,'ytick.color':FG,'axes.edgecolor':MUTED,'axes.facecolor':BG,
      'figure.facecolor':BG,'savefig.facecolor':BG,'grid.color':'#496789','grid.alpha':.65})


def clean(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis='x',lw=.6)


def figures(sm, pr):
    style();dest=OUT/'figures';dest.mkdir(exist_ok=True)
    data,asset,base,book=load_base()
    fig,axes=plt.subplots(2,1,figsize=(9.6,5),sharex=True,gridspec_kw={'height_ratios':[2,1]})
    for ap,color in [('A1',MUTED),('A2',ORANGE)]:
        r=m1_book(base[ap],data,asset,book,START,DEV_END,'official').net
        equity=(1+r).cumprod();dd=(equity/equity.cummax()-1)*100
        axes[0].plot(r.index,equity,color=color,lw=1.35,label=ap)
        axes[1].plot(r.index,dd,color=color,lw=1.2)
    for ax in axes:clean(ax)
    axes[0].set_ylabel('Growth of 1');axes[1].set_ylabel('Drawdown (%)')
    axes[0].legend(loc='upper left',frameon=False,ncols=2)
    fig.tight_layout();fig.savefig(dest/'baseline.png',dpi=180);plt.close(fig)
    test=sm.query("window=='test' and cost_model=='official'")
    one=test[test.trial.eq('M4-4')].set_index('approach').loc[['A2','R','A3']]
    fig,axs=plt.subplots(1,2,figsize=(11.6,4.15))
    for ax,metric,label in zip(axs,['sharpe','ann_vol'],['Net Sharpe','Annual volatility (%)']):
        vals=one[metric]*(100 if metric=='ann_vol' else 1)
        ax.bar(range(3),vals,color=[MUTED,BLUE,ORANGE],width=.6)
        ax.set_xticks(range(3),one.index);ax.set_ylabel(label)
        ax.axhline(0,color=MUTED,lw=.8);clean(ax);ax.grid(axis='y',lw=.6)
        ax.grid(axis='x',visible=False)
        for x,v in enumerate(vals):ax.text(x,v, f'{v:.2f}',ha='center',va='top' if v<0 else 'bottom',fontsize=14)
        ax.margins(y=.17)
    fig.tight_layout();fig.savefig(dest/'iv.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11.9,3.72))
    ids=['M5-1','M5-2','M5-3','M5-4'];x=np.arange(4)
    for j,(ap,col) in enumerate([('A2',MUTED),('R',BLUE),('A3',ORANGE)]):
        vals=test[test.approach.eq(ap)].set_index('trial').loc[ids,'sharpe']
        ax.bar(x+(j-1)*.24,vals,.22,label=ap,color=col)
    ax.set_xticks(x,['Class-pooled HMM','Global-pooled HMM','Seasonality','Expiry (identity)'])
    ax.set_ylabel('Test-period net Sharpe');ax.axhline(0,color=MUTED,lw=.8)
    clean(ax);ax.grid(axis='x',visible=False);ax.grid(axis='y',lw=.6)
    ax.legend(frameon=False,ncols=3,loc='lower left');fig.tight_layout()
    fig.savefig(dest/'m5.png',dpi=180);plt.close(fig)
    active=pr.query("window=='test' and cost_model=='official' and comparison=='A3-A2' and not identity").copy()
    active['label']=[k+'  '+SHORT[k] for k in active.trial]
    fig,ax=plt.subplots(figsize=(10.22,5.35))
    ys=np.arange(len(active))[::-1]
    for y,r in zip(ys,active.itertuples()):
        color=ORANGE if r.trial.startswith('M4') else ('#77C3A4' if r.trial.startswith('M5') else BLUE)
        ax.plot([r.ci_low,r.ci_high],[y,y],color=color,lw=2)
        ax.scatter([r.d_sharpe],[y],color=color,s=22,zorder=3)
    ax.set_yticks(ys,active.label);ax.set_xlim(-.85,1.7);ax.axvline(0,color=MUTED,ls='--',lw=1)
    ax.set_xlabel('A3 minus A2 Sharpe, unadjusted 95% interval')
    clean(ax);fig.tight_layout();fig.savefig(dest/'forest.png',dpi=180);plt.close(fig)
    active['mde']=2.8*active.se_boot
    active=active.sort_values('mde',ascending=False)
    fig,ax=plt.subplots(figsize=(8.89,4.27));ys=np.arange(len(active))[::-1]
    ax.barh(ys,active.mde,color='#607D9F',label='80%-power detectable effect')
    ax.scatter(active.d_sharpe.abs(),ys,color=BLUE,label='Absolute estimated effect',zorder=3,s=25)
    ax.set_yticks(ys,active.trial);ax.set_xlabel('Sharpe difference');clean(ax)
    ax.legend(frameon=False,loc='lower right',fontsize=11);fig.tight_layout()
    fig.savefig(dest/'power.png',dpi=180);plt.close(fig)


def main():
    sm=pd.read_csv(OUT/'summary.csv');pr=pd.read_csv(OUT/'comparisons.csv')
    extra=pd.read_csv(OUT/'extra_strategy_comparisons.csv');bas=pd.read_csv(OUT/'baselines.csv')
    passed=audit(sm,pr);figures(sm,pr)
    data=dict(summary=sm.to_dict('records'),comparisons=pr.fillna('').to_dict('records'),
       extra=extra.to_dict('records'),baselines=bas.to_dict('records'),
       gates=json.loads((OUT/'gate_decisions.json').read_text()),
       season=pd.read_csv(OUT/'season_gates.csv').to_dict('records'),
       overlap=pd.read_csv(OUT/'overlap.csv').fillna('').to_dict('records'),audit=passed)
    (OUT/'slide_data.json').write_text(json.dumps(data,indent=2))
    lines=['# Post-hoc 80/20 rerun','',
      '**Development:** 2007-07-02–2022-12-05 (3,984 common trading dates).',
      '**Test:** 2022-12-06–2026-10-15 (997 dates). The split is floor(80% × 4,981), after every included learned model and all 13 contracts have warmed up. Earlier observations remain available for training.', '',
      'This is a post-hoc sensitivity analysis. The test years were previously viewed. Changing the split does not create a new untouched confirmation sample.', '',
      '## Design','',
      '- A2 = M2 combo × contract volatility targeting. R = the same combo × detector filter. A3 = the same combo × volatility targeting × the same filter.',
      '- 14 regime rules plus the portfolio-volatility benchmark M3-4 produce 15 comparable overlays. M3-5 turning points and M3-8 pure HMM change the signal itself and appear separately against their original controls and M2 A2.',
      '- Existing hyperparameters remain fixed. Annual, quarterly and online updates use only past observations available at the update date, including past test observations once available. This is rolling out-of-sample prediction, not a single model frozen for four years.',
      '- HMM, macro/news HMM, BOCPD, portfolio volatility and the two pooled HMMs already have causal full-history filter files. Their fixed predictions are reused by hash, and their portfolio returns and statistical comparisons are recalculated on the new split. Meta-labeling predictions, external-data filters and gate-dependent rules are extended or rebuilt as needed.',
      '- Original gate rules use pre-test data only. Decisions are frozen for test. Development results are descriptive because they use the selected gates.',
      '- Same 13 contracts, fixed 1/13 weights, one M1 engine, delay 1 and separate flat starts for each evaluation window. No ex-post risk matching.',
      '- Two cost scenarios: fixed research basis points and a 2026 fee-schedule snapshot. The snapshot excludes spread and slippage and is not an exact historical transaction-cost series.',
      '- Paired circular block bootstrap: 20-day blocks, 5,000 draws, seed 0, two-sided p. Intervals below are unadjusted 95% percentile intervals. Holm over 45 overlay comparisons per cost and 90 across both costs, separately by period. The focused family of 14 A3-minus-A2 regime tests is also reported.', '',
      '## What changes','',
      'No regime A3-minus-A2 gain survives Holm under either cost model. This supports the narrow statement that an incremental benefit is not established. It does not establish that every detector is useless or that volatility information overlap explains the outcome.',
      'The former claim that R has lower sample Sharpe than A2 and A3 for every active filter no longer holds. In the new test period, R beats A3 for five of nine active regime filters and trails it for the four HMM filters, under both cost models. No R-minus-A3 result survives Holm. Concurrent cuts show overlapping exposure decisions, not proof of harmful duplication.',
      'Agricultural seasonality now passes its development gate (Holm p = 0.0404, out-of-sample R² = 0.0112). Energy seasonality fails. Four M4 gates and expiry still fail, giving five identity rules with A3=A2 and R=A1.',
      'Fee-schedule baseline: development Sharpe A1 0.120, A2 0.343. Test Sharpe A1 −0.240, A2 −0.298. Test volatility falls from 8.02% to 4.82%, and drawdown from −22.65% to −11.31%. Lower risk does not imply better Sharpe.', '',
      '## Baselines','',md_table(bas.round(4)), '', '## All overlay results','']
    for window in ['development','test']:
      for cost in ['official','research']:
        s=sm[(sm.window==window)&(sm.cost_model==cost)]
        w=s.pivot(index=['trial','name'],columns='approach',values='sharpe').reset_index()
        p=pr[(pr.window==window)&(pr.cost_model==cost)&(pr.comparison=='A3-A2')]
        w=w.merge(p[['trial','d_sharpe','ci_low','ci_high','p_boot','p_holm_increment_14','p_holm_joint']],on='trial')
        lines+=['### '+window+' / '+cost,'',md_table(w.round(4).fillna('benchmark')),'']
    lines+=['## R comparisons in the test period','',md_table(pr.query("window=='test' and comparison!='A3-A2'")[['trial','cost_model','comparison','d_sharpe','ci_low','ci_high','p_boot','p_holm_joint']].round(4)), '',
      '## Additional signal strategies','',md_table(extra.round(4)), '',
      'These strategies also show no significant test improvement. They are not regime-filter ablations, because signal direction differs. Their multiplicity family is separate from the overlay tests.', '',
      '## Inputs and availability','',
      'CFTC legacy reports, scheduled FOMC meetings and implied-volatility series were extended through the test period. sources.json stores URLs and hashes. COT release delays use the original seven-day buffer plus conservative catch-up dates for the 2018/19, 2023 and 2025 publication backlogs. These are conservative availability assumptions, not reconstructed individual releases.',
      'CBOE supplies new VIX3M/OVX/GVZ observations after the old FRED prefix. Yahoo supplies daily ICE BofA MOVE after the old Investing.com prefix. The overlapping MOVE series pass the scale/correlation audit in audit.json. Public historical macro series remain subject to possible revisions, as in the original project.',
      'Meta-labeling extends the original causal yearly refits. Predictions reproduce the original completed-week prefix exactly through 2014-12-26. The old terminal partial week ending Wednesday 2014-12-31 is replaced by its natural complete week in the longer sample.', '',
      '## Reproduction','',
      'Run from the project root with the project Python environment. Set OMP_NUM_THREADS and OPENBLAS_NUM_THREADS to 4. Preserve the completed run before a rerun: the evaluator deliberately refuses to overwrite summary.csv.', '',
      '```text\npython -m backtest.resplit_common\npython -m backtest.resplit_data\npython -m backtest.resplit_data --prepare\npython -m backtest.resplit_models\npython -m backtest.resplit_gates\npython -m backtest.resplit_models --extras\npython -m backtest.resplit_evaluate\npython -m backtest.resplit_evaluate --extras\npython -m backtest.resplit_report\n```','',
      'Saved evidence: analysis_spec.json, code_manifest.json, evaluated_filter_manifest.json, audit.json, meta_training_audit.csv, gate_decisions.json, filter_coverage.csv, identity_checks.csv, summary.csv, comparisons.csv and daily return files. The original split outputs remain intact. The written project report has not been regenerated as part of this slide revision.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(passed,indent=2));print('Wrote report, slide data and five figures')


if __name__=='__main__':main()
