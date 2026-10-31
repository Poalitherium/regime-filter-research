"""Readable report for the completed frozen-filter ablation (no new backtest)."""
from pathlib import Path
import json
import pandas as pd

root=Path(__file__).resolve().parents[1]
out=root/'backtest_outputs/regime_ablation'
s=pd.read_csv(out/'summary.csv')
p=pd.read_csv(out/'comparisons.csv')
o=pd.read_csv(out/'overlap.csv')
v=pd.read_csv(out/'reproduction_checks.csv')
reg=s[(s.cost_model=='official') & (s.kind=='regime_filter')]
ids=list(dict.fromkeys(reg.trial))
wide=s[s.kind.eq('regime_filter')].pivot(index=['trial','cost_model'],columns='approach',values='sharpe')
r_below=bool(((wide['R']<wide['A2']) & (wide['R']<wide['A3'])).all())
assert len(ids)==8 and len(wide)==16 and r_below
assert len(v)==36 and v.max_abs_daily_error.max()<1e-12

def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])

rows=[]
for tid in ids:
    a=reg[reg.trial==tid].set_index('approach')
    rows.append([tid,a.loc['R','name'],a.loc['R','start'],*[f'{a.loc[ap,"sharpe"]:.3f}' for ap in ['A1','A2','R','A3']]])
result=["# Frozen-filter ablation: removing volatility targeting", "",
"This is a post-hoc supplementary analysis requested on 30 October 2026. It is separate from the original 17-trial registered family. It adds R = raw trend signal × existing filter and retains A1, A2 and A3 for comparison.", "",
"## Result", "",
"For all eight active regime filters, R has a lower sample Sharpe than both A2 and A3 under both existing cost models. The portfolio-volatility benchmark gives the same ordering. Every new unadjusted 95% interval contains zero; none of the new differences is significant after Holm correction. This does not establish population-level superiority of A2 or A3.", "",
"The ablation does not support removing volatility targeting as a remedy for excessive risk reduction. Simultaneous cuts occur, but they do not by themselves show harmful redundancy or identify information overlap.", "",
"## Sample Sharpe ratios (exact contract fees)", "",
table(['Trial','Filter','Start','A1','A2','R (no VT)','A3'],rows), "",
"All samples end on 2014-12-31. The three strategies within each trial share the same dates and contract book. Start dates differ across trials, so this table must not be used to rank different filters on a common sample.", "",
"## Risk and exposure", "",
"Example: M3-1, the frozen in-market HMM, on its matched 2004-07-01 to 2014-12-31 sample:", ""]
a=reg[reg.trial=='M3-1'].set_index('approach')
result.append(table(['Approach','Mean return p.a.','Volatility p.a.','Sharpe','Max drawdown','Mean gross exposure','Turnover p.a.','Costs p.a.'],[[ap,f'{a.loc[ap,"ann_mean_return"]:.2%}',f'{a.loc[ap,"ann_vol"]:.2%}',f'{a.loc[ap,"sharpe"]:.3f}',f'{a.loc[ap,"max_drawdown"]:.2%}',f'{a.loc[ap,"gross_exposure"]:.3f}x',f'{a.loc[ap,"turnover_pa"]:.2f}x',f'{a.loc[ap,"cost_pa"]:.3%}'] for ap in ['A1','A2','R','A3']]))
result+=['',"R and A3 have almost identical mean gross exposure (about 0.256x), but R has 4.42% volatility versus A3's 2.68%. Removing volatility targeting changes the distribution of risk across contracts and dates, rather than simply restoring exposure. The HMM filter itself is responsible for much of the reduction relative to A1/A2.", '',
"The reported overlap measure counts active contract-days with a nonzero trend signal, V < 1 and F < 1 (tolerance 1e-10). It measures simultaneous reductions relative to unit exposure, not evidence that both rules contain the same information. Soft filters slightly below one count as reductions. The separate weighted measure weights by the absolute raw signal.", '',
table(['Trial','Both rules cut','A3 / A2 absolute target exposure','A3 / R absolute target exposure'],[[r.trial,f'{r.both_cut_share:.1%}',f'{r.a3_target_abs_ratio_a2:.3f}',f'{r.a3_target_abs_ratio_r:.3f}'] for r in o.itertuples()]), '',
"## New paired comparisons (exact contract fees)", '',
table(['Trial','Comparison','Delta Sharpe','95% CI','Raw p','Holm p'],[[r.trial,r.comparison,f'{r.d_sharpe:+.3f}',f'[{r.ci_low:+.3f}, {r.ci_high:+.3f}]',f'{r.p_boot:.4f}',f'{r.p_holm_new:.4f}'] for r in p[(p.cost_model=='official') & p.new_comparison].itertuples()]), '',
"## Method and boundaries", '',
"- Existing combo trend signals, filter CSVs and all parameter choices are reused unchanged. No model is retrained, no parameter is selected, and no ex-post risk rescaling is applied.",
"- M1 accounting: close-t target executes at close t+1 and first earns at t+2, sample starts flat, transaction costs accrue on execution, and roll turnover includes closing and opening positions. Portfolio weights remain fixed at 1/13.",
"- Existing official fees (zero added slippage) and research bps costs are both reported. Risk, gross exposure, turnover, net/gross Sharpe and costs are in summary.csv. Max drawdown includes initial wealth of 1. Calmar uses CAGR divided by absolute maximum drawdown, following m1.backtest.performance; older backtest.metrics reports an arithmetic-return version.",
"- New inference uses the existing paired circular block bootstrap, 20-day blocks, 5,000 replications, seed 0, two-sided p-values and unadjusted 95% percentile intervals. Holm correction covers 18 new R-A2 and R-A3 comparisons per cost model (nine overlays including the portfolio-volatility benchmark). A3-A2 is reproduced as a reference and does not enter this new family.",
"- Six failed-gate rules remain identity filters: M4-1, M4-2, M4-3, M4-5, M5-3 and M5-4. Their registered rule implies R=A1 and A3=A2 exactly, so there is no active filter to ablate. Exploratory rejected filters are not reactivated.",
"- M3-5 and M3-8 change the underlying trading signal and are excluded from this frozen-filter ablation. M3-4 is a portfolio-volatility scaling benchmark, not one of the eight regime filters; its multiplier can exceed one and was originally calculated using A2 exposures.",
"- Existing filters may have been estimated or designed for A2 returns. This tests those frozen rules after removing the A2 scaler, not the best independently trained regime-only strategy.",
"- Development ends in 2014. No new evaluation uses the 2015-2026 hold-out. These results cannot support a new out-of-sample confirmation claim.",
"- All 36 original A2/A3 daily-return reconstructions match archived returns to a maximum absolute error of "+f'{v.max_abs_daily_error.max():.3g}'+". Checks abort on a date mismatch or numerical discrepancy. The input manifest records SHA-256 hashes before evaluation.", '',
"## Reproduction", '',
"Run `.venv/Scripts/python.exe -m backtest.regime_ablation --out-dir backtest_outputs/regime_ablation_new` from the project root. A fresh output directory is required. The frozen specification is analysis_spec.json; full metrics are summary.csv; inference is comparisons.csv; overlap diagnostics are overlap.csv; and per-trial daily ledgers are the *_daily.csv files.", '',
"## Slide 2", '',
"Keep A3 versus A2 as the original incremental comparison. Add R explicitly as the post-hoc ablation, and distinguish substitution from adding a filter. Say that removing volatility targeting did not improve sample Sharpe in any of the eight active filters, while the differences remain statistically inconclusive. Do not describe this as proof that regime information is redundant."]
(out/'REPORT.md').write_text('\n'.join(result)+'\n',encoding='utf-8')
slide_data=dict(active_filter_count=8,r_below_a2_and_a3_both_costs=r_below,all_new_ci_include_zero=bool(((p[p.new_comparison].ci_low<0)&(p[p.new_comparison].ci_high>0)).all()),hmm={ap:{k:float(a.loc[ap,k]) for k in ['sharpe','ann_vol','gross_exposure','max_drawdown']} for ap in ['A2','R','A3']})
(out/'slide_data.json').write_text(json.dumps(slide_data,indent=2),encoding='utf-8')
print(json.dumps(slide_data,indent=2))
