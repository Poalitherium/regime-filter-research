"""Price supplied executions and optionally convert cash fees using USD/EUR."""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .cost_model import execution_cost
from .fx import FXRates,convert_amount


def price_ledger(ledger,*,fx=None,base_currency=None,fx_policy='exact',max_age_days=3,
                 default_slippage=0.,allow_proxy=False):
    needed={'trade_date','code','contract_id','side','quantity','order_type'}
    if not needed.issubset(ledger) or ledger.empty:
        raise ValueError('nonempty execution ledger and all required fields needed')
    if not np.isfinite(default_slippage) or default_slippage<0:
        raise ValueError('default cash slippage must be finite and nonnegative')
    rows=[]
    for event_id,row in enumerate(ledger.to_dict('records')):
        q=float(row['quantity'])
        if not np.isfinite(q) or not q.is_integer() or q<=0:
            raise ValueError('positive integer executed quantities required')
        if row['side'] not in {'buy','sell'} or not isinstance(row['contract_id'],str) or not row['contract_id'].strip():
            raise ValueError('buy/sell and actual contract identifier required')
        slip=row.get('slippage_per_contract',np.nan)
        defaulted=pd.isna(slip)
        slip=float(default_slippage if defaulted else slip)
        r=execution_cost(row['code'],int(q),str(row['trade_date']),mode='snapshot_scenario',
                         order_type=row['order_type'],allow_proxy=allow_proxy,slippage_per_contract=slip)
        r.update(event_id=event_id,contract_id=row['contract_id'],side=row['side'],
                 slippage_per_contract=slip,slippage_assumption='ZERO_SLIPPAGE_ASSUMPTION' if slip==0 else 'USER_SPECIFIED_SLIPPAGE_ASSUMPTION',
                 slippage_source='CLI default' if defaulted else 'ledger column')
        if base_currency:
            conv=convert_amount(r['known_fee_subtotal'],r['currency'],base_currency,r['trade_date'],fx,fx_policy,max_age_days)
            r.update(base_currency=base_currency,known_fee_subtotal_base=conv['amount_base'],
                     fx_date=conv['fx_date'],conversion_factor=conv['conversion_factor'],fx_policy=fx_policy,
                     slippage_base=r['slippage']*conv['conversion_factor'],
                     modeled_total_base=None if r['modeled_total'] is None else r['modeled_total']*conv['conversion_factor'])
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--executions',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('m1/report/tables/cash_execution_costs.csv'))
    p.add_argument('--fx-csv',type=Path)
    p.add_argument('--base-currency',choices=['USD','EUR'])
    p.add_argument('--fx-policy',choices=['exact','backward'],default='exact')
    p.add_argument('--max-fx-age-days',type=int,default=3)
    p.add_argument('--default-slippage-per-contract',type=float,default=0.)
    p.add_argument('--allow-proxy',action='store_true')
    a=p.parse_args()
    fx=FXRates.from_csv(a.fx_csv) if a.fx_csv else None
    result=price_ledger(pd.read_csv(a.executions),fx=fx,base_currency=a.base_currency,
        fx_policy=a.fx_policy,max_age_days=a.max_fx_age_days,
        default_slippage=a.default_slippage_per_contract,allow_proxy=a.allow_proxy)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(a.output,index=False)
    config={'fee_mode':'snapshot_scenario','fx_quote':'usdeur = EUR per 1 USD',
            'base_currency':a.base_currency,'fx_policy':a.fx_policy,'max_fx_age_days':a.max_fx_age_days,
            'default_slippage_per_contract':a.default_slippage_per_contract,
            'zero_slippage_is_an_assumption':True,'unknown_fee_components_remain_null':True}
    a.output.with_suffix('.json').write_text(json.dumps(config,indent=2)+'\n')
    print(f'{len(result)} executions priced; unknown totals remain blank; output: {a.output}')

if __name__=='__main__': main()
