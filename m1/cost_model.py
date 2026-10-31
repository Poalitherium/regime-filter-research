"""Cash fees for explicit futures executions; no signals, timing or P&L."""
import json
import math
from datetime import date
from numbers import Integral
from pathlib import Path


def load_schedule():
    return {r['code']: r for r in json.loads(
        (Path(__file__).parent/'config/fee_schedule.json').read_text())}


def _nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be finite and nonnegative')
    return float(value)


def position_legs(before, after):
    """Signed integer holdings keyed by actual expiry ID -> executed quantities.

    Same expiry: abs(after-before). Different expiries: close old/open new.
    These are ordinary outright legs, not exchange-listed spread orders.
    """
    for holdings in (before, after):
        for contract, count in holdings.items():
            if not isinstance(contract, str) or not contract.strip():
                raise ValueError('actual expiry ID required')
            if isinstance(count, bool) or not isinstance(count, Integral):
                raise ValueError('holdings must be integer contract counts')
    return [{'contract_id': k, 'side': 'buy' if after.get(k,0)-before.get(k,0)>0 else 'sell',
             'quantity': abs(after.get(k,0)-before.get(k,0))}
            for k in sorted(before.keys() | after.keys()) if after.get(k,0)!=before.get(k,0)]


def execution_cost(code, quantity, trade_date, *, mode, order_type='outright',
                   allow_proxy=False, slippage_per_contract=None):
    """Apply a sourced snapshot scenario, NOT a historically matched tariff.

    quantity is a positive executed contract count (one side), never exposure.
    Slippage is a separately supplied assumption in the fee's native currency.
    Unknown fee components remain None; known_subtotal is never an all-in fee.
    """
    schedule=load_schedule()
    if code not in schedule:
        raise ValueError(f'unknown product: {code}')
    if isinstance(quantity,bool) or not isinstance(quantity,Integral) or quantity<=0:
        raise ValueError('quantity must be a positive integer')
    date.fromisoformat(trade_date)
    if mode != 'snapshot_scenario':
        raise ValueError('historical fee coverage unverified; explicitly select snapshot_scenario')
    if order_type != 'outright':
        raise ValueError('spread/block/delivery fees require a separately verified schedule')
    r=schedule[code]
    if r['proxy'] and not allow_proxy:
        raise ValueError('proxy data: explicit hypothetical futures permission required')
    result={k:r[k] for k in ['code','fee_symbol','currency','profile','observed_date','proxy']}
    result.update(quantity=int(quantity),trade_date=trade_date,mode=mode)
    for key in ['exchange_fee','broker_fee','regulatory_fee']:
        result[key]=None if r[key] is None else round(quantity*r[key],10)
    complete=all(result[k] is not None for k in ['exchange_fee','broker_fee','regulatory_fee'])
    result['fee_components_complete']=complete
    result['known_fee_subtotal']=round(sum(result[k] or 0 for k in ['exchange_fee','broker_fee','regulatory_fee']),10)
    result['slippage']=None if slippage_per_contract is None else quantity*_nonnegative(slippage_per_contract,'slippage_per_contract')
    result['modeled_total']=None if not complete or result['slippage'] is None else round(result['known_fee_subtotal']+result['slippage'],10)
    return result


def convert_cash(amount, currency, base_currency, fx=None):
    """fx maps native currency -> units of base currency per one native unit.

    Caller must provide date-matched FX rates. Never adds USD to EUR directly.
    """
    amount=_nonnegative(amount,'amount')
    if currency==base_currency:
        return amount
    if fx is None or currency not in fx:
        raise ValueError(f'missing {currency}/{base_currency} conversion rate')
    rate=_nonnegative(fx[currency],'FX rate')
    if rate==0:
        raise ValueError('FX rate must be positive')
    return amount*rate


def cash_to_bps(cost, actual_traded_notional):
    """Same-currency executed notional required; do not use adjusted proxy prices."""
    cost=_nonnegative(cost,'cost')
    n=_nonnegative(actual_traded_notional,'actual_traded_notional')
    if n==0:
        raise ValueError('actual traded notional must be positive')
    return 10000*cost/n
