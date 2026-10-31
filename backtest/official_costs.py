"""Exact per-contract fees (m1/config/fee_schedule.json) as per-side bps of traded notional.

    bps_t = 10,000 x known fee subtotal per contract per side / (close_t x contract size)

`close` is data/final's unadjusted price of the held contract (never `close_adj`), so the
notional is that contract's value on day t, in the fee's own currency: every US product
is USD-quoted, and FESX/FGBL fees and notionals are both EUR, so no FX is needed.

Caveats, all making these rates a lower bound on real costs:
- the fees are M1's snapshot scenario (sources read 2026-10-20) applied to every past date;
- zero slippage and no bid-ask spread;
- FESX/FGBL are stand-ins (index level, modelled bond price) and only Eurex's exchange fee
  is known for them, so their rate is exchange-only;
- costs are charged on M1's traded exposure, so only detected rolls pay roll legs.
"""

from pathlib import Path

import pandas as pd

from m1.cost_model import execution_cost

# currency units per price point, from the exchanges' contract specifications
CONTRACT_SIZE = {
    "ES": 50,             # CME E-mini S&P 500: USD 50 x index
    "NK": 5,              # CME Nikkei 225 (USD, NKD): USD 5 x index
    "CL": 1_000,          # NYMEX WTI crude: 1,000 barrels, USD per barrel
    "NG": 10_000,         # NYMEX Henry Hub: 10,000 MMBtu, USD per MMBtu
    "GC": 100,            # COMEX gold: 100 troy oz, USD per oz
    "SI": 5_000,          # COMEX silver: 5,000 troy oz, USD per oz
    "ZN": 1_000,          # CBOT 10-year note: USD 100,000 face, price in points of par
    "6E": 125_000,        # CME euro FX: EUR 125,000, price in USD per EUR
    "6J": 12_500_000,     # CME Japanese yen: JPY 12,500,000, price in USD per JPY
    "ZC": 50,             # CBOT corn: 5,000 bushels, price in US cents per bushel
    "ZS": 50,             # CBOT soybeans: 5,000 bushels, price in US cents per bushel
    "FESX": 10,           # Eurex EURO STOXX 50: EUR 10 x index
    "FGBL": 1_000,        # Eurex Euro-Bund: EUR 100,000 nominal, price in percent of par
}


def fee_per_contract(code: str) -> tuple[float, str]:
    """Known fee subtotal per contract per side and its currency (M1's cash model)."""
    r = execution_cost(code, 1, "2026-10-20", mode="snapshot_scenario", allow_proxy=True, slippage_per_contract=0.0)
    return r["known_fee_subtotal"], r["currency"]


def fee_bps(close: pd.Series, code: str) -> pd.Series:
    """Per-side fee in bps of one contract's notional on each day."""
    fee, _ = fee_per_contract(code)
    notional = close.astype(float) * CONTRACT_SIZE[code]
    if (notional <= 0).any():
        raise ValueError(f"{code}: nonpositive unadjusted price")
    return 1e4 * fee / notional


def all_fee_bps(data_dir: Path) -> dict[str, pd.Series]:
    uni = pd.read_csv(data_dir / "universe.csv")
    return {c: fee_bps(pd.read_csv(data_dir / f"{c}.csv", index_col="date", parse_dates=True).close, c)
            for c in uni.code}
