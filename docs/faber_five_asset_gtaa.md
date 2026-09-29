# Faber five-asset GTAA reference

This module implements the five-asset rule in Meb Faber, *A Quantitative Approach to Tactical Asset Allocation* (February 2013 update), sections “Step 2” and “Step 3”:
https://mebfaber.com/wp-content/uploads/2016/05/SSRN-id962461.pdf

| CSV column | Paper's monthly total-return series |
| --- | --- |
| `us_stocks` | S&P 500 |
| `foreign_stocks` | MSCI EAFE |
| `us_10y_bonds` | US 10-year government bonds |
| `real_estate` | NAREIT Index |
| `commodities` | Goldman Sachs Commodity Index |
| `tbill_return` | Monthly decimal return of 90-day Treasury bills |

Add `month` as an ISO date (e.g. `2020-01-31`). Supply at least 11 consecutive monthly rows. The first five series must be positive total-return **index levels**, including distributions and income, not monthly percentage returns or unadjusted ETF prices. `tbill_return` is a decimal monthly return (`0.002` means 0.2%), not a quoted annual yield. The paper used Global Financial Data; these source series are not bundled here.

At each month end, compare each index level with the simple average of its latest ten monthly levels, including this month. Above means its 20% allocation holds that asset for the next month; below means it holds T-bills. An exact equality retains the prior allocation (the paper does not state a tie rule). The first signal occurs after ten levels; the first realized portfolio return is the eleventh month. Each month, rebalance the five allocations to 20% of portfolio equity. The comparison curve rebalances all five asset classes to 20% each month. Trading at the signal-day close is assumed; commissions, slippage, and taxes are excluded as in the paper.

Run from the repository root:

```bash
python -m unittest discover -s tests -p 'test_published_faber_gtaa.py' -v
python -m scanner.published_faber_gtaa monthly_five_assets.csv gtaa_result.csv
```

Example input header:

```csv
month,us_stocks,foreign_stocks,us_10y_bonds,real_estate,commodities,tbill_return
```

The output records each asset's index level, moving average, allocation held during the month and signal for the following month, plus the strategy and equal-weight comparison returns and equities. The initial warmup row has blank portfolio returns. A numerical replication of the paper requires its historical series and cash returns; substituting ETFs or estimated T-bill returns produces a separate proxy study, even when the trading rule is identical.
