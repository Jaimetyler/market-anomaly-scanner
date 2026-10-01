# GEM next-session execution audit

This extension measures the difference between the existing idealized month-end
close calculation and trading at the following exchange session's opening price.
It also marks the portfolio at every daily close to expose losses hidden by
monthly sampling. The original GEM signal rules are unchanged.

## Run and reproduce

```bash
python -m unittest discover -s tests -p 'test_gem_execution*.py' -v
python -m scanner.gem_execution --as-of 2026-09-30
```

The delivery includes the frozen daily OHLC snapshot and reports under
`data/research/gem_execution`. The command above replays that snapshot offline.
For a new snapshot in a separate directory:

```bash
python -m scanner.gem_execution --refresh --output-dir data/research/gem_execution_latest
```

The daily fetch excludes the as-of day's bar. Confirmed monthly signals and the
audit end date exclude the current calendar month. A saved snapshot requires
its original as-of date. No package installation or new API key is required.

## Execution order

1. Calculate GEM's 12-month signal after the final session's close each month.
2. Leave the existing holding in place overnight.
3. At the next exchange session's open, value the old holding at its own open.
4. If the target changed, sell the old holding and buy the new one at its open,
   deducting the configured cost estimate. Unchanged targets do not trade.
5. Mark the portfolio at every daily close.

Initial funding remains cash at the June 30, 2008 baseline, with its first ETF
entry on July 1. Uninvested initial cash earns zero for that overnight interval.
The terminal month-end signal has no fill inside the historical window. No
forced final sale or exit cost is included. SPY buy-and-hold enters at the same
next open as GEM, rather than inheriting the earlier closing-price entry.

The signal-close reference follows the same holding sequence and cost model but
switches at the signal date's close. Its close valuation is recorded before the
new trade cost, matching the existing monthly reference convention. The software
requires every monthly reference value to reconcile, and checks that multiplying
all trade-event overnight timing factors reproduces the difference in final wealth.

## Price and accounting conventions

The provider supplies daily open, close and dividend/split-adjusted close. The
audit calculates `adjusted_open = open * adjusted_close / close` to put both ends
of a holding period on the same adjustment basis. This common adjustment convention
is also implemented by [yfinance auto_adjust](https://github.com/ranaroussi/yfinance/blob/main/yfinance/utils.py).
No yfinance dependency is introduced. Calculations use fractional total-return
index units. They are not actual share orders or a complete dividend cash ledger.

Adjustments represent reinvested distributions through historical price factors,
not actual cash entitlement/payment dates. An ex-dividend day's adjusted-open
factor can affect overnight attribution. Whole-share rounding, distribution cash,
splits in a share ledger, settlement and order lifecycle still require separate
paper-account implementation. Provider history may be revised; a frozen snapshot
and SHA-256 digest allow this run to be reproduced.

Default costs are 10 basis points per traded buy/sell dollar, using pretrade equity
as the approximation. Entry costs 0.1%; a full switch costs 0.2%; holding incurs no
additional trading charge. This matches the monthly study's convention. The provider
daily open is a reference price, not a guaranteed auction fill. Liquidity, partial
fills, spreads beyond the selected cost allowance, taxes, and market impact are
not modeled. No actual or simulated paper-account orders are created by this audit.

## Data validation and calendar

All four tickers must contain every expected daily session throughout the shared
full-month history. The code rejects duplicate dates, stale snapshots, zero/null/
non-finite prices, and absent sessions. It detects a day missing from all tickers,
not just differences between their date sets. It never forward-fills prices or
silently delays an order to an arbitrary later day. First observed months are
excluded, then 13 month-end levels establish the first signal, as in the prior study.

The calendar covers 2007–2028 regular US equity holidays, including the New Year
Saturday exception and Juneteenth from 2022, plus known full-market closures in
2007, 2012, 2018 and 2025. Early-close days remain trading sessions. Calendar
rules were checked against [NYSE trading hours](https://www.nyse.com/markets/hours-calendars)
and the [exchange_calendars XNYS implementation](https://github.com/gerrymanoim/exchange_calendars/blob/master/exchange_calendars/exchange_calendar_xnys.py).
Unexpected future exchange closures should cause validation to stop until the
calendar is updated. This small scoped calendar is not a general exchange-calendar
library. Dates outside the supported years are rejected.

## Statistics

Daily drawdown includes the original capital baseline and uses daily closes.
Intraday drawdowns may be larger. The report separately retains month-end
drawdown, since the two measures must not be compared as though identical.
Daily volatility uses the sample standard deviation of daily returns times
sqrt(252). CAGR uses the same count of monthly return intervals as the monthly
comparison. Monthly summary measures use actual monthly portfolio marks; the
Sharpe measure in JSON uses excess monthly returns over the BIL proxy.

## Completed validation and findings

The 17 new tests cover delayed entry, ownership of overnight moves, switching
costs, unchanged holdings, terminal signals, causal fills, hidden daily drawdowns,
calendar anomalies, missing sessions, price adjustment consistency, and offline
reproducibility. The historical daily signal-close path also reconciled every
monthly equity mark against the existing GEM calculation.

Snapshot as-of September 30, 2026. Baseline June 30, 2008; returns through
August 31, 2026: 218 months and 4,570 daily return sessions. $100,000 initial
capital; 10 bps per traded side:

| Model | Ending value | CAGR | Worst daily-close drawdown | Worst month-end drawdown |
| --- | ---: | ---: | ---: | ---: |
| GEM signal-close reference | $479,986.40 | 9.02% | -33.72% | -19.85% |
| GEM next-session open | $448,210.45 | 8.61% | -33.72% | -21.64% |
| SPY from the same next open | $841,922.16 | 12.44% | -47.17% | -41.80% |

Next-open execution reduced GEM's final wealth by $31,775.95 (6.62% relative
to the signal-close result). There were 29 switches plus initial entry. A second
calculation, independent of the strategy/execution modules, recomputed wealth
from each trade's provider open/close data and matched $448,210.45.

The worst daily-close GEM decline ran from February 19 to March 23, 2020.
The simulated account fell from $253,663.63 to $168,135.14 while holding SPY.
GEM switched to AGG at the April 1 open. This illustrates that a monthly exit
rule can leave the portfolio exposed to a large loss inside the month.

The refreshed same-snapshot signal-close result differs from the earlier report
by less than $1. That small provider adjustment difference is separate from
the measured opening-execution effect. The report does not establish real-market
fills or validate a live paper ledger.
