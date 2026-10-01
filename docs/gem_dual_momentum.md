# GEM dual momentum: locked specification and ETF study

Specification locked October 1, 2026 UTC, before running this implementation
against real historical ETF returns. No parameter search or result-based tuning.
Base repository inspected: `58f6bf3c60959de6b0b0084546b53271435f0ced`.

## Rules

At each completed calendar month-end, calculate trailing 12-month total returns
using 13 month-end levels. Include the latest month; do not skip a month.
First compare US stock return with Treasury-bill return over the SAME window.
If US does not beat bills, select aggregate bonds regardless of foreign strength.
Otherwise select the stronger US or ex-US equity market. Hold one asset at 100%.
Repeat monthly. An allocation selected at month t earns only month t+1 returns.
This follows the absolute-first version described in Antonacci's FAQ, rather
than the relative-first variation which can produce different allocations.

Explicit tie conventions: US equal to bills selects bonds; US equal to foreign
selects US. These choices are documented implementation conventions, not claims
that the cited sources specify all equality cases. No stops, leverage, lookback
optimization, or bond momentum gate is added. Bonds can lose money.

## Source and instrument mapping

Original rule sources (accessed October 1, 2026 UTC):

- https://www.optimalmomentum.com/extended-backtest-of-global-equities-momentum/
- https://www.optimalmomentum.com/faq/
- https://www.optimalmomentum.com/global-equities-momentum/

| Role | Original model exposure | Research ETF proxy |
| --- | --- | --- |
| US equities | S&P 500 | SPY |
| Foreign equities | MSCI ACWI ex-US | VEU, FTSE All-World ex-US (index substitution) |
| Defensive holding | US Aggregate Bond index | AGG |
| Absolute momentum hurdle | Treasury-bill total return | BIL, 1–3 month Treasury-bill ETF |

VEU includes developed and emerging markets; it is not the original MSCI index.
BIL is used as a total-return hurdle, never as GEM's defensive holding. Its fees
and index construction differ from the original bill series. Never substitute
an annualized bill yield for a trailing 12-month total return. Faber's comparison
uses its existing SPY/EFA/IEF/VNQ/GSG mix, with BIL replacing estimated cash returns.

Provider: Yahoo daily dividend/split-adjusted closes, through the existing
`scanner.gtaa_signals.fetch_daily` function. Adjusted closes are total-return
proxies subject to provider revisions, not independently audited index history.
No new package dependency or API key is needed. Existing `tzdata` is required on
Windows, as with the Faber runner. Fetching still requires internet/provider access.

The common sample starts after every ETF has data. The first observed month of
each ETF is excluded conservatively; twelve more intervals establish GEM's first
signal, and returns begin the following month. No synthetic pre-inception history
is added. Exact sample dates are printed after data retrieval.

## Run

From the repository root with the existing virtual environment active:

```bash
python -m unittest discover -s tests -p 'test_gem*.py' -v
python -m scanner.strategy_cli list
python -m scanner.gem_research --refresh
```

The first command tests known-rule outcomes and report/data behavior using
synthetic fixtures; these tests do not establish real-market performance.

The research command defaults to $100,000 and 10 bps per traded dollar per side.
It prints a comparison and saves `data/research/gem_etf_proxy/report.md`,
`summary.json`, five strategy path CSVs, `gem_results.csv`, `gem_input.csv`,
`monthly_panel.csv`, and `daily_snapshot.json`.

For an identical offline rerun, use the same as-of date shown in the report:

```bash
python -m scanner.gem_research --as-of 2026-09-30
```

On October 1 use `--refresh` to capture September's completed month. The fetcher
excludes the as-of day's bar. The offline runner rejects stale input at a month
boundary: either refresh or use the original as-of date deliberately.
Use a separate `--output-dir` to retain multiple studies/snapshots. Successful
runs replace outputs in the selected research directory; they do not touch Faber's
paper ledger or create a GEM account.

Cost sensitivity without changing strategy rules:

```bash
python -m scanner.gem_research --input-csv data/research/gem_etf_proxy/monthly_panel.csv --as-of 2026-09-30 --cost-bps 0 --output-dir data/research/gem_no_cost
```

The generic catalog also supports `python -m scanner.strategy_cli run gem
INPUT.csv OUTPUT.csv` (one line). Its input columns are `month,us_stocks,
foreign_stocks,aggregate_bonds,tbill_index`, all positive total-return LEVELS.
It defaults to zero trading costs and $100,000, like the existing reference CLIs.
The research runner's `gem_input.csv` has exactly that schema.

## Comparison conventions

Five paths are compared on identical dates and identical starting capital:
GEM; Faber GTAA5; Faber's five-ETF equal-weight monthly rebalanced benchmark;
GEM's 45% US/28% foreign/27% bonds monthly rebalanced benchmark; SPY buy and hold.
The 45/28/27 weights follow the official GEM benchmark page; monthly rebalancing
is our explicit comparison convention. All initial purchases incur costs.

Signals are causal, but these monthly reference calculations assume the signal
close as the rebalance price. Actual next-session fills and overnight gaps cannot
be reproduced from monthly data. Before paper trading, validate a daily execution
layer separately; do not call this an executable trading backtest.

Costs apply to gross bought plus sold weight relative to pretrade equity, then
remaining capital is allocated proportionally to target weights. This is an
approximation, not an exact fee-aware order solver. An initial allocation trades
100%; a complete switch trades 200%. Continuing single-asset holdings have no
rebalance costs. Multi-asset costs account for portfolio weight drift. No final
liquidation is assumed. ETF expenses are embedded in adjusted returns; taxes,
additional management fees and market impact are omitted.

CAGR uses the number of actual monthly return intervals. Maximum drawdown includes
the initial $100,000 baseline but observes month-ends only. Volatility is sample
monthly standard deviation times sqrt(12). Sharpe uses the mean and sample
standard deviation of monthly excess returns over BIL, times sqrt(12). Worst
calendar year includes only years with all twelve return months. Undefined
statistics are null/n/a. Gross and cost-adjusted results are reported separately.

Daily data must share an actual month-end session; no forward filling or silently
dropping missing internal months. The compact end-of-month calendar covers the
ETF history from 2007, including month-end Good Friday and Memorial Day closures.
An exceptional future month-end exchange closure may require a calendar update.
Offline files may use actual trading month-end or calendar month-end labels;
the caller is responsible for their provenance and completeness. A SHA-256
digest records the normalized monthly input used in each report.

## Next gate

Review the first real-data report and spot-check switching months and distributions.
Only then add GEM current sizing and a separate paper ledger. This patch does not
change the Faber paper workflow, data-fetch cutoff, or existing Faber calculations.
