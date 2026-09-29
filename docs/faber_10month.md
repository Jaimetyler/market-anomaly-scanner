# Published strategy reference: Faber 10-month timing model

Source: Mebane T. Faber, *A Quantitative Approach to Tactical Asset Allocation*, 2013 revision, pp. 20–21. https://mebfaber.com/wp-content/uploads/2016/05/SSRN-id962461.pdf

This is the project's first published strategy reference. It implements the paper's single-asset timing rule, not an optimized variant. It runs independently from the anomaly scanner and the experimental Turtle stock backtest.

## Fixed rules

- At each month-end, compare that month's total-return index level with the simple average of its latest 10 month-end levels, including the current level.
- Above average: hold the asset for the following month. Below average: hold 90-day Treasury bills for the following month.
- Equality is unspecified in the paper. This implementation retains the previous allocation (cash when no allocation exists); this convention must be disclosed in any replication.
- Start only after 10 observed months. No leverage, shorting, parameter tuning, or anomaly overlay.
- The paper assumes execution at the signal-day close and omits costs. The monthly return calculation represents the resulting next-month exposure; live execution at that same close may not be achievable after computing the final closing signal.

Input CSV must have `month,total_return_index,tbill_return`, where `month` is ISO `YYYY-MM-DD` for each consecutive calendar month, the index level includes reinvested income, and `tbill_return` is the decimal total return over that month (e.g. `0.003`). The asset's return in month *t* is `index[t]/index[t-1]-1`. The T-bill return in month *t* applies when cash was chosen at month-end *t-1*. Data sourcing and benchmark period must match the paper before claiming numerical replication.

Run: `python -m scanner.published_faber monthly.csv output.csv`. Test: `python -m unittest discover -s tests -p test_published_faber.py`.

This repository's existing split-adjusted daily stock bars are not the paper's S&P 500 total-return index or 90-day T-bill monthly returns. Do not substitute SPY closing prices or zero cash returns and call it a paper replication. Obtain and record an authoritative monthly total-return index and bill-return series, source/version, and comparison period before reporting performance. No published performance figures have been replicated yet.

## Existing Turtle experiment

Keep the current Turtle System 2 stock run unchanged as an experimental engine regression case. It is not an exact replication of the original futures strategy and should not be listed as a validated published strategy. Guarded seed 0/1/2 ending equity was approximately $116,258/$48,331/$49,706 from $100,000, respectively, before trading costs. Seven unresolved long-gap tickers were quarantined. No further ticker-by-ticker work is required to begin this independent benchmark.
