# Independent GTAA ETF proxy run

Run on 2026-09-29 UTC. Investable months: November 2007 through August 2026 (226 months), starting equity $100,000. The warmup begins January 2007. Rule: the unmodified five-asset 10-month monthly signal and 20% allocation per sleeve in `scanner/published_faber_gtaa.py`.

| Month-end metric | Timing portfolio | Five-ETF equal-weight comparison |
| --- | ---: | ---: |
| Ending equity | $257,563.40 | $259,768.67 |
| Annualized compound return | 5.15% | 5.20% |
| Maximum month-end drawdown | -10.54% | -46.56% |
| Annualized standard deviation of monthly returns | 6.72% | 12.84% |
| 2008 calendar return | -1.29% | -29.45% |
| 2022 calendar return | -3.73% | -10.64% |

This is **not a replication of Faber's paper**. The author says Global Financial Data's license prevents sharing the original data: https://mebfaber.com/2013/06/14/qtaa-update-conclustions-and-faqs/. His paper specifies S&P 500, MSCI EAFE, US ten-year government bonds, GSCI, NAREIT, and 90-day bills: https://mebfaber.com/wp-content/uploads/2016/05/SSRN-id962461.pdf.

This study substitutes Yahoo Finance adjusted closing prices of SPY, EFA, IEF, GSG, and VNQ, respectively. In particular, IEF tracks a 7–10 year Treasury ETF rather than Faber's ten-year bond index; GSG and VNQ are investable ETF exposures instead of the paper's GSCI and NAREIT index histories. Yahoo adjusted-close series may be revised and the chart endpoint is unofficial. Last available trading close is sampled each calendar month. For cash, FRED TB3MS is the monthly three-month Treasury bill bank-discount rate. The script converts the **previous month's** quoted annual rate to an estimated 91-day purchase price and accrues it for the current calendar month's number of days. This is an estimate, not a measured bill total-return series. Data are fetched fresh by the script and outputs may change when data providers revise histories.

No tax, commissions, bid-ask spreads, or slippage. Both curves rebalance monthly to 20% per sleeve. Drawdowns use month-end equity only. The comparison is to the same five ETF proxies, not S&P 500. Results cannot establish a future edge or match the 1973–2012 paper study.

Run from the repository root with the reference module already installed and the included input CSV extracted:

```bash
python run_faber_gtaa_etf_proxy.py
```

This default calculation uses the saved input without network access. To fetch fresh Yahoo and FRED data and overwrite the input CSV, use `python run_faber_gtaa_etf_proxy.py --refresh`. A provider timeout or revised data can change or interrupt a refresh.

The script saves `faber_gtaa_etf_proxy_input.csv` and `faber_gtaa_etf_proxy_results.csv`. The input CSV is included in the accompanying research archive so the recorded result can be recalculated without Yahoo or FRED access:

```bash
python -m scanner.published_faber_gtaa faber_gtaa_etf_proxy_input.csv recalculated.csv
```
