# Current Faber GTAA5 ETF proxy signals

From the repository root, with the existing Python environment activated:

```bash
python -m scanner.strategy_cli signals faber-gtaa5 --equity 100000
```

The command fetches Yahoo daily adjusted closes for SPY, EFA, IEF, VNQ and GSG,
using the same ETF mapping as the existing proxy study. No API key or new
package is required. It writes `signals.md`, `signals.csv`, `signals.json`, and
`daily_snapshot.json` under `data/research/faber_current_signals/`, and prints
the readable report. A provider failure exits with status 2 rather than
substituting unadjusted prices or stale signals. Existing report files are not
removed on failure; always check their `as_of` and `signal_date`.

The default as-of date is today in America/New_York. Only the **previous
calendar month** can generate confirmed targets; the as-of day's daily bar is
excluded to avoid partial data. Run on October 1 to confirm September's
month-end signal. A midmonth run reports the allocation already applicable
to that month and a clearly labeled developing preview, not a new trigger.
Require the latest completed month, consecutive unique months, aligned ETF
month-end dates within five calendar days of month-end, and daily quotes no
older than seven calendar days. This is a conservative freshness check, not
an exchange-calendar certification.

Signals reuse `scanner.published_faber_gtaa.backtest`, including its equality
rule. Each asset independently receives either 20% of total equity or zero;
the unused sleeve stays in cash. Signal comparisons use adjusted levels;
whole-share target estimates use recent **unadjusted** daily closing prices,
round down, and leave the remainder in cash. Adjusted levels are not executable
prices. Rebalance all active sleeves monthly, including sleeves whose model
signal did not change.

Without holdings the report gives model changes and target positions, not
actual order quantities. To create a paper rebalance against your holdings,
write a CSV with columns `ticker,shares`; omit zero positions if preferred:

```csv
ticker,shares
SPY,30
EFA,20
```

```bash
python -m scanner.strategy_cli signals faber-gtaa5 --equity 100000 --holdings-csv holdings.csv
```

Equity means total current account value **including cash**; it is supplied by
the user, not fetched from a broker. Sizing is indicative at the quoted daily
close. Refresh prices, execute paper sells before buys, and account for costs
and cash before recording fills. Signals formed at month-end can first be
acted on after the close, in the next session. The research backtest's
signal-close fill assumption does not establish executable paper/live results.
This command submits no broker orders and does not track fills.

For an offline monthly CSV (same schema as the existing strategy input):

```bash
python -m scanner.strategy_cli signals faber-gtaa5 --input-csv faber_gtaa_etf_proxy_input.csv --as-of 2026-09-29
```

Offline input generates allocations without share sizing or developing
preview. It must contain at least 11 consecutive months through the latest
completed month. As-of is a data cutoff, not point-in-time-vintage replication;
current Yahoo adjusted historical prices can reflect later adjustments.
Snapshot JSON preserves fetched observations for audit. Zero `tbill_return`
is used internally because only signals are taken from the backtest; no
performance or estimated cash returns are reported.

ETF proxies differ from Faber's original total-return indices. Rule source:
https://mebfaber.com/wp-content/uploads/2016/05/SSRN-id962461.pdf

Verification:

```bash
python -m unittest discover -s tests -p 'test_gtaa_signals.py' -v
```
