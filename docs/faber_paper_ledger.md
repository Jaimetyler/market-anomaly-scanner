# Faber paper account ledger

This local account records simulated fills, share holdings, cash, average-cost
basis, realized P&L, income, and dated market-value snapshots in SQLite. It
never connects to a broker. Monetary balances are integer cents; fills use
decimal arithmetic. The default account is
`data/research/faber_paper/account.sqlite`.

## First run

From the repository root with your existing virtual environment active:

```bash
python -m scanner.paper_cli init --equity 100000
python -m scanner.strategy_cli signals faber-gtaa5 --equity 100000
python -m scanner.paper_cli plan
```

Initialization is cash-only and refuses to overwrite an existing account.
Starting equity is an illustrative paper-account size. Planning reads the
current confirmed signals JSON and derives target whole-share positions
from the **ledger's own cash and positions**, ignoring the signal report's
illustrative account equity. Each active ETF keeps its 20% sleeve, with the
remaining sleeves in cash. Every month's plan also rebalances existing
holdings to their sleeve weights. Planning records no fills or P&L.

Optional assumptions are set when the plan is created:

```bash
python -m scanner.paper_cli plan --fee 1 --slippage-bps 5
```

The fee is dollars per filled order. Slippage is adverse basis points on
each user-supplied raw closing price: higher buys and lower sells. Defaults
are zero, explicitly frictionless. Target quantities are fixed at planning;
they are not recalculated from subsequent fill prices.

## Record a simulated fill

Fill dates must be later than **plan creation**, later than the confirmed
signal, and in the allocation month. They must also be completed dates
before today in America/New_York. For example, a plan created September 29
can use September 30's close when recorded October 1. It cannot buy at
September 28's already-known price. A September plan cannot fill in October
using August's signal. Regenerate signals and plan for the new month.

```bash
python -m scanner.paper_cli prices
python -m scanner.paper_cli fill
```

`prices` fetches Yahoo raw daily closes, excludes today's daily bar, and saves
`prices.csv`. It requires all five ETF dates to agree and be no older than
seven calendar days. It does not certify exchange calendars. Alternatively,
provide your own CSV and retain its source externally:

```csv
ticker,date,close
SPY,2026-09-30,770.00
EFA,2026-09-30,105.00
IEF,2026-09-30,90.00
VNQ,2026-09-30,91.00
GSG,2026-09-30,35.50
```

These are **example prices**, not market observations. Use actual dated data.

```bash
python -m scanner.paper_cli fill --prices-csv your_prices.csv
```

This is a simulated completed-close fill assumption, not an observed broker
execution. The ledger preserves reference close, modeled price, date,
quantity, fees, cash after each fill, and realized P&L. Sells precede buys.
The entire rebalance commits in one transaction. Insufficient cash, stale
prices, changed account state, or a duplicate execution rejects the batch
without recording any partial fills. No leverage or short positions.

## Mark and review

```bash
python -m scanner.paper_cli prices
python -m scanner.paper_cli mark
python -m scanner.paper_cli status
```

Each mark records raw-price market values plus cash. Identical repeated
marks are harmless; changing a previously recorded valuation is rejected.
Record fills and income before marking that session. Chronology cannot move
backward. `status` works offline and states the last recorded valuation date;
it does not fetch a current quote. Marks form a sparse record of observations,
not a complete daily return/drawdown series.

All state-changing commands export `account.md`, `ledger.json`, `holdings.csv`,
`fills.csv`, and `equity.csv` beside the database. Planning additionally exports
`plan.json`. Monetary CSV fields ending in `_cents` are integer cents, not
dollars. `holdings.csv` is compatible with the signal command's
`--holdings-csv` option. The SQLite database is authoritative; JSON/CSV exports
are reports, not an import/restore interface. Keep the account database when
moving machines: it lives in git-ignored `data/research/` and is not pushed
with the code.

## Income and monthly updates

Yahoo adjusted levels are for signal calculations; paper holdings are valued
with raw prices. ETF distributions and interest are **not automatically
credited**. Record observed paper-account credits explicitly, before the
session's mark, using a unique reference to prevent duplicates:

```bash
python -m scanner.paper_cli income --type dividend --amount 25.50 --date 2026-09-30 --reference SPY-2026-09-30
```

Income is a fixed starting-account return component, not an external deposit.
The ledger doesn't infer entitlement or validate credit amounts. Unrecorded
distributions/interest are omitted from P&L. Splits and other share-changing
corporate actions are not handled; do not use affected positions until that
handling is added. This is a basic paper ledger, not a full broker simulator.

At a new month, refresh signals, create a new plan, and fill it in a later
session. The current-date command intentionally confirms only the preceding
calendar month. A plan made October 1 can use October 2's close, recorded on
October 3; this conservative convention adds an execution delay relative to
the historical backtest's signal-close assumption.

```bash
python -m scanner.strategy_cli signals faber-gtaa5
python -m scanner.paper_cli plan
```

One non-cancelled plan per signal month prevents duplicate monthly trades.
To replace an unfilled plan after quotes/account state change:

```bash
python -m scanner.paper_cli cancel
python -m scanner.paper_cli plan
```

`fill` and `cancel` default to the most recent pending plan; use `--plan-id`
for an explicit one. Filled plans cannot be cancelled. Custom account paths
use global options before the subcommand:

```bash
python -m scanner.paper_cli --db data/research/another_account/account.sqlite init --equity 10000
```

No new Python packages are required. Tests:

```bash
python -m pytest -q tests/test_paper_ledger.py tests/test_gtaa_signals.py tests/test_strategy_cli.py tests/test_published_faber.py tests/test_published_faber_gtaa.py
```
