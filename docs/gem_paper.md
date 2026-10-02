# GEM paper account

This adds a separate $100,000 GEM paper account. It does not connect to a broker.
The Faber commands and account are unchanged. GEM uses the audited decision
function in `published_gem.py`: compare 12-month US total return with bills first;
if US passes, choose the stronger of SPY and VEU; otherwise choose AGG.
BIL is the signal threshold, not a held position.

## Start and run

From the repository root, with the existing virtual environment activated:

```bash
python -m unittest discover -s tests -p 'test_gem*.py' -v
python -m scanner.gem_paper init
python -m scanner.gem_paper daily
```

`init` fetches current data, creates the account and records its initial plan.
It refuses to overwrite any existing account file. Defaults are $100,000 and
10 basis points (0.10%) of each buy/sell notional as modeled transaction costs.
Choose different initial assumptions with `init --equity 100000 --cost-bps 0`.
Costs are fixed when the account is created. They are charged separately from
the recorded opening price. No extra commission or slippage is added.

Daily use:

```bash
python -m scanner.paper_daily
python -m scanner.gem_paper daily
```

Read the GEM account without fetching data or changing positions:

```bash
python -m scanner.gem_paper status
```

Open `data/research/gem_paper/dashboard.html` in a browser. The folder also
contains the authoritative `account.sqlite`, `ledger.json`, `plan.json`,
`holdings.csv`, `fills.csv`, `income.csv`, `equity.csv`, `account.txt`, and the
latest `daily_snapshot.json`. Back up the SQLite file; exports can be regenerated.
An offline rerun uses `daily --snapshot PATH` with a paper snapshot whose as-of
date matches today's New York date. Price-only research snapshots are rejected.
Use `--db PATH` and `--output-dir PATH` before the subcommand for another account.
Do not direct GEM exports into the Faber account directory.

## Timing

All dates are New York dates. Current-day bars are excluded, including after
market close. An account created October 1 records an initial plan for the
October 2 open; that fill is available to the runner October 3. Initialization
never backdates an entry to October 1 or the original month-end signal date.

Initialization also records a standing instruction to follow subsequent monthly
GEM signals. Each future confirmed month-end signal applies at the next exchange
session's open. Weekend month ends and exchange holidays use the same calendar
as the GEM execution audit. Its supported range is 2007–2028; extend the calendar
before using this beyond 2028 or after an unexpected exchange closure.

If runs are missed, the account processes every intervening session in order.
It can generate and execute intervening monthly plans under the standing rule,
without substituting today's price. Catch-up decisions use the current saved
adjusted-history snapshot; provider revisions can differ from what would have
been downloaded that day. Previously recorded plans are retained unchanged.
No historical plan before account creation is manufactured.

On a switch, the old ETF is sold at its opening price (including its overnight
move); proceeds and cash buy the new ETF at that session's opening price.
The maximum affordable whole-share quantity is calculated using those opening
prices and costs, with the remainder left as cash. Planning quantities are
estimates. This is an opening-price allocation simulation, not evidence of a
pre-sized market-on-open order, broker auction execution, settlement or fills.

An unchanged selected ETF produces NO_TRADE: no monthly top-up, cash sweep or
dividend reinvestment. Cash earns zero. Close marks value the whole-share position
and residual cash. Initial cost basis includes buy costs; realized P&L includes
both buy and sell costs.

## Dividends, splits, and data checks

Signals use adjusted levels. The ledger uses unadjusted opening and closing
quotes and explicitly modeled income, avoiding dividend double counting.
Yahoo dividend events are credited using the shares held before the ex-date
open. A position sold at that open still receives the credit; a new position
bought then does not. **Cash is credited on the ex-dividend date, not its actual
payment date.** This simplification may make cash available earlier than in a
broker account. It is an explicit modeling assumption, not exact cash settlement.

Any SPY/VEU/AGG split dated on or after account creation stops the run before
changing the ledger. Automatic split conversion and fractional cash-in-lieu are
not implemented. This also prevents using Yahoo's potentially split-restated
historical quotes for previous paper fills. A revised or late-reported dividend
in an already processed period also stops the run for review. Do not delete the
database to bypass these checks; preserve it for a targeted accounting repair.

All four ETFs must have complete aligned sessions through the previous exchange
session. Missing/stale bars, nonpositive or nonfinite prices, duplicates, future
bars, and unsupported snapshot schemas fail before account changes. Fetches
request dividends and splits explicitly; provider omissions cannot be detected
until the provider supplies or revises an event.

The entire daily update is one SQLite transaction. Repeated runs do not duplicate
fills, income or valuations. Initialization and daily processing reject a Faber
database. Export failure after a committed run can be recovered with `status`.

## Difference from the historical execution audit

The audit holds fractional total-return index units and models dividends/splits
through adjusted price factors. This account holds whole ETF shares, leaves
residual cash, models ex-date income explicitly, and charges costs on actual
traded notional. Results therefore will not match the audit dollar for dollar.
Both share the same signal decision, next-session opening timing, overnight
exposure on switches, and no-trade behavior for unchanged allocations.

## Verification

Tests cover initial timing, opening versus closing prices, whole-share sizing,
cost reconciliation, month-end switches, overnight losses, cash/basis accounting,
unchanged allocations, missing sessions, catch-up, idempotency, ex-date entitlement,
late dividend corrections, split stops, transaction rollback, weekend/holiday
timing, preserving recorded signals, future-data independence of earlier fills,
Faber database isolation, offline commands and report exports.

Live Yahoo fetching must also be exercised on the user's machine. The release
was verified with deterministic offline data; no current-market signal is claimed
until `init` successfully fetches and validates the latest snapshot.
