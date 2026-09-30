# Daily Faber paper runner and local dashboard

From the repository root, with the existing environment activated:

```bash
python -m scanner.paper_daily
```

Requires the existing account at `data/research/faber_paper/account.sqlite`.
It never initializes/resets an account or connects to a broker. No new Python
package is required. This is a runnable daily command, not a scheduled task;
it runs only when invoked.

The runner fetches the same five ETF histories used by current signals. It
validates aligned, fresh daily closes and monthly history before touching
account state. Today's daily bar is excluded; default dates are in
America/New_York. The existing monthly signal calculation and ETF proxy
limitations remain unchanged.

In order, it processes eligible existing pending plans, marks the account,
then creates a current monthly plan if one is needed. September 29's pending
plan can fill using September 30's completed close on October 1, **before**
the runner creates the new October plan from September's confirmed signal.
The October plan waits for a later completed session. Repeated runs neither
fill a plan twice nor create duplicate monthly plans. Identical marks are
harmless. An OS advisory lock prevents two daily runners from overlapping
for one account and releases automatically if a process exits.

When prices predate plan creation, the runner prints **WAITING** and exits
successfully. Initial account cash/holdings stay intact. An old pending plan
is cancelled only once the fetched price session is beyond its allocation
month; the runner never fills an expired plan using a later month's prices.
When an account revision changed or fill prices exceed available cash,
the batch remains pending with no partial fills. The runner reports **REVIEW**
and exits 2, rather than silently rewriting quantities or assumptions. The
latest eligible date must not precede already recorded account history.

Exports under the account folder:

- `dashboard.html`: local account overview, pending plans, fills, confirmed
  signals, developing preview, and a chart of recorded account values.
- `daily_status.json` and `daily_runs.jsonl`: latest status and append-only
  run history, including waiting/cancellation/review reasons.
- `signals/`: fresh signal reports plus the fetched daily snapshot.
- Existing account reports and `prices.csv`.

Open `data/research/faber_paper/dashboard.html` in your browser. Refresh it
after running the command again. It is standalone HTML, works offline, and
does not fetch live data or alter the account. Every mark and quote is dated;
the recorded-value chart is sparse observations, not a complete daily curve
or a max-drawdown calculation. Monthly preview signals are informational.

Fees/slippage for new plans inherit the most recent saved plan,
defaulting to zero when no plan exists. Optional overrides affect only new
plans; a pending plan keeps its saved assumptions:

```bash
python -m scanner.paper_daily --fee 1 --slippage-bps 5
```

Custom paths:

```bash
python -m scanner.paper_daily --db data/research/custom/account.sqlite --output-dir data/research/custom/reports
```

For an offline rerun of a saved daily snapshot:

```bash
python -m scanner.paper_daily --snapshot-json data/research/faber_current_signals/daily_snapshot.json
```

The actual current date is still used; all freshness and timing checks apply.
There is no CLI date override for manufacturing historical paper fills. A
snapshot containing future bars is ignored beyond the current data cutoff.

For REVIEW, inspect `daily_status.json` and the ledger before continuing. The
existing `paper_cli cancel` and `paper_cli plan` commands can replace a
pending plan deliberately. Account SQLite remains authoritative and is
git-ignored along with the reports. Income must still be recorded manually;
unrecorded dividends/interest and share-changing corporate actions are not
handled automatically. See `docs/faber_paper_ledger.md`.

Tests use synthetic future sessions solely to verify month-change sequencing
and accounting. They are not market observations or reported strategy results.

```bash
python -m pytest -q tests/test_paper_daily.py tests/test_paper_ledger.py tests/test_gtaa_signals.py tests/test_strategy_cli.py tests/test_published_faber.py tests/test_published_faber_gtaa.py
```
