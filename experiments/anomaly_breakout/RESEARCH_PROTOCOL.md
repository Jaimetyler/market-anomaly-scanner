# Anomaly breakout A/B research — development protocol

Status: design and repository integration review; **not a validated backtest**.

## Hypothesis
Compare a price/volume volatility-contraction breakout baseline (A) with the identical setup requiring a recent point-in-time anomaly (B). A must include every qualifying baseline opportunity; B is a subset, not a separate parameter-optimized system.

## Frozen development rules (provisional until implementation)
- Develop on 2019–2023 only. Seal 2024–2025 until rules, costs and quality gates are frozen.
- Prior 10-session high is computed with a one-session shift; breakout is signal-session close above that high.
- Prior 20-session average volume excludes the current session; breakout volume >= 1.5 times that average.
- Five-session high-low range contracts relative to preceding five sessions; document exact non-overlapping window definitions before running.
- SPY market filter uses information available by the signal-session close; SPY close > prior/current 200-session SMA (no future bar).
- Enter at next actual trading-session open; never fill at signal close.
- B requires an eligible anomaly on the same verified security within the five trading sessions ending on the breakout signal date; do not use forward-return/outcome columns to tag.
- Position risk budget <= 1% of account equity, at most 10 positions, constrained by available cash; stop is consolidation low. Specify deterministic tie-breaking and same-day ordering.
- Account for entry/exit slippage and commissions; conservative stop fills on gap-through and intraday touch. Record assumptions.
- Freeze a single exit policy (e.g., stop or 20-session close) before looking at holdout.

## Repository interfaces verified on GitHub main
- `scanner.flatfile_history.read_many_ticker_histories`: bulk, case-sensitive, local OHLCV loading.
- `scanner.flatfile_adjustments.adjust_flatfile_history_for_splits`: split adjustment; requires vetted event data.
- `scanner.turtle_history.split_continuous_history`: evidence-confirmed lineage/distribution breaks; not a blanket gap filter.
- `scanner.historical_universe`: point-in-time universe support, but verify completeness and ticker/FIGI identity across dates.
- `scanner.backtest.compute_event_outcomes`: event outcome analysis, **not** portfolio simulation.

## Must-pass quality gates before performance claims
1. Local maturity-rebuild observations and manifest are inputs; the local phase-1 script and CSV are **not** on GitHub. GitHub main may lag the user's uncommitted corrections.
2. No future columns (`research_return_*`, `executable_return_*`, MFE/MAE, future highs/lows) in signals, eligibility or sizing.
3. Verify `security_id` mapping for each date; reject ticker reuse, ambiguous identifiers, confirmed lineage breaks and missing critical sessions rather than silently stitching.
4. Verify corporate-action cache completeness and local code parity. Do not trust an uppercase-ticker cache key when two provider tickers differ by case.
5. Ensure full-market baseline universe, not just anomaly-observed tickers; otherwise the comparison is selection-biased.
6. Avoid same-bar stop/entry ordering errors; next-open fills and gaps are modeled conservatively.
7. Track rejected candidates and reasons, missing prices, corporate actions, unavailable SPY data, and unresolved delisting/terminal values.
8. Compare portfolios at equal starting capital and costs; report exposure and trade counts as well as CAGR, drawdown, Sharpe, profit factor and benchmarks.
9. Run synthetic tests for no lookahead, split, ticker reuse, stop gap, competing entries, missing data and portfolio cash constraints.
10. Never alter existing paper accounts, scanner production files, or the user's dirty working tree.

## Next implementation
Create independent `experiments/anomaly_breakout/` modules and tests. Start with signal generation and a reproducible input manifest, then implement portfolio simulation. Local execution is required for data-dependent integration tests.

## Offline lineage diagnostic

The raw diagnostic now splits AAC, AAN, AABA and ACET at the reviewed dates in
`lineage.py`, shared with the offline audit. A boundary date belongs to the new
segment. Each segment starts fresh price/volume and anomaly lookbacks, and the
last bar of an old segment cannot enter at the new segment's open.

Observation IDs are checked within each segment, including excluded observations.
Conflicting or blank IDs within one segment still fail. Unreviewed multi-ID
tickers still fail. The standalone `read_anomaly_dates` API keeps its conservative
whole-ticker check unless `lineage_aware=True` is explicitly requested.

This implements boundary isolation, not full identity certification. A segment
without observations may still emit baseline diagnostic signals, but receives no
anomaly tags. Prices remain unadjusted; split completeness, point-in-time universe,
missing sessions and terminal values still block performance claims. The master
observations CSV is not present in this repository; full anomaly-tagged execution
still requires that local input. The 2024–2025 holdout remains excluded.

## One-command local results bundle

From the research checkout with the main scanner's Python environment active:

```bash
git pull --ff-only && python -m experiments.anomaly_breakout.run_local
```

The runner searches the current repository's data folder, then the sibling
`market-anomaly-scanner/data` folder, for both the master observations and daily
archive. For another layout pass `--data-root /path/to/data`. Default tickers are
A, AA, AAL, AAC, AAN; override using `--tickers` with exact provider letter case.

It reads the archives once, compares raw and available split-adjusted diagnostic
signals, audits split-cache structure/ratios/identity, and inventories development
universe snapshots. Each lineage segment uses only its own split events. Only
2019–2023 signals are exported; 2018 is warmup. Future outcome columns are not
signal inputs. The master file is hashed in full only for input provenance.

Split helpers now import the API client only inside the network convenience
wrapper. The runner never calls that wrapper and needs no API key or new package.
Existing malformed/missing caches are reported without API fallback or changes.
Legacy empty split lists are explicitly unverified, not proof of no splits.
Mixed-case symbols cannot use uppercase legacy cache filenames. Universe
snapshots need matching dates, active-only provenance and valid ticker records;
missing dates stay unknown rather than borrowing future snapshots. Listing with
a FIGI is reported separately from full observation-to-security reconciliation.

The timestamped ZIP under `data/research/anomaly_breakout/local_runs/` contains
only `summary.json`, `signals.csv` and `README.txt`. It includes hashes, cache
coverage/errors and sampled diagnostics, never the master dataset, cache bodies,
credentials or paper accounts. Existing outputs cannot be overwritten. Missing
inputs still generate a blocked report (exit code 2) for troubleshooting.

This bundle is an input-review handoff, not a full-market A/B backtest. Full
universe completeness, corporate-action coverage, date-specific security mapping,
missing sessions and terminal values remain gates before performance reporting.

### Missing split-cache repair

If the bundle reports a missing split cache (for example SPY), run:

```bash
git pull --ff-only && python -m experiments.anomaly_breakout.run_local --fetch-missing-splits
```

This flag opts in to the existing bounded Massive split client for **absent**
sample/benchmark cache files only (maximum 20 symbols per run, existing client
pagination bound 20 pages per symbol). It loads `MASSIVE_API_KEY` from the current
environment or the main scanner's `.env` without exporting it. Existing caches,
including rejected files, are never intentionally refreshed. Fetch errors export
only the exception type; authenticated HTTP URLs are omitted. Network requests
do not occur without this flag. Missing credentials or request failures still
produce a results ZIP with the blocker recorded.

Each signal basis now has `completed`, `partial` or `blocked` status and an
evaluated-ticker list. A blocked basis reports null counts and prints `NOT RUN`,
rather than reporting a misleading zero. A completed run with no setups retains
a real zero. Successful cache retrieval does not certify unrelated legacy caches
or mark the overall report ready for portfolio performance analysis.

### Exploratory candidate outcomes

After a completed or partial adjusted diagnostic the bundle includes independent
candidate outcomes, grouped into all sample breakouts, the anomaly subset and
breakouts without anomalies. These candidates may overlap and are not portfolio
returns. No new parameter search or holdout reads are performed.

The frozen diagnostic rule is next-session open entry, fixed signal stop, or close
of the 20th held session **including** entry day. Costs are 10 bps on each fill and
$0.005 per share each way. Gap-through stops fill at the opening price; entry-day
stop touches are charged. Identity listings without FIGIs are excluded. Missing
market sessions, lineage changes and unavailable development-period exits censor
candidates rather than silently assigning returns. Completed, censored and
excluded counts are reported separately; averages and win rates use completed
candidates only. Net R divides net P&L per share by entry fill minus signal stop.

This is a descriptive check of the selected ticker sample. Security mapping and
legacy corporate-action completeness remain unverified. It cannot establish an
anomaly advantage, annual performance or the feasibility of taking overlapping
signals under cash/risk constraints.

### Broader fixed development sample

```bash
git pull --ff-only && python -m experiments.anomaly_breakout.run_local --universe-sample 250
```

This chooses up to 250 stocks by a fixed SHA-256 rank (seed 20261029) from the
union of eligible symbols in valid 2019–2023 universe snapshots. Selection does
not inspect anomalies, prices or trade outcomes; it does not require a stock to
survive until the final date. Actual candidate eligibility remains an exact-date
snapshot check. The selected list, union fingerprint, method and seed are recorded.
This is a broader sample, not a complete full-market test. `--tickers` and
`--universe-sample` are mutually exclusive; sample size is bounded at 250.

The run remains offline by default. Missing split caches and identity conflicts
are counted as exclusions/partial coverage, without substituting other stocks
into the fixed sample. Symbols without anomaly observations stay in the baseline.
An ambiguous observation identity excludes the affected symbol from both groups
while other symbols continue; the standalone anomaly loader still raises on
ambiguities unless a rejection-report dictionary is explicitly supplied.

No new API fetches are needed for the command above. The existing optional
missing-cache fetch remains bounded at 20 symbols, so a large missing-cache list
is reported rather than triggering an unbounded retrieval. Price fingerprints
are streamed without constructing a second full copy of all bar dictionaries.
