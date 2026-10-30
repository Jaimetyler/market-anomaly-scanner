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
