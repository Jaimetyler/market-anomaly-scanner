"""Standalone, deterministic breakout signal detection and long-only execution.

Input bars must be split-adjusted and lineage-verified *before* calling this module.
No future outcome columns are read. Holdout control is the caller's responsibility.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import floor, isfinite
from typing import Mapping, Sequence


@dataclass(frozen=True)
class Bar:
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class Signal:
    ticker: str
    signal_date: str
    entry_date: str
    stop: float
    anomaly: bool


@dataclass(frozen=True)
class Trade:
    ticker: str
    entry_date: str
    exit_date: str
    shares: int
    entry_price: float
    exit_price: float
    pnl: float
    reason: str


def _valid(b: Bar) -> bool:
    return (
        all(isfinite(v) for v in (b.open, b.high, b.low, b.close, b.volume))
        and b.low > 0 and b.low <= min(b.open, b.close)
        and b.high >= max(b.open, b.close) and b.volume >= 0
    )


def detect_signals(
    ticker: str,
    bars: Sequence[Bar],
    anomaly_dates: set[str],
    spy_regime: Mapping[str, bool],
) -> list[Signal]:
    """All windows end at or before signal close. Anomaly window includes signal day."""
    if len({b.date for b in bars}) != len(bars):
        raise ValueError("Duplicate sessions")
    if list(b.date for b in bars) != sorted(b.date for b in bars):
        raise ValueError("Bars must be sorted")
    if not all(_valid(b) for b in bars):
        raise ValueError("Invalid OHLCV")
    out: list[Signal] = []
    for i in range(20, len(bars) - 1):
        b = bars[i]
        prior = bars[i - 10:i]
        old = bars[i - 10:i - 5]
        recent = bars[i - 5:i]
        old_range = max(x.high for x in old) - min(x.low for x in old)
        new_range = max(x.high for x in recent) - min(x.low for x in recent)
        avg_volume = sum(x.volume for x in bars[i - 20:i]) / 20
        if (
            spy_regime.get(b.date) is not True
            or old_range <= 0 or new_range >= old_range
            or avg_volume <= 0 or b.volume < 1.5 * avg_volume
            or b.close <= max(x.high for x in prior)
        ):
            continue
        stop = min(x.low for x in prior)
        if stop >= bars[i + 1].open:
            continue
        dates = {x.date for x in bars[max(0, i - 4):i + 1]}
        out.append(Signal(ticker, b.date, bars[i + 1].date, stop, bool(dates & anomaly_dates)))
    return out


def simulate(
    signals: Sequence[Signal],
    bars_by_ticker: Mapping[str, Sequence[Bar]],
    *,
    anomaly_only: bool = False,
    initial_cash: float = 100_000.0,
    risk_fraction: float = 0.01,
    max_positions: int = 10,
    slippage_bps: float = 10.0,
    commission_per_share: float = 0.005,
    max_hold_sessions: int = 20,
) -> tuple[list[Trade], list[tuple[str, float]]]:
    """Daily open entries, intraday stops, close time exits; no margin.

    Conservative ordering: exits processed before new entries on a session;
    open-gap stops fill at open with exit slippage; entry-day stops are active.
    Equity marks to close. No terminal liquidation: caller must supply sufficient
    future bars and audit any still-open positions separately.
    """
    if initial_cash <= 0 or not (0 < risk_fraction <= 1) or max_positions < 1 or max_hold_sessions < 1:
        raise ValueError("Invalid portfolio parameters")
    if slippage_bps < 0 or commission_per_share < 0:
        raise ValueError("Invalid trading costs")
    by_date: dict[str, dict[str, Bar]] = {}
    indices: dict[str, dict[str, int]] = {}
    for ticker, bars in bars_by_ticker.items():
        if not all(_valid(b) for b in bars):
            raise ValueError(f"Invalid OHLCV: {ticker}")
        dates = [b.date for b in bars]
        if dates != sorted(set(dates)):
            raise ValueError(f"Unsorted or duplicate bars: {ticker}")
        indices[ticker] = {d: i for i, d in enumerate(dates)}
        for b in bars:
            by_date.setdefault(b.date, {})[ticker] = b
    entry_map: dict[str, list[Signal]] = {}
    for s in signals:
        if anomaly_only and not s.anomaly:
            continue
        if s.ticker not in indices or s.entry_date not in indices[s.ticker]:
            raise ValueError(f"Missing entry bar: {s}")
        if s.entry_date <= s.signal_date:
            raise ValueError("Entry must follow signal")
        entry_map.setdefault(s.entry_date, []).append(s)
    cash = float(initial_cash)
    positions: dict[str, dict] = {}
    trades: list[Trade] = []
    curve: list[tuple[str, float]] = []
    slip = slippage_bps / 10_000
    for day in sorted(by_date):
        today = by_date[day]
        opening_cash = cash
        # Missing a bar while a position is open is a data-integrity failure,
        # not permission to use a future close or silently skip a session.
        for ticker in positions:
            if ticker not in today:
                raise ValueError(f"Missing held-position bar {ticker} on {day}")
        # Existing positions: check open-gap and intraday stops before time exit.
        for ticker, pos in list(positions.items()):
            b = today.get(ticker)
            if b is None:
                raise ValueError(f"Missing held-position bar {ticker} on {day}")
            idx = indices[ticker][day]
            if idx <= pos["entry_idx"]:
                continue
            reason = None
            if b.open <= pos["stop"]:
                exit_px, reason = b.open * (1 - slip), "gap_stop"
            elif b.low <= pos["stop"]:
                exit_px, reason = pos["stop"] * (1 - slip), "stop"
            elif idx - pos["entry_idx"] >= max_hold_sessions:
                exit_px, reason = b.close * (1 - slip), "time"
            if reason:
                shares = pos["shares"]
                cash += shares * (exit_px - commission_per_share)
                pnl = shares * (exit_px - pos["entry_px"] - 2 * commission_per_share)
                trades.append(Trade(ticker, pos["entry_date"], day, shares, pos["entry_px"], exit_px, pnl, reason))
                del positions[ticker]
        # Do not recycle proceeds from intraday or close exits into morning entries.
        # Deterministic ordering, no overlapping positions in the same ticker.
        entry_budget = opening_cash
        for s in sorted(entry_map.get(day, []), key=lambda x: (x.ticker, x.signal_date)):
            if len(positions) >= max_positions or s.ticker in positions:
                continue
            b = today[s.ticker]
            px = b.open * (1 + slip)
            risk_per_share = px - s.stop
            if risk_per_share <= 0:
                continue
            equity = cash + sum(
                p["shares"] * today[t].close for t, p in positions.items()
            )
            shares = min(
                floor(equity * risk_fraction / risk_per_share),
                floor(entry_budget / (px + commission_per_share)),
            )
            if shares < 1:
                continue
            entry_cost = shares * (px + commission_per_share)
            cash -= entry_cost
            entry_budget -= entry_cost
            positions[s.ticker] = dict(
                shares=shares, stop=s.stop, entry_px=px,
                entry_date=day, entry_idx=indices[s.ticker][day],
            )
            # Entry-day intraday stop; cannot assume favorable high came first.
            if b.low <= s.stop:
                exit_px = s.stop * (1 - slip)
                cash += shares * (exit_px - commission_per_share)
                pnl = shares * (exit_px - px - 2 * commission_per_share)
                trades.append(Trade(s.ticker, day, day, shares, px, exit_px, pnl, "entry_day_stop"))
                del positions[s.ticker]
        equity = cash + sum(
            p["shares"] * today[t].close for t, p in positions.items()
        )
        curve.append((day, equity))
    if positions:
        raise ValueError(f"Open positions at end of data: {sorted(positions)}; extend bars or define terminal policy")
    return trades, curve
