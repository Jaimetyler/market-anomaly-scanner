from __future__ import annotations

from dataclasses import dataclass
from math import erf, exp, lgamma, log, log1p, sqrt
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SignificanceConfig:
    """
    Statistical confidence settings for research segments.

    alpha:
        Nominal significance level used for raw p-values / intervals.

    fdr_alpha:
        False-discovery-rate threshold used by Benjamini-Hochberg.

    bootstrap_samples:
        Number of bootstrap resamples used for the median-return CI.

    random_seed:
        Fixed seed keeps reports reproducible.
    """

    alpha: float = 0.05
    fdr_alpha: float = 0.05
    bootstrap_samples: int = 4000
    random_seed: int = 42


def _validate_config(config: SignificanceConfig) -> None:
    if not 0 < config.alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")
    if not 0 < config.fdr_alpha < 1:
        raise ValueError("fdr_alpha must be between 0 and 1.")
    if config.bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be at least 100.")


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def wilson_interval(
    wins: int,
    n: int,
    *,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """
    Wilson score interval for a binomial proportion.

    Uses common z critical values for the confidence levels we expect here,
    with a normal approximation fallback.
    """
    if n <= 0:
        return np.nan, np.nan
    if wins < 0 or wins > n:
        raise ValueError("wins must be between 0 and n.")

    # Common two-sided critical values.
    common = {
        0.10: 1.6448536269514722,
        0.05: 1.959963984540054,
        0.02: 2.3263478740408408,
        0.01: 2.5758293035489004,
    }

    rounded = round(float(alpha), 8)
    z = common.get(rounded)

    if z is None:
        # Binary-search inverse normal CDF without scipy.
        target = 1.0 - alpha / 2.0
        lo, hi = -8.0, 8.0
        for _ in range(100):
            mid = (lo + hi) / 2.0
            if _normal_cdf(mid) < target:
                lo = mid
            else:
                hi = mid
        z = (lo + hi) / 2.0

    p = wins / n
    z2 = z * z

    denominator = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denominator
    margin = (
        z
        * sqrt(
            p * (1.0 - p) / n
            + z2 / (4.0 * n * n)
        )
        / denominator
    )

    return max(0.0, center - margin), min(1.0, center + margin)


def exact_binomial_greater_pvalue(
    wins: int,
    n: int,
    *,
    null_p: float = 0.50,
) -> float:
    """
    Exact one-sided binomial p-value for H1: win_rate > null_p.

    Uses log-space probabilities so large samples do not overflow when
    evaluating binomial coefficients. This remains a pure-Python exact
    binomial tail calculation and does not require scipy.
    """
    if n <= 0:
        return np.nan
    if wins < 0 or wins > n:
        raise ValueError("wins must be between 0 and n.")
    if not 0 <= null_p <= 1:
        raise ValueError("null_p must be between 0 and 1.")

    if null_p == 0.0:
        return 1.0 if wins == 0 else 0.0
    if null_p == 1.0:
        return 1.0

    log_p = log(null_p)
    log_q = log1p(-null_p)

    def _log_pmf(k: int) -> float:
        return (
            lgamma(n + 1.0)
            - lgamma(k + 1.0)
            - lgamma(n - k + 1.0)
            + k * log_p
            + (n - k) * log_q
        )

    null_mean = n * null_p

    if wins <= null_mean:
        if wins == 0:
            return 1.0

        logs = [_log_pmf(k) for k in range(0, wins)]
        max_log = max(logs)
        lower_tail = exp(max_log) * sum(
            exp(value - max_log) for value in logs
        )
        probability = 1.0 - lower_tail
    else:
        logs = [_log_pmf(k) for k in range(wins, n + 1)]
        max_log = max(logs)
        probability = exp(max_log) * sum(
            exp(value - max_log) for value in logs
        )

    return float(min(1.0, max(0.0, probability)))


def bootstrap_median_interval(
    values: Iterable[float],
    *,
    alpha: float = 0.05,
    samples: int = 4000,
    random_seed: int = 42,
) -> tuple[float, float]:
    """
    Percentile bootstrap confidence interval for the median.
    """
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return np.nan, np.nan

    if arr.size == 1:
        value = float(arr[0])
        return value, value

    rng = np.random.default_rng(random_seed)

    medians = np.empty(samples, dtype=float)

    # Chunking avoids a giant samples x n allocation on large segments.
    chunk = 250
    written = 0

    while written < samples:
        batch = min(chunk, samples - written)
        indices = rng.integers(
            0,
            arr.size,
            size=(batch, arr.size),
        )
        medians[written : written + batch] = np.median(
            arr[indices],
            axis=1,
        )
        written += batch

    lower = float(np.quantile(medians, alpha / 2.0))
    upper = float(np.quantile(medians, 1.0 - alpha / 2.0))

    return lower, upper


def benjamini_hochberg(
    p_values: pd.Series,
    *,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Benjamini-Hochberg false-discovery-rate correction.

    Returns:
        p_value
        q_value
        fdr_significant
    """
    values = pd.to_numeric(p_values, errors="coerce")

    result = pd.DataFrame(
        {
            "p_value": values,
            "q_value": np.nan,
            "fdr_significant": False,
        },
        index=p_values.index,
    )

    valid = values.dropna()

    if valid.empty:
        return result

    ordered = valid.sort_values()
    m = len(ordered)

    raw_q = np.empty(m, dtype=float)

    for rank, (_, p) in enumerate(ordered.items(), start=1):
        raw_q[rank - 1] = float(p) * m / rank

    # Enforce monotonicity from largest rank backward.
    adjusted = np.minimum.accumulate(raw_q[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    for (idx, _), q in zip(ordered.items(), adjusted):
        result.at[idx, "q_value"] = float(q)
        result.at[idx, "fdr_significant"] = bool(q <= alpha)

    return result


def evaluate_segment_significance(
    outcomes: pd.DataFrame,
    *,
    group_by: Sequence[str],
    horizons: Iterable[int] = (1, 2, 3, 5, 10, 20),
    min_n: int = 30,
    config: SignificanceConfig | None = None,
) -> pd.DataFrame:
    """
    Evaluate statistical confidence of contrarian-return segments.

    For each segment/horizon:
      - sample size
      - win rate
      - Wilson 95% interval for win rate
      - exact one-sided binomial test against 50%
      - mean / median contrarian return
      - bootstrap confidence interval for median return

    Multiple-testing correction is applied across all returned rows.

    `statistically_supported` requires:
      - n >= min_n
      - BH-adjusted q <= fdr_alpha
      - lower Wilson win-rate bound > 50%
      - lower bootstrap median-return bound > 0

    This is intentionally conservative.
    """
    config = config or SignificanceConfig()
    _validate_config(config)

    group_by = list(group_by)
    if not group_by:
        raise ValueError("group_by must contain at least one column.")
    if min_n < 1:
        raise ValueError("min_n must be at least 1.")

    missing = [column for column in group_by if column not in outcomes.columns]
    if missing:
        raise ValueError(f"Missing grouping columns: {missing}")

    horizons = tuple(sorted({int(h) for h in horizons}))
    if not horizons or horizons[0] <= 0:
        raise ValueError("horizons must contain positive integers.")

    rows: list[dict] = []

    # Statistical tests require a fully specified segment definition.
    # Missing bucket values represent unavailable source data, not a segment.
    complete_outcomes = outcomes.dropna(subset=group_by)

    grouped = complete_outcomes.groupby(
        group_by[0] if len(group_by) == 1 else group_by,
        observed=True,
        sort=True,
    )

    for group_key, frame in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        group_values = dict(zip(group_by, group_key))

        for horizon in horizons:
            suffix = f"{horizon}d"
            available_col = f"outcome_available_{suffix}"
            return_col = f"contrarian_return_{suffix}"

            required = [available_col, return_col]
            missing_outcomes = [
                column for column in required if column not in frame.columns
            ]
            if missing_outcomes:
                raise ValueError(
                    f"Missing outcome columns for {horizon}d: "
                    f"{missing_outcomes}"
                )

            usable = frame[
                frame[available_col].fillna(False)
            ].copy()

            returns = pd.to_numeric(
                usable[return_col],
                errors="coerce",
            ).dropna()

            n = int(len(returns))
            if n < min_n:
                continue

            wins = int((returns > 0).sum())
            win_rate = wins / n

            win_ci_low, win_ci_high = wilson_interval(
                wins,
                n,
                alpha=config.alpha,
            )

            p_value = exact_binomial_greater_pvalue(
                wins,
                n,
                null_p=0.50,
            )

            median_ci_low, median_ci_high = bootstrap_median_interval(
                returns.to_numpy(),
                alpha=config.alpha,
                samples=config.bootstrap_samples,
                random_seed=(
                    config.random_seed
                    + horizon
                    + len(rows) * 997
                ),
            )

            rows.append(
                {
                    **group_values,
                    "horizon_days": horizon,
                    "n": n,
                    "wins": wins,
                    "win_rate": float(win_rate),
                    "win_rate_ci_low": float(win_ci_low),
                    "win_rate_ci_high": float(win_ci_high),
                    "win_rate_p_value": float(p_value),
                    "mean_contrarian_return": float(returns.mean()),
                    "median_contrarian_return": float(returns.median()),
                    "median_return_ci_low": float(median_ci_low),
                    "median_return_ci_high": float(median_ci_high),
                }
            )

    result = pd.DataFrame(rows)

    if result.empty:
        return result

    correction = benjamini_hochberg(
        result["win_rate_p_value"],
        alpha=config.fdr_alpha,
    )

    result["q_value"] = correction["q_value"].to_numpy()
    result["fdr_significant"] = (
        correction["fdr_significant"].to_numpy()
    )

    result["statistically_supported"] = (
        (result["n"] >= min_n)
        & result["fdr_significant"]
        & (result["win_rate_ci_low"] > 0.50)
        & (result["median_return_ci_low"] > 0.0)
    )

    # Conservative ranking score: reward the lower confidence bounds,
    # rather than the rosy point estimates.
    result["confidence_score"] = (
        np.maximum(result["win_rate_ci_low"] - 0.50, 0.0)
        * np.maximum(result["median_return_ci_low"], 0.0)
        * np.sqrt(result["n"])
    )

    return result.sort_values(
        [
            "horizon_days",
            "statistically_supported",
            "confidence_score",
            "n",
        ],
        ascending=[True, False, False, False],
        kind="stable",
    ).reset_index(drop=True)
