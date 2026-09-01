from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations
from math import exp, lgamma
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from scanner.segmentation import add_research_buckets
from scanner.significance import benjamini_hochberg


SEGMENTATION_COLUMNS: dict[str, tuple[str, ...]] = {
    "initial_setup": ("initial_setup",),
    "price": ("price_bucket",),
    "rvol": ("rvol_bucket",),
    "sma20_extension": ("sma20_extension_bucket",),
    "move_1d": ("move_1d_bucket",),
    "move_3d": ("move_3d_bucket",),
    "move_5d": ("move_5d_bucket",),
    "setup_x_rvol": ("initial_setup", "rvol_bucket"),
    "setup_x_price": ("initial_setup", "price_bucket"),
    "setup_x_sma20": ("initial_setup", "sma20_extension_bucket"),
    "price_x_rvol": ("price_bucket", "rvol_bucket"),
    "rvol_x_move5": ("rvol_bucket", "move_5d_bucket"),
    "setup_x_rvol_x_move5": (
        "initial_setup",
        "rvol_bucket",
        "move_5d_bucket",
    ),
}


@dataclass(frozen=True)
class ConsolidationConfig:
    disposition: str = "PROMOTE"
    jaccard_threshold: float = 0.85
    nested_containment_threshold: float = 0.90
    nested_jaccard_ceiling: float = 0.85
    min_events_for_overlap: int = 1
    first_oos_year: int = 2021
    min_oos_n: int = 10
    permutation_samples: int = 4000
    bootstrap_samples: int = 4000
    random_seed: int = 42
    modifier_min_child_n: int = 50
    modifier_min_remainder_n: int = 30
    modifier_min_oos_years: int = 3
    modifier_min_retention: float = 0.10
    modifier_max_informative_retention: float = 0.85
    modifier_min_oos_win_lift: float = 0.02
    modifier_min_oos_median_lift: float = 0.25
    modifier_min_consistency: float = 0.60
    modifier_worst_win_delta_floor: float = -0.05
    modifier_worst_median_delta_floor: float = -1.0
    modifier_redundant_win_band: float = 0.01
    modifier_redundant_median_band: float = 0.10
    modifier_degrading_win_lift: float = -0.02
    modifier_degrading_median_lift: float = -0.25
    modifier_degrading_max_consistency: float = 0.40


class UnionFind:
    def __init__(self, items: Iterable[int]) -> None:
        self.parent = {item: item for item in items}
        self.rank = {item: 0 for item in items}

    def find(self, item: int) -> int:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, a: int, b: int) -> None:
        root_a = self.find(a)
        root_b = self.find(b)

        if root_a == root_b:
            return

        if self.rank[root_a] < self.rank[root_b]:
            root_a, root_b = root_b, root_a

        self.parent[root_b] = root_a

        if self.rank[root_a] == self.rank[root_b]:
            self.rank[root_a] += 1


def _clean_value(value: object) -> object:
    if pd.isna(value):
        return None
    return value


def _ensure_event_id(events: pd.DataFrame) -> pd.DataFrame:
    out = events.copy()

    if "event_id" in out.columns:
        out["event_id"] = out["event_id"].astype(str)
        return out

    required = {"ticker", "date"}
    missing = required.difference(out.columns)
    if missing:
        raise ValueError(
            "Events need event_id, or ticker/date columns. "
            f"Missing: {sorted(missing)}"
        )

    if "event_number" in out.columns:
        out["event_id"] = (
            out["ticker"].astype(str)
            + "|"
            + out["date"].astype(str)
            + "|"
            + out["event_number"].astype(str)
        )
    else:
        out["event_id"] = (
            out["ticker"].astype(str)
            + "|"
            + out["date"].astype(str)
            + "|"
            + out.index.astype(str)
        )

    return out


def ensure_research_buckets(events: pd.DataFrame) -> pd.DataFrame:
    """
    Add research bucket columns if the adapted event file does not already
    contain them.
    """
    needed = {
        "price_bucket",
        "rvol_bucket",
        "sma20_extension_bucket",
        "move_1d_bucket",
        "move_3d_bucket",
        "move_5d_bucket",
    }

    out = _ensure_event_id(events)

    if needed.issubset(out.columns):
        return out

    return add_research_buckets(out)


def build_signal_definitions(
    promoted: pd.DataFrame,
    *,
    disposition: str = "PROMOTE",
) -> pd.DataFrame:
    """
    Normalize final-evidence rows into signal definitions.

    One row remains one promoted segment/horizon definition.
    """
    if "disposition" not in promoted.columns:
        raise ValueError("Promotion file is missing disposition.")

    work = promoted.loc[
        promoted["disposition"].astype(str).str.upper()
        == disposition.upper()
    ].copy()

    if work.empty:
        return work.assign(signal_id=pd.Series(dtype=str))

    unknown = sorted(
        set(work["segmentation"].dropna().astype(str))
        - set(SEGMENTATION_COLUMNS)
    )
    if unknown:
        raise ValueError(
            "Unsupported segmentation(s): "
            + ", ".join(unknown)
        )

    for index, row in work.iterrows():
        columns = SEGMENTATION_COLUMNS[str(row["segmentation"])]
        missing = [col for col in columns if pd.isna(row.get(col))]
        if missing:
            raise ValueError(
                "Promoted signal has missing required grouping value(s): "
                + ", ".join(missing)
            )

    work = work.reset_index(drop=True)
    identities = work.apply(_deterministic_signal_id, axis=1)
    if identities.duplicated().any():
        raise ValueError("Promotion file contains duplicate signal definitions.")
    work.insert(0, "signal_id", identities)

    return work


def _deterministic_signal_id(row: pd.Series) -> str:
    segmentation = str(row["segmentation"])
    conditions = ";".join(
        f"{column}={row[column]}"
        for column in SEGMENTATION_COLUMNS[segmentation]
    )
    canonical = (
        f"{segmentation}|horizon={int(row['horizon_days'])}|{conditions}"
    )
    return "S_" + sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _condition_map(row: pd.Series) -> dict[str, str]:
    segmentation = str(row["segmentation"])
    conditions: dict[str, str] = {}
    for column in SEGMENTATION_COLUMNS[segmentation]:
        value = row.get(column)
        if pd.isna(value):
            raise ValueError(
                f"Signal {row.get('signal_id', '<unknown>')} is missing "
                f"required grouping value {column}."
            )
        conditions[column] = str(value)
    return conditions


def build_structural_relationships(
    signals: pd.DataFrame,
    event_sets: dict[str, set[str]],
) -> pd.DataFrame:
    """Build directional, same-horizon, strict parent-child relationships."""
    rows: list[dict[str, object]] = []
    records = [row for _, row in signals.iterrows()]

    for parent in records:
        parent_conditions = _condition_map(parent)
        for child in records:
            if parent["signal_id"] == child["signal_id"]:
                continue
            if int(parent["horizon_days"]) != int(child["horizon_days"]):
                continue

            child_conditions = _condition_map(child)
            if len(child_conditions) <= len(parent_conditions):
                continue
            if not all(
                child_conditions.get(key) == value
                for key, value in parent_conditions.items()
            ):
                continue

            parent_set = event_sets[str(parent["signal_id"])]
            child_set = event_sets[str(child["signal_id"])]
            outside_parent = child_set - parent_set
            added = {
                key: value
                for key, value in child_conditions.items()
                if key not in parent_conditions
            }
            rows.append(
                {
                    "parent_signal_id": parent["signal_id"],
                    "child_signal_id": child["signal_id"],
                    "horizon_days": int(parent["horizon_days"]),
                    "parent_segmentation": parent["segmentation"],
                    "child_segmentation": child["segmentation"],
                    "parent_condition": signal_condition_text(parent),
                    "child_condition": signal_condition_text(child),
                    "added_columns": ";".join(sorted(added)),
                    "added_conditions": ";".join(
                        f"{key}={added[key]}" for key in sorted(added)
                    ),
                    "relationship_depth": len(added),
                    "parent_event_count": len(parent_set),
                    "child_event_count": len(child_set),
                    "historical_containment": (
                        len(parent_set & child_set) / len(child_set)
                        if child_set else np.nan
                    ),
                    "integrity_check_passed": not outside_parent,
                    "child_events_outside_parent": len(outside_parent),
                }
            )

    return pd.DataFrame(rows)


def _return_values(
    events: pd.DataFrame,
    event_ids: set[str],
    horizon: int,
) -> np.ndarray:
    available = f"outcome_available_{horizon}d"
    returns = f"contrarian_return_{horizon}d"
    missing = [col for col in (available, returns) if col not in events.columns]
    if missing:
        raise ValueError(f"Events are missing outcome columns: {missing}")
    selected = events.loc[
        events["event_id"].astype(str).isin(event_ids)
        & events[available].fillna(False),
        returns,
    ]
    values = pd.to_numeric(selected, errors="coerce").dropna().to_numpy(float)
    return values


def _sample_metrics(values: np.ndarray) -> dict[str, float | int]:
    if values.size == 0:
        return {"n": 0, "win_rate": np.nan, "median_return": np.nan,
                "mean_return": np.nan}
    return {
        "n": int(values.size),
        "win_rate": float(np.mean(values > 0)),
        "median_return": float(np.median(values)),
        "mean_return": float(np.mean(values)),
    }


def fisher_exact_greater(child: np.ndarray, remainder: np.ndarray) -> float:
    """One-sided Fisher exact p-value for child win rate > remainder."""
    a = int(np.sum(child > 0))
    b = int(child.size - a)
    c = int(np.sum(remainder > 0))
    d = int(remainder.size - c)
    n1, n2, wins = a + b, c + d, a + c
    if n1 == 0 or n2 == 0:
        return np.nan

    def log_choose(n: int, k: int) -> float:
        if k < 0 or k > n:
            return -np.inf
        return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)

    upper = min(n1, wins)
    logs = [
        log_choose(wins, x) + log_choose(n1 + n2 - wins, n1 - x)
        - log_choose(n1 + n2, n1)
        for x in range(a, upper + 1)
    ]
    maximum = max(logs)
    return float(min(1.0, exp(maximum) * sum(exp(x - maximum) for x in logs)))


def median_permutation_greater(
    child: np.ndarray,
    remainder: np.ndarray,
    *,
    samples: int,
    random_seed: int,
) -> float:
    if child.size == 0 or remainder.size == 0:
        return np.nan
    observed = float(np.median(child) - np.median(remainder))
    pooled = np.concatenate([child, remainder])
    rng = np.random.default_rng(random_seed)
    exceedances = 0
    for _ in range(samples):
        shuffled = rng.permutation(pooled)
        delta = np.median(shuffled[: child.size]) - np.median(
            shuffled[child.size :]
        )
        exceedances += int(delta >= observed)
    return (exceedances + 1) / (samples + 1)


def _bootstrap_median_delta(
    child: np.ndarray,
    remainder: np.ndarray,
    *,
    samples: int,
    random_seed: int,
) -> tuple[float, float]:
    if child.size == 0 or remainder.size == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(random_seed)
    deltas = np.empty(samples)
    for index in range(samples):
        child_sample = rng.choice(child, child.size, replace=True)
        remainder_sample = rng.choice(remainder, remainder.size, replace=True)
        deltas[index] = np.median(child_sample) - np.median(remainder_sample)
    return tuple(np.quantile(deltas, [0.025, 0.975]).astype(float))


def analyze_incremental_edges(
    relationships: pd.DataFrame,
    events: pd.DataFrame,
    event_sets: dict[str, set[str]],
    *,
    config: ConsolidationConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate structural children against their disjoint parent remainder."""
    if relationships.empty:
        return pd.DataFrame(), pd.DataFrame()

    work = events.copy()
    if "date" not in work.columns:
        raise ValueError("Events are missing date for OOS evaluation.")
    work["_incremental_year"] = pd.to_datetime(
        work["date"], errors="coerce"
    ).dt.year
    if work["_incremental_year"].isna().any():
        raise ValueError("Events contain invalid/missing dates.")

    aggregate_rows: list[dict[str, object]] = []
    yearly_rows: list[dict[str, object]] = []

    for relation_number, relation in relationships.reset_index(drop=True).iterrows():
        parent_id = str(relation["parent_signal_id"])
        child_id = str(relation["child_signal_id"])
        horizon = int(relation["horizon_days"])
        parent_ids = event_sets[parent_id]
        child_ids = event_sets[child_id]
        remainder_ids = parent_ids - child_ids

        parent_values = _return_values(work, parent_ids, horizon)
        child_values = _return_values(work, child_ids, horizon)
        remainder_values = _return_values(work, remainder_ids, horizon)
        parent_metrics = _sample_metrics(parent_values)
        child_metrics = _sample_metrics(child_values)
        remainder_metrics = _sample_metrics(remainder_values)

        seed = config.random_seed + relation_number * 1009
        win_p = fisher_exact_greater(child_values, remainder_values)
        median_p = median_permutation_greater(
            child_values,
            remainder_values,
            samples=config.permutation_samples,
            random_seed=seed,
        )
        ci_low, ci_high = _bootstrap_median_delta(
            child_values,
            remainder_values,
            samples=config.bootstrap_samples,
            random_seed=seed + 1,
        )

        years = sorted(
            int(year) for year in work.loc[
                work["event_id"].astype(str).isin(parent_ids)
                & (work["_incremental_year"] >= config.first_oos_year),
                "_incremental_year",
            ].unique()
        )
        relation_years: list[dict[str, object]] = []
        for year in years:
            year_events = work.loc[work["_incremental_year"] == year]
            p = _sample_metrics(_return_values(year_events, parent_ids, horizon))
            c = _sample_metrics(_return_values(year_events, child_ids, horizon))
            r = _sample_metrics(_return_values(year_events, remainder_ids, horizon))
            comparable = bool(c["n"] > 0 and r["n"] > 0)
            year_row = {
                "parent_signal_id": parent_id,
                "child_signal_id": child_id,
                "horizon_days": horizon,
                "test_year": year,
                "parent_n": p["n"],
                "child_n": c["n"],
                "remainder_n": r["n"],
                "sample_retention": c["n"] / p["n"] if p["n"] else np.nan,
                "parent_win_rate": p["win_rate"],
                "child_win_rate": c["win_rate"],
                "remainder_win_rate": r["win_rate"],
                "child_vs_parent_win_rate_delta": c["win_rate"] - p["win_rate"],
                "child_vs_remainder_win_rate_delta": c["win_rate"] - r["win_rate"],
                "parent_median_return": p["median_return"],
                "child_median_return": c["median_return"],
                "remainder_median_return": r["median_return"],
                "child_vs_parent_median_delta": c["median_return"] - p["median_return"],
                "child_vs_remainder_median_delta": c["median_return"] - r["median_return"],
                "parent_validated": bool(
                    p["n"] >= config.min_oos_n and p["win_rate"] > 0.5
                    and p["median_return"] > 0
                ),
                "child_validated": bool(
                    c["n"] >= config.min_oos_n and c["win_rate"] > 0.5
                    and c["median_return"] > 0
                ),
                "comparable_child_remainder": comparable,
            }
            yearly_rows.append(year_row)
            relation_years.append(year_row)

        yearly = pd.DataFrame(relation_years)
        comparable = (
            yearly.loc[yearly["comparable_child_remainder"]].copy()
            if not yearly.empty else pd.DataFrame()
        )
        child_observed = (
            yearly.loc[yearly["child_n"] > 0].copy()
            if not yearly.empty else pd.DataFrame()
        )

        def safe_min(frame: pd.DataFrame, column: str) -> float:
            return float(frame[column].min()) if not frame.empty else np.nan

        aggregate_rows.append(
            {
                **relation.to_dict(),
                "historical_parent_n": parent_metrics["n"],
                "historical_child_n": child_metrics["n"],
                "historical_remainder_n": remainder_metrics["n"],
                "historical_sample_retention": (
                    child_metrics["n"] / parent_metrics["n"]
                    if parent_metrics["n"] else np.nan
                ),
                "historical_child_win_rate": child_metrics["win_rate"],
                "historical_remainder_win_rate": remainder_metrics["win_rate"],
                "historical_win_rate_delta": (
                    child_metrics["win_rate"] - remainder_metrics["win_rate"]
                ),
                "historical_child_median_return": child_metrics["median_return"],
                "historical_remainder_median_return": remainder_metrics["median_return"],
                "historical_median_return_delta": (
                    child_metrics["median_return"] - remainder_metrics["median_return"]
                ),
                "win_rate_p_value": win_p,
                "median_delta_p_value": median_p,
                "median_delta_ci_low": ci_low,
                "median_delta_ci_high": ci_high,
                "eligible_oos_years": len(yearly),
                "comparable_oos_years": len(comparable),
                "child_observed_oos_years": len(child_observed),
                "oos_mean_sample_retention": (
                    float(yearly["sample_retention"].mean())
                    if not yearly.empty else np.nan
                ),
                "oos_weighted_child_win_rate": (
                    float(np.average(child_observed["child_win_rate"], weights=child_observed["child_n"]))
                    if not child_observed.empty else np.nan
                ),
                "oos_weighted_remainder_win_rate": (
                    float(np.average(comparable["remainder_win_rate"], weights=comparable["remainder_n"]))
                    if not comparable.empty else np.nan
                ),
                "oos_mean_win_rate_delta": (
                    float(comparable["child_vs_remainder_win_rate_delta"].mean())
                    if not comparable.empty else np.nan
                ),
                "oos_median_return_delta": (
                    float(comparable["child_vs_remainder_median_delta"].median())
                    if not comparable.empty else np.nan
                ),
                "oos_win_rate_improvement_consistency": (
                    float((comparable["child_vs_remainder_win_rate_delta"] > 0).mean())
                    if not comparable.empty else np.nan
                ),
                "oos_median_improvement_consistency": (
                    float((comparable["child_vs_remainder_median_delta"] > 0).mean())
                    if not comparable.empty else np.nan
                ),
                "parent_validation_rate": (
                    float(yearly["parent_validated"].mean()) if not yearly.empty else np.nan
                ),
                "child_validation_rate": (
                    float(yearly["child_validated"].mean()) if not yearly.empty else np.nan
                ),
                "validation_rate_delta": (
                    float(yearly["child_validated"].mean() - yearly["parent_validated"].mean())
                    if not yearly.empty else np.nan
                ),
                "worst_oos_child_win_rate": safe_min(child_observed, "child_win_rate"),
                "worst_oos_win_rate_delta": safe_min(comparable, "child_vs_remainder_win_rate_delta"),
                "worst_oos_child_median_return": safe_min(child_observed, "child_median_return"),
                "worst_oos_median_return_delta": safe_min(comparable, "child_vs_remainder_median_delta"),
                "zero_size_remainder": remainder_metrics["n"] == 0,
            }
        )

    aggregate = pd.DataFrame(aggregate_rows)
    for p_column, q_column, flag_column in (
        ("win_rate_p_value", "win_rate_q_value", "win_rate_fdr_significant"),
        ("median_delta_p_value", "median_delta_q_value", "median_delta_fdr_significant"),
    ):
        correction = benjamini_hochberg(aggregate[p_column], alpha=0.05)
        aggregate[q_column] = correction["q_value"].to_numpy()
        aggregate[flag_column] = correction["fdr_significant"].to_numpy()

    aggregate["incrementally_supported"] = (
        aggregate["integrity_check_passed"]
        & ~aggregate["zero_size_remainder"]
        & aggregate["win_rate_fdr_significant"]
        & aggregate["median_delta_fdr_significant"]
        & (aggregate["historical_win_rate_delta"] > 0)
        & (aggregate["historical_median_return_delta"] > 0)
    )
    return aggregate, pd.DataFrame(yearly_rows)


def classify_modifier_relationships(
    incremental: pd.DataFrame,
    *,
    config: ConsolidationConfig,
) -> pd.DataFrame:
    """Assign transparent rule-based modifier labels to immediate edges."""
    if incremental.empty:
        return incremental.assign(
            modifier_classification=pd.Series(dtype=str),
            classification_reason=pd.Series(dtype=str),
        )

    def classify(row: pd.Series) -> tuple[str, str]:
        retention = row["historical_sample_retention"]
        win_lift = row["oos_mean_win_rate_delta"]
        median_lift = row["oos_median_return_delta"]
        win_consistency = row["oos_win_rate_improvement_consistency"]
        median_consistency = row["oos_median_improvement_consistency"]

        power_failures: list[str] = []
        if int(row["historical_child_n"]) < config.modifier_min_child_n:
            power_failures.append("child_n")
        if int(row["historical_remainder_n"]) < config.modifier_min_remainder_n:
            power_failures.append("remainder_n")
        if int(row["comparable_oos_years"]) < config.modifier_min_oos_years:
            power_failures.append("comparable_oos_years")
        if pd.isna(retention) or retention < config.modifier_min_retention:
            power_failures.append("sample_retention")
        if power_failures:
            return "UNDERPOWERED", "insufficient:" + ",".join(power_failures)

        fdr_supported = bool(
            row["win_rate_fdr_significant"]
            and row["median_delta_fdr_significant"]
            and row["historical_win_rate_delta"] > 0
            and row["historical_median_return_delta"] > 0
        )
        consistent = bool(
            win_consistency >= config.modifier_min_consistency
            and median_consistency >= config.modifier_min_consistency
        )
        worst_year_ok = bool(
            row["worst_oos_win_rate_delta"]
            >= config.modifier_worst_win_delta_floor
            and row["worst_oos_median_return_delta"]
            >= config.modifier_worst_median_delta_floor
        )
        material_oos = bool(
            win_lift >= config.modifier_min_oos_win_lift
            and median_lift >= config.modifier_min_oos_median_lift
        )

        if (
            fdr_supported
            and material_oos
            and consistent
            and worst_year_ok
            and retention <= config.modifier_max_informative_retention
        ):
            return "INFORMATIVE", "fdr+material_oos+consistent+worst_year_ok"

        if (
            win_lift <= config.modifier_degrading_win_lift
            and median_lift <= config.modifier_degrading_median_lift
            and win_consistency <= config.modifier_degrading_max_consistency
            and median_consistency <= config.modifier_degrading_max_consistency
        ):
            return "DEGRADING", "material_negative_oos+low_consistency"

        negligible = bool(
            abs(win_lift) <= config.modifier_redundant_win_band
            and abs(median_lift) <= config.modifier_redundant_median_band
        )
        barely_selective = bool(
            retention > config.modifier_max_informative_retention
            and not fdr_supported
        )
        if negligible or barely_selective:
            reason = "negligible_oos_effect" if negligible else "high_retention_without_fdr_support"
            return "REDUNDANT", reason

        aggregate_positive = bool(win_lift > 0 and median_lift > 0)
        if aggregate_positive and (not consistent or not worst_year_ok):
            failed = []
            if not consistent:
                failed.append("consistency")
            if not worst_year_ok:
                failed.append("worst_year")
            return "FRAGILE", "positive_aggregate_failed:" + ",".join(failed)

        return "MIXED", "conflicting_or_inconclusive_evidence"

    out = incremental.copy()
    labels = out.apply(classify, axis=1, result_type="expand")
    out["modifier_classification"] = labels[0]
    out["classification_reason"] = labels[1]
    out["classification_rule_version"] = "modifier_rules_v1"
    for column, value in {
        "rule_min_child_n": config.modifier_min_child_n,
        "rule_min_remainder_n": config.modifier_min_remainder_n,
        "rule_min_oos_years": config.modifier_min_oos_years,
        "rule_min_retention": config.modifier_min_retention,
        "rule_max_informative_retention": config.modifier_max_informative_retention,
        "rule_min_oos_win_lift": config.modifier_min_oos_win_lift,
        "rule_min_oos_median_lift": config.modifier_min_oos_median_lift,
        "rule_min_consistency": config.modifier_min_consistency,
        "rule_worst_win_delta_floor": config.modifier_worst_win_delta_floor,
        "rule_worst_median_delta_floor": config.modifier_worst_median_delta_floor,
        "rule_redundant_win_band": config.modifier_redundant_win_band,
        "rule_redundant_median_band": config.modifier_redundant_median_band,
        "rule_degrading_win_lift": config.modifier_degrading_win_lift,
        "rule_degrading_median_lift": config.modifier_degrading_median_lift,
        "rule_degrading_max_consistency": config.modifier_degrading_max_consistency,
    }.items():
        out[column] = value
    return out


def summarize_modifiers(immediate: pd.DataFrame) -> pd.DataFrame:
    """Summarize immediate modifier behavior across parents and horizons."""
    if immediate.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for (columns, conditions), group in immediate.groupby(
        ["added_columns", "added_conditions"], sort=True, dropna=False
    ):
        counts = group["modifier_classification"].value_counts()
        contexts = group[
            ["parent_segmentation", "parent_condition", "horizon_days"]
        ].drop_duplicates()
        rows.append(
            {
                "added_columns": columns,
                "added_conditions": conditions,
                "relationship_count": len(group),
                "parent_context_count": len(contexts),
                "horizon_count": int(group["horizon_days"].nunique()),
                "horizons": ";".join(
                    map(str, sorted(group["horizon_days"].astype(int).unique()))
                ),
                "informative_count": int(counts.get("INFORMATIVE", 0)),
                "informative_rate": float(
                    (group["modifier_classification"] == "INFORMATIVE").mean()
                ),
                "redundant_count": int(counts.get("REDUNDANT", 0)),
                "fragile_count": int(counts.get("FRAGILE", 0)),
                "underpowered_count": int(counts.get("UNDERPOWERED", 0)),
                "degrading_count": int(counts.get("DEGRADING", 0)),
                "mixed_count": int(counts.get("MIXED", 0)),
                "median_historical_retention": float(
                    group["historical_sample_retention"].median()
                ),
                "median_oos_win_rate_lift": float(
                    group["oos_mean_win_rate_delta"].median()
                ),
                "median_oos_median_return_lift": float(
                    group["oos_median_return_delta"].median()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["informative_rate", "informative_count", "relationship_count"],
        ascending=[False, False, False],
        kind="stable",
    ).reset_index(drop=True)


def signal_condition_text(row: pd.Series) -> str:
    segmentation = str(row["segmentation"])
    columns = SEGMENTATION_COLUMNS[segmentation]

    parts: list[str] = []
    for col in columns:
        value = _clean_value(row.get(col))
        if value is not None:
            parts.append(f"{col}={value}")

    return " | ".join(parts)


def match_signal_events(
    signal_row: pd.Series,
    events: pd.DataFrame,
) -> set[str]:
    """
    Return the event IDs satisfying one promoted signal definition.

    Outcome horizon is not part of the event population definition. This is
    intentional: identical market populations promoted at multiple horizons
    should be recognized as overlapping evidence.
    """
    segmentation = str(signal_row["segmentation"])
    columns = SEGMENTATION_COLUMNS[segmentation]

    mask = pd.Series(True, index=events.index)

    for col in columns:
        if col not in events.columns:
            raise ValueError(
                f"Events are missing required bucket column: {col}"
            )

        value = _clean_value(signal_row.get(col))
        if value is None:
            raise ValueError(
                f"Signal {signal_row.get('signal_id', '<unknown>')} "
                f"is missing required grouping value {col}."
            )

        mask &= events[col].astype(str) == str(value)

    return set(events.loc[mask, "event_id"].astype(str))


def build_signal_event_sets(
    signals: pd.DataFrame,
    events: pd.DataFrame,
) -> tuple[dict[str, set[str]], pd.DataFrame]:
    event_sets: dict[str, set[str]] = {}
    rows: list[dict[str, object]] = []

    for _, row in signals.iterrows():
        signal_id = str(row["signal_id"])
        event_ids = match_signal_events(row, events)
        event_sets[signal_id] = event_ids

        rows.append(
            {
                "signal_id": signal_id,
                "condition": signal_condition_text(row),
                "matched_event_count": len(event_ids),
            }
        )

    return event_sets, pd.DataFrame(rows)


def overlap_statistics(
    a: set[str],
    b: set[str],
) -> dict[str, float | int]:
    intersection = len(a & b)
    union = len(a | b)
    min_size = min(len(a), len(b))
    max_size = max(len(a), len(b))

    jaccard = intersection / union if union else np.nan
    containment = intersection / min_size if min_size else np.nan
    larger_coverage = intersection / max_size if max_size else np.nan

    return {
        "intersection_n": intersection,
        "union_n": union,
        "jaccard": jaccard,
        "containment": containment,
        "larger_coverage": larger_coverage,
    }


def build_overlap_pairs(
    signals: pd.DataFrame,
    event_sets: dict[str, set[str]],
    *,
    config: ConsolidationConfig | None = None,
) -> pd.DataFrame:
    config = config or ConsolidationConfig()

    lookup = signals.set_index("signal_id", drop=False)
    rows: list[dict[str, object]] = []

    for signal_a, signal_b in combinations(signals["signal_id"], 2):
        set_a = event_sets[str(signal_a)]
        set_b = event_sets[str(signal_b)]

        if (
            len(set_a) < config.min_events_for_overlap
            or len(set_b) < config.min_events_for_overlap
        ):
            continue

        stats = overlap_statistics(set_a, set_b)

        if stats["intersection_n"] == 0:
            continue

        row_a = lookup.loc[signal_a]
        row_b = lookup.loc[signal_b]

        near_duplicate = bool(
            stats["jaccard"] >= config.jaccard_threshold
        )
        nested = bool(
            stats["containment"] >= config.nested_containment_threshold
            and stats["jaccard"] < config.nested_jaccard_ceiling
        )

        rows.append(
            {
                "signal_a": signal_a,
                "signal_b": signal_b,
                "segmentation_a": row_a["segmentation"],
                "segmentation_b": row_b["segmentation"],
                "horizon_a": row_a["horizon_days"],
                "horizon_b": row_b["horizon_days"],
                "condition_a": signal_condition_text(row_a),
                "condition_b": signal_condition_text(row_b),
                "events_a": len(set_a),
                "events_b": len(set_b),
                **stats,
                "near_duplicate": near_duplicate,
                "nested_relationship": nested,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "signal_a",
                "signal_b",
                "segmentation_a",
                "segmentation_b",
                "horizon_a",
                "horizon_b",
                "condition_a",
                "condition_b",
                "events_a",
                "events_b",
                "intersection_n",
                "union_n",
                "jaccard",
                "containment",
                "larger_coverage",
                "near_duplicate",
                "nested_relationship",
            ]
        )

    return pd.DataFrame(rows).sort_values(
        ["near_duplicate", "jaccard", "containment"],
        ascending=[False, False, False],
        kind="stable",
    ).reset_index(drop=True)


def assign_archetypes(
    signals: pd.DataFrame,
    overlap_pairs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Collapse near-duplicate definitions into connected components.

    Only high-Jaccard pairs create archetype links. Nested-but-materially-
    narrower definitions are reported separately and are NOT automatically
    collapsed.
    """
    if signals.empty:
        return signals.assign(archetype_id=pd.Series(dtype=str))

    signal_ids = signals["signal_id"].astype(str).tolist()
    uf = UnionFind(range(len(signal_ids)))
    position = {signal_id: i for i, signal_id in enumerate(signal_ids)}

    if not overlap_pairs.empty:
        dupes = overlap_pairs.loc[overlap_pairs["near_duplicate"]]
        for _, row in dupes.iterrows():
            uf.union(
                position[str(row["signal_a"])],
                position[str(row["signal_b"])],
            )

    groups: dict[int, list[str]] = {}
    for i, signal_id in enumerate(signal_ids):
        root = uf.find(i)
        groups.setdefault(root, []).append(signal_id)

    ordered_groups = sorted(
        groups.values(),
        key=lambda members: (-len(members), members[0]),
    )

    archetype_for: dict[str, str] = {}
    for number, members in enumerate(ordered_groups, start=1):
        archetype_id = f"A{number:03d}"
        for signal_id in members:
            archetype_for[signal_id] = archetype_id

    out = signals.copy()
    out.insert(
        1,
        "archetype_id",
        out["signal_id"].map(archetype_for),
    )
    return out


def _representative_sort_columns(frame: pd.DataFrame) -> list[str]:
    preferred = [
        "promotion_score",
        "validation_rate",
        "weighted_test_win_rate",
        "median_test_median_return",
        "total_test_n",
        "confidence_score",
        "n",
    ]
    return [col for col in preferred if col in frame.columns]


def summarize_archetypes(
    members: pd.DataFrame,
    event_sets: dict[str, set[str]],
) -> pd.DataFrame:
    if members.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []

    for archetype_id, group in members.groupby(
        "archetype_id",
        sort=False,
        dropna=False,
    ):
        sort_cols = _representative_sort_columns(group)
        if sort_cols:
            representative = group.sort_values(
                sort_cols,
                ascending=[False] * len(sort_cols),
                na_position="last",
                kind="stable",
            ).iloc[0]
        else:
            representative = group.iloc[0]

        union_events: set[str] = set()
        for signal_id in group["signal_id"].astype(str):
            union_events |= event_sets[signal_id]

        segmentation_types = sorted(
            group["segmentation"].dropna().astype(str).unique()
        )
        horizons = sorted(
            {
                int(value)
                for value in group["horizon_days"].dropna().tolist()
            }
        )

        rows.append(
            {
                "archetype_id": archetype_id,
                "member_count": len(group),
                "unique_event_count": len(union_events),
                "segmentation_count": len(segmentation_types),
                "segmentations": ";".join(segmentation_types),
                "horizons": ";".join(map(str, horizons)),
                "representative_signal_id": representative["signal_id"],
                "representative_segmentation": representative["segmentation"],
                "representative_horizon_days": representative["horizon_days"],
                "representative_condition": signal_condition_text(
                    representative
                ),
                "promotion_score": representative.get(
                    "promotion_score", np.nan
                ),
                "historical_n": representative.get("n", np.nan),
                "historical_win_rate": representative.get(
                    "win_rate", np.nan
                ),
                "historical_median_return": representative.get(
                    "median_contrarian_return", np.nan
                ),
                "win_rate_ci_low": representative.get(
                    "win_rate_ci_low", np.nan
                ),
                "median_return_ci_low": representative.get(
                    "median_return_ci_low", np.nan
                ),
                "folds_tested": representative.get(
                    "folds_tested", np.nan
                ),
                "validated_folds": representative.get(
                    "validated_folds", np.nan
                ),
                "validation_rate": representative.get(
                    "validation_rate", np.nan
                ),
                "oos_n": representative.get("total_test_n", np.nan),
                "oos_win_rate": representative.get(
                    "weighted_test_win_rate", np.nan
                ),
                "oos_median_return": representative.get(
                    "median_test_median_return", np.nan
                ),
                "worst_oos_median_return": representative.get(
                    "worst_test_median_return", np.nan
                ),
            }
        )

    out = pd.DataFrame(rows)

    sort_cols = [
        col
        for col in [
            "promotion_score",
            "validation_rate",
            "oos_win_rate",
            "oos_median_return",
            "oos_n",
        ]
        if col in out.columns
    ]
    if sort_cols:
        out = out.sort_values(
            sort_cols,
            ascending=[False] * len(sort_cols),
            na_position="last",
            kind="stable",
        )

    return out.reset_index(drop=True)


def consolidate_promoted_signals(
    promoted: pd.DataFrame,
    events: pd.DataFrame,
    *,
    config: ConsolidationConfig | None = None,
) -> dict[str, pd.DataFrame]:
    config = config or ConsolidationConfig()

    events_buckets = ensure_research_buckets(events)
    signals = build_signal_definitions(
        promoted,
        disposition=config.disposition,
    )

    event_sets, counts = build_signal_event_sets(
        signals,
        events_buckets,
    )

    signals = signals.merge(
        counts,
        on="signal_id",
        how="left",
        validate="one_to_one",
    )

    overlap_pairs = build_overlap_pairs(
        signals,
        event_sets,
        config=config,
    )
    members = assign_archetypes(signals, overlap_pairs)
    archetypes = summarize_archetypes(members, event_sets)
    structural = build_structural_relationships(signals, event_sets)
    incremental, incremental_by_year = analyze_incremental_edges(
        structural,
        events_buckets,
        event_sets,
        config=config,
    )
    if incremental.empty:
        immediate = classify_modifier_relationships(incremental, config=config)
        transitive = incremental.copy()
    else:
        immediate = incremental.loc[
            incremental["relationship_depth"] == 1
        ].copy()
        immediate = classify_modifier_relationships(immediate, config=config)
        transitive = incremental.loc[
            incremental["relationship_depth"] > 1
        ].copy()
    modifier_summary = summarize_modifiers(immediate)

    nested = overlap_pairs.loc[
        overlap_pairs["nested_relationship"]
    ].copy()

    duplicates = overlap_pairs.loc[
        overlap_pairs["near_duplicate"]
    ].copy()

    return {
        "archetypes": archetypes,
        "members": members,
        "overlap_pairs": overlap_pairs,
        "near_duplicates": duplicates,
        "nested_relationships": nested,
        "incremental_edges": incremental,
        "incremental_edges_by_year": incremental_by_year,
        "immediate_modifier_relationships": immediate,
        "transitive_incremental_relationships": transitive,
        "modifier_summary": modifier_summary,
    }


def write_consolidation_reports(
    reports: dict[str, pd.DataFrame],
    output_dir: str | Path,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    filenames = {
        "archetypes": "archetypes.csv",
        "members": "archetype_members.csv",
        "overlap_pairs": "signal_overlap_pairs.csv",
        "near_duplicates": "near_duplicate_signals.csv",
        "nested_relationships": "nested_signal_relationships.csv",
        "incremental_edges": "incremental_edge_relationships.csv",
        "incremental_edges_by_year": "incremental_edge_by_year.csv",
        "immediate_modifier_relationships": "immediate_modifier_relationships.csv",
        "transitive_incremental_relationships": "transitive_incremental_relationships.csv",
        "modifier_summary": "modifier_summary.csv",
    }

    for key, filename in filenames.items():
        reports[key].to_csv(output / filename, index=False)
