from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scanner.consolidation import (
    ConsolidationConfig,
    consolidate_promoted_signals,
    write_consolidation_reports,
)


DEFAULT_PROMOTED = Path(
    "data/research/final_evidence/promoted_research_setups.csv"
)
DEFAULT_EVENTS = Path(
    "data/research/edge_reports/event_outcomes_adapted.csv"
)
DEFAULT_OUTPUT = Path(
    "data/research/consolidation_reports"
)


def run_consolidation(
    promoted_path: Path = DEFAULT_PROMOTED,
    events_path: Path = DEFAULT_EVENTS,
    output_dir: Path = DEFAULT_OUTPUT,
    *,
    jaccard_threshold: float = 0.85,
    nested_containment_threshold: float = 0.90,
    first_oos_year: int = 2021,
    min_oos_n: int = 10,
    permutation_samples: int = 4000,
    bootstrap_samples: int = 4000,
) -> dict[str, pd.DataFrame]:
    print(f"Loading final evidence: {promoted_path}")
    promoted = pd.read_csv(promoted_path)

    print(f"Loading adapted events: {events_path}")
    events = pd.read_csv(events_path)

    promote_count = int(
        promoted["disposition"]
        .astype(str)
        .str.upper()
        .eq("PROMOTE")
        .sum()
    )

    print(f"PROMOTE definitions:      {promote_count:,}")
    print(f"Historical events:        {len(events):,}")
    print(f"Jaccard collapse gate:    {jaccard_threshold:.0%}")
    print(
        "Nested containment gate: "
        f"{nested_containment_threshold:.0%}"
    )
    print()

    config = ConsolidationConfig(
        disposition="PROMOTE",
        jaccard_threshold=jaccard_threshold,
        nested_containment_threshold=nested_containment_threshold,
        nested_jaccard_ceiling=jaccard_threshold,
        first_oos_year=first_oos_year,
        min_oos_n=min_oos_n,
        permutation_samples=permutation_samples,
        bootstrap_samples=bootstrap_samples,
    )

    reports = consolidate_promoted_signals(
        promoted,
        events,
        config=config,
    )

    write_consolidation_reports(reports, output_dir)

    archetypes = reports["archetypes"]
    members = reports["members"]
    dupes = reports["near_duplicates"]
    nested = reports["nested_relationships"]
    incremental = reports["incremental_edges"]
    immediate = reports["immediate_modifier_relationships"]
    modifier_summary = reports["modifier_summary"]

    multi_member = int(
        (archetypes["member_count"] > 1).sum()
    ) if not archetypes.empty else 0

    largest = int(
        archetypes["member_count"].max()
    ) if not archetypes.empty else 0

    print("=" * 92)
    print("PROMOTED SIGNAL CONSOLIDATION COMPLETE")
    print("=" * 92)
    print(f"Promoted definitions:     {len(members):,}")
    print(f"Distinct archetypes:      {len(archetypes):,}")
    print(f"Multi-signal archetypes:  {multi_member:,}")
    print(f"Largest archetype:        {largest:,} promoted definitions")
    print(f"Near-duplicate pairs:     {len(dupes):,}")
    print(f"Nested relationships:     {len(nested):,}")
    print(f"Structural increments:    {len(incremental):,}")
    print(f"Immediate modifiers:      {len(immediate):,}")
    print(f"Distinct modifiers:       {len(modifier_summary):,}")
    print()
    print(f"Reports: {output_dir}")
    print()
    print(
        "Near-duplicates are collapsed only when event-population "
        "Jaccard overlap clears the gate."
    )
    print(
        "Nested subsets are reported separately so a narrower, "
        "potentially stronger edge is not automatically erased."
    )

    return reports


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collapse overlapping PROMOTE definitions into historical "
            "anomaly archetypes."
        )
    )
    parser.add_argument(
        "--promoted",
        type=Path,
        default=DEFAULT_PROMOTED,
    )
    parser.add_argument(
        "--events",
        type=Path,
        default=DEFAULT_EVENTS,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--jaccard-threshold",
        type=float,
        default=0.85,
    )
    parser.add_argument(
        "--nested-containment-threshold",
        type=float,
        default=0.90,
    )
    parser.add_argument("--first-oos-year", type=int, default=2021)
    parser.add_argument("--min-oos-n", type=int, default=10)
    parser.add_argument("--permutation-samples", type=int, default=4000)
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
    return parser


def main() -> int:
    args = build_parser().parse_args()

    run_consolidation(
        promoted_path=args.promoted,
        events_path=args.events,
        output_dir=args.output_dir,
        jaccard_threshold=args.jaccard_threshold,
        nested_containment_threshold=(
            args.nested_containment_threshold
        ),
        first_oos_year=args.first_oos_year,
        min_oos_n=args.min_oos_n,
        permutation_samples=args.permutation_samples,
        bootstrap_samples=args.bootstrap_samples,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
