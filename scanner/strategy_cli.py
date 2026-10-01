"""Small command line catalog of published strategy reference calculations."""

import argparse
from datetime import date, datetime
from zoneinfo import ZoneInfo


STRATEGIES = {
    "faber-10month": ("Single-asset 10-month moving average", "scanner.published_faber"),
    "faber-gtaa5": ("Five-asset GTAA, equal 20% sleeves", "scanner.published_faber_gtaa"),
    "gem": ("Antonacci GEM, 12-month absolute then relative momentum", "scanner.published_gem"),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Published strategy references")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="show available published strategies")
    run = commands.add_parser("run", help="calculate one strategy from monthly CSV data")
    run.add_argument("strategy", choices=STRATEGIES)
    run.add_argument("input_csv", help="monthly input CSV")
    run.add_argument("output_csv", help="result CSV")
    signals = commands.add_parser("signals", help="current monthly ETF proxy targets and paper sizing")
    signals.add_argument("strategy", choices=["faber-gtaa5"])
    signals.add_argument("--as-of", type=date.fromisoformat,
                         default=datetime.now(ZoneInfo("America/New_York")).date())
    signals.add_argument("--equity", type=float, default=100_000)
    signals.add_argument("--output-dir", default="data/research/faber_current_signals")
    signals.add_argument("--input-csv", help="offline monthly input; requires latest completed month")
    signals.add_argument("--holdings-csv", help="optional ticker,shares CSV for paper rebalance actions")
    args = parser.parse_args(argv)
    if args.command == "list":
        for name, (description, _) in STRATEGIES.items():
            print(f"{name}: {description}")
        print("Turtle: experimental stock adaptation, outside the published catalog")
        return
    if args.command == "signals":
        from scanner.gtaa_signals import run_signals
        try:
            run_signals(args.as_of, args.output_dir, args.equity, args.input_csv, args.holdings_csv)
        except (ValueError, KeyError, IndexError, TypeError, OSError) as error:
            parser.exit(2, f"Signals unavailable: {error}\n")
        return
    # Import only the selected calculation. Both modules retain their original
    # independent interfaces and validation; no rules are changed here.
    if args.strategy == "faber-10month":
        from scanner.published_faber import main as calculate
    elif args.strategy == "faber-gtaa5":
        from scanner.published_faber_gtaa import main as calculate
    else:
        from scanner.published_gem import main as calculate
    calculate(args.input_csv, args.output_csv)
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
