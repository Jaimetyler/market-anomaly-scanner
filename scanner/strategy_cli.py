"""Small command line catalog of published strategy reference calculations."""

import argparse


STRATEGIES = {
    "faber-10month": ("Single-asset 10-month moving average", "scanner.published_faber"),
    "faber-gtaa5": ("Five-asset GTAA, equal 20% sleeves", "scanner.published_faber_gtaa"),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Published strategy references")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="show available published strategies")
    run = commands.add_parser("run", help="calculate one strategy from monthly CSV data")
    run.add_argument("strategy", choices=STRATEGIES)
    run.add_argument("input_csv", help="monthly input CSV")
    run.add_argument("output_csv", help="result CSV")
    args = parser.parse_args(argv)
    if args.command == "list":
        for name, (description, _) in STRATEGIES.items():
            print(f"{name}: {description}")
        print("Turtle: experimental stock adaptation, outside the published catalog")
        return
    # Import only the selected calculation. Both modules retain their original
    # independent interfaces and validation; no rules are changed here.
    if args.strategy == "faber-10month":
        from scanner.published_faber import main as calculate
    else:
        from scanner.published_faber_gtaa import main as calculate
    calculate(args.input_csv, args.output_csv)
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
