import argparse
from datetime import date, timedelta

from scanner.data import get_daily_bars


def main():
    parser = argparse.ArgumentParser(
        description="Analyze a stock for abnormal market behavior."
    )

    parser.add_argument(
        "ticker",
        nargs="?",
        default="PHAT",
        help="Ticker symbol to analyze (default: PHAT)",
    )

    args = parser.parse_args()

    ticker = args.ticker.upper().strip()

    end_date = date.today()
    start_date = end_date - timedelta(days=365)

    print(f"Fetching {ticker} from Massive...")

    bars = get_daily_bars(
        ticker=ticker,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
    )

    if not bars:
        print(f"No daily bars returned for {ticker}.")
        return

    latest = bars[-1]

    print()
    print("=" * 45)
    print(f"  MARKET ANOMALY SCANNER — {ticker}")
    print("=" * 45)

    print(f"Daily bars received: {len(bars)}")
    print()

    print("LATEST DAILY BAR")
    print("-" * 45)
    print(f"Open:       ${latest['o']:,.2f}")
    print(f"High:       ${latest['h']:,.2f}")
    print(f"Low:        ${latest['l']:,.2f}")
    print(f"Close:      ${latest['c']:,.2f}")
    print(f"Volume:     {latest['v']:,.0f}")

    if "vw" in latest:
        print(f"VWAP:       ${latest['vw']:,.2f}")

    print("=" * 45)


if __name__ == "__main__":
    main()