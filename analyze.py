import argparse
from datetime import date, timedelta

from scanner.data import get_daily_bars
from scanner.detector import detect_anomaly
from scanner.indicators import build_snapshot


def format_pct(value):
    if value is None:
        return "N/A"

    sign = "+" if value > 0 else ""

    return f"{sign}{value:.2f}%"


def format_number(value):
    if value is None:
        return "N/A"

    return f"{value:,.2f}"


def format_money(value):
    if value is None:
        return "N/A"

    return f"${value:,.2f}"


def main():
    parser = argparse.ArgumentParser(
        description="Analyze a stock for abnormal market behavior."
    )

    parser.add_argument(
        "ticker",
        nargs="?",
        default="PHAT",
        help="Ticker symbol to analyze",
    )

    args = parser.parse_args()

    ticker = args.ticker.upper().strip()

    end_date = date.today()
    start_date = end_date - timedelta(days=450)

    print()
    print(f"Fetching {ticker} from Massive...")

    bars = get_daily_bars(
        ticker=ticker,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
    )

    if not bars:
        print(f"No daily bars returned for {ticker}.")
        return

    snapshot = build_snapshot(bars)
    detection = detect_anomaly(snapshot)

    print()
    print("=" * 58)
    print(f"  MARKET ANOMALY SCANNER — {ticker}")
    print("=" * 58)

    print(f"As of: {snapshot['as_of']}")
    print(f"Daily bars loaded: {len(bars)}")

    print()
    print("PRICE")
    print("-" * 58)

    print(
        f"Close:              "
        f"{format_money(snapshot['price'])}"
    )

    print(
        f"Open:               "
        f"{format_money(snapshot['open'])}"
    )

    print(
        f"High:               "
        f"{format_money(snapshot['high'])}"
    )

    print(
        f"Low:                "
        f"{format_money(snapshot['low'])}"
    )

    print()
    print("RETURNS")
    print("-" * 58)

    print(
        f"1 Day:              "
        f"{format_pct(snapshot['return_1d'])}"
    )

    print(
        f"2 Day:              "
        f"{format_pct(snapshot['return_2d'])}"
    )

    print(
        f"3 Day:              "
        f"{format_pct(snapshot['return_3d'])}"
    )

    print(
        f"5 Day:              "
        f"{format_pct(snapshot['return_5d'])}"
    )

    print(
        f"10 Day:             "
        f"{format_pct(snapshot['return_10d'])}"
    )

    print(
        f"20 Day:             "
        f"{format_pct(snapshot['return_20d'])}"
    )

    print()
    print("VOLUME")
    print("-" * 58)

    print(
        f"Current Volume:     "
        f"{snapshot['volume']:,.0f}"
    )

    avg_volume = snapshot["avg_volume_20d"]

    if avg_volume is not None:
        print(
            f"20D Avg Volume:     "
            f"{avg_volume:,.0f}"
        )
    else:
        print(
            "20D Avg Volume:     "
            "N/A"
        )

    rvol = snapshot["relative_volume"]

    if rvol is not None:
        print(
            f"Relative Volume:    "
            f"{rvol:.2f}x"
        )
    else:
        print(
            "Relative Volume:    "
            "N/A"
        )

    print(
        f"Dollar Volume:      "
        f"{format_money(snapshot['dollar_volume'])}"
    )

    print()
    print("MOMENTUM")
    print("-" * 58)

    rsi = snapshot["rsi_14"]

    if rsi is not None:
        print(
            f"RSI (14):           "
            f"{rsi:.2f}"
        )
    else:
        print(
            "RSI (14):           "
            "N/A"
        )

    print()
    print("MOVING AVERAGES")
    print("-" * 58)

    moving_averages = [
        (
            "SMA 10",
            snapshot["sma_10"],
            snapshot["distance_sma_10_pct"],
        ),
        (
            "SMA 20",
            snapshot["sma_20"],
            snapshot["distance_sma_20_pct"],
        ),
        (
            "SMA 50",
            snapshot["sma_50"],
            snapshot["distance_sma_50_pct"],
        ),
        (
            "SMA 200",
            snapshot["sma_200"],
            snapshot["distance_sma_200_pct"],
        ),
    ]

    for label, average, distance in moving_averages:
        if average is None:
            print(
                f"{label:<20} "
                "N/A"
            )
            continue

        print(
            f"{label:<20} "
            f"{format_money(average):>10}   "
            f"{format_pct(distance):>10}"
        )

    print()
    print("VOLATILITY")
    print("-" * 58)

    atr = snapshot["atr_14"]

    if atr is not None:
        print(
            f"ATR (14):           "
            f"{format_money(atr)}"
        )
    else:
        print(
            "ATR (14):           "
            "N/A"
        )

    atr_expansion = snapshot["atr_expansion"]

    if atr_expansion is not None:
        print(
            f"ATR Expansion:      "
            f"{atr_expansion:.2f}x"
        )
    else:
        print(
            "ATR Expansion:      "
            "N/A"
        )

    print()
    print("ANOMALY DETECTION")
    print("-" * 58)

    if not detection.eligible:
        print(
            "Eligible:            "
            "NO"
        )

        print(
            "Candidate:           "
            "NO"
        )

        print()
        print("Failed filters:")

        for failure in detection.failed_filters:
            print(
                f"  - {failure}"
            )

    else:
        print(
            "Eligible:            "
            "YES"
        )

        if detection.candidate:
            print(
                "Candidate:           "
                "YES"
            )
        else:
            print(
                "Candidate:           "
                "NO"
            )

        if detection.triggers:
            print()
            print("Primary triggers:")

            for trigger in detection.triggers:
                print(
                    f"  - {trigger}"
                )

        if detection.evidence:
            print()
            print("Supporting evidence:")

            for item in detection.evidence:
                print(
                    f"  - {item}"
                )

        if (
            not detection.triggers
            and not detection.evidence
        ):
            print()
            print(
                "No meaningful upside anomaly "
                "detected."
            )

    print()
    print("=" * 58)


if __name__ == "__main__":
    main()