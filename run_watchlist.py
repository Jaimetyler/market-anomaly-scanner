"""Run the existing watchlist with Massive Stocks Basic request pacing."""

import sys
import time
from datetime import date, datetime, timedelta
from unittest.mock import patch
from zoneinfo import ZoneInfo

import httpx
import scan_market
from scanner import grouped


def main():
    session_cutoff = datetime.now(ZoneInfo("America/New_York")).date() - timedelta(days=1)

    class WatchlistDate(date):
        @classmethod
        def today(cls):
            return session_cutoff

    original_send = httpx.Client.send
    next_request = 0.0

    def paced_send(client, request, *args, **kwargs):
        nonlocal next_request
        if request.url.host != "api.massive.com":
            return original_send(client, request, *args, **kwargs)

        key = request.url.params.get("apiKey")
        if key:
            request.url = request.url.copy_remove_param("apiKey")
            request.headers["Authorization"] = f"Bearer {key}"

        for attempt in (1, 2):
            time.sleep(max(0, next_request - time.monotonic()))
            next_request = time.monotonic() + 13
            response = original_send(client, request, *args, **kwargs)
            if response.status_code != 429 or attempt == 2:
                return response
            response.close()
            print("Massive rate limit reached. Waiting 65 seconds before retrying.")
            next_request = time.monotonic() + 65

    # The older scanner uses date.today() for both the grouped request and the
    # per-stock range. Newer grouped code already handles today's data cutoff.
    args = [
        sys.argv[0], "--limit", "5", "--show", "5",
        "--paper-plan", "data/research/paper_watchlist/ledger.json", *sys.argv[1:],
    ]
    with (
        patch.object(httpx.Client, "send", paced_send),
        patch.object(grouped, "date", WatchlistDate, create=True),
        patch.object(scan_market, "date", WatchlistDate),
        patch.object(sys, "argv", args),
    ):
        scan_market.main()


if __name__ == "__main__":
    main()
