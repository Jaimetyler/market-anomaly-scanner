import httpx

from scanner.config import MASSIVE_API_KEY


BASE_URL = "https://api.massive.com"


def get_daily_bars(
    ticker: str,
    start_date: str,
    end_date: str,
) -> list[dict]:
    """Fetch adjusted daily OHLCV bars from Massive."""

    ticker = ticker.upper().strip()

    url = (
        f"{BASE_URL}/v2/aggs/ticker/{ticker}"
        f"/range/1/day/{start_date}/{end_date}"
    )

    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": MASSIVE_API_KEY,
    }

    with httpx.Client(timeout=30.0) as client:
        response = client.get(url, params=params)
        response.raise_for_status()
        payload = response.json()

    if payload.get("status") not in {"OK", "DELAYED"}:
        raise RuntimeError(
            f"Massive returned an unexpected response: {payload}"
        )

    return payload.get("results", [])