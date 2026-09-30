# Published strategy command catalog

From the repository root:

```bash
python -m scanner.strategy_cli list
python -m scanner.strategy_cli run faber-gtaa5 faber_gtaa_etf_proxy_input.csv gtaa_output.csv
python -m scanner.strategy_cli run faber-10month single_asset_monthly.csv single_output.csv
```

`faber-10month` uses `scanner.published_faber`; `faber-gtaa5` uses `scanner.published_faber_gtaa`. This command only selects one of the existing reference calculations. It does not fetch data, change the rules, execute trades, or run the anomaly scanner. Turtle is shown as experimental and is not a published-strategy selection.

Input formats and data limitations: `docs/faber_10month.md`, `docs/faber_five_asset_gtaa.md`, and `docs/faber_gtaa_etf_proxy_2007_2026.md`. The included five-ETF input CSV is a proxy study, not Faber's original Global Financial Data history.

Current ETF targets and indicative paper sizing:

```bash
python -m scanner.strategy_cli signals faber-gtaa5 --equity 100000
```

This separate command fetches data and creates a report; it never submits
orders. Monthly confirmation, holdings input, and offline use are documented
in `docs/faber_current_signals.md`.
