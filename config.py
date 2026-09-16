#!/usr/bin/env python3
"""Configuration file for TickerWatcher - hardcoded tickers."""

TICKERS = ["AAPL", "CRM", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA", "AMD", "VGT", "VTI", "VDE", "VFINX", "VTSMX",
           "VOO", "VOX", "VNQ", "VTWAX", "VDE"]
CRYPTO = ["BTC", "ETH"]

# Ticker the app opens to when no ?ticker= is given (e.g. visiting "/").
DEFAULT_TICKER = "VGT"
