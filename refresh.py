#!/usr/bin/env python3
from datetime import datetime, timedelta
import logging
from db import init_db, get_last_price_date, needs_backfill, fetch_and_store_prices, get_stats, MIN_YEARS
from config import TICKERS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def update_ticker(conn, ticker):
    """Update a single ticker with new data."""
    last_date = get_last_price_date(conn, ticker)
    today = datetime.now().date()

    if last_date is None:
        start_date = today - timedelta(days=MIN_YEARS * 365 + 30)
        logger.info(f"{ticker}: No existing data, backfilling from {start_date}")
    else:
        start_date = last_date + timedelta(days=1)
        if (today - last_date).days <= 1:
            logger.info(f"{ticker}: Already up to date")
            return
        logger.info(f"{ticker}: Last update {last_date}, fetching new data")

    fetch_and_store_prices(conn, ticker, start_date, today)

    if needs_backfill(conn, ticker):
        min_date = get_last_price_date(conn, ticker)
        if min_date:
            backfill_start = today - timedelta(days=MIN_YEARS * 365 + 30)
            logger.info(f"{ticker}: Backfilling to meet {MIN_YEARS}-year requirement")
            fetch_and_store_prices(conn, ticker, backfill_start, min_date)


def main():
    logger.info("Starting TickerWatcher")
    conn = init_db()

    try:
        for ticker in TICKERS:
            update_ticker(conn, ticker)

        logger.info("\nData Summary:")
        get_stats(conn, TICKERS)
        logger.info("Completed successfully")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
