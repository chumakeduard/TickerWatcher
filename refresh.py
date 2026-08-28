#!/usr/bin/env python3
from datetime import datetime, timedelta
from db import (
    init_db, get_last_price_date, get_earliest_price_date, needs_backfill,
    fetch_and_store_prices, get_stats, MIN_YEARS, CRYPTO_MIN_YEARS
)
from config import TICKERS, CRYPTO
from logging_config import get_data_refresh_logger

logger = get_data_refresh_logger()


def _update(conn, ticker, is_crypto, min_years):
    """Shared delta-sync + backfill logic for both stocks and crypto.

    1. Fetch only the gap between the last stored date and today (delta sync).
    2. Regardless of whether a delta fetch happened, separately check whether
       enough total history is on hand yet — a ticker can be "up to date" for
       today's data while still being short on years of backfilled history
       (e.g. right after being added, or if a first backfill was interrupted).
    """
    last_date = get_last_price_date(conn, ticker, is_crypto=is_crypto)
    today = datetime.now().date()

    if last_date is None:
        start_date = today - timedelta(days=min_years * 365 + 30)
        logger.info(f"{ticker}: No existing data, backfilling from {start_date}")
        fetch_and_store_prices(conn, ticker, start_date, today, is_crypto=is_crypto)
    else:
        days_gap = (today - last_date).days
        if days_gap > 1:
            start_date = last_date + timedelta(days=1)
            logger.info(f"{ticker}: Fetching {days_gap}-day gap (from {start_date} to {today})")
            fetch_and_store_prices(conn, ticker, start_date, today, is_crypto=is_crypto)
        else:
            logger.info(f"{ticker}: Already up to date (last: {last_date})")

    # Always check backfill depth, even when the delta fetch above was skipped —
    # otherwise a ticker that's current on recent data but short on history would
    # never get backfilled after its first (possibly partial) fetch.
    if needs_backfill(conn, ticker, is_crypto=is_crypto):
        earliest_date = get_earliest_price_date(conn, ticker, is_crypto=is_crypto)
        if earliest_date:
            backfill_start = today - timedelta(days=min_years * 365 + 30)
            if backfill_start < earliest_date:
                logger.info(f"{ticker}: Backfilling to meet {min_years}-year requirement "
                            f"(from {backfill_start} to {earliest_date})")
                fetch_and_store_prices(conn, ticker, backfill_start, earliest_date, is_crypto=is_crypto)


def update_ticker(conn, ticker):
    """Update a single stock ticker with delta sync (only fetch what's missing)."""
    _update(conn, ticker, is_crypto=False, min_years=MIN_YEARS)


def update_crypto(conn, ticker):
    """Update a single crypto ticker with delta sync (only fetch what's missing).

    Mirrors update_ticker(): keeps full history (as far back as Yahoo Finance has
    it, e.g. BTC-USD since 2014) rather than a rolling window, so period selectors
    like 1Y/5Y/MAX behave the same as they do for stocks.
    """
    _update(conn, ticker, is_crypto=True, min_years=CRYPTO_MIN_YEARS)


def main():
    logger.info("=" * 70)
    logger.info("Starting TickerWatcher Data Refresh")
    logger.info("=" * 70)
    conn = init_db()

    try:
        # Stock tickers
        logger.info(f"\nUpdating {len(TICKERS)} stock tickers...")
        stock_stats = {}
        for ticker in TICKERS:
            # Capture initial record count
            import sqlite3
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM prices WHERE ticker = ?", (ticker,))
            before = cursor.fetchone()[0]

            update_ticker(conn, ticker)

            cursor.execute("SELECT COUNT(*) FROM prices WHERE ticker = ?", (ticker,))
            after = cursor.fetchone()[0]
            records_added = after - before
            stock_stats[ticker] = records_added
            if records_added > 0:
                logger.info(f"  {ticker}: +{records_added} records")

        # Crypto tickers
        logger.info(f"\nUpdating {len(CRYPTO)} crypto tickers...")
        crypto_stats = {}
        for ticker in CRYPTO:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM crypto_prices WHERE ticker = ?", (ticker,))
            before = cursor.fetchone()[0]

            update_crypto(conn, ticker)

            cursor.execute("SELECT COUNT(*) FROM crypto_prices WHERE ticker = ?", (ticker,))
            after = cursor.fetchone()[0]
            records_added = after - before
            crypto_stats[ticker] = records_added
            if records_added > 0:
                logger.info(f"  {ticker}: +{records_added} records")

        # Summary statistics
        logger.info("\n" + "=" * 70)
        logger.info("REFRESH SUMMARY")
        logger.info("=" * 70)
        total_stock_records = sum(stock_stats.values())
        total_crypto_records = sum(crypto_stats.values())
        logger.info(f"Stocks: +{total_stock_records} records across {len(TICKERS)} tickers")
        logger.info(f"Crypto: +{total_crypto_records} records across {len(CRYPTO)} tickers")
        logger.info(f"Total: +{total_stock_records + total_crypto_records} records")
        logger.info("Completed successfully")
        logger.info("=" * 70)
    except Exception as e:
        logger.error(f"Error during refresh: {e}", exc_info=True)
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
