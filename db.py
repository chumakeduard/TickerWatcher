#!/usr/bin/env python3
import sqlite3
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent / "database" / "prices.db"
MIN_YEARS = 50
# Crypto markets (BTC-USD, ETH-USD on Yahoo Finance) only go back to ~2014-2017,
# so a 50-year backfill window is wasted range — yfinance just returns whatever
# exists from the actual listing date either way, but a smaller window keeps the
# backfill query intentional. 15 years comfortably covers any crypto's full history.
CRYPTO_MIN_YEARS = 15


def init_db():
    """Initialize the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            date DATE NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            UNIQUE(ticker, date)
        )
    ''')
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_ticker_date ON prices (ticker, date)
    ''')

    # Crypto prices table (separate from stocks, keeps only 20 days)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS crypto_prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            date DATE NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            UNIQUE(ticker, date)
        )
    ''')
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_crypto_ticker_date ON crypto_prices (ticker, date)
    ''')
    conn.commit()
    return conn


def get_last_price_date(conn, ticker, is_crypto=False):
    """Get the last (most recent) date we have price data for a ticker."""
    cursor = conn.cursor()
    table = 'crypto_prices' if is_crypto else 'prices'
    cursor.execute(
        f'SELECT MAX(date) FROM {table} WHERE ticker = ?',
        (ticker,)
    )
    result = cursor.fetchone()[0]
    return datetime.strptime(result, '%Y-%m-%d').date() if result else None


def get_earliest_price_date(conn, ticker, is_crypto=False):
    """Get the earliest date we have price data for a ticker (used as the end
    boundary when backfilling further back in history)."""
    cursor = conn.cursor()
    table = 'crypto_prices' if is_crypto else 'prices'
    cursor.execute(
        f'SELECT MIN(date) FROM {table} WHERE ticker = ?',
        (ticker,)
    )
    result = cursor.fetchone()[0]
    return datetime.strptime(result, '%Y-%m-%d').date() if result else None


def needs_backfill(conn, ticker, is_crypto=False):
    """Check if ticker needs data backfilled to meet minimum year requirement."""
    cursor = conn.cursor()
    table = 'crypto_prices' if is_crypto else 'prices'
    cursor.execute(
        f'SELECT MIN(date), MAX(date) FROM {table} WHERE ticker = ?',
        (ticker,)
    )
    min_date, max_date = cursor.fetchone()

    if not max_date:
        return True

    max_date_obj = datetime.strptime(max_date, '%Y-%m-%d').date()
    min_date_obj = datetime.strptime(min_date, '%Y-%m-%d').date() if min_date else None

    if not min_date_obj:
        return True

    days_span = (max_date_obj - min_date_obj).days
    min_years = CRYPTO_MIN_YEARS if is_crypto else MIN_YEARS
    return days_span < (min_years * 365)


def fetch_and_store_prices(conn, ticker, start_date, end_date, is_crypto=False):
    """Fetch prices from Yahoo Finance and store in database."""
    try:
        logger.info(f"Fetching {ticker} from {start_date} to {end_date}")
        # For crypto, use ticker-usd format for yfinance
        fetch_ticker = f"{ticker}-USD" if is_crypto else ticker
        data = yf.download(fetch_ticker, start=start_date, end=end_date, progress=False)

        if data.empty:
            logger.warning(f"No data retrieved for {ticker}")
            return 0

        cursor = conn.cursor()
        inserted = 0
        table = 'crypto_prices' if is_crypto else 'prices'

        for date, row in data.iterrows():
            try:
                cursor.execute(f'''
                    INSERT INTO {table} (ticker, date, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    ticker,
                    date.date(),
                    float(row['Open']),
                    float(row['High']),
                    float(row['Low']),
                    float(row['Close']),
                    int(float(row['Volume']))
                ))
                inserted += 1
            except sqlite3.IntegrityError:
                pass

        conn.commit()
        logger.info(f"Inserted {inserted} new records for {ticker}")
        return inserted
    except Exception as e:
        logger.error(f"Error fetching {ticker}: {e}")
        return 0


def get_stats(conn, tickers, is_crypto=False):
    """Get statistics about data in the database."""
    cursor = conn.cursor()
    table = 'crypto_prices' if is_crypto else 'prices'
    for ticker in tickers:
        cursor.execute(f'''
            SELECT COUNT(*), MIN(date), MAX(date) FROM {table} WHERE ticker = ?
        ''', (ticker,))
        count, min_date, max_date = cursor.fetchone()
        if count > 0:
            days_span = (datetime.strptime(max_date, '%Y-%m-%d').date() -
                        datetime.strptime(min_date, '%Y-%m-%d').date()).days
            logger.info(f"{ticker}: {count} records, {min_date} to {max_date} ({days_span} days)")
        else:
            logger.info(f"{ticker}: No data")
