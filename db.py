#!/usr/bin/env python3
import sqlite3
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent / "database" / "prices.db"
MIN_YEARS = 50


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
    conn.commit()
    return conn


def get_last_price_date(conn, ticker):
    """Get the last date we have price data for a ticker."""
    cursor = conn.cursor()
    cursor.execute(
        'SELECT MAX(date) FROM prices WHERE ticker = ?',
        (ticker,)
    )
    result = cursor.fetchone()[0]
    return datetime.strptime(result, '%Y-%m-%d').date() if result else None


def needs_backfill(conn, ticker):
    """Check if ticker needs data backfilled to meet minimum year requirement."""
    cursor = conn.cursor()
    cursor.execute(
        'SELECT MIN(date), MAX(date) FROM prices WHERE ticker = ?',
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
    return days_span < (MIN_YEARS * 365)


def fetch_and_store_prices(conn, ticker, start_date, end_date):
    """Fetch prices from Yahoo Finance and store in database."""
    try:
        logger.info(f"Fetching {ticker} from {start_date} to {end_date}")
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)

        if data.empty:
            logger.warning(f"No data retrieved for {ticker}")
            return 0

        cursor = conn.cursor()
        inserted = 0

        for date, row in data.iterrows():
            try:
                cursor.execute('''
                    INSERT INTO prices (ticker, date, open, high, low, close, volume)
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


def get_stats(conn, tickers):
    """Get statistics about data in the database."""
    cursor = conn.cursor()
    for ticker in tickers:
        cursor.execute('''
            SELECT COUNT(*), MIN(date), MAX(date) FROM prices WHERE ticker = ?
        ''', (ticker,))
        count, min_date, max_date = cursor.fetchone()
        if count > 0:
            days_span = (datetime.strptime(max_date, '%Y-%m-%d').date() -
                        datetime.strptime(min_date, '%Y-%m-%d').date()).days
            logger.info(f"{ticker}: {count} records, {min_date} to {max_date} ({days_span} days)")
        else:
            logger.info(f"{ticker}: No data")
