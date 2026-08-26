#!/usr/bin/env python3
"""Flask web app for viewing stock charts."""

from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify
from io import BytesIO
import os
from config import TICKERS
from draw import draw_chart
from garch_model import forecast_volatility, get_garch_stats

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Available options
PERIODS = ['1D', '5D', '1M', '3M', '6M', 'YTD', '1Y', '5Y', 'MAX']
GROUPINGS = ['daily', 'weekly', 'monthly']

# In-memory cache for generated charts and their data
chart_cache = {}
chart_data_cache = {}


def get_chart_key(ticker, period, grouping):
    """Generate cache key for chart."""
    return f"{ticker}_{period.lower()}_{grouping}"


def get_chart_data_for_tooltip(ticker, period_days):
    """Get OHLC data for chart tooltips."""
    import sqlite3
    from db import DB_PATH

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    from datetime import datetime, timedelta
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=period_days)

    cursor.execute('''
        SELECT date, open, high, low, close, volume
        FROM prices
        WHERE ticker = ? AND date BETWEEN ? AND ?
        ORDER BY date
    ''', (ticker, start_date, end_date))

    rows = cursor.fetchall()
    conn.close()

    data = []
    for row in rows:
        data.append({
            'date': row['date'],
            'open': round(float(row['open']), 2),
            'high': round(float(row['high']), 2),
            'low': round(float(row['low']), 2),
            'close': round(float(row['close']), 2),
            'volume': int(row['volume'])
        })

    return data


def ensure_chart_exists(ticker, period, grouping='daily'):
    """Generate and cache chart if it doesn't exist in cache."""
    cache_key = get_chart_key(ticker, period, grouping)

    if cache_key not in chart_cache:
        try:
            chart_bytes = draw_chart(ticker, period, get_period_days(period), grouping, include_forecast=True)
            chart_cache[cache_key] = chart_bytes
            # Also cache the data for tooltips
            chart_data_cache[cache_key] = get_chart_data_for_tooltip(ticker, get_period_days(period))
        except Exception as e:
            print(f"Error generating chart: {e}")
            return None

    return cache_key if chart_cache.get(cache_key) else None


def get_period_days(period):
    """Map period to days."""
    period_map = {
        '1D': 1,
        '5D': 5,
        '1M': 30,
        '3M': 90,
        '6M': 180,
        'YTD': 365,
        '1Y': 365,
        '5Y': 1825,
        'MAX': 1825
    }
    return period_map.get(period, 180)


@app.route('/')
def index():
    """Home page - redirect to first ticker."""
    return redirect(url_for('chart', ticker=TICKERS[0]))


@app.route('/chart')
def chart():
    """Display chart for selected ticker."""
    ticker = request.args.get('ticker', TICKERS[0]).upper()
    period = request.args.get('period', '6M')
    grouping = 'daily'  # Always use daily grouping

    # Validate inputs
    if ticker not in TICKERS:
        ticker = TICKERS[0]
    if period not in PERIODS:
        period = '6M'

    # Ensure chart exists in cache
    chart_key = ensure_chart_exists(ticker, period, grouping)

    if not chart_key:
        error_msg = f"Could not generate chart for {ticker}"
        return render_template('error.html', error=error_msg), 500

    return render_template('chart_sidebar.html',
                          ticker=ticker,
                          period=period,
                          chart_key=chart_key,
                          tickers=TICKERS,
                          periods=PERIODS)


@app.route('/api/chart')
def api_chart():
    """API endpoint for getting chart info."""
    ticker = request.args.get('ticker', TICKERS[0]).upper()
    period = request.args.get('period', '6M')
    grouping = request.args.get('grouping', 'daily')

    # Validate inputs
    if ticker not in TICKERS:
        return {'error': f'Invalid ticker: {ticker}'}, 400
    if period not in PERIODS:
        return {'error': f'Invalid period: {period}'}, 400
    if grouping not in GROUPINGS:
        return {'error': f'Invalid grouping: {grouping}'}, 400

    # Ensure chart exists in cache
    chart_key = ensure_chart_exists(ticker, period, grouping)

    if not chart_key:
        return {'error': f'Could not generate chart for {ticker}'}, 500

    return {
        'ticker': ticker,
        'period': period,
        'grouping': grouping,
        'chart_key': chart_key,
        'chart_url': f'/chart-image/{chart_key}'
    }


@app.route('/chart-image/<chart_key>')
def serve_chart_image(chart_key):
    """Serve chart image from memory cache."""
    if chart_key not in chart_cache:
        return {'error': 'Chart not found'}, 404

    img_bytes = chart_cache[chart_key]
    return send_file(
        BytesIO(img_bytes),
        mimetype='image/png',
        as_attachment=False
    )


@app.route('/api/garch/<ticker>')
def api_garch(ticker):
    """Get GARCH volatility forecast for a ticker."""
    ticker = ticker.upper()

    if ticker not in TICKERS:
        return {'error': f'Invalid ticker: {ticker}'}, 400

    periods = request.args.get('periods', 5, type=int)

    result = forecast_volatility(ticker, periods=periods)
    return result


@app.route('/api/garch-stats/<ticker>')
def api_garch_stats(ticker):
    """Get GARCH model statistics for a ticker."""
    ticker = ticker.upper()

    if ticker not in TICKERS:
        return {'error': f'Invalid ticker: {ticker}'}, 400

    stats = get_garch_stats(ticker)

    if stats is None:
        return {'error': f'Could not calculate GARCH stats for {ticker}'}, 500

    return stats


@app.route('/api/chart-data/<chart_key>')
def api_chart_data(chart_key):
    """Get OHLC data for chart tooltips."""
    if chart_key not in chart_data_cache:
        return {'error': 'Chart data not found'}, 404

    return {'data': chart_data_cache[chart_key]}


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(debug=True, host='0.0.0.0', port=port)
