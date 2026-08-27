#!/usr/bin/env python3
"""Flask web app for viewing stock charts."""

from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify
from io import BytesIO
import os
import threading
from datetime import datetime
from config import TICKERS
from draw import draw_chart
from garch_model import forecast_volatility, get_garch_stats
from garch_config import (
    VALID_GARCH_ORDERS,
    VALID_VOL_MODELS,
    DEFAULT_VOL_SCALE,
    MIN_VOL_SCALE,
    MAX_VOL_SCALE,
    FORECAST_DAYS_BY_PERIOD,
    DEFAULT_PRICE_MOVE_THRESHOLD,
    MIN_PRICE_MOVE_THRESHOLD,
    MAX_PRICE_MOVE_THRESHOLD
)

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Available options
PERIODS = ['1D', '5D', '1M', '3M', '6M', 'YTD', '1Y', '5Y', 'MAX']
GROUPINGS = ['daily', 'weekly', 'monthly']

# In-memory cache for generated charts and their data
chart_cache = {}
chart_data_cache = {}

# Background data-refresh state, guarded by refresh_lock
refresh_lock = threading.Lock()
refresh_state = {
    'running': False,
    'last_run': None,
    'last_result': None,
    'error': None
}


def _run_refresh():
    """Runs refresh.py's update logic in a background thread, then clears
    chart caches so subsequent chart requests regenerate with the new data."""
    from db import init_db
    from refresh import update_ticker

    try:
        conn = init_db()
        try:
            for ticker in TICKERS:
                update_ticker(conn, ticker)
        finally:
            conn.close()

        # Force charts (and their cached tooltip data) to regenerate on next view
        chart_cache.clear()
        chart_data_cache.clear()

        with refresh_lock:
            refresh_state['running'] = False
            refresh_state['last_result'] = 'success'
            refresh_state['last_run'] = datetime.now().isoformat()
            refresh_state['error'] = None
    except Exception as e:
        with refresh_lock:
            refresh_state['running'] = False
            refresh_state['last_result'] = 'error'
            refresh_state['error'] = str(e)


def get_chart_key(ticker, period, grouping, threshold_pct=10.0, forecast_days=None, garch_p=1, garch_q=1,
                   vol_model='garch', vol_scale=0.8, show_historical=False):
    """Generate cache key for chart, including parameters that affect rendering."""
    if forecast_days is None:
        forecast_days = get_forecast_days(period)
    # Include threshold, forecast_days, GARCH order, vol_model, vol_scale, and show_historical
    # in the key so cache doesn't serve a stale chart when only these params change
    hist_suffix = "_hist" if show_historical else ""
    return f"{ticker}_{period.lower()}_{grouping}_th{threshold_pct:.1f}_fd{forecast_days}_p{garch_p}q{garch_q}_{vol_model}_vs{vol_scale:.2f}{hist_suffix}"


def get_garch_order(p_val, q_val):
    """Parse and validate GARCH order (p, q) from query string. Default (1, 1)."""
    try:
        p = int(p_val) if p_val else 1
        q = int(q_val) if q_val else 1
    except (ValueError, TypeError):
        return 1, 1
    return (p, q) if (p, q) in VALID_GARCH_ORDERS else (1, 1)


def get_vol_model(val):
    """Parse and validate volatility model type from query string. Default 'garch'."""
    if not val:
        return 'garch'
    val = val.lower()
    return val if val in VALID_VOL_MODELS else 'garch'


def get_vol_scale(val):
    """Parse and validate volatility calibration multiplier from query string.

    Default from garch_config.DEFAULT_VOL_SCALE (0.8) — backtesting found
    GARCH(1,1) consistently over-forecasts realized volatility by ~20-25%.
    """
    try:
        scale = float(val) if val else DEFAULT_VOL_SCALE
    except (ValueError, TypeError):
        return DEFAULT_VOL_SCALE
    return max(MIN_VOL_SCALE, min(MAX_VOL_SCALE, scale))


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


def ensure_chart_exists(ticker, period, grouping='daily', threshold_pct=10.0, forecast_days_override=None,
                         garch_p=1, garch_q=1, vol_model='garch', vol_scale=0.8, show_historical=False):
    """Generate and cache chart if it doesn't exist in cache.

    Args:
        forecast_days_override: If provided, use this instead of computing from period
        garch_p, garch_q: GARCH model order (default (1,1))
        vol_model: 'garch' (default) or 'egarch'
        vol_scale: volatility calibration multiplier (default 0.8, see get_vol_scale())
        show_historical: If True, overlay historical predictions on the chart
    """
    # Determine the actual forecast_days to use
    forecast_days = forecast_days_override
    if forecast_days is None:
        forecast_days = get_forecast_days(period)
    else:
        try:
            forecast_days = int(forecast_days_override)
        except (ValueError, TypeError):
            forecast_days = get_forecast_days(period)

    cache_key = get_chart_key(ticker, period, grouping, threshold_pct=threshold_pct, forecast_days=forecast_days,
                               garch_p=garch_p, garch_q=garch_q, vol_model=vol_model, vol_scale=vol_scale,
                               show_historical=show_historical)

    if cache_key not in chart_cache:
        try:
            period_days = get_period_days(period)
            result = draw_chart(
                ticker, period, period_days, grouping,
                include_forecast=True, forecast_days=forecast_days, threshold_pct=threshold_pct,
                garch_p=garch_p, garch_q=garch_q, vol_model=vol_model, vol_scale=vol_scale,
                show_historical=show_historical
            )
            # Unpack chart bytes, predictions data, and axis metadata
            if isinstance(result, tuple):
                if len(result) == 3:
                    chart_bytes, predictions_data, chart_meta = result
                elif len(result) == 2:
                    chart_bytes, predictions_data = result
                    chart_meta = {}
                else:
                    chart_bytes, predictions_data, chart_meta = result[0], [], {}
            else:
                chart_bytes, predictions_data, chart_meta = result, [], {}
            chart_cache[cache_key] = chart_bytes
            # Also cache the data for tooltips, predictions, and axis metadata
            chart_data_cache[cache_key] = {
                'ohlcv': get_chart_data_for_tooltip(ticker, period_days),
                'predictions': predictions_data,
                'meta': chart_meta
            }
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
        # MAX must actually mean "all available history", not the same 5 years as
        # 5Y — some tickers (e.g. MSFT) have 30-40+ years of data. 36500 days
        # (100 years) safely covers any ticker's full history without a DB query.
        'MAX': 36500
    }
    return period_map.get(period, 180)


def get_forecast_days(period):
    """Map period to forecast days.

    - 1D → 3 days
    - 5D → 5 days
    - 1M/3M/6M → 14 days
    - YTD/1Y/5Y/MAX → 21 days
    """
    period_map = {
        '1D': 3,
        '5D': 5,
        '1M': 14,
        '3M': 14,
        '6M': 14,
        'YTD': 21,
        '1Y': 21,
        '5Y': 21,
        'MAX': 21
    }
    return period_map.get(period, 14)


def get_threshold_pct(query_val):
    """Parse threshold percentage from query string (default 10)."""
    try:
        val = float(query_val) if query_val else 10.0
        return max(0.1, min(100.0, val))  # Clamp to 0.1-100%
    except (ValueError, TypeError):
        return 10.0


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
    threshold_pct = get_threshold_pct(request.args.get('threshold', '10'))
    forecast_days_override = request.args.get('forecast_days')
    garch_p, garch_q = get_garch_order(request.args.get('garch_p'), request.args.get('garch_q'))
    vol_model = get_vol_model(request.args.get('vol_model'))
    vol_scale = get_vol_scale(request.args.get('vol_scale'))
    show_historical = request.args.get('show_historical', '0') == '1'

    # Validate inputs
    if ticker not in TICKERS:
        ticker = TICKERS[0]
    if period not in PERIODS:
        period = '6M'

    # Calculate the effective forecast_days that will be used
    if forecast_days_override:
        try:
            effective_forecast_days = int(forecast_days_override)
        except (ValueError, TypeError):
            effective_forecast_days = get_forecast_days(period)
    else:
        effective_forecast_days = get_forecast_days(period)

    # Ensure chart exists in cache, passing threshold, forecast_days, and GARCH model settings
    chart_key = ensure_chart_exists(ticker, period, grouping, threshold_pct=threshold_pct,
                                   forecast_days_override=forecast_days_override,
                                   garch_p=garch_p, garch_q=garch_q, vol_model=vol_model, vol_scale=vol_scale,
                                   show_historical=show_historical)

    if not chart_key:
        error_msg = f"Could not generate chart for {ticker}"
        return render_template('error.html', error=error_msg), 500

    return render_template('chart_sidebar.html',
                          ticker=ticker,
                          period=period,
                          chart_key=chart_key,
                          tickers=TICKERS,
                          periods=PERIODS,
                          threshold_pct=threshold_pct,
                          forecast_days_override=forecast_days_override,
                          effective_forecast_days=effective_forecast_days,
                          garch_p=garch_p,
                          garch_q=garch_q,
                          vol_model=vol_model,
                          vol_scale=vol_scale,
                          show_historical=show_historical,
                          valid_orders=VALID_GARCH_ORDERS)


@app.route('/api/chart')
def api_chart():
    """API endpoint for getting chart info."""
    ticker = request.args.get('ticker', TICKERS[0]).upper()
    period = request.args.get('period', '6M')
    grouping = request.args.get('grouping', 'daily')
    threshold_pct = get_threshold_pct(request.args.get('threshold', '10'))
    forecast_days_override = request.args.get('forecast_days')

    # Validate inputs
    if ticker not in TICKERS:
        return {'error': f'Invalid ticker: {ticker}'}, 400
    if period not in PERIODS:
        return {'error': f'Invalid period: {period}'}, 400
    if grouping not in GROUPINGS:
        return {'error': f'Invalid grouping: {grouping}'}, 400

    # Ensure chart exists in cache
    chart_key = ensure_chart_exists(ticker, period, grouping, threshold_pct=threshold_pct,
                                   forecast_days_override=forecast_days_override)

    if not chart_key:
        return {'error': f'Could not generate chart for {ticker}'}, 500

    return {
        'ticker': ticker,
        'period': period,
        'grouping': grouping,
        'chart_key': chart_key,
        'chart_url': f'/chart-image/{chart_key}',
        'threshold_pct': threshold_pct,
        'forecast_days': forecast_days_override or get_forecast_days(period)
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
    garch_p, garch_q = get_garch_order(request.args.get('garch_p'), request.args.get('garch_q'))
    vol_model = get_vol_model(request.args.get('vol_model'))

    result = forecast_volatility(ticker, periods=periods, p=garch_p, q=garch_q, vol_model=vol_model)
    return result


@app.route('/api/garch-stats/<ticker>')
def api_garch_stats(ticker):
    """Get GARCH model statistics for a ticker, including fitted coefficients.

    Accepts optional ?garch_p=&garch_q=&vol_model= query params to match the
    model settings the chart itself was generated with.
    """
    ticker = ticker.upper()

    if ticker not in TICKERS:
        return {'error': f'Invalid ticker: {ticker}'}, 400

    garch_p, garch_q = get_garch_order(request.args.get('garch_p'), request.args.get('garch_q'))
    vol_model = get_vol_model(request.args.get('vol_model'))

    stats = get_garch_stats(ticker, p=garch_p, q=garch_q, vol_model=vol_model)
    # Note: get_garch_stats reports current/average/max realized volatility from the
    # model fit itself (not a forward forecast), so vol_scale calibration doesn't apply here

    if stats is None:
        return {'error': f'Could not calculate GARCH stats for {ticker}'}, 500

    return stats


@app.route('/api/chart-data/<chart_key>')
def api_chart_data(chart_key):
    """Get OHLC data for chart tooltips and prediction data."""
    if chart_key not in chart_data_cache:
        return {'error': 'Chart data not found'}, 404

    cached_data = chart_data_cache[chart_key]
    # Handle both old format (list) and new format (dict)
    if isinstance(cached_data, dict):
        return {
            'data': cached_data.get('ohlcv', []),
            'predictions': cached_data.get('predictions', []),
            'meta': cached_data.get('meta', {})
        }
    else:
        return {'data': cached_data, 'predictions': [], 'meta': {}}


@app.route('/api/refresh', methods=['POST'])
def api_refresh_start():
    """Kick off a background data refresh (catches up all tickers to today)."""
    with refresh_lock:
        if refresh_state['running']:
            return {'status': 'already_running'}, 409
        refresh_state['running'] = True
        refresh_state['last_result'] = None
        refresh_state['error'] = None

    thread = threading.Thread(target=_run_refresh, daemon=True)
    thread.start()
    return {'status': 'started'}


@app.route('/api/refresh/status')
def api_refresh_status():
    """Poll the status of a background data refresh."""
    with refresh_lock:
        return dict(refresh_state)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(debug=True, host='0.0.0.0', port=port)
