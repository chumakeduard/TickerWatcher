#!/usr/bin/env python3
"""Flask web app for viewing stock charts."""

from flask import Flask, render_template, request, redirect, url_for
from pathlib import Path
import sys
import os
from config import TICKERS
from draw import draw_chart

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Available options
PERIODS = ['1D', '5D', '1M', '3M', '6M', 'YTD', '1Y', '5Y', 'MAX']
GROUPINGS = ['daily', 'weekly', 'monthly']


def ensure_chart_exists(ticker, period, grouping='daily'):
    """Generate chart if it doesn't exist."""
    images_dir = Path(__file__).parent / 'images'
    chart_file = images_dir / f'{ticker}_{period.lower()}.png'

    if not chart_file.exists():
        try:
            draw_chart(ticker, period, get_period_days(period), grouping)
        except Exception as e:
            print(f"Error generating chart: {e}")
            return None

    return chart_file.name if chart_file.exists() else None


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
    grouping = request.args.get('grouping', 'daily')

    # Validate inputs
    if ticker not in TICKERS:
        ticker = TICKERS[0]
    if period not in PERIODS:
        period = '6M'
    if grouping not in GROUPINGS:
        grouping = 'daily'

    # Ensure chart exists
    chart_file = ensure_chart_exists(ticker, period, grouping)

    if not chart_file:
        error_msg = f"Could not generate chart for {ticker}"
        return render_template('error.html', error=error_msg), 500

    return render_template('chart.html',
                          ticker=ticker,
                          period=period,
                          grouping=grouping,
                          chart_file=chart_file,
                          tickers=TICKERS,
                          periods=PERIODS,
                          groupings=GROUPINGS)


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

    # Ensure chart exists
    chart_file = ensure_chart_exists(ticker, period, grouping)

    if not chart_file:
        return {'error': f'Could not generate chart for {ticker}'}, 500

    return {
        'ticker': ticker,
        'period': period,
        'grouping': grouping,
        'chart_file': chart_file,
        'chart_url': f'/images/{chart_file}'
    }


@app.route('/images/<filename>')
def serve_image(filename):
    """Serve chart images."""
    from flask import send_from_directory
    images_dir = Path(__file__).parent / 'images'
    return send_from_directory(images_dir, filename)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
