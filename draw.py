#!/usr/bin/env python3
"""Generate professional stock market chart images."""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta
from io import BytesIO
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for Flask
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import sqlite3
from db import DB_PATH


def get_ticker_data(ticker, period_days):
    """Fetch ticker data from database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

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

    if not rows:
        print(f"No data found for {ticker}")
        sys.exit(1)

    dates = [row['date'] for row in rows]
    opens = [row['open'] for row in rows]
    highs = [row['high'] for row in rows]
    lows = [row['low'] for row in rows]
    closes = [row['close'] for row in rows]
    volumes = [row['volume'] for row in rows]

    return dates, opens, highs, lows, closes, volumes


def get_price_stats(opens, highs, lows, closes, volumes):
    """Calculate price statistics."""
    current_close = closes[-1]
    prev_close = opens[0]
    change = current_close - prev_close
    change_pct = (change / prev_close) * 100 if prev_close != 0 else 0

    return {
        'current': current_close,
        'change': change,
        'change_pct': change_pct,
        'high_52w': max(highs),
        'low_52w': min(lows),
        'avg_volume': np.mean(volumes),
        'ohlc': {
            'open': opens[-1],
            'high': highs[-1],
            'low': lows[-1],
            'close': closes[-1]
        }
    }


def draw_chart(ticker, period_name, period_days, grouping='daily', include_forecast=True, forecast_days=14, threshold_pct=10.0):
    """Generate professional stock chart image with optional GARCH forecast.

    Args:
        forecast_days: Number of days to forecast
            - 1D → 3 days
            - 5D → 5 days
            - 1M/3M/6M → 14 days
            - YTD/1Y/5Y/MAX → 21 days (1 month)
        threshold_pct: Percentage threshold for marking significant price moves (default 10%)
    """

    # Fetch data
    dates, opens, highs, lows, closes, volumes = get_ticker_data(ticker, period_days)
    stats = get_price_stats(opens, highs, lows, closes, volumes)

    # Identify days with significant price moves (drops/rises > threshold)
    significant_moves = []
    for i in range(len(closes)):
        if i == 0:
            prev_close = closes[0]
        else:
            prev_close = closes[i - 1]

        pct_change = ((closes[i] - prev_close) / prev_close * 100) if prev_close != 0 else 0

        if abs(pct_change) >= threshold_pct:
            is_drop = pct_change < 0
            significant_moves.append({
                'index': i,
                'date': dates[i],
                'pct_change': pct_change,
                'is_drop': is_drop
            })

    # Get GARCH forecast if requested
    forecast_data = None
    forecast_volatility_value = None

    if include_forecast:
        try:
            from garch_model import forecast_volatility as garch_forecast_func
            from datetime import timedelta

            forecast_result = garch_forecast_func(ticker, periods=forecast_days, days=1825)  # 5 years data
            if forecast_result.get('status') == 'success':
                current_price = closes[-1]

                # Per-day volatility term structure from the GARCH horizon forecast
                # (falls back to the flat current volatility if the array is short/missing)
                current_vol = forecast_result.get('current_volatility', 0) / 100
                per_day_vol = [v / 100 for v in forecast_result.get('forecasted_volatility', [])]
                if not per_day_vol:
                    per_day_vol = [current_vol] * forecast_days
                forecast_volatility_value = per_day_vol[0] if per_day_vol else current_vol

                # Historical mean daily return used as forecast drift (was previously ignored,
                # so every horizon was a zero-mean random walk indistinguishable from any other)
                drift = forecast_result.get('returns_mean', 0.0) / 100

                forecast_data = {
                    'dates': [],
                    'closes': [],
                    'volatility': []  # per-day vol actually used, reused by the candle renderer below
                }

                # Generate forecast: random walk with GARCH-forecasted, day-specific volatility + drift
                last_date = datetime.strptime(dates[-1], '%Y-%m-%d').date()

                # Seed the RNG deterministically from (ticker, last_date) only — NOT from
                # forecast_days/threshold. This makes the random draw sequence identical
                # regardless of how many days are requested, so a 51-day forecast is the
                # same 50-day forecast plus one more step, instead of an unrelated redraw.
                # (Previously np.random was left unseeded here, so changing forecast_days
                # by even 1 reshuffled the entire path and made adjacent-day forecasts look
                # completely unrelated to each other.)
                import hashlib
                seed_str = f"{ticker}_{last_date.isoformat()}"
                seed = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2**32)
                rng = np.random.RandomState(seed)

                for i in range(forecast_days):
                    forecast_date = last_date + timedelta(days=i+1)
                    day_vol = per_day_vol[i] if i < len(per_day_vol) else per_day_vol[-1]

                    drift_component = current_price * drift
                    shock_component = rng.normal(0, day_vol * current_price)
                    price_change = drift_component + shock_component
                    current_price = max(current_price + price_change, 0.01)

                    forecast_data['dates'].append(forecast_date.strftime('%Y-%m-%d'))
                    forecast_data['closes'].append(current_price)
                    forecast_data['volatility'].append(day_vol)
        except Exception as e:
            print(f"Could not generate GARCH forecast: {e}")
            forecast_data = None
            forecast_volatility_value = None

    # Setup figure with dark background
    fig = plt.figure(figsize=(14, 9), facecolor='#1a1a1a')
    fig.suptitle('', y=0.98)

    # Create grid layout
    gs = fig.add_gridspec(12, 10, left=0.08, right=0.95, top=0.92, bottom=0.08,
                          hspace=0.4, wspace=0.3)

    # Header area
    ax_header = fig.add_subplot(gs[0:1, :])
    ax_header.axis('off')

    # Main chart
    ax_chart = fig.add_subplot(gs[1:8, :])

    # Volume chart
    ax_volume = fig.add_subplot(gs[8:10, :])

    # ===== HEADER =====
    header_text = f"{ticker} — Stock Price"
    price_text = f"${stats['current']:.2f}"
    change_color = '#00d84f' if stats['change'] >= 0 else '#ff3333'
    change_text = f"{stats['change']:+.2f} ({stats['change_pct']:+.2f}%)"

    ax_header.text(0.02, 0.6, header_text, fontsize=18, fontweight='bold',
                   color='white', va='top', transform=ax_header.transAxes)
    ax_header.text(0.02, 0.1, price_text, fontsize=24, fontweight='bold',
                   color='white', va='top', transform=ax_header.transAxes)
    ax_header.text(0.15, 0.2, change_text, fontsize=14, fontweight='bold',
                   color=change_color, va='top', transform=ax_header.transAxes)
    ax_header.text(0.35, 0.2, "Market Open", fontsize=11, color='#888888',
                   va='top', transform=ax_header.transAxes)

    # Time period buttons
    periods = ['1D', '5D', '1M', '3M', '6M', 'YTD', '1Y', '5Y', 'MAX']
    button_x = 0.5
    for i, p in enumerate(periods):
        x_pos = button_x + (i * 0.04)
        color = '#00d84f' if p == period_name else '#555555'
        weight = 'bold' if p == period_name else 'normal'
        ax_header.text(x_pos, 0.15, p, fontsize=9, fontweight=weight,
                      color=color, va='center', transform=ax_header.transAxes)

    # SMA indicators
    ax_header.text(0.5, 0.5, "SMA 20", fontsize=9, color='#888888',
                   transform=ax_header.transAxes)
    ax_header.text(0.58, 0.5, "SMA 50", fontsize=9, color='#888888',
                   transform=ax_header.transAxes)
    ax_header.text(0.66, 0.5, "SMA 200", fontsize=9, color='#888888',
                   transform=ax_header.transAxes)

    # Ticker selector
    ax_header.text(0.88, 0.4, f"{ticker} ▾", fontsize=11, color='white',
                   fontweight='bold', transform=ax_header.transAxes)

    # ===== CANDLESTICK CHART =====
    ax_chart.set_facecolor('#1a1a1a')
    ax_chart.grid(True, color='#333333', linestyle='-', linewidth=0.3, alpha=0.3)

    # Plot candlesticks
    width = 0.6
    for i in range(len(dates)):
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]

        # Color based on up/down
        color = '#00d84f' if c >= o else '#ff3333'

        # Draw wick (high-low line)
        ax_chart.plot([i, i], [l, h], color=color, linewidth=1)

        # Draw body (open-close rectangle)
        body_top = max(o, c)
        body_bottom = min(o, c)
        rect = Rectangle((i - width/2, body_bottom), width, body_top - body_bottom,
                         facecolor=color, edgecolor=color, linewidth=0.5)
        ax_chart.add_patch(rect)

    # Plot forecasted candlesticks (future predictions)
    forecast_highs = []
    forecast_lows = []
    if forecast_data and 'closes' in forecast_data and len(forecast_data['closes']) > 0:
        forecast_closes = forecast_data['closes']
        forecast_dates = forecast_data['dates']
        forecast_vols = forecast_data.get('volatility', [])

        # Use the same deterministic per-(ticker, last_date) seeded RNG as the price-path
        # generation above, so OHLC wick sizes also stay identical across forecast_days
        # changes instead of reshuffling every candle each time the horizon is adjusted.
        import hashlib
        last_date_for_seed = datetime.strptime(dates[-1], '%Y-%m-%d').date()
        seed_str = f"{ticker}_{last_date_for_seed.isoformat()}_ohlc"
        ohlc_seed = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2**32)
        ohlc_rng = np.random.RandomState(ohlc_seed)

        # Generate realistic OHLC for forecasts using GARCH volatility
        for idx, close_price in enumerate(forecast_closes):
            i = len(dates) + idx + 1  # Position after historical data

            # Use this day's own forecasted volatility (not a single flat value for
            # every day) so uncertainty actually widens/narrows across the horizon
            day_vol = forecast_vols[idx] if idx < len(forecast_vols) else (forecast_volatility_value or 0.02)
            daily_vol = day_vol * close_price

            # Generate simple OHLC with random walk
            o = close_price + ohlc_rng.normal(0, daily_vol * 0.3)
            h = max(close_price, o) + abs(ohlc_rng.normal(0, daily_vol * 0.5))
            l = min(close_price, o) - abs(ohlc_rng.normal(0, daily_vol * 0.5))
            c = close_price

            # Ensure realistic values
            h = max(h, max(o, c))
            l = min(l, min(o, c))
            forecast_highs.append(h)
            forecast_lows.append(l)

            # Color based on up/down
            color_forecast = '#00d84f' if c >= o else '#ff3333'
            # Make forecast colors more transparent (lighter shade)
            alpha_color = '#66ff99' if c >= o else '#ff7777'

            # Draw wick (high-low line)
            ax_chart.plot([i, i], [l, h], color=alpha_color, linewidth=0.8, alpha=0.6)

            # Draw body (open-close rectangle) with dashed outline
            body_top = max(o, c)
            body_bottom = min(o, c)
            rect = Rectangle((i - width/2, body_bottom), width, body_top - body_bottom,
                             facecolor=alpha_color, edgecolor=alpha_color, linewidth=0.5,
                             alpha=0.4, linestyle='--')
            ax_chart.add_patch(rect)

    # OHLC info box (top-left of chart)
    ohlc = stats['ohlc']
    ohlc_text = f"O {ohlc['open']:.2f}  H {ohlc['high']:.2f}  L {ohlc['low']:.2f}  C {ohlc['close']:.2f}"
    ax_chart.text(0.01, 0.97, ohlc_text, fontsize=9, color='#cccccc',
                  transform=ax_chart.transAxes, va='top',
                  bbox=dict(boxstyle='round,pad=0.5', facecolor='#222222', edgecolor='#444444', alpha=0.8))

    # Crosshair on last candle
    last_i = len(dates) - 1
    ax_chart.plot([last_i, last_i], [ax_chart.get_ylim()[0], ax_chart.get_ylim()[1]],
                  color='#666666', linewidth=1, linestyle='--', alpha=0.5)
    ax_chart.plot([ax_chart.get_xlim()[0], ax_chart.get_xlim()[1]], [closes[-1], closes[-1]],
                  color='#666666', linewidth=1, linestyle='--', alpha=0.5)

    # Extend x-axis to include forecast
    x_max = len(dates) + (forecast_days if forecast_data else 0) + 1
    ax_chart.set_xlim(-1, x_max)

    # Y-axis must also account for forecast highs/lows, which can (and often do,
    # now that drift + real volatility are used) extend beyond the historical range
    y_low = min(lows + forecast_lows) if forecast_lows else min(lows)
    y_high = max(highs + forecast_highs) if forecast_highs else max(highs)
    ax_chart.set_ylim(y_low * 0.98, y_high * 1.02)
    ax_chart.set_ylabel('Price (USD)', color='#888888', fontsize=10)
    ax_chart.tick_params(colors='#666666', labelsize=9)
    ax_chart.spines['top'].set_visible(False)
    ax_chart.spines['right'].set_visible(False)
    ax_chart.spines['left'].set_color('#333333')
    ax_chart.spines['bottom'].set_color('#333333')

    # Now that y-axis limits are finalized, identify and mark significant moves in the FORECAST portion
    # (in addition to the historical significant_moves already identified earlier)
    if forecast_data and 'closes' in forecast_data:
        forecast_closes = forecast_data['closes']
        forecast_dates = forecast_data['dates']

        # Chain from the last historical close
        prev_close_for_forecast = closes[-1]

        for idx, close_price in enumerate(forecast_closes):
            pct_change = ((close_price - prev_close_for_forecast) / prev_close_for_forecast * 100) if prev_close_for_forecast != 0 else 0

            if abs(pct_change) >= threshold_pct:
                is_drop = pct_change < 0
                # Index is len(dates) + idx + 1 (to account for the position after historical data)
                x_index = len(dates) + idx + 1
                significant_moves.append({
                    'index': x_index,
                    'date': forecast_dates[idx],
                    'pct_change': pct_change,
                    'is_drop': is_drop
                })

            prev_close_for_forecast = close_price

    # Mark significant price moves (drops/rises > threshold) with vertical lines and labels
    # This is done AFTER set_ylim() so y_pos calculations use the correct, final axis limits
    y_min, y_max = ax_chart.get_ylim()
    for move in significant_moves:
        line_color = '#ff3333' if move['is_drop'] else '#00d84f'  # Red for drop, green for rise
        ax_chart.axvline(x=move['index'], color=line_color, linewidth=1.5, alpha=0.6, linestyle='-')

        # Add date label above/below the line, using the finalized y-axis range
        label_text = f"{move['date']}\n{move['pct_change']:+.1f}%"
        y_pos = y_max * 0.95 if move['is_drop'] else y_max * 0.92
        ax_chart.text(move['index'], y_pos, label_text, fontsize=8, color=line_color,
                      ha='center', va='top',
                      bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a1a', edgecolor=line_color, alpha=0.8))

    # ===== VOLUME CHART =====
    ax_volume.set_facecolor('#1a1a1a')
    ax_volume.grid(True, color='#333333', linestyle='-', linewidth=0.3, alpha=0.3)

    colors = ['#00d84f' if closes[i] >= opens[i] else '#ff3333' for i in range(len(dates))]
    ax_volume.bar(range(len(volumes)), volumes, color=colors, alpha=0.6, width=0.8)

    # Plot forecasted volume (estimated based on average historical volume)
    if forecast_data and len(forecast_data.get('closes', [])) > 0:
        avg_volume = np.mean(volumes)
        # Same deterministic per-(ticker, last_date) seeding as the price path/OHLC above,
        # so forecast volume bars also stay consistent when only forecast_days changes.
        import hashlib
        last_date_for_seed = datetime.strptime(dates[-1], '%Y-%m-%d').date()
        seed_str = f"{ticker}_{last_date_for_seed.isoformat()}_volume"
        vol_seed = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2**32)
        vol_rng = np.random.RandomState(vol_seed)
        for idx in range(len(forecast_data['closes'])):
            i = len(dates) + idx + 1
            # Generate forecast volume with some variation
            forecast_vol = avg_volume * vol_rng.uniform(0.8, 1.2)
            # Alternate colors for forecast volume (lighter shades)
            color_forecast = '#66ff99' if idx % 2 == 0 else '#ff7777'
            ax_volume.bar(i, forecast_vol, color=color_forecast, alpha=0.3, width=0.8)

    ax_volume.set_xlim(-1, x_max)
    ax_volume.set_ylabel('Volume', color='#888888', fontsize=9)
    ax_volume.tick_params(colors='#666666', labelsize=8)
    ax_volume.spines['top'].set_visible(False)
    ax_volume.spines['right'].set_visible(False)
    ax_volume.spines['left'].set_color('#333333')
    ax_volume.spines['bottom'].set_color('#333333')

    # ===== STATS PANEL (bottom right) =====
    ax_stats = fig.add_subplot(gs[10:, 5:])
    ax_stats.axis('off')

    stats_text = f"""52W High: ${stats['high_52w']:.2f}
52W Low: ${stats['low_52w']:.2f}
YTD Return: {(stats['change_pct']/12):.2f}%
1Y Return: {stats['change_pct']:.2f}%
Avg Volume: {stats['avg_volume']/1e6:.1f}M"""

    ax_stats.text(0.05, 0.95, stats_text, fontsize=9, color='#cccccc',
                  transform=ax_stats.transAxes, va='top', family='monospace',
                  bbox=dict(boxstyle='round,pad=0.8', facecolor='#222222', edgecolor='#444444', alpha=0.9))

    # Generate image as bytes instead of saving to disk
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png', facecolor='#1a1a1a', dpi=150, bbox_inches='tight')
    img_buffer.seek(0)
    plt.close()

    return img_buffer.getvalue()


def main():
    parser = argparse.ArgumentParser(description='Generate stock market chart images')
    parser.add_argument('ticker', help='Stock ticker symbol')
    parser.add_argument('--period', default='6M',
                        choices=['1D', '5D', '1M', '3M', '6M', 'YTD', '1Y', '5Y', 'MAX'],
                        help='Time period for chart (default: 6M)')
    parser.add_argument('--grouping', default='daily',
                        choices=['daily', 'weekly', 'monthly'],
                        help='Chart grouping option (default: daily)')

    args = parser.parse_args()

    # Map period to days
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

    period_days = period_map.get(args.period, 180)

    draw_chart(args.ticker.upper(), args.period, period_days, args.grouping)


if __name__ == '__main__':
    main()
