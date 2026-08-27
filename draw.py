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
from garch_config import GARCH_TRAINING_DAYS


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


def compute_historical_predictions(ticker, dates, closes, garch_p, garch_q, vol_model, vol_scale=0.8, step=None):
    """Compute per-day OHLC prediction candles covering the ENTIRE displayed
    historical period (not just a sparse sample), so the overlay spans the same
    left-to-right range as the main chart — plus a smoothed trend line derived
    from the same data.

    Efficiency approach: fitting GARCH fresh for every single day would be far too
    slow for long periods, so instead we fit once per "window" of `step` days and
    use that single fit's multi-step forecast (horizon=step) to derive per-day
    volatility for every day inside the window.

    Deliberately blind to what actually happened: each window's price path is
    generated with the SAME random-walk simulation as the real forward forecast
    (drift + a Gaussian shock sized by that day's GARCH volatility, plus a smaller
    jitter for the candle's open/high/low) — not a "we know the real close" trick.
    It's re-anchored to the actual close only at each window's START (since that's
    the one piece of information a walk-forward backtest is allowed to know: "here
    is today, predict forward from here"), then simulated exactly as blindly as the
    future forecast is. The RNG is seeded deterministically per (ticker, window
    start date) so the page renders the same simulated path on every reload.

    Args:
        ticker: Stock ticker
        dates: List of date strings (aligned with `closes`)
        closes: List of closing prices (the actual historical data being charted)
        garch_p, garch_q: GARCH model order
        vol_model: 'garch' or 'egarch'
        vol_scale: Calibration multiplier applied to predicted volatility (matches
            the same factor used for the forward forecast, for a fair comparison)
        step: Days per fit window (auto-calculated if None)

    Returns:
        dict with:
          'candles': list of {'index', 'open', 'high', 'low', 'close'} — one per
              historical day that could be predicted, full displayed range
          'line': list of {'index', 'close'} — smoothed version of the same closes
    """
    import sqlite3
    import hashlib

    raw_candles = []
    n = len(closes)

    # Minimum rows required for a usable GARCH fit (checked per-window against the DB,
    # which typically holds years of history before the displayed period even starts —
    # see config.py MIN_YEARS — so this does NOT limit how far back predictions can begin
    # within the displayed chart itself).
    min_train = 50
    empty_result = {'candles': [], 'line': []}
    if n < 2:
        return empty_result

    # Auto-calculate step (fit-window size) based on total days to keep cost reasonable.
    if step is None:
        if n <= 80:
            step = 3
        elif n <= 200:
            step = 5
        elif n <= 500:
            step = 8
        else:
            step = 15

    try:
        from arch import arch_model
    except ImportError:
        return empty_result

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Walk forward from the very first displayed day to the most recent, fitting
    # once per window and simulating a random-walk price path forward from the
    # actual close at that window's start (same method as the real forecast).
    w_start = 0

    while w_start < n - 1:
        window_len = min(step, n - 1 - w_start)
        cutoff_date = dates[w_start]

        try:
            cursor.execute('''
                SELECT date, close FROM prices
                WHERE ticker = ? AND date <= ?
                ORDER BY date DESC LIMIT 1825
            ''', (ticker, cutoff_date))
            rows = cursor.fetchall()

            if len(rows) < min_train:
                w_start += window_len
                continue

            train_closes = np.array([float(r[1]) for r in reversed(rows)])
            train_returns = np.diff(np.log(train_closes)) * 100

            model = arch_model(train_returns, vol='Garch' if vol_model.lower() == 'garch' else 'EGARCH',
                                p=garch_p, q=garch_q)
            fitted = model.fit(disp='off')

            forecast_method = 'simulation' if vol_model.lower() == 'egarch' else 'analytic'
            forecast = fitted.forecast(horizon=window_len, method=forecast_method, reindex=False)
            variance_forecast = forecast.variance.values[-1, :] if hasattr(forecast.variance, 'values') else forecast.variance[-1, :]
            day_vols = (np.sqrt(variance_forecast) / 100) * vol_scale  # fraction, calibrated

            drift = float(np.mean(train_returns)) / 100
            current_price = train_closes[-1]  # actual close at w_start — the one thing this window "knows"

            # Deterministic per-window seed (ticker + window start date) so the same
            # simulated path renders on every reload — mirrors the future forecast's
            # (ticker, last_date) seeding pattern.
            seed_str = f"{ticker}_{cutoff_date}_hist"
            seed = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2**32)
            rng = np.random.RandomState(seed)

            for day_offset in range(1, window_len + 1):
                idx = w_start + day_offset
                if idx >= n:
                    break

                day_vol = day_vols[day_offset - 1] if (day_offset - 1) < len(day_vols) else day_vols[-1]

                # Same random-walk step as the real forward forecast: drift + a
                # Gaussian shock sized by this day's GARCH volatility.
                drift_component = current_price * drift
                shock_component = rng.normal(0, day_vol * current_price)
                current_price = max(current_price + drift_component + shock_component, 0.01)
                close_price = current_price

                # Same OHLC jitter formula as the real forward forecast candles.
                daily_vol = day_vol * close_price
                o = close_price + rng.normal(0, daily_vol * 0.3)
                h = max(close_price, o) + abs(rng.normal(0, daily_vol * 0.5))
                l = min(close_price, o) - abs(rng.normal(0, daily_vol * 0.5))
                c = close_price
                h = max(h, max(o, c))
                l = min(l, min(o, c))

                raw_candles.append({'index': idx, 'open': float(o), 'high': float(h),
                                     'low': float(l), 'close': float(c)})
        except Exception:
            pass

        w_start += window_len

    conn.close()

    if not raw_candles:
        return empty_result

    # Smoothed trend line: moving average over the raw predicted closes, erasing
    # the window-boundary jumps while staying close to the underlying values.
    # A wider window (and a second smoothing pass) makes the curve noticeably smoother.
    smooth_window = max(5, min(step * 2, 15))
    raw_closes = np.array([c['close'] for c in raw_candles])
    kernel = np.ones(smooth_window) / smooth_window
    padded = np.pad(raw_closes, (smooth_window // 2, smooth_window // 2), mode='edge')
    smoothed_closes = np.convolve(padded, kernel, mode='valid')[:len(raw_closes)]
    # Second pass with a smaller kernel to further round off any remaining kinks
    padded2 = np.pad(smoothed_closes, (2, 2), mode='edge')
    smoothed_closes = np.convolve(padded2, np.ones(5) / 5, mode='valid')[:len(raw_closes)]

    line = [{'index': raw_candles[i]['index'], 'close': float(smoothed_closes[i])}
            for i in range(len(raw_candles))]

    return {'candles': raw_candles, 'line': line}


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


def draw_chart(ticker, period_name, period_days, grouping='daily', include_forecast=True, forecast_days=14,
                threshold_pct=10.0, garch_p=1, garch_q=1, vol_model='garch', vol_scale=0.8, show_historical=False):
    """Generate professional stock chart image with optional GARCH forecast and historical predictions.

    Args:
        forecast_days: Number of days to forecast
            - 1D → 3 days
            - 5D → 5 days
            - 1M/3M/6M → 14 days
            - YTD/1Y/5Y/MAX → 21 days (1 month)
        threshold_pct: Percentage threshold for marking significant price moves (default 10%)
        garch_p, garch_q: GARCH model order (default (1,1), backtested as best AIC)
        vol_model: 'garch' (default, stable) or 'egarch' (asymmetric, backtested as
            numerically fragile on short windows — see backtest_garch.py notes)
        vol_scale: Volatility calibration factor (default 0.8)
        show_historical: If True, overlay blue line showing model's past predictions on historical data
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

            forecast_result = garch_forecast_func(ticker, periods=forecast_days, days=GARCH_TRAINING_DAYS,
                                                   p=garch_p, q=garch_q, vol_model=vol_model, vol_scale=vol_scale)
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
    forecast_ohlc_by_index = {}  # index -> {open, high, low, close}, for tooltip lookup
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
            forecast_ohlc_by_index[i] = {'open': float(o), 'high': float(h), 'low': float(l), 'close': float(c),
                                          'date': forecast_dates[idx] if idx < len(forecast_dates) else None}

            # Forecast candles are all blue — same color family as the historical
            # prediction overlay, since both represent "model prediction" rather than
            # real observed data (distinct from real green/red actual candlesticks)
            forecast_color = '#4a90e2'

            # Draw wick (high-low line)
            ax_chart.plot([i, i], [l, h], color=forecast_color, linewidth=0.8, alpha=0.6)

            # Draw body (open-close rectangle) with dashed outline
            body_top = max(o, c)
            body_bottom = min(o, c)
            rect = Rectangle((i - width/2, body_bottom), width, body_top - body_bottom,
                             facecolor=forecast_color, edgecolor=forecast_color, linewidth=0.5,
                             alpha=0.4, linestyle='--')
            ax_chart.add_patch(rect)

    # OHLC info box (top-left of chart)
    ohlc = stats['ohlc']
    ohlc_text = f"O {ohlc['open']:.2f}  H {ohlc['high']:.2f}  L {ohlc['low']:.2f}  C {ohlc['close']:.2f}"
    ax_chart.text(0.01, 0.97, ohlc_text, fontsize=9, color='#cccccc',
                  transform=ax_chart.transAxes, va='top',
                  bbox=dict(boxstyle='round,pad=0.5', facecolor='#222222', edgecolor='#444444', alpha=0.8))

    # "Today" separator: the boundary between real historical data and forecast.
    # Positioned just past the RIGHT EDGE of the last historical candle's body
    # (not through its center) so that candle sits entirely on the historical side —
    # everything left of this line is real data, everything right is prediction only.
    last_i = len(dates) - 1
    today_x = last_i + width / 2 + 0.05
    ax_chart.plot([today_x, today_x], [ax_chart.get_ylim()[0], ax_chart.get_ylim()[1]],
                  color='#888888', linewidth=1.2, linestyle='--', alpha=0.6)
    ax_chart.plot([ax_chart.get_xlim()[0], ax_chart.get_xlim()[1]], [closes[-1], closes[-1]],
                  color='#666666', linewidth=1, linestyle='--', alpha=0.5)

    # Historical predictions overlay: blue candlesticks (raw per-day OHLC prediction)
    # PLUS a green dotted trend line (smoothed), covering the entire displayed period —
    # same left-to-right range as the real data.
    historical_predictions = []
    pred_highs = []
    pred_lows = []
    if show_historical and len(dates) > 50:
        try:
            hist_result = compute_historical_predictions(ticker, dates, closes, garch_p, garch_q, vol_model, vol_scale=vol_scale)
            hist_candles = hist_result.get('candles', [])
            hist_line = hist_result.get('line', [])

            if hist_candles:
                pred_width = width * 0.9  # slightly narrower than actual candles so both are visible when overlapping
                for pred in hist_candles:
                    idx, o, h, l, c = pred['index'], pred['open'], pred['high'], pred['low'], pred['close']
                    pred_highs.append(h)
                    pred_lows.append(l)

                    # Blue, semi-transparent — visually distinct from actual (green/red) and
                    # future forecast (light green/red) candles
                    ax_chart.plot([idx, idx], [l, h], color='#4a90e2', linewidth=0.9, alpha=0.55, zorder=3)
                    body_top = max(o, c)
                    body_bottom = min(o, c)
                    rect = Rectangle((idx - pred_width/2, body_bottom), pred_width, max(body_top - body_bottom, 0.01),
                                      facecolor='#4a90e2', edgecolor='#4a90e2', linewidth=0.4, alpha=0.4, zorder=3)
                    ax_chart.add_patch(rect)

            forecast_line_points = []  # (index, close) pairs for the forecast-region trend extension
            if hist_line:
                line_indices = [p['index'] for p in hist_line]
                line_closes = [p['close'] for p in hist_line]

                # Continue the trend line into the future forecast region so it connects
                # seamlessly with the forward forecast instead of stopping at "today".
                if forecast_data and forecast_data.get('closes'):
                    for idx, close_price in enumerate(forecast_data['closes']):
                        f_index = len(dates) + idx + 1
                        line_indices.append(f_index)
                        line_closes.append(close_price)
                        forecast_line_points.append((f_index, close_price))

                pred_highs.append(max(line_closes))
                pred_lows.append(min(line_closes))

                ax_chart.plot(line_indices, line_closes, color='#4a90e2', linewidth=1.8, alpha=0.9,
                              linestyle=':', label='Model Predictions (historical)', zorder=4)

            if hist_candles or hist_line or forecast_line_points:
                # Merge candle OHLC + smoothed close into one lookup dict per index for tooltips.
                # Historical indices get their predicted OHLC + trend; forecast-region indices
                # (beyond the "today" boundary) get the actual forecast candle's OHLC + trend.
                merged = {}
                for c in hist_candles:
                    merged[c['index']] = {'index': c['index'], 'open': c['open'], 'high': c['high'],
                                           'low': c['low'], 'close': c['close']}
                for p in hist_line:
                    merged.setdefault(p['index'], {'index': p['index']})['line_close'] = p['close']
                for f_index, f_close in forecast_line_points:
                    entry = merged.setdefault(f_index, {'index': f_index})
                    entry['line_close'] = f_close
                    if f_index in forecast_ohlc_by_index:
                        entry.update(forecast_ohlc_by_index[f_index])
                historical_predictions = list(merged.values())

                # Legend: proxy patch for candles + the actual line handle
                legend_patch = Rectangle((0, 0), 1, 1, facecolor='#4a90e2', edgecolor='#4a90e2', alpha=0.5)
                line_handle = plt.Line2D([0], [0], color='#4a90e2', linewidth=1.8, linestyle=':')
                ax_chart.legend([legend_patch, line_handle],
                                ['Predicted OHLC (historical)', 'Predicted trend (smoothed)'],
                                loc='lower right', fontsize=9, framealpha=0.85)
        except Exception:
            pass  # Silently skip if historical predictions fail

    # Extend x-axis to include forecast
    x_max = len(dates) + (forecast_days if forecast_data else 0) + 1
    ax_chart.set_xlim(-1, x_max)

    # Y-axis must also account for forecast highs/lows, which can (and often do,
    # now that drift + real volatility are used) extend beyond the historical range
    all_lows = lows + forecast_lows + pred_lows
    all_highs = highs + forecast_highs + pred_highs
    y_low = min(all_lows)
    y_high = max(all_highs)
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

            # This runs after historical_predictions was already built, so patch the
            # matching entry (if any) with volume for tooltip display.
            for entry in historical_predictions:
                if entry.get('index') == i:
                    entry['volume'] = float(forecast_vol)
                    break

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

    # Return image bytes, prediction data, and axis metadata for accurate tooltip
    # hover-to-date mapping (the image includes the forecast region too, so a naive
    # "pixel fraction of image width = fraction of historical_count" assumption drifts
    # further off the more forecast_days extends the total plotted width)
    chart_meta = {'x_min': -1, 'x_max': x_max, 'historical_count': len(dates)}
    return img_buffer.getvalue(), historical_predictions, chart_meta


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
