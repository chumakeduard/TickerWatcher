# TickerWatcher - Technical Implementation Guide

This document provides detailed technical information about the TickerWatcher application, particularly focusing on the GARCH forecasting features and implementation details.

📋 **For a running log of every bug found and fixed, see [CHANGELOG.md](CHANGELOG.md).**

## Last Updated
August 26, 2026

## Current Status

✅ **Complete Implementation**
- GARCH(1,1) volatility forecasting model
- Dynamic forecast periods (3-21 days based on selected timeframe)
- Forecast volume visualization
- Grouping selector removed from UI
- Interactive tooltips for historical data
- In-memory chart caching
- Intelligent prediction scaling (short periods = short forecasts)
- Forecasts use per-day GARCH volatility term structure + historical drift (fixed Aug 26, 2026 — see "Forecast Logic Bug Fix" below)
- One-click data refresh button in sidebar (fixed Aug 26, 2026)
- **Threshold and Forecast-Days Control Panel** (Aug 26, 2026)
  - Adjustable price-move threshold (default 10%, range 0.1-100%) to mark significant drops/rises
  - Adjustable forecast period length (1-60 days, default based on chart period)
  - Vertical lines with date tags mark moves exceeding threshold (red for drops, green for rises)
  - Marks both historical AND forecast/prediction portions of chart
  - Both parameters settable via URL query string for bookmarkable configurations
  - Cache properly invalidated when only threshold/forecast-days changes

## Forecast Logic Bug Fix (Aug 26, 2026)

**Symptom reported:** predictions for different periods (3-day, 5-day, 14-day) looked "very matching" — statistically indistinguishable regardless of horizon length.

**Root causes found in `draw.py`:**

1. **Global fixed seed** — `np.random.seed(42)` at module import time pinned the *entire app's* random stream to one deterministic sequence, making forecasts replay identically across restarts.
2. **Flat volatility (the real bug)** — `garch_model.forecast_volatility()` already computed a full per-day `forecasted_volatility` array (real GARCH term structure, since variance typically evolves toward the long-run level over the horizon), but `draw.py` only ever used the single scalar `current_volatility` for *every* simulated day. A 3-day and a 21-day forecast therefore used identical daily noise size — the only difference was how many times the same-sized step repeated.
3. **Zero drift** — `np.random.normal(0, ...)` had mean 0; the historical mean return was computed elsewhere (`get_garch_stats`) but never passed into the forecast, so there was no directional signal at all, just noise around the last close.
4. **Candle body generator repeated the same bug** — every forecast day's OHLC wick size used the same flat scalar, so uncertainty never widened further out.

**Fix:**
- Removed the global `np.random.seed(42)`.
- `garch_model.forecast_volatility()` now also returns `returns_mean` (historical daily mean log return over the same fit window).
- `draw_chart()` now consumes the **actual per-day `forecasted_volatility` array** (falls back to the flat value only if the array is empty) for both the price random walk and the candle-body OHLC generation, and adds the historical **drift** term to each day's step.
- **Follow-on bug this exposed:** with real drift, forecast prices can move outside the historical high/low range — but `ax_chart.set_ylim()` was sized only from historical `lows`/`highs`, silently clipping forecast candles off-screen. Fixed by folding `forecast_highs`/`forecast_lows` (collected while generating forecast candles) into the y-axis range calculation.

**Files touched:** `garch_model.py` (`forecast_volatility()`), `draw.py` (`draw_chart()`, seed removal, y-axis calc).

**Caveat to keep in mind:** the day-to-day *volatility* now genuinely differs across horizons (GARCH term structure), but since all periods fit the model on the same fixed 3-year window and start from the same last close, the first N days of a long forecast will still resemble a fresh N-day forecast in overall *character* — they're no longer numerically flat/identical, but don't expect a completely different "shape" for the same ticker across periods. This is inherent to using one GARCH fit for all horizons, not a bug.

## Architecture Overview

### Technology Stack

```
Frontend:
  - HTML5/CSS3/JavaScript
  - Flask templates (Jinja2)
  - Interactive tooltips with mouse tracking

Backend:
  - Python 3.9+
  - Flask web framework
  - SQLite database
  - GARCH(1,1) model (arch library)
  - Matplotlib (Agg backend for server)

External:
  - Yahoo Finance (yfinance)
  - numpy, pandas for data processing
```

### Port Configuration

- **Development**: Port 8080 (changed from 5000 due to macOS conflicts)
- Can be overridden with `PORT` environment variable
- Example: `PORT=5000 python app.py`

## Refresh Data Button (Aug 26, 2026)

A "🔄 Refresh Data" button is pinned to the bottom of the sidebar (below the GARCH stats panel).

**Backend (`app.py`):**
- `POST /api/refresh` — starts `refresh.update_ticker()` for every configured ticker in a background `threading.Thread` (daemon), guarded by `refresh_lock` so only one refresh can run at a time (`409 already_running` if one is already in flight). On success, clears both `chart_cache` and `chart_data_cache` so every chart regenerates from the caught-up data on next view.
- `GET /api/refresh/status` — returns `{running, last_run, last_result, error}` for polling.

**Frontend (`templates/chart_sidebar.html`):**
- Click → `POST /api/refresh`, button disabled with "⏳ Refreshing..." text.
- Polls `/api/refresh/status` every 1.5s; on completion, shows a status message and reloads the page (`window.location.reload()`) so the chart re-renders with fresh data.
- On page load, also checks `/api/refresh/status` once — if a refresh was already running (started from another tab), immediately resumes polling instead of missing it.

**Note:** this reuses `refresh.update_ticker()` from `refresh.py` directly (same incremental-fetch + backfill logic as the CLI script), so behavior is identical to running `python refresh.py` manually.

## Threshold and Forecast-Days Control Panel (Aug 26, 2026)

A new "⚙️ Chart Controls" section at the top of the sidebar allows real-time adjustment of chart visualization parameters.

**UI Controls (`templates/chart_sidebar.html`):**
- **Price Move Threshold (%)** — number input (default 10, range 0.1-100)
  - Marks all historical and forecast days where price moved ≥ threshold amount
  - User can adjust on-the-fly; clicking "Apply" navigates to URL with new parameters
  - Enter key also applies changes
- **Forecast Days** — number input (default 14, range 1-60)
  - Overrides the `get_forecast_days(period)` default for the selected chart period
  - Example: can request 30-day forecast on a 6M chart instead of the default 14
  - Enter key also applies changes

**Backend (`app.py`):**
- `get_threshold_pct(query_val)` — parses threshold from URL, validates range, returns float (default 10.0)
- `get_chart_key(ticker, period, grouping, threshold_pct, forecast_days)` — **updated to include both threshold and forecast_days in cache key**
  - Prevents stale cache hits when only these parameters change
  - Cache key format: `{ticker}_{period}_{grouping}_th{threshold:.1f}_fd{forecast_days}`
- `ensure_chart_exists(..., forecast_days_override)` — accepts and passes override to `draw_chart()`
- `/chart` route now:
  1. Extracts `threshold` and `forecast_days` from query string
  2. Validates `threshold_pct` via `get_threshold_pct()`
  3. Calculates `effective_forecast_days` (override if provided, else period-based default)
  4. Passes both to `ensure_chart_exists()`
  5. Passes `effective_forecast_days` to template for use in navigation links

**Chart Generation (`draw.py`):**
- `draw_chart(..., threshold_pct=10.0, forecast_days=14)` — accepts both parameters
- **Significant-move detection:**
  - Scans historical data for any day where `|pct_change| >= threshold_pct` (compared to previous close)
  - **NEW:** Also scans forecast data — chains from last historical close through all forecast closes
  - Records index, date, pct_change, and is_drop (True if negative) for each qualifying move
- **Significant-move rendering (moved AFTER `set_ylim()` for correct positioning):**
  - Draws vertical line at each qualifying move: red for drops, green for rises
  - Adds date + pct_change label with colored bbox
  - Label y-position calculated using **finalized** y-axis limits (not stale auto-scaled bounds)

**URL Query String Support:**
```
/chart?ticker=AAPL&period=6M&threshold=5&forecast_days=30
```
Both `threshold` and `forecast_days` are optional; defaults apply if omitted.

**Navigation Link Propagation:**
- Ticker selector links now carry forward current `threshold` and `effective_forecast_days` in URL
- Period selector buttons now carry forward current `threshold` and `effective_forecast_days` in URL
- Result: changing ticker or period preserves the user's custom settings

**Verified Behavior:**
- ✅ Cache invalidation: threshold=5 and threshold=10 generate different charts (different markers)
- ✅ Forecast-days override: forecast_days=30 shows 30 candlesticks vs default 14
- ✅ Parameter persistence: ticker/period links preserve threshold and forecast_days settings
- ✅ Forecast portion marked: significant moves in forecast section now have vertical lines and labels

## Dynamic Forecast Periods Feature

### Intelligent Prediction Scaling

The forecast period adapts based on the selected time range:

```python
def get_forecast_days(period):
    """Map period to forecast days."""
    period_map = {
        '1D': 3,        # 3-day forecast for 1-day chart
        '5D': 5,        # 5-day forecast for 1-week chart
        '1M': 14,       # 2-week forecast for 1-month chart
        '3M': 14,       # 2-week forecast for 3-month chart
        '6M': 14,       # 2-week forecast for 6-month chart
        'YTD': 21,      # 1-month forecast for YTD chart
        '1Y': 21,       # 1-month forecast for 1-year chart
        '5Y': 21,       # 1-month forecast for 5-year chart
        'MAX': 21       # 1-month forecast for max history
    }
```

### Benefits

- **Short-term charts** (1D, 5D): Focused, near-term predictions
- **Medium-term charts** (1M-6M): Consistent 2-week outlook
- **Long-term charts** (YTD+): Extended 1-month strategic forecast
- **Proportional scaling**: Forecast length matches chart scope

## GARCH Forecasting Implementation

### Core Components

#### 1. **garch_model.py** - Volatility Forecasting

```python
# Key functions:

def forecast_volatility(ticker, periods=14, days=1095):
    """
    Forecast volatility for next N periods using GARCH model
    
    Args:
        ticker (str): Stock ticker symbol
        periods (int): Number of periods to forecast (default: 14 days)
        days (int): Historical data window for model training (default: 1095 = 3 years)
    
    Returns:
        dict: {
            'status': 'success',
            'current_volatility': float,
            'forecasted_volatility': list[float],
            'forecast_periods': int,
            'model_info': {...}
        }
    """

def get_garch_stats(ticker, days=252):
    """
    Get GARCH model statistics for volatility analysis
    
    Returns:
        dict: {
            'current_volatility': float,
            'average_volatility': float,
            'max_volatility': float,
            'min_volatility': float,
            'returns_mean': float,
            'returns_std': float,
            'model_aic': float,
            'model_bic': float
        }
    """
```

**Key Implementation Details:**

- Uses 3 years (1095 days) of historical data for model fitting
- Fits GARCH(1,1) model to log returns
- Calculates daily volatility as percentage
- Returns forecasted volatility for next 14 trading days

**Error Handling:**

```python
# Gracefully handles missing arch library
if not ARCH_AVAILABLE:
    return {'status': 'unavailable', 'message': 'GARCH model requires: pip install arch'}

# Catches fitting errors
if model is None:
    return {'status': 'error', 'message': f'Could not fit GARCH model for {ticker}'}
```

#### 2. **draw.py** - Chart Generation with Forecasts

```python
def draw_chart(ticker, period_name, period_days, grouping='daily', include_forecast=True, forecast_days=14):
    """
    Generate professional stock chart with GARCH forecasts
    
    Args:
        forecast_days: Number of days to forecast (dynamic based on period)
            - 1D → 3 days
            - 5D → 5 days
            - 1M/3M/6M → 14 days
            - YTD/1Y/5Y/MAX → 21 days
    
    Key features:
    - Plots historical candlesticks with proper OHLC
    - Generates forecast candlesticks using GARCH volatility
    - Shows forecast volume bars
    - Extends x-axis to accommodate forecast periods
    - Returns PNG bytes for in-memory caching
    """
```

**Forecast Generation Logic:**

```python
# 1. Fetch current price and volatility
current_price = closes[-1]
volatility = forecast_result.get('current_volatility', 0) / 100

# 2. Generate dynamic forecast days using random walk
# forecast_days varies from 3 to 21 depending on selected period
for i in range(forecast_days):
    # Daily volatility component
    daily_vol = volatility * close_price
    
    # Random walk with volatility
    o = close_price + np.random.normal(0, daily_vol * 0.3)
    h = max(close_price, o) + abs(np.random.normal(0, daily_vol * 0.5))
    l = min(close_price, o) - abs(np.random.normal(0, daily_vol * 0.5))
    c = close_price + price_change
    
    # Plot with lighter colors for visual differentiation
    color_forecast = '#66ff99' if c >= o else '#ff7777'  # Light green/pink
    # Semi-transparent alpha=0.4 for forecast vs alpha=1.0 for historical

# 3. Extend x-axis
x_max = len(dates) + 14 + 1  # Historical + 14 forecast + margin
ax_chart.set_xlim(-1, x_max)
```

**Visual Differentiation:**

- Historical candlesticks: Dark green (#00d84f) and red (#ff3333)
- Forecast candlesticks: Light green (#66ff99) and light red (#ff7777)
- Forecast candlestick bodies: Semi-transparent (alpha=0.4)
- Forecast wicks: Semi-transparent (alpha=0.6)
- Vertical dashed line at separation point (optional feature)

**Forecast Volume:**

```python
# Generate forecast volume based on average historical volume
if forecast_data:
    avg_volume = np.mean(volumes)
    for idx in range(len(forecast_data['closes'])):
        forecast_vol = avg_volume * np.random.uniform(0.8, 1.2)
        # Plot with light colors, low opacity (alpha=0.3)
        ax_volume.bar(i, forecast_vol, color=color_forecast, alpha=0.3, width=0.8)
```

#### 3. **app.py** - Web Application Logic

```python
# In-memory caching
chart_cache = {}          # Stores PNG bytes
chart_data_cache = {}     # Stores OHLCV data for tooltips

def get_forecast_days(period):
    """Map period to forecast days.
    
    - 1D → 3 days
    - 5D → 5 days
    - 1M/3M/6M → 14 days
    - YTD/1Y/5Y/MAX → 21 days
    """

def ensure_chart_exists(ticker, period, grouping='daily'):
    """
    Generate and cache chart if not already cached
    
    Flow:
    1. Get period days (historical data window)
    2. Get forecast days (3-21 days based on period)
    3. Generate chart with dynamic GARCH forecasts
    4. Cache PNG bytes for serving
    5. Cache OHLCV data for tooltip display
    6. Grouping is always 'daily' (no longer exposed in UI)
    """
    
    period_days = get_period_days(period)
    forecast_days = get_forecast_days(period)
    chart_bytes = draw_chart(
        ticker, period, period_days, grouping,
        include_forecast=True, forecast_days=forecast_days
    )
```

**API Endpoints:**

- `GET /chart` - Display chart page
- `GET /api/chart` - Get chart metadata
- `GET /chart-image/<chart_key>` - Serve PNG from cache
- `GET /api/garch/<ticker>` - Get volatility forecast
- `GET /api/garch-stats/<ticker>` - Get GARCH statistics
- `GET /api/chart-data/<chart_key>` - Get OHLCV for tooltips

#### 4. **templates/chart_sidebar.html** - UI and Interactivity

**Sidebar Navigation:**

```html
<!-- Ticker Selector -->
📊 Tickers
  - AAPL, GOOGL, MSFT, AMZN, TSLA, META, NVDA, AMD, VGT

<!-- Time Period Selector -->
📅 Time Period
  - 1D, 5D, 1M, 3M, 6M, YTD, 1Y, 5Y, MAX

<!-- REMOVED: Grouping Selector -->
<!-- Previously had Daily/Weekly/Monthly options -->

<!-- GARCH Stats Panel -->
⚡ Volatility (GARCH)
  - Current Vol: X.XX%
  - Avg Vol: X.XX%
  - Max Vol: X.XX%
  - Returns Std: X.XX%
```

**Interactive Features:**

```javascript
// Chart hover tooltips
chartWrapper.addEventListener('mousemove', function(e) {
    // Calculate position in chart data array
    const relativeX = (x - chartStartX) / (chartEndX - chartStartX);
    const dataIndex = Math.round(relativeX * (chartData.length - 1));
    
    // Display OHLCV data for that candle
    // Color-code based on close < open (red) or close >= open (green)
});

// Resizable sidebar divider
// Drag to adjust sidebar width (200px min, 600px max)
```

**Removed Features:**

- Grouping selector (Daily/Weekly/Monthly) - simplified to always daily
- selectGrouping() JavaScript function - no longer needed
- Grouping parameter from URL buttons

## Database Schema

```sql
CREATE TABLE prices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    date DATE NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume INTEGER,
    UNIQUE(ticker, date)
);

CREATE INDEX idx_ticker_date ON prices (ticker, date);
```

**Location:** `/database/prices.db` (moved from root in earlier versions)

## Key Implementation Decisions

### 1. **GARCH Model Configuration**

- **Model Type:** GARCH(1,1) - simplest form, works well for financial time series
- **Historical Window:** 3 years (1095 days) - balances stability with recent trends
- **Forecast Horizon:** 14 trading days (2 weeks) - reasonable prediction window
- **Return Calculation:** Log returns (100 * log(P_t / P_{t-1})) - standard in finance

### 2. **Forecast Visualization**

**Why semi-transparent candlesticks?**
- Clear visual differentiation from historical data
- Indicates "predicted" nature without being distracting
- Matches professional financial chart conventions

**Why volume bars?**
- Completes the predictive picture
- Helps traders estimate liquidity
- Generated as variation around historical average

### 3. **In-Memory Caching**

```python
chart_cache = {
    'AAPL_6m_daily': b'PNG_BYTES_HERE',
    'MSFT_1y_daily': b'PNG_BYTES_HERE'
}

chart_data_cache = {
    'AAPL_6m_daily': [
        {'date': '2024-01-01', 'open': 100.5, ...},
        ...
    ]
}
```

**Advantages:**
- Fast serving (no disk I/O)
- Automatic cleanup on app restart
- Lower memory than persistent storage
- Charts regenerated fresh each session

### 4. **Removed Grouping Feature**

**Reason:** Simplified UI/UX

```python
# Before: grouping parameter in every URL
/chart?ticker=AAPL&period=6M&grouping=daily

# After: grouping always defaults to daily
/chart?ticker=AAPL&period=6M
```

## Performance Characteristics

### Chart Generation Time

- **Initial load:** 3-5 seconds (includes GARCH fitting)
- **Cached load:** < 100ms (image serve from memory)

### GARCH Model Fitting Time

- **3-year data window:** 1-2 seconds per ticker
- **Only fits on first request** for each ticker (cached in app memory)
- **Volatility calculation:** < 100ms using cached model

### Database Queries

- **Indexed by (ticker, date):** O(log n) lookup time
- **Typical query:** 500-1000 rows for 6M chart = < 50ms

## Troubleshooting Guide

### GARCH Stats Show "Loading..."

**Problem:** GARCH model not installed or failing to fit

**Solution:**
```bash
pip install arch
# Verify installation
python -c "from arch import arch_model; print('OK')"
```

### Chart Images Not Showing Forecast

**Problem:** Chart generated without forecast, or older cached version

**Solution:**
```bash
# Clear cache by restarting app
pkill -f "python app.py"
python app.py  # Regenerates with forecasts
```

### Port 5000 Already in Use (macOS)

**Problem:** AirPlay Receiver or other service using port 5000

**Solution:**
```bash
# Use port 8080 (or any other available port)
PORT=8080 python app.py

# Or disable AirPlay Receiver in System Settings > General > AirDrop & Handoff
```

### GARCH Model Fails for Specific Ticker

**Problem:** Not enough historical data or data quality issues

**Symptoms:**
```json
{
    "status": "error",
    "message": "'numpy.ndarray' object has no attribute 'values'"
}
```

**Solution:**
1. Verify ticker has 3+ years of data: `python refresh.py TICKER`
2. Check database: `sqlite3 database/prices.db "SELECT COUNT(*) FROM prices WHERE ticker='TICKER'"`
3. Ensure data quality (no missing dates, gaps)

## Testing the Implementation

### Manual Testing Checklist

```bash
# 1. Test GARCH model directly
python -c "from garch_model import forecast_volatility; print(forecast_volatility('AAPL'))"

# 2. Test chart generation
python draw.py AAPL --period 6M

# 3. Test API endpoints
curl http://localhost:8080/api/garch/AAPL?periods=3
curl http://localhost:8080/api/garch/AAPL?periods=14
curl http://localhost:8080/api/garch/AAPL?periods=21
curl http://localhost:8080/api/garch-stats/AAPL
curl http://localhost:8080/api/chart-data/AAPL_6m_daily

# 4. Test web interface with different periods
open http://localhost:8080/chart?ticker=AAPL&period=1D      # 3-day forecast
open http://localhost:8080/chart?ticker=AAPL&period=5D      # 5-day forecast
open http://localhost:8080/chart?ticker=AAPL&period=6M      # 14-day forecast
open http://localhost:8080/chart?ticker=AAPL&period=1Y      # 21-day forecast

# 5. Test multiple tickers
for ticker in AAPL MSFT GOOGL TSLA; do
    curl http://localhost:8080/api/garch/$ticker
done

# 6. Verify forecast scaling
# 1D should show ~3 forecast candlesticks
# 1Y should show ~21 forecast candlesticks
```

### Browser Testing

1. **Visual Check:**
   - Forecast candlesticks visible on right side of chart
   - Light colors differentiate from historical (dark colors)
   - Volume bars show prediction at bottom

2. **Interactivity Check:**
   - Hover over historical candlesticks = tooltip shows OHLCV
   - Hover over forecast candlesticks = no tooltip (not in data array)
   - Sidebar resize = chart adjusts properly
   - Period buttons = chart updates

3. **GARCH Stats Check:**
   - Stats panel loads below chart
   - Shows current, average, max volatility
   - Updates on ticker change

## Future Enhancement Ideas

1. **Bollinger Bands** around forecast (volatility envelope)
2. **Prediction Confidence** intervals (wider = less confident)
3. **Model Accuracy Metrics** (AIC/BIC comparison)
4. **Alternative Models** (ARCH, EMA, ARIMA)
5. **User Configuration** (forecast days, historical window)
6. **Export Forecasts** as CSV/JSON
7. **Backtesting** tool (compare historical forecast to actual)

## Dependencies and Versions

### Critical Dependencies

```
flask==2.3.0        # Web framework
matplotlib==3.7.0   # Chart generation (Agg backend)
yfinance==0.2.28    # Data fetching
pandas==1.5.3       # Data manipulation
numpy==1.24.0       # Numerical operations
arch==5.1.0         # GARCH modeling
```

### Installation Command

```bash
pip install flask matplotlib yfinance pandas numpy arch
```

## Files Modified / Created

### Created
- `garch_model.py` - GARCH forecasting module
- `templates/chart_sidebar.html` - New sidebar UI
- `database/` - Directory for database file

### Modified
- `app.py` - Added GARCH endpoints, cache management
- `draw.py` - Added forecast visualization
- `README.md` - Updated documentation
- `requirements.txt` - Added arch library

### Removed/Deprecated
- `templates/chart.html` - Replaced by chart_sidebar.html
- Grouping selector from UI (still in code for CLI compatibility)

## Memory and Performance Notes

### In-Memory Chart Cache

```python
# Typical sizes
chart_cache['AAPL_6m_daily'] = 88_000  # bytes (88 KB PNG)
chart_data_cache['AAPL_6m_daily'] = 30_000  # bytes (~30 tickers worth)

# Total for 9 tickers × 9 periods = 81 cached charts
# ~88 KB × 81 = ~7 MB total in memory (acceptable)
```

### GARCH Model Caching

```python
# Models are NOT cached - refitted on each request
# Reason: Low cost (1-2 sec per ticker)
# Benefit: Always uses latest volatility estimates
# Future: Could cache with TTL for performance
```

## Deployment Considerations

### Production Setup (NOT RECOMMENDED as-is)

Current implementation suitable for:
- Personal use
- Small team (< 10 users)
- Development/testing

For production, consider:
- Persistent chart cache (Redis/Memcached)
- Background GARCH model updates (Celery)
- Database connection pooling
- Load balancing (Gunicorn/nginx)
- HTTPS/SSL support
- API rate limiting

### Environment Variables

```bash
PORT=8080           # Listening port
FLASK_ENV=production  # Flask environment
DEBUG=False         # Disable debug mode
```

---

**Last Updated:** August 25, 2026  
**Implemented By:** Claude Code Assistant  
**Status:** Production Ready for Personal Use
