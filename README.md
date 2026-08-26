# TickerWatcher

A comprehensive stock market data collection, visualization, and web interface system for analyzing ticker prices with professional-grade charts.

## Features

- 📊 **Professional Chart Generation**: Creates high-quality PNG images with candlestick charts, volume histograms, and financial metrics
- 🌐 **Web Interface**: Interactive Flask web app for browsing charts with button controls and query string parameters
- 💾 **Data Management**: SQLite database for storing historical price data with efficient querying
- 🔄 **Automated Data Refresh**: Fetch and update ticker data from Yahoo Finance
- ⚙️ **Flexible Configuration**: Easily switch between tickers and chart parameters
- 🤖 **Automation Ready**: Direct URL support for programmatic access and automation
- 🔮 **GARCH Forecasting**: Dynamic-length price predictions using the model's actual per-day volatility term structure plus historical drift, trained on 5-year historical data
- 📈 **Predictive Visualization**: Extended charts showing forecasted candlesticks and volume alongside historical data
- ⚡ **Volatility Analytics**: Real-time GARCH statistics (current, average, max volatility) + fitted model coefficients
- 📊 **Model Configurability**: Switch between GARCH orders (1,1)/(1,2)/(2,1)/(2,2) and volatility models (GARCH/EGARCH) from sidebar
- 🎚️ **Volatility Calibration**: Adjust the vol_scale factor (0.3–1.5×, default 0.8×) to correct for GARCH's systematic over-forecast bias
- 🎯 **Price-Move Thresholds**: Mark significant price movements (>10% default) with vertical lines and date labels
- 🔄 **One-Click Refresh**: Sidebar button fetches the latest data for all tickers and reloads the chart automatically

## Project Structure

```
TickerWatcher/
├── config.py                      # Hardcoded ticker list
├── db.py                         # Database operations and data fetching
├── refresh.py                    # Data refresh script
├── draw.py                       # Chart image generation with GARCH forecasts
├── garch_model.py               # GARCH volatility forecasting model
├── app.py                        # Flask web application
├── templates/
│   ├── chart_sidebar.html       # Main chart page with sidebar navigation
│   └── error.html               # Error page
├── database/
│   └── prices.db               # SQLite database
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── QUICK_START.md              # Quick start guide
```

## Installation

1. **Clone or navigate to the project directory**
   ```bash
   cd TickerWatcher
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On macOS/Linux
   # or
   .venv\Scripts\activate     # On Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Refresh Data

Update the database with the latest ticker prices:

```bash
python refresh.py
```

This will:
- Initialize the SQLite database if needed
- Fetch historical data for all configured tickers (default: 5 years)
- Store data with OHLCV (Open, High, Low, Close, Volume) values
- Handle incremental updates on subsequent runs

### 2. Generate Charts (CLI)

Generate a single chart image with forecasts:

```bash
python draw.py AAPL --period 6M
```

Parameters:
- **ticker**: Stock symbol (e.g., AAPL, TSLA, GOOGL)
- **--period**: Time period (1D, 5D, 1M, 3M, 6M, YTD, 1Y, 5Y, MAX) - default: 6M
- **--grouping**: Chart type (daily, weekly, monthly) - default: daily - *optional, for CLI only*

Output: Generated PNG with GARCH forecasts saved to memory (used by web app)

The chart includes:
- Historical candlesticks and volume
- 14-day forecasted candlesticks (lighter colors)
- Forecasted volume bars
- Vertical dashed line separating historical from predicted data

### 3. Web Application

Start the Flask web server:

```bash
# Default port 5000
python app.py

# Or use a custom port
PORT=8000 python app.py
```

Access the web interface:
- Home: `http://localhost:5000/`
- Chart page: `http://localhost:5000/chart`
- API endpoint: `http://localhost:5000/api/chart`

## Web Interface

### Features

- **Ticker Selector**: Switch between available tickers with button clicks
- **Period Controls**: Choose time ranges (1D to MAX)
- **Grouping Options**: Select chart granularity (daily, weekly, monthly)
- **Direct URLs**: All parameters support query strings for automation

### URL Structure

```
/chart?ticker=AAPL&period=6M
```

### Example URLs

```
# AAPL 6-month chart, default settings
http://localhost:8080/chart?ticker=AAPL&period=6M

# TSLA 1-year chart with custom GARCH order and vol_scale
http://localhost:8080/chart?ticker=TSLA&period=1Y&garch_p=1&garch_q=2&vol_scale=0.9

# MSFT with EGARCH model (experimental)
http://localhost:8080/chart?ticker=MSFT&period=MAX&vol_model=egarch

# GOOGL with 15% price-move threshold and 10-day forecast
http://localhost:8080/chart?ticker=GOOGL&period=3M&threshold=15&forecast_days=10
```

**Note:** Charts always display daily granularity. Grouping selector has been removed from the UI for simplicity.

### API Endpoints

#### Get Chart Information
```bash
curl http://localhost:5000/api/chart?ticker=AAPL&period=6M
```

Response:
```json
{
    "ticker": "AAPL",
    "period": "6M",
    "grouping": "daily",
    "chart_key": "AAPL_6m_daily",
    "chart_url": "/chart-image/AAPL_6m_daily"
}
```

#### Get GARCH Volatility Forecast
```bash
curl http://localhost:5000/api/garch/AAPL?periods=14
```

Response:
```json
{
    "status": "success",
    "ticker": "AAPL",
    "current_volatility": 1.29,
    "forecasted_volatility": [1.609, 1.611, 1.613, ...],
    "forecast_periods": 14,
    "model_info": {
        "p": 1,
        "q": 1,
        "aic": 12345.67,
        "bic": 12356.78
    }
}
```

#### Get GARCH Statistics
```bash
curl http://localhost:5000/api/garch-stats/AAPL
```

Response:
```json
{
    "ticker": "AAPL",
    "current_volatility": 1.29,
    "average_volatility": 1.18,
    "max_volatility": 2.45,
    "min_volatility": 0.89,
    "returns_mean": 0.08,
    "returns_std": 1.42,
    "model_aic": 12345.67,
    "model_bic": 12356.78
}
```

#### Get Chart Data for Tooltips
```bash
curl http://localhost:5000/api/chart-data/AAPL_6m_daily
```

Response:
```json
{
    "data": [
        {
            "date": "2024-01-01",
            "open": 100.50,
            "high": 101.20,
            "low": 99.80,
            "close": 101.00,
            "volume": 50000000
        }
    ]
}
```

#### Refresh Data (all tickers)

Also available as a "🔄 Refresh Data" button at the bottom of the sidebar in the web UI.

```bash
# Start a background refresh
curl -X POST http://localhost:5000/api/refresh
# {"status": "started"}   (or {"status": "already_running"} with HTTP 409)

# Poll until it finishes
curl http://localhost:5000/api/refresh/status
# {"running": false, "last_run": "2026-08-26T10:14:27", "last_result": "success", "error": null}
```

Runs the same incremental-fetch logic as `python refresh.py`, in a background thread. On success, clears the in-memory chart cache so the next chart view regenerates with the newly fetched data.

## Model Calibration & Tuning

### Volatility Calibration Factor

The GARCH model systematically over-forecasts realized volatility. A calibration factor (`vol_scale`, default **0.8×**) is applied to correct this bias.

**Tuning the Calibration Factor:**

1. **UI Slider** (recommended):
   - Sidebar panel: "⚙️ Chart Controls" → "Vol. Calibration"
   - Range: 0.3× to 1.5×
   - Default: 0.8× (pre-tuned via walk-forward backtesting)
   - Change the slider and click "Apply" to test different scales

2. **Query Parameter**:
   ```
   http://localhost:8080/chart?ticker=NVDA&vol_scale=0.75
   ```

3. **Backtest via CLI**:
   ```bash
   python backtest_garch.py --ticker NVDA --windows 36 --vol-scale 0.75
   ```

**When to Recalibrate:**
- New market regime or volatility regime shift
- Adding new asset classes (e.g., crypto) with different volatility patterns
- After major data outages/corrections

### GARCH Model Order (p, q)

Select from (1,1), (1,2), (2,1), (2,2) via sidebar dropdown or query parameter.

**Backtesting Results** (Aug 26, 2026):
- **(1,1)** — Best AIC/BIC across NVDA, AAPL, MSFT, TSLA ✅ **recommended**
- (1,2) — Slightly worse fit, more parameters
- (2,1) — Slightly worse fit, more parameters
- (2,2) — Worst fit, overfitting risk

Default is (1,1). Change via:
- **UI:** Sidebar dropdown "GARCH Order (p,q)"
- **Query:** `?garch_p=1&garch_q=2`

### Volatility Model Type (GARCH vs EGARCH)

- **GARCH** (default): Symmetric, numerically stable, recommended for production
- **EGARCH** (experimental): Asymmetric, captures leverage effect, but numerically fragile on short training windows

Change via:
- **UI:** Sidebar dropdown "Volatility Model"
- **Query:** `?vol_model=garch` or `?vol_model=egarch`

### Run Walk-Forward Backtests

Validate model settings on historical data using the `calibrate-model` skill:

```bash
# Run backtest for a single ticker
python backtest_garch.py --ticker NVDA --windows 36 --train-years 2 --vol-scale 0.8

# Run for multiple tickers and export results
for ticker in VDE VOO VTI; do
  python backtest_garch.py --ticker $ticker --windows 36 --vol-scale 0.8 --csv /tmp/${ticker}_results.csv
done
```

Metrics reported:
- **Price forecast:** MAE, MAPE, RMSE, direction accuracy (%)
- **Volatility forecast:** mean forecasted vol vs realized vol, % error
- **Model fit:** AIC, BIC

See `.claude/skills/calibrate-model/SKILL.md` for detailed backtest methodology and latest results.

## Configuration

### Tickers

Edit `config.py` to change the list of tracked tickers:

```python
TICKERS = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA", "AMD", "VGT"]
```

### Data Retention

In `db.py`, modify `MIN_YEARS` to change historical data span for GARCH fitting:

```python
MIN_YEARS = 5  # Train GARCH model on 5 years of data (default)
```

## Chart Features

Generated charts include:

- **Candlestick Chart**: OHLC with color-coded candles (green = up, red = down)
- **Volume Histogram**: Aligned with price candles
- **Header Information**: Current price, daily change, market status
- **Time Period Controls**: Visual buttons for different time ranges (1D to MAX)
- **Technical Indicators**: SMA 20/50/200 options shown
- **OHLC Display**: Open, High, Low, Close values
- **Crosshair & Tooltip**: Interactive position indicator with stats
- **52-Week Stats**: High/Low, YTD Return, Volume metrics
- **Professional Styling**: Dark theme with financial-grade aesthetics

### GARCH Forecasting Features

- **Dynamic Forecast Periods**: Forecast length adapts to selected time period
  - **1D** → 3-day forecast
  - **5D** → 5-day forecast  
  - **1M/3M/6M** → 14-day forecast
  - **YTD/1Y/5Y/MAX** → 21-day forecast (1 month)
- **Volatility-Based**: GARCH(1,1) model trained on 3-year historical data
- **Visual Differentiation**: 
  - Forecast candlesticks shown with lighter colors (semi-transparent)
  - Vertical dashed line separates historical from predicted data
  - Forecast volume bars displayed alongside price predictions
- **Dynamic Calculation**: Forecasts updated in real-time based on current market conditions
- **Term-Structure Aware**: Each forecasted day uses the model's own per-day volatility forecast (not a single flat number), so uncertainty genuinely evolves across the horizon
- **Drift-Aware**: Forecast includes the ticker's historical mean return as drift, not just zero-mean noise
- **Volatility Analytics**: GARCH stats panel shows current/average/max volatility

## Database

### Schema

```sql
CREATE TABLE prices (
    id INTEGER PRIMARY KEY,
    ticker TEXT NOT NULL,
    date DATE NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume INTEGER,
    UNIQUE(ticker, date)
);
```

### Queries

Access data programmatically:

```python
from db import init_db, get_last_price_date, fetch_and_store_prices
import sqlite3

conn = init_db()
cursor = conn.cursor()

# Get data for a ticker
cursor.execute('''
    SELECT date, open, high, low, close, volume
    FROM prices
    WHERE ticker = ? AND date BETWEEN ? AND ?
    ORDER BY date
''', ('AAPL', '2024-01-01', '2024-12-31'))

for row in cursor.fetchall():
    print(row)

conn.close()
```

## Automation Examples

### Schedule Daily Updates

Create a cron job to run every morning:

```bash
# Edit crontab
crontab -e

# Add this line (runs at 9 AM daily)
0 9 * * * cd /path/to/TickerWatcher && python refresh.py
```

### Generate Charts on Demand

Create charts programmatically:

```python
from draw import draw_chart

# Generate AAPL 6-month chart
draw_chart('AAPL', '6M', 180, 'daily')
```

### Web Integration

Embed charts in web pages:

```html
<!-- Display chart for a specific ticker -->
<img src="http://localhost:5000/images/AAPL_6m.png" alt="AAPL Chart">

<!-- Or use the API endpoint -->
<script>
  fetch('/api/chart?ticker=AAPL&period=6M')
    .then(r => r.json())
    .then(data => {
      document.querySelector('img').src = data.chart_url;
    });
</script>
```

### Screenshot API

Direct URL for screenshots or embedding:

```
# Return chart as PNG directly
GET /images/AAPL_6m.png

# Return metadata as JSON
GET /api/chart?ticker=AAPL&period=6M&grouping=daily
```

## Troubleshooting

### No data appears after running refresh.py

1. Check database connection: `ls -la prices.db`
2. Verify ticker symbols in `config.py`
3. Check network connection (yfinance requires internet)
4. Review logs: `python refresh.py 2>&1 | grep ERROR`

### Port already in use

Use a different port:
```bash
PORT=8001 python app.py
```

### Chart generation fails

1. Ensure matplotlib is installed: `pip install matplotlib`
2. Check disk space in `/images` directory
3. Verify ticker exists in database: Run `refresh.py` first

### Images not loading in web app

1. Verify Flask app is running: `curl http://localhost:5000/`
2. Check `images/` directory exists with PNG files
3. Clear browser cache and refresh

## Dependencies

- **yfinance**: Fetch stock data from Yahoo Finance
- **pandas**: Data manipulation and analysis
- **matplotlib**: Chart generation
- **numpy**: Numerical operations
- **flask**: Web framework
- **arch**: GARCH model for volatility forecasting

See `requirements.txt` for versions.

### Installing GARCH Support

The GARCH forecasting feature requires the `arch` library:

```bash
pip install arch
```

If not installed, the application will still work but GARCH forecasts will be unavailable.

## Performance Notes

- Chart generation typically takes 1-3 seconds per image
- Database queries are indexed for fast lookups
- Images are cached after generation (no regeneration on repeat requests)
- Web interface caches charts during session

## License

This project is provided as-is for educational and personal use.

## Support

For issues or questions:
1. Check the Troubleshooting section above
2. Check [CHANGELOG.md](CHANGELOG.md) — many past issues are already documented there with root cause and fix
3. Review Flask logs: `tail -f /tmp/flask.log`
4. Verify all dependencies are installed: `pip list | grep -E "(flask|matplotlib|yfinance)"`

---

**TickerWatcher** - Professional stock market analysis tools made simple.
