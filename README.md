# TickerWatcher

A comprehensive stock market data collection, visualization, and web interface system for analyzing ticker prices with professional-grade charts.

## Features

- 📊 **Professional Chart Generation**: Creates high-quality PNG images with candlestick charts, volume histograms, and financial metrics
- 🌐 **Web Interface**: Interactive Flask web app for browsing charts with button controls and query string parameters
- 💾 **Data Management**: SQLite database for storing historical price data with efficient querying
- 🔄 **Automated Data Refresh**: Fetch and update ticker data from Yahoo Finance
- ⚙️ **Flexible Configuration**: Easily switch between tickers and chart parameters
- 🤖 **Automation Ready**: Direct URL support for programmatic access and automation

## Project Structure

```
TickerWatcher/
├── config.py           # Hardcoded ticker list
├── db.py              # Database operations and data fetching
├── refresh.py         # Data refresh script
├── draw.py            # Chart image generation
├── app.py             # Flask web application
├── templates/
│   ├── chart.html     # Chart display page
│   └── error.html     # Error page
├── images/            # Generated chart images
├── prices.db          # SQLite database
└── requirements.txt   # Python dependencies
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

Generate a single chart image:

```bash
python draw.py AAPL --period 6M --grouping daily
```

Parameters:
- **ticker**: Stock symbol (e.g., AAPL, TSLA, GOOGL)
- **--period**: Time period (1D, 5D, 1M, 3M, 6M, YTD, 1Y, 5Y, MAX)
- **--grouping**: Chart type (daily, weekly, monthly)

Output: Generated PNG saved to `images/TICKER_PERIOD.png`

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
/chart?ticker=AAPL&period=6M&grouping=daily
```

### Example URLs

```
# AAPL 6-month chart
http://localhost:5000/chart?ticker=AAPL&period=6M&grouping=daily

# TSLA 1-year chart
http://localhost:5000/chart?ticker=TSLA&period=1Y&grouping=daily

# MSFT maximum history
http://localhost:5000/chart?ticker=MSFT&period=MAX&grouping=daily

# GOOGL 3-month with weekly grouping
http://localhost:5000/chart?ticker=GOOGL&period=3M&grouping=weekly
```

### API Endpoint

Get chart information as JSON:

```bash
curl http://localhost:5000/api/chart?ticker=AAPL&period=6M
```

Response:
```json
{
    "ticker": "AAPL",
    "period": "6M",
    "grouping": "daily",
    "chart_file": "AAPL_6m.png",
    "chart_url": "/images/AAPL_6m.png"
}
```

## Configuration

### Tickers

Edit `config.py` to change the list of tracked tickers:

```python
TICKERS = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA", "AMD", "VGT"]
```

### Data Retention

In `db.py`, modify `MIN_YEARS` to change historical data span:

```python
MIN_YEARS = 5  # Store 5 years of data
```

## Chart Features

Generated charts include:

- **Candlestick Chart**: OHLC with color-coded candles (green = up, red = down)
- **Volume Histogram**: Aligned with price candles
- **Header Information**: Current price, daily change, market status
- **Time Period Controls**: Visual buttons for different time ranges
- **Technical Indicators**: SMA 20/50/200 options shown
- **OHLC Display**: Open, High, Low, Close values
- **Crosshair & Tooltip**: Interactive position indicator with stats
- **52-Week Stats**: High/Low, YTD Return, Volume metrics
- **Professional Styling**: Dark theme with financial-grade aesthetics

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

See `requirements.txt` for versions.

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
2. Review Flask logs: `tail -f /tmp/flask.log`
3. Verify all dependencies are installed: `pip list | grep -E "(flask|matplotlib|yfinance)"`

---

**TickerWatcher** - Professional stock market analysis tools made simple.
