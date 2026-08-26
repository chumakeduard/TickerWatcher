# TickerWatcher - Quick Start Guide

## Start the Web App

```bash
# Option 1: Default port (5000)
python app.py

# Option 2: Custom port
PORT=8000 python app.py
```

Access at: **http://localhost:8000** (or your chosen port)

---

## Web Interface URLs

### Main Chart Page
```
http://localhost:8000/chart
http://localhost:8000/chart?ticker=AAPL
http://localhost:8000/chart?ticker=AAPL&period=6M
http://localhost:8000/chart?ticker=AAPL&period=6M&grouping=daily
```

### API Endpoint
```
http://localhost:8000/api/chart?ticker=AAPL&period=6M&grouping=daily
```

---

## URL Parameter Guide

### Ticker
Use any configured ticker from the system:
- AAPL, GOOGL, MSFT, AMZN, TSLA, META, NVDA, AMD, VGT

```
?ticker=AAPL
?ticker=TSLA
?ticker=MSFT
```

### Period
Time range for the chart:
- **1D** - 1 day
- **5D** - 5 days
- **1M** - 1 month
- **3M** - 3 months
- **6M** - 6 months (default)
- **YTD** - Year-to-date
- **1Y** - 1 year
- **5Y** - 5 years
- **MAX** - Maximum available

```
?period=1D
?period=6M
?period=1Y
?period=MAX
```

### Grouping
Chart data grouping method:
- **daily** - Daily candles (default)
- **weekly** - Weekly candles
- **monthly** - Monthly candles

```
?grouping=daily
?grouping=weekly
?grouping=monthly
```

---

## Example URLs for Automation

### Individual Tickers with Different Periods

```bash
# Apple - 6 months (default)
http://localhost:8000/chart?ticker=AAPL&period=6M

# Tesla - 1 year
http://localhost:8000/chart?ticker=TSLA&period=1Y

# Microsoft - 3 months
http://localhost:8000/chart?ticker=MSFT&period=3M

# Google - All available data
http://localhost:8000/chart?ticker=GOOGL&period=MAX

# Amazon - Daily view
http://localhost:8000/chart?ticker=AMZN&period=1D

# Meta - Weekly grouping
http://localhost:8000/chart?ticker=META&period=6M&grouping=weekly

# NVIDIA - Monthly grouping
http://localhost:8000/chart?ticker=NVDA&period=1Y&grouping=monthly
```

### API Queries

```bash
# Get chart info as JSON
curl http://localhost:8000/api/chart?ticker=AAPL&period=6M

# With custom period
curl http://localhost:8000/api/chart?ticker=TSLA&period=1Y&grouping=daily

# Response example:
{
    "ticker": "AAPL",
    "period": "6M",
    "grouping": "daily",
    "chart_file": "AAPL_6m.png",
    "chart_url": "/images/AAPL_6m.png"
}
```

---

## Web Interface Features

### Ticker Selector
- **Interactive buttons** for all configured tickers
- **Active indicator** shows current selection
- **One-click switching** between tickers

### Period Controls
- **Time period buttons**: 1D, 5D, 1M, 3M, 6M, YTD, 1Y, 5Y, MAX
- **Active button highlights** currently selected period
- **Direct URL updates** when changed

### Chart Grouping
- **Dropdown selector** for daily/weekly/monthly
- **Affects candle grouping** on the chart
- **Preserved in URL** for sharing

### Chart Display
- **Professional candlestick chart** (green up, red down)
- **Volume histogram** below price chart
- **OHLC information** (Open, High, Low, Close)
- **Current price** and daily change
- **Market status** indicator
- **Interactive tooltip** on hover
- **52-week statistics** panel
- **Direct URL display** for easy sharing/automation

---

## Automation Examples

### Shell Script - Generate Reports
```bash
#!/bin/bash
TICKERS=("AAPL" "TSLA" "MSFT" "GOOGL")
for ticker in "${TICKERS[@]}"; do
    echo "Fetching $ticker chart..."
    curl -s "http://localhost:8000/api/chart?ticker=$ticker&period=1Y" \
        | jq '.chart_url'
done
```

### Python - Download Charts
```python
import requests
from urllib.request import urlretrieve

tickers = ["AAPL", "TSLA", "MSFT"]
for ticker in tickers:
    response = requests.get(
        f"http://localhost:8000/api/chart",
        params={"ticker": ticker, "period": "6M"}
    )
    data = response.json()
    chart_url = f"http://localhost:8000{data['chart_url']}"
    urlretrieve(chart_url, f"{ticker}_chart.png")
    print(f"✓ Saved {ticker}_chart.png")
```

### cURL - Batch Download
```bash
# Download all ticker charts for different periods
for ticker in AAPL TSLA MSFT GOOGL; do
    for period in 1M 6M 1Y; do
        url="http://localhost:8000/images/${ticker,,}_${period,,}.png"
        wget -q "$url" -O "${ticker}_${period}.png" && echo "✓ $ticker $period"
    done
done
```

### Schedule Daily Updates
```bash
# Add to crontab (runs every day at 9 AM)
0 9 * * * cd /path/to/TickerWatcher && python refresh.py

# View existing crontab
crontab -l

# Edit crontab
crontab -e
```

---

## Web App Response Times

- **First chart load**: ~2-3 seconds (generates if needed)
- **Subsequent loads**: <100ms (cached)
- **API endpoint**: ~50ms (JSON response)
- **Image serving**: <50ms (direct file)

---

## Dark Theme Features

The web app includes:
- ✓ Dark charcoal background (#0a0e27)
- ✓ Subtle grid lines
- ✓ Professional typography
- ✓ Minimal borders
- ✓ Green highlights for active elements
- ✓ High contrast for important data
- ✓ Responsive mobile design

---

## Troubleshooting

### Port already in use
```bash
# Kill existing Flask process
pkill -f "python app.py"

# Or use different port
PORT=8001 python app.py
```

### Chart not loading
1. Ensure `refresh.py` has been run to fetch data
2. Check `/images` directory exists
3. Verify ticker is in config.py

### Database errors
```bash
# Reset database
rm prices.db

# Regenerate data
python refresh.py
```

### API returns empty response
- Check if Flask app is running: `ps aux | grep "python app.py"`
- Verify network connection
- Check logs: `tail -f /tmp/flask.log`

---

## Key Points for Automation

1. **All parameters are URL-based** - no login required
2. **Query strings are persistent** - bookmark or share directly
3. **API endpoint returns JSON** - easy to parse programmatically
4. **Charts auto-generate** - request any ticker/period combination
5. **Caching prevents bottlenecks** - repeated requests are instant

---

## Next Steps

1. **Run the web app**: `python app.py`
2. **Visit the homepage**: http://localhost:8000
3. **Use the controls** to switch tickers and periods
4. **Copy the URL** for any chart combination
5. **Use the URL** in your automation tools

Enjoy! 📊
