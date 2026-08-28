# TickerWatcher Logging System

Comprehensive unified logging framework integrated into TickerWatcher. All logs are stored in separate files per calendar day in `./logs` directory.

## Log Files

**One file per calendar day:** `tickerwatcher-YYYY-MM-DD.log`

Each date gets its own log file:
- `tickerwatcher-2026-08-28.log` — Today's operations
- `tickerwatcher-2026-08-27.log` — Yesterday's operations
- `tickerwatcher-2026-08-26.log` — Day before (etc.)

Each file captures all operations:
- Application startup events
- Data refresh operations (refresh.py)
- Model calibration & backtests (backtest_garch.py)
- Web refresh operations (app.py)

**Example log entries:**
```
2026-08-28 14:13:12 | INFO     | AAPL: Already up to date (last: 2026-08-27)
2026-08-28 14:13:12 | INFO     | AAPL: Backfilling to meet 50-year requirement (from 1976-08-10 to 1980-12-12)
2026-08-28 14:11:43 | INFO     | MODEL CALIBRATION SWEEP: AAPL (STOCK)
2026-08-28 14:11:43 | INFO     | Testing 2 vol models × 4 orders = 8 configurations
2026-08-28 14:11:43 | INFO     | Stocks: +5 records across 18 tickers
2026-08-28 14:11:43 | INFO     | GARCH | Config: garch(1,1) | MAPE: 1.75% | Vol Err: 77.2% | BIC: 1945.3 | Direction Acc: 50.0% | Windows: 2
```

## Daily File Rotation

New log files are created automatically when the calendar date changes (at midnight):
- **Current date:** `tickerwatcher-YYYY-MM-DD.log` (actively written to)
- **Previous dates:** `tickerwatcher-YYYY-MM-DD.log` (archived by date)
- **Retention:** Keep all historical files; manually delete if storage becomes constrained

Browse historical logs by date:
```bash
ls logs/tickerwatcher-*.log              # View all log files
tail -100 logs/tickerwatcher-2026-08-28.log   # Today's log
cat logs/tickerwatcher-2026-08-27.log    # Yesterday's complete log
```

## Log Format

All log entries follow this consistent format:
```
TIMESTAMP | LEVEL | MESSAGE
```

- **TIMESTAMP**: `YYYY-MM-DD HH:MM:SS` (24-hour format)
- **LEVEL**: `INFO`, `WARNING`, `ERROR`, `DEBUG` (currently using INFO)
- **MESSAGE**: Descriptive text with operation data

## Usage

### Access Logs Directly
```bash
# View today's complete log (most recent)
tail -100 logs/tickerwatcher-2026-08-28.log

# Search for specific operations in today's log
grep "REFRESH SUMMARY" logs/tickerwatcher-2026-08-28.log
grep "RECOMMENDED CONFIGURATIONS" logs/tickerwatcher-2026-08-28.log

# View historical logs by date
cat logs/tickerwatcher-2026-08-27.log     # Yesterday
cat logs/tickerwatcher-2026-08-26.log     # Two days ago
tail -50 logs/tickerwatcher-2026-08-25.log

# List all available log files (by date)
ls -lh logs/tickerwatcher-*.log

# Monitor logs in real-time (today)
tail -f logs/tickerwatcher-2026-08-28.log
```

### Programmatic Access
```python
from logging_config import (
    get_data_refresh_logger,
    get_model_calibration_logger,
    get_app_logger
)

# All three functions return the same unified logger
refresh_logger = get_data_refresh_logger()
refresh_logger.info("Custom refresh message")

calibration_logger = get_model_calibration_logger()
calibration_logger.info("Custom calibration message")

app_logger = get_app_logger()
app_logger.info("Custom app message")
```

## What Gets Logged

### Application Startup (`app.py`)
- ✅ Port and startup configuration
- ✅ Number of stocks and crypto tickers loaded
- ✅ Initialization status

### Data Refresh (`refresh.py`)
- ✅ Refresh start (CLI or web-triggered)
- ✅ Individual ticker updates with record counts
- ✅ Summary statistics:
  - Total records added per asset type
  - Total tickers updated
  - Overall completion status

### Model Calibration (`backtest_garch.py`)
- ✅ Sweep initialization (ticker, asset type)
- ✅ Configuration matrix (vol models × GARCH orders)
- ✅ Recommended models per vol_model with metrics:
  - MAPE (mean absolute percent error)
  - Vol Error (volatility forecast error %)
  - BIC (model fit quality)
  - Direction Accuracy (drift prediction accuracy)
  - Windows evaluated
  - Vol Scale used

### Web Refresh (`app.py`)
- ✅ Refresh initiation from web interface
- ✅ Stock/crypto ticker updates (record counts)
- ✅ Cache clearing notification
- ✅ Success/error status

## Configuration

Logging is configured in `logging_config.py`:
- **Log directory:** `./logs` (created automatically if missing)
- **Log files:** One per calendar date: `tickerwatcher-YYYY-MM-DD.log`
- **Log level:** INFO (configurable in `setup_unified_logger()`)
- **Rotation:** Automatic at midnight (new file created for new date)
- **Retention:** Unlimited (keep all historical files; delete manually as needed)
- **Console output:** Enabled (logs appear in terminal + file)
- **Custom handler:** `DailyRotatingFileHandler` checks date on each log entry

## Troubleshooting

**Logs not appearing:**
1. Verify `logs/` directory exists: `ls -la logs/`
2. Check file permissions: `ls -l logs/tickerwatcher-*.log`
3. Ensure `logging_config.py` is in the project root
4. Verify imports: `from logging_config import get_*_logger()`

**Log files growing large (old dates):**
- Each file is limited to one calendar day's activity
- No automatic archival/deletion (manual cleanup needed)
- Safe to delete or compress old files: `rm logs/tickerwatcher-2026-07-*.log`
- Or archive them: `tar -czf logs-2026-07.tar.gz logs/tickerwatcher-2026-07-*.log`

**Logs showing wrong timestamps or mixed dates:**
- Logs use system clock for timestamps
- Set system time correctly for accurate dating
- Logs for a given date are in `tickerwatcher-YYYY-MM-DD.log`
- If date changes mid-operation, remaining logs go to next day's file

## Future Enhancements

- [ ] Log different severity levels (WARNING, ERROR, DEBUG with different formatting)
- [ ] Structured logging (JSON format for machine parsing)
- [ ] Remote logging to centralized server
- [ ] Performance metrics (execution time per operation)
- [ ] Log analysis dashboard with search/filter UI
- [ ] Automatic log compression for archived files

---

**Last Updated:** August 28, 2026  
**Log System:** Unified daily rotation to single file per date  
**Implemented in:** Latest TickerWatcher with full crypto support
