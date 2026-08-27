---
name: calibrate-model
description: Runs walk-forward backtests on volatility calibration factor and adjusts prediction settings
aliases:
  - calibrate model
  - calibrate the model
  - model calibration
  - recalibrate volatility
---

# Model Calibration Skill

## Purpose

Validates the GARCH volatility forecasting model's calibration factor and accuracy across multiple tickers using 36-window walk-forward backtests. Updates the model configuration based on backtest results.

## Trigger

User says phrases like:
- "calibrate the model"
- "recalibrate volatility"
- "run calibration"
- "backtest and calibrate"

## What It Does

1. **Runs backtests** for a set of test tickers (VDE, VOO, VTI — diversified funds; NVDA, AAPL, MSFT, TSLA — individual stocks)
   - Each: 36-window walk-forward, 2-year training window, 5-day forecast horizon
   - Tests both baseline (`vol_scale=1.0`) and calibrated (`vol_scale=0.8`) models
   - Reports volatility error % (mean forecasted vs realized)

2. **Computes calibration effectiveness**
   - If vol_scale=0.8 reduces error by >10 percentage points consistently, keeps it as default
   - If a better scale factor emerges, updates `DEFAULT_VOL_SCALE` in `garch_config.py`

3. **Updates configuration** (if needed):
   - Adjusts `DEFAULT_VOL_SCALE` in `garch_config.py`
   - Updates documentation in `CLAUDE.md` with new findings
   - Notes any GARCH order (p,q) or vol_model recommendations if discovered

4. **Generates report**
   - Summary table: ticker, baseline error, calibrated error, improvement
   - Direction accuracy stats (are weekly drift predictions accurate?)
   - Model fit quality (mean AIC/BIC per ticker)

## Technical Details

### Backtest Script

Uses `backtest_garch.py` with:
- `--ticker <name>` — test ticker
- `--windows 36` — 36 rolling windows
- `--train-years 2` — 2-year training window
- `--vol-scale 1.0` or `0.8` — test baseline vs calibrated
- `--csv <path>` — optional detailed CSV export

### Test Tickers

**Diversified Funds** (lower volatility, broad exposure):
- VDE (Vanguard Energy, sector-specific)
- VOO (Vanguard S&P 500, broad market)
- VTI (Vanguard Total Market, broadest)

**Individual Growth/Tech Stocks** (higher volatility, sector-concentrated):
- NVDA (Nvidia, semiconductors)
- AAPL (Apple, tech/consumer)
- MSFT (Microsoft, tech/software)
- TSLA (Tesla, automotive/tech)

### Key Metrics

**Price Forecast Error:**
- MAE (mean absolute error in $)
- MAPE (mean absolute percentage error %)
- RMSE (root mean square error)

**Direction Accuracy:**
- % of weeks where predicted drift direction matched actual price move
- Expected baseline: ~50% (coin flip) for historical mean drift

**Volatility Forecast Error:**
- Mean forecasted vol (daily %/day)
- Mean realized vol (daily %/day)
- Mean absolute error (percentage points)
- **Mean percent error (%)** — key metric for calibration
  - Baseline (vol_scale=1.0): typically 40-82% over-forecast
  - Calibrated (vol_scale=0.8): typically 30-62% over-forecast
  - Goal: >10 percentage point improvement

**Model Fit:**
- AIC (Akaike Information Criterion)
- BIC (Bayesian Information Criterion)
- Lower = better fit

## Backtest Results Summary (Aug 26, 2026)

| Ticker | Baseline Error | Calibrated Error | Improvement |
|---|---|---|---|
| VDE | 41.1% | 31.5% | -9.6 pts |
| VOO | 62.1% | 41.7% | -20.4 pts |
| VTI | 60.7% | 40.7% | -20.0 pts |
| NVDA | 82.5% | 59.4% | -23.1 pts |
| AAPL | 80.9% | 56.8% | -24.1 pts |
| MSFT | 75.4% | 59.1% | -16.3 pts |
| TSLA | 62.7% | 43.5% | -19.2 pts |

**Conclusion:** vol_scale=0.8 consistently reduces volatility forecast error by 9-24 percentage points across all 7 tested tickers (funds and individual stocks alike). Generalizes well. Remains the default.

**Direction accuracy note:** re-running this backtest also reconfirmed direction accuracy sits at ~44-56% across AAPL/MSFT/TSLA — essentially a coin flip, consistent with every previous run. No ticker showed a persistent directional edge from historical-mean drift.

## Implementation Notes

1. **Cache busting:** Every chart parameter change (including vol_scale) is included in the cache key, so no stale charts are served after recalibration.

2. **UI control:** Vol. Calibration slider in sidebar (0.3× to 1.5×) with default 0.8× visible to user — they can experiment with other scales if desired.

3. **Query string support:** vol_scale parameter in all chart URLs, API endpoints, and backtest CLI for reproducibility.

4. **Forward forecast only:** vol_scale only applies to the forward-looking `forecasted_volatility` array, not to historical/in-sample `current_volatility` (which reflects the raw model fit).

5. **Drift not calibrated:** Historical mean return (drift) has no predictive power at weekly horizon (~50% direction accuracy), so no calibration factor is applied to it. Future improvement: consider damping drift over longer forecast horizons.

## Running the Skill

```bash
# Run backtests for a single ticker
python backtest_garch.py --ticker VDE --windows 36 --train-years 2 --vol-scale 0.8

# Run for multiple tickers with CSV export
for ticker in VDE VOO VTI; do
  python backtest_garch.py --ticker $ticker --windows 36 --train-years 2 --vol-scale 1.0 --csv /tmp/${ticker}_baseline.csv
  python backtest_garch.py --ticker $ticker --windows 36 --train-years 2 --vol-scale 0.8 --csv /tmp/${ticker}_calibrated.csv
done

# Generate comprehensive report (see instructions in this file)
```

## Future Enhancements

1. **GJR-GARCH (vol='GARCH', o=1)** — more numerically stable than EGARCH for capturing leverage effect
2. **Drift damping** — taper historical mean return to zero over forecast horizon (since it's a coin flip anyway)
3. **Ticker-specific calibration** — if backtest shows individual tickers need different vol_scale values
4. **QLIKE loss metric** — more statistically robust volatility-forecast accuracy measure than mean % error
5. **Auto-rebalance on new data** — periodically re-run backtest as new data arrives to detect regime changes

## See Also

- `backtest_garch.py` — the walk-forward backtest implementation
- `garch_config.py` — centralized GARCH configuration, containing `DEFAULT_VOL_SCALE` and all other tunable defaults
- `garch_model.py` — the volatility forecasting model (imports its defaults from `garch_config.py`)
- `CLAUDE.md` — technical documentation and backtest findings
- `/chart?vol_scale=0.8` — example chart URL with calibration factor

