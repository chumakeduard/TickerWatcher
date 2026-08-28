#!/usr/bin/env python3
"""
GARCH Model Configuration

Centralized configuration for all GARCH volatility forecasting settings.
Modify these values to tune the model behavior across the entire application.
"""

# ============================================================================
# TRAINING DATA WINDOW
# ============================================================================

# Historical data window (in days) used to fit the GARCH model
# Longer window = more stable estimates but may miss recent regime changes
# Default: 1825 days = 5 years (recommended for production)
GARCH_TRAINING_DAYS = 1825

# Crypto training window (in days)
# Crypto now keeps full history (same retention policy as stocks — see
# refresh.py update_crypto()), so it uses the same 5-year default window.
# Crypto trades 7 days/week (no market closures), so this window actually
# contains more data points than the stock window over the same calendar span.
CRYPTO_GARCH_TRAINING_DAYS = 1825

# ============================================================================
# MODEL ORDER (p, q)
# ============================================================================

# Valid GARCH order combinations to expose in UI and CLI
# (1,1) is default and backtested as best by AIC/BIC across all tickers
VALID_GARCH_ORDERS = [(1, 1), (1, 2), (2, 1), (2, 2)]

# Default GARCH order
# Backtesting (Aug 26, 2026) confirmed (1,1) has lowest AIC/BIC
# Higher orders add complexity without improving fit or accuracy
DEFAULT_GARCH_P = 1
DEFAULT_GARCH_Q = 1

# ============================================================================
# VOLATILITY MODEL TYPE
# ============================================================================

# Valid volatility model types
VALID_VOL_MODELS = ['garch', 'egarch']

# Default volatility model
# GARCH: symmetric, numerically stable, recommended for production
# EGARCH: asymmetric (captures leverage effect), fragile on short windows, experimental
DEFAULT_VOL_MODEL = 'garch'

# ============================================================================
# VOLATILITY CALIBRATION FACTOR
# ============================================================================

# Default calibration multiplier for forecasted volatility
# Corrects for GARCH's systematic over-forecast bias (~20-25% at weekly horizon)
#
# Backtesting results (Aug 25-26, 2026, 36 windows per ticker):
#   - NVDA: 82.5% error → 59.4% (0.8×), improvement -23.1 pts
#   - AAPL: ~75% → ~52%, improvement ~-23 pts
#   - MSFT: ~65% → ~45%, improvement ~-20 pts
#   - TSLA: ~70% → ~48%, improvement ~-22 pts
#   - VDE: 41.1% → 31.5%, improvement -9.6 pts
#   - VOO: 62.1% → 41.7%, improvement -20.4 pts
#   - VTI: 60.7% → 40.7%, improvement -20.0 pts
#
# Conclusion: 0.8× reduces error 10-23 pct points across all asset types
# User-configurable range: 0.3–2.0 (slider in sidebar shows 0.3–1.5)
DEFAULT_VOL_SCALE = 0.8
MIN_VOL_SCALE = 0.3
MAX_VOL_SCALE = 2.0
UI_MAX_VOL_SCALE = 1.5  # Slider limit (can go higher via query string)

# Crypto volatility calibration (separate from stocks)
# Crypto typically has higher volatility and different dynamics
# Start with 0.9× (less aggressive correction than stocks)
DEFAULT_CRYPTO_VOL_SCALE = 0.9
MIN_CRYPTO_VOL_SCALE = 0.3
MAX_CRYPTO_VOL_SCALE = 2.0
UI_MAX_CRYPTO_VOL_SCALE = 1.5

# ============================================================================
# EGARCH ASYMMETRY ORDER
# ============================================================================

# Asymmetry order (o) for EGARCH models, unused for GARCH
# Default: 1 (captures first-order leverage effect)
DEFAULT_EGARCH_O = 1

# ============================================================================
# FORECAST HORIZON MAPPING
# ============================================================================

# Map selected chart period to default forecast day length
# Rationale:
#   - Short-term charts (1D, 5D): focused, near-term predictions
#   - Medium-term charts (1M-6M): consistent 2-week outlook
#   - Long-term charts (YTD+): extended 1-month strategic forecast
FORECAST_DAYS_BY_PERIOD = {
    '1D': 3,      # 1-day chart → 3-day forecast
    '5D': 5,      # 5-day chart → 5-day forecast
    '1M': 14,     # 1-month chart → 2-week forecast
    '3M': 14,     # 3-month chart → 2-week forecast
    '6M': 14,     # 6-month chart → 2-week forecast
    'YTD': 21,    # YTD chart → 1-month forecast
    '1Y': 21,     # 1-year chart → 1-month forecast
    '5Y': 21,     # 5-year chart → 1-month forecast
    'MAX': 21     # Max history → 1-month forecast
}

# Crypto forecast periods (shorter horizons due to shorter data window)
# Progressively longer forecasts for longer periods to show variation
CRYPTO_FORECAST_DAYS_BY_PERIOD = {
    '1D': 2,      # 1-day chart → 2-day forecast (crypto)
    '5D': 3,      # 5-day chart → 3-day forecast (crypto)
    '1M': 5,      # 1-month chart → 5-day forecast (crypto)
    '3M': 7,      # 3-month chart → 7-day forecast (crypto)
    '6M': 7,      # 6-month chart → 7-day forecast (crypto)
    'YTD': 10,    # YTD chart → 10-day forecast (crypto)
    '1Y': 10,     # 1-year chart → 10-day forecast (crypto)
    '5Y': 14,     # 5-year chart → 14-day forecast (crypto)
    'MAX': 14     # Max history → 14-day forecast (crypto)
}

# ============================================================================
# THRESHOLDS
# ============================================================================

# Default price-move threshold (%) for marking significant moves on chart
# Mark candlesticks where |price_change| >= threshold
DEFAULT_PRICE_MOVE_THRESHOLD = 10.0
MIN_PRICE_MOVE_THRESHOLD = 0.1
MAX_PRICE_MOVE_THRESHOLD = 100.0

# ============================================================================
# PROFIT TARGET MARKING
# ============================================================================

# Default profit target (%) for marking forecast candles that hit profit goal
# Marks the closest candle where price reaches: last_close * (1 ± profit_pct/100)
# Shows both upside (sell profit) and downside (short/buy profit) targets
DEFAULT_PROFIT_PCT = 10.0
MIN_PROFIT_PCT = 0.1
MAX_PROFIT_PCT = 100.0

# ============================================================================
# NOTES & FUTURE ENHANCEMENTS
# ============================================================================

"""
Direction Accuracy Notes:
- Across all tickers (NVDA, AAPL, MSFT, TSLA, VDE, VOO, VTI):
  Historical mean drift predicts direction at ~50% accuracy (coin flip)
- Suggests mean return has no real predictive power at weekly horizon
- Future enhancement: consider damping drift to zero over forecast horizons

Model Stability Notes:
- GARCH(1,1) is robust and numerically stable across all tickers
- EGARCH is more elegant (captures asymmetry/leverage) but fragile:
  - Fails to converge on short windows (< 1 year data)
  - Requires simulation-based multi-step forecasting (slower, less accurate)
  - Only marginally better AIC than GARCH on long windows
  - No accuracy improvement at weekly horizon
- Future consideration: GJR-GARCH (vol='Garch', o=1) as stable asymmetric alternative

Calibration Notes:
- vol_scale=0.8 generalizes across individual stocks and diversified funds
- Ratio of realized/forecasted vol ranges 0.74–0.87 across tickers
- Lower-vol assets (funds) benefit less (9.6 pts) than high-vol stocks (23 pts)
- Future enhancement: ticker-specific or regime-adaptive calibration

Accuracy Metrics Notes:
- Currently using mean % error: (|forecasted - realized| / realized) × 100
- Future enhancement: QLIKE loss (more statistically robust for vol forecasts)
"""
