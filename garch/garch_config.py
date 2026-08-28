#!/usr/bin/env python3
"""
GARCH Model Configuration

Centralized configuration for all volatility forecasting settings.

Two kinds of values live here:

  * Structural settings (valid ranges, UI limits, forecast-horizon maps) — edit by hand.
  * CALIBRATED DEFAULTS — the block between the BEGIN/END markers below is written
    by `backtest_garch.py --calibrate --apply` (the `calibrate-model` skill).
    Hand edits inside that block will be overwritten on the next calibration run;
    change the search space in `backtest_garch.py` instead.

Stocks and crypto are calibrated independently — different model family, order,
error distribution, training window and vol_scale. Crypto trades 7 days a week and
has fatter tails, so there is no reason to expect a shared optimum.
"""

# ============================================================================
# MODEL SEARCH SPACE (what the UI/API will accept, and what calibration may pick)
# ============================================================================

# Volatility model families supported by garch_model.py, all backed by `arch`.
#   arch    — pure ARCH(p), no persistence term
#   garch   — symmetric GARCH(p,q); with o>0 this is GJR-GARCH (asymmetric)
#   egarch  — log-variance, asymmetric; needs simulation for multi-step forecasts
#   aparch  — asymmetric power ARCH, estimates the power exponent
#   figarch — fractionally integrated, long-memory volatility
#   harch   — heterogeneous ARCH over daily/weekly/monthly lag buckets
VALID_VOL_MODELS = ['arch', 'garch', 'egarch', 'aparch', 'figarch', 'harch']

# (p, q) order pairs exposed in the UI dropdown and accepted by the API.
VALID_GARCH_ORDERS = [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (1, 0), (2, 0), (3, 0)]

# Asymmetry order. o>0 on the `garch` family turns GARCH into GJR-GARCH.
VALID_ASYMMETRY_ORDERS = [0, 1]

# Error distribution for the innovations. Fat tails (t/skewt/ged) matter for crypto
# and high-beta single names; `normal` is the classic textbook default.
VALID_DISTRIBUTIONS = ['normal', 't', 'skewt', 'ged']

# Lag buckets used by the HARCH family (daily / weekly / monthly).
HARCH_LAGS = [1, 5, 22]

# Families with no closed-form multi-step variance forecast — these fall back to
# simulation-based forecasting, which is slower and slightly noisier.
SIMULATION_ONLY_MODELS = ['egarch', 'aparch']


# ============================================================================
# --- BEGIN CALIBRATED DEFAULTS (managed by the calibrate-model skill) -------
# ==========================================================================
# Written by: garch/garch_backtest.py --calibrate --apply
# Objective:  QLIKE (out-of-sample variance forecast loss) at each config's own
#             optimal vol_scale, averaged with equal weight across all tickers.
# Do not hand-edit — re-run the skill instead.

# ---- Stocks ----
DEFAULT_VOL_MODEL = 'garch'
DEFAULT_GARCH_P = 1
DEFAULT_GARCH_O = 0
DEFAULT_GARCH_Q = 1
DEFAULT_GARCH_DIST = 'normal'
DEFAULT_VOL_SCALE = 0.931
GARCH_TRAINING_DAYS = 1825

# ---- Crypto ----
DEFAULT_CRYPTO_VOL_MODEL = 'garch'
DEFAULT_CRYPTO_GARCH_P = 1
DEFAULT_CRYPTO_GARCH_O = 0
DEFAULT_CRYPTO_GARCH_Q = 1
DEFAULT_CRYPTO_GARCH_DIST = 'normal'
DEFAULT_CRYPTO_VOL_SCALE = 0.923
CRYPTO_GARCH_TRAINING_DAYS = 1825

# Provenance of the values above — updated on every calibration run.
CALIBRATION_PROVENANCE = {   'crypto': {   'bias_ratio': 1.08,
                  'calibrated_on': '2026-08-28',
                  'dm_p_vs_incumbent': None,
                  'free_parameters': 4,
                  'holdout_qlike': 2.7331,
                  'method': 'QLIKE walk-forward sweep; selection/holdout split with '
                            'Diebold-Mariano test vs incumbent',
                  'model': 'GARCH(1,1)-normal',
                  'selection_qlike': 3.1314,
                  'tickers': 2,
                  'vol_scale_range': [0.86, 0.986]},
    'stock': {   'bias_ratio': 0.91,
                 'calibrated_on': '2026-08-28',
                 'dm_p_vs_incumbent': None,
                 'free_parameters': 4,
                 'holdout_qlike': 2.1845,
                 'method': 'QLIKE walk-forward sweep; selection/holdout split with '
                           'Diebold-Mariano test vs incumbent',
                 'model': 'GARCH(1,1)-normal',
                 'selection_qlike': 1.7511,
                 'tickers': 18,
                 'vol_scale_range': [0.747, 1.159]}}
# --- END CALIBRATED DEFAULTS ------------------------------------------------
# ============================================================================


# ============================================================================
# VOLATILITY CALIBRATION RANGES
# ============================================================================

# vol_scale corrects the systematic over-forecast bias in GARCH-family models.
# It is applied ONLY to the forward-looking forecast, never to the in-sample
# conditional volatility (which should reflect the raw model fit).
MIN_VOL_SCALE = 0.3
MAX_VOL_SCALE = 2.0
UI_MAX_VOL_SCALE = 1.5  # Slider limit (higher still reachable via query string)

MIN_CRYPTO_VOL_SCALE = 0.3
MAX_CRYPTO_VOL_SCALE = 2.0
UI_MAX_CRYPTO_VOL_SCALE = 1.5

# Backwards-compatible alias: older code refers to the EGARCH asymmetry order by
# this name. Asymmetry is now a first-class dimension (DEFAULT_GARCH_O).
DEFAULT_EGARCH_O = 1


# ============================================================================
# FORECAST HORIZON MAPPING
# ============================================================================

# Map selected chart period to default forecast day length.
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

# Crypto uses deliberately shorter horizons — not a data limitation (crypto keeps
# full history now), but a forecast-confidence choice given higher volatility.
CRYPTO_FORECAST_DAYS_BY_PERIOD = {
    '1D': 2,
    '5D': 3,
    '1M': 5,
    '3M': 7,
    '6M': 7,
    'YTD': 10,
    '1Y': 10,
    '5Y': 14,
    'MAX': 14
}

# Forecast horizon used by the calibration backtest when scoring configs.
CALIBRATION_FORECAST_DAYS = 7


# ============================================================================
# THRESHOLDS
# ============================================================================

# Default price-move threshold (%) for marking significant moves on the chart.
DEFAULT_PRICE_MOVE_THRESHOLD = 10.0
MIN_PRICE_MOVE_THRESHOLD = 0.1
MAX_PRICE_MOVE_THRESHOLD = 100.0


# ============================================================================
# PROFIT TARGET MARKING
# ============================================================================

# Default profit target (%) for marking forecast candles that hit the goal.
DEFAULT_PROFIT_PCT = 10.0
MIN_PROFIT_PCT = 0.1
MAX_PROFIT_PCT = 100.0


# ============================================================================
# HELPERS
# ============================================================================

def get_model_defaults(is_crypto=False):
    """Return the calibrated model defaults for an asset class as a dict."""
    if is_crypto:
        return {
            'vol_model': DEFAULT_CRYPTO_VOL_MODEL,
            'p': DEFAULT_CRYPTO_GARCH_P,
            'o': DEFAULT_CRYPTO_GARCH_O,
            'q': DEFAULT_CRYPTO_GARCH_Q,
            'dist': DEFAULT_CRYPTO_GARCH_DIST,
            'vol_scale': DEFAULT_CRYPTO_VOL_SCALE,
            'training_days': CRYPTO_GARCH_TRAINING_DAYS,
        }
    return {
        'vol_model': DEFAULT_VOL_MODEL,
        'p': DEFAULT_GARCH_P,
        'o': DEFAULT_GARCH_O,
        'q': DEFAULT_GARCH_Q,
        'dist': DEFAULT_GARCH_DIST,
        'vol_scale': DEFAULT_VOL_SCALE,
        'training_days': GARCH_TRAINING_DAYS,
    }


# ============================================================================
# NOTES
# ============================================================================

"""
Direction Accuracy:
- Historical mean drift predicts direction at ~50% (coin flip) on every ticker
  tested, at every horizon tested. It carries no directional edge and is therefore
  not part of model selection. Damping drift toward zero over longer horizons
  remains an open idea.

Model Selection Objective:
- Configs are ranked by QLIKE, the standard robust loss for volatility forecasting.
  MAE/MAPE/RMSE cannot rank volatility models here: the forecast price path is
  driven by drift alone, so those metrics are identical for every config.
- BIC is reported as a secondary diagnostic only — it measures in-sample fit, not
  forecast accuracy.

vol_scale:
- Derived in closed form as c* = sqrt(mean(r2 / sigma2)), the value minimizing
  QLIKE. Each config is scored at its own optimum so the comparison is fair.
"""
