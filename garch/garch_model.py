#!/usr/bin/env python3
"""GARCH model for volatility forecasting."""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
import sqlite3
from db import DB_PATH
from garch.garch_config import (
    GARCH_TRAINING_DAYS,
    CRYPTO_GARCH_TRAINING_DAYS,
    VALID_GARCH_ORDERS,
    VALID_ASYMMETRY_ORDERS,
    VALID_DISTRIBUTIONS,
    DEFAULT_GARCH_P,
    DEFAULT_GARCH_Q,
    VALID_VOL_MODELS,
    DEFAULT_VOL_MODEL,
    DEFAULT_VOL_SCALE,
    MIN_VOL_SCALE,
    MAX_VOL_SCALE,
    DEFAULT_CRYPTO_VOL_SCALE,
    MIN_CRYPTO_VOL_SCALE,
    MAX_CRYPTO_VOL_SCALE,
    DEFAULT_EGARCH_O,
    HARCH_LAGS,
    SIMULATION_ONLY_MODELS,
    get_model_defaults,
)
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False


def get_ticker_returns(ticker, days=252, is_crypto=False):
    """Fetch ticker data and calculate log returns."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days + 30)

    table = 'crypto_prices' if is_crypto else 'prices'
    cursor.execute(f'''
        SELECT date, close FROM {table}
        WHERE ticker = ? AND date BETWEEN ? AND ?
        ORDER BY date
    ''', (ticker, start_date, end_date))

    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return None

    prices = np.array([float(row[1]) for row in rows])
    returns = np.diff(np.log(prices)) * 100  # Log returns in percentage

    return returns


# Import valid GARCH orders and volatility models from config
# GARCH(1,1) is the default — backtesting confirmed it has the lowest AIC/BIC
# Volatility model types: GARCH (default, stable) vs EGARCH (experimental, fragile)
# See garch_config.py for full rationale and backtesting details


def build_arch_model(returns, p, o, q, vol_model, dist):
    """Construct an arch model for any supported volatility family.

    The families take different parameters — ARCH has no q, HARCH uses lag buckets
    instead of p/o/q — so the mapping is explicit rather than an if/else on two cases.
    Note `o > 0` on the 'garch' family is what makes it GJR-GARCH.
    """
    vm = (vol_model or DEFAULT_VOL_MODEL).lower()
    if vm == 'harch':
        return arch_model(returns, vol='HARCH', lags=list(HARCH_LAGS), dist=dist)
    if vm == 'arch':
        return arch_model(returns, vol='ARCH', p=p, dist=dist)
    family = {
        'garch': 'GARCH',
        'egarch': 'EGARCH',
        'aparch': 'APARCH',
        'figarch': 'FIGARCH',
    }.get(vm, 'GARCH')
    return arch_model(returns, vol=family, p=p, o=o, q=q, dist=dist)


def forecast_variance(fitted, vol_model, periods):
    """Multi-step variance forecast, using the right method for the family.

    EGARCH and APARCH have no closed-form multi-step forecast and must be
    simulated; the rest support the (faster, exact) analytic path. Falls back to
    simulation if analytic is refused, so a new family can't silently break.
    """
    vm = (vol_model or DEFAULT_VOL_MODEL).lower()
    use_sim = vm in SIMULATION_ONLY_MODELS
    try:
        method = 'simulation' if use_sim else 'analytic'
        fc = fitted.forecast(horizon=periods, method=method, reindex=False)
    except Exception:
        fc = fitted.forecast(horizon=periods, method='simulation', reindex=False)
    if hasattr(fc.variance, 'values'):
        return fc.variance.values[-1, :]
    return fc.variance[-1, :]


def _resolve_params(p, q, o, vol_model, days, dist, is_crypto):
    """Fill in any unspecified model parameter from the calibrated per-asset
    defaults, and coerce invalid values back to those defaults."""
    d = get_model_defaults(is_crypto=is_crypto)
    p = d['p'] if p is None else p
    q = d['q'] if q is None else q
    o = d['o'] if o is None else o
    vol_model = d['vol_model'] if vol_model is None else vol_model
    dist = d['dist'] if dist is None else dist
    days = d['training_days'] if days is None else days

    if str(vol_model).lower() not in VALID_VOL_MODELS:
        vol_model = d['vol_model']
    if str(dist).lower() not in VALID_DISTRIBUTIONS:
        dist = d['dist']
    if o not in VALID_ASYMMETRY_ORDERS:
        o = d['o']
    # ARCH and HARCH don't take a q term; (p, q) validity only applies to the rest.
    if str(vol_model).lower() not in ('arch', 'harch') and (p, q) not in VALID_GARCH_ORDERS:
        p, q = d['p'], d['q']
    return p, q, o, str(vol_model).lower(), days, str(dist).lower()


def fit_garch(ticker, p=None, q=None, o=None, vol_model=None, days=None,
              is_crypto=False, dist=None):
    """Fit a volatility model to ticker returns.

    Any argument left as None is taken from the calibrated defaults for the asset
    class (see garch_config.get_model_defaults), so stocks and crypto can run
    entirely different models.

    Args:
        p, q: model order of the variance equation
        o: asymmetry order (o>0 on the 'garch' family gives GJR-GARCH)
        vol_model: one of garch_config.VALID_VOL_MODELS
        days: training data window (in days)
        dist: error distribution, one of garch_config.VALID_DISTRIBUTIONS
        is_crypto: if True, read crypto data and crypto calibrated defaults
    """
    if not ARCH_AVAILABLE:
        return None

    p, q, o, vol_model, days, dist = _resolve_params(p, q, o, vol_model, days, dist, is_crypto)

    returns = get_ticker_returns(ticker, days, is_crypto=is_crypto)
    if returns is None or len(returns) < 50:
        return None

    try:
        model = build_arch_model(returns, p, o, q, vol_model, dist)
        return model.fit(disp='off', show_warning=False)
    except Exception as e:
        logger.warning(f"Error fitting {vol_model} model for {ticker}: {e}")
        return None


def extract_coefficients(fitted_model):
    """Extract the fitted model's coefficients into a plain dict for display.

    GARCH(1,1): omega, alpha[1], beta[1]
    EGARCH(1,1,1) adds: gamma[1] (asymmetry/leverage term)
    mu (mean equation constant) is included for both.
    """
    if fitted_model is None:
        return {}
    params = fitted_model.params
    coeffs = {}
    for name in params.index:
        # arch library names params like 'omega', 'alpha[1]', 'beta[1]', 'gamma[1]', 'mu'
        key = name.replace('[', '').replace(']', '')
        coeffs[key] = float(params[name])
    return coeffs


def forecast_volatility(ticker, periods=5, p=None, q=None, o=None, vol_model=None,
                        days=None, vol_scale=None, is_crypto=False, dist=None):
    """Forecast volatility for next N periods using GARCH or EGARCH.

    Args:
        periods: Number of days to forecast (default 5)
        p, q: GARCH model order (defaults to config DEFAULT_GARCH_P/Q)
        o: EGARCH asymmetry order (defaults to config DEFAULT_EGARCH_O)
        vol_model: 'garch' or 'egarch' (defaults to config DEFAULT_VOL_MODEL)
        days: Training window in days (defaults to config GARCH_TRAINING_DAYS or CRYPTO_GARCH_TRAINING_DAYS)
        vol_scale: Calibration multiplier (defaults to config DEFAULT_VOL_SCALE or DEFAULT_CRYPTO_VOL_SCALE)
                   Corrects for GARCH's systematic over-forecast bias (~20-25%)
        is_crypto: If True, use crypto data and crypto-specific defaults
    """
    if not ARCH_AVAILABLE:
        return {
            'status': 'unavailable',
            'message': 'GARCH model requires: pip install arch'
        }

    defaults = get_model_defaults(is_crypto=is_crypto)
    if vol_scale is None:
        vol_scale = defaults['vol_scale']
    p, q, o, vol_model, days, dist = _resolve_params(p, q, o, vol_model, days, dist, is_crypto)

    # Validate vol_scale against crypto or stock limits
    lo, hi = (MIN_CRYPTO_VOL_SCALE, MAX_CRYPTO_VOL_SCALE) if is_crypto else (MIN_VOL_SCALE, MAX_VOL_SCALE)
    try:
        vol_scale = max(lo, min(hi, float(vol_scale)))
    except (ValueError, TypeError):
        vol_scale = defaults['vol_scale']

    model = fit_garch(ticker, p, q, o, vol_model, days, is_crypto=is_crypto, dist=dist)
    if model is None:
        return {
            'status': 'error',
            'message': f'Could not fit {vol_model} model for {ticker}'
        }

    try:
        variance_forecast = forecast_variance(model, vol_model, periods)
        volatility_forecast = np.sqrt(variance_forecast) * vol_scale

        # Get current volatility (NOT calibrated — this reflects the model's actual
        # in-sample fit, calibration only applies to the forward-looking forecast)
        current_vol = float(np.sqrt(model.conditional_volatility[-1]))

        # Historical mean return, used as forecast drift (in % per day, same units as returns)
        returns = get_ticker_returns(ticker, days, is_crypto=is_crypto)
        returns_mean = float(np.mean(returns)) if returns is not None and len(returns) > 0 else 0.0

        return {
            'status': 'success',
            'ticker': ticker,
            'current_volatility': current_vol,
            'forecasted_volatility': volatility_forecast.tolist(),
            'forecast_periods': periods,
            'returns_mean': returns_mean,
            'vol_scale': vol_scale,
            'model_info': {
                'p': p,
                'q': q,
                'o': o,
                'dist': dist,
                'vol_model': vol_model,
                'vol_scale': vol_scale,
                'training_days': days,
                'aic': float(model.aic),
                'bic': float(model.bic),
                'coefficients': extract_coefficients(model)
            }
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }


def get_garch_stats(ticker, p=None, q=None, o=None, vol_model=None, days=None,
                    is_crypto=False, dist=None):
    """Get volatility model statistics, including fitted coefficients.

    Any argument left as None comes from the calibrated defaults for the asset
    class (see garch_config.get_model_defaults).

    Args:
        ticker: Stock or crypto ticker symbol
        p, q: model order of the variance equation
        o: asymmetry order (o>0 on the 'garch' family gives GJR-GARCH)
        vol_model: one of garch_config.VALID_VOL_MODELS
        days: training window in days
        dist: error distribution, one of garch_config.VALID_DISTRIBUTIONS
        is_crypto: if True, use crypto data and crypto calibrated defaults
    """
    if not ARCH_AVAILABLE:
        return None

    p, q, o, vol_model, days, dist = _resolve_params(p, q, o, vol_model, days, dist, is_crypto)

    model = fit_garch(ticker, p, q, o, vol_model, days=days, is_crypto=is_crypto, dist=dist)
    if model is None:
        return None

    returns = get_ticker_returns(ticker, days, is_crypto=is_crypto)

    return {
        'ticker': ticker,
        'current_volatility': float(np.sqrt(model.conditional_volatility[-1])),
        'average_volatility': float(np.sqrt(np.mean(model.conditional_volatility))),
        'max_volatility': float(np.sqrt(np.max(model.conditional_volatility))),
        'min_volatility': float(np.sqrt(np.min(model.conditional_volatility))),
        'returns_mean': float(np.mean(returns)),
        'returns_std': float(np.std(returns)),
        'model_aic': float(model.aic),
        'model_bic': float(model.bic),
        'p': p,
        'q': q,
        'o': o,
        'dist': dist,
        'vol_model': vol_model,
        'training_days': days,
        'coefficients': extract_coefficients(model)
    }
