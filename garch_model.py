#!/usr/bin/env python3
"""GARCH model for volatility forecasting."""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sqlite3
from db import DB_PATH
from garch_config import (
    GARCH_TRAINING_DAYS,
    VALID_GARCH_ORDERS,
    DEFAULT_GARCH_P,
    DEFAULT_GARCH_Q,
    VALID_VOL_MODELS,
    DEFAULT_VOL_MODEL,
    DEFAULT_VOL_SCALE,
    MIN_VOL_SCALE,
    MAX_VOL_SCALE,
    DEFAULT_EGARCH_O
)
import warnings
warnings.filterwarnings('ignore')

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False


def get_ticker_returns(ticker, days=252):
    """Fetch ticker data and calculate log returns."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days + 30)

    cursor.execute('''
        SELECT date, close FROM prices
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


def fit_garch(ticker, p=None, q=None, o=None, vol_model=None, days=None):
    """Fit a GARCH or EGARCH model to ticker returns.

    Args:
        p, q: model order (AR/MA terms of the variance equation)
              Defaults to config DEFAULT_GARCH_P, DEFAULT_GARCH_Q
        o: asymmetry order, only used when vol_model='egarch'
           Defaults to config DEFAULT_EGARCH_O
        vol_model: 'garch' (symmetric, default) or 'egarch' (asymmetric)
                   Defaults to config DEFAULT_VOL_MODEL
        days: training data window (in days)
              Defaults to config GARCH_TRAINING_DAYS
    """
    # Use config defaults if not specified
    if p is None:
        p = DEFAULT_GARCH_P
    if q is None:
        q = DEFAULT_GARCH_Q
    if o is None:
        o = DEFAULT_EGARCH_O
    if vol_model is None:
        vol_model = DEFAULT_VOL_MODEL
    if days is None:
        days = GARCH_TRAINING_DAYS

    if not ARCH_AVAILABLE:
        return None

    returns = get_ticker_returns(ticker, days)
    if returns is None or len(returns) < 50:
        return None

    try:
        if vol_model.lower() == 'egarch':
            model = arch_model(returns, vol='EGARCH', p=p, o=o, q=q)
        else:
            model = arch_model(returns, vol='Garch', p=p, q=q)
        fitted = model.fit(disp='off')
        return fitted
    except Exception as e:
        print(f"Error fitting GARCH model: {e}")
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


def forecast_volatility(ticker, periods=5, p=None, q=None, o=None, vol_model=None, days=None, vol_scale=None):
    """Forecast volatility for next N periods using GARCH or EGARCH.

    Args:
        periods: Number of days to forecast (default 5)
        p, q: GARCH model order (defaults to config DEFAULT_GARCH_P/Q)
        o: EGARCH asymmetry order (defaults to config DEFAULT_EGARCH_O)
        vol_model: 'garch' or 'egarch' (defaults to config DEFAULT_VOL_MODEL)
        days: Training window in days (defaults to config GARCH_TRAINING_DAYS)
        vol_scale: Calibration multiplier (defaults to config DEFAULT_VOL_SCALE)
                   Corrects for GARCH's systematic over-forecast bias (~20-25%)
    """
    # Use config defaults if not specified
    if p is None:
        p = DEFAULT_GARCH_P
    if q is None:
        q = DEFAULT_GARCH_Q
    if o is None:
        o = DEFAULT_EGARCH_O
    if vol_model is None:
        vol_model = DEFAULT_VOL_MODEL
    if days is None:
        days = GARCH_TRAINING_DAYS
    if vol_scale is None:
        vol_scale = DEFAULT_VOL_SCALE

    if not ARCH_AVAILABLE:
        return {
            'status': 'unavailable',
            'message': 'GARCH model requires: pip install arch'
        }

    if vol_model.lower() not in VALID_VOL_MODELS:
        vol_model = DEFAULT_VOL_MODEL
    if (p, q) not in VALID_GARCH_ORDERS:
        p, q = DEFAULT_GARCH_P, DEFAULT_GARCH_Q
    try:
        vol_scale = max(MIN_VOL_SCALE, min(MAX_VOL_SCALE, float(vol_scale)))
    except (ValueError, TypeError):
        vol_scale = DEFAULT_VOL_SCALE

    model = fit_garch(ticker, p, q, o, vol_model, days)
    if model is None:
        return {
            'status': 'error',
            'message': f'Could not fit GARCH model for {ticker}'
        }

    try:
        # EGARCH (and other asymmetric models) have no closed-form multi-step
        # forecast; they require simulation-based forecasting for horizon > 1.
        forecast_method = 'simulation' if vol_model.lower() == 'egarch' else 'analytic'
        forecast = model.forecast(horizon=periods, method=forecast_method, reindex=False)
        # Handle both DataFrame and ndarray returns
        if hasattr(forecast.variance, 'values'):
            variance_forecast = forecast.variance.values[-1, :]
        else:
            variance_forecast = forecast.variance[-1, :]

        volatility_forecast = np.sqrt(variance_forecast) * vol_scale

        # Get current volatility (NOT calibrated — this reflects the model's actual
        # in-sample fit, calibration only applies to the forward-looking forecast)
        current_vol = float(np.sqrt(model.conditional_volatility[-1]))

        # Historical mean return, used as forecast drift (in % per day, same units as returns)
        returns = get_ticker_returns(ticker, days)
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
                'o': o if vol_model.lower() == 'egarch' else None,
                'vol_model': vol_model.lower(),
                'vol_scale': vol_scale,
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


def get_garch_stats(ticker, p=None, q=None, o=None, vol_model=None, days=None):
    """Get GARCH model statistics, including fitted coefficients.

    Args:
        ticker: Stock ticker symbol
        p, q: GARCH model order (defaults to config DEFAULT_GARCH_P/Q)
        o: EGARCH asymmetry order (defaults to config DEFAULT_EGARCH_O)
        vol_model: 'garch' or 'egarch' (defaults to config DEFAULT_VOL_MODEL)
        days: Training window in days (defaults to config GARCH_TRAINING_DAYS)
    """
    # Use config defaults if not specified
    if p is None:
        p = DEFAULT_GARCH_P
    if q is None:
        q = DEFAULT_GARCH_Q
    if o is None:
        o = DEFAULT_EGARCH_O
    if vol_model is None:
        vol_model = DEFAULT_VOL_MODEL
    if days is None:
        days = GARCH_TRAINING_DAYS

    if not ARCH_AVAILABLE:
        return None

    if vol_model.lower() not in VALID_VOL_MODELS:
        vol_model = DEFAULT_VOL_MODEL
    if (p, q) not in VALID_GARCH_ORDERS:
        p, q = DEFAULT_GARCH_P, DEFAULT_GARCH_Q

    model = fit_garch(ticker, p, q, o, vol_model, days=days)
    if model is None:
        return None

    returns = get_ticker_returns(ticker, days)

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
        'o': o if vol_model.lower() == 'egarch' else None,
        'vol_model': vol_model.lower(),
        'coefficients': extract_coefficients(model)
    }
