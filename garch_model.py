#!/usr/bin/env python3
"""GARCH model for volatility forecasting."""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sqlite3
from db import DB_PATH
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


def fit_garch(ticker, p=1, q=1, days=252):
    """Fit GARCH model to ticker returns."""
    if not ARCH_AVAILABLE:
        return None

    returns = get_ticker_returns(ticker, days)
    if returns is None or len(returns) < 50:
        return None

    try:
        model = arch_model(returns, vol='Garch', p=p, q=q)
        fitted = model.fit(disp='off')
        return fitted
    except Exception as e:
        print(f"Error fitting GARCH model: {e}")
        return None


def forecast_volatility(ticker, periods=5, p=1, q=1, days=252):
    """Forecast volatility for next N periods."""
    if not ARCH_AVAILABLE:
        return {
            'status': 'unavailable',
            'message': 'GARCH model requires: pip install arch'
        }

    model = fit_garch(ticker, p, q, days)
    if model is None:
        return {
            'status': 'error',
            'message': f'Could not fit GARCH model for {ticker}'
        }

    try:
        forecast = model.forecast(horizon=periods)
        variance_forecast = forecast.variance.values[-1, :]
        volatility_forecast = np.sqrt(variance_forecast)

        return {
            'status': 'success',
            'ticker': ticker,
            'current_volatility': float(np.sqrt(model.conditional_volatility.values[-1])),
            'forecasted_volatility': volatility_forecast.tolist(),
            'forecast_periods': periods,
            'model_info': {
                'p': p,
                'q': q,
                'aic': float(model.aic),
                'bic': float(model.bic)
            }
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }


def get_garch_stats(ticker, days=252):
    """Get GARCH model statistics."""
    if not ARCH_AVAILABLE:
        return None

    model = fit_garch(ticker, days=days)
    if model is None:
        return None

    returns = get_ticker_returns(ticker, days)

    return {
        'ticker': ticker,
        'current_volatility': float(np.sqrt(model.conditional_volatility.values[-1])),
        'average_volatility': float(np.sqrt(model.conditional_volatility.mean())),
        'max_volatility': float(np.sqrt(model.conditional_volatility.max())),
        'min_volatility': float(np.sqrt(model.conditional_volatility.min())),
        'returns_mean': float(np.mean(returns)),
        'returns_std': float(np.std(returns)),
        'model_aic': float(model.aic),
        'model_bic': float(model.bic)
    }
