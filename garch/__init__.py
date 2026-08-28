"""
GARCH volatility forecasting package.

Modules:
    garch_config    — centralized configuration; the calibrated-defaults block is
                      written by the `calibrate-model` skill
    garch_model     — production forecaster used by the chart pipeline
    garch_backtest  — walk-forward backtest + full model-selection sweep

Common entry points are re-exported here so callers can use `from garch import ...`.
"""

from garch.garch_model import (
    forecast_volatility,
    get_garch_stats,
    fit_garch,
    extract_coefficients,
)
from garch.garch_config import get_model_defaults

__all__ = [
    'forecast_volatility',
    'get_garch_stats',
    'fit_garch',
    'extract_coefficients',
    'get_model_defaults',
]
