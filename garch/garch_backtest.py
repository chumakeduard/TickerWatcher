#!/usr/bin/env python3
"""
Walk-forward backtest and model calibration for volatility forecasting.

Two modes:

1. SINGLE MODEL (legacy) — backtest one (vol_model, p, q) config on one ticker and
   print price/vol/direction error metrics.

2. CALIBRATION SWEEP (--calibrate) — the real model-selection engine. For every
   ticker in an asset class, fits EVERY volatility model the `arch` library
   supports (ARCH / GARCH / GJR-GARCH / EGARCH / APARCH / FIGARCH / HARCH) across
   every order and every error distribution (normal / t / skewt / ged), scores
   them on out-of-sample forecast accuracy, and recommends the config + vol_scale
   that best matches realized volatility.

Walk-forward protocol (both modes):
  - anchor_date = last_available_date - i * step_weeks
  - train on `train_years` of data ending at anchor_date
  - forecast the following `forecast_days` trading days
  - compare against what actually happened in the database

Why the sweep scores what it does
---------------------------------
The forecast's PRICE path is driven by historical-mean drift, which does not
depend on the volatility model at all — so MAE/MAPE/RMSE/direction-accuracy are
identical for every config and cannot rank them. The volatility model only
affects the FORECAST VARIANCE, so that is what gets scored, using QLIKE:

    QLIKE = mean( ln(sigma2_forecast) + r2_actual / sigma2_forecast )

QLIKE is the standard robust loss for volatility forecasting (Patton 2011): it is
a proper scoring rule even though squared returns are a very noisy proxy for true
variance, which is exactly the regime here (7-day forecast windows). Lower = better.

Optimal vol_scale in closed form
--------------------------------
vol_scale c rescales sigma -> c*sigma, i.e. sigma2 -> c^2*sigma2. Substituting:

    QLIKE(c) = 2*ln(c) + mean(ln sigma2) + (1/c^2) * mean(r2 / sigma2)

    d/dc = 0  =>  c* = sqrt( mean(r2 / sigma2) )

So each config is scored at ITS OWN optimal calibration rather than at an
arbitrary fixed 0.8x. This makes the comparison fair AND produces the recommended
vol_scale as a by-product. At the optimum, QLIKE* = 2*ln(c*) + mean(ln sigma2) + 1.
"""

import sys
import json
import pprint
import sqlite3
import numpy as np
import pandas as pd
import warnings
import concurrent.futures
from scipy import stats as scipy_stats
from datetime import datetime, timedelta
from pathlib import Path

# This module lives in the `garch` package but depends on top-level modules (db,
# config, logging_config). Put the project root on sys.path so it works whether it
# is imported as `garch.garch_backtest` or run directly as
# `python3 garch/garch_backtest.py`. Also required for the ProcessPoolExecutor
# workers, which re-import __main__ under the macOS spawn start method.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from db import DB_PATH
from garch.garch_config import (
    VALID_GARCH_ORDERS,
    VALID_VOL_MODELS,
    DEFAULT_VOL_SCALE,
    DEFAULT_CRYPTO_VOL_SCALE,
    get_model_defaults,
)
from config import TICKERS, CRYPTO
from logging_config import get_model_calibration_logger

warnings.filterwarnings('ignore')

logger = get_model_calibration_logger()

try:
    from arch import arch_model
except ImportError:
    print("ERROR: requires `pip install arch`")
    sys.exit(1)


# ============================================================================
# MODEL SPACE
# ============================================================================

# Every volatility structure the arch library supports, verified to both fit and
# produce finite positive multi-step forecasts (arch 7.2.0).
#   vol   — arch `vol` family
#   p     — lag order of the symmetric innovation
#   o     — lag order of the asymmetric innovation (o>0 on GARCH == GJR-GARCH)
#   q     — lag order of lagged volatility
#   lags  — HARCH only (heterogeneous ARCH uses lag buckets, not p/o/q)
MODEL_STRUCTURES = [
    # Pure ARCH — no volatility persistence term
    {'vol': 'ARCH', 'p': 1, 'o': 0, 'q': 0},
    {'vol': 'ARCH', 'p': 2, 'o': 0, 'q': 0},
    {'vol': 'ARCH', 'p': 3, 'o': 0, 'q': 0},
    # Symmetric GARCH
    {'vol': 'GARCH', 'p': 1, 'o': 0, 'q': 1},
    {'vol': 'GARCH', 'p': 1, 'o': 0, 'q': 2},
    {'vol': 'GARCH', 'p': 2, 'o': 0, 'q': 1},
    {'vol': 'GARCH', 'p': 2, 'o': 0, 'q': 2},
    # GJR-GARCH — asymmetric via o, numerically stabler than EGARCH
    {'vol': 'GARCH', 'p': 1, 'o': 1, 'q': 1},
    {'vol': 'GARCH', 'p': 1, 'o': 1, 'q': 2},
    {'vol': 'GARCH', 'p': 2, 'o': 1, 'q': 1},
    {'vol': 'GARCH', 'p': 2, 'o': 1, 'q': 2},
    # EGARCH — log-variance, asymmetric
    {'vol': 'EGARCH', 'p': 1, 'o': 0, 'q': 1},
    {'vol': 'EGARCH', 'p': 1, 'o': 1, 'q': 1},
    {'vol': 'EGARCH', 'p': 1, 'o': 1, 'q': 2},
    {'vol': 'EGARCH', 'p': 2, 'o': 1, 'q': 1},
    # APARCH — asymmetric power ARCH, estimates the power exponent
    {'vol': 'APARCH', 'p': 1, 'o': 0, 'q': 1},
    {'vol': 'APARCH', 'p': 1, 'o': 1, 'q': 1},
    # FIGARCH — fractionally integrated, long-memory volatility
    {'vol': 'FIGARCH', 'p': 0, 'o': 0, 'q': 1},
    {'vol': 'FIGARCH', 'p': 1, 'o': 0, 'q': 1},
    # HARCH — heterogeneous ARCH over daily/weekly/monthly lag buckets
    {'vol': 'HARCH', 'p': 0, 'o': 0, 'q': 0, 'lags': (1, 5, 22)},
]

# Error distributions. Fat tails (t / skewt / ged) matter a lot for crypto and for
# high-beta single names; the previous sweep tested none of them.
DISTRIBUTIONS = ['normal', 't', 'skewt', 'ged']

# Families with no closed-form multi-step variance forecast.
SIMULATION_FAMILIES = {'EGARCH', 'APARCH'}
N_SIMULATIONS = 250          # keeps Monte-Carlo noise well below model differences
SIM_SEED = 20260828          # fixed so the ranking is reproducible


def config_complexity(cfg):
    """Free-parameter count, used as a parsimony tie-break.

    When two configs are statistically indistinguishable, the simpler one is the
    safer production default — it is less prone to the selection overfitting this
    whole sweep is trying to avoid.
    """
    n = 2  # mu (mean equation) + omega (variance constant)
    vol = cfg['vol']
    if vol == 'HARCH':
        n += len(cfg.get('lags', (1, 5, 22)))
    elif vol == 'ARCH':
        n += cfg['p']
    else:
        n += cfg['p'] + cfg['o'] + cfg['q']
    if vol == 'APARCH':
        n += 1   # estimated power exponent
    if vol == 'FIGARCH':
        n += 1   # fractional integration parameter d
    n += {'normal': 0, 't': 1, 'ged': 1, 'skewt': 2}[cfg['dist']]
    return n


def qlike_losses(var, r2, scale=1.0):
    """Per-observation QLIKE loss for a variance forecast at a given vol_scale.

    QLIKE = ln(sigma2) + r2/sigma2, with sigma2 scaled by scale**2.
    """
    s2 = var * (scale ** 2)
    return np.log(s2) + r2 / s2


def optimal_scale(var, r2):
    """Closed-form vol_scale minimizing QLIKE: c* = sqrt(mean(r2/sigma2))."""
    return float(np.sqrt(np.mean(r2 / var)))


def diebold_mariano(loss_a, loss_b, lag=None):
    """Diebold-Mariano test on two loss series (a - b), Newey-West corrected.

    Returns (stat, p_value, mean_diff). Negative stat means `a` has lower loss.

    The HAC correction matters here: consecutive forecast windows overlap, so the
    loss differentials are autocorrelated and a naive standard error would be too
    small — which would make noise look significant, exactly the error this test
    exists to prevent.
    """
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    d = d[np.isfinite(d)]
    n = d.size
    if n < 20:
        return np.nan, np.nan, float(np.mean(d)) if n else np.nan
    dbar = float(np.mean(d))
    dc = d - dbar
    if lag is None:
        lag = max(1, int(round(n ** (1.0 / 3.0))))
    gamma0 = float(np.dot(dc, dc) / n)
    var_hac = gamma0
    for k in range(1, min(lag, n - 1) + 1):
        gk = float(np.dot(dc[k:], dc[:-k]) / n)
        var_hac += 2.0 * (1.0 - k / (lag + 1.0)) * gk
    if var_hac <= 0:
        return np.nan, np.nan, dbar
    stat = dbar / np.sqrt(var_hac / n)
    p = 2.0 * (1.0 - scipy_stats.norm.cdf(abs(stat)))
    return float(stat), float(p), dbar


def config_label(cfg):
    """Compact human-readable name for a config."""
    vol = cfg['vol']
    if vol == 'HARCH':
        core = f"HARCH{list(cfg.get('lags', (1, 5, 22)))}"
    elif vol == 'ARCH':
        core = f"ARCH({cfg['p']})"
    elif cfg['o'] > 0:
        prefix = 'GJR-GARCH' if vol == 'GARCH' else vol
        core = f"{prefix}({cfg['p']},{cfg['o']},{cfg['q']})"
    else:
        core = f"{vol}({cfg['p']},{cfg['q']})"
    return f"{core}-{cfg['dist']}"


def build_config_grid(train_years_grid=(2,)):
    """Full cartesian grid: structure x distribution x training window."""
    grid = []
    for ty in train_years_grid:
        for struct in MODEL_STRUCTURES:
            for dist in DISTRIBUTIONS:
                cfg = dict(struct)
                cfg['dist'] = dist
                cfg['train_years'] = ty
                grid.append(cfg)
    return grid


def _make_model(returns, cfg):
    """Instantiate an arch model from a config dict."""
    vol = cfg['vol']
    dist = cfg['dist']
    if vol == 'HARCH':
        return arch_model(returns, vol='HARCH', lags=list(cfg.get('lags', (1, 5, 22))), dist=dist)
    if vol == 'ARCH':
        return arch_model(returns, vol='ARCH', p=cfg['p'], dist=dist)
    return arch_model(returns, vol=vol, p=cfg['p'], o=cfg['o'], q=cfg['q'], dist=dist)


def fit_and_forecast_variance(returns, cfg, horizon):
    """Fit `cfg` on `returns` (log returns in %) and return the h=1..horizon
    variance forecast, or None if the fit/forecast fails or is degenerate.

    Returns (variance_array, bic) on success.
    """
    try:
        model = _make_model(returns, cfg)
        fitted = model.fit(disp='off', show_warning=False)
    except Exception:
        return None

    use_sim = cfg['vol'] in SIMULATION_FAMILIES
    try:
        if use_sim:
            fc = fitted.forecast(horizon=horizon, method='simulation',
                                 simulations=N_SIMULATIONS,
                                 rng=np.random.default_rng(SIM_SEED).standard_normal,
                                 reindex=False)
        else:
            fc = fitted.forecast(horizon=horizon, method='analytic', reindex=False)
    except Exception:
        # Some families refuse analytic at horizon>1; fall back to simulation.
        try:
            fc = fitted.forecast(horizon=horizon, method='simulation',
                                 simulations=N_SIMULATIONS,
                                 rng=np.random.default_rng(SIM_SEED).standard_normal,
                                 reindex=False)
        except Exception:
            return None

    var = fc.variance.values[-1, :] if hasattr(fc.variance, 'values') else fc.variance[-1, :]
    var = np.asarray(var, dtype=float)
    if var.shape[0] != horizon or not np.all(np.isfinite(var)) or np.any(var <= 0):
        return None

    bic = float(fitted.bic)
    if not np.isfinite(bic):
        bic = np.nan
    return var, bic


# ============================================================================
# DATA ACCESS
# ============================================================================

def get_prices_in_range(ticker, start_date, end_date, is_crypto=False, conn=None):
    """Fetch (date, close) rows for ticker between start_date and end_date inclusive."""
    own_conn = conn is None
    if own_conn:
        conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    table = 'crypto_prices' if is_crypto else 'prices'
    cursor.execute(f'''
        SELECT date, close FROM {table}
        WHERE ticker = ? AND date BETWEEN ? AND ?
        ORDER BY date
    ''', (ticker, start_date.isoformat(), end_date.isoformat()))
    rows = cursor.fetchall()
    if own_conn:
        conn.close()
    return rows


def get_latest_date(ticker, is_crypto=False, conn=None):
    own_conn = conn is None
    if own_conn:
        conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    table = 'crypto_prices' if is_crypto else 'prices'
    cursor.execute(f'SELECT MAX(date) FROM {table} WHERE ticker = ?', (ticker,))
    row = cursor.fetchone()
    if own_conn:
        conn.close()
    if not row or not row[0]:
        return None
    return datetime.strptime(row[0], '%Y-%m-%d').date()


# ============================================================================
# LEGACY SINGLE-MODEL BACKTEST (unchanged behaviour, kept for the CLI)
# ============================================================================

def run_backtest(ticker='NVDA', n_windows=36, train_years=2, forecast_days=7,
                 p=1, q=1, vol_model='Garch', o=1, vol_scale=1.0, step_weeks=1,
                 verbose=True, is_crypto=False):
    """
    Rolling walk-forward backtest of a single model configuration.

    Window i: anchor_date = last_available_date - i*step_weeks weeks
      train:    [anchor_date - train_years, anchor_date]
      forecast: the forecast_days trading days AFTER anchor_date
    """
    latest_date = get_latest_date(ticker, is_crypto=is_crypto)
    if latest_date is None:
        raise ValueError(f"No data found for {ticker}")

    results = []
    conn = sqlite3.connect(DB_PATH)
    try:
        for i in range(1, n_windows + 1):
            anchor_date = latest_date - timedelta(weeks=i * step_weeks)
            train_start = anchor_date - timedelta(days=train_years * 365)

            train_rows = get_prices_in_range(ticker, train_start, anchor_date,
                                             is_crypto=is_crypto, conn=conn)
            if len(train_rows) < 100:
                if verbose:
                    print(f"[window {i}] Skipping anchor={anchor_date}: insufficient training data ({len(train_rows)} rows)")
                continue

            train_closes = np.array([float(r[1]) for r in train_rows])
            train_returns = np.diff(np.log(train_closes)) * 100

            cfg = {'vol': 'EGARCH' if vol_model.lower() == 'egarch' else 'GARCH',
                   'p': p, 'o': o if vol_model.lower() == 'egarch' else 0, 'q': q,
                   'dist': 'normal'}
            out = fit_and_forecast_variance(train_returns, cfg, forecast_days)
            if out is None:
                if verbose:
                    print(f"[window {i}] Fit/forecast failed at anchor={anchor_date}")
                continue
            variance_forecast, bic = out
            forecasted_vol = np.sqrt(variance_forecast) / 100 * vol_scale

            drift = float(np.mean(train_returns)) / 100
            last_train_close = train_closes[-1]

            actual_end = anchor_date + timedelta(days=forecast_days * 3)
            actual_rows = get_prices_in_range(ticker, anchor_date + timedelta(days=1),
                                              actual_end, is_crypto=is_crypto, conn=conn)
            actual_rows = actual_rows[:forecast_days]
            if len(actual_rows) < forecast_days:
                if verbose:
                    print(f"[window {i}] Skipping anchor={anchor_date}: insufficient actual data ({len(actual_rows)} rows)")
                continue

            actual_closes = np.array([float(r[1]) for r in actual_rows])
            actual_dates = [r[0] for r in actual_rows]

            expected_path = last_train_close * np.cumprod(np.full(forecast_days, 1 + drift))

            abs_errors = np.abs(expected_path - actual_closes)
            mae = float(np.mean(abs_errors))
            mape = float(np.mean(abs_errors / actual_closes * 100))
            rmse = float(np.sqrt(np.mean((expected_path - actual_closes) ** 2)))

            actual_week_return = (actual_closes[-1] - last_train_close) / last_train_close
            predicted_direction = 1 if drift > 0 else (-1 if drift < 0 else 0)
            actual_direction = 1 if actual_week_return > 0 else (-1 if actual_week_return < 0 else 0)
            direction_correct = predicted_direction == actual_direction

            actual_returns = np.diff(np.log(np.concatenate([[last_train_close], actual_closes])))
            realized_vol_daily = float(np.std(actual_returns, ddof=1))
            forecasted_vol_avg = float(np.mean(forecasted_vol))
            vol_error = abs(forecasted_vol_avg - realized_vol_daily)
            vol_pct_error = vol_error / realized_vol_daily * 100 if realized_vol_daily != 0 else np.nan

            results.append({
                'window': i,
                'anchor_date': anchor_date.isoformat(),
                'forecast_start': actual_dates[0],
                'forecast_end': actual_dates[-1],
                'last_train_close': last_train_close,
                'drift_daily_pct': drift * 100,
                'mae': mae, 'mape': mape, 'rmse': rmse,
                'direction_correct': direction_correct,
                'forecasted_vol_avg_pct': forecasted_vol_avg * 100,
                'realized_vol_daily_pct': realized_vol_daily * 100,
                'vol_error_pct_points': vol_error * 100,
                'vol_pct_error': vol_pct_error,
                'bic': bic,
            })

            if verbose:
                print(f"[window {i:2d}] anchor={anchor_date} | MAPE={mape:5.2f}% | "
                      f"dir_correct={str(direction_correct):5s} | "
                      f"fcst_vol={forecasted_vol_avg*100:.2f}% vs realized={realized_vol_daily*100:.2f}%")
    finally:
        conn.close()

    return pd.DataFrame(results)


def summarize(df, p=1, q=1, o=None, vol_model='Garch'):
    """Print aggregate statistics across all backtest windows."""
    if df.empty:
        print("No valid windows to summarize.")
        return

    o_str = f",{o}" if o else ""
    print("\n" + "=" * 70)
    print(f"BACKTEST SUMMARY — {vol_model}({p}{o_str},{q}) — {len(df)} windows")
    print("=" * 70)
    print("Price forecast (drift-only expected path vs actual):")
    print(f"  Mean MAE:   ${df['mae'].mean():.2f}")
    print(f"  Mean MAPE:  {df['mape'].mean():.2f}%")
    print(f"  Mean RMSE:  ${df['rmse'].mean():.2f}")
    print("\nDirection accuracy (did drift sign match the actual move?):")
    print(f"  {df['direction_correct'].sum()}/{len(df)} correct = {df['direction_correct'].mean()*100:.1f}%")
    print("\nVolatility forecast (forecasted vs realized):")
    print(f"  Mean forecasted vol: {df['forecasted_vol_avg_pct'].mean():.2f}%/day")
    print(f"  Mean realized vol:   {df['realized_vol_daily_pct'].mean():.2f}%/day")
    print(f"  Mean abs error:      {df['vol_error_pct_points'].mean():.2f} pct points")
    print(f"  Mean pct error:      {df['vol_pct_error'].mean():.1f}%")
    print(f"\nModel fit quality:\n  Mean BIC: {df['bic'].mean():.1f}")
    print("=" * 70)


# ============================================================================
# CALIBRATION ENGINE
# ============================================================================

def evaluate_ticker(task):
    """Worker: evaluate every config in the grid against one ticker.

    Runs the walk-forward loop ONCE per (window, train_years), fitting all configs
    that share that training window against the same cached return series — the old
    code re-queried SQLite and recomputed returns for every (config, window) pair.

    Returns {config_label: metrics_dict} plus per-ticker metadata.
    """
    ticker = task['ticker']
    is_crypto = task['is_crypto']
    n_windows = task['n_windows']
    forecast_days = task['forecast_days']
    step_weeks = task['step_weeks']
    grid = task['grid']

    latest_date = get_latest_date(ticker, is_crypto=is_crypto)
    if latest_date is None:
        return {'ticker': ticker, 'error': 'no data', 'configs': {}}

    # Group configs by training window so each window's data is fetched once.
    by_train_years = {}
    for cfg in grid:
        by_train_years.setdefault(cfg['train_years'], []).append(cfg)

    # accum[label] -> {'var': [...], 'r2': [...], 'bic': [...], 'fails': int}
    accum = {config_label(c) + f"|ty{c['train_years']}": {
        'cfg': c, 'var': [], 'r2': [], 'win': [], 'bic': [], 'fails': 0, 'windows': 0
    } for c in grid}

    conn = sqlite3.connect(DB_PATH)
    windows_used = 0
    try:
        for i in range(1, n_windows + 1):
            anchor_date = latest_date - timedelta(weeks=i * step_weeks)

            # Actual realized path for this window (shared by all configs)
            actual_end = anchor_date + timedelta(days=forecast_days * 4)
            actual_rows = get_prices_in_range(ticker, anchor_date + timedelta(days=1),
                                              actual_end, is_crypto=is_crypto, conn=conn)
            actual_rows = actual_rows[:forecast_days]
            if len(actual_rows) < forecast_days:
                continue

            for train_years, cfgs in by_train_years.items():
                train_start = anchor_date - timedelta(days=train_years * 365)
                train_rows = get_prices_in_range(ticker, train_start, anchor_date,
                                                 is_crypto=is_crypto, conn=conn)
                if len(train_rows) < 250:   # need a meaningful sample to fit on
                    continue

                train_closes = np.array([float(r[1]) for r in train_rows])
                train_returns = np.diff(np.log(train_closes)) * 100
                last_train_close = train_closes[-1]

                actual_closes = np.array([float(r[1]) for r in actual_rows])
                # Realized returns h=1..H, in %, aligned with the variance forecast
                actual_r = np.diff(np.log(np.concatenate([[last_train_close], actual_closes]))) * 100
                actual_r2 = actual_r ** 2

                for cfg in cfgs:
                    key = config_label(cfg) + f"|ty{cfg['train_years']}"
                    out = fit_and_forecast_variance(train_returns, cfg, forecast_days)
                    if out is None:
                        accum[key]['fails'] += 1
                        continue
                    var, bic = out
                    accum[key]['var'].append(var)
                    accum[key]['r2'].append(actual_r2)
                    # Window index is retained so the parent can split the losses
                    # chronologically into selection vs held-out validation sets.
                    accum[key]['win'].append(np.full(var.shape[0], i, dtype=int))
                    accum[key]['bic'].append(bic)
                    accum[key]['windows'] += 1

            windows_used += 1
    finally:
        conn.close()

    # Return RAW per-observation forecasts, not scores. Scoring needs the
    # chronological selection/validation split, which only the parent knows, and
    # the significance tests need the individual loss series rather than a mean.
    results = {}
    for key, a in accum.items():
        if not a['var']:
            continue
        var = np.concatenate(a['var'])
        r2 = np.concatenate(a['r2'])
        win = np.concatenate(a['win'])
        mask = np.isfinite(var) & np.isfinite(r2) & (var > 0)
        var, r2, win = var[mask], r2[mask], win[mask]
        if var.size < 20:
            continue
        results[key] = {
            'label': config_label(a['cfg']),
            'vol': a['cfg']['vol'], 'p': a['cfg']['p'], 'o': a['cfg']['o'],
            'q': a['cfg']['q'], 'dist': a['cfg']['dist'],
            'train_years': a['cfg']['train_years'],
            'lags': list(a['cfg'].get('lags', ())) or None,
            'complexity': config_complexity(a['cfg']),
            'var': var, 'r2': r2, 'win': win,
            'mean_bic': float(np.nanmean(a['bic'])) if a['bic'] else np.nan,
            'windows': a['windows'],
            'fails': a['fails'],
        }

    return {'ticker': ticker, 'windows_used': windows_used, 'configs': results}


def run_calibration(asset_type='stock', tickers=None, n_windows=72, forecast_days=7,
                    train_years_grid=(2,), step_weeks=1, max_workers=None, top_n=15,
                    validation_frac=0.25, alpha=0.05):
    """Full model-selection sweep across every ticker in an asset class.

    Selection happens on the older windows; the most recent `validation_frac` of
    windows are held out and used only to confirm the winner, with a
    Diebold-Mariano test against the incumbent production config. A change is
    recommended only if it is significant at `alpha` on that held-out data.

    Returns (ranked_configs, per_ticker_results, recommendation).
    """
    is_crypto = asset_type.lower() == 'crypto'
    if tickers is None:
        tickers = CRYPTO if is_crypto else TICKERS

    grid = build_config_grid(train_years_grid)

    header = (f"MODEL CALIBRATION — {asset_type.upper()} — {len(tickers)} tickers "
              f"x {len(grid)} configs x {n_windows} windows")
    logger.info("=" * 78)
    logger.info(header)
    logger.info("=" * 78)
    logger.info(f"Families: {sorted({s['vol'] for s in MODEL_STRUCTURES})}")
    logger.info(f"Distributions: {DISTRIBUTIONS} | Train windows (yrs): {list(train_years_grid)}")
    logger.info(f"Horizon: {forecast_days} trading days | Objective: QLIKE at optimal vol_scale")
    print(f"\n{'='*78}\n{header}\n{'='*78}")
    print(f"Fitting up to {len(tickers) * len(grid) * n_windows:,} models...\n")

    tasks = [{'ticker': t, 'is_crypto': is_crypto, 'n_windows': n_windows,
              'forecast_days': forecast_days, 'step_weeks': step_weeks, 'grid': grid}
             for t in tickers]

    per_ticker = {}
    t0 = datetime.now()
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as ex:
        for res in ex.map(evaluate_ticker, tasks):
            per_ticker[res['ticker']] = res
            n_ok = len(res.get('configs', {}))
            msg = f"  {res['ticker']:6s} — {n_ok:3d} configs scored over {res.get('windows_used', 0)} windows"
            print(msg)
            logger.info(msg)
    elapsed = (datetime.now() - t0).total_seconds()

    # ------------------------------------------------------------------
    # Chronological split. Window i=1 is the most recent, i=N the oldest, so the
    # RECENT windows form the held-out validation set and the older ones are used
    # for selection. Selecting and reporting on the same data would let a config
    # win by chance across 160 candidates; the winner must prove itself on data
    # that played no part in choosing it.
    # ------------------------------------------------------------------
    val_cut = max(1, int(round(n_windows * validation_frac)))
    logger.info(f"Split: validation = windows 1..{val_cut} (most recent), "
                f"selection = windows {val_cut + 1}..{n_windows}")
    print(f"\n  Selection windows: {val_cut + 1}..{n_windows}   "
          f"Held-out validation: 1..{val_cut}")

    def _score(m):
        """Score one config: fit vol_scale on selection, evaluate on validation.

        The scale is deliberately NOT refit on the validation set — doing so would
        leak held-out information into the very parameter being recommended.
        """
        sel = m['win'] > val_cut
        val = ~sel
        if sel.sum() < 30 or val.sum() < 20:
            return None
        c = optimal_scale(m['var'][sel], m['r2'][sel])
        sel_loss = qlike_losses(m['var'][sel], m['r2'][sel], c)
        val_loss = qlike_losses(m['var'][val], m['r2'][val], c)
        v, r = m['var'][val], m['r2'][val]
        return {
            'vol_scale': c,
            'sel_qlike': float(np.mean(sel_loss)),
            'val_qlike': float(np.mean(val_loss)),
            'val_loss': val_loss,
            'bias': float(np.sqrt(np.mean(v * c * c) / np.mean(r))) if np.mean(r) > 0 else np.nan,
        }

    # Aggregate across tickers, every ticker weighted equally.
    agg = {}
    for tk, res in per_ticker.items():
        for key, m in res.get('configs', {}).items():
            s = _score(m)
            if s is None:
                continue
            a = agg.setdefault(key, {'label': m['label'], 'meta': m, 'sel': [], 'val': [],
                                     'scale': [], 'bias': [], 'bic': [], 'losses': {},
                                     'tickers': 0, 'fails': 0})
            a['sel'].append(s['sel_qlike'])
            a['val'].append(s['val_qlike'])
            a['scale'].append(s['vol_scale'])
            a['bias'].append(s['bias'])
            a['bic'].append(m['mean_bic'])
            a['losses'][tk] = s['val_loss']
            a['tickers'] += 1
            a['fails'] += m['fails']

    # Only configs that scored on EVERY ticker are eligible. One that silently
    # fails to converge on part of the universe is not a usable production default.
    n_tick = len(tickers)
    pool = [a for a in agg.values() if a['tickers'] == n_tick] or list(agg.values())

    ranked = []
    for a in pool:
        m = a['meta']
        ranked.append({
            'label': a['label'], 'vol': m['vol'], 'p': m['p'], 'o': m['o'], 'q': m['q'],
            'dist': m['dist'], 'train_years': m['train_years'], 'lags': m['lags'],
            'complexity': m['complexity'],
            'sel_qlike': float(np.mean(a['sel'])),
            'val_qlike': float(np.mean(a['val'])),
            'median_vol_scale': float(np.median(a['scale'])),
            'mean_vol_scale': float(np.mean(a['scale'])),
            'vol_scale_min': float(np.min(a['scale'])),
            'vol_scale_max': float(np.max(a['scale'])),
            'mean_bias_ratio': float(np.nanmean(a['bias'])),
            'mean_bic': float(np.nanmean(a['bic'])),
            'tickers': a['tickers'], 'fails': a['fails'],
            '_losses': a['losses'],
        })
    # Rank on SELECTION only — validation must stay untouched by the choice.
    ranked.sort(key=lambda r: r['sel_qlike'])

    # Incumbent: whatever is configured in production right now, so the test asks
    # the question that actually matters — "is this better than what we ship?"
    inc = get_model_defaults(is_crypto=is_crypto)
    incumbent = next(
        (r for r in ranked
         if r['vol'].lower() == inc['vol_model'] and r['p'] == inc['p']
         and r['o'] == inc['o'] and r['q'] == inc['q'] and r['dist'] == inc['dist']
         and r['train_years'] * 365 == inc['training_days']),
        None)
    if incumbent is None:   # current default outside the sweep grid
        incumbent = next((r for r in ranked if r['vol'] == 'GARCH' and r['p'] == 1
                          and r['o'] == 0 and r['q'] == 1 and r['dist'] == 'normal'), None)

    def _pooled(r):
        return np.concatenate([r['_losses'][t] for t in sorted(r['_losses'])])

    # Significance-test the top candidates against the incumbent on VALIDATION data.
    for r in ranked[:top_n]:
        if incumbent is None or r is incumbent:
            r['dm_stat'], r['dm_p'] = np.nan, np.nan
            continue
        r['dm_stat'], r['dm_p'], _ = diebold_mariano(
            _pooled(r), _pooled(incumbent), lag=forecast_days)

    # Report
    print(f"\n{'='*96}\nTOP {top_n} CONFIGS — {asset_type.upper()} "
          f"(ranked by SELECTION QLIKE; validation is held out)\n{'='*96}")
    hdr = (f"{'#':>3} {'config':<26} {'ty':>3} {'selQLIKE':>9} {'valQLIKE':>9} "
           f"{'vol_scale':>10} {'bias':>6} {'DM p':>7} {'np':>3}")
    print(hdr)
    print("-" * 96)
    logger.info("Top configs (rank|config|ty|selQLIKE|valQLIKE|vol_scale|bias|DM_p|params):")
    for i, r in enumerate(ranked[:top_n], 1):
        dmp = r.get('dm_p', np.nan)
        dmp_s = '  incumb' if incumbent is not None and r is incumbent else (
            f"{dmp:>7.4f}" if np.isfinite(dmp) else "      -")
        line = (f"{i:>3} {r['label']:<26} {r['train_years']:>3} {r['sel_qlike']:>9.4f} "
                f"{r['val_qlike']:>9.4f} {r['median_vol_scale']:>10.3f} "
                f"{r['mean_bias_ratio']:>6.2f} {dmp_s} {r['complexity']:>3}")
        print(line)
        logger.info("  " + line.strip())

    # ------------------------------------------------------------------
    # Decision rule. A change is only recommended when the candidate
    #   (a) beats the incumbent on held-out validation data, AND
    #   (b) does so significantly (Diebold-Mariano p < alpha).
    # Among survivors that are statistically tied with the best, the simplest
    # model wins — parsimony is the tie-break, not a coin flip.
    # ------------------------------------------------------------------
    recommendation, reason = None, ''
    if incumbent is None:
        candidates = [r for r in ranked[:top_n]]
        recommendation = candidates[0] if candidates else None
        reason = 'no incumbent found in grid; taking best by selection QLIKE'
    else:
        survivors = [r for r in ranked[:top_n]
                     if r is not incumbent
                     and r['val_qlike'] < incumbent['val_qlike']
                     and np.isfinite(r.get('dm_p', np.nan))
                     and r['dm_p'] < alpha]
        if survivors:
            best_val = min(s['val_qlike'] for s in survivors)
            # Statistically tied with the best survivor -> prefer fewer parameters.
            tied = []
            best_s = min(survivors, key=lambda s: s['val_qlike'])
            for s in survivors:
                if s is best_s:
                    tied.append(s)
                    continue
                _, p_tie, _ = diebold_mariano(_pooled(s), _pooled(best_s), lag=forecast_days)
                if not np.isfinite(p_tie) or p_tie >= alpha:
                    tied.append(s)
            recommendation = min(tied, key=lambda s: (s['complexity'], s['val_qlike']))
            reason = (f'beats incumbent on held-out data (DM p={recommendation["dm_p"]:.4f}); '
                      f'simplest of {len(tied)} statistically tied candidate(s)')
        else:
            recommendation = incumbent
            reason = ('no candidate beat the incumbent significantly on held-out data '
                      f'(alpha={alpha}) — keeping current production config')

    print(f"\n{'='*96}\nRECOMMENDATION — {asset_type.upper()}\n{'='*96}")
    if incumbent is not None:
        print(f"  Incumbent : {incumbent['label']} (ty{incumbent['train_years']})  "
              f"selQLIKE {incumbent['sel_qlike']:.4f}  valQLIKE {incumbent['val_qlike']:.4f}  "
              f"vol_scale {incumbent['median_vol_scale']:.3f}")
    if recommendation is not None:
        changed = incumbent is None or recommendation is not incumbent
        print(f"  {'CHANGE TO' if changed else 'KEEP      '}: {recommendation['label']} "
              f"(ty{recommendation['train_years']})  "
              f"selQLIKE {recommendation['sel_qlike']:.4f}  "
              f"valQLIKE {recommendation['val_qlike']:.4f}  "
              f"vol_scale {recommendation['median_vol_scale']:.3f} "
              f"(range {recommendation['vol_scale_min']:.2f}-{recommendation['vol_scale_max']:.2f})")
        print(f"  Reason    : {reason}")
        if changed and incumbent is not None:
            print(f"  Held-out QLIKE improvement: "
                  f"{incumbent['val_qlike'] - recommendation['val_qlike']:.4f}")
        logger.info(f"RECOMMENDED {asset_type.upper()}: {recommendation['label']} "
                    f"ty={recommendation['train_years']} "
                    f"vol_scale={recommendation['median_vol_scale']:.3f} "
                    f"selQLIKE={recommendation['sel_qlike']:.4f} "
                    f"valQLIKE={recommendation['val_qlike']:.4f} | {reason}")
    print(f"\n  Elapsed: {elapsed:.0f}s")
    print("=" * 96)

    # Drop the heavy loss arrays before returning/serialising.
    for r in ranked:
        r.pop('_losses', None)
    return ranked, per_ticker, recommendation


CONFIG_PATH = Path(__file__).resolve().parent / 'garch_config.py'
BEGIN_MARKER = '# --- BEGIN CALIBRATED DEFAULTS'
END_MARKER = '# --- END CALIBRATED DEFAULTS'


def _render_calibrated_block(values, provenance):
    """Render the managed block of garch_config.py from calibrated values."""
    s, c = values['stock'], values['crypto']
    lines = [
        BEGIN_MARKER + ' (managed by the calibrate-model skill) -------',
        '# ' + '=' * 74,
        '# Written by: garch/garch_backtest.py --calibrate --apply',
        '# Objective:  QLIKE (out-of-sample variance forecast loss) at each config\'s own',
        '#             optimal vol_scale, averaged with equal weight across all tickers.',
        '# Do not hand-edit — re-run the skill instead.',
        '',
        '# ---- Stocks ----',
        f"DEFAULT_VOL_MODEL = {s['vol_model']!r}",
        f"DEFAULT_GARCH_P = {s['p']}",
        f"DEFAULT_GARCH_O = {s['o']}",
        f"DEFAULT_GARCH_Q = {s['q']}",
        f"DEFAULT_GARCH_DIST = {s['dist']!r}",
        f"DEFAULT_VOL_SCALE = {s['vol_scale']}",
        f"GARCH_TRAINING_DAYS = {s['training_days']}",
        '',
        '# ---- Crypto ----',
        f"DEFAULT_CRYPTO_VOL_MODEL = {c['vol_model']!r}",
        f"DEFAULT_CRYPTO_GARCH_P = {c['p']}",
        f"DEFAULT_CRYPTO_GARCH_O = {c['o']}",
        f"DEFAULT_CRYPTO_GARCH_Q = {c['q']}",
        f"DEFAULT_CRYPTO_GARCH_DIST = {c['dist']!r}",
        f"DEFAULT_CRYPTO_VOL_SCALE = {c['vol_scale']}",
        f"CRYPTO_GARCH_TRAINING_DAYS = {c['training_days']}",
        '',
        '# Provenance of the values above — updated on every calibration run.',
        # pformat, NOT json.dumps: this is embedded in a .py file, and JSON writes
        # null/true/false, which are not Python literals. A None in provenance would
        # produce a config that no longer imports.
        'CALIBRATION_PROVENANCE = ' + pprint.pformat(provenance, indent=4, width=88),
        END_MARKER + ' ------------------------------------------------',
    ]
    return '\n'.join(lines)


def apply_recommendations(recommendations):
    """Write recommended configs into the managed block of garch_config.py.

    `recommendations` maps asset_type -> the top-ranked config dict. An asset class
    absent from it keeps whatever is currently configured, so calibrating only
    crypto does not clobber the stock settings.
    """
    import importlib
    import garch.garch_config as gc
    importlib.reload(gc)

    # Start from what is configured today, then overlay whatever was calibrated.
    values = {
        'stock': dict(gc.get_model_defaults(is_crypto=False)),
        'crypto': dict(gc.get_model_defaults(is_crypto=True)),
    }
    provenance = dict(getattr(gc, 'CALIBRATION_PROVENANCE', {}))

    for asset_type, best in recommendations.items():
        if not best:
            continue
        values[asset_type] = {
            'vol_model': best['vol'].lower(),
            'p': int(best['p']),
            'o': int(best['o']),
            'q': int(best['q']),
            'dist': best['dist'],
            'vol_scale': round(float(best['median_vol_scale']), 3),
            'training_days': int(best['train_years'] * 365),
        }
        # Read defensively: this dict is built from run_calibration's ranked entries,
        # and a field rename there previously crashed the applier *after* a 30-minute
        # sweep had already succeeded. Missing keys should degrade the provenance
        # record, never lose the run.
        def _num(key, nd=4):
            v = best.get(key)
            try:
                return round(float(v), nd)
            except (TypeError, ValueError):
                return None

        provenance[asset_type] = {
            'calibrated_on': datetime.now().strftime('%Y-%m-%d'),
            'method': ('QLIKE walk-forward sweep; selection/holdout split with '
                       'Diebold-Mariano test vs incumbent'),
            'model': best.get('label'),
            'selection_qlike': _num('sel_qlike'),
            'holdout_qlike': _num('val_qlike'),
            'dm_p_vs_incumbent': _num('dm_p'),
            'free_parameters': best.get('complexity'),
            'tickers': best.get('tickers'),
            'vol_scale_range': [_num('vol_scale_min', 3), _num('vol_scale_max', 3)],
            'bias_ratio': _num('mean_bias_ratio', 3),
        }

    source = CONFIG_PATH.read_text()
    lines = source.splitlines()
    try:
        b = next(i for i, l in enumerate(lines) if l.startswith(BEGIN_MARKER))
        e = next(i for i, l in enumerate(lines) if l.startswith(END_MARKER))
    except StopIteration:
        print(f"  !! Could not find the managed block markers in {CONFIG_PATH}; nothing applied.")
        logger.error("apply failed: calibrated-defaults markers missing in garch_config.py")
        return None

    block = _render_calibrated_block(values, provenance)

    # Validate BEFORE touching the file. Writing first and checking afterwards once
    # left an unimportable config on disk (a None rendered as JSON `null`), which
    # takes the whole app down rather than just failing the calibration.
    try:
        exec(compile(block, '<calibrated-block>', 'exec'), {})
    except Exception as exc:
        print(f"  !! Refusing to write: rendered config block is invalid Python ({exc})")
        logger.error(f"apply aborted: rendered block failed validation: {exc}")
        return None

    new_source = '\n'.join(lines[:b] + block.splitlines() + lines[e + 1:])
    if not new_source.endswith('\n'):
        new_source += '\n'

    backup = source
    CONFIG_PATH.write_text(new_source)
    try:
        importlib.reload(gc)
    except Exception as exc:
        # Restore rather than leave production config broken.
        CONFIG_PATH.write_text(backup)
        importlib.reload(gc)
        print(f"  !! Wrote config but it failed to import ({exc}) — rolled back.")
        logger.error(f"apply rolled back: reload failed: {exc}")
        return None

    print(f"\n  Applied to {CONFIG_PATH}:")
    for at in ('stock', 'crypto'):
        v = values[at]
        print(f"    {at:<7} {v['vol_model']}(p={v['p']},o={v['o']},q={v['q']})-{v['dist']} "
              f"| vol_scale={v['vol_scale']} | train={v['training_days']}d")
        logger.info(f"APPLIED {at}: {v['vol_model']}(p={v['p']},o={v['o']},q={v['q']})-{v['dist']} "
                    f"vol_scale={v['vol_scale']} training_days={v['training_days']}")
    return CONFIG_PATH


def save_results(asset_type, ranked, per_ticker, recommendation=None,
                 out_dir='calibration_results'):
    """Persist the full ranking so the skill / future runs can diff against it."""
    out = Path(_PROJECT_ROOT) / out_dir
    out.mkdir(exist_ok=True)
    stamp = datetime.now().strftime('%Y-%m-%d')
    path = out / f"calibration-{asset_type}-{stamp}.json"
    payload = {
        'asset_type': asset_type,
        'generated': datetime.now().isoformat(),
        'recommendation': recommendation,
        'ranked': ranked,
        'tickers_evaluated': sorted(per_ticker),
    }
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"  Full results written to {path}")
    logger.info(f"Calibration results saved: {path}")
    return path


# ============================================================================
# CLI
# ============================================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='GARCH walk-forward backtest & model calibration')
    parser.add_argument('--ticker', default='NVDA')
    parser.add_argument('--windows', type=int, default=72)
    parser.add_argument('--validation-frac', type=float, default=0.25,
                        help='Fraction of most-recent windows held out to confirm the winner')
    parser.add_argument('--alpha', type=float, default=0.05,
                        help='Diebold-Mariano significance level required to change config')
    parser.add_argument('--train-years', type=int, default=2)
    parser.add_argument('--forecast-days', type=int, default=7)
    parser.add_argument('--p', type=int, default=1)
    parser.add_argument('--q', type=int, default=1)
    parser.add_argument('--o', type=int, default=1, help='Asymmetry order for EGARCH')
    parser.add_argument('--vol-model', default='Garch', choices=['Garch', 'EGARCH', 'garch', 'egarch'])
    parser.add_argument('--vol-scale', type=float, default=None)
    parser.add_argument('--csv', default=None, help='Save single-model detailed results as CSV')
    parser.add_argument('--asset-type', default='stock', choices=['stock', 'crypto', 'both'])
    parser.add_argument('--calibrate', action='store_true',
                        help='Full model-selection sweep across all tickers of the asset class')
    parser.add_argument('--apply', action='store_true',
                        help='Write the winning configs into garch_config.py (use with --calibrate)')
    parser.add_argument('--apply-from-results', metavar='DATE', nargs='?', const='latest',
                        help='Apply previously saved results from calibration_results/ '
                             '(YYYY-MM-DD, or omit for the most recent) without re-running '
                             'the sweep. Use when a sweep succeeded but the write-back failed.')
    parser.add_argument('--train-years-grid', default='2',
                        help='Comma-separated training windows to test, e.g. "2,5"')
    parser.add_argument('--max-workers', type=int, default=None, help='Parallel worker processes')
    parser.add_argument('--top-n', type=int, default=15)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    if args.apply_from_results:
        results_dir = Path(_PROJECT_ROOT) / 'calibration_results'
        recs = {}
        for at in ('stock', 'crypto'):
            if args.apply_from_results == 'latest':
                files = sorted(results_dir.glob(f'calibration-{at}-*.json'))
            else:
                files = sorted(results_dir.glob(f'calibration-{at}-{args.apply_from_results}.json'))
            if not files:
                print(f"  no saved results for {at}")
                continue
            payload = json.loads(files[-1].read_text())
            rec = payload.get('recommendation')
            if rec:
                recs[at] = rec
                print(f"  {at:<7} <- {files[-1].name}: {rec.get('label')} "
                      f"(ty{rec.get('train_years')}) vol_scale={rec.get('median_vol_scale'):.3f}")
            else:
                print(f"  {at}: saved results contain no recommendation")
        if recs:
            apply_recommendations(recs)
        else:
            print("  nothing to apply")
        sys.exit(0)

    if args.calibrate:
        ty_grid = tuple(int(x) for x in args.train_years_grid.split(','))
        asset_types = ['stock', 'crypto'] if args.asset_type == 'both' else [args.asset_type]
        recommendations = {}
        for at in asset_types:
            ranked, per_ticker, rec = run_calibration(
                asset_type=at,
                n_windows=args.windows,
                forecast_days=args.forecast_days,
                train_years_grid=ty_grid,
                max_workers=args.max_workers,
                top_n=args.top_n,
                validation_frac=args.validation_frac,
                alpha=args.alpha,
            )
            if ranked:
                save_results(at, ranked, per_ticker, recommendation=rec)
            if rec:
                recommendations[at] = rec

        if args.apply and recommendations:
            apply_recommendations(recommendations)
        elif recommendations:
            print("\n  (re-run with --apply to write these into garch_config.py)")
    else:
        is_crypto = args.asset_type == 'crypto'
        if args.vol_scale is None:
            args.vol_scale = DEFAULT_CRYPTO_VOL_SCALE if is_crypto else DEFAULT_VOL_SCALE

        df = run_backtest(
            ticker=args.ticker, n_windows=args.windows, train_years=args.train_years,
            forecast_days=args.forecast_days, p=args.p, q=args.q,
            vol_model=args.vol_model, o=args.o, vol_scale=args.vol_scale,
            verbose=args.verbose, is_crypto=is_crypto,
        )
        summarize(df, p=args.p, q=args.q,
                  o=args.o if args.vol_model.lower() == 'egarch' else None,
                  vol_model=args.vol_model)

        if not df.empty:
            logger.info(f"SINGLE MODEL {args.ticker} ({args.asset_type}): "
                        f"{args.vol_model}({args.p},{args.q}) | windows={len(df)} | "
                        f"MAPE={df['mape'].mean():.2f}% | vol_err={df['vol_pct_error'].mean():.1f}% | "
                        f"BIC={df['bic'].mean():.1f} | dir_acc={df['direction_correct'].mean()*100:.1f}%")

        if args.csv:
            df.to_csv(args.csv, index=False)
            print(f"\nDetailed results saved to {args.csv}")
