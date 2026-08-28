#!/usr/bin/env python3
"""
Walk-forward backtest for the GARCH(1,1) forecasting model.

For each of N rolling windows (default 36, weekly steps):
  - anchor_date = (today - i weeks)
  - train on the 2 years of data ending at anchor_date
  - forecast the following 7 trading days (volatility + drift-based price path)
  - compare forecast against what actually happened in the database

This validates whether the GARCH volatility term structure and the historical-mean
drift assumption actually track reality, and reports aggregate error metrics so the
model (or its p/q order, training window length, etc.) can be tuned.
"""

import sys
import sqlite3
import numpy as np
import pandas as pd
import warnings
from datetime import datetime, timedelta
from db import DB_PATH
from garch_config import (
    VALID_GARCH_ORDERS,
    VALID_VOL_MODELS,
    DEFAULT_VOL_SCALE,
    DEFAULT_CRYPTO_VOL_SCALE
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


def get_prices_in_range(ticker, start_date, end_date, is_crypto=False):
    """Fetch (date, close) rows for ticker between start_date and end_date inclusive."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    table = 'crypto_prices' if is_crypto else 'prices'
    cursor.execute(f'''
        SELECT date, close FROM {table}
        WHERE ticker = ? AND date BETWEEN ? AND ?
        ORDER BY date
    ''', (ticker, start_date.isoformat(), end_date.isoformat()))
    rows = cursor.fetchall()
    conn.close()
    return rows


def run_backtest(ticker='NVDA', n_windows=36, train_years=2, forecast_days=7,
                  p=1, q=1, vol_model='Garch', o=1, vol_scale=1.0, step_weeks=1, verbose=True, is_crypto=False):
    """
    Rolling walk-forward backtest.

    Window i (0-indexed): anchor_date = last_available_date - (i+1)*step_weeks weeks
      train:    [anchor_date - train_years, anchor_date]
      forecast: the forecast_days trading days AFTER anchor_date
    """
    # Determine the latest date we have data for (acts as "today")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    table = 'crypto_prices' if is_crypto else 'prices'
    cursor.execute(f'SELECT MAX(date) FROM {table} WHERE ticker = ?', (ticker,))
    row = cursor.fetchone()
    conn.close()
    if not row or not row[0]:
        raise ValueError(f"No data found for {ticker}")
    latest_date = datetime.strptime(row[0], '%Y-%m-%d').date()

    results = []

    for i in range(1, n_windows + 1):
        anchor_date = latest_date - timedelta(weeks=i * step_weeks)
        train_start = anchor_date - timedelta(days=train_years * 365)
        train_end = anchor_date

        # --- Fetch training data and fit GARCH ---
        train_rows = get_prices_in_range(ticker, train_start, train_end, is_crypto=is_crypto)
        if len(train_rows) < 100:
            if verbose:
                print(f"[window {i}] Skipping anchor={anchor_date}: insufficient training data ({len(train_rows)} rows)")
            continue

        train_closes = np.array([float(r[1]) for r in train_rows])
        train_returns = np.diff(np.log(train_closes)) * 100  # log returns, %

        try:
            if vol_model.lower() == 'egarch':
                model = arch_model(train_returns, vol='EGARCH', p=p, o=o, q=q)
            else:
                model = arch_model(train_returns, vol='Garch', p=p, q=q)
            fitted = model.fit(disp='off')
        except Exception as e:
            if verbose:
                print(f"[window {i}] GARCH fit failed at anchor={anchor_date}: {e}")
            continue

        # EGARCH (and other asymmetric models) have no closed-form multi-step forecast;
        # they require simulation-based forecasting for horizon > 1.
        forecast_method = 'simulation' if vol_model.lower() == 'egarch' else 'analytic'
        try:
            forecast = fitted.forecast(horizon=forecast_days, method=forecast_method, reindex=False)
        except Exception as e:
            if verbose:
                print(f"[window {i}] Forecast failed at anchor={anchor_date}: {e}")
            continue
        variance_forecast = forecast.variance.values[-1, :] if hasattr(forecast.variance, 'values') else forecast.variance[-1, :]
        forecasted_vol = np.sqrt(variance_forecast) / 100 * vol_scale  # back to fraction, calibrated

        drift = float(np.mean(train_returns)) / 100  # per-day drift as fraction
        last_train_close = train_closes[-1]

        # --- Fetch actual data for the forecast window ---
        # Forecast window: the forecast_days trading days strictly after anchor_date
        actual_end = anchor_date + timedelta(days=forecast_days * 3)  # generous buffer for weekends/holidays
        actual_rows = get_prices_in_range(ticker, anchor_date + timedelta(days=1), actual_end, is_crypto=is_crypto)
        actual_rows = actual_rows[:forecast_days]  # take exactly forecast_days trading days

        if len(actual_rows) < forecast_days:
            if verbose:
                print(f"[window {i}] Skipping anchor={anchor_date}: insufficient actual data ({len(actual_rows)} rows)")
            continue

        actual_closes = np.array([float(r[1]) for r in actual_rows])
        actual_dates = [r[0] for r in actual_rows]

        # --- Build the deterministic (expected/drift-only) forecast path ---
        expected_path = []
        price = last_train_close
        for day_idx in range(forecast_days):
            price = price * (1 + drift)
            expected_path.append(price)
        expected_path = np.array(expected_path)

        # --- Price-path error metrics ---
        abs_errors = np.abs(expected_path - actual_closes)
        pct_errors = abs_errors / actual_closes * 100
        mae = float(np.mean(abs_errors))
        mape = float(np.mean(pct_errors))
        rmse = float(np.sqrt(np.mean((expected_path - actual_closes) ** 2)))

        # --- Direction accuracy (did the week move the direction the drift predicted?) ---
        actual_week_return = (actual_closes[-1] - last_train_close) / last_train_close
        predicted_direction = 1 if drift > 0 else (-1 if drift < 0 else 0)
        actual_direction = 1 if actual_week_return > 0 else (-1 if actual_week_return < 0 else 0)
        direction_correct = predicted_direction == actual_direction

        # --- Volatility error metrics ---
        actual_returns = np.diff(np.log(np.concatenate([[last_train_close], actual_closes])))
        realized_vol_daily = float(np.std(actual_returns))  # fraction
        forecasted_vol_avg = float(np.mean(forecasted_vol))
        vol_error = abs(forecasted_vol_avg - realized_vol_daily)
        vol_pct_error = vol_error / realized_vol_daily * 100 if realized_vol_daily != 0 else np.nan

        results.append({
            'window': i,
            'anchor_date': anchor_date.isoformat(),
            'train_start': train_start.isoformat(),
            'forecast_start': actual_dates[0],
            'forecast_end': actual_dates[-1],
            'last_train_close': last_train_close,
            'drift_daily_pct': drift * 100,
            'mae': mae,
            'mape': mape,
            'rmse': rmse,
            'direction_correct': direction_correct,
            'forecasted_vol_avg_pct': forecasted_vol_avg * 100,
            'realized_vol_daily_pct': realized_vol_daily * 100,
            'vol_error_pct_points': vol_error * 100,
            'vol_pct_error': vol_pct_error,
            'aic': float(fitted.aic),
            'bic': float(fitted.bic),
        })

        if verbose:
            print(f"[window {i:2d}] anchor={anchor_date} | MAPE={mape:5.2f}% | "
                  f"dir_correct={str(direction_correct):5s} | "
                  f"fcst_vol={forecasted_vol_avg*100:.2f}% vs realized={realized_vol_daily*100:.2f}%")

    return pd.DataFrame(results)


def summarize(df, p=1, q=1, o=None, vol_model='Garch'):
    """Print aggregate statistics across all backtest windows."""
    if df.empty:
        print("No valid windows to summarize.")
        return

    o_str = f", o={o}" if o else ""
    model_label = f"{vol_model}({p},{q}{o_str})"

    print("\n" + "=" * 70)
    print(f"BACKTEST SUMMARY — {model_label} — {len(df)} windows")
    print("=" * 70)
    print(f"Price forecast (drift-only expected path vs actual):")
    print(f"  Mean MAE:   ${df['mae'].mean():.2f}")
    print(f"  Mean MAPE:  {df['mape'].mean():.2f}%")
    print(f"  Mean RMSE:  ${df['rmse'].mean():.2f}")
    print()
    print(f"Direction accuracy (did drift sign match the week's actual move?):")
    print(f"  {df['direction_correct'].sum()}/{len(df)} correct = {df['direction_correct'].mean()*100:.1f}%")
    print()
    print(f"Volatility forecast (GARCH forecasted vol vs realized vol that week):")
    print(f"  Mean forecasted vol: {df['forecasted_vol_avg_pct'].mean():.2f}%/day")
    print(f"  Mean realized vol:   {df['realized_vol_daily_pct'].mean():.2f}%/day")
    print(f"  Mean abs error:      {df['vol_error_pct_points'].mean():.2f} pct points")
    print(f"  Mean pct error:      {df['vol_pct_error'].mean():.1f}%")
    print()
    print(f"Model fit quality:")
    print(f"  Mean AIC: {df['aic'].mean():.1f}")
    print(f"  Mean BIC: {df['bic'].mean():.1f}")
    print("=" * 70)


def run_backtest_sweep(ticker, asset_type='stock', n_windows=36, verbose=False):
    """
    Run walk-forward backtest across all supported GARCH model variants.

    Tests all combinations of:
    - Vol models: GARCH, EGARCH
    - GARCH orders: (1,1), (1,2), (2,1), (2,2)
    - Vol scale: default for asset type (0.8 for stocks, 0.9 for crypto)

    Returns a DataFrame with best model by vol_model type and aggregate results.
    """
    is_crypto = asset_type.lower() == 'crypto'
    default_vol_scale = DEFAULT_CRYPTO_VOL_SCALE if is_crypto else DEFAULT_VOL_SCALE

    results_summary = []
    all_models = []

    # Log start of backtest sweep
    logger.info("=" * 70)
    logger.info(f"MODEL CALIBRATION SWEEP: {ticker} ({asset_type.upper()})")
    logger.info("=" * 70)
    logger.info(f"Testing {len(VALID_VOL_MODELS)} vol models × {len(VALID_GARCH_ORDERS)} orders = {len(VALID_VOL_MODELS) * len(VALID_GARCH_ORDERS)} configurations")
    logger.info(f"Windows: {n_windows} | Vol Scale: {default_vol_scale}×")

    print(f"\n{'='*70}")
    print(f"BACKTEST SWEEP: {ticker} ({asset_type.upper()})")
    print(f"{'='*70}")
    print(f"Testing all GARCH model variants ({len(VALID_VOL_MODELS)} vol models × {len(VALID_GARCH_ORDERS)} orders)...\n")

    for vol_model in VALID_VOL_MODELS:
        print(f"\n--- {vol_model.upper()} Variants ---")
        vol_model_results = []

        for p, q in VALID_GARCH_ORDERS:
            o = 1 if vol_model.lower() == 'egarch' else None
            model_name = f"{vol_model}({p},{q})" + (f",o={o}" if o else "")

            try:
                df = run_backtest(
                    ticker=ticker,
                    n_windows=n_windows,
                    train_years=2,
                    forecast_days=7,
                    p=p,
                    q=q,
                    vol_model=vol_model,
                    o=o,
                    vol_scale=default_vol_scale,
                    step_weeks=1,
                    verbose=verbose,
                    is_crypto=is_crypto
                )

                if not df.empty:
                    mean_mape = df['mape'].mean()
                    mean_vol_error = df['vol_pct_error'].mean()
                    mean_bic = df['bic'].mean()
                    dir_acc = df['direction_correct'].mean() * 100

                    vol_model_results.append({
                        'model': model_name,
                        'p': p,
                        'q': q,
                        'o': o,
                        'vol_model': vol_model,
                        'windows': len(df),
                        'mean_mape': mean_mape,
                        'mean_vol_error': mean_vol_error,
                        'mean_bic': mean_bic,
                        'dir_accuracy': dir_acc,
                        'dataframe': df
                    })

                    if not verbose:
                        print(f"  {model_name:20s} | MAPE: {mean_mape:6.2f}% | Vol Err: {mean_vol_error:6.1f}% | BIC: {mean_bic:7.1f} | Dir: {dir_acc:5.1f}%")
                else:
                    print(f"  {model_name:20s} | FAILED (insufficient windows)")
            except Exception as e:
                print(f"  {model_name:20s} | ERROR: {e}")

        # Find best model for this vol_model type by BIC (lower is better)
        if vol_model_results:
            vol_model_results.sort(key=lambda x: x['mean_bic'])
            best = vol_model_results[0]
            results_summary.append(best)
            all_models.extend(vol_model_results)

            print(f"\n  ✓ Best {vol_model.upper()} model: {best['model']}")
            print(f"    Mean MAPE: {best['mean_mape']:.2f}%")
            print(f"    Mean Vol Error: {best['mean_vol_error']:.1f}%")
            print(f"    Mean BIC: {best['mean_bic']:.1f}")
            print(f"    Direction Accuracy: {best['dir_accuracy']:.1f}%")

    # Summary of best-per-volmodel
    print(f"\n{'='*70}")
    print(f"RECOMMENDED CONFIGURATIONS FOR {ticker} ({asset_type.upper()})")
    print(f"{'='*70}")

    # Log and print summary
    logger.info("\n" + "=" * 70)
    logger.info(f"RECOMMENDED CONFIGURATIONS FOR {ticker} ({asset_type.upper()})")
    logger.info("=" * 70)

    for best in results_summary:
        summary_line = (f"{best['vol_model'].upper()} | Config: {best['model']} | "
                       f"MAPE: {best['mean_mape']:.2f}% | Vol Err: {best['mean_vol_error']:.1f}% | "
                       f"BIC: {best['mean_bic']:.1f} | Direction Acc: {best['dir_accuracy']:.1f}% | "
                       f"Windows: {best['windows']}")
        logger.info(summary_line)

        print(f"\n{best['vol_model'].upper()} Model:")
        print(f"  Config: {best['model']}")
        print(f"  Mean MAPE: {best['mean_mape']:.2f}%")
        print(f"  Mean Vol Error: {best['mean_vol_error']:.1f}%")
        print(f"  Mean BIC: {best['mean_bic']:.1f}")
        print(f"  Direction Accuracy: {best['dir_accuracy']:.1f}%")
        print(f"  Windows: {best['windows']}")

    logger.info("=" * 70)

    return results_summary, all_models


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Walk-forward GARCH backtest (single model or sweep all variants)')
    parser.add_argument('--ticker', default='NVDA')
    parser.add_argument('--windows', type=int, default=36)
    parser.add_argument('--train-years', type=int, default=2)
    parser.add_argument('--forecast-days', type=int, default=7)
    parser.add_argument('--p', type=int, default=1)
    parser.add_argument('--q', type=int, default=1)
    parser.add_argument('--o', type=int, default=1, help='Asymmetry order for EGARCH')
    parser.add_argument('--vol-model', default='Garch', choices=['Garch', 'EGARCH', 'garch', 'egarch'])
    parser.add_argument('--vol-scale', type=float, default=None, help='Calibration multiplier applied to forecasted volatility (defaults to asset-specific value)')
    parser.add_argument('--csv', default=None, help='Optional path to save detailed results as CSV')
    parser.add_argument('--asset-type', default='stock', choices=['stock', 'crypto'],
                       help='Asset type: stock or crypto (determines data table and defaults)')
    parser.add_argument('--sweep', action='store_true',
                       help='Run backtest across all GARCH model variants instead of single model')
    parser.add_argument('--verbose', action='store_true', help='Print window-by-window results')
    args = parser.parse_args()

    is_crypto = args.asset_type.lower() == 'crypto'

    if args.sweep:
        # Run sweep across all model variants
        best_models, all_models = run_backtest_sweep(
            ticker=args.ticker,
            asset_type=args.asset_type,
            n_windows=args.windows,
            verbose=args.verbose
        )
    else:
        # Run single model (legacy behavior)
        if args.vol_scale is None:
            args.vol_scale = DEFAULT_CRYPTO_VOL_SCALE if is_crypto else DEFAULT_VOL_SCALE

        df = run_backtest(
            ticker=args.ticker,
            n_windows=args.windows,
            train_years=args.train_years,
            forecast_days=args.forecast_days,
            p=args.p,
            q=args.q,
            vol_model=args.vol_model,
            o=args.o,
            vol_scale=args.vol_scale,
            verbose=not args.verbose,
            is_crypto=is_crypto
        )

        summarize(df, p=args.p, q=args.q, o=args.o if args.vol_model.lower() == 'egarch' else None, vol_model=args.vol_model)

        # Log single-model results
        if not df.empty:
            logger.info("=" * 70)
            logger.info(f"SINGLE MODEL BACKTEST: {args.ticker} ({args.asset_type.upper()})")
            logger.info("=" * 70)
            model_label = f"{args.vol_model}({args.p},{args.q})" + (f",o={args.o}" if args.vol_model.lower() == 'egarch' else "")
            logger.info(f"Model: {model_label}")
            logger.info(f"Windows: {len(df)}")
            logger.info(f"Mean MAPE: {df['mape'].mean():.2f}%")
            logger.info(f"Mean Vol Error: {df['vol_pct_error'].mean():.1f}%")
            logger.info(f"Mean BIC: {df['bic'].mean():.1f}")
            logger.info(f"Direction Accuracy: {df['direction_correct'].mean()*100:.1f}%")
            logger.info("=" * 70)

        if args.csv:
            df.to_csv(args.csv, index=False)
            print(f"\nDetailed results saved to {args.csv}")
            logger.info(f"Results saved to {args.csv}")
