"""Tests for the production volatility forecaster and its configuration.

These guard the contract between garch_config (machine-written by the calibration
skill) and garch_model (which must be able to actually serve whatever the
calibration picks). A recommendation the app cannot serve is a silent outage.
"""

import numpy as np
import pytest

from garch import garch_config as gc
from garch.garch_model import (
    _resolve_params,
    build_arch_model,
    forecast_variance,
    forecast_volatility,
    get_garch_stats,
)


# --------------------------------------------------------- config integrity

def test_calibrated_defaults_are_within_the_declared_search_space():
    """Whatever the calibration wrote must be something the UI/API will accept."""
    for is_crypto in (False, True):
        d = gc.get_model_defaults(is_crypto=is_crypto)
        assert d['vol_model'] in gc.VALID_VOL_MODELS
        assert d['dist'] in gc.VALID_DISTRIBUTIONS
        assert d['o'] in gc.VALID_ASYMMETRY_ORDERS
        if d['vol_model'] not in ('arch', 'harch'):
            assert (d['p'], d['q']) in gc.VALID_GARCH_ORDERS
        assert d['training_days'] > 0


def test_calibrated_vol_scale_is_within_allowed_bounds():
    s = gc.get_model_defaults(is_crypto=False)['vol_scale']
    c = gc.get_model_defaults(is_crypto=True)['vol_scale']
    assert gc.MIN_VOL_SCALE <= s <= gc.MAX_VOL_SCALE
    assert gc.MIN_CRYPTO_VOL_SCALE <= c <= gc.MAX_CRYPTO_VOL_SCALE


def test_stock_and_crypto_defaults_are_independent_keys():
    """Crypto must not silently inherit stock settings."""
    s = gc.get_model_defaults(is_crypto=False)
    c = gc.get_model_defaults(is_crypto=True)
    assert set(s) == set(c)


# ------------------------------------------------------- parameter resolution

def test_resolve_params_fills_from_asset_defaults():
    p, q, o, vm, days, dist = _resolve_params(None, None, None, None, None, None, False)
    d = gc.get_model_defaults(is_crypto=False)
    assert (p, q, o, vm, days, dist) == (
        d['p'], d['q'], d['o'], d['vol_model'], d['training_days'], d['dist'])


def test_resolve_params_rejects_garbage_and_falls_back():
    _, _, o, vm, _, dist = _resolve_params(
        1, 1, 99, 'not-a-model', None, 'not-a-dist', False)
    d = gc.get_model_defaults(is_crypto=False)
    assert vm == d['vol_model']
    assert dist == d['dist']
    assert o == d['o']


def test_resolve_params_allows_arch_without_a_q_term():
    """ARCH/HARCH have no q; the (p,q) whitelist must not force them back."""
    p, q, _, vm, _, _ = _resolve_params(3, 0, 0, 'arch', None, 'normal', False)
    assert vm == 'arch' and p == 3 and q == 0


def test_resolve_params_uses_crypto_defaults_when_flagged():
    _, _, _, vm, days, _ = _resolve_params(None, None, None, None, None, None, True)
    assert days == gc.get_model_defaults(is_crypto=True)['training_days']
    assert vm == gc.get_model_defaults(is_crypto=True)['vol_model']


# ------------------------------------------------- every family is servable

@pytest.mark.parametrize("vol_model", gc.VALID_VOL_MODELS)
@pytest.mark.parametrize("dist", gc.VALID_DISTRIBUTIONS)
def test_every_family_and_distribution_fits_and_forecasts(vol_model, dist):
    """Synthetic data so this does not depend on the database being populated."""
    rng = np.random.default_rng(7)
    n = 600
    r = np.zeros(n)
    s = 1.5
    for i in range(1, n):
        s = np.sqrt(0.05 + 0.08 * r[i - 1] ** 2 + 0.90 * s ** 2)
        r[i] = rng.normal(0.03, s)

    o = 1 if vol_model in ('garch', 'egarch', 'aparch') else 0
    model = build_arch_model(r, p=1, o=o, q=1, vol_model=vol_model, dist=dist)
    fitted = model.fit(disp='off', show_warning=False)
    var = forecast_variance(fitted, vol_model, periods=7)

    assert len(var) == 7
    assert np.all(np.isfinite(var)), f"{vol_model}-{dist} produced non-finite variance"
    assert np.all(var > 0), f"{vol_model}-{dist} produced non-positive variance"


# ------------------------------------------------------------ live DB smoke

def _has_data(ticker, is_crypto):
    import sqlite3
    from db import DB_PATH
    table = 'crypto_prices' if is_crypto else 'prices'
    con = sqlite3.connect(DB_PATH)
    try:
        n = con.execute(f"SELECT COUNT(*) FROM {table} WHERE ticker=?", (ticker,)).fetchone()[0]
    except sqlite3.Error:
        return False
    finally:
        con.close()
    return n > 300


@pytest.mark.parametrize("ticker,is_crypto", [('AAPL', False), ('BTC', True)])
def test_forecast_volatility_against_real_data(ticker, is_crypto):
    if not _has_data(ticker, is_crypto):
        pytest.skip(f"no local data for {ticker}")
    res = forecast_volatility(ticker, periods=7, is_crypto=is_crypto)
    assert res['status'] == 'success'
    vols = res['forecasted_volatility']
    assert len(vols) == 7
    assert all(v > 0 and np.isfinite(v) for v in vols)
    assert res['model_info']['vol_model'] in gc.VALID_VOL_MODELS
    assert res['model_info']['dist'] in gc.VALID_DISTRIBUTIONS


@pytest.mark.parametrize("ticker,is_crypto", [('AAPL', False), ('BTC', True)])
def test_garch_stats_against_real_data(ticker, is_crypto):
    if not _has_data(ticker, is_crypto):
        pytest.skip(f"no local data for {ticker}")
    stats = get_garch_stats(ticker, is_crypto=is_crypto)
    assert stats is not None
    assert stats['current_volatility'] > 0
    assert stats['coefficients']


def test_vol_scale_is_clamped_to_bounds():
    if not _has_data('AAPL', False):
        pytest.skip("no local data")
    hi = forecast_volatility('AAPL', periods=3, vol_scale=999)
    lo = forecast_volatility('AAPL', periods=3, vol_scale=-5)
    assert hi['vol_scale'] == gc.MAX_VOL_SCALE
    assert lo['vol_scale'] == gc.MIN_VOL_SCALE


def test_unknown_ticker_errors_cleanly_rather_than_raising():
    res = forecast_volatility('NOSUCHTICKER', periods=3)
    assert res['status'] == 'error'
