"""Tests for the volatility model calibration engine.

The pure-math tests here are the important ones: the calibration writes
production config automatically, so a silent regression in the scoring maths
would ship a bad model without anyone noticing.
"""

import numpy as np
import pytest

from garch.garch_backtest import (
    optimal_scale,
    qlike_losses,
    diebold_mariano,
    config_complexity,
    config_label,
    build_config_grid,
    _render_calibrated_block,
    MODEL_STRUCTURES,
    DISTRIBUTIONS,
)


# ---------------------------------------------------------------- QLIKE maths

def test_optimal_scale_recovers_known_bias():
    """If returns really have k^2x the forecast variance, c* must recover k."""
    rng = np.random.default_rng(0)
    n = 50_000
    for true_k in (0.7, 1.0, 1.5):
        sigma2 = rng.gamma(4.0, 0.5, n)
        r2 = (rng.standard_normal(n) * np.sqrt(sigma2 * true_k ** 2)) ** 2
        assert optimal_scale(sigma2, r2) == pytest.approx(true_k, rel=0.02)


def test_optimal_scale_is_a_true_qlike_minimum():
    """The closed form must beat any neighbouring scale."""
    rng = np.random.default_rng(1)
    sigma2 = rng.gamma(3.0, 0.7, 20_000)
    r2 = (rng.standard_normal(20_000) * np.sqrt(sigma2 * 1.3)) ** 2
    c = optimal_scale(sigma2, r2)
    best = qlike_losses(sigma2, r2, c).mean()
    for k in (0.8, 0.9, 0.95, 1.05, 1.1, 1.25):
        assert qlike_losses(sigma2, r2, c * k).mean() > best


def test_qlike_analytic_identity_at_optimum():
    """QLIKE* == 2*ln(c*) + mean(ln sigma2) + 1, the identity the engine relies on."""
    rng = np.random.default_rng(2)
    sigma2 = rng.gamma(2.0, 1.0, 10_000)
    r2 = (rng.standard_normal(10_000) * np.sqrt(sigma2)) ** 2
    c = optimal_scale(sigma2, r2)
    analytic = 2 * np.log(c) + np.mean(np.log(sigma2)) + 1.0
    assert qlike_losses(sigma2, r2, c).mean() == pytest.approx(analytic, rel=1e-9)


def test_qlike_penalises_both_over_and_under_forecasting():
    rng = np.random.default_rng(3)
    sigma2 = np.full(5_000, 1.0)
    r2 = (rng.standard_normal(5_000)) ** 2
    perfect = qlike_losses(sigma2, r2, 1.0).mean()
    assert qlike_losses(sigma2, r2, 0.5).mean() > perfect
    assert qlike_losses(sigma2, r2, 2.0).mean() > perfect


# ------------------------------------------------------- Diebold-Mariano test

def test_dm_does_not_false_positive_on_pure_noise():
    """Under the null of equal accuracy the test must not reject at 5%."""
    rng = np.random.default_rng(4)
    rejections = 0
    trials = 40
    for _ in range(trials):
        a = rng.standard_normal(2_000)
        b = a + rng.standard_normal(2_000) * 0.05
        _, p, _ = diebold_mariano(a, b, lag=7)
        if np.isfinite(p) and p < 0.05:
            rejections += 1
    # Expect ~5% false positives; allow generous slack for 40 trials.
    assert rejections <= 8, f"too many false positives: {rejections}/{trials}"


def test_dm_detects_a_genuine_accuracy_difference():
    rng = np.random.default_rng(5)
    a = rng.standard_normal(3_000)
    b = a + rng.standard_normal(3_000) * 0.5 + 0.2   # genuinely worse on average
    stat, p, mean_diff = diebold_mariano(a, b, lag=7)
    assert p < 0.01
    assert stat < 0          # negative => `a` has the lower loss
    assert mean_diff < 0


def test_dm_returns_nan_on_insufficient_data():
    _, p, _ = diebold_mariano(np.arange(5.0), np.arange(5.0) + 1, lag=7)
    assert np.isnan(p)


def test_dm_hac_widens_errors_under_autocorrelation():
    """Overlapping forecast windows autocorrelate the loss differential; the HAC
    correction must produce a less significant result than assuming independence."""
    rng = np.random.default_rng(6)
    n = 4_000
    e = rng.standard_normal(n)
    ac = np.convolve(e, np.ones(20) / 20, mode='same')   # strongly autocorrelated
    a, b = ac, np.zeros(n)
    stat_hac, _, _ = diebold_mariano(a, b, lag=40)
    stat_naive, _, _ = diebold_mariano(a, b, lag=1)
    assert abs(stat_hac) < abs(stat_naive)


# ------------------------------------------------------------- Model space

def test_every_structure_distribution_combo_is_labelled_uniquely():
    grid = build_config_grid((2, 5))
    labels = [config_label(c) + f"|ty{c['train_years']}" for c in grid]
    assert len(labels) == len(set(labels)), "config labels must be unique keys"
    assert len(grid) == len(MODEL_STRUCTURES) * len(DISTRIBUTIONS) * 2


def test_complexity_orders_models_sensibly():
    simple = {'vol': 'GARCH', 'p': 1, 'o': 0, 'q': 1, 'dist': 'normal'}
    asym = {'vol': 'GARCH', 'p': 1, 'o': 1, 'q': 1, 'dist': 'normal'}
    fat = {'vol': 'GARCH', 'p': 1, 'o': 1, 'q': 1, 'dist': 'skewt'}
    assert config_complexity(simple) < config_complexity(asym) < config_complexity(fat)


def test_gjr_is_labelled_distinctly_from_plain_garch():
    plain = config_label({'vol': 'GARCH', 'p': 1, 'o': 0, 'q': 1, 'dist': 'normal'})
    gjr = config_label({'vol': 'GARCH', 'p': 1, 'o': 1, 'q': 1, 'dist': 'normal'})
    assert plain != gjr and 'GJR' in gjr


# ------------------------------------------------------ Config write-back

def _sample_values():
    return {
        'stock': {'vol_model': 'garch', 'p': 1, 'o': 1, 'q': 1, 'dist': 'skewt',
                  'vol_scale': 0.771, 'training_days': 730},
        'crypto': {'vol_model': 'egarch', 'p': 2, 'o': 1, 'q': 1, 'dist': 't',
                   'vol_scale': 0.912, 'training_days': 1825},
    }


def test_rendered_config_block_is_valid_python_with_exact_values():
    """The block is machine-written into a live module — it must exec cleanly and
    round-trip every value, or the app boots with a corrupted config."""
    values = _sample_values()
    block = _render_calibrated_block(values, {'stock': {'calibrated_on': '2026-08-28'}})
    ns = {}
    exec(compile(block, '<block>', 'exec'), ns)   # noqa: S102 - deliberate

    assert ns['DEFAULT_VOL_MODEL'] == 'garch'
    assert (ns['DEFAULT_GARCH_P'], ns['DEFAULT_GARCH_O'], ns['DEFAULT_GARCH_Q']) == (1, 1, 1)
    assert ns['DEFAULT_GARCH_DIST'] == 'skewt'
    assert ns['DEFAULT_VOL_SCALE'] == 0.771
    assert ns['GARCH_TRAINING_DAYS'] == 730

    assert ns['DEFAULT_CRYPTO_VOL_MODEL'] == 'egarch'
    assert ns['DEFAULT_CRYPTO_GARCH_DIST'] == 't'
    assert ns['DEFAULT_CRYPTO_VOL_SCALE'] == 0.912
    assert ns['CRYPTO_GARCH_TRAINING_DAYS'] == 1825
    assert 'CALIBRATION_PROVENANCE' in ns


@pytest.mark.parametrize("provenance", [
    # Regression: None must render as Python `None`, not JSON `null`. Rendering the
    # provenance dict with json.dumps once produced a config that would not import,
    # and the original version of this test never exercised a None so it passed.
    {'stock': {'dm_p_vs_incumbent': None, 'vol_scale_range': [None, None]}},
    {'stock': {'flag': True, 'other': False, 'missing': None}},
    {'stock': {'nested': {'a': None, 'b': [1, None, True]}}},
    {},
])
def test_rendered_block_survives_none_and_bools_in_provenance(provenance):
    block = _render_calibrated_block(_sample_values(), provenance)
    ns = {}
    exec(compile(block, '<block>', 'exec'), ns)   # noqa: S102 - deliberate
    assert ns['CALIBRATION_PROVENANCE'] == provenance
    assert 'null' not in block and 'true' not in block.replace('true_', '')


def test_config_file_still_has_the_managed_markers():
    """apply_recommendations locates its write target by these markers; if they are
    ever renamed by hand the applier silently stops working."""
    from garch.garch_backtest import CONFIG_PATH, BEGIN_MARKER, END_MARKER
    text = CONFIG_PATH.read_text()
    assert text.count(BEGIN_MARKER) == 1
    assert text.count(END_MARKER) == 1
    assert text.index(BEGIN_MARKER) < text.index(END_MARKER)
