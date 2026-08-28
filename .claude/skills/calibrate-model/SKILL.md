---
name: calibrate-model
description: Runs a full walk-forward model-selection sweep over every GARCH-family model the arch library supports, for stocks and crypto separately, and writes the winning config and vol_scale into garch/garch_config.py
aliases:
  - calibrate model
  - calibrate the model
  - model calibration
  - recalibrate volatility
  - recalibrate the model
  - find the best garch config
---

# Model Calibration Skill

Finds the volatility model configuration that makes forecasts track realized
volatility as closely as possible — **separately for stocks and crypto** — and
writes the winners into `garch/garch_config.py`.

This is a real model-selection sweep, not a single-parameter A/B. It fits every
volatility family `arch` supports, at every order, under every error distribution,
across every ticker, and ranks them on out-of-sample forecast loss.

---

## RUN THIS

One command does everything: sweep both asset classes, then write the results into
config.

```bash
cd /Users/eduardchumak/MyProjects/TickerWatcher
python3 garch/garch_backtest.py --calibrate --asset-type both \
    --windows 72 --train-years-grid 2,5 --top-n 20 --max-workers 8 --apply
```

**This is a long job** — on the order of 100k model fits for the stock universe,
roughly 30-60 minutes. Run it in the background and watch the log:

```bash
nohup python3 garch/garch_backtest.py --calibrate --asset-type both \
    --windows 72 --train-years-grid 2,5 --top-n 20 --max-workers 8 --apply \
    > /tmp/calibration.log 2>&1 &

# progress (one line per ticker as it finishes)
tail -f /tmp/calibration.log
```

Drop `--apply` to preview the ranking without touching config. Narrow with
`--asset-type stock` or `--asset-type crypto` to calibrate one class; the other
keeps its current settings.

### Steps to follow when invoked

1. Run the command above (background it, then poll the log — do not block).
2. When it finishes, read the two `RECOMMENDATION` blocks from the output.
3. Confirm `--apply` reported `Applied to .../garch_config.py` for both classes.
4. Verify the app still imports and serves the new config:
   ```bash
   python3 -c "
   from garch.garch_config import get_model_defaults, CALIBRATION_PROVENANCE
   from garch.garch_model import forecast_volatility
   for c in (False, True):
       print(get_model_defaults(c))
   print(forecast_volatility('AAPL', periods=7)['status'])
   print(forecast_volatility('BTC', periods=7, is_crypto=True)['status'])
   "
   ```
5. Restart the app so charts regenerate under the new model (the chart cache keys
   on model params, so nothing stale survives, but the running process holds the
   old config in memory).
6. Report to the user: winning config per asset class, the QLIKE improvement over
   the previous default, and the new `vol_scale`.

---

## What the sweep covers

| Dimension | Values |
|---|---|
| Families | `ARCH`, `GARCH`, `GJR-GARCH` (GARCH with `o>0`), `EGARCH`, `APARCH`, `FIGARCH`, `HARCH` |
| Orders | p ∈ {1,2,3}, o ∈ {0,1}, q ∈ {0,1,2}, per family's valid combinations |
| Distributions | `normal`, `t`, `skewt`, `ged` |
| Training window | 2-year and 5-year |

= **160 configurations**, each evaluated on every ticker over 72 walk-forward windows.

The search space lives in `MODEL_STRUCTURES` / `DISTRIBUTIONS` in
`garch/garch_backtest.py`. Widen it there, not by hand-editing config.

## Methodology

### Why QLIKE, and not MAPE or BIC

The forecast **price path** is driven entirely by historical-mean drift, which does
not depend on the volatility model. So MAE / MAPE / RMSE / direction-accuracy are
*byte-identical for every config* and cannot rank them — an earlier version of this
skill ranked on them and produced meaningless ties across all candidates. Only the
**forecast variance** depends on the model, so that is what gets scored.

BIC is also the wrong objective: it measures in-sample likelihood fit with a
complexity penalty, not out-of-sample accuracy. It is kept as a secondary diagnostic.

The objective is QLIKE (Patton 2011), the standard robust loss for volatility
forecasting:

```
QLIKE = mean( ln(sigma2_forecast) + r2_actual / sigma2_forecast )
```

It stays a proper scoring rule even though squared returns are a very noisy proxy
for true variance — exactly the regime here, with 7-day forecast windows. Lower is
better.

### Optimal vol_scale in closed form

`vol_scale` c rescales sigma → c·sigma, so sigma² → c²·sigma². Substituting:

```
QLIKE(c) = 2·ln(c) + mean(ln sigma2) + (1/c^2)·mean(r2 / sigma2)
d/dc = 0  =>  c* = sqrt( mean(r2 / sigma2) )
```

Every config is therefore scored **at its own best calibration** rather than at an
arbitrary fixed 0.8×, which makes the comparison fair and yields the recommended
`vol_scale` as a by-product. At the optimum,
`QLIKE* = 2·ln(c*) + mean(ln sigma2) + 1`.

The applied value is the **median across tickers** (robust to outliers); the
per-ticker range is recorded in provenance so a config needing wildly different
scales per ticker is visible.

### Bias ratio

Measured in **variance space**: `sqrt( mean(sigma2_forecast) / mean(r2_actual) )`.
Comparing `mean(sigma)` to `mean(|r|)` would not be apples-to-apples — for a normal,
`E|r| = sigma·sqrt(2/pi) ≈ 0.8·sigma`, so even a perfect forecast would look ~1.25×
biased.

### Guarding against selection overfitting

This is the part that keeps the recommendation honest. Picking the best of 160
candidates on a single dataset means the winner is partly winning **by luck** —
with 160 draws, some config will look good by chance alone. A raw point estimate
cannot tell that apart from a real edge. Three safeguards:

**1. Held-out validation split.** Windows are ordered most-recent-first, so the
sweep selects on the *older* windows and holds out the most recent
`--validation-frac` (default 25%) entirely. The winner is chosen without ever
seeing that data, then must prove itself on it.

Crucially the `vol_scale` is fitted on the selection windows and applied
*unchanged* to validation. Refitting it there would leak held-out information
into the very number being recommended.

**2. Diebold-Mariano significance test.** Each candidate's per-observation QLIKE
losses on the held-out set are tested against the **incumbent production config**
(not an arbitrary baseline) — so the question asked is "is this actually better
than what we currently ship?" The test uses a Newey-West HAC standard error,
because consecutive forecast windows overlap and their loss differentials are
autocorrelated; a naive standard error would be too small and would make noise
look significant, which is precisely the failure being guarded against.

**3. Parsimony tie-break.** Among candidates that are statistically tied with the
best survivor, the one with the fewest free parameters wins. A simpler model is
the safer default and is less prone to the same overfitting.

**Decision rule** — config only changes when a candidate:
- beats the incumbent on **held-out** data, **and**
- does so with DM `p < --alpha` (default 0.05).

If nothing clears that bar, the run **keeps the incumbent** and says so. "No
significant improvement" is a valid, expected outcome — not a failure.

Additional guards:
- A config is only eligible if it scored on **every** ticker. One that silently
  fails to converge on part of the universe is not a usable production default.
- EGARCH and APARCH have no closed-form multi-step forecast and use simulation with
  a **fixed seed** (`SIM_SEED`), so the ranking is reproducible run to run.

## How --apply writes config

`garch/garch_config.py` contains a managed block:

```
# --- BEGIN CALIBRATED DEFAULTS (managed by the calibrate-model skill) ---
...
# --- END CALIBRATED DEFAULTS ---
```

`--apply` regenerates that whole block and reloads the module to prove the result
still imports. It writes, per asset class:

| Written constant | Source |
|---|---|
| `DEFAULT_VOL_MODEL` / `DEFAULT_CRYPTO_VOL_MODEL` | winning family |
| `DEFAULT_GARCH_P` / `_O` / `_Q` (+ crypto equivalents) | winning order |
| `DEFAULT_GARCH_DIST` / `DEFAULT_CRYPTO_GARCH_DIST` | winning distribution |
| `DEFAULT_VOL_SCALE` / `DEFAULT_CRYPTO_VOL_SCALE` | median optimal vol_scale |
| `GARCH_TRAINING_DAYS` / `CRYPTO_GARCH_TRAINING_DAYS` | winning window × 365 |
| `CALIBRATION_PROVENANCE` | date, QLIKE, ranks, ticker count, scale range |

An asset class not included in the run keeps its current values — calibrating only
crypto never clobbers the stock settings.

Everything outside the markers (valid ranges, UI limits, forecast-horizon maps) is
hand-maintained and left alone. `garch/garch_model.py` reads all defaults through
`get_model_defaults()`, so no code change is needed for a new winner as long as its
family is in `VALID_VOL_MODELS` and its order in `VALID_GARCH_ORDERS` — widen those
lists if the sweep space is widened.

## Metrics reference

| Metric | Meaning | Good value |
|---|---|---|
| `selQLIKE` | QLIKE on the selection windows — what ranking is done on | lower |
| `valQLIKE` | QLIKE on **held-out** windows — the honest number | lower |
| `DM p` | Diebold-Mariano p-value vs the incumbent, on held-out data | `< 0.05` to justify a change |
| `np` | Free parameter count — the parsimony tie-break | lower |
| `vol_scale` | Multiplier best matching forecast to realized vol | reported, not judged |
| `bias` | `sqrt(mean(sigma2)/mean(r2))` — over/under-forecast in variance space | 1.0 |
| `wstRk` | Worst per-ticker rank — catches erratic configs | lower |
| `BIC` | In-sample fit (secondary diagnostic only) | lower |

**Direction accuracy** is reported by the legacy single-model path only. It has sat
at ~50% (coin flip) in every run, on every ticker — historical-mean drift carries no
directional edge at this horizon. It is deliberately not part of model selection.

## Legacy single-model backtest

For inspecting one config in detail, window by window:

```bash
python3 garch/garch_backtest.py --ticker NVDA --windows 36 \
    --p 1 --q 1 --vol-model garch --vol-scale 0.8 --csv /tmp/nvda.csv
```

## Implementation notes

1. **Cache busting:** every chart parameter (vol_scale, order, vol model) is in the
   chart cache key, so no stale chart survives a recalibration.
2. **Forward forecast only:** `vol_scale` applies only to the forward-looking
   `forecasted_volatility`, never to in-sample `current_volatility`.
3. **Drift is not calibrated:** ~50% direction accuracy means there is nothing to
   calibrate. Damping drift toward zero over longer horizons remains an open idea.
4. **Stocks and crypto are independent:** separate family, order, distribution,
   training window and vol_scale. Crypto trades 7 days a week and has fatter tails;
   there is no reason to expect a shared optimum.

## See also

- `garch/garch_backtest.py` — sweep engine (`run_calibration`), applier (`apply_recommendations`), legacy backtest
- `garch/garch_config.py` — all defaults; the managed block is this skill's output
- `garch/garch_model.py` — production forecaster; reads defaults via `get_model_defaults()`
- `calibration_results/` — dated JSON of every sweep, for diffing runs
- `logs/tickerwatcher-{date}.log` — full ranking, logged on every run
