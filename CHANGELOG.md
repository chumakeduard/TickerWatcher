# TickerWatcher - Bug Fix & Feature Log

Running log of every bug found, fixed, and feature added, in chronological order. See `CLAUDE.md` for architecture/implementation details and `README.md` for user-facing docs.

---

## 2026-08-28 (Model Calibration Rewrite, `garch` Package, Production Hardening)

### Fix: the model-selection sweep was ranking on metrics that could not rank anything

The previous `--sweep` compared 8 configs and reported the *same* `MAPE: 1.75%` for
every one of them. That was not a coincidence — it was the bug. The forecast price
path is built purely from historical-mean drift, which has no dependence on the
volatility model, so MAE / MAPE / RMSE / direction-accuracy are **byte-identical for
every config by construction**. Ranking on them is meaningless; the tie-break then
fell to BIC, which measures in-sample fit rather than forecast accuracy.

Seven flaws found in total:

| # | Flaw | Consequence |
|---|---|---|
| 1 | Ranked on drift-only metrics | identical scores for all configs |
| 2 | Tie-broke on BIC | optimised in-sample fit, not forecasting |
| 3 | `vol_scale` frozen, never searched | configs judged at an arbitrary calibration |
| 4 | 8 of ~80 available configs | no GJR-GARCH, APARCH, FIGARCH, HARCH, ARCH, and no fat-tailed distributions |
| 5 | Swept a single ticker | recommendation overfit to one name |
| 6 | Re-queried SQLite per (config × window) | ~55k redundant queries |
| 7 | `np.std` of 7 points as the vol target | very noisy, and biased low via `ddof=0` |

**Rewritten as a proper calibration engine:**
- **160 configurations** — every `arch` family (ARCH / GARCH / GJR-GARCH / EGARCH /
  APARCH / FIGARCH / HARCH) × orders × 4 error distributions (normal / t / skewt /
  ged) × 2 training windows. Each verified to both fit *and* produce finite positive
  multi-step forecasts before inclusion.
- **QLIKE objective** (Patton 2011) — a proper scoring rule for volatility that stays
  valid when squared returns are a noisy variance proxy, which is exactly this regime.
- **`vol_scale` solved in closed form.** Scaling σ → c·σ gives
  `QLIKE(c) = 2ln(c) + mean(ln σ²) + (1/c²)·mean(r²/σ²)`, so `c* = sqrt(mean(r²/σ²))`.
  Every config is scored at *its own* optimum, making the comparison fair and
  producing the recommended `vol_scale` as a by-product.
- Runs **all tickers**, parallel across cores, fetching each window's data once.
- Bias measured in **variance space** — comparing `mean(σ)` to `mean(|r|)` is not
  apples-to-apples, since `E|r| = σ·sqrt(2/π) ≈ 0.8σ` makes a perfect forecast look
  1.25× biased.

### Fix: selection overfitting made the "winner" untrustworthy

Choosing the best of 160 candidates on one dataset means the winner is partly winning
by chance; a point estimate cannot distinguish that from a real edge. An early
small-sample run had ARCH(3) beating GARCH(1,1), which was almost certainly noise.

- **Held-out validation split** — selection happens on older windows; the most recent
  25% are held out entirely. `vol_scale` is fitted on selection and applied
  *unchanged* to validation, since refitting there would leak held-out data into the
  recommended number.
- **Diebold-Mariano test** against the **incumbent production config**, with a
  Newey-West HAC standard error. The HAC part matters: overlapping forecast windows
  autocorrelate the loss differentials, and a naive standard error would be too small
  — making noise look significant, the exact failure being guarded against.
- **Parsimony tie-break** among statistically tied survivors.
- **Decision rule:** config changes only if a candidate beats the incumbent on
  held-out data *and* clears DM `p < 0.05`. Otherwise the incumbent is kept and the
  run says so — "no significant improvement" is a valid outcome.

### Feature: the calibration now writes its own config

`garch_config.py` gained a machine-managed block between `BEGIN/END CALIBRATED
DEFAULTS` markers. `--apply` regenerates it and reloads the module to prove the
result still imports. An asset class absent from a run keeps its current values, so
calibrating only crypto never clobbers stocks. Provenance (date, QLIKE, DM p,
ticker count, `vol_scale` range) is recorded alongside.

The config schema previously could not *express* a calibration result — no error
distribution, no asymmetry order on GARCH, no per-asset model family. Added those,
plus `get_model_defaults()`, and generalized `garch_model.py` from a hardcoded
`garch`/`egarch` if-else to all 6 families × 4 distributions (all 12 spot-checked
combinations verified to serve).

### Refactor: `garch/` package

`backtest_garch.py` → `garch/garch_backtest.py`; `garch_config.py` and
`garch_model.py` moved alongside, with `__init__.py` re-exporting the common entry
points. Imports updated across `app.py` and `draw.py`. Runnable both as a package
and as a direct script.

### Production hardening

- **`requirements.txt` was missing `arch` entirely** — the core forecasting
  dependency. A fresh `pip install -r requirements.txt` produced an app where
  `ARCH_AVAILABLE = False` and all volatility forecasting silently did nothing.
  Added, along with `scipy`; floors realigned to the versions actually tested against.
- **`sys.exit(1)` inside `draw.py:get_ticker_data()`** — this runs inside Flask
  request handling, and `SystemExit` derives from `BaseException`, so the
  `except Exception` in `ensure_chart_exists()` would *not* catch it: an unknown
  ticker would tear down the worker instead of returning an error. Now raises
  `NoDataError`.
- `print()` in library code (`garch_model.py`, `draw.py`, `app.py`) replaced with
  module loggers.
- **Test suite added** (`tests/`, 50 tests) — none existed. Covers the calibration
  maths against known-answer cases (`optimal_scale` recovers a planted 1.5× bias;
  QLIKE optimum verified against neighbours and against its analytic identity), DM
  false-positive rate under the null, every family × distribution fitting and
  forecasting, config/​model contract integrity, and the config write-back round-trip.
- `.gitignore` written; **`database/prices.db` and `logs/` were tracked in git** and
  are now untracked (files preserved on disk).
- `pyproject.toml` added with pytest config; `requirements-dev.txt` split out.

**Files:** `garch/garch_backtest.py`, `garch/garch_config.py`, `garch/garch_model.py`,
`garch/__init__.py`, `tests/`, `app.py`, `draw.py`, `pyproject.toml`, `.gitignore`,
`requirements.txt`, `requirements-dev.txt`, `.claude/skills/calibrate-model/SKILL.md`.

---

## 2026-08-28 (Cryptocurrency Support)

### Feature: Crypto Monitoring (BTC, ETH)

Added cryptocurrency price tracking and GARCH forecasting alongside stocks, with its own database table, GARCH configuration, and a sidebar tab to switch between the two.

**Database (`db.py`):** New `crypto_prices` table (same schema as `prices`, separate for data isolation). All read/write helpers (`get_last_price_date()`, `fetch_and_store_prices()`, `get_stats()`, `needs_backfill()`) take an `is_crypto` flag to pick the right table. Crypto tickers are fetched from Yahoo Finance as `<TICKER>-USD`.

**Refresh (`refresh.py`):** `update_crypto()` mirrors `update_ticker()` — delta sync (only fetches the gap between the last stored date and today) plus a full backfill on first run. Crypto keeps **full history** (as far back as Yahoo Finance has it — BTC-USD since 2014-09-17, ETH-USD since 2017-11-09), the same retention policy as stocks; an earlier version of this that pruned crypto to a rolling 20-day window was abandoned once it became clear it made the period selector (`1Y`/`5Y`/`MAX`) meaningless for crypto (see bug below).

**GARCH config (`garch_config.py`):** Crypto gets its own `CRYPTO_MIN_YEARS` (15, vs stocks' `MIN_YEARS` 50 — plenty for any crypto's actual listing history) and `DEFAULT_CRYPTO_VOL_SCALE` (0.9×, vs stocks' 0.8×, separately configurable). Training window (`CRYPTO_GARCH_TRAINING_DAYS`) and forecast-day mapping now match stocks' 5-year window; `CRYPTO_FORECAST_DAYS_BY_PERIOD` stays deliberately shorter (2-14 days vs stocks' 3-21) as a forecast-horizon choice, not a data limitation.

**App routing (`app.py`):** New `?asset_type=stock|crypto` query param threads through chart generation, caching (cache keys include asset type), and the sidebar. `/api/garch/<ticker>` and `/api/garch-stats/<ticker>` now accept crypto tickers (auto-detected, or via `?is_crypto=true`).

**UI (`templates/chart_sidebar.html`):** Added a "📈 Stocks" / "🪙 Crypto" tab pair above the ticker list; switching tabs swaps the ticker list and all navigation links carry `asset_type` forward (period buttons, ticker links, chart-control checkboxes).

### Bugs found and fixed during crypto rollout

1. **Period selector for crypto only ever showed ~20-25 days, no matter which period was picked.** Root cause was the (later abandoned) 20-day retention window — `1Y`/`5Y`/`MAX` all silently rendered the same handful of days. Fixed by switching crypto to full-history retention (see above), which also required reverting several crypto-specific shortcuts (GARCH min-training-data floor, historical-prediction fit window/step, SQL row limits) back to stock-identical values now that the data volume is comparable.

2. **Historical-predictions overlay and buy/sell-signal checkboxes appeared to do nothing for crypto.** The overlay's `len(dates) > 50` gate and the internal GARCH fit's `min_train = 50` requirement were both unreachable with only ~20-25 crypto rows on hand. (Buy/sell signals were actually working correctly all along — with the default 10% profit target, BTC's simulated path just didn't cross it within a short forecast window; confirmed by testing with a lower target.)

3. **No tooltips on crypto charts.** `get_chart_data_for_tooltip()` always queried the `prices` table, never `crypto_prices`, so the tooltip cache was silently empty for every crypto chart. Added an `is_crypto` parameter.

4. **Pre-existing bug, not crypto-specific: `update_ticker()`/`update_crypto()` in `refresh.py` returned early on "already up to date" *before* ever checking `needs_backfill()`.** A ticker current on recent data but short on backfilled history (e.g. right after being added) would never get backfilled past its first, possibly partial, fetch. Also, the backfill fetch used `get_last_price_date()` (MAX date) as its end boundary instead of the true earliest stored date — added `get_earliest_price_date()` (MIN date) in `db.py` and fixed the boundary. This is what let BTC/ETH actually backfill to their full 2014/2017 history once the retention policy changed.

5. **Legend covering candlesticks.** The "Predicted OHLC (historical)" / "Predicted trend (smoothed)" legend for the historical-predictions overlay sat at `loc='lower right'` *inside* the price chart, on top of the candles. Moved it to a figure-level legend anchored just below the price chart's x-axis (in the existing gridspec gap before the volume panel) — discovered along the way that an Axes-level legend with a `bbox_to_anchor` point outside that axes still gets silently clipped to the axes' box, so the fix had to switch to `fig.legend()`.

**Files touched:** `db.py`, `config.py`, `refresh.py`, `garch_config.py`, `garch_model.py`, `draw.py`, `app.py`, `templates/chart_sidebar.html`.

---

## 2026-08-26 (Buy/Sell Trading Signals & Profit Targets)

### Feature: Profit Target Marking

Added a "Profit %" slider (default 10%, range 0.1-50%) that marks the forecast prices where a profit % target is first achievable.

**Visual elements:**
- Green horizontal dashed line at upside target: `last_close × (1 + profit_pct/100)` — "sell for profit" opportunity
- Red horizontal dashed line at downside target: `last_close × (1 - profit_pct/100)` — "short/buy for profit" opportunity
- Vertical dotted lines marking the exact forecast candle that first hits each target
- Labels showing date, target price, and profit %
- Only marked in the forecast region (right of "today" boundary)

**Implementation:**
- Slider in Chart Controls sidebar, settable via `?profit_pct=X` query string
- Included in cache key so chart regenerates when changed
- Uses candle high/low (not just close) to detect target hit

**Files:** `garch_config.py` (new constants), `app.py`, `templates/chart_sidebar.html`, `draw.py`.

---

### Feature: Buy/Sell Trading Signals (Alternating Cycle)

Added a "Show Buy/Sell Signals" checkbox that labels every forecast candle with alternating BUY and SELL signals, where each consecutive pair achieves the configured profit % target.

**Algorithm (state machine):**
1. Start looking for a BUY: price must drop ≥profit_pct from the reference price
2. Once BUY found, switch to looking for SELL: price must rise ≥profit_pct from the BUY
3. Once SELL found, switch back to looking for BUY
4. Continue alternating through entire forecast: BUY → SELL → BUY → SELL...

**Visual labels:**
- Green "BUY" labels (positioned below candles) when entry point is hit
  - Shows date and price
- Red "SELL" labels (positioned above candles) when exit point is hit
  - Shows date, price, and actual profit % achieved
- Each label has a colored box matching the signal type
- Small, compact formatting to avoid crowding the chart

**Features:**
- Checkbox in Chart Controls, settable via `?show_signals=1`
- Strict alternating pattern — no consecutive BUYs or SELLs
- Each signal represents a real trading opportunity
- Actual profit % calculated and displayed on SELL signals
- Works with any profit_pct value

**Files:** `app.py`, `templates/chart_sidebar.html`, `draw.py`.

---

### Cache Key Updates

Updated cache key format to include `profit_pct` and `show_signals` parameters:
- Old: `{ticker}_{period}_{grouping}_th{threshold:.1f}_fd{forecast_days}_p{p}q{q}_{vol_model}_vs{vol_scale:.2f}`
- New: `{ticker}_{period}_{grouping}_th{threshold:.1f}_fd{forecast_days}_pr{profit_pct:.1f}_p{p}q{q}_{vol_model}_vs{vol_scale:.2f}_{hist_suffix}{sig_suffix}`

This ensures charts are regenerated when these parameters change, not served stale from cache.

**File:** `app.py` (`get_chart_key()`).

---

## 2026-08-26 (Historical Predictions Overlay & Config Refactor)

### Feature: Historical Predictions Overlay

Added a toggle ("Show Historical Predictions" checkbox, or `?show_historical=1`) that overlays what the model would have predicted at each point in the past directly on the real candlesticks — a visual complement to `backtest_garch.py`'s numeric output.

**What it draws (all blue, the unified "prediction" color):**
- Predicted OHLC candlesticks for every historical day, semi-transparent, overlaid on the real candles at the same index
- A smoothed trend line (dotted) — a moving average over the raw predicted closes — that continues seamlessly across the "today" boundary into the future forecast, so past and future predictions read as one line
- The future forecast candlesticks (previously light green/red) are now also solid blue, unifying "predicted" styling everywhere

**Files:** `draw.py` (new `compute_historical_predictions()`), `app.py`, `templates/chart_sidebar.html`.

---

### Bug fix: Historical predictions "cheated" using real outcomes

**Symptom:** historical prediction candles looked flat/unrealistic — thin bodies with disproportionately long wicks — visually inconsistent with the future forecast's more naturally-proportioned candles.

**Root cause:** the first implementation computed historical predicted closes with pure deterministic drift compounding (`close = prev_close × (1 + drift)`, no randomness), while the future forecast uses a genuine random walk (`drift + Gaussian shock sized by GARCH volatility`). Since historical drift is tiny, the candle body was nearly flat, and the entire GARCH volatility magnitude got dumped into an oversized wick instead.

**Fix:** `compute_historical_predictions()` now uses the *exact same* random-walk formula as the real forecast — re-anchored to the actual close only at the **start** of each rolling fit window (the one piece of information a walk-forward backtest is allowed to know), then simulated exactly as blindly as the future forecast, using the same OHLC jitter formula (`±0.3×`/`±0.5×` day_vol). Each window's RNG is seeded deterministically on `(ticker, window_start_date)` for reproducibility across reloads.

**File:** `draw.py` (`compute_historical_predictions()`).

---

### Bug fix: "Today" line bled into the last historical candle

**Symptom:** the grey dashed line marking the boundary between real data and prediction-only appeared to run through the middle of the last real candle rather than cleanly separating the two regions.

**Root cause:** the line was drawn at `x = last_i` (the candle's center x-position), so half the candle's body (± half its rendered width) visually crossed to the "prediction" side.

**Fix:** moved the line to `today_x = last_i + width/2 + 0.05` — just past the full right edge of the last candle's body.

**File:** `draw.py`.

---

### Bug fix: Tooltip hover date drifted increasingly off with larger `forecast_days`

**Symptom:** hovering near the "today" boundary on a chart with `forecast_days=30` reported a date almost a month earlier than the actual last date in the database.

**Root cause:** the frontend JS mapped mouse pixel position to a data index assuming the whole chart image width corresponds only to the historical candle count — but the rendered image also includes the forecast region. The larger `forecast_days` is relative to the historical window, the further this assumption drifts.

**Fix:** `draw_chart()` now returns axis metadata (`x_min`, `x_max`, `historical_count`) alongside the image bytes; `/api/chart-data/<chart_key>` exposes it as `meta`; the frontend hover handler now maps pixel position to the true matplotlib axis coordinate using that metadata instead of assuming historical-only width.

**Files:** `draw.py` (`draw_chart()` now returns a 3-tuple), `app.py`, `templates/chart_sidebar.html`.

---

### Feature: Tooltip content — full OHLC + volume + trend on both sides

Restructured the hover tooltip to show, depending on which side of the "today" line is hovered:
- **Historic side:** real OHLCV in green/red (unchanged) **+** predicted Open/High/Low/Close in blue **+** the smoothed trend value in blue
- **Forecast side:** date, full predicted Open/High/Low/Close, Volume, and the smoothed trend value — all in blue, no green/red block (nothing "actual" exists out there)

**Files:** `draw.py`, `templates/chart_sidebar.html`.

---

### Refactor: Centralized GARCH configuration (`garch_config.py`)

Extracted all previously-scattered GARCH constants (training window length, valid/default orders, valid/default vol models, vol_scale defaults and bounds, forecast-day-by-period mapping, price-move-threshold bounds) into a single new module, `garch_config.py`. `garch_model.py`, `draw.py`, `app.py`, and `backtest_garch.py` now import from it instead of hardcoding values inline.

**File:** `garch_config.py` (new), plus import updates across `garch_model.py`, `draw.py`, `app.py`, `backtest_garch.py`.

---

### Backtest re-run: filled in pending AAPL/MSFT/TSLA calibration numbers

`calibrate-model` skill's results table had "(backtest pending)" placeholders for AAPL/MSFT/TSLA. Ran the full 36-window walk-forward backtest for all three:

| Ticker | Baseline Error | Calibrated Error | Improvement |
|---|---|---|---|
| AAPL | 80.9% | 56.8% | -24.1 pts |
| MSFT | 75.4% | 59.1% | -16.3 pts |
| TSLA | 62.7% | 43.5% | -19.2 pts |

Consistent with all previously-tested tickers: vol_scale=0.8 improves volatility forecast accuracy, and direction accuracy remains ~44-56% (coin-flip) across the board.

---

## 2026-08-26 (Model Calibration & Configurability)

### Feature: Volatility Calibration Factor (vol_scale)

Walk-forward backtesting (36 windows on 7 tickers: NVDA, AAPL, MSFT, TSLA, VDE, VOO, VTI) revealed that GARCH(1,1) systematically over-forecasts realized volatility by ~20-25% at a weekly forecast horizon. Added a **calibration multiplier** (`vol_scale`, default **0.8×**) to correct this bias, reducing mean % error by 10-23 percentage points across all tested tickers.

**Implementation:**
- New parameter `vol_scale` in `garch_model.py::forecast_volatility()` (default 0.8, range 0.3–2.0)
- Applied *only* to the forward-looking `forecasted_volatility` array (not to in-sample fit)
- **UI control:** Slider in sidebar "⚙️ Chart Controls" → "Vol. Calibration" (0.3×–1.5×)
- **Query parameter support:** `?vol_scale=0.8`
- **CLI support:** `backtest_garch.py --vol-scale 0.8`
- **Cache key includes vol_scale** to prevent stale-chart bugs

**Backtest results** (mean % volatility forecast error, lower is better):
| Ticker | Baseline (1.0×) | Calibrated (0.8×) | Improvement |
|---|---|---|---|
| NVDA | 82.5% | 59.4% | -23.1 pts |
| VDE | 41.1% | 31.5% | -9.6 pts |
| VOO | 62.1% | 41.7% | -20.4 pts |
| VTI | 60.7% | 40.7% | -20.0 pts |

**Files:** `garch_model.py`, `draw.py`, `app.py`, `templates/chart_sidebar.html`, `backtest_garch.py`.

---

### Feature: GARCH Order Configurability

Added ability to select between GARCH model orders (1,1)/(1,2)/(2,1)/(2,2), with (1,1) as the default (backtested as best AIC/BIC fit).

**Implementation:**
- New parameters `p`, `q` in `garch_model.py::fit_garch()`, `forecast_volatility()`, `get_garch_stats()`
- **UI control:** Dropdown in sidebar "⚙️ Chart Controls" → "GARCH Order (p,q)" with "default, best AIC" label on (1,1)
- **Query parameter support:** `?garch_p=1&garch_q=2`
- **Cache key includes p,q** to regenerate chart when order changes

**Files:** `garch_model.py`, `draw.py`, `app.py`, `templates/chart_sidebar.html`.

---

### Feature: Volatility Model Selection (GARCH vs EGARCH)

Added ability to select between symmetric GARCH (default, stable) and asymmetric EGARCH (experimental, captures leverage effect but numerically fragile on short windows).

**Implementation:**
- New parameter `vol_model` in `garch_model.py::fit_garch()`, `forecast_volatility()`, `get_garch_stats()`
- GARCH model automatically selects forecast method: `'analytic'` for GARCH, `'simulation'` for EGARCH (multi-step)
- **UI control:** Dropdown in sidebar "⚙️ Chart Controls" → "Volatility Model" with labels: "GARCH — default, stable" vs "EGARCH — asymmetric, experimental"
- **Query parameter support:** `?vol_model=garch` or `?vol_model=egarch`
- **Cache key includes vol_model** to regenerate chart when model type changes

**Backtesting findings:**
- EGARCH fails to converge on short training windows (1-year data), produces `ConvergenceWarning` and inflated/NaN forecasts
- EGARCH only marginally better AIC than GARCH on long windows
- No improvement in accuracy at weekly forecast horizon
- Recommendation: use GARCH for production, keep EGARCH for research

**Files:** `garch_model.py`, `draw.py`, `app.py`, `templates/chart_sidebar.html`.

---

### Feature: Fitted Coefficients Display

GARCH model's fitted coefficients (μ, ω, α, β, γ) are now extracted and displayed in the stats panel, allowing users to inspect model behavior and inform tuning decisions.

**Implementation:**
- New function `garch_model.py::extract_coefficients()` — extracts fitted parameters from arch model
- Updated `forecast_volatility()` and `get_garch_stats()` to include `coefficients` dict in return
- **UI display:** Stats panel below volatility metrics shows fitted coefficients with labels
- Coefficient values rounded to 4 decimal places

**Files:** `garch_model.py`, `app.py`, `templates/chart_sidebar.html`.

---

### Feature: Project-Level Skill "calibrate-model"

Created a project-level Claude skill at `.claude/skills/calibrate-model/SKILL.md` that encapsulates the walk-forward backtesting process. Users can now trigger model calibration with voice/text phrases like "calibrate the model" or "run calibration".

**What it does:**
- Runs 36-window backtests on a set of test tickers (VDE, VOO, VTI, NVDA, AAPL, MSFT, TSLA)
- Tests both baseline (vol_scale=1.0) and calibrated (vol_scale=0.8) models
- Generates comparison table and metrics
- Suggests adjustments to `DEFAULT_VOL_SCALE` if needed
- Updates documentation with findings

**Skill file:** `.claude/skills/calibrate-model/SKILL.md`.

---

### Documentation Updates

Updated `CLAUDE.md`, `README.md`, and created comprehensive skill documentation to reflect:
- Volatility calibration methodology and backtest results
- GARCH order and vol_model selection rationale
- Fitted coefficients and their interpretation
- Cache key design (now includes all parametric changes)
- Walk-forward backtest findings across 7 tickers

---

## 2026-08-26

### Bug: Forecast predictions statistically identical across time periods

**Symptom:** Switching a single ticker between period selections (1D→3-day forecast, 5D→5-day, 1M/3M/6M→14-day) produced predictions that "look very matching" — the only apparent difference was length, not shape or character.

**Root causes** (all in the forecast-generation path):
1. `np.random.seed(42)` was set globally at `draw.py` module-import time, pinning the entire app's random stream to one deterministic sequence — forecasts replayed identically across app restarts.
2. **The core bug:** `garch_model.forecast_volatility()` already computed a genuine per-day volatility array (`forecasted_volatility` — the actual GARCH multi-step term structure, which evolves toward the long-run variance over the horizon), but `draw.py` discarded it and reused a single flat scalar (`current_volatility`) for *every* simulated day. A 3-day and a 21-day forecast therefore used identically-sized daily noise; forecast length was the only thing that varied.
3. Zero drift: `np.random.normal(0, ...)` had mean 0. The historical mean return was computed elsewhere (`get_garch_stats`) but never fed into the forecast, so there was no directional signal — just noise centered on the last close.
4. The candle-body (OHLC) generator repeated bug #2 — every forecast day's wick size used the same flat scalar, so displayed uncertainty never widened further into the horizon.

**Fix:**
- Removed the global `np.random.seed(42)`.
- `garch_model.forecast_volatility()` now also returns `returns_mean` (historical daily mean log return over the same fit window).
- `draw.py`'s `draw_chart()` now consumes the actual per-day `forecasted_volatility` array (falling back to the flat value only if empty) for both the price random-walk and the OHLC candle generation, and adds the drift term to each day's step.

**Files:** `garch_model.py` (`forecast_volatility()`), `draw.py` (`draw_chart()`).

**Caveat (not a bug, a limitation):** all periods still fit the GARCH model on the same fixed 3-year window and start from the same last close, so a 3-day forecast will still resemble the first 3 days of a fresh 21-day forecast in general character — the noise is no longer flat/identical, and drift is now real, but don't expect a wildly different "shape" per period for the same ticker. Inherent to using one model fit for all horizons.

---

### Bug (follow-on, exposed by the fix above): Forecast candles clipped off-screen

**Symptom:** After adding real drift, some forecast candlesticks for 5D/short-history charts disappeared from the chart entirely (only their volume bar remained visible).

**Root cause:** `ax_chart.set_ylim(min(lows) * 0.98, max(highs) * 1.02)` sized the y-axis using only *historical* highs/lows. With real drift now applied, forecast prices could move outside that historical range (e.g. day-1 forecast of $318.64 when the historical high was $313.36) and got silently clipped by the axis limits.

**Fix:** Forecast highs/lows are now collected while generating forecast candles and folded into the y-axis range: `y_low = min(lows + forecast_lows)`, `y_high = max(highs + forecast_highs)`.

**File:** `draw.py` (`draw_chart()`).

---

### Feature added: "🔄 Refresh Data" button

Not a bug fix, but logged here for continuity: added a one-click button at the bottom of the sidebar that triggers `refresh.py`'s update logic in a background thread (`POST /api/refresh`, polled via `GET /api/refresh/status`), clears the chart cache on success, and the page auto-reloads. See `CLAUDE.md` → "Refresh Data Button" for full details.

---

## 2026-08-25

### Bug: `/api/garch-stats/<ticker>` returned HTTP 500

**Symptom:** `curl http://localhost:8080/api/garch-stats/AAPL` → `AttributeError: 'numpy.ndarray' object has no attribute 'values'`.

**Root cause:** `model.conditional_volatility` (and `forecast.variance`) from the `arch` library return plain `numpy.ndarray` in this version, not a pandas Series/DataFrame — code assumed `.values` / `.iloc` accessors existed unconditionally.

**Fix:** Removed the `.values`/`.iloc` assumptions; index/aggregate the ndarray directly (`model.conditional_volatility[-1]`, `np.mean(...)`, `np.max(...)`, `np.min(...)`).

**Files:** `garch_model.py` (`forecast_volatility()`, `get_garch_stats()`).

---

### Bug: `arch` library not installed

**Symptom:** GARCH endpoints returned `{"status": "unavailable", "message": "GARCH model requires: pip install arch"}`.

**Fix:** `pip install arch`.

---

### Bug: Flask wouldn't bind to port 5000

**Symptom:** `Address already in use` / `Port 5000 is in use by another program` on macOS.

**Root cause:** macOS `ControlCenter` (AirPlay Receiver) squats on port 5000 by default.

**Fix:** Switched the app's default port to 8080 (`PORT` env var still overrides). Documented as a known macOS gotcha rather than something to "fix" system-side.

**File:** `app.py` (`if __name__ == '__main__':` block).

---

## Earlier (pre-summary, exact dates not recorded)

### Bug: Clicking any button crashed the app

**Root cause:** `<button>` elements nested inside `<a>` tags — invalid HTML that some browsers mis-render/mis-handle on click.

**Fix:** Replaced with `onclick` handlers on plain buttons/divs instead of button-in-anchor nesting.

---

### Bug: Matplotlib crashed on macOS when serving charts via Flask

**Root cause:** Matplotlib's default GUI backend isn't safe to use from a non-main thread / server context on macOS.

**Fix:** `matplotlib.use('Agg')` set before importing `pyplot` — forces the non-GUI, thread-safe rendering backend.

**File:** `draw.py` (top of file).

---

### Bug: Chart grouping selector (daily/weekly/monthly) had no effect

**Root cause:** Native `<select>` dropdown's change events weren't firing reliably across browsers in the way the page expected.

**Fix:** Replaced the native dropdown with button-based selectors plus a polling mechanism (`setInterval` checking for value changes every 200ms) to reliably detect and apply the selection.

**File:** `templates/chart.html` (later superseded by `chart_sidebar.html`; grouping selector itself was subsequently removed from the UI entirely on 2026-08-25 per user request — see below).

---

### Bug: Chart URLs broke due to quote-escaping in Jinja2 templates

**Root cause:** `url_for()` calls embedded inside `onchange`/`onclick` attribute strings had inconsistent quote nesting (double quotes inside double-quoted attributes).

**Fix:** Corrected quoting in the Jinja2 templates so generated attribute strings are valid HTML.

---

### Bug: Charts served incorrectly / inconsistently

**Root cause:** Charts were being written to and read from disk (`images/` folder), which introduced staleness and path issues.

**Fix:** Switched to in-memory caching — `draw_chart()` returns PNG bytes directly (`BytesIO`), stored in `chart_cache` dict keyed by `{ticker}_{period}_{grouping}`, served via `/chart-image/<chart_key>`. No disk writes for chart images.

**Files:** `draw.py`, `app.py`.

---

## Non-bug UI change (logged for context)

### Grouping selector removed from UI (2026-08-25)

Per explicit user request ("totally remove grouping"), the Daily/Weekly/Monthly button group was removed from the sidebar. Charts now always render with `grouping='daily'` internally; the CLI (`draw.py --grouping`) still accepts the flag for compatibility, it's just no longer exposed in the web UI.

**Files:** `templates/chart_sidebar.html`, `app.py` (`chart()` route).
