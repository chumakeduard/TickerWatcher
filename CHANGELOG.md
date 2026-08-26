# TickerWatcher - Bug Fix & Feature Log

Running log of every bug found, fixed, and feature added, in chronological order. See `CLAUDE.md` for architecture/implementation details and `README.md` for user-facing docs.

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
