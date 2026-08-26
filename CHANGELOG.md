# TickerWatcher - Bug Fix Log

Running log of every bug found and fixed, in chronological order. See `CLAUDE.md` for architecture/implementation details and `README.md` for user-facing docs.

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
