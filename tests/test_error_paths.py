"""Regression tests for the production error paths.

Both cases here were live bugs:

  * `get_ticker_data` used to call `sys.exit(1)`. It runs inside Flask request
    handling, and SystemExit derives from BaseException, so the `except Exception`
    in `ensure_chart_exists` did not catch it — an unknown ticker tore down the
    worker instead of returning an error page.
  * `requirements.txt` omitted `arch`, the core forecasting dependency, so a fresh
    install produced an app where ARCH_AVAILABLE was False and volatility
    forecasting silently did nothing.
"""

import ast
import pathlib
import sqlite3

import pytest

import app as app_module
from draw import NoDataError, get_ticker_data

ROOT = pathlib.Path(__file__).resolve().parent.parent


# ------------------------------------------------- missing data must not exit

def test_missing_ticker_raises_instead_of_exiting():
    """The critical property: an Exception subclass, never SystemExit."""
    with pytest.raises(NoDataError):
        get_ticker_data('DEFINITELY_NOT_A_TICKER', 90)


def test_missing_ticker_error_is_catchable_as_exception():
    """`except Exception` must catch it — SystemExit would slip through."""
    try:
        get_ticker_data('DEFINITELY_NOT_A_TICKER', 90)
    except Exception as e:                      # noqa: BLE001 - that is the point
        assert isinstance(e, NoDataError)
    else:
        pytest.fail("expected NoDataError")


def test_no_sys_exit_in_request_path_modules():
    """Guard against sys.exit creeping back into code Flask calls per-request."""
    for mod in ('draw.py', 'app.py'):
        tree = ast.parse((ROOT / mod).read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            is_sys_exit = (
                isinstance(fn, ast.Attribute) and fn.attr == 'exit'
                and isinstance(fn.value, ast.Name) and fn.value.id == 'sys'
            )
            if not is_sys_exit:
                continue
            # Only permitted under `if __name__ == '__main__'`, i.e. the CLI entry.
            assert mod == 'draw.py' and node.lineno > _main_guard_line(tree), (
                f"{mod}:{node.lineno} calls sys.exit() outside the CLI entry point"
            )


def _main_guard_line(tree):
    for node in tree.body:
        if isinstance(node, ast.If) and ast.dump(node.test).find("__main__") != -1:
            return node.lineno
    return float('inf')


def test_chart_route_returns_404_not_500_for_unknown_ticker(monkeypatch):
    """Missing data is a client condition; it must not be reported as a server fault."""
    monkeypatch.setitem(app_module.app.config, 'TESTING', True)
    # Make the ticker pass whitelist validation but have no rows behind it.
    monkeypatch.setattr(app_module, 'TICKERS', list(app_module.TICKERS) + ['NODATAXY'])
    client = app_module.app.test_client()
    resp = client.get('/chart?ticker=NODATAXY&period=6M')
    assert resp.status_code == 404, f"expected 404, got {resp.status_code}"


def test_chart_route_still_serves_a_real_ticker():
    con = sqlite3.connect(__import__('db').DB_PATH)
    try:
        n = con.execute("SELECT COUNT(*) FROM prices WHERE ticker='AAPL'").fetchone()[0]
    finally:
        con.close()
    if n < 300:
        pytest.skip("no local AAPL data")
    client = app_module.app.test_client()
    assert client.get('/chart?ticker=AAPL&period=3M').status_code == 200


# ------------------------------------------------------ dependency manifest

def _declared():
    text = (ROOT / 'requirements.txt').read_text().lower()
    text += (ROOT / 'requirements-dev.txt').read_text().lower()
    return text


@pytest.mark.parametrize("dist", ['arch', 'scipy', 'numpy', 'pandas',
                                  'matplotlib', 'flask', 'yfinance'])
def test_core_dependency_is_declared(dist):
    assert dist in _declared(), f"{dist} missing from requirements"


def test_arch_is_actually_importable_and_enabled():
    """`arch` missing does not crash — it silently disables forecasting, which is
    why its absence from requirements went unnoticed. Assert it is really on."""
    from garch.garch_model import ARCH_AVAILABLE
    assert ARCH_AVAILABLE, "arch not installed: all volatility forecasting is a no-op"


def test_every_third_party_import_is_declared():
    """Catch a new undeclared dependency the moment it is introduced."""
    import sys
    import sysconfig
    import importlib.util

    stdlib_dir = sysconfig.get_paths()['stdlib']
    local = {'app', 'draw', 'db', 'refresh', 'config', 'logging_config', 'garch', 'tests'}

    def is_stdlib(mod):
        if mod in sys.builtin_module_names:
            return True
        try:
            spec = importlib.util.find_spec(mod)
        except Exception:
            return False
        if spec is None:
            return False
        if not spec.origin:
            return True
        return spec.origin == 'built-in' or (
            stdlib_dir in spec.origin and 'site-packages' not in spec.origin)

    declared = _declared()
    undeclared = set()
    for path in ROOT.rglob('*.py'):
        if '.venv' in str(path):
            continue
        try:
            tree = ast.parse(path.read_text())
        except Exception:
            continue
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name.split('.')[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                mods = [node.module.split('.')[0]]
            for m in mods:
                if m not in local and not is_stdlib(m) and m.lower() not in declared:
                    undeclared.add(m)
    assert not undeclared, f"undeclared third-party imports: {sorted(undeclared)}"
