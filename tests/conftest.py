"""
Pytest configuration for PIGLasso test suite.

Tests that require NODIS are skipped automatically when NODIS is not installed,
so `pytest tests/` works in any environment and only reports failures on
functionality that belongs to PIGLasso itself.
"""

from __future__ import annotations

import inspect

import pytest

_NODIS_AVAILABLE = False
try:
    import nodis  # noqa: F401
    _NODIS_AVAILABLE = True
except ImportError:
    pass


def pytest_collection_modifyitems(items):
    """Mark NODIS-dependent tests as skip when nodis is not installed."""
    if _NODIS_AVAILABLE:
        return
    skip = pytest.mark.skip(reason="nodis not installed — pip install nodis")
    for item in items:
        if _needs_nodis(item):
            item.add_marker(skip)


def _needs_nodis(item) -> bool:
    """Return True if this test item requires nodis (via fixture, direct import, or subprocess)."""
    # Tests that depend on the fitted_estimator fixture (nodis import is inside the fixture)
    if "fitted_estimator" in getattr(item, "fixturenames", []):
        return True
    # End-to-end tests exercise the full CLI stack, which always requires nodis
    if "end_to_end" in item.name:
        return True
    # Tests whose own body imports nodis
    fn = getattr(item, "function", None)
    if fn is None:
        return False
    try:
        return "nodis" in inspect.getsource(fn)
    except (OSError, TypeError):
        return False
