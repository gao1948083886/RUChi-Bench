"""Pytest conftest for tests/unit/metrics."""
import pytest
from pathlib import Path


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_makereport(item, call):
    """Add debug info to test reports."""
    pass


def pytest_runtest_setup(item):
    """Debug setup."""
    # Check if Path's __truediv__ is working
    p = Path("/tmp")
    result = p / "test.txt"
    print(f"\n>>> Path / str type: {type(result)}")
    print(f">>> Path / str has open: {hasattr(result, 'open')}")
