"""
pytest conftest for the spherical_eeg_sim test suite.

Adds the repository root to sys.path so that:
  - ``pytest`` works from any working directory
  - each test file is also runnable directly:
        python tests/test_forward.py
        python tests/test_sampling.py
        ...
without requiring ``pip install -e .`` or a manual ``PYTHONPATH``.
"""

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def pytest_report_collectionfinish(items):
    """Show how many distinct test functions the selected cases exercise."""
    cases = [item for item in items if isinstance(item, pytest.Function)]
    if not cases:
        return None
    functions = {(case.parent.nodeid, case.originalname) for case in cases}
    function_unit = "test function" if len(functions) == 1 else "test functions"
    case_unit = "case" if len(cases) == 1 else "cases"
    return f"Selected {len(functions):,} {function_unit} covering {len(cases):,} {case_unit}."
