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

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
