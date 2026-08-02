"""
Three-layer verification strategy:
  Layer 1 (forward_check)      : internal algebraic and numerical checks
  Layer 2 (distribution_check) : agreement with configured bounds and moments
  Layer 3 (test suite)         : byte-identical reruns with the same seed

The first two layers are implementation checks, not independent physical
validation. Independent evidence is documented separately.
"""

from .forward_check import (
    validate_forward_model,
    ForwardValidationResult,
)
from .distribution_check import (
    validate_distributions,
    DistributionValidationResult,
    CheckResult,
)

__all__ = [
    'validate_forward_model',
    'ForwardValidationResult',
    'validate_distributions',
    'DistributionValidationResult',
    'CheckResult',
]
