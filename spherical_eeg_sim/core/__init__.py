"""
Core physics: 4-layer spherical forward model and electrode handling.
"""

from .forward import (
    compute_transfer_coefficients,
    forward_potential,
    forward_potential_basis,
    forward_potential_batch,
    potentials_from_basis,
)
from .electrodes import (
    ElectrodeArray,
    load_electrodes,
    save_electrodes_csv,
    generate_standard_montage,
    validate_electrodes,
    resolve_electrode_path,
    resolve_reference,
    apply_reference,
)

__all__ = [
    'compute_transfer_coefficients',
    'forward_potential',
    'forward_potential_basis',
    'forward_potential_batch',
    'potentials_from_basis',
    'ElectrodeArray',
    'load_electrodes',
    'save_electrodes_csv',
    'generate_standard_montage',
    'validate_electrodes',
    'resolve_electrode_path',
    'resolve_reference',
    'apply_reference',
]
