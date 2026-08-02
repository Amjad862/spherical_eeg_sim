"""
Source sampling: positions, orientations, magnitudes, and source-mode dispatchers.
"""

from .sampling import (
    fibonacci_sphere,
    sample_cortical_positions,
    sample_subcortical_positions,
    sample_unit_vectors_s2,
    sample_orientations,
    sample_magnitudes,
    cartesian_to_spherical,
    world_orientation_to_spherical_frame,
    make_position_split,
    PositionSplit,
)
from .modes import (
    SampleBatch,
    generate_point_samples,
    generate_multi_point_samples,
    generate_patch_samples,
    compute_background_leadfield_pool,
)

__all__ = [
    'fibonacci_sphere',
    'sample_cortical_positions',
    'sample_subcortical_positions',
    'sample_unit_vectors_s2',
    'sample_orientations',
    'sample_magnitudes',
    'cartesian_to_spherical',
    'world_orientation_to_spherical_frame',
    'make_position_split',
    'PositionSplit',
    'SampleBatch',
    'generate_point_samples',
    'generate_multi_point_samples',
    'generate_patch_samples',
    'compute_background_leadfield_pool',
]
