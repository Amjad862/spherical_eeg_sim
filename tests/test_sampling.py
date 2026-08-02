"""
Unit tests for source sampling: positions, orientations, magnitudes, splits.
"""

import sys as _sys
from pathlib import Path as _Path
_repo_root = _Path(__file__).resolve().parent.parent
if str(_repo_root) not in _sys.path:
    _sys.path.insert(0, str(_repo_root))

import numpy as np
import pytest

from spherical_eeg_sim.sources.sampling import (
    fibonacci_sphere,
    sample_cortical_positions,
    sample_subcortical_positions,
    sample_unit_vectors_s2,
    sample_orientations,
    sample_magnitudes,
    make_position_split,
    cartesian_to_spherical,
    world_orientation_to_spherical_frame,
    _local_to_world,
)


def test_fibonacci_sphere_unit_norm():
    pts = fibonacci_sphere(500)
    assert pts.shape == (500, 3)
    norms = np.linalg.norm(pts, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-10)


def test_unit_vectors_s2_uniform():
    """Normalized isotropic Gaussian draws are uniform on S2."""
    rng = np.random.default_rng(0)
    v = sample_unit_vectors_s2(50_000, rng)
    norms = np.linalg.norm(v, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-10)
    # z component uniform in [-1, 1] => mean 0, std 1/sqrt(3)
    assert abs(v[:, 2].mean()) < 0.01
    assert abs(v[:, 2].std() - 1/np.sqrt(3)) < 0.01


def test_cortical_positions_within_range():
    rng = np.random.default_rng(0)
    pos = sample_cortical_positions(1000, (70.5, 79.5), 'fibonacci', rng)
    radii = np.linalg.norm(pos, axis=1) * 1000
    assert radii.min() >= 70.4
    assert radii.max() <= 79.6


def test_subcortical_positions_within_range():
    rng = np.random.default_rng(0)
    pos = sample_subcortical_positions(1000, (30, 65), 'uniform_volume', rng)
    radii = np.linalg.norm(pos, axis=1) * 1000
    assert radii.min() >= 29.9
    assert radii.max() <= 65.1


def test_orientation_world_local_roundtrip():
    rng = np.random.default_rng(42)
    n = 100
    pos = rng.standard_normal((n, 3)) * 0.05
    pos += 0.05 * np.sign(pos)  # avoid origin
    world, local = sample_orientations(n, 'uniform_s2', pos, rng)
    # World -> local -> world should round-trip
    world_back = _local_to_world(local, pos)
    assert np.max(np.abs(world - world_back)) < 1e-12


def test_codebook_orientation_split():
    rng = np.random.default_rng(0)
    n = 10000
    pos = rng.standard_normal((n, 3)) * 0.05
    pos += 0.05 * np.sign(pos)
    world, local = sample_orientations(n, 'codebook', pos, rng)
    n_radial = (np.abs(local[:, 0]) > 0.5).sum()
    n_tang = ((local[:, 1]**2 + local[:, 2]**2) > 0.5).sum()
    assert n_radial + n_tang == n
    assert 0.28 < n_radial / n < 0.38, f"radial fraction {n_radial/n}"
    golden = np.pi * (1.0 + np.sqrt(5.0))
    expected_angles = (np.arange(256) * golden) % (2 * np.pi)
    observed_angles = np.mod(
        np.arctan2(local[~(np.abs(local[:, 0]) > 0.5), 2],
                   local[~(np.abs(local[:, 0]) > 0.5), 1]),
        2 * np.pi,
    )
    expected_codes = set(np.round(expected_angles, 12))
    observed_codes = set(np.round(observed_angles, 12))
    assert observed_codes <= expected_codes
    assert len(observed_codes) > 240


def test_magnitude_log_uniform_range():
    rng = np.random.default_rng(0)
    m = sample_magnitudes(10_000, (5, 300), 'log_uniform', rng)
    assert m.min() >= 5.0
    assert m.max() <= 300.0
    # Geometric mean ~= sqrt(5*300) = ~38.7
    assert 30 < np.exp(np.log(m).mean()) < 50


def test_cartesian_spherical_roundtrip():
    rng = np.random.default_rng(0)
    xyz = rng.standard_normal((100, 3))
    rtp = cartesian_to_spherical(xyz)
    # Reconstruct
    r, theta, phi = rtp[:, 0], rtp[:, 1], rtp[:, 2]
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    rec = np.stack([x, y, z], axis=1)
    assert np.allclose(xyz, rec, atol=1e-12)


def test_position_split_disjoint():
    rng = np.random.default_rng(0)
    P = 1000
    pos = rng.standard_normal((P, 3)) * 0.05
    sp = make_position_split(pos, 800, 100, 100, True, 10, rng)
    counts = np.bincount(sp.split, minlength=3)
    # Approximate fractions
    assert abs(counts[0]/P - 0.8) < 0.02
    assert abs(counts[1]/P - 0.1) < 0.02
    assert abs(counts[2]/P - 0.1) < 0.02
    # Every position assigned to exactly one split
    assert (sp.split >= 0).all() and (sp.split <= 2).all()


@pytest.mark.parametrize(
    "call",
    [
        lambda: fibonacci_sphere(-1),
        lambda: fibonacci_sphere(3, radius=np.nan),
        lambda: sample_cortical_positions(3, (80.0, 70.0)),
        lambda: sample_subcortical_positions(3, (0.0, 65.0)),
        lambda: sample_unit_vectors_s2(True),
        lambda: sample_magnitudes(2, (1.0, np.inf)),
    ],
)
def test_public_sampling_functions_reject_invalid_scalar_inputs(call):
    with pytest.raises(ValueError):
        call()


def test_coordinate_and_orientation_functions_reject_invalid_arrays():
    with pytest.raises(ValueError, match="shape"):
        cartesian_to_spherical(np.ones((2, 2)))
    with pytest.raises(ValueError, match="finite"):
        cartesian_to_spherical(np.array([[0.0, np.nan, 1.0]]))
    with pytest.raises(ValueError, match="same shape"):
        world_orientation_to_spherical_frame(
            np.ones((2, 3)), np.ones((3, 3)),
        )
    with pytest.raises(ValueError, match="origin"):
        world_orientation_to_spherical_frame(
            np.array([[1.0, 0.0, 0.0]]), np.zeros((1, 3)),
        )
    with pytest.raises(ValueError, match="shape"):
        sample_orientations(2, "uniform_s2", np.ones((1, 3)))


@pytest.mark.parametrize('kwargs', [
    {'positions_xyz': np.ones((2, 2))},
    {'positions_xyz': np.array([[1.0, 0.0, 0.0], [0.0, np.nan, 1.0], [0.0, 1.0, 0.0]])},
    {'n_train_target': 0},
    {'n_val_target': 1.5},
    {'depth_bins': 0},
    {'population': np.array([0, 1])},
    {'population': np.array([0, 1, 2])},
])
def test_position_split_rejects_malformed_public_inputs(kwargs):
    base = {
        'positions_xyz': np.eye(3),
        'n_train_target': 1,
        'n_val_target': 1,
        'n_test_target': 1,
        'rng': np.random.default_rng(1),
    }
    base.update(kwargs)
    with pytest.raises(ValueError):
        make_position_split(**base)


if __name__ == '__main__':
    test_fibonacci_sphere_unit_norm()
    test_unit_vectors_s2_uniform()
    test_cortical_positions_within_range()
    test_subcortical_positions_within_range()
    test_orientation_world_local_roundtrip()
    test_codebook_orientation_split()
    test_magnitude_log_uniform_range()
    test_cartesian_spherical_roundtrip()
    test_position_split_disjoint()
    print("All sampling tests passed!")
