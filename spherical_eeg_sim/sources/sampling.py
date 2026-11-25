"""
Source sampling: positions, orientations, magnitudes.

This module is responsible for choosing where dipoles live (cortical shell
and subcortical volume), what direction they point (uniform-on-sphere or
codebook), what magnitude they have (log-uniform default), and how they
are partitioned into train/val/test splits with disjoint position indices.

All randomness flows from a numpy.random.Generator passed in by the caller,
so the entire sampling stage is reproducible from the master seed.
"""

from __future__ import annotations
from dataclasses import dataclass
import numbers
from typing import Optional, Sequence, Tuple
import numpy as np


def _validate_count(name: str, value: int) -> int:
    """Validate a public sampling count while allowing an empty population."""
    if (isinstance(value, bool) or not isinstance(value, numbers.Integral)
            or value < 0):
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _validate_positive_range(name: str, value) -> Tuple[float, float]:
    """Validate a finite positive [low, high] sampling interval."""
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != 2:
        raise ValueError(f"{name} must be a two-value (low, high) pair")
    low, high = value
    if (isinstance(low, bool) or isinstance(high, bool)
            or not isinstance(low, numbers.Real)
            or not isinstance(high, numbers.Real)):
        raise ValueError(f"{name} must contain finite values with 0 < low <= high")
    low_float, high_float = float(low), float(high)
    if (
        not np.isfinite(low_float)
        or not np.isfinite(high_float)
        or low_float <= 0.0
        or high_float < low_float
    ):
        raise ValueError(f"{name} must contain finite values with 0 < low <= high")
    return low_float, high_float


# =============================================================================
# Position sampling
# =============================================================================

def fibonacci_sphere(n_points: int, radius: float = 1.0) -> np.ndarray:
    """Quasi-uniform points on a sphere via the Fibonacci lattice.

    Produces points whose angular spacing is approximately uniform, with
    much better coverage than independent uniform sampling at the same
    point count.

    Parameters
    ----------
    n_points : int
    radius : float
        Sphere radius (linear scaling of output).

    Returns
    -------
    points : ndarray, shape (n_points, 3)
        Cartesian coordinates on the sphere of given radius.
    """
    n_points = _validate_count('n_points', n_points)
    if (isinstance(radius, bool) or not isinstance(radius, numbers.Real)
            or not np.isfinite(radius) or radius < 0.0):
        raise ValueError("radius must be a finite non-negative number")
    if n_points < 1:
        return np.zeros((0, 3))

    # Standard Fibonacci lattice on unit sphere
    indices = np.arange(n_points, dtype=np.float64) + 0.5
    phi = np.arccos(1.0 - 2.0 * indices / n_points)             # polar
    golden = np.pi * (1.0 + 5.0 ** 0.5)                          # golden angle (rad)
    theta = golden * indices                                     # azimuthal

    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)

    return radius * np.column_stack([x, y, z])


def sample_cortical_positions(
    n_positions: int,
    radius_range_mm: Sequence[float],
    pattern: str = 'fibonacci',
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Sample positions on the cortical shell.

    The cortical shell spans radii in radius_range_mm. We sample radii
    uniformly within the range and angular positions either via Fibonacci
    lattice or independent uniform sampling.

    Parameters
    ----------
    n_positions : int
    radius_range_mm : (low, high)
        Inclusive radial range of the cortical shell (mm).
    pattern : str
        'fibonacci' for quasi-uniform angular coverage; 'uniform_random'
        for independent draws (worse coverage).
    rng : numpy Generator

    Returns
    -------
    positions_xyz : ndarray, shape (n_positions, 3)
        Cartesian positions in meters.
    """
    if rng is None:
        rng = np.random.default_rng()
    n_positions = _validate_count('n_positions', n_positions)
    r_low_mm, r_high_mm = _validate_positive_range(
        'radius_range_mm', radius_range_mm,
    )
    r_low_m = r_low_mm * 1e-3
    r_high_m = r_high_mm * 1e-3

    # Sample radius uniformly so radial depth receives equal probability. This
    # is not volume-uniform sampling; it is the declared superficial benchmark
    # distribution documented in docs/parameters.md.
    radii = rng.uniform(r_low_m, r_high_m, size=n_positions)

    if pattern == 'fibonacci':
        # Use Fibonacci lattice on a unit sphere, then scale per-position by
        # the sampled radius. We also randomise the lattice's overall
        # rotation to break symmetry across calls.
        unit_pts = fibonacci_sphere(n_positions, radius=1.0)
        # Random rotation matrix from the rng (uniform on SO(3)).
        R = _random_rotation_matrix(rng)
        unit_pts = unit_pts @ R.T
        positions = unit_pts * radii[:, None]
    elif pattern == 'uniform_random':
        # Independent isotropic directions from normalized Gaussian draws.
        directions = sample_unit_vectors_s2(n_positions, rng)
        positions = directions * radii[:, None]
    else:
        raise ValueError(f"Unknown cortical_sampling pattern: {pattern!r}")

    return positions


def sample_subcortical_positions(
    n_positions: int,
    radius_range_mm: Sequence[float],
    pattern: str = 'uniform_volume',
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Sample positions in the subcortical volume.

    Deep-source positions are sampled inside a spherical radial volume. The
    population is not an anatomical segmentation and does not identify nuclei
    or hippocampal anatomy. This package does not accept anatomical masks.

    Parameters
    ----------
    n_positions : int
    radius_range_mm : (low, high)
    pattern : str
        'uniform_volume'  -- volume-uniform within the shell (r ~ r^2 dr)
        'fibonacci_shells'-- Fibonacci lattice at multiple radii
    rng : numpy Generator

    Returns
    -------
    positions_xyz : ndarray, shape (n_positions, 3)
        Cartesian positions in meters.
    """
    if rng is None:
        rng = np.random.default_rng()
    n_positions = _validate_count('n_positions', n_positions)
    r_low_mm, r_high_mm = _validate_positive_range(
        'radius_range_mm', radius_range_mm,
    )
    r_low_m = r_low_mm * 1e-3
    r_high_m = r_high_mm * 1e-3

    if pattern == 'uniform_volume':
        # Inverse-CDF sampling for r ~ r^2 in [r_low, r_high]:
        # CDF(r) = (r^3 - r_low^3) / (r_high^3 - r_low^3)
        # so r = (u * (r_high^3 - r_low^3) + r_low^3)^(1/3)
        u = rng.uniform(0, 1, size=n_positions)
        radii = np.cbrt(u * (r_high_m ** 3 - r_low_m ** 3) + r_low_m ** 3)
        directions = sample_unit_vectors_s2(n_positions, rng)
        positions = directions * radii[:, None]
    elif pattern == 'fibonacci_shells':
        # Stack Fibonacci lattices at evenly spaced radii. The cube-root shell
        # count is a declared balanced 3-D discretization rule; Gonzalez (2010)
        # motivates the angular lattice, not this radial shell count.
        n_shells = max(3, int(np.cbrt(n_positions)))
        per_shell = n_positions // n_shells
        remainder = n_positions - per_shell * n_shells
        shell_radii = np.linspace(r_low_m, r_high_m, n_shells)
        chunks = []
        for i, r_m in enumerate(shell_radii):
            count = per_shell + (1 if i < remainder else 0)
            unit = fibonacci_sphere(count, radius=1.0)
            R = _random_rotation_matrix(rng)
            chunks.append((unit @ R.T) * r_m)
        positions = np.concatenate(chunks, axis=0)
        # Shuffle so split stratification is well-mixed across shells.
        order = rng.permutation(positions.shape[0])
        positions = positions[order]
    else:
        raise ValueError(f"Unknown subcortical_sampling pattern: {pattern!r}")

    return positions


def cartesian_to_spherical(xyz: np.ndarray) -> np.ndarray:
    """Convert Cartesian (x, y, z) -> spherical (r, theta, phi).

    theta is the polar angle from +z; phi is the azimuthal angle from +x.
    Both in radians. r in same units as input.

    Parameters
    ----------
    xyz : ndarray, shape (..., 3)

    Returns
    -------
    rtp : ndarray, shape (..., 3)
        (r, theta, phi) per input row.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim < 1 or xyz.shape[-1] != 3:
        raise ValueError(f"xyz must have shape (..., 3); got {xyz.shape}")
    if not np.all(np.isfinite(xyz)):
        raise ValueError("xyz must contain only finite values")
    r = np.linalg.norm(xyz, axis=-1)
    # Clamp r to floor for safe division.
    r_safe = np.where(r > 0, r, 1.0)
    theta = np.arccos(np.clip(xyz[..., 2] / r_safe, -1.0, 1.0))
    phi = np.arctan2(xyz[..., 1], xyz[..., 0])
    # Normalize phi to [0, 2*pi).
    phi = np.where(phi < 0, phi + 2 * np.pi, phi)
    return np.stack([r, theta, phi], axis=-1)


# =============================================================================
# Orientation sampling
# =============================================================================

def sample_unit_vectors_s2(
    n: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Sample n uniformly distributed unit vectors on S^2.

    Gaussian samples are normalized, exploiting the rotational invariance of
    the multivariate Gaussian to obtain a uniform distribution on the sphere.

    Parameters
    ----------
    n : int
    rng : numpy Generator

    Returns
    -------
    vecs : ndarray, shape (n, 3)
        Unit vectors uniformly distributed on the unit sphere.
    """
    if rng is None:
        rng = np.random.default_rng()
    n = _validate_count('n', n)
    g = rng.standard_normal(size=(n, 3))
    norms = np.linalg.norm(g, axis=1, keepdims=True)
    # Reject the (zero-probability) zero vectors and resample.
    bad = (norms.squeeze(-1) == 0)
    while np.any(bad):
        n_bad = int(bad.sum())
        g[bad] = rng.standard_normal(size=(n_bad, 3))
        norms = np.linalg.norm(g, axis=1, keepdims=True)
        bad = (norms.squeeze(-1) == 0)
    return g / norms


def world_orientation_to_spherical_frame(
    orientation_world: np.ndarray,
    position_xyz: np.ndarray,
) -> np.ndarray:
    """Project a world-frame orientation onto the local spherical frame.

    The forward model takes dipole moments expressed in the LOCAL spherical
    frame at the dipole position, where the basis is:
       e_r     = position direction (radially outward)
       e_theta = direction of increasing polar angle theta
       e_phi   = direction of increasing azimuthal angle phi

    Given a world-frame unit orientation vector and the dipole's Cartesian
    position, this function returns the (p_r, p_theta, p_phi) components in
    the local frame.

    Parameters
    ----------
    orientation_world : ndarray, shape (..., 3)
        Unit vector(s) in world Cartesian frame.
    position_xyz : ndarray, shape (..., 3)
        Dipole position(s) in world Cartesian frame.

    Returns
    -------
    components : ndarray, shape (..., 3)
        (p_r, p_theta, p_phi) in local frame, with the same magnitude as
        the input (unit if input was unit).
    """
    orientation_world = np.asarray(orientation_world, dtype=np.float64)
    position_xyz = np.asarray(position_xyz, dtype=np.float64)
    if (orientation_world.shape != position_xyz.shape
            or orientation_world.ndim < 1
            or orientation_world.shape[-1] != 3):
        raise ValueError(
            "orientation_world and position_xyz must have the same shape (..., 3)"
        )
    if (not np.all(np.isfinite(orientation_world))
            or not np.all(np.isfinite(position_xyz))):
        raise ValueError("orientation_world and position_xyz must be finite")
    if np.any(np.linalg.norm(position_xyz, axis=-1) <= 0.0):
        raise ValueError("position_xyz must not contain the model origin")

    r = np.linalg.norm(position_xyz, axis=-1, keepdims=True)
    r_safe = np.where(r > 0, r, 1.0)
    e_r = position_xyz / r_safe  # radial unit vector

    # Polar and azimuthal angles
    rtp = cartesian_to_spherical(position_xyz)
    theta = rtp[..., 1]
    phi = rtp[..., 2]

    # Local theta direction:
    # e_theta = (cos theta cos phi, cos theta sin phi, -sin theta)
    e_theta = np.stack([
        np.cos(theta) * np.cos(phi),
        np.cos(theta) * np.sin(phi),
        -np.sin(theta),
    ], axis=-1)

    # Local phi direction:
    # e_phi = (-sin phi, cos phi, 0)
    e_phi = np.stack([
        -np.sin(phi),
        np.cos(phi),
        np.zeros_like(phi),
    ], axis=-1)

    # Project orientation_world onto each basis vector.
    p_r = np.sum(orientation_world * e_r, axis=-1)
    p_theta = np.sum(orientation_world * e_theta, axis=-1)
    p_phi = np.sum(orientation_world * e_phi, axis=-1)

    return np.stack([p_r, p_theta, p_phi], axis=-1)


def sample_orientations(
    n: int,
    mode: str,
    positions_xyz: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
    *,
    codebook_tangential_directions: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample dipole orientations.

    Returns BOTH the world-frame unit vectors (for ground truth) AND the
    local-spherical-frame components (for the forward-model evaluation).

    Parameters
    ----------
    n : int
    mode : str
        'uniform_s2'    -- uniform on the unit sphere from normalized
                           isotropic Gaussian draws
        'codebook'      -- discrete: each sample is purely radial OR purely
                           tangential at an azimuth from a finite golden-angle
                           codebook
        'cortex_normal' -- aligned with local radial direction
    positions_xyz : ndarray (n, 3), required for 'cortex_normal' and 'codebook'
    rng : numpy Generator
    codebook_tangential_directions : int
        Number of directions in the local tangential codebook.

    Returns
    -------
    world_unit : ndarray (n, 3)
        World-frame unit orientation.
    local_components : ndarray (n, 3)
        (p_r, p_theta, p_phi) in local spherical frame at each position.
    """
    if rng is None:
        rng = np.random.default_rng()
    n = _validate_count('n', n)
    if mode not in ('uniform_s2', 'codebook', 'cortex_normal'):
        raise ValueError(f"Unknown orientation_mode: {mode!r}")
    if positions_xyz is None:
        raise ValueError(f"{mode} requires positions_xyz")
    positions_xyz = np.asarray(positions_xyz, dtype=np.float64)
    if positions_xyz.shape != (n, 3):
        raise ValueError(f"positions_xyz must have shape ({n}, 3)")
    if (not np.all(np.isfinite(positions_xyz))
            or np.any(np.linalg.norm(positions_xyz, axis=1) <= 0.0)):
        raise ValueError("positions_xyz must contain finite non-origin positions")

    if mode == 'uniform_s2':
        world_unit = sample_unit_vectors_s2(n, rng)
        local = world_orientation_to_spherical_frame(world_unit, positions_xyz)
        return world_unit, local

    elif mode == 'cortex_normal':
        r = np.linalg.norm(positions_xyz, axis=-1, keepdims=True)
        r_safe = np.where(r > 0, r, 1.0)
        world_unit = positions_xyz / r_safe  # purely radial
        # In local spherical frame: p_r = +/- 1, p_theta = p_phi = 0
        # Random sign so half point inward, half outward.
        signs = rng.choice([-1.0, 1.0], size=n)
        world_unit = world_unit * signs[:, None]
        local = np.zeros((n, 3))
        local[:, 0] = signs
        return world_unit, local

    elif mode == 'codebook':
        codebook_size = _validate_count(
            'codebook_tangential_directions',
            codebook_tangential_directions,
        )
        if codebook_size < 1:
            raise ValueError("codebook_tangential_directions must be at least 1")

        is_radial = rng.uniform(0, 1, size=n) < (1.0 / 3.0)
        n_tang = int((~is_radial).sum())
        golden = np.pi * (1.0 + 5.0 ** 0.5)
        codebook_index = rng.integers(0, codebook_size, size=n_tang)
        psi = (codebook_index * golden) % (2 * np.pi)
        signs = rng.choice([-1.0, 1.0], size=int(is_radial.sum()))

        local = np.zeros((n, 3))
        local[is_radial, 0] = signs
        local[~is_radial, 1] = np.cos(psi)
        local[~is_radial, 2] = np.sin(psi)

        # Convert to world frame for ground truth.
        world_unit = _local_to_world(local, positions_xyz)
        return world_unit, local

    raise AssertionError("unreachable orientation mode")

def _local_to_world(
    local_components: np.ndarray,
    positions_xyz: np.ndarray,
) -> np.ndarray:
    """Inverse of world_orientation_to_spherical_frame."""
    rtp = cartesian_to_spherical(positions_xyz)
    theta = rtp[..., 1]
    phi = rtp[..., 2]

    e_r = np.stack([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ], axis=-1)
    e_theta = np.stack([
        np.cos(theta) * np.cos(phi),
        np.cos(theta) * np.sin(phi),
        -np.sin(theta),
    ], axis=-1)
    e_phi = np.stack([
        -np.sin(phi),
        np.cos(phi),
        np.zeros_like(phi),
    ], axis=-1)

    pr = local_components[..., 0:1]
    pt = local_components[..., 1:2]
    pp = local_components[..., 2:3]
    return pr * e_r + pt * e_theta + pp * e_phi


def _random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    """Sample a uniformly random rotation matrix in SO(3).

    Uses QR decomposition of a Gaussian matrix; the orthogonal factor is
    uniform on O(3), and we flip sign of the last column if needed to
    ensure determinant +1.
    """
    A = rng.standard_normal(size=(3, 3))
    Q, R = np.linalg.qr(A)
    # Ensure positive diagonal of R (canonicalization)
    d = np.sign(np.diag(R))
    d[d == 0] = 1
    Q = Q * d[None, :]
    if np.linalg.det(Q) < 0:
        Q[:, -1] *= -1
    return Q


# =============================================================================
# Magnitude sampling
# =============================================================================

def sample_magnitudes(
    n: int,
    range_nAm: Sequence[float],
    distribution: str = 'log_uniform',
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Sample dipole magnitudes (in nA*m).

    Parameters
    ----------
    n : int
    range_nAm : (low, high)
    distribution : str
        'log_uniform' -- uniform in log-space (default benchmark for coverage
                         across the configured magnitude range)
        'uniform'     -- uniform in linear space
        'fixed'       -- constant equal to range_nAm[0]
    rng : numpy Generator

    Returns
    -------
    magnitudes : ndarray, shape (n,)
        Magnitudes in nA*m. Multiply by 1e-9 to convert to A*m for the
        forward model.
    """
    if rng is None:
        rng = np.random.default_rng()
    n = _validate_count('n', n)
    low, high = _validate_positive_range('range_nAm', range_nAm)

    if distribution == 'log_uniform':
        return np.exp(rng.uniform(np.log(low), np.log(high), size=n))
    elif distribution == 'uniform':
        return rng.uniform(low, high, size=n)
    elif distribution == 'fixed':
        return np.full(n, low)
    else:
        raise ValueError(f"Unknown magnitude_distribution: {distribution!r}")


# =============================================================================
# Position-level train/val/test split
# =============================================================================

@dataclass
class PositionSplit:
    """Per-position split assignment.

    split[i] is one of:
      0 -- train
      1 -- val
      2 -- test
    """
    split: np.ndarray  # shape (P,), int8


def make_position_split(
    positions_xyz: np.ndarray,
    n_train_target: int,
    n_val_target: int,
    n_test_target: int,
    stratify_by_depth: bool = True,
    depth_bins: int = 10,
    rng: Optional[np.random.Generator] = None,
    population: Optional[np.ndarray] = None,
) -> PositionSplit:
    """Stratified position-level split.

    Each position is assigned to ONE of train, val, or test. The split is
    chosen to match the requested fractional split sizes (n_train_target /
    total, etc.) approximately, while ensuring stratification by radial
    depth so all splits have balanced superficial/deep representation.

    ``n_*_target`` are positive allocation weights, commonly the desired
    numbers of generated samples. Multiple samples may use the same assigned
    position with different orientations, magnitudes, and signals. Only the
    normalized target ratios affect the position allocation.

    Parameters
    ----------
    positions_xyz : ndarray (P, 3)
    n_train_target, n_val_target, n_test_target : int
        Used only to determine fractional split sizes.
    stratify_by_depth : bool
    depth_bins : int
    rng : numpy Generator
    population : ndarray (P,), optional
        Per-position population label (0=cortical, 1=subcortical). When given,
        EACH population is split independently with the same fractions, so every
        split receives a proportional share of both populations. This guarantees
        the configured ``subcortical_fraction`` is achievable in every split
        whenever a population has at least as many positions as there are splits
        that need it. When omitted, all positions are split together.

    Returns
    -------
    PositionSplit with split array of length P.
    """
    positions_xyz = np.asarray(positions_xyz, dtype=np.float64)
    if positions_xyz.ndim != 2 or positions_xyz.shape[1:] != (3,):
        raise ValueError("positions_xyz must have shape (n_positions, 3)")
    if positions_xyz.shape[0] < 3:
        raise ValueError("positions_xyz must contain at least three positions")
    if (not np.all(np.isfinite(positions_xyz))
            or np.any(np.linalg.norm(positions_xyz, axis=1) <= 0.0)):
        raise ValueError("positions_xyz must contain finite non-origin positions")
    targets = (n_train_target, n_val_target, n_test_target)
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value <= 0
        for value in targets
    ):
        raise ValueError("n_train_target, n_val_target, and n_test_target must be positive integers")
    if not isinstance(stratify_by_depth, (bool, np.bool_)):
        raise ValueError("stratify_by_depth must be a boolean")
    if (isinstance(depth_bins, (bool, np.bool_))
            or not isinstance(depth_bins, (int, np.integer))
            or depth_bins < 1):
        raise ValueError("depth_bins must be a positive integer")
    if population is not None:
        population = np.asarray(population)
        if population.shape != (positions_xyz.shape[0],):
            raise ValueError(
                "population must have shape (n_positions,) matching positions_xyz"
            )
        if (not np.issubdtype(population.dtype, np.number)
                or not np.all(np.isfinite(population))
                or not np.all(np.isin(population, (0, 1)))):
            raise ValueError("population labels must be finite and equal to 0 or 1")
    if rng is None:
        rng = np.random.default_rng()

    P = positions_xyz.shape[0]
    total_target = n_train_target + n_val_target + n_test_target
    f_train = n_train_target / total_target
    f_val = n_val_target / total_target
    # f_test = the rest

    split = np.full(P, -1, dtype=np.int8)

    # Split each population independently (when labels are given) so every
    # split gets a proportional share of cortical and subcortical positions.
    if population is not None:
        groups = [np.where(population == p)[0] for p in np.unique(population)]
    else:
        groups = [np.arange(P)]

    for group_idx in groups:
        if group_idx.size == 0:
            continue
        _assign_depth_stratified(
            group_idx, positions_xyz, f_train, f_val,
            stratify_by_depth, depth_bins, rng, split,
        )
        _rebalance_group_counts(group_idx, f_train, f_val, split, rng)

    if (split == -1).any():
        raise RuntimeError("Some positions were not assigned a split (bug)")

    # Repair pass: stratified rounding at extreme split ratios can send 0
    # positions to a small-fraction split (val or test) across every depth bin,
    # leaving that split empty. Guarantee that each population group large
    # enough to cover all three splits contributes at least one position to
    # each, by moving a single position from the richest split to any empty one.
    # This preserves disjointness and only perturbs counts by +/-1 per group, so
    # it is a no-op for normally-sized pools.
    _ensure_all_splits_populated(groups, split, rng)

    # If a split is still globally empty, the pool is simply too small for the
    # requested fractions; fail with an actionable message instead of letting a
    # downstream stage raise an opaque "empty pool" error.
    for s, name in ((0, 'train'), (1, 'val'), (2, 'test')):
        if int((split == s).sum()) == 0:
            raise ValueError(
                f"Position-level split produced an empty {name!r} pool. The "
                f"requested split fractions (train:val:test derived from "
                f"n_train:n_val:n_test) are too small for the position pool "
                f"({P} positions) under depth stratification. Increase "
                f"n_cortical_positions / n_subcortical_positions, reduce "
                f"depth_bins, or raise the {name} split's share."
            )

    return PositionSplit(split=split)


def _rebalance_group_counts(
    group_idx: np.ndarray,
    f_train: float,
    f_val: float,
    split_out: np.ndarray,
    rng: np.random.Generator,
) -> None:
    """Match group-level fractional targets after per-depth-bin rounding."""
    fractions = np.array([f_train, f_val, 1.0 - f_train - f_val])
    raw = group_idx.size * fractions
    target = np.floor(raw).astype(int)
    remainder = group_idx.size - int(target.sum())
    for s in np.argsort(-(raw - target), kind='stable')[:remainder]:
        target[s] += 1

    counts = np.array([np.sum(split_out[group_idx] == s) for s in range(3)])
    for recipient in range(3):
        while counts[recipient] < target[recipient]:
            donors = [s for s in range(3) if counts[s] > target[s]]
            if not donors:
                raise RuntimeError("Could not rebalance position split counts")
            donor = max(donors, key=lambda s: counts[s] - target[s])
            donor_pool = group_idx[np.flatnonzero(split_out[group_idx] == donor)]
            member = int(rng.choice(donor_pool))
            split_out[member] = recipient
            counts[donor] -= 1
            counts[recipient] += 1


def _ensure_all_splits_populated(
    groups, split_out: np.ndarray, rng: np.random.Generator
) -> None:
    """Guarantee each split (0/1/2) gets >= 1 position from every group that has
    at least 3 positions, by reassigning from the richest split. In place."""
    SPLITS = (0, 1, 2)
    for group_idx in groups:
        if group_idx.size < len(SPLITS):
            continue  # too few to seed all splits; another group must cover
        for s in SPLITS:
            if int((split_out[group_idx] == s).sum()) > 0:
                continue
            # Donor: the split with the most members in this group (and >= 2 so
            # it is not itself emptied).
            counts = {x: int((split_out[group_idx] == x).sum()) for x in SPLITS}
            donor = max(SPLITS, key=lambda x: counts[x])
            if counts[donor] < 2:
                continue  # cannot safely donate; leave for the global check
            donor_members = group_idx[split_out[group_idx] == donor]
            split_out[int(rng.choice(donor_members))] = s


def _assign_depth_stratified(
    member_indices: np.ndarray,
    positions_xyz: np.ndarray,
    f_train: float,
    f_val: float,
    stratify_by_depth: bool,
    depth_bins: int,
    rng: np.random.Generator,
    split_out: np.ndarray,
) -> None:
    """Assign train/val/test (0/1/2) to ``member_indices`` in ``split_out``.

    Equal-count quantile bins on radial depth are used when stratifying, so the
    fractions hold within each depth band. Operates in place on ``split_out``.
    """
    member_indices = np.asarray(member_indices)
    M = member_indices.size
    if M == 0:
        return

    if stratify_by_depth:
        bin_idx = _depth_bin_indices(positions_xyz[member_indices], depth_bins)
    else:
        bin_idx = np.zeros(M, dtype=np.int64)

    fractions = np.array([f_train, f_val, 1.0 - f_train - f_val])
    for bin_number, b in enumerate(np.unique(bin_idx)):
        local = np.where(bin_idx == b)[0]
        members = member_indices[local]
        n = members.size
        rng.shuffle(members)
        raw = n * fractions
        counts = np.floor(raw).astype(int)
        remainder = n - int(counts.sum())
        # Largest-remainder apportionment keeps every bin close to the target
        # fractions.  Cycling exact ties prevents the same split from losing
        # one position in every depth bin (important for equal-density patch
        # support meshes).
        frac = raw - counts
        order = sorted(
            range(3),
            key=lambda s: (-frac[s], (s - bin_number) % 3),
        )
        for s in order[:remainder]:
            counts[s] += 1
        n_train_bin, n_val_bin, n_test_bin = map(int, counts)
        split_out[members[:n_train_bin]] = 0
        split_out[members[n_train_bin:n_train_bin + n_val_bin]] = 1
        split_out[members[n_train_bin + n_val_bin:]] = 2


def _depth_bin_indices(positions_xyz: np.ndarray, depth_bins: int) -> np.ndarray:
    """Equal-count radial bin labels used by the position splitter."""
    if positions_xyz.shape[0] < depth_bins:
        return np.zeros(positions_xyz.shape[0], dtype=np.int64)
    radii = np.linalg.norm(positions_xyz, axis=1)
    quantiles = np.quantile(radii, np.linspace(0, 1, depth_bins + 1))
    return np.digitize(radii, quantiles[1:-1])
