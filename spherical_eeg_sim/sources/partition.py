"""Plan source-position eligibility without computing lead fields or EEG."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Optional
import warnings

import numpy as np
from scipy.spatial import cKDTree

from ..config import Config
from .sampling import (
    _random_rotation_matrix,
    fibonacci_sphere,
    make_position_split,
    sample_cortical_positions,
    sample_subcortical_positions,
)


SPLIT_NAMES = ('train', 'val', 'test')
EXCLUDED = -1
SHARED = -2
# A split whose largest patches fit at fewer centres than this draws a warning.
_FEW_PATCH_CENTRES = 50
# Relative shrink of the containment radius, so that a footprint never touches
# a sector edge or a position of another split through rounding.
_CONTAINMENT_SHRINK = 1e-9


@dataclass(frozen=True)
class RegionPartition:
    """Rotated unit directions and the pre-buffer Voronoi assignment."""

    sector_directions: np.ndarray
    sector_split: np.ndarray
    sector_index: np.ndarray
    pre_buffer_split_labels: np.ndarray


@dataclass(frozen=True)
class SourcePartition:
    """Global positions (metres) and the eligible indices for each split.

    ``split_labels`` is 0/1/2 for train/val/test, -1 for buffer exclusions,
    and -2 for positions eligible in all three splits. ``membership`` gives
    explicit boolean columns in train/val/test order for the shared policy.
    ``patch_containment_radius_mm`` is set only for patch mode under the region
    policy: the largest radius admitted by the planner at that position
    (geodesic, on the position's own radius). The bound ensures containment
    and can be conservative near sector corners. It is NaN for excluded positions.
    """

    positions_xyz: np.ndarray
    population: np.ndarray
    split_labels: np.ndarray
    indices: dict[str, np.ndarray]
    allocation_basis: str
    membership: Optional[np.ndarray] = None
    region: Optional[RegionPartition] = None
    patch_containment_radius_mm: Optional[np.ndarray] = None


def _setup_rngs(cfg: Config) -> tuple[
    np.random.Generator, np.random.Generator, np.random.Generator
]:
    seed = cfg.master_seed if cfg.position_seed is None else cfg.position_seed
    master_rng = np.random.default_rng(seed)
    # Keep this draw order independent of split policy. The third draw seeds
    # the common background lead-field pool even when the split RNG is unused.
    rng_pos = np.random.default_rng(master_rng.integers(0, 2**31))
    rng_split = np.random.default_rng(master_rng.integers(0, 2**31))
    rng_background = np.random.default_rng(master_rng.integers(0, 2**31))
    return rng_pos, rng_split, rng_background


def plan_position_split(cfg: Config) -> SourcePartition:
    """Return the generator's source-position pool and split assignments.

    This runs configuration validation and the seeded geometry steps only.
    Region labels record both the sector assignment and buffer exclusions.
    Positions keep the same row indices across split policies with the same
    position configuration and seed. The returned arrays are read-only by
    convention.
    """
    cfg.validate()
    rng_pos, rng_split, _ = _setup_rngs(cfg)
    return _plan_partition(cfg, rng_pos, rng_split)


def _plan_partition(
    cfg: Config, rng_pos: np.random.Generator, rng_split: np.random.Generator,
) -> SourcePartition:
    cort = sample_cortical_positions(
        cfg.sources.n_cortical_positions, cfg.sources.cortical_radius_range_mm,
        cfg.sources.cortical_sampling, rng_pos,
    )
    if cfg.source_mode == 'patch':
        sub = np.empty((0, 3), dtype=np.float64)
    else:
        sub = sample_subcortical_positions(
            cfg.sources.n_subcortical_positions, cfg.sources.subcortical_radius_range_mm,
            cfg.sources.subcortical_sampling, rng_pos,
        )
    positions = np.concatenate([cort, sub], axis=0)
    population = np.concatenate([
        np.zeros(cort.shape[0], dtype=np.int8),
        np.ones(sub.shape[0], dtype=np.int8),
    ])

    targets = (cfg.dataset.n_train, cfg.dataset.n_val, cfg.dataset.n_test)
    region = None
    membership = None
    containment = None
    if cfg.split.policy == 'region':
        assert cfg.split.region_count is not None
        assert cfg.split.region_rotation_seed is not None
        assert cfg.split.buffer_mm is not None
        region = _plan_regions(
            positions, targets, cfg.split.region_count, cfg.split.region_rotation_seed,
        )
        labels = _buffer_regions(positions * 1000.0, region, cfg.split.buffer_mm)
        allocation_basis = 'latitude-banded sectors with Euclidean buffer'
        if cfg.source_mode == 'patch':
            containment = _patch_containment_radius_mm(positions, labels, region)
            _require_patch_capacity(containment, labels, cfg.patch.diameter_range_mm[1])
            allocation_basis += ' and patch-support containment'
    elif cfg.split.policy == 'shared' and cfg.split.shared_position_fraction == 1.0:
        labels = np.full(positions.shape[0], SHARED, dtype=np.int8)
        allocation_basis = 'common finite pool'
    else:
        # Patch support is recruited from split-local meshes of equal density.
        if cfg.source_mode == 'patch':
            targets = (1, 1, 1)
            allocation_basis = 'equal-density patch support meshes'
        else:
            allocation_basis = 'sample-count proportions'
        labels = make_position_split(
            positions, *targets, cfg.split.stratify_by_depth, cfg.split.depth_bins,
            rng_split, population=population,
        ).split
        if cfg.split.policy == 'shared':
            assert cfg.split.shared_position_fraction is not None
            for pop in np.unique(population):
                members = np.flatnonzero(population == pop)
                count = int(Decimal(members.size) * Decimal(str(cfg.split.shared_position_fraction)))
                labels[rng_split.choice(members, size=count, replace=False)] = SHARED
                if count == 0:
                    warnings.warn(
                        f'The shared fraction selects zero positions in population {int(pop)}. '
                        'Increase the fraction or pool size to share this population.',
                        RuntimeWarning,
                    )
            if not np.any(labels == SHARED):
                raise ValueError(
                    'split.shared_position_fraction selects zero shared positions; '
                    'increase the fraction or the source pool size'
                )
            allocation_basis = 'common core plus split-specific positions'

    if cfg.split.policy == 'shared':
        membership = (labels[:, None] == np.arange(3)) | (labels[:, None] == SHARED)
    indices = {
        name: np.flatnonzero(labels == split_id if membership is None else membership[:, split_id])
        for split_id, name in enumerate(SPLIT_NAMES)
    }
    if cfg.split.policy in ('region', 'shared'):
        # Patch pools hold cortical positions only, whatever the configured fraction.
        subcortical_fraction = 0.0 if cfg.source_mode == 'patch' else cfg.sources.subcortical_fraction
        _require_populations(indices, population, subcortical_fraction, cfg.split.policy)
    return SourcePartition(
        positions, population, labels, indices, allocation_basis, membership, region, containment,
    )


def _plan_regions(
    positions: np.ndarray, targets: tuple[int, int, int], count: int, seed: int,
) -> RegionPartition:
    n_train, n_val, n_test = (int(n) for n in targets)
    total = n_train + n_val + n_test
    n_test_sectors = max(1, round(count * (n_test / total)))
    n_val_sectors = max(1, round(count * (n_val / total)))
    if n_test_sectors + n_val_sectors >= count:
        raise ValueError(
            'split.region_count and dataset ratios must leave at least one sector '
            'for each of train, val, and test; increase region_count or change the ratios'
        )

    rng = np.random.default_rng(seed)
    directions = fibonacci_sphere(count) @ _random_rotation_matrix(rng).T
    order = np.argsort(directions[:, 2])
    sector_split = np.zeros(count, dtype=np.int8)
    for band in np.array_split(order, n_test_sectors):
        sector_split[rng.choice(band)] = 2
    for band in np.array_split(order[sector_split[order] == 0], n_val_sectors):
        sector_split[rng.choice(band)] = 1

    unit = positions / np.linalg.norm(positions, axis=1, keepdims=True)
    sector_index = np.empty(unit.shape[0], dtype=np.int64)
    # Keep the pilot's maximum-dot-product assignment and bound working memory.
    chunk = max(1, 1_000_000 // count)
    for start in range(0, unit.shape[0], chunk):
        block = slice(start, start + chunk)
        sector_index[block] = np.argmax(unit[block] @ directions.T, axis=1)
    return RegionPartition(directions, sector_split, sector_index, sector_split[sector_index])


def _buffer_regions(positions_mm: np.ndarray, region: RegionPartition, buffer_mm: float) -> np.ndarray:
    labels = region.pre_buffer_split_labels.copy()
    if buffer_mm == 0:
        return labels
    # Priority is test, then validation, then training. Buffer in 3-D across
    # both populations together. Equality at the requested clearance is valid.
    for lower_priority in (1, 0):
        candidates = np.flatnonzero(labels == lower_priority)
        held = positions_mm[labels > lower_priority]
        if candidates.size and held.size:
            distances = cKDTree(held).query(positions_mm[candidates])[0]
            labels[candidates[distances < buffer_mm]] = EXCLUDED
    return labels


def sector_margin_rad(
    unit: np.ndarray, split: np.ndarray, sector_directions: np.ndarray, sector_split: np.ndarray,
) -> np.ndarray:
    """Angular distance from each direction to the nearest sector of another split.

    ``unit`` holds unit directions and ``split`` the split each belongs to. The
    Voronoi cell of sector j lies in the half-sphere of directions at least as
    close to its centre d_j as to any other centre d_k, so the distance from u
    to cell j is at least the signed distance from u to every bisecting great
    circle, arcsin(u . (d_k - d_j) / |d_k - d_j|). The maximum over k bounds
    the distance to cell j from below, and the minimum over the sectors of
    other splits bounds the distance to any of them. The bound is exact along
    sector edges and conservative near sector corners, so a cap whose
    half-angle does not exceed it stays inside the sectors of its own split.
    """
    margin = np.full(unit.shape[0], np.inf)
    count = sector_directions.shape[0]
    # Bound the working array to about eight million entries.
    chunk = max(1, 8_000_000 // count)
    for j in range(count):
        members = np.flatnonzero(split != sector_split[j])
        if not members.size:
            continue
        others = np.arange(count) != j
        normals = sector_directions[others] - sector_directions[j]
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        for start in range(0, members.size, chunk):
            block = members[start:start + chunk]
            signed = np.arcsin(np.clip(unit[block] @ normals.T, -1.0, 1.0)).max(axis=1)
            margin[block] = np.minimum(margin[block], signed)
    return margin


def _patch_containment_radius_mm(
    positions: np.ndarray, labels: np.ndarray, region: RegionPartition,
) -> np.ndarray:
    """Maximum patch radius in millimetres admitted by the planner at each centre.

    Two margins limit the footprint, the cap of directions around the centre.
    The sector margin keeps it inside the sectors of the centre's split, which
    makes footprints of different splits disjoint. The pool margin, the
    angular distance to the nearest position with any other final label,
    keeps excluded buffer positions and other splits out of it, so the patch
    recruits exactly the pool positions in its footprint. The angle is
    converted with the centre's own radius, the convention of patch mode.
    """
    radius_mm = np.linalg.norm(positions, axis=1) * 1000.0
    unit = positions / np.linalg.norm(positions, axis=1, keepdims=True)
    result = np.full(positions.shape[0], np.nan)
    eligible = labels >= 0
    angle = np.full(positions.shape[0], np.inf)
    angle[eligible] = sector_margin_rad(
        unit[eligible], labels[eligible], region.sector_directions, region.sector_split,
    )
    for split_id in range(3):
        own = np.flatnonzero(labels == split_id)
        foreign = np.flatnonzero(labels != split_id)
        if own.size and foreign.size:
            chord = cKDTree(unit[foreign]).query(unit[own])[0]
            angle[own] = np.minimum(angle[own], 2.0 * np.arcsin(np.minimum(chord / 2.0, 1.0)))
    angle = np.maximum(angle, 0.0) * (1.0 - _CONTAINMENT_SHRINK)
    result[eligible] = np.minimum(angle[eligible], np.pi) * radius_mm[eligible]
    return result


def patch_capacity(
    containment_radius_mm: np.ndarray, labels: np.ndarray, diameter_mm: float,
) -> dict[str, int]:
    """Number of centres in each split admitted by the planner for this diameter."""
    return {
        name: int(np.sum(containment_radius_mm[labels == split_id] >= diameter_mm / 2.0))
        for split_id, name in enumerate(SPLIT_NAMES)
    }


def _require_patch_capacity(
    containment_radius_mm: np.ndarray, labels: np.ndarray, largest_diameter_mm: float,
) -> None:
    for split_id, name in enumerate(SPLIT_NAMES):
        members = containment_radius_mm[labels == split_id]
        admissible = 2.0 * float(members.max()) if members.size else 0.0
        if admissible < largest_diameter_mm:
            raise ValueError(
                f'The conservative containment rule admits no centre for a '
                f'{largest_diameter_mm:g} mm patch in the {name} split '
                f'(largest admitted diameter {admissible:.1f} mm); reduce '
                'split.region_count, split.buffer_mm or patch.diameter_range_mm[1]'
            )
        centres = int(np.sum(members >= largest_diameter_mm / 2.0))
        if centres < _FEW_PATCH_CENTRES:
            warnings.warn(
                f'The containment rule admits only {centres} centres in the {name} split '
                f'for a {largest_diameter_mm:g} mm patch; the largest patches in that split will '
                'share few centres. Consider fewer, larger sectors or a smaller buffer.',
                RuntimeWarning,
            )


def _require_populations(
    indices: dict[str, np.ndarray], population: np.ndarray,
    subcortical_fraction: float, policy: str,
) -> None:
    required = []
    if subcortical_fraction < 1.0:
        required.append((0, 'cortical'))
    if subcortical_fraction > 0.0:
        required.append((1, 'subcortical'))
    for split, members in indices.items():
        for code, name in required:
            if not np.any(population[members] == code):
                raise ValueError(
                    f'{policy.capitalize()} split has no eligible {name} positions in {split}; '
                    'adjust the split settings or increase the source pool'
                )
