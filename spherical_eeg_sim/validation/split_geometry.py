"""Audit the source geometry eligible for and actually used by each split."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from scipy.spatial import cKDTree

from ..config import Config
from ..sources.partition import (
    SHARED, SPLIT_NAMES, SourcePartition, patch_capacity, sector_margin_rad,
)
from ..sources.sampling import _depth_bin_indices


SPLIT_PAIRS = (('test', 'train'), ('val', 'train'), ('test', 'val'))
POPULATIONS = ((0, 'cortical'), (1, 'subcortical'))


def distance_summary(distances: np.ndarray) -> dict[str, Any]:
    if not distances.size:
        return {'n': 0, 'min': None, 'p10': None, 'median': None, 'p90': None, 'max': None}
    p10, median, p90 = np.percentile(distances, [10, 50, 90])
    return {
        'n': int(distances.size), 'min': float(distances.min()), 'p10': float(p10),
        'median': float(median), 'p90': float(p90), 'max': float(distances.max()),
    }


def _nearest_distances(
    positions_mm: np.ndarray, population: np.ndarray, indices: dict[str, np.ndarray],
) -> dict[str, Any]:
    result = {}
    for query, reference in SPLIT_PAIRS:
        query_indices = indices[query]
        reference_indices = indices[reference]
        measures = {}
        for label, pop in (('all', None), ('cortical', 0), ('subcortical', 1)):
            q, r = query_indices, reference_indices
            if pop is not None:
                q = q[population[q] == pop]
                r = r[population[r] == pop]
            d = (
                cKDTree(positions_mm[r]).query(positions_mm[q])[0]
                if q.size and r.size else np.empty(0)
            )
            measures[label] = distance_summary(d)
        result[f'{query}_to_{reference}'] = measures
    return result


def audit_split_geometry(
    cfg: Config, partition: SourcePartition, serialized_indices: dict[str, np.ndarray],
    position_rows: dict[str, np.ndarray],
    patch_diameter_mm: Optional[dict[str, np.ndarray]] = None,
) -> dict[str, Any]:
    """Report eligible and used nearest-neighbour distances and policy facts.

    ``patch_diameter_mm`` carries the serialized patch diameters of each split
    for patch mode under the region policy; the containment of every written
    patch is then verified and reported.
    """
    population = partition.population
    positions_mm = partition.positions_xyz * 1000.0
    used = {name: np.unique(values[values >= 0]) for name, values in serialized_indices.items()}
    eligible_nn = _nearest_distances(positions_mm, population, partition.indices)
    used_nn = _nearest_distances(positions_mm, population, used)
    report: dict[str, Any] = {
        'policy': cfg.split.policy,
        'distance_metric': (
            'Euclidean distance in mm to the nearest reference position. '
            'Population-specific comparisons restrict both sets to that population.'
        ),
        'eligible_position_nearest_neighbour_mm': eligible_nn,
        'serialized_used_position_nearest_neighbour_mm': used_nn,
    }
    if partition.region is not None:
        assert cfg.split.buffer_mm is not None
        eligible_min = min(pair['all']['min'] for pair in eligible_nn.values())
        written_min = min(pair['all']['min'] for pair in used_nn.values())
        if min(eligible_min, written_min) < cfg.split.buffer_mm:
            raise RuntimeError('Region split clearance is below split.buffer_mm')
        report['region'] = _region_report(cfg, partition, eligible_min, written_min)
        if partition.patch_containment_radius_mm is not None:
            if patch_diameter_mm is None:
                raise ValueError('patch containment needs the serialized patch diameters')
            report['region']['patch_containment'] = _patch_containment_report(
                cfg, partition, position_rows, patch_diameter_mm,
            )
    if cfg.split.policy == 'shared':
        report['shared'] = _shared_report(cfg, partition, position_rows, used)
    return report


def _region_report(
    cfg: Config, partition: SourcePartition, eligible_min: float, written_min: float,
) -> dict[str, Any]:
    region = partition.region
    assert region is not None
    total = sum(len(indices) for indices in partition.indices.values())
    retained: dict[str, Any] = {name: {} for name in SPLIT_NAMES}
    for pop, name in POPULATIONS:
        members = np.flatnonzero(partition.population == pop)
        bins = _depth_bin_indices(partition.positions_xyz[members], cfg.split.depth_bins)
        before = region.pre_buffer_split_labels[members]
        after = partition.split_labels[members]
        for split_id, split in enumerate(SPLIT_NAMES):
            n_before = int(np.sum(before == split_id))
            n_after = int(np.sum(after == split_id))
            retained[split][name] = {
                'pre_buffer': n_before,
                'post_buffer': n_after,
                'excluded': n_before - n_after,
                'equal_count_depth_bins': [
                    {
                        'bin': int(band),
                        'pre_buffer': int(np.sum((before == split_id) & (bins == band))),
                        'post_buffer': int(np.sum((after == split_id) & (bins == band))),
                    }
                    for band in np.unique(bins)
                ],
            }
    return {
        'sector_directions': region.sector_directions,
        'sector_split': region.sector_split,
        'sector_index': region.sector_index,
        'pre_buffer_split_labels': region.pre_buffer_split_labels,
        'rotation_seed': cfg.split.region_rotation_seed,
        'buffer_mm': cfg.split.buffer_mm,
        'minimum_clearance_mm': eligible_min,
        'minimum_serialized_used_clearance_mm': written_min,
        'retention_by_split_and_population': retained,
        'realized_position_fractions': {
            name: len(indices) / total for name, indices in partition.indices.items()
        },
    }


def _patch_containment_report(
    cfg: Config, partition: SourcePartition, centre_rows: dict[str, np.ndarray],
    diameter_mm: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Verify, from the written truth, that every patch stayed inside its split."""
    region = partition.region
    limit = partition.patch_containment_radius_mm
    assert region is not None and limit is not None
    labels = partition.split_labels
    positions = partition.positions_xyz
    unit = positions / np.linalg.norm(positions, axis=1, keepdims=True)
    radius_mm = np.linalg.norm(positions, axis=1) * 1000.0
    low, high = (float(value) for value in cfg.patch.diameter_range_mm)
    smallest = patch_capacity(limit, labels, low)
    largest = patch_capacity(limit, labels, high)
    report: dict[str, Any] = {
        'rule': (
            'A patch is centred only where its whole footprint, the cap of directions with '
            'half-angle radius / centre radius, lies inside the sectors of its own split and '
            'contains no source position of another split or of the buffer. The diameter is '
            'drawn first and the centre uniformly from the positions that can contain it.'
        ),
        'splits': {},
    }
    failed = []
    for split_id, name in enumerate(SPLIT_NAMES):
        centres = centre_rows[name].astype(np.int64, copy=False).ravel()
        half_angle = np.minimum(np.pi, (diameter_mm[name] / 2.0) / radius_mm[centres])
        foreign = np.flatnonzero(labels != split_id)
        # Chord length that corresponds to the footprint half-angle.
        reach = 2.0 * np.sin(half_angle / 2.0)
        clipped = (
            cKDTree(unit[foreign]).query(unit[centres])[0] <= reach
            if foreign.size else np.zeros(centres.size, dtype=bool)
        )
        beyond = half_angle > sector_margin_rad(
            unit[centres], np.full(centres.size, split_id), region.sector_directions,
            region.sector_split,
        )
        wrong_split = labels[centres] != split_id
        margin = limit[centres] - diameter_mm[name] / 2.0
        members = limit[labels == split_id]
        report['splits'][name] = {
            'eligible_positions': int(members.size),
            'positions_admitting_smallest_diameter': smallest[name],
            'positions_admitting_largest_diameter': largest[name],
            'largest_admissible_diameter_mm': 2.0 * float(members.max()),
            'patches': int(centres.size),
            'distinct_centres': int(np.unique(centres).size),
            'clipped_patches': int(np.sum(clipped)),
            'patches_beyond_sector_margin': int(np.sum(beyond)),
            'centres_outside_split': int(np.sum(wrong_split)),
            'minimum_margin_mm': float(margin.min()) if margin.size else None,
        }
        if np.any(clipped) or np.any(beyond) or np.any(wrong_split):
            failed.append(name)
    if failed:
        raise RuntimeError(
            'Patch containment failed for split(s) ' + ', '.join(failed)
            + '; staged dataset was not published'
        )
    return report


def _shared_report(
    cfg: Config, partition: SourcePartition, rows: dict[str, np.ndarray],
    used: dict[str, np.ndarray],
) -> dict[str, Any]:
    common = partition.split_labels == SHARED
    reuse = {}
    for query, reference in SPLIT_PAIRS:
        q = rows[query].reshape(rows[query].shape[0], -1)
        reused = np.isin(q, used[reference])
        any_reused = np.any(reused, axis=1)
        reuse[f'{query}_to_{reference}'] = {
            'eligible_position_overlap': int(common.sum()),
            'used_position_overlap': int(np.intersect1d(used[query], used[reference]).size),
            'query_windows': len(q),
            'query_windows_with_any_position_seen_in_reference': int(any_reused.sum()),
            'query_window_reuse_fraction': float(any_reused.mean()),
        }
    return {
        'design': 'One common finite pool, with independent split-specific window RNGs.',
        'requested_common_pool_fraction': cfg.split.shared_position_fraction,
        'realized_common_pool_fraction': float(common.mean()),
        'common_positions_by_population': {
            name: int(np.sum(common & (partition.population == pop))) for pop, name in POPULATIONS
        },
        'eligible_positions': {name: len(indices) for name, indices in partition.indices.items()},
        'used_unique_positions': {name: len(indices) for name, indices in used.items()},
        'observed_reuse': reuse,
    }
