"""Exact continuous spherical-cap overlap of serialized patch footprints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from .split_geometry import distance_summary


_FULL_SPHERE = 4 * np.pi
_HEMISPHERE = np.pi / 2
_SMALL_CAP_PAIR_BUDGET = 1_200_000_000
_GENERAL_CAP_PAIR_BUDGET = 8_000_000
_BATCH_PAIR_LIMIT = 500_000
_ROUNDING_MARGIN = 64 * np.finfo(float).eps


@dataclass(frozen=True)
class _Caps:
    xyz_mm: np.ndarray
    unit: np.ndarray
    radii_mm: np.ndarray
    half_angles: np.ndarray

    @classmethod
    def from_truth(cls, xyz_m: np.ndarray, diameter_mm: np.ndarray) -> _Caps:
        radii_m = np.linalg.norm(xyz_m, axis=1)
        radii_mm = radii_m * 1000.0
        return cls(
            xyz_m * 1000.0, xyz_m / radii_m[:, None], radii_mm,
            np.minimum(np.pi, (diameter_mm / 2) / radii_mm),
        )


def _area(angle: np.ndarray) -> np.ndarray:
    return _FULL_SPHERE * np.sin(angle / 2) ** 2


def _small_cap_intersection(a: np.ndarray, b: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Intersection area for two caps of at most a hemisphere each."""
    area = np.zeros_like(d)
    contained = d <= np.abs(a - b)
    area[contained] = _area(np.minimum(a[contained], b[contained]))
    lens = (~contained) & (d < a + b)
    a, b, d = a[lens], b[lens], d[lens]

    # Two cap sectors minus twice the area of their spherical triangle.
    # Half-angle formulae and L'Huilier's formula avoid subtracting nearly
    # equal cosines for small caps or for centres close to each other.
    s = (a + b + d) / 2
    sa = np.maximum(0.0, s - a)
    sb = np.maximum(0.0, s - b)
    sd = np.maximum(0.0, s - d)
    angle_a = 2 * np.arctan2(np.sqrt(np.sin(sa) * np.sin(sd)), np.sqrt(np.sin(s) * np.sin(sb)))
    angle_b = 2 * np.arctan2(np.sqrt(np.sin(sb) * np.sin(sd)), np.sqrt(np.sin(s) * np.sin(sa)))
    excess = 4 * np.arctan(np.sqrt(np.tan(s / 2) * np.tan(sa / 2) * np.tan(sb / 2) * np.tan(sd / 2)))
    area[lens] = (_area(a) * angle_a + _area(b) * angle_b) / np.pi - 2 * excess
    return area


def _cap_iou(a: np.ndarray, b: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """Exact unit-sphere cap IoU, including caps larger than a hemisphere."""
    a, b, distance = np.broadcast_arrays(
        np.asarray(a, dtype=float), np.asarray(b, dtype=float), np.asarray(distance, dtype=float),
    )
    a_big, b_big = a > _HEMISPHERE, b > _HEMISPHERE
    complement_distance = np.where(a_big ^ b_big, np.pi - distance, distance)
    intersection = _small_cap_intersection(
        np.where(a_big, np.pi - a, a), np.where(b_big, np.pi - b, b), complement_distance,
    )
    area_a, area_b = _area(a), _area(b)
    intersection = np.select(
        (a_big & b_big, a_big, b_big),
        (area_a + area_b - _FULL_SPHERE + intersection,
         area_b - intersection, area_a - intersection),
        default=intersection,
    )
    intersection = np.clip(intersection, 0.0, np.minimum(area_a, area_b))
    union = area_a + area_b - intersection
    return np.divide(intersection, union, out=np.zeros_like(union), where=union > 0)


def _angular_distance(chord: np.ndarray) -> np.ndarray:
    return 2 * np.arcsin(np.minimum(chord / 2, 1.0))


def _chord_for_reach(reach: np.ndarray) -> np.ndarray:
    return 2 * np.sin(np.minimum(reach, np.pi) / 2) + _ROUNDING_MARGIN


def _lens_iou_upper_bound(
    a: np.ndarray, b: np.ndarray, distance: np.ndarray,
    area_a: np.ndarray, area_b: np.ndarray,
) -> np.ndarray:
    """Lens bound for caps of at most a hemisphere, where a + b <= pi."""
    # For a partial lens, a subcap of radius (d + a - b)/2 can be placed
    # inside cap a but outside cap b, opposite cap b along the centre arc.
    # Since a+b <= pi, that construction does not wrap around the antipode.
    # The same argument applies to b. Containment uses only the size bound.
    rho_a = np.maximum(0.0, (distance + a - b) / 2)
    rho_b = np.maximum(0.0, (distance + b - a) / 2)
    size_limit = np.minimum(area_a, area_b)
    overlap = np.where(
        distance <= np.abs(a - b), size_limit,
        np.maximum(0.0, np.minimum(area_a - _area(rho_a), area_b - _area(rho_b))),
    )
    union = area_a + area_b - overlap
    return np.divide(overlap, union, out=np.zeros_like(union), where=union > 0)


def _scores_for_pairs(
    q: _Caps, r: _Caps, query_indices: np.ndarray, reference_indices: np.ndarray,
    lower: np.ndarray, small_caps: bool,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Filter candidates with conservative bounds and evaluate the remainder."""
    a, b = q.half_angles[query_indices], r.half_angles[reference_indices]
    area_a, area_b = _area(a), _area(b)
    larger = np.maximum(area_a, area_b)
    size_bound = np.divide(np.minimum(area_a, area_b), larger, out=np.zeros_like(larger), where=larger > 0)
    keep = size_bound + _ROUNDING_MARGIN >= lower
    a, b, area_a, area_b = a[keep], b[keep], area_a[keep], area_b[keep]
    qi, ri = query_indices[keep], reference_indices[keep]
    distance = _angular_distance(np.linalg.norm(q.unit[qi] - r.unit[ri], axis=1))
    if small_caps:
        keep = _lens_iou_upper_bound(a, b, distance, area_a, area_b) + _ROUNDING_MARGIN >= lower[keep]
        a, b, qi, distance = a[keep], b[keep], qi[keep], distance[keep]
    return qi, _cap_iou(a, b, distance), len(qi)


def _best_match(
    q: _Caps, r: _Caps, tree: cKDTree, selected: np.ndarray, small_caps: bool,
) -> tuple[np.ndarray, dict[str, int]]:
    """Compute each selected query's global maximum against the full reference."""
    nq, nr = len(q.unit), len(r.unit)
    best = np.zeros(nq)
    balls = exact = 0
    largest_batch = 0
    max_reference_angle = r.half_angles.max()
    nearest = min(16, nr)
    for offset in range(0, len(selected), 512):
        block = selected[offset:offset + 512]
        chord, near = tree.query(q.unit[block], k=list(range(1, nearest + 1)))
        lower = _cap_iou(q.half_angles[block, None], r.half_angles[near], _angular_distance(chord)).max(axis=1)
        best[block] = lower
        reach = q.half_angles[block] + max_reference_angle
        if small_caps:
            # A lens contains a non-overlapping subcap of radius >= d/2 in
            # one of the two caps. To beat lower, its area must be below
            # query_area * (1 - lower) / lower. Invert the stable cap area.
            base = np.maximum(0.0, lower - _ROUNDING_MARGIN)
            bound = np.divide(1 - base, base, out=np.full_like(base, np.inf), where=base > 0)
            np.multiply(bound, np.sin(q.half_angles[block] / 2) ** 2, out=bound, where=base > 0)
            reach = np.minimum(reach, 4 * np.arcsin(np.sqrt(np.clip(bound, 0.0, 1.0))))
        chord_reach = _chord_for_reach(reach)
        sizes = tree.query_ball_point(q.unit[block], chord_reach, return_length=True)
        bounds = np.r_[0, np.cumsum(sizes)]
        balls += int(bounds[-1])

        start = 0
        while start < len(block):
            stop = int(np.searchsorted(bounds, bounds[start] + _BATCH_PAIR_LIMIT, side='right')) - 1
            if stop == start:
                # A single dense custom query still has bounded memory.
                query = block[start]
                for ref_start in range(0, nr, _BATCH_PAIR_LIMIT):
                    ri = np.arange(ref_start, min(nr, ref_start + _BATCH_PAIR_LIMIT))
                    qi = np.full(ri.size, query)
                    _, values, count = _scores_for_pairs(q, r, qi, ri, np.full(ri.size, best[query]), small_caps)
                    best[query] = max(best[query], values.max(initial=0.0))
                    exact += count
                    largest_batch = max(largest_batch, len(ri))
                start += 1
                continue
            neighbours = tree.query_ball_point(
                q.unit[block[start:stop]], chord_reach[start:stop], return_sorted=False,
            )
            qi = np.repeat(block[start:stop], sizes[start:stop])
            ri = np.fromiter((idx for row in neighbours for idx in row), dtype=np.int64, count=len(qi))
            updated, values, count = _scores_for_pairs(q, r, qi, ri, best[qi], small_caps)
            np.maximum.at(best, updated, values)
            exact += count
            largest_batch = max(largest_batch, len(qi))
            start = stop
    return best[selected], {
        'evaluated_ball_candidates': balls,
        'exact_iou_evaluations': exact + len(selected) * nearest,
        'seed_iou_evaluations': len(selected) * nearest,
        'subsequent_candidate_iou_evaluations': exact,
        'largest_materialized_candidate_batch': largest_batch,
    }


def _minimum_footprint_gap_mm(q: _Caps, r: _Caps, tree: cKDTree) -> float:
    """Smallest edge-to-edge gap between a query and a reference footprint.

    The gap of a pair is its angular centre separation minus both half-angles,
    times the query centre radius. Nearest centres are examined in growing
    batches; a query is settled once the largest reference cap placed at the
    last examined distance could not come closer than the best gap found.
    """
    n_ref = len(r.unit)
    best = np.full(len(q.unit), np.inf)
    pending = np.arange(len(q.unit))
    largest = r.half_angles.max()
    k = min(16, n_ref)
    while pending.size:
        chord, near = tree.query(q.unit[pending], k=k)
        if k == 1:
            chord, near = chord[:, None], near[:, None]
        angle = _angular_distance(chord)
        gap = (angle - r.half_angles[near] - q.half_angles[pending, None]).min(axis=1)
        best[pending] = np.minimum(best[pending], gap)
        settled = (angle[:, -1] - largest - q.half_angles[pending] >= best[pending]) | (k == n_ref)
        pending = pending[~settled]
        k = min(n_ref, 4 * k)
    return float((best * q.radii_mm).min())


def _possible_pairs(q: _Caps, r: _Caps, tree: cKDTree) -> np.ndarray:
    return tree.query_ball_point(
        q.unit, _chord_for_reach(q.half_angles + r.half_angles.max()), return_length=True,
    )


def _query_selection(possible_pairs: np.ndarray, budget: int) -> np.ndarray:
    n = len(possible_pairs)
    total = int(possible_pairs.sum())
    count = n if total <= budget else max(1, n * budget // total)
    while True:
        selected = np.arange(count, dtype=np.int64) * n // count
        work = int(possible_pairs[selected].sum())
        if work <= budget or count == 1:
            return selected
        count = max(1, min(count - 1, count * budget // work))


def patch_overlap_report(
    positions_xyz: np.ndarray, split_truth: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    """Compare centres and nominal recruitment caps for all split pairs."""
    caps = {
        name: _Caps.from_truth(
            positions_xyz[truth['position_pool_idx'].astype(np.int64).ravel()],
            truth['patch_diameter_mm'].ravel(),
        )
        for name, truth in split_truth.items()
    }
    trees = {name: cKDTree(caps[name].unit) for name in ('train', 'val')}
    xyz_trees = {name: cKDTree(caps[name].xyz_mm) for name in trees}
    pairs = (('test', 'train'), ('test', 'val'), ('val', 'train'))
    possible = {(q, r): _possible_pairs(caps[q], caps[r], trees[r]) for q, r in pairs}
    small = all(np.all(split.half_angles <= _HEMISPHERE) for split in caps.values())
    budget = _SMALL_CAP_PAIR_BUDGET if small else _GENERAL_CAP_PAIR_BUDGET
    selections = {
        'test': _query_selection(possible['test', 'train'] + possible['test', 'val'], budget),
        'val': _query_selection(possible['val', 'train'], budget),
    }
    report = {}
    gaps: dict[tuple[str, str], Any] = {}
    test_best: list[np.ndarray] = []
    test_angular: list[np.ndarray] = []
    test_euclidean: list[np.ndarray] = []
    for q_name, r_name in pairs:
        q, r = caps[q_name], caps[r_name]
        selected = selections[q_name]
        angular = _angular_distance(trees[r_name].query(q.unit)[0]) * q.radii_mm
        euclidean = xyz_trees[r_name].query(q.xyz_mm)[0]
        best, counts = _best_match(q, r, trees[r_name], selected, small)
        # Reported only when every query was scored and none overlaps a reference.
        gap = (
            _minimum_footprint_gap_mm(q, r, trees[r_name])
            if len(selected) == len(q.unit) and not np.any(best > 0) else None
        )
        gaps[q_name, r_name] = gap
        report[f'{q_name}_to_{r_name}'] = {
            'nearest_centre_geodesic_mm': distance_summary(angular),
            'nearest_centre_euclidean_mm': distance_summary(euclidean),
            'best_cap_iou': distance_summary(best),
            'minimum_footprint_gap_mm': gap,
            'reference_patches': len(r.unit),
            'query_patches': len(q.unit),
            'iou_query_patches': len(selected),
            'iou_scope': 'all query patches' if len(selected) == len(q.unit) else 'deterministic query subset',
            'possible_overlap_pair_upper_count': int(possible[q_name, r_name].sum()),
            **counts,
        }
        if q_name == 'test':
            test_best.append(best)
            test_angular.append(angular)
            test_euclidean.append(euclidean)
    report['test_to_train_or_val'] = {
        'nearest_centre_geodesic_mm': distance_summary(np.minimum(*test_angular)),
        'nearest_centre_euclidean_mm': distance_summary(np.minimum(*test_euclidean)),
        'best_cap_iou': distance_summary(np.maximum(*test_best)),
        'minimum_footprint_gap_mm': (
            None if gaps['test', 'train'] is None or gaps['test', 'val'] is None
            else min(gaps['test', 'train'], gaps['test', 'val'])
        ),
        'query_patches': len(caps['test'].unit),
        'iou_query_patches': len(selections['test']),
        'iou_scope': report['test_to_train']['iou_scope'],
    }
    return {
        'footprint': 'Continuous cap on the unit sphere of directions, half-angle min(diameter / (2 * centre radius), pi).',
        'geodesic_distance': 'Angular nearest-centre separation times the query centre radius.',
        'euclidean_distance': 'Independently minimized Cartesian nearest-centre distance.',
        'footprint_gap': (
            'Smallest angular centre separation minus both half-angles, times the query centre '
            'radius; reported only when all query patches were scored and none overlaps a '
            'reference patch.'
        ),
        'algorithm': 'exact small-cap lens and size pruning' if small else 'exact general caps with size pruning',
        'iou_candidate_budget': budget,
        'iou_reference_scope': 'All reference patches for each scored query.',
        'pairs': report,
    }
