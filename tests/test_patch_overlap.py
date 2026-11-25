"""Analytic and brute-force checks for the patch overlap diagnostic."""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import cKDTree

from spherical_eeg_sim import run_pipeline
from spherical_eeg_sim.validation.patch_overlap import (
    _Caps, _angular_distance, _best_match, _cap_iou, _query_selection, patch_overlap_report,
)
from spherical_eeg_sim.validation import patch_overlap
from test_release_arrays import tiny_config


def test_analytic_cap_intersections():
    a = np.array([0.3, np.pi, np.pi / 2, 2.0, 0.1, 0.2])
    b = np.array([0.3, 1.0, np.pi / 2, 2.0, 0.2, 0.2])
    d = np.array([0.0, 2.0, np.pi / 2, 3.0, 0.02, 0.4])
    assert np.allclose(_cap_iou(a, b, d), [
        1.0, np.sin(0.5) ** 2, 1 / 3, -np.cos(2.0),
        (np.sin(0.05) / np.sin(0.1)) ** 2, 0.0,
    ], rtol=0, atol=1e-13)


def _caps(unit, angles):
    return _Caps.from_truth(unit / 10.0, angles * 200.0)


def test_large_cap_pruning_counterexample_is_exact():
    query = _caps(np.array([[0.0, 0.0, 1.0]]), np.array([2.5]))
    distances = np.r_[np.full(16, 0.01), 3.0]
    ref = _caps(np.c_[np.sin(distances), np.zeros(17), np.cos(distances)], np.r_[np.full(16, 2.0), 2.8])
    actual, _ = _best_match(query, ref, cKDTree(ref.unit), np.arange(1), False)
    assert actual[0] == pytest.approx(-(np.cos(2.5) + np.cos(2.8)) / 2, abs=1e-13)


@pytest.mark.parametrize('max_angle', [0.426, np.pi / 2, np.pi])
def test_best_matches_agree_with_exhaustive_search(max_angle):
    rng = np.random.default_rng(267)
    units = rng.normal(size=(183, 3))
    units /= np.linalg.norm(units, axis=1, keepdims=True)
    angles = rng.uniform(0.06, max_angle, size=len(units))
    angles[-1] = max_angle
    q = _caps(units[:37], angles[:37])
    r = _caps(units[37:], angles[37:])
    d = _angular_distance(np.linalg.norm(q.unit[:, None] - r.unit[None, :], axis=-1))
    expected = _cap_iou(q.half_angles[:, None], r.half_angles, d).max(axis=1)
    actual, _ = _best_match(q, r, cKDTree(r.unit), np.arange(len(q.unit)), max_angle <= np.pi / 2)
    assert np.allclose(actual, expected, rtol=0, atol=1e-13)


def test_patch_report_minimizes_3d_and_angular_distance_independently():
    xyz = np.array([[0.07, 0.0, 0.0], [0.07999, 0.001, 0.0], [0.08, 0.0, 0.0]])
    truth = {
        'train': {'position_pool_idx': np.array([0, 1]), 'patch_diameter_mm': np.array([40., 40.])},
        'val': {'position_pool_idx': np.array([0]), 'patch_diameter_mm': np.array([40.])},
        'test': {'position_pool_idx': np.array([2]), 'patch_diameter_mm': np.array([40.])},
    }
    result = patch_overlap_report(xyz, truth)['pairs']['test_to_train']
    assert result['nearest_centre_geodesic_mm']['min'] == 0.0
    assert result['nearest_centre_euclidean_mm']['min'] == pytest.approx(np.hypot(1.0, 0.01))
    assert result['iou_query_patches'] == 1


def test_query_budget_is_deterministic():
    candidates = np.arange(1, 101)
    selected = _query_selection(candidates, 300)
    assert 1 < selected.size < candidates.size
    assert candidates[selected].sum() <= 300
    assert np.array_equal(selected, _query_selection(candidates, 300))


def test_dense_custom_query_streams_reference_batches(monkeypatch):
    monkeypatch.setattr(patch_overlap, '_BATCH_PAIR_LIMIT', 3)
    q = _caps(np.array([[0., 0., 1.]]), np.array([np.pi]))
    r = _caps(np.tile([0., 0., 1.], (20, 1)), np.full(20, np.pi))
    actual, counts = _best_match(q, r, cKDTree(r.unit), np.array([0]), False)
    assert actual.tolist() == [1.0]
    assert counts['largest_materialized_candidate_batch'] <= 3
    assert counts['exact_iou_evaluations'] == (
        counts['seed_iou_evaluations'] + counts['subsequent_candidate_iou_evaluations']
    )


def test_subset_report_uses_identical_test_queries_against_both_references(monkeypatch):
    monkeypatch.setattr(patch_overlap, '_SMALL_CAP_PAIR_BUDGET', 10)
    positions = np.array([[0.08, 0., 0.]])
    truth = {
        name: {'position_pool_idx': np.zeros(count, dtype=int), 'patch_diameter_mm': np.full(count, 40.)}
        for name, count in (('train', 6), ('val', 4), ('test', 5))
    }
    result = patch_overlap_report(positions, truth)['pairs']
    for pair in ('test_to_train', 'test_to_val', 'test_to_train_or_val'):
        assert result[pair]['iou_scope'] == 'deterministic query subset'
        assert result[pair]['iou_query_patches'] == 1
        assert result[pair]['best_cap_iou']['median'] == 1.0


def test_patch_report_is_deterministic_and_timing_is_operational(tmp_path):
    cfg = tiny_config(tmp_path / 'first', 'patch', 'hdf5')
    first = run_pipeline(cfg)
    cfg.output_dir = str(tmp_path / 'second')
    second = run_pipeline(cfg)
    assert Path(first['validation_report']).read_bytes() == Path(second['validation_report']).read_bytes()
    report = json.loads(Path(first['validation_report']).read_text())
    runtime = json.loads(Path(first['runtime_provenance']).read_text())
    assert 'patch_overlap' in report
    assert 'patch_overlap_wall_time_seconds' not in report
    assert runtime['patch_overlap_wall_time_seconds'] >= 0.0
    cfg.patch.report_overlap = False
    cfg.output_dir = str(tmp_path / 'disabled')
    disabled = run_pipeline(cfg)
    assert 'patch_overlap' not in json.loads(Path(disabled['validation_report']).read_text())
