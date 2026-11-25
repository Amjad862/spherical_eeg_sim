"""Region split with support containment for patch mode."""

import hashlib
import json
import warnings
from pathlib import Path

import h5py
import numpy as np
import pytest

from spherical_eeg_sim import Config, plan_position_split, run_pipeline
from spherical_eeg_sim.sources.partition import EXCLUDED, patch_capacity, sector_margin_rad
from spherical_eeg_sim.validation.patch_overlap import _Caps, _minimum_footprint_gap_mm
from scipy.spatial import cKDTree


SPLITS = ('train', 'val', 'test')


def patch_region_config(output_dir, positions=3000, count=6, buffer_mm=10.0, fmt='hdf5'):
    cfg = Config()
    cfg.output_dir = str(output_dir)
    cfg.source_mode = 'patch'
    cfg.master_seed = 862
    cfg.sources.n_cortical_positions = positions
    cfg.dataset.n_train = 60
    cfg.dataset.n_val = 30
    cfg.dataset.n_test = 30
    cfg.noise.background_pool_size = 16
    cfg.noise.n_background_sources = 8
    cfg.head.n_legendre = 40
    cfg.output.format = fmt
    cfg.output.chunk_samples = 16
    cfg.split.policy = 'region'
    cfg.split.region_count = count
    cfg.split.buffer_mm = buffer_mm
    cfg.split.region_rotation_seed = 24000
    return cfg


def _unit(positions):
    return positions / np.linalg.norm(positions, axis=1, keepdims=True)


def _directory_sha256(path):
    return {
        str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path(path).rglob('*')) if p.is_file()
    }


def test_patch_mode_accepts_region_and_rejects_shared(tmp_path):
    cfg = patch_region_config(tmp_path)
    plan = plan_position_split(cfg)
    assert plan.patch_containment_radius_mm is not None
    assert plan.allocation_basis.endswith('patch-support containment')
    # The patch pool is cortical only; the default subcortical fraction must not be required.
    assert cfg.sources.subcortical_fraction > 0 and np.all(plan.population == 0)

    cfg.split.policy = 'shared'
    cfg.split.region_count = cfg.split.buffer_mm = cfg.split.region_rotation_seed = None
    cfg.split.shared_position_fraction = 0.5
    with pytest.raises(ValueError, match="Patch mode supports split.policy='position' or 'region'"):
        plan_position_split(cfg)


def test_containment_radius_is_defined_exactly_for_eligible_positions(tmp_path):
    cfg = patch_region_config(tmp_path)
    plan = plan_position_split(cfg)
    radius = plan.patch_containment_radius_mm
    eligible = plan.split_labels >= 0
    assert radius.shape == plan.split_labels.shape
    assert np.any(plan.split_labels == EXCLUDED)
    assert np.all(np.isnan(radius[~eligible]))
    assert np.all(np.isfinite(radius[eligible])) and np.all(radius[eligible] >= 0)
    assert np.array_equal(radius, plan_position_split(cfg).patch_containment_radius_mm, equal_nan=True)
    # Point mode with the same region settings carries no containment field.
    cfg.source_mode = 'point'
    assert plan_position_split(cfg).patch_containment_radius_mm is None


def test_footprint_at_the_containment_radius_holds_only_its_own_split(tmp_path):
    plan = plan_position_split(patch_region_config(tmp_path))
    unit = _unit(plan.positions_xyz)
    radius_mm = np.linalg.norm(plan.positions_xyz, axis=1) * 1000.0
    labels = plan.split_labels
    rng = np.random.default_rng(1)
    for centre in rng.choice(np.flatnonzero(labels >= 0), size=300, replace=False):
        half_angle = plan.patch_containment_radius_mm[centre] / radius_mm[centre]
        inside = unit @ unit[centre] >= np.cos(half_angle)
        assert np.all(labels[inside] == labels[centre])


def test_footprint_at_the_containment_radius_stays_in_its_own_sectors(tmp_path):
    plan = plan_position_split(patch_region_config(tmp_path))
    unit = _unit(plan.positions_xyz)
    radius_mm = np.linalg.norm(plan.positions_xyz, axis=1) * 1000.0
    region = plan.region
    rng = np.random.default_rng(2)
    for centre in rng.choice(np.flatnonzero(plan.split_labels >= 0), size=150, replace=False):
        half_angle = plan.patch_containment_radius_mm[centre] / radius_mm[centre]
        # Uniform directions inside the cap, built in the tangent frame of the centre.
        axis = unit[centre]
        helper = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        e1 = np.cross(axis, helper)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(axis, e1)
        cos_polar = 1.0 - rng.uniform(size=2000) * (1.0 - np.cos(half_angle))
        sin_polar = np.sqrt(np.maximum(0.0, 1.0 - cos_polar ** 2))
        azimuth = rng.uniform(0.0, 2.0 * np.pi, size=2000)
        points = (cos_polar[:, None] * axis
                  + sin_polar[:, None] * (np.cos(azimuth)[:, None] * e1 + np.sin(azimuth)[:, None] * e2))
        sector = np.argmax(points @ region.sector_directions.T, axis=1)
        assert np.all(region.sector_split[sector] == plan.split_labels[centre])


def test_sector_margin_matches_the_bisector_of_two_sectors():
    directions = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    sector_split = np.array([0, 2], dtype=np.int8)
    polar = np.deg2rad([10.0, 60.0, 89.0])
    unit = np.stack([np.sin(polar), np.zeros(3), np.cos(polar)], axis=1)
    margin = sector_margin_rad(unit, np.zeros(3, dtype=np.int8), directions, sector_split)
    # Two antipodal sectors meet on the equator, so the margin is the latitude.
    assert np.allclose(margin, np.pi / 2 - polar, atol=1e-12, rtol=0)


def test_sectors_too_small_for_the_largest_patch_are_rejected(tmp_path):
    cfg = patch_region_config(tmp_path, count=24)
    with pytest.raises(ValueError, match=r'The conservative containment rule admits no centre for a '
                                         r'60 mm patch in the (train|val|test) split '
                                         r'\(largest admitted diameter'):
        plan_position_split(cfg)
    cfg.patch.diameter_range_mm = [10.0, 20.0]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        plan = plan_position_split(cfg)
    capacity = patch_capacity(plan.patch_containment_radius_mm, plan.split_labels, 20.0)
    assert all(count > 0 for count in capacity.values())


def test_few_centres_for_the_largest_patch_draw_a_warning(tmp_path):
    cfg = patch_region_config(tmp_path, positions=400, count=6, buffer_mm=5.0)
    with pytest.warns(RuntimeWarning, match='containment rule admits only .* for a 60 mm patch'):
        plan_position_split(cfg)


def _read_patches(path):
    with h5py.File(path, 'r') as file:
        truth = file['ground_truth']
        support = truth['patch_active_position_idx']
        return (truth['position_pool_idx'][:].ravel().astype(np.int64),
                truth['patch_diameter_mm'][:].ravel(),
                np.split(support['values'][:].astype(np.int64), support['row_splits'][1:-1]))


def test_patch_region_dataset_is_contained_and_separated(tmp_path):
    cfg = patch_region_config(tmp_path / 'dataset')
    summary = run_pipeline(cfg)
    out = Path(cfg.output_dir)
    report = json.loads(Path(summary['validation_report']).read_text())
    region = report['source_split']['region']
    containment = region['patch_containment']
    assert set(containment['splits']) == set(SPLITS)
    for name in SPLITS:
        entry = containment['splits'][name]
        assert entry['patches'] == getattr(cfg.dataset, f'n_{name}')
        assert entry['clipped_patches'] == 0
        assert entry['patches_beyond_sector_margin'] == 0
        assert entry['centres_outside_split'] == 0
        assert entry['minimum_margin_mm'] >= 0
        assert entry['largest_admissible_diameter_mm'] >= cfg.patch.diameter_range_mm[1]
    assert region['minimum_clearance_mm'] >= cfg.split.buffer_mm
    assert region['minimum_serialized_used_clearance_mm'] >= cfg.split.buffer_mm
    for pair in ('test_to_train', 'test_to_val', 'val_to_train', 'test_to_train_or_val'):
        assert report['patch_overlap']['pairs'][pair]['best_cap_iou']['max'] == 0.0
        assert report['patch_overlap']['pairs'][pair]['minimum_footprint_gap_mm'] > 0
    names = [check['name'] for check in report['distributions']['checks']]
    assert 'patch_orientation_radial' in names and 'orientation_s2_moments' not in names

    with h5py.File(out / 'source_pool.h5', 'r') as pool:
        positions = pool['positions_xyz'][:]
        labels = pool['split'][:]
        limit = pool['patch_containment_radius_mm'][:]
    plan = plan_position_split(cfg)
    assert np.array_equal(limit, plan.patch_containment_radius_mm, equal_nan=True)
    unit = _unit(positions)
    radius_mm = np.linalg.norm(positions, axis=1) * 1000.0
    low, high = cfg.patch.diameter_range_mm
    for split_id, name in enumerate(SPLITS):
        centres, diameters, supports = _read_patches(out / f'{name}.h5')
        assert diameters.min() < low + 0.25 * (high - low)
        assert diameters.max() > high - 0.25 * (high - low)
        assert np.all(limit[centres] >= diameters / 2)
        for centre, diameter, support in zip(centres, diameters, supports):
            # With this sparse pool the density cap never binds, so the written
            # support must equal the footprint taken from the complete pool.
            half_angle = (diameter / 2) / radius_mm[centre]
            complete = np.flatnonzero(unit @ unit[centre] >= np.cos(half_angle))
            assert np.array_equal(np.sort(support), complete)
            assert np.all(labels[support] == split_id)


def test_patch_region_output_is_independent_of_chunks_and_workers(tmp_path):
    first = patch_region_config(tmp_path / 'a')
    second = patch_region_config(tmp_path / 'b')
    first.output.chunk_samples, first.output.n_workers = 7, 1
    second.output.chunk_samples, second.output.n_workers = 16, 2
    run_pipeline(first)
    run_pipeline(second)
    assert _directory_sha256(tmp_path / 'a') == _directory_sha256(tmp_path / 'b')


@pytest.mark.parametrize('fmt', ['npz', 'mat'])
def test_other_formats_store_the_containment_radius(tmp_path, fmt):
    if fmt == 'mat':
        pytest.importorskip('hdf5storage')
    cfg = patch_region_config(tmp_path / 'dataset', fmt=fmt)
    summary = run_pipeline(cfg)
    expected = plan_position_split(cfg).patch_containment_radius_mm
    out = Path(cfg.output_dir)
    if fmt == 'npz':
        with np.load(out / 'source_pool.npz', allow_pickle=False) as pool:
            stored = pool['patch_containment_radius_mm']
    else:
        with h5py.File(out / 'source_pool.mat', 'r') as pool:
            stored = pool['patch_containment_radius_mm'][:].ravel()
    assert np.array_equal(stored, expected, equal_nan=True)
    report = json.loads(Path(summary['validation_report']).read_text())
    assert all(entry['clipped_patches'] == 0
               for entry in report['source_split']['region']['patch_containment']['splits'].values())


def test_default_policy_patch_dataset_is_unchanged_in_layout(tmp_path):
    cfg = patch_region_config(tmp_path / 'dataset')
    cfg.split.policy = 'position'
    cfg.split.region_count = cfg.split.buffer_mm = cfg.split.region_rotation_seed = None
    summary = run_pipeline(cfg)
    with h5py.File(Path(cfg.output_dir) / 'source_pool.h5', 'r') as pool:
        assert 'patch_containment_radius_mm' not in pool
    report = json.loads(Path(summary['validation_report']).read_text())
    assert 'region' not in report['source_split']
    pairs = report['patch_overlap']['pairs']
    # Interleaved meshes overlap, so no footprint gap is reported.
    assert pairs['test_to_train']['best_cap_iou']['max'] > 0
    assert pairs['test_to_train']['minimum_footprint_gap_mm'] is None


def test_minimum_footprint_gap_of_known_caps():
    radius_m = 0.075
    polar = np.deg2rad([0.0, 40.0, 90.0])
    centres = radius_m * np.stack([np.sin(polar), np.zeros(3), np.cos(polar)], axis=1)
    query = _Caps.from_truth(centres[:1], np.array([20.0]))
    reference = _Caps.from_truth(centres[1:], np.array([30.0, 60.0]))
    gap = _minimum_footprint_gap_mm(query, reference, cKDTree(reference.unit))
    expected = min(np.deg2rad(40.0) * 75.0 - 10.0 - 15.0, np.deg2rad(90.0) * 75.0 - 10.0 - 30.0)
    assert gap == pytest.approx(expected, abs=1e-9)
