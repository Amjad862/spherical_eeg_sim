"""Regression checks for split eligibility and serialized split audits."""

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from spherical_eeg_sim import Config, plan_position_split, run_pipeline
from spherical_eeg_sim import pipeline
from test_release_arrays import tiny_config


def _region(cfg, count=12, clearance=1.0, seed=24000):
    cfg.split.policy = 'region'
    cfg.split.region_count = count
    cfg.split.buffer_mm = clearance
    cfg.split.region_rotation_seed = seed
    return cfg


@pytest.mark.parametrize(('seed', 'pre_digest', 'post_digest'), [
    (862, 'b33ce97c311dc777b4c201f024b0765031ee03db996de30466600e70bfec7738',
     'ac07191653afc00d2974090a7eec5e86e884473890c7e0e3fa2b5813706e99c2'),
    (863, '199d977fb2ee7ebc2c7671b68aaaff30f1820fb842e632a4266a5d376dc57841',
     'f6d59adf161ef3108526c0b74ea393e4d75f82c17b5c2d862d0be1e7e418f54b'),
    (864, 'e5fc81a913be06835d2eb1bfc49d893645d69f311fcef8a0cae60e4337c64adc',
     'e602c9ee83a1c7734d64c5056c434e88d428d63d7cc3ae8bc6a8dfcf96d39eb6'),
])
def test_main_region_assignments_match_geometry_pilot(seed, pre_digest, post_digest):
    # Independently captured from the pre-experiment geometry pilot.
    cfg = _region(Config(), 24, 15.0, 24000 + seed - 862)
    cfg.position_seed = seed
    cfg.sources.n_cortical_positions = 8000
    cfg.sources.n_subcortical_positions = 1500
    cfg.dataset.n_train = 20_000
    cfg.dataset.n_val = 2000
    cfg.dataset.n_test = 3000
    plan = plan_position_split(cfg)
    assert hashlib.sha256(plan.region.pre_buffer_split_labels.tobytes()).hexdigest() == pre_digest
    assert hashlib.sha256(plan.split_labels.tobytes()).hexdigest() == post_digest


def test_region_uses_the_same_position_and_background_pools(tmp_path, monkeypatch):
    backgrounds = []
    original = pipeline.compute_background_leadfield_pool

    def capture_background(**kwargs):
        pool = original(**kwargs)
        backgrounds.append(pool.copy())
        return pool

    monkeypatch.setattr(pipeline, 'compute_background_leadfield_pool', capture_background)
    cfg = tiny_config(tmp_path / 'position', 'point', 'hdf5')
    position = plan_position_split(cfg)
    run_pipeline(cfg)
    cfg.output_dir = str(tmp_path / 'region')
    _region(cfg)
    region = plan_position_split(cfg)
    summary = run_pipeline(cfg)
    assert np.array_equal(region.positions_xyz, position.positions_xyz)
    assert np.array_equal(region.population, position.population)
    assert np.array_equal(backgrounds[0], backgrounds[1])
    assert np.array_equal(region.split_labels, plan_position_split(cfg).split_labels)
    with h5py.File(Path(cfg.output_dir) / 'source_pool.h5') as pool:
        assert np.array_equal(pool['split'][:], region.split_labels)
        assert 'split_membership' not in pool
    report = json.loads(Path(summary['validation_report']).read_text())['source_split']
    assert report['region']['minimum_clearance_mm'] >= cfg.split.buffer_mm
    assert report['region']['minimum_serialized_used_clearance_mm'] >= cfg.split.buffer_mm
    assert report['region']['sector_index'] == region.region.sector_index.tolist()


@pytest.mark.parametrize('mode', ['point', 'multi_point'])
def test_region_requires_each_configured_population(tmp_path, mode):
    cfg = _region(tiny_config(tmp_path, mode, 'hdf5'), clearance=1000.0)
    with pytest.raises(ValueError, match='no eligible cortical positions in train'):
        plan_position_split(cfg)
    cfg.split.buffer_mm = 1.0
    cfg.sources.subcortical_fraction = 1.0
    cfg.sources.n_cortical_positions = 0
    plan = plan_position_split(cfg)
    assert all(np.all(plan.population[indices] == 1) for indices in plan.indices.values())


def test_patch_rejects_the_shared_policy(tmp_path):
    # The region policy for patches is covered in test_patch_region_split.py.
    cfg = tiny_config(tmp_path, 'patch', 'hdf5')
    cfg.split.policy = 'shared'
    cfg.split.shared_position_fraction = 0.5
    with pytest.raises(ValueError, match="Patch mode supports split.policy='position' or 'region'"):
        plan_position_split(cfg)


@pytest.mark.parametrize('fmt', ['hdf5', 'npz', 'mat'])
def test_shared_core_and_realized_reuse(tmp_path, fmt):
    if fmt == 'mat':
        pytest.importorskip('hdf5storage')
    cfg = tiny_config(tmp_path / 'shared', 'multi_point', fmt)
    position = plan_position_split(cfg)
    cfg.split.policy = 'shared'
    with pytest.raises(ValueError, match='shared_position_fraction'):
        plan_position_split(cfg)
    cfg.split.shared_position_fraction = 0.2
    plan = plan_position_split(cfg)
    assert np.array_equal(plan.positions_xyz, position.positions_xyz)
    assert np.all(plan.membership.sum(axis=1) == np.where(plan.split_labels == -2, 3, 1))
    common = np.flatnonzero(plan.split_labels == -2)
    assert len(common) == 18
    for name in plan.indices:
        assert np.array_equal(plan.indices[name], np.union1d(position.indices[name], common))

    summary = run_pipeline(cfg)
    suffix = {'hdf5': 'h5', 'npz': 'npz', 'mat': 'mat'}[fmt]
    for stem in ('source_pool', 'leadfield'):
        path = Path(cfg.output_dir) / f'{stem}.{suffix}'
        if fmt == 'npz':
            with np.load(path, allow_pickle=False) as pool:
                assert np.array_equal(pool['split_membership'], plan.membership)
        else:
            with h5py.File(path) as pool:
                data = pool['split_membership'][:]
                assert np.array_equal(data.T if fmt == 'mat' else data, plan.membership)

    train = pipeline._read_serialized_ground_truth(Path(summary['splits']['train']), ['position_pool_idx'])
    test = pipeline._read_serialized_ground_truth(Path(summary['splits']['test']), ['position_pool_idx'])
    train_locations = train['position_pool_idx'].ravel()
    test_locations = test['position_pool_idx'].reshape(cfg.dataset.n_test, -1)
    n_reused = np.any(np.isin(test_locations, train_locations[train_locations >= 0]), axis=1).sum()
    report = json.loads(Path(summary['validation_report']).read_text())['source_split']['shared']
    assert report['realized_common_pool_fraction'] == 0.2
    assert report['observed_reuse']['test_to_train']['eligible_position_overlap'] == len(common)
    assert report['observed_reuse']['test_to_train']['query_windows_with_any_position_seen_in_reference'] == n_reused


def test_shared_location_does_not_reuse_eeg_windows(tmp_path):
    cfg = tiny_config(tmp_path / 'shared', 'point', 'hdf5')
    cfg.sources.n_cortical_positions = 1
    cfg.sources.n_subcortical_positions = 0
    cfg.sources.subcortical_fraction = 0.0
    cfg.split.policy = 'shared'
    cfg.split.shared_position_fraction = 1.0
    result = run_pipeline(cfg)
    with h5py.File(result['splits']['train']) as train, h5py.File(result['splits']['test']) as test:
        assert train['ground_truth/position_pool_idx'][0] == test['ground_truth/position_pool_idx'][0] == 0
        assert not np.array_equal(train['eeg'][0], test['eeg'][0])


def test_shared_fraction_uses_decimal_floor_and_warns_for_empty_population(tmp_path):
    cfg = tiny_config(tmp_path, 'point', 'hdf5')
    cfg.sources.n_cortical_positions = 100
    cfg.sources.n_subcortical_positions = 3
    cfg.split.policy = 'shared'
    cfg.split.shared_position_fraction = 0.29
    with pytest.warns(RuntimeWarning, match='zero positions in population 1'):
        plan = plan_position_split(cfg)
    assert np.sum((plan.population == 0) & (plan.split_labels == -2)) == 29


def test_nondefault_split_files_identify_policy(tmp_path):
    cfg = tiny_config(tmp_path / 'shared', 'point', 'hdf5')
    cfg.split.policy = 'shared'
    cfg.split.shared_position_fraction = 0.2
    result = run_pipeline(cfg)
    with h5py.File(result['splits']['test']) as data:
        assert data['metadata'].attrs['split_policy'] == 'shared'
    assert 'Split policy     : shared' in (Path(cfg.output_dir) / 'README.txt').read_text()
