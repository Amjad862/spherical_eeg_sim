"""
End-to-end pipeline integration test.

Runs the full pipeline at minimal scale and verifies output integrity.
"""

import sys as _sys
from pathlib import Path as _Path
_repo_root = _Path(__file__).resolve().parent.parent
if str(_repo_root) not in _sys.path:
    _sys.path.insert(0, str(_repo_root))

from pathlib import Path
import hashlib
import json
import tempfile

import h5py
import numpy as np
import pytest

from spherical_eeg_sim import Config, __version__, run_pipeline
from spherical_eeg_sim.pipeline import (
    _directory_checksum_complete,
    _write_rollback_marker,
)


def _assert_h5_node_equal(left, right):
    """Recursively compare dense datasets and CSR ground-truth groups."""
    if isinstance(left, h5py.Dataset):
        assert isinstance(right, h5py.Dataset)
        assert np.array_equal(
            np.nan_to_num(left[:]), np.nan_to_num(right[:])
        )
        return
    assert isinstance(left, h5py.Group)
    assert isinstance(right, h5py.Group)
    assert set(left.keys()) == set(right.keys())
    assert dict(left.attrs) == dict(right.attrs)
    for key in left:
        _assert_h5_node_equal(left[key], right[key])


def _tiny_config(output_dir: str, source_mode: str = 'point') -> Config:
    cfg = Config()
    cfg.source_mode = source_mode
    cfg.dataset.n_train = 100
    cfg.dataset.n_val = 30
    cfg.dataset.n_test = 30
    cfg.sources.n_cortical_positions = 200
    cfg.sources.n_subcortical_positions = 50
    cfg.head.n_legendre = 40
    cfg.electrode_file = 'standard_1010_69.csv'
    cfg.output_dir = output_dir
    cfg.master_seed = 862
    cfg.output.chunk_samples = 50  # exercise chunking with several chunks per split
    cfg.output.diagnostic_plots = False
    return cfg


def test_pipeline_point_mode():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _tiny_config(tmp, 'point')
        summary = run_pipeline(cfg)
        assert summary['output_dir'] == str(Path(tmp).resolve())
        assert '.staging-' not in json.dumps(summary)
        runtime = json.loads(Path(summary['runtime_provenance']).read_text())
        assert runtime['n_workers_requested'] == 1
        assert runtime['n_workers_used'] == 1
        # A source checkout has no installed distribution metadata, so the
        # recorded version carries an explicit '+source' suffix there.
        recorded = runtime['software_versions']['spherical_eeg_sim']
        assert recorded.split('+')[0] == __version__
        assert recorded in (__version__, f'{__version__}+source')
        assert (Path(tmp) / 'train.h5').exists()
        assert (Path(tmp) / 'val.h5').exists()
        assert (Path(tmp) / 'test.h5').exists()
        assert (Path(tmp) / 'metadata.json').exists()
        assert (Path(tmp) / 'validation_report.json').exists()
        assert (Path(tmp) / 'source_pool.h5').exists()
        assert (Path(tmp) / 'checksums.sha256').exists()
        assert not (Path(tmp) / '.spherical_eeg_sim.transaction.json').exists()
        assert 'Magnitude convention: rms' in (Path(tmp) / 'README.txt').read_text()
        manifest = json.loads((Path(tmp) / 'metadata.json').read_text())
        actual_files = sorted(
            str(path.relative_to(tmp).as_posix())
            for path in Path(tmp).rglob('*') if path.is_file()
        )
        assert manifest['extras']['output_files'] == actual_files
        checksum_lines = (Path(tmp) / 'checksums.sha256').read_text().splitlines()
        checksums = dict(line.split('  ', 1) for line in checksum_lines)
        expected_hashed_files = [
            item for item in actual_files if item != 'checksums.sha256'
        ]
        assert sorted(checksums.values()) == expected_hashed_files
        for digest, relative in checksums.items():
            assert hashlib.sha256((Path(tmp) / relative).read_bytes()).hexdigest() == digest
        report = json.loads((Path(tmp) / 'validation_report.json').read_text())
        checks = {c['name']: c for c in report['distributions']['checks']}
        assert checks['serialized_split_position_audit']['passed']
        with h5py.File(Path(tmp) / 'train.h5', 'r') as f:
            assert f['metadata'].attrs['magnitude_convention'] == 'rms'
            assert f['eeg'].shape == (100, 69, 100)
            assert f['ground_truth/positions_xyz'].shape == (100, 3)
            assert f['ground_truth/orientations_world'].shape == (100, 3)
            assert f['ground_truth/magnitudes_nAm'].shape == (100,)
            assert f['ground_truth/snr_db'].shape == (100,)
            assert f['ground_truth/position_pool_idx'].shape == (100,)
            assert np.all(np.isfinite(f['eeg'][:]))
            assert np.all(np.isfinite(f['ground_truth/magnitudes_nAm'][:]))


def test_pipeline_multi_point_mode():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _tiny_config(tmp, 'multi_point')
        cfg.multi_point.n_sources_min = 2
        cfg.multi_point.n_sources_max = 3
        run_pipeline(cfg)
        with h5py.File(Path(tmp) / 'train.h5', 'r') as f:
            assert f['ground_truth/positions_xyz'].shape == (100, 3, 3)
            assert 'n_dipoles_per_sample' in f['ground_truth']
            assert 'source_time_courses' in f['ground_truth']
            assert 'source_base_band_idx' in f['ground_truth']
            assert 'source_base_frequencies_hz' in f['ground_truth']
            assert 'source_base_phases_rad' in f['ground_truth']
            assert 'source_base_component_amplitudes' in f['ground_truth']
            assert 'source_base_n_components' in f['ground_truth']
            assert 'source_primary_blend_weight' in f['ground_truth']
            assert 'source_independent_blend_weight' in f['ground_truth']
            assert 'source_blend_std' in f['ground_truth']
            assert 'time_coupling' in f['ground_truth']
            assert f['ground_truth/position_pool_idx'].shape == (100, 3)
            assert f['ground_truth/source_base_band_idx'].shape == (100, 3)
            assert f['ground_truth/source_base_frequencies_hz'].shape == (100, 3, 3)
            assert f['ground_truth/source_time_courses'].shape == (100, 3, 100)
            n_dip = f['ground_truth/n_dipoles_per_sample'][:]
            assert n_dip.min() >= 2 and n_dip.max() <= 3
            src_band = f['ground_truth/source_base_band_idx'][:]
            active = np.arange(src_band.shape[1])[None, :] < n_dip[:, None]
            # Default low_freq_band_policy='restrict' drops delta/theta, so
            # every active source in multi-point mode should use alpha or above.
            assert (src_band[active] >= 2).all()
            assert (src_band[~active] == -1).all()


def test_pipeline_patch_mode():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _tiny_config(tmp, 'patch')
        cfg.output.save_leadfield = True
        cfg.output.save_components = True
        run_pipeline(cfg)
        with h5py.File(Path(tmp) / 'train.h5', 'r') as f:
            assert f['ground_truth/positions_xyz'].shape == (100, 3)
            assert 'patch_diameter_mm' in f['ground_truth']
            assert 'position_pool_idx' in f['ground_truth']
            assert 'patch_active_position_idx' in f['ground_truth']
            assert 'patch_active_moments_nAm' in f['ground_truth']
            assert f['ground_truth/position_pool_idx'].shape == (100,)
            d = f['ground_truth/patch_diameter_mm'][:]
            assert d.min() >= 10.0 and d.max() <= 60.0
            n_active = f['ground_truth/n_active_per_patch'][:]
            center_idx = f['ground_truth/position_pool_idx'][:]
            idx = f['ground_truth/patch_active_position_idx']
            moments = f['ground_truth/patch_active_moments_nAm']
            idx_values = idx['values'][:]
            idx_splits = idx['row_splits'][:]
            moment_values = moments['values'][:]
            moment_splits = moments['row_splits'][:]
            assert idx.attrs['encoding'] == 'csr_rows'
            assert np.array_equal(idx_splits, moment_splits)
            assert np.array_equal(np.diff(idx_splits), n_active)
            assert np.all(np.isfinite(moment_values))
            magnitudes = f['ground_truth/magnitudes_nAm'][:]
            courses = f['ground_truth/source_time_course'][:]
            stored_signal = f['ground_truth/eeg_signal_only'][:]
            with h5py.File(Path(tmp) / 'leadfield.h5', 'r') as leadfield_file:
                leadfield = leadfield_file['leadfield'][:]
            for sample in range(n_active.size):
                start, stop = idx_splits[sample:sample + 2]
                assert center_idx[sample] in idx_values[start:stop]
                assert np.isclose(
                    np.abs(moment_values[start:stop]).sum(),
                    magnitudes[sample],
                )
                if sample < 5:
                    topography = np.sum(
                        leadfield[idx_values[start:stop], :, 0]
                        * (moment_values[start:stop] * 1e-9)[:, None],
                        axis=0,
                    )
                    rebuilt = topography[:, None] * courses[sample][None, :]
                    assert np.allclose(
                        rebuilt, stored_signal[sample], rtol=0.0, atol=1e-18
                    )
        assert (Path(tmp) / 'source_pool.h5').is_file()

        used = {}
        for split in ('train', 'val', 'test'):
            with h5py.File(Path(tmp) / f'{split}.h5', 'r') as split_file:
                used[split] = set(
                    split_file[
                        'ground_truth/patch_active_position_idx/values'
                    ][:].tolist()
                )
        assert used['train'].isdisjoint(used['val'])
        assert used['train'].isdisjoint(used['test'])
        assert used['val'].isdisjoint(used['test'])


def test_pipeline_reproducibility():
    """Two runs with the same seed must produce identical EEG."""
    with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
        cfg1 = _tiny_config(tmp1, 'point')
        cfg2 = _tiny_config(tmp2, 'point')
        run_pipeline(cfg1)
        run_pipeline(cfg2)
        with h5py.File(Path(tmp1) / 'train.h5', 'r') as f1, \
             h5py.File(Path(tmp2) / 'train.h5', 'r') as f2:
            assert np.array_equal(f1['eeg'][:], f2['eeg'][:])
            assert np.array_equal(
                f1['ground_truth/positions_xyz'][:],
                f2['ground_truth/positions_xyz'][:],
            )


def test_pipeline_chunk_invariance():
    """The streaming chunk size must not change the produced data.

    Per-sample seeding makes every sample a pure function of
    (master_seed, split_id, global_index), so two runs that differ ONLY in
    cfg.output.chunk_samples must yield byte-identical EEG and ground truth.
    """
    for mode in ('point', 'multi_point', 'patch'):
        with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
            cfg1 = _tiny_config(tmp1, mode)
            cfg2 = _tiny_config(tmp2, mode)
            cfg1.output.chunk_samples = 50   # several chunks per split
            cfg2.output.chunk_samples = 17   # different chunk boundaries
            run_pipeline(cfg1)
            run_pipeline(cfg2)
            for split in ('train', 'val', 'test'):
                with h5py.File(Path(tmp1) / f'{split}.h5', 'r') as f1, \
                     h5py.File(Path(tmp2) / f'{split}.h5', 'r') as f2:
                    assert np.array_equal(f1['eeg'][:], f2['eeg'][:]), \
                        f"{mode}/{split}: EEG differs across chunk sizes"
                    _assert_h5_node_equal(
                        f1['ground_truth'], f2['ground_truth']
                    )


def _micro_config(output_dir: str, source_mode: str = 'point') -> Config:
    cfg = Config()
    cfg.source_mode = source_mode
    cfg.dataset.n_train = 24
    cfg.dataset.n_val = 8
    cfg.dataset.n_test = 8
    cfg.sources.n_cortical_positions = 80
    cfg.sources.n_subcortical_positions = 20
    cfg.head.n_legendre = 40
    cfg.electrode_file = 'standard_1010_32.csv'
    cfg.output_dir = output_dir
    cfg.master_seed = 123
    cfg.output.chunk_samples = 7
    cfg.output.hdf5_chunk_rows = 4
    cfg.output.compress = False
    cfg.output.diagnostic_plots = False
    return cfg


def test_pipeline_n_workers_invariance():
    """Parallel generation must be byte-identical to serial generation."""
    with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
        cfg1 = _micro_config(tmp1, 'point')
        cfg2 = _micro_config(tmp2, 'point')
        cfg1.output.n_workers = 1
        cfg2.output.n_workers = 2
        run_pipeline(cfg1)
        run_pipeline(cfg2)
        with h5py.File(Path(tmp1) / 'train.h5', 'r') as f1, \
             h5py.File(Path(tmp2) / 'train.h5', 'r') as f2:
            assert np.array_equal(f1['eeg'][:], f2['eeg'][:])
            _assert_h5_node_equal(f1['ground_truth'], f2['ground_truth'])


def test_pipeline_npz_and_mat_output():
    pytest.importorskip(
        "hdf5storage",
        reason="MATLAB v7.3 output requires the optional hdf5storage backend",
    )
    """NPZ and genuine MATLAB v7.3 outputs contain EEG and GT arrays."""
    with tempfile.TemporaryDirectory() as tmp_npz:
        cfg = _micro_config(tmp_npz, 'point')
        cfg.output.format = 'npz'
        run_pipeline(cfg)
        with np.load(Path(tmp_npz) / 'train.npz', allow_pickle=True) as z:
            assert z['eeg'].shape == (24, 32, 100)
            assert z['gt_positions_xyz'].shape == (24, 3)
            assert 'electrode_names' in z

    with tempfile.TemporaryDirectory() as tmp_mat:
        from spherical_eeg_sim._compat import import_hdf5storage
        hdf5storage = import_hdf5storage()
        cfg = _micro_config(tmp_mat, 'point')
        cfg.output.format = 'mat'
        run_pipeline(cfg)
        d = hdf5storage.loadmat(Path(tmp_mat) / 'train.mat')
        assert d['eeg'].shape == (24, 32, 100)
        gt = d['ground_truth'][0]
        electrodes = d['electrodes'][0]
        assert gt['positions_xyz'].shape == (24, 3)
        assert 'names' in electrodes.dtype.names


def test_patch_csr_round_trip_npz_and_mat():
    pytest.importorskip(
        "hdf5storage",
        reason="MATLAB v7.3 output requires the optional hdf5storage backend",
    )
    """Padding-free patch support must survive both non-HDF5 backends."""
    with tempfile.TemporaryDirectory() as tmp_npz:
        cfg = _micro_config(tmp_npz, 'patch')
        cfg.output.format = 'npz'
        run_pipeline(cfg)
        with np.load(Path(tmp_npz) / 'train.npz') as data:
            idx_values = data['gt_patch_active_position_idx_values']
            idx_splits = data['gt_patch_active_position_idx_row_splits']
            moment_values = data['gt_patch_active_moments_nAm_values']
            moment_splits = data['gt_patch_active_moments_nAm_row_splits']
            n_active = data['gt_n_active_per_patch']
            magnitude = data['gt_magnitudes_nAm']
            assert np.array_equal(idx_splits, moment_splits)
            assert np.array_equal(np.diff(idx_splits), n_active)
            for sample, (start, stop) in enumerate(zip(idx_splits[:-1], idx_splits[1:])):
                assert np.isclose(np.abs(moment_values[start:stop]).sum(), magnitude[sample])
            assert idx_values.shape == moment_values.shape

    with tempfile.TemporaryDirectory() as tmp_mat:
        from spherical_eeg_sim._compat import import_hdf5storage
        hdf5storage = import_hdf5storage()
        cfg = _micro_config(tmp_mat, 'patch')
        cfg.output.format = 'mat'
        run_pipeline(cfg)
        data = hdf5storage.loadmat(Path(tmp_mat) / 'train.mat')
        gt = data['ground_truth'][0]
        idx = gt['patch_active_position_idx'][0]
        moments = gt['patch_active_moments_nAm'][0]
        idx_splits = idx['row_splits'].reshape(-1)
        moment_splits = moments['row_splits'].reshape(-1)
        moment_values = moments['values'].reshape(-1)
        n_active = gt['n_active_per_patch'].reshape(-1)
        magnitude = gt['magnitudes_nAm'].reshape(-1)
        assert idx['encoding'].reshape(-1)[0] == 'csr_rows'
        assert np.array_equal(idx_splits, moment_splits)
        assert np.array_equal(np.diff(idx_splits), n_active)
        for sample, (start, stop) in enumerate(zip(idx_splits[:-1], idx_splits[1:])):
            assert np.isclose(np.abs(moment_values[start:stop]).sum(), magnitude[sample])


def test_pipeline_custom_electrode_file():
    """A user-supplied electrode CSV path should work, not only bundled names."""
    import shutil
    import spherical_eeg_sim.montages as montages
    from importlib.resources import files

    with tempfile.TemporaryDirectory() as tmp:
        custom = Path(tmp) / 'custom_20.csv'
        shutil.copyfile(files(montages).joinpath('standard_1020_20.csv'), custom)
        cfg = _micro_config(str(Path(tmp) / 'out'), 'point')
        cfg.electrode_file = str(custom)
        run_pipeline(cfg)
        with h5py.File(Path(cfg.output_dir) / 'train.h5', 'r') as f:
            assert f['eeg'].shape == (24, 20, 100)
            assert f['electrodes/cartesian_xyz'].shape == (20, 3)


def test_writer_rejects_ground_truth_sample_axis_mismatch(tmp_path):
    """A split cannot serialize labels with a different sample count."""
    from spherical_eeg_sim.io.writer import open_split_writer

    writer = open_split_writer(tmp_path, 'bad', fmt='hdf5')
    with writer:
        with pytest.raises(ValueError, match='EEG has 2 samples'):
            writer.append(np.zeros((2, 3, 4)), {'label': np.zeros(1)})


def test_pipeline_save_components_and_snr_output():
    """Saved signal/noise components should reconstruct EEG and match SNR."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _micro_config(tmp, 'point')
        cfg.output.save_components = True
        cfg.noise.noise_model = 'snr_target'
        cfg.noise.snr_db_range = [10.0, 10.001]
        run_pipeline(cfg)
        with h5py.File(Path(tmp) / 'train.h5', 'r') as f:
            eeg = f['eeg'][:]
            sig = f['ground_truth/eeg_signal_only'][:]
            noise = f['ground_truth/eeg_noise_only'][:]
            target = f['ground_truth/snr_db'][:]
            assert np.allclose(sig + noise, eeg, atol=1e-12)
            measured = 10 * np.log10((sig ** 2).mean(axis=(1, 2)) / (noise ** 2).mean(axis=(1, 2)))
            assert np.max(np.abs(measured - target)) < 1e-5


def test_pipeline_float32_output_dtype():
    """cfg.output.dtype='float32' should write /eeg and saved components as float32."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _micro_config(tmp, 'point')
        cfg.output.dtype = 'float32'
        cfg.output.save_components = True
        run_pipeline(cfg)
        with h5py.File(Path(tmp) / 'train.h5', 'r') as f:
            assert f['eeg'].dtype == np.dtype('float32')
            assert f['ground_truth/eeg_signal_only'].dtype == np.dtype('float32')
            assert f['ground_truth/eeg_noise_only'].dtype == np.dtype('float32')
            assert np.array_equal(
                f['ground_truth/eeg_signal_only'][:]
                + f['ground_truth/eeg_noise_only'][:],
                f['eeg'][:],
            )
            # Core labels remain high precision / semantic dtypes.
            assert f['ground_truth/positions_xyz'].dtype == np.dtype('float64')
            assert f['ground_truth/population'].dtype == np.dtype('int8')


def test_pipeline_absolute_floor_realized_rms_and_snr():
    """absolute_floor: fixed noise RMS is realized and the emergent SNR is
    self-consistent with the written signal/noise components."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _micro_config(tmp, 'point')
        cfg.output.save_components = True
        cfg.noise.noise_model = 'absolute_floor'
        cfg.noise.absolute_noise_rms_uV = 3.16
        # reference_mode defaults to 'none' -> the written montage IS the raw
        # montage on which the RMS is fixed, so realized RMS is exact.
        run_pipeline(cfg)
        with h5py.File(Path(tmp) / 'train.h5', 'r') as f:
            eeg = f['eeg'][:]
            sig = f['ground_truth/eeg_signal_only'][:]
            noise = f['ground_truth/eeg_noise_only'][:]
            snr_db = f['ground_truth/snr_db'][:]
            md = dict(f['metadata'].attrs)
        # Components reconstruct the EEG exactly.
        assert np.allclose(sig + noise, eeg, atol=1e-12)
        # Realized combined-noise RMS equals absolute_noise_rms_uV (in volts).
        realized_rms = np.sqrt((noise ** 2).mean(axis=(1, 2)))
        assert np.allclose(realized_rms, 3.16e-6, rtol=1e-9, atol=0.0)
        # Emergent SNR written to ground truth matches the SNR of the components.
        measured = 10 * np.log10((sig ** 2).mean(axis=(1, 2)) / (noise ** 2).mean(axis=(1, 2)))
        assert np.max(np.abs(measured - snr_db)) < 1e-9
        # Dataset is self-describing via metadata.
        assert md['noise_model'] == 'absolute_floor'
        assert abs(float(md['absolute_noise_rms_uV']) - 3.16) < 1e-12


def test_pipeline_absolute_floor_magnitude_controls_snr():
    """absolute_floor: with a fixed noise floor, scaling every dipole magnitude
    by 10x must raise the emergent SNR by exactly 20 dB per sample (signal power
    scales as magnitude^2 while the noise is unchanged)."""
    def cfg_for(tmp, mag_scale):
        cfg = _micro_config(tmp, 'point')
        cfg.noise.noise_model = 'absolute_floor'
        cm = [v * mag_scale for v in cfg.dipole.cortical_magnitude_range_nAm]
        sm = [v * mag_scale for v in cfg.dipole.subcortical_magnitude_range_nAm]
        cfg.dipole.cortical_magnitude_range_nAm = cm
        cfg.dipole.subcortical_magnitude_range_nAm = sm
        return cfg

    with tempfile.TemporaryDirectory() as t1, tempfile.TemporaryDirectory() as t10:
        run_pipeline(cfg_for(t1, 1.0))
        run_pipeline(cfg_for(t10, 10.0))
        with h5py.File(Path(t1) / 'train.h5', 'r') as f1, \
             h5py.File(Path(t10) / 'train.h5', 'r') as f10:
            snr1 = f1['ground_truth/snr_db'][:]
            snr10 = f10['ground_truth/snr_db'][:]
            mag1 = f1['ground_truth/magnitudes_nAm'][:]
            mag10 = f10['ground_truth/magnitudes_nAm'][:]
        # Same seed + same RNG consumption => magnitudes scaled exactly 10x.
        assert np.allclose(mag10 / mag1, 10.0, rtol=1e-9)
        # Emergent SNR rises by exactly 20 dB (= 10*log10(10^2)).
        assert np.max(np.abs((snr10 - snr1) - 20.0)) < 1e-6


def test_pipeline_absolute_floor_deeper_lower_snr():
    """absolute_floor: at fixed dipole magnitude, deeper sources yield lower
    emergent SNR than superficial ones (Goldenholz-type depth dependence)."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Config()
        cfg.source_mode = 'point'
        cfg.dataset.n_train = 300
        cfg.dataset.n_val = 40
        cfg.dataset.n_test = 40
        cfg.sources.n_cortical_positions = 400
        cfg.sources.n_subcortical_positions = 150
        cfg.head.n_legendre = 40
        cfg.electrode_file = 'standard_1010_69.csv'
        cfg.output_dir = tmp
        cfg.master_seed = 7
        cfg.output.diagnostic_plots = False
        cfg.noise.noise_model = 'absolute_floor'
        # Hold magnitude fixed so only depth (lead-field gain) drives the SNR.
        cfg.dipole.cortical_magnitude_range_nAm = [100.0, 100.0001]
        cfg.dipole.subcortical_magnitude_range_nAm = [100.0, 100.0001]
        run_pipeline(cfg)
        with h5py.File(Path(tmp) / 'train.h5', 'r') as f:
            pos = f['ground_truth/positions_xyz'][:]
            snr = f['ground_truth/snr_db'][:]
        radius = np.linalg.norm(pos, axis=1)
        order = np.argsort(radius)
        k = len(radius) // 3
        deep_median = np.median(snr[order[:k]])        # smallest radius = deepest
        shallow_median = np.median(snr[order[-k:]])    # largest radius = shallowest
        assert shallow_median > deep_median
        assert np.corrcoef(radius, snr)[0, 1] > 0.0


def test_pipeline_snr_target_chunk_invariance():
    """snr_target must also be byte-identical across chunk sizes.

    (The default-noise-model invariance tests above exercise absolute_floor;
    this locks the snr_target path too.)"""
    with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
        cfg1 = _micro_config(tmp1, 'point')
        cfg2 = _micro_config(tmp2, 'point')
        cfg1.noise.noise_model = 'snr_target'
        cfg2.noise.noise_model = 'snr_target'
        cfg1.output.chunk_samples = 5
        cfg2.output.chunk_samples = 13
        run_pipeline(cfg1)
        run_pipeline(cfg2)
        for split in ('train', 'val', 'test'):
            with h5py.File(Path(tmp1) / f'{split}.h5', 'r') as f1, \
                 h5py.File(Path(tmp2) / f'{split}.h5', 'r') as f2:
                assert np.array_equal(f1['eeg'][:], f2['eeg'][:]), \
                    f"snr_target/{split}: EEG differs across chunk sizes"
                _assert_h5_node_equal(
                    f1['ground_truth'], f2['ground_truth']
                )


def test_target_snr_source_truth_aligns_across_montages():
    """Sensor count must not perturb source-side truth or target-SNR draws."""
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        cfg_a = _micro_config(a, 'point')
        cfg_b = _micro_config(b, 'point')
        cfg_a.electrode_file = 'standard_1010_32.csv'
        cfg_b.electrode_file = 'standard_1010_69.csv'
        cfg_a.noise.noise_model = cfg_b.noise.noise_model = 'snr_target'
        run_pipeline(cfg_a)
        run_pipeline(cfg_b)
        keys = (
            'positions_xyz', 'orientations_world', 'magnitudes_nAm',
            'band_idx', 'frequencies_hz', 'phases_rad',
            'component_amplitudes', 'source_time_course', 'snr_db',
        )
        with h5py.File(Path(a) / 'train.h5', 'r') as left, \
             h5py.File(Path(b) / 'train.h5', 'r') as right:
            for key in keys:
                assert np.array_equal(
                    left[f'ground_truth/{key}'][:],
                    right[f'ground_truth/{key}'][:],
                    equal_nan=True,
                ), key


def test_point_time_course_parameters_reconstruct_stored_course():
    """Stored sinusoid parameters must reconstruct every point-source course."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _micro_config(tmp, 'point')
        run_pipeline(cfg)
        with h5py.File(Path(tmp) / 'train.h5', 'r') as data:
            freq = data['ground_truth/frequencies_hz'][:]
            phase = data['ground_truth/phases_rad'][:]
            amplitude = data['ground_truth/component_amplitudes'][:]
            n_components = data['ground_truth/n_components'][:]
            actual = data['ground_truth/source_time_course'][:]
        time = np.arange(cfg.signal.window_samples) / cfg.signal.sampling_freq_hz
        for sample in range(actual.shape[0]):
            count = int(n_components[sample])
            rebuilt = (
                amplitude[sample, :count, None]
                * np.sin(
                    2 * np.pi * freq[sample, :count, None] * time
                    + phase[sample, :count, None]
                )
            ).sum(axis=0)
            if cfg.signal.zero_mean_signal:
                rebuilt -= rebuilt.mean()
            rebuilt /= rebuilt.std()
            assert np.allclose(rebuilt, actual[sample], rtol=0.0, atol=1e-14)


def test_codebook_pipeline_spans_tangent_azimuths():
    """Pipeline draws span, but never leave, the finite tangent codebook."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _tiny_config(tmp, 'point')
        cfg.dataset.n_train = 500
        cfg.dipole.orientation_mode = 'codebook'
        run_pipeline(cfg)
        with h5py.File(Path(tmp) / 'train.h5', 'r') as data:
            positions = data['ground_truth/positions_xyz'][:]
            orientations = data['ground_truth/orientations_world'][:]
        from spherical_eeg_sim.sources.sampling import (
            world_orientation_to_spherical_frame,
        )
        local = world_orientation_to_spherical_frame(orientations, positions)
        tangential = np.abs(local[:, 0]) < 0.5
        azimuth = np.arctan2(local[tangential, 2], local[tangential, 1])
        assert tangential.sum() > 200
        unique_azimuth = np.unique(np.round(np.mod(azimuth, 2 * np.pi), 12))
        golden = np.pi * (1.0 + np.sqrt(5.0))
        expected = set(np.round(
            (np.arange(cfg.dipole.codebook_tangential_directions) * golden)
            % (2 * np.pi),
            12,
        ))
        assert set(unique_azimuth) <= expected
        assert unique_azimuth.size > 100


def _magnitude_scaled_target_snr_config(path, factor, normalization='none'):
    cfg = _micro_config(path, 'point')
    cfg.noise.noise_model = 'snr_target'
    cfg.output.eeg_normalization = normalization
    cfg.dipole.cortical_magnitude_range_nAm = [5 * factor, 300 * factor]
    cfg.dipole.subcortical_magnitude_range_nAm = [30 * factor, 500 * factor]
    return cfg


def test_snr_target_preserves_physical_magnitude_scale():
    """Target SNR scales noise only and keeps physical signal voltage."""
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        run_pipeline(_magnitude_scaled_target_snr_config(a, 1.0))
        run_pipeline(_magnitude_scaled_target_snr_config(b, 10.0))
        with h5py.File(Path(a) / 'train.h5', 'r') as f1, \
             h5py.File(Path(b) / 'train.h5', 'r') as f10:
            eeg1 = f1['eeg'][:]
            eeg10 = f10['eeg'][:]
            scale1 = f1['ground_truth/eeg_normalization_scale_V'][:]
            scale10 = f10['ground_truth/eeg_normalization_scale_V'][:]
            assert f1['metadata'].attrs['eeg_units'] == 'volts'
        assert np.allclose(eeg10 / eeg1, 10.0, rtol=2e-11, atol=2e-11)
        assert np.array_equal(scale1, np.ones_like(scale1))
        assert np.array_equal(scale10, np.ones_like(scale10))


def test_explicit_rms_normalization_removes_magnitude_scale():
    """Explicit RMS normalization is reversible and independent of SNR mode."""
    def make(path, factor):
        return _magnitude_scaled_target_snr_config(
            path, factor, normalization='per_sample_rms'
        )

    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        run_pipeline(make(a, 1.0))
        run_pipeline(make(b, 10.0))
        with h5py.File(Path(a) / 'train.h5', 'r') as f1, \
             h5py.File(Path(b) / 'train.h5', 'r') as f10:
            eeg1 = f1['eeg'][:]
            eeg10 = f10['eeg'][:]
            scale1 = f1['ground_truth/eeg_normalization_scale_V'][:]
            scale10 = f10['ground_truth/eeg_normalization_scale_V'][:]
            assert f1['metadata'].attrs['eeg_units'] == 'dimensionless_rms_normalized'
        assert np.allclose(eeg1, eeg10, rtol=2e-12, atol=2e-12)
        assert np.allclose(scale10 / scale1, 10.0, rtol=2e-12, atol=0.0)
        assert np.allclose(
            np.sqrt(np.mean(eeg1 ** 2, axis=(1, 2))), 1.0,
            rtol=2e-12, atol=2e-12,
        )


def test_patch_support_mesh_density_is_split_independent():
    """Patch-mode position meshes are equal-sized across the three splits."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _tiny_config(tmp, 'patch')
        run_pipeline(cfg)
        with h5py.File(Path(tmp) / 'source_pool.h5', 'r') as f:
            labels = f['split'][:]
            pop = f['population'][:]
        cortical_counts = [int(np.sum((labels == i) & (pop == 0))) for i in range(3)]
        assert max(cortical_counts) - min(cortical_counts) <= 2
        with open(Path(tmp) / 'metadata.json') as f:
            metadata = __import__('json').load(f)
        assert metadata['extras']['source_pool']['allocation_basis'] == \
            'equal-density patch support meshes'


def test_multi_point_temporal_ground_truth_reconstructs_blends():
    """Stored bases, amplitudes, weights, and scale reproduce every course."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _micro_config(tmp, 'multi_point')
        cfg.multi_point.n_sources_min = 3
        cfg.multi_point.n_sources_max = 3
        cfg.multi_point.time_coupling = 0.4
        run_pipeline(cfg)
        with h5py.File(Path(tmp) / 'train.h5', 'r') as f:
            g = f['ground_truth']
            actual = g['source_time_courses'][:]
            freqs = g['source_base_frequencies_hz'][:]
            phases = g['source_base_phases_rad'][:]
            amps = g['source_base_component_amplitudes'][:]
            ncomp = g['source_base_n_components'][:]
            wp = g['source_primary_blend_weight'][:]
            wi = g['source_independent_blend_weight'][:]
            blend_sd = g['source_blend_std'][:]
        t = np.arange(cfg.signal.window_samples) / cfg.signal.sampling_freq_hz
        for i in range(actual.shape[0]):
            for k in range(actual.shape[1]):
                n = int(ncomp[i, k])
                raw = np.sum(
                    amps[i, k, :n, None] * np.sin(
                        2 * np.pi * freqs[i, k, :n, None] * t
                        + phases[i, k, :n, None]
                    ), axis=0,
                )
                if cfg.signal.zero_mean_signal:
                    raw -= raw.mean()
                raw /= raw.std()
                expected = raw if k == 0 else (
                    wp[i, k] * actual[i, 0] + wi[i, k] * raw
                ) / blend_sd[i, k]
                assert np.allclose(expected, actual[i, k], rtol=2e-12, atol=2e-12)


def test_noncentered_source_courses_declare_waveform_scale_magnitudes():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _micro_config(tmp, 'point')
        cfg.signal.zero_mean_signal = False
        run_pipeline(cfg)
        with h5py.File(Path(tmp) / 'train.h5', 'r') as f:
            course = f['ground_truth/source_time_course'][:]
            assert f['metadata'].attrs['magnitude_convention'] == 'waveform_scale'
        assert 'Magnitude convention: waveform_scale' in (
            Path(tmp) / 'README.txt'
        ).read_text()
        assert np.allclose(course.std(axis=1), 1.0, rtol=1e-12, atol=1e-12)
        assert np.any(np.abs(np.sqrt(np.mean(course ** 2, axis=1)) - 1.0) > 1e-6)


def test_multi_point_coupling_design_varies_per_sample():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _micro_config(tmp, 'multi_point')
        cfg.multi_point.n_sources_min = 2
        cfg.multi_point.n_sources_max = 2
        cfg.multi_point.time_coupling_values = [0.0, 0.5, 1.0]
        run_pipeline(cfg)
        with h5py.File(Path(tmp) / 'train.h5', 'r') as f:
            realized = f['ground_truth/time_coupling'][:]
            primary = f['ground_truth/source_primary_blend_weight'][:, 1]
        assert set(np.unique(realized)).issubset({0.0, 0.5, 1.0})
        assert np.unique(realized).size > 1
        assert np.array_equal(realized, primary)


def test_position_and_sample_seed_domains_are_independent():
    with tempfile.TemporaryDirectory() as root:
        a, b = Path(root) / 'a', Path(root) / 'b'
        cfg_a = _micro_config(str(a), 'point')
        cfg_b = _micro_config(str(b), 'point')
        for cfg, sample_seed in ((cfg_a, 901), (cfg_b, 902)):
            cfg.position_seed = 777
            cfg.sample_seed = sample_seed
            cfg.dataset.n_train = 8
            cfg.dataset.n_val = 3
            cfg.dataset.n_test = 3
        run_pipeline(cfg_a)
        run_pipeline(cfg_b)
        with h5py.File(a / 'source_pool.h5', 'r') as left, \
             h5py.File(b / 'source_pool.h5', 'r') as right:
            assert np.array_equal(left['positions_xyz'][:], right['positions_xyz'][:])
            assert np.array_equal(left['split'][:], right['split'][:])
        with h5py.File(a / 'train.h5', 'r') as left, \
             h5py.File(b / 'train.h5', 'r') as right:
            assert not np.array_equal(
                left['ground_truth/position_pool_idx'][:],
                right['ground_truth/position_pool_idx'][:],
            )


def _directory_sha256(path):
    import hashlib
    return {
        str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path(path).rglob('*')) if p.is_file()
    }


def test_complete_directory_hash_invariance_all_formats():
    pytest.importorskip(
        "hdf5storage",
        reason="MATLAB v7.3 output requires the optional hdf5storage backend",
    )
    """Every mode/backend hashes identically across chunk and worker controls."""
    for mode in ('point', 'multi_point', 'patch'):
        for fmt in ('hdf5', 'npz', 'mat'):
            with tempfile.TemporaryDirectory() as root:
                a = Path(root) / 'a'
                b = Path(root) / 'b'
                c1 = _micro_config(str(a), mode)
                c2 = _micro_config(str(b), mode)
                for c in (c1, c2):
                    c.dataset.n_train = 8
                    c.dataset.n_val = 3
                    c.dataset.n_test = 3
                    c.output.format = fmt
                    c.output.save_leadfield = True
                    c.output.save_graph = True
                    c.output.diagnostic_plots = True
                c1.output.chunk_samples = 3
                c1.output.n_workers = 1
                c2.output.chunk_samples = 5
                c2.output.n_workers = 2
                run_pipeline(c1)
                run_pipeline(c2)
                assert _directory_sha256(a) == _directory_sha256(b), (mode, fmt)


def test_failed_distribution_validation_is_not_published(tmp_path, monkeypatch):
    """A failed Layer-2 check must abort staged directory publication."""
    from spherical_eeg_sim.validation.distribution_check import (
        CheckResult, DistributionValidationResult,
    )
    import spherical_eeg_sim.pipeline as pipeline

    def fail_validation(**kwargs):
        return DistributionValidationResult([
            CheckResult('injected_failure', False, 'test failure'),
        ])

    monkeypatch.setattr(pipeline, 'validate_distributions', fail_validation)
    cfg = _micro_config(str(tmp_path / 'dataset'), 'point')
    cfg.dataset.n_train = 4
    cfg.dataset.n_val = 2
    cfg.dataset.n_test = 2
    with pytest.raises(RuntimeError, match='injected_failure'):
        run_pipeline(cfg)
    assert not (tmp_path / 'dataset').exists()


def test_serialized_split_audit_detects_writer_side_leakage():
    """The audit must fail when written indices disagree with assignments."""
    from spherical_eeg_sim.validation.distribution_check import (
        validate_distributions,
    )
    result = validate_distributions(
        positions_xyz=np.array([
            [0.075, 0.0, 0.0],
            [0.0, 0.075, 0.0],
            [0.0, 0.0, 0.075],
        ]),
        populations=np.zeros(3, dtype=np.int8),
        magnitudes_nAm=np.full(3, 10.0),
        snr_db=np.full(3, 5.0),
        cortical_radius_range_mm=(70.5, 79.5),
        subcortical_radius_range_mm=(30.0, 65.0),
        cortical_magnitude_range_nAm=(5.0, 300.0),
        subcortical_magnitude_range_nAm=(30.0, 500.0),
        snr_db_range=(0.0, 20.0),
        expected_subcortical_fraction=0.0,
        orientation_mode='uniform_s2',
        assigned_position_indices={
            'train': np.array([0]), 'val': np.array([1]), 'test': np.array([2]),
        },
        serialized_position_indices={
            'train': np.array([0]), 'val': np.array([0]), 'test': np.array([2]),
        },
    )
    check = next(
        item for item in result.checks
        if item.name == 'serialized_split_position_audit'
    )
    assert not check.passed
    assert check.statistics['overlap_train_val'] == 1
    assert check.statistics['outside_assigned_pool']['val'] == 1


def test_canonical_manifest_config_round_trip_reproduces_directory():
    """Loading metadata.json's config must preserve every scientific label."""
    with tempfile.TemporaryDirectory() as root:
        a = Path(root) / 'a'
        b = Path(root) / 'b'
        original = _micro_config(str(a), 'point')
        original.dataset.n_train = 8
        original.dataset.n_val = 3
        original.dataset.n_test = 3
        run_pipeline(original)

        manifest = json.loads((a / 'metadata.json').read_text())
        replay = Config.from_dict(manifest['config'])
        replay.output_dir = str(b)
        replay.output.chunk_samples = 5
        replay.output.n_workers = 2
        run_pipeline(replay)

        assert replay.signal.band_order == [
            'delta', 'theta', 'alpha', 'beta', 'gamma',
        ]
        assert _directory_sha256(a) == _directory_sha256(b)


def test_nonempty_output_requires_explicit_overwrite():
    with tempfile.TemporaryDirectory() as root:
        out = Path(root) / 'dataset'
        cfg = _micro_config(str(out), 'point')
        cfg.dataset.n_train = 4
        cfg.dataset.n_val = 2
        cfg.dataset.n_test = 2
        run_pipeline(cfg)
        stale = out / 'stale.txt'
        stale.write_text('must not survive replacement')
        with pytest.raises(FileExistsError):
            run_pipeline(cfg)
        cfg.output.overwrite = True
        with pytest.raises(FileExistsError, match='unowned file'):
            run_pipeline(cfg)
        assert stale.read_text() == 'must not survive replacement'
        stale.unlink()
        run_pipeline(cfg)
        assert (out / 'checksums.sha256').is_file()


def test_interrupted_publish_backup_is_restored_before_any_regeneration():
    with tempfile.TemporaryDirectory() as root:
        out = Path(root) / 'dataset'
        cfg = _micro_config(str(out), 'point')
        cfg.dataset.n_train = 4
        cfg.dataset.n_val = 2
        cfg.dataset.n_test = 2
        run_pipeline(cfg)
        backup = Path(root) / '.dataset.previous-crash'
        _write_rollback_marker(out)
        out.replace(backup)
        with pytest.raises(FileExistsError, match='not empty'):
            run_pipeline(cfg)
        assert out.is_dir()
        assert (out / 'checksums.sha256').is_file()
        assert not backup.exists()


def test_unowned_rollback_like_directory_is_never_deleted():
    with tempfile.TemporaryDirectory() as root:
        out = Path(root) / 'dataset'
        cfg = _micro_config(str(out), 'point')
        cfg.dataset.n_train = 4
        cfg.dataset.n_val = 2
        cfg.dataset.n_test = 2
        run_pipeline(cfg)
        backup = Path(root) / '.dataset.previous-not-ours'
        backup.mkdir()
        sentinel = backup / 'user-file.txt'
        sentinel.write_text('keep me')
        with pytest.raises(RuntimeError, match='Refusing to use or delete unverified'):
            run_pipeline(cfg)
        assert sentinel.read_text() == 'keep me'


def test_checksum_verification_rejects_an_incomplete_inventory():
    with tempfile.TemporaryDirectory() as root:
        out = Path(root) / 'dataset'
        cfg = _micro_config(str(out), 'point')
        cfg.dataset.n_train = 4
        cfg.dataset.n_val = 2
        cfg.dataset.n_test = 2
        run_pipeline(cfg)
        checksum_path = out / 'checksums.sha256'
        lines = checksum_path.read_text(encoding='ascii').splitlines()
        assert _directory_checksum_complete(out)
        checksum_path.write_text('\n'.join(lines[1:]) + '\n', encoding='ascii')
        assert not _directory_checksum_complete(out)


def test_checksum_complete_staging_is_published_and_reused():
    with tempfile.TemporaryDirectory() as root:
        out = Path(root) / 'dataset'
        cfg = _micro_config(str(out), 'point')
        cfg.dataset.n_train = 4
        cfg.dataset.n_val = 2
        cfg.dataset.n_test = 2
        run_pipeline(cfg)
        original_hash = hashlib.sha256((out / 'train.h5').read_bytes()).hexdigest()
        staging = Path(root) / '.dataset.staging-crash'
        out.replace(staging)
        summary = run_pipeline(cfg)
        assert hashlib.sha256((out / 'train.h5').read_bytes()).hexdigest() == original_hash
        runtime = json.loads(Path(summary['runtime_provenance']).read_text())
        assert runtime['recovered_completed_staging'] is True
        assert not staging.exists()


def test_owned_incomplete_staging_is_removed_before_clean_restart():
    with tempfile.TemporaryDirectory() as root:
        out = Path(root) / 'dataset'
        cfg = _micro_config(str(out), 'point')
        staging = Path(root) / '.dataset.staging-crash'
        staging.mkdir()
        (staging / '.spherical_eeg_sim.transaction.json').write_text(
            json.dumps({
                'package': 'spherical_eeg_sim',
                'transaction_kind': 'dataset_staging',
                'target_name': 'dataset',
                'canonical_config': {},
            })
        )
        (staging / 'partial.h5').write_bytes(b'partial')
        run_pipeline(cfg)
        assert out.is_dir()
        assert not staging.exists()


def test_unowned_staging_like_directory_is_never_deleted():
    with tempfile.TemporaryDirectory() as root:
        out = Path(root) / 'dataset'
        cfg = _micro_config(str(out), 'point')
        staging = Path(root) / '.dataset.staging-not-ours'
        staging.mkdir()
        sentinel = staging / 'user-file.txt'
        sentinel.write_text('keep me')
        with pytest.raises(RuntimeError, match='Refusing to delete unverified'):
            run_pipeline(cfg)
        assert sentinel.read_text() == 'keep me'


def test_output_path_must_be_directory():
    with tempfile.TemporaryDirectory() as root:
        out = Path(root) / 'dataset'
        out.write_text('not a directory')
        cfg = _micro_config(str(out), 'point')
        with pytest.raises(NotADirectoryError, match='not a directory'):
            run_pipeline(cfg)


def test_noise_model_config_validation():
    """Config.validate() accepts the two noise models and rejects bad values."""
    import pytest

    cfg = Config()
    assert cfg.noise.noise_model == 'absolute_floor'   # default
    assert cfg.noise.absolute_noise_rms_uV == 1.0
    cfg.validate()                                      # default is valid

    cfg.noise.noise_model = 'snr_target'
    cfg.validate()

    cfg.noise.noise_model = 'not_a_model'
    with pytest.raises(ValueError):
        cfg.validate()

    cfg = Config()
    cfg.noise.absolute_noise_rms_uV = 0.0
    with pytest.raises(ValueError):
        cfg.validate()

    cfg.noise.absolute_noise_rms_uV = -1.0
    with pytest.raises(ValueError):
        cfg.validate()

    cfg = Config()
    cfg.output.eeg_normalization = 'per_sample_rms'
    cfg.validate()
    cfg.output.eeg_normalization = 'implicit_target_side_effect'
    with pytest.raises(ValueError):
        cfg.validate()


@pytest.mark.parametrize('mutation', [
    lambda c: setattr(c.head, 'radii_m', [0.080, 0.081, 0.086]),
    lambda c: setattr(c.head, 'conductivities_S_per_m', [0.33, 1.79, 0.0042, np.nan]),
    lambda c: setattr(c.head, 'n_legendre', 121),
    lambda c: setattr(c.head, 'truncation_tail_tolerance', 0.0),
    lambda c: setattr(c.head, 'truncation_tail_tolerance', np.nan),
    lambda c: setattr(c.signal, 'sampling_freq_hz', np.inf),
    lambda c: setattr(c.signal, 'window_samples', 100.5),
    lambda c: c.signal.bands_hz.__setitem__('gamma', [30.0, 251.0]),
    lambda c: setattr(c.signal, 'band_order', ['delta', 'theta', 'alpha', 'beta']),
    lambda c: setattr(c.signal, 'band_order', ['delta', 'theta', 'alpha', 'beta', 'beta']),
    lambda c: setattr(c.signal, 'min_band_cycles', 1000.0),
    lambda c: setattr(c.noise, 'background_pool_size', 5),
    lambda c: setattr(c.multi_point, 'max_separation_attempts', 0),
    lambda c: setattr(c.signal, 'component_dirichlet_concentration', 0.0),
    lambda c: setattr(c.noise, 'background_amplitude_concentration', np.nan),
    lambda c: setattr(c.noise, 'background_radius_range_mm', [70.5, 80.0]),
    lambda c: setattr(c.signal, 'component_amplitude_jitter', 1),
    lambda c: setattr(c.signal, 'zero_mean_signal', 'yes'),
    lambda c: setattr(c.output, 'graph_distance_threshold_m', np.nan),
    lambda c: setattr(c.output, 'diagnostic_sample_count', 0),
    lambda c: setattr(c.sources, 'cortical_radius_range_mm', [-1.0, 79.5]),
    lambda c: setattr(c.sources, 'n_cortical_positions', 0),
    lambda c: setattr(c.sources, 'n_subcortical_positions', 0),
    lambda c: setattr(c.sources, 'n_cortical_positions', 2),
    lambda c: setattr(c.sources, 'n_subcortical_positions', 2),
    lambda c: setattr(c.dataset, 'n_train', True),
    lambda c: setattr(c.output, 'chunk_samples', 1.5),
    lambda c: setattr(c.output, 'reference_weights', 1.0),
    lambda c: (
        setattr(c, 'source_mode', 'patch'),
        setattr(c.sources, 'n_cortical_positions', 2),
    ),
])
def test_config_rejects_malformed_or_nonfinite_values(mutation):
    cfg = Config()
    mutation(cfg)
    with pytest.raises(ValueError):
        cfg.validate()


def test_head_preset_rejects_nonfinite_ratio_and_noninteger_order():
    from spherical_eeg_sim.config import HeadModelConfig

    with pytest.raises(ValueError):
        HeadModelConfig.with_skull_ratio(np.nan)
    with pytest.raises(ValueError):
        HeadModelConfig.with_skull_ratio(20.0, n_legendre=80.5)


def test_config_rejects_slow_legendre_tail_unless_explicitly_accepted():
    cfg = Config()
    cfg.head.radii_m = [0.0800, 0.0802, 0.0804, 0.0806]
    with pytest.raises(ValueError, match="geometric-tail indicator"):
        cfg.validate()

    cfg.head.truncation_tail_tolerance = 100.0
    cfg.validate()


def test_config_from_dict_rejects_unknown_nested_key():
    with pytest.raises(ValueError, match="signal.*typo_field"):
        Config.from_dict({'signal': {'typo_field': 1}})
    with pytest.raises(ValueError, match='must be an object'):
        Config.from_dict({'signal': []})


if __name__ == '__main__':
    test_pipeline_point_mode()
    print("point OK")
    test_pipeline_multi_point_mode()
    print("multi_point OK")
    test_pipeline_patch_mode()
    print("patch OK")
    test_pipeline_reproducibility()
    print("reproducibility OK")
    test_pipeline_chunk_invariance()
    print("chunk invariance OK")
    print("All pipeline tests passed!")
