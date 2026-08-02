"""
Tests for the auxiliary exports and format matrix:
  - lead-field operator export (save_leadfield)
  - MNE-Python export helpers (skipped if MNE is not installed)
  - signal_topographic_concentration ground-truth field
  - NPZ / MAT output formats and custom electrode CSVs
"""

import sys as _sys
from pathlib import Path as _Path
_repo_root = _Path(__file__).resolve().parent.parent
if str(_repo_root) not in _sys.path:
    _sys.path.insert(0, str(_repo_root))

import numpy as np
import pytest

from spherical_eeg_sim import Config, run_pipeline
from spherical_eeg_sim.core.electrodes import load_electrodes, save_electrodes_csv


def _tiny_cfg(tmp_path, **over):
    cfg = Config()
    cfg.dataset.n_train = 120
    cfg.dataset.n_val = 30
    cfg.dataset.n_test = 30
    cfg.sources.n_cortical_positions = 400
    cfg.sources.n_subcortical_positions = 100
    cfg.head.n_legendre = 40
    cfg.output.diagnostic_plots = False
    cfg.output_dir = str(tmp_path)
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


# ---------------------------------------------------------------------------
# B3: lead-field operator export
# ---------------------------------------------------------------------------

def test_leadfield_export_matches_forward_model(tmp_path):
    import h5py
    from spherical_eeg_sim.core.forward import compute_transfer_coefficients, forward_potential
    from spherical_eeg_sim.sources.sampling import (
        cartesian_to_spherical, world_orientation_to_spherical_frame,
    )
    cfg = _tiny_cfg(tmp_path)
    cfg.output.save_leadfield = True
    cfg.output.reference_mode = 'average'
    run_pipeline(cfg)

    lf = tmp_path / 'leadfield.h5'
    assert lf.is_file()
    with h5py.File(lf, 'r') as f:
        L = f['leadfield'][:]
        pos = f['positions_xyz'][:]
        split = f['split'][:]
        assert L.shape[0] == pos.shape[0]
        assert L.shape[2] == 3
        # split labels cover all three sets
        assert set(np.unique(split).tolist()) == {0, 1, 2}

    # L @ moment must reproduce the direct forward potential at machine level.
    elec = load_electrodes(cfg.electrode_file, target_radius_m=cfg.head.radii_m[-1])
    M = compute_transfer_coefficients(
        cfg.head.radii_m, cfg.head.conductivities_S_per_m, cfg.head.n_legendre,
    )
    i = 17
    rtp = cartesian_to_spherical(pos[i])
    o = np.array([0.2, 0.5, 0.84]); o = o / np.linalg.norm(o)
    loc = world_orientation_to_spherical_frame(o, pos[i])
    v_direct = forward_potential(
        rtp[0], rtp[1], rtp[2], loc[0], loc[1], loc[2],
        elec.theta_rad, elec.phi_rad, M,
        brain_radius_m=cfg.head.radii_m[0],
    )
    v_direct = v_direct - v_direct.mean()
    v_from_L = L[i] @ np.array([loc[0], loc[1], loc[2]])
    assert np.max(np.abs(v_direct - v_from_L)) < 1e-10
    assert np.max(np.abs(L.mean(axis=1))) < 1e-8


# ---------------------------------------------------------------------------
# Electrode-graph export (save_graph) for GNN workflows
# ---------------------------------------------------------------------------

def test_build_electrode_graph_methods():
    from spherical_eeg_sim.io.graph_export import build_electrode_graph
    el = load_electrodes('standard_1010_69.csv', target_radius_m=0.092)
    xyz = np.asarray(el.cartesian)

    for method, kw in (('knn', dict(k=6)),
                       ('delaunay', {}),
                       ('distance', dict(distance_threshold_m=0.05))):
        g = build_electrode_graph(xyz, method=method, **kw)
        ei = g['edge_index']
        assert ei.shape[0] == 2
        assert ei.shape[1] == g['edge_weight'].shape[0] == g['edge_length_m'].shape[0]
        # Bidirectional/symmetric: every (i, j) has a matching (j, i).
        edge_set = set(map(tuple, ei.T.tolist()))
        assert all((j, i) in edge_set for i, j in edge_set), f"{method}: not symmetric"
        # No self-loops.
        assert not np.any(ei[0] == ei[1]), f"{method}: has self-loops"
        # Weights in (0, 1]; nearer edges weigh more (Gaussian of distance).
        assert np.all(g['edge_weight'] > 0) and np.all(g['edge_weight'] <= 1.0 + 1e-12)
        assert int(g['n_nodes']) == xyz.shape[0]
        assert g['degree'].shape[0] == xyz.shape[0]

    # kNN: the union symmetrisation can only add edges, so every node has
    # degree at least k.
    g = build_electrode_graph(xyz, method='knn', k=6)
    assert g['degree'].min() >= 6


def test_build_electrode_graph_rejects_invalid_direct_inputs():
    from spherical_eeg_sim.io.graph_export import build_electrode_graph
    xyz = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    with pytest.raises(ValueError, match='finite'):
        build_electrode_graph(
            np.array([[0.0, 0.0, 1.0], [np.nan, 1.0, 0.0]])
        )
    with pytest.raises(ValueError, match='positive integer'):
        build_electrode_graph(xyz, k=1.5)
    with pytest.raises(ValueError, match='smaller than'):
        build_electrode_graph(xyz, k=2)
    with pytest.raises(ValueError, match='positive distance'):
        build_electrode_graph(xyz, method='distance', distance_threshold_m=np.inf)
    with pytest.raises(ValueError, match='duplicate'):
        build_electrode_graph(np.array([[0.0, 0.0, 1.0]] * 2), k=1)


def test_graph_export_pipeline_hdf5(tmp_path):
    import h5py
    cfg = _tiny_cfg(tmp_path)
    cfg.output.save_graph = True
    cfg.output.graph_method = 'knn'
    cfg.output.graph_k = 6
    run_pipeline(cfg)

    gp = tmp_path / 'graph.h5'
    assert gp.is_file()
    n_elec = len(load_electrodes(cfg.electrode_file,
                                 target_radius_m=cfg.head.radii_m[-1]).names)
    with h5py.File(gp, 'r') as f:
        ei = f['edge_index'][:]
        w = f['edge_weight'][:]
        assert ei.shape[0] == 2 and ei.shape[1] == w.shape[0]
        # Node indices are valid electrode indices.
        assert ei.min() >= 0 and ei.max() < n_elec
        # Electrode block travels with the graph, node order matches.
        assert f['electrodes/cartesian_xyz'].shape[0] == n_elec
        assert f['metadata'].attrs['method'] == 'knn'


def test_graph_export_pipeline_npz(tmp_path):
    cfg = _tiny_cfg(tmp_path)
    cfg.output.format = 'npz'
    cfg.output.save_graph = True
    cfg.output.graph_method = 'delaunay'
    run_pipeline(cfg)

    gp = tmp_path / 'graph.npz'
    assert gp.is_file()
    with np.load(gp, allow_pickle=True) as d:
        for key in ('edge_index', 'edge_weight', 'edge_length_m', 'degree',
                    'electrode_names', 'electrode_cartesian_xyz'):
            assert key in d.files, f"missing {key} in graph.npz"
        assert d['edge_index'].shape[0] == 2


def test_graph_config_validation():
    cfg = Config()
    cfg.output.graph_method = 'not_a_method'
    with pytest.raises(ValueError):
        cfg.validate()
    cfg = Config()
    cfg.output.graph_method = 'distance'   # threshold required
    with pytest.raises(ValueError):
        cfg.validate()
    cfg = Config()
    cfg.output.graph_method = 'distance'
    cfg.output.graph_distance_threshold_m = 0.05
    cfg.validate()


# ---------------------------------------------------------------------------
# Descriptive sensor-topography concentration ground-truth field
# ---------------------------------------------------------------------------

def test_signal_topographic_concentration_present_and_finite(tmp_path):
    import h5py
    cfg = _tiny_cfg(tmp_path)
    run_pipeline(cfg)
    with h5py.File(tmp_path / 'train.h5', 'r') as f:
        foc = f['ground_truth/signal_topographic_concentration'][:]
    assert foc.shape == (cfg.dataset.n_train,)
    assert np.all(np.isfinite(foc))
    assert np.all(foc >= 0)


def test_signal_topographic_concentration_magnitude_invariant(tmp_path):
    """Concentration is a property of the spatial pattern, so scaling every dipole
    magnitude must leave it byte-identical at a fixed seed."""
    import h5py

    def run(scale, sub):
        cfg = _tiny_cfg(tmp_path / sub)
        cfg.dipole.cortical_magnitude_range_nAm = [5.0 * scale, 300.0 * scale]
        cfg.dipole.subcortical_magnitude_range_nAm = [30.0 * scale, 500.0 * scale]
        run_pipeline(cfg)
        with h5py.File(tmp_path / sub / 'train.h5', 'r') as f:
            return f['ground_truth/signal_topographic_concentration'][:]

    foc_1x = run(1.0, 'a')
    foc_50x = run(50.0, 'b')
    # The concentration statistic is a ratio in which the common magnitude
    # factor cancels, up to rounding of the intermediate scaled map.
    assert np.allclose(foc_1x, foc_50x, rtol=1e-9, atol=0.0)


# ---------------------------------------------------------------------------
# C7: output-format matrix and custom electrode CSV
# ---------------------------------------------------------------------------

def test_npz_output(tmp_path):
    cfg = _tiny_cfg(tmp_path)
    cfg.output.format = 'npz'
    run_pipeline(cfg)
    p = tmp_path / 'train.npz'
    assert p.is_file()
    with np.load(p, allow_pickle=True) as d:
        assert 'eeg' in d
        assert 'gt_signal_topographic_concentration' in d
        assert 'electrode_names' in d


def test_mat_output(tmp_path):
    hdf5storage = pytest.importorskip('hdf5storage')
    cfg = _tiny_cfg(tmp_path)
    cfg.output.format = 'mat'
    run_pipeline(cfg)
    p = tmp_path / 'train.mat'
    assert p.is_file()
    d = hdf5storage.loadmat(p)
    assert 'eeg' in d
    assert 'ground_truth' in d
    assert 'signal_topographic_concentration' in d['ground_truth'].dtype.names
    assert b'Created on: Thu Jan 01 00:00:00 1970' in p.read_bytes()[:128]


def test_pipeline_schematic_is_byte_reproducible(tmp_path):
    """Both shipped schematic formats must omit wall-clock metadata."""
    pytest.importorskip('matplotlib')
    import runpy

    namespace = runpy.run_path(
        str(_repo_root / 'scripts' / 'make_validation_figures.py')
    )
    first = tmp_path / 'first'
    second = tmp_path / 'second'
    first.mkdir()
    second.mkdir()
    namespace['fig_pipeline_schematic'](first)
    namespace['fig_pipeline_schematic'](second)
    for name in ('pipeline_schematic.pdf', 'pipeline_schematic.png'):
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_custom_electrode_csv(tmp_path):
    import h5py
    # Build a valid custom CSV from a subset of a bundled montage.
    arr = load_electrodes('standard_1010_69.csv', target_radius_m=0.092)
    keep = list(range(24))
    from spherical_eeg_sim.core.electrodes import ElectrodeArray
    sub = ElectrodeArray(
        names=[arr.names[i] for i in keep],
        cartesian=arr.cartesian[keep],
        spherical=arr.spherical[keep],
        radius_m=arr.radius_m,
    )
    csv_path = tmp_path / 'custom_24.csv'
    save_electrodes_csv(sub, csv_path)

    cfg = _tiny_cfg(tmp_path / 'run')
    cfg.electrode_file = str(csv_path)
    run_pipeline(cfg)
    with h5py.File(tmp_path / 'run' / 'train.h5', 'r') as f:
        assert f['eeg'].shape[1] == 24
        assert len(f['electrodes/names'][:]) == 24


# ---------------------------------------------------------------------------
# B2: MNE export (skipped if MNE is not installed)
# ---------------------------------------------------------------------------

def test_mne_dataset_to_epochs(tmp_path):
    pytest.importorskip('mne')
    from spherical_eeg_sim.io.mne_export import dataset_to_epochs
    cfg = _tiny_cfg(tmp_path)
    run_pipeline(cfg)
    epochs = dataset_to_epochs(tmp_path / 'train.h5')
    assert len(epochs) == cfg.dataset.n_train
    assert len(epochs.ch_names) == 69
    assert abs(epochs.info['sfreq'] - cfg.signal.sampling_freq_hz) < 1e-9
    # data preserved in volts
    import h5py
    with h5py.File(tmp_path / 'train.h5', 'r') as f:
        raw = f['eeg'][:]
    assert np.allclose(epochs.get_data(copy=False), raw)
    subset = dataset_to_epochs(tmp_path / 'train.h5', selection=slice(3, 8))
    assert len(subset) == 5
    assert np.allclose(subset.get_data(copy=False), raw[3:8])
    with pytest.raises(ValueError, match='no epochs'):
        dataset_to_epochs(tmp_path / 'train.h5', selection=slice(3, 3))


def test_mne_restores_normalized_output_to_volts(tmp_path):
    pytest.importorskip('mne')
    import h5py
    from spherical_eeg_sim.io.mne_export import dataset_to_epochs
    cfg = _tiny_cfg(tmp_path)
    cfg.noise.noise_model = 'snr_target'
    cfg.output.eeg_normalization = 'per_sample_rms'
    run_pipeline(cfg)
    with h5py.File(tmp_path / 'train.h5', 'r') as f:
        stored = f['eeg'][:]
        scale = f['ground_truth/eeg_normalization_scale_V'][:]
    epochs = dataset_to_epochs(tmp_path / 'train.h5')
    assert np.allclose(
        epochs.get_data(copy=False), stored * scale[:, None, None]
    )
    with pytest.raises(ValueError, match='require EEG in volts'):
        dataset_to_epochs(
            tmp_path / 'train.h5', restore_physical_units=False,
        )


def test_mne_to_epochs_from_arrays():
    pytest.importorskip('mne')
    from spherical_eeg_sim.io.mne_export import to_epochs
    rng = np.random.default_rng(0)
    eeg = rng.standard_normal((10, 8, 50)) * 1e-6
    names = [f'E{i}' for i in range(8)]
    epochs = to_epochs(eeg, sfreq=250.0, names=names)
    assert len(epochs) == 10
    assert len(epochs.ch_names) == 8


def test_mne_array_helpers_reject_invalid_direct_inputs():
    pytest.importorskip('mne')
    from spherical_eeg_sim.io.mne_export import make_montage, to_epochs
    with pytest.raises(ValueError, match='unique'):
        to_epochs(np.ones((1, 2, 4)), 250.0, ['Cz', 'Cz'])
    with pytest.raises(ValueError, match='finite'):
        to_epochs(np.array([[[np.nan, 0.0], [0.0, 0.0]]]), 250.0, ['A', 'B'])
    with pytest.raises(ValueError, match='finite coordinates'):
        make_montage(['A', 'B'], np.array([[0.0, 0.0, 1.0], [np.inf, 0.0, 1.0]]))


@pytest.mark.parametrize('mode', ['none', 'average', 'cz', 'custom'])
def test_mne_dataset_records_stored_reference_state(tmp_path, mode):
    pytest.importorskip('mne')
    from spherical_eeg_sim.io.mne_export import dataset_to_epochs
    cfg = _tiny_cfg(tmp_path)
    cfg.dataset.n_train = 8
    cfg.dataset.n_val = 3
    cfg.dataset.n_test = 3
    cfg.output.reference_mode = mode
    if mode == 'custom':
        cfg.output.reference_weights = [1.0] * 69
    run_pipeline(cfg)
    epochs = dataset_to_epochs(tmp_path / 'train.h5')
    if mode == 'average':
        assert any('Average EEG reference' in proj['desc'] for proj in epochs.info['projs'])
        assert not bool(epochs.info['custom_ref_applied'])
    elif mode in ('cz', 'custom'):
        assert bool(epochs.info['custom_ref_applied'])
        assert len(epochs.info['projs']) == 0
    else:
        assert not bool(epochs.info['custom_ref_applied'])
        assert len(epochs.info['projs']) == 0
        assert 'zero-spherical-mean' in epochs.info['description']


if __name__ == '__main__':
    import tempfile
    d = _Path(tempfile.mkdtemp())
    test_leadfield_export_matches_forward_model(d / '1')
    test_signal_topographic_concentration_present_and_finite(d / '2')
    test_signal_topographic_concentration_magnitude_invariant(d / '3')
    test_npz_output(d / '4')
    test_mat_output(d / '5')
    test_custom_electrode_csv(d / '6')
    print("export tests passed (MNE tests require pytest)")
