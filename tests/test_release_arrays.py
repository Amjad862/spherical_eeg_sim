"""Regression oracle for numeric arrays from the released default split."""

from pathlib import Path
import hashlib
import json
import logging
import platform
import tempfile
import warnings
import sys

import h5py
import numpy as np
import pytest
import scipy

from spherical_eeg_sim import Config, run_pipeline


RELEASE_COMMIT = '1a732114ab5a13cca5d4808ddf406274420b7a50'
FORMATS = {'hdf5': 'h5', 'npz': 'npz', 'mat': 'mat'}
MODES = ('point', 'multi_point', 'patch')
FIXTURE = Path(__file__).with_name('v1_0_0_array_digests.json')
PORTABLE_FIXTURE = Path(__file__).with_name('v1_0_0_portable_arrays.npz')


def numeric_environment(fmt):
    versions = {
        'os': platform.system(),
        'machine': platform.machine(),
        'python': platform.python_version(),
        'numpy': np.__version__,
        'scipy': scipy.__version__,
        'h5py': h5py.__version__,
        'hdf5': h5py.version.hdf5_version,
    }
    if fmt == 'mat':
        import hdf5storage
        versions['hdf5storage'] = hdf5storage.__version__
    return versions


def tiny_config(output_dir, mode, fmt):
    cfg = Config()
    cfg.output_dir = str(output_dir)
    cfg.source_mode = mode
    cfg.master_seed = 862
    cfg.dataset.n_train = 12
    cfg.dataset.n_val = 6
    cfg.dataset.n_test = 6
    cfg.sources.n_cortical_positions = 60
    cfg.sources.n_subcortical_positions = 30
    cfg.noise.background_pool_size = 16
    cfg.noise.n_background_sources = 8
    cfg.head.n_legendre = 40
    cfg.output.format = fmt
    cfg.output.chunk_samples = 5
    cfg.output.save_components = True
    cfg.output.save_leadfield = True
    return cfg


def array_digests(path, stem):
    result = {}

    def record(name, arr):
        arr = np.asarray(arr)
        if arr.dtype.kind not in 'biufc':
            return
        arr = np.ascontiguousarray(arr.astype(arr.dtype.newbyteorder('<'), copy=False))
        digest = hashlib.sha256()
        digest.update(str((arr.dtype.str, arr.shape)).encode('ascii'))
        digest.update(arr.tobytes())
        result[name] = digest.hexdigest()

    if path.suffix == '.npz':
        with np.load(path, allow_pickle=False) as payload:
            for name in payload.files:
                if name == 'eeg' or name.startswith('gt_') or (
                    stem in ('source_pool', 'leadfield') and
                    name in ('positions_xyz', 'population', 'split', 'leadfield')
                ):
                    record(name, payload[name])
    else:
        with h5py.File(path, 'r') as payload:
            def collect(name, obj):
                if isinstance(obj, h5py.Dataset):
                    record(name, obj[()])

            if stem in ('source_pool', 'leadfield'):
                for name in ('positions_xyz', 'population', 'split', 'leadfield'):
                    if name in payload:
                        record(name, payload[name][()])
            else:
                record('eeg', payload['eeg'][()])
                payload['ground_truth'].visititems(
                    lambda name, obj: collect('ground_truth/' + name, obj)
                )
    assert result
    return result


def generate_digests(output_dir, mode, fmt):
    cfg = tiny_config(output_dir, mode, fmt)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Patch support is sparse')
        run_pipeline(cfg, log_level=logging.ERROR)
    suffix = FORMATS[fmt]
    return {
        stem: array_digests(output_dir / f'{stem}.{suffix}', stem)
        for stem in ('source_pool', 'leadfield', 'train', 'val', 'test')
    }


def portable_config(output_dir, mode):
    cfg = tiny_config(output_dir, mode, 'hdf5')
    cfg.dataset.n_train, cfg.dataset.n_val, cfg.dataset.n_test = 3, 2, 2
    cfg.electrode_file = 'standard_1020_19.csv'
    cfg.signal.window_samples = 32
    cfg.noise.background_pool_size = 8
    cfg.noise.n_background_sources = 4
    return cfg


def portable_arrays(output_dir):
    arrays = {}
    for stem in ('source_pool', 'leadfield', 'train', 'val', 'test'):
        with h5py.File(output_dir / f'{stem}.h5', 'r') as file:
            def collect(name, node):
                if isinstance(node, h5py.Dataset) and node.dtype.kind in 'biufc':
                    arrays[f'{stem}/{name}'] = node[()]

            if stem in ('source_pool', 'leadfield'):
                for name in ('positions_xyz', 'population', 'split', 'leadfield'):
                    if name in file:
                        collect(name, file[name])
            else:
                collect('eeg', file['eeg'])
                file['ground_truth'].visititems(lambda name, node: collect('ground_truth/' + name, node))
    return arrays


def _require_released_package():
    import subprocess

    repo = Path(__file__).resolve().parents[1]
    actual = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=repo, text=True
    ).strip()
    changed = subprocess.check_output(
        ['git', 'diff', '--name-only', 'HEAD', '--', 'spherical_eeg_sim'],
        cwd=repo, text=True,
    ).strip()
    if actual != RELEASE_COMMIT or changed:
        raise RuntimeError('Capture requires unmodified released package source')


def capture_fixture():
    """Capture from an unchanged checkout of RELEASE_COMMIT."""
    _require_released_package()

    capture_current_files()


def capture_portable_fixture():
    """Capture portable values after copying this helper into v1.0.0."""
    _require_released_package()
    arrays = {'source_commit': np.array(RELEASE_COMMIT)}
    with tempfile.TemporaryDirectory() as temporary:
        for mode in MODES:
            output = Path(temporary) / mode
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='Patch support is sparse')
                run_pipeline(portable_config(output, mode), log_level=logging.ERROR)
            arrays.update({mode + '/' + name: value for name, value in portable_arrays(output).items()})
    np.savez_compressed(PORTABLE_FIXTURE, **arrays)


def capture_current_files():
    fixture = {'commit': RELEASE_COMMIT, 'formats': {}}
    with tempfile.TemporaryDirectory() as tmp:
        for fmt in FORMATS:
            cases = {'environment': numeric_environment(fmt), 'modes': {}}
            for mode in MODES:
                output = Path(tmp) / f'{fmt}_{mode}'
                cases['modes'][mode] = generate_digests(output, mode, fmt)
                print(f'Captured {fmt} {mode}', flush=True)
            fixture['formats'][fmt] = cases
    with FIXTURE.open('w', encoding='utf-8', newline='\n') as handle:
        json.dump(fixture, handle, indent=2, sort_keys=True)
        handle.write('\n')


@pytest.mark.parametrize('fmt', FORMATS)
@pytest.mark.parametrize('mode', MODES)
def test_default_arrays_match_v1_0_0(tmp_path, fmt, mode):
    if fmt == 'mat':
        pytest.importorskip('hdf5storage')
    fixture = json.loads(FIXTURE.read_text(encoding='utf-8'))
    assert fixture['commit'] == RELEASE_COMMIT
    expected = fixture['formats'][fmt]
    if numeric_environment(fmt) != expected['environment']:
        pytest.skip('Exact array digests require the recorded reference environment')
    assert generate_digests(tmp_path / 'dataset', mode, fmt) == expected['modes'][mode]


@pytest.mark.parametrize('mode', MODES)
def test_portable_numeric_continuity_against_v1_0_0(tmp_path, mode):
    output = tmp_path / mode
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Patch support is sparse')
        run_pipeline(portable_config(output, mode), log_level=logging.ERROR)
    actual = portable_arrays(output)
    with np.load(PORTABLE_FIXTURE, allow_pickle=False) as fixture:
        assert str(fixture['source_commit']) == RELEASE_COMMIT
        expected_keys = {name[len(mode) + 1:] for name in fixture.files if name.startswith(mode + '/')}
        assert set(actual) == expected_keys
        for name, value in actual.items():
            expected = fixture[f'{mode}/{name}']
            assert value.shape == expected.shape and value.dtype == expected.dtype
            if value.dtype.kind in 'biu':
                np.testing.assert_array_equal(value, expected)
            else:
                scale = max(float(np.nanmax(np.abs(expected), initial=0.0)), np.finfo(float).tiny)
                np.testing.assert_allclose(value, expected, rtol=1e-10, atol=1e-12 * scale, equal_nan=True)


if __name__ == '__main__':
    if '--capture-portable' in sys.argv:
        capture_portable_fixture()
    else:
        capture_fixture()
