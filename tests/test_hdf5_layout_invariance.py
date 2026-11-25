"""The HDF5 container must not record how rows were handed to the writer.

``output.chunk_samples`` and ``output.n_workers`` decide how many windows reach
the writer in one call.  The arrays never depended on that, but in version
1.0.0 the file bytes did: the writer stored every generation chunk as it
arrived, so the first append fixed the initial extent of each dataset and
partly filled storage chunks were stored and replaced later.  With the default
settings the files differed between ``chunk_samples`` values under HDF5 1.14.5
and under HDF5 2.0.0, with and without compression.  The writer now stores
blocks of complete storage chunks in one fixed order, and the block size
depends on ``hdf5_chunk_rows`` alone.
"""

import hashlib
from pathlib import Path

import numpy as np
import pytest

from spherical_eeg_sim.io.writer import ChunkedHDF5Writer
from spherical_eeg_sim.ragged import RaggedArray

h5py = pytest.importorskip('h5py')

DENSE_KEYS = ('positions_xyz', 'snr_db', 'pool_index', 'population', 'band_idx', 'separation_violated')


def _dataset(rows, channels, times, seed=20261112):
    rng = np.random.default_rng(seed)
    eeg = np.round(rng.standard_normal((rows, channels, times)).cumsum(axis=-1), 1)
    counts = rng.integers(1, 6, size=rows)
    row_splits = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    truth = {
        'positions_xyz': rng.standard_normal((rows, 3)),
        'snr_db': rng.uniform(0.0, 30.0, size=rows),
        'pool_index': rng.integers(0, 1000, size=rows).astype(np.int64),
        'population': rng.integers(0, 2, size=rows).astype(np.int8),
        'band_idx': rng.integers(0, 5, size=rows).astype(np.int32),
        'separation_violated': rng.random(rows) < 0.2,
        'support': RaggedArray(rng.integers(0, 1000, size=int(counts.sum())).astype(np.int32), row_splits),
    }
    return eeg, truth


def _rows(value, start, stop):
    if isinstance(value, RaggedArray):
        low, high = int(value.row_splits[start]), int(value.row_splits[stop])
        return RaggedArray(value.values[low:high], value.row_splits[start:stop + 1] - low)
    return value[start:stop]


def _write(path: Path, eeg, truth, sizes, *, compress, chunk_rows, eeg_dtype='float64',
           reverse_keys_after_first=False) -> Path:
    assert sum(sizes) == eeg.shape[0]
    start = 0
    with ChunkedHDF5Writer(path, compress=compress, chunk_rows=chunk_rows, eeg_dtype=eeg_dtype) as writer:
        for number, size in enumerate(sizes):
            stop = start + size
            keys = list(truth)
            if reverse_keys_after_first and number:
                keys.reverse()
            writer.append(eeg[start:stop], {key: _rows(truth[key], start, stop) for key in keys})
            start = stop
        writer.finalize({'n_samples': int(eeg.shape[0])})
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_content(path, eeg, truth, eeg_dtype='float64'):
    with h5py.File(path, 'r') as handle:
        assert handle['eeg'].dtype == np.dtype(eeg_dtype)
        assert np.array_equal(handle['eeg'][:], eeg.astype(eeg_dtype))
        for key in DENSE_KEYS:
            stored = handle['ground_truth'][key]
            assert stored.dtype == truth[key].dtype, key
            assert np.array_equal(stored[:], truth[key]), key
        group = handle['ground_truth/support']
        assert np.array_equal(group['values'][:], truth['support'].values)
        assert np.array_equal(group['row_splits'][:], truth['support'].row_splits)


# 50 rows of 0.5 MiB: with 20 rows a storage chunk has 10 MiB, more than the
# chunk cache of either HDF5 version; with 64 rows it is longer than the data.
LARGE = dict(rows=50, channels=64, times=1024)
LARGE_PARTITIONS = {
    'one-call': [50],
    'sevens': [7] * 7 + [1],
    'sixteens': [16, 16, 16, 2],
    'storage-aligned': [20, 20, 10],
    'single-rows': [1] * 50,
    'uneven': [3, 29, 1, 17],
}


@pytest.mark.parametrize('compress', [True, False])
@pytest.mark.parametrize('chunk_rows', [20, 64])
def test_file_bytes_do_not_depend_on_how_rows_are_appended(tmp_path, compress, chunk_rows):
    eeg, truth = _dataset(**LARGE)
    digests = {
        name: _sha256(_write(tmp_path / f'{name}.h5', eeg, truth, sizes, compress=compress,
                             chunk_rows=chunk_rows))
        for name, sizes in LARGE_PARTITIONS.items()
    }
    assert len(set(digests.values())) == 1, digests
    _assert_content(tmp_path / 'uneven.h5', eeg, truth)


# 300 rows of 4 KiB: many blocks for small storage chunks.  With chunk_rows of
# 1 and 3 a block has 64 and 66 rows; with 64 and 100 it is one storage chunk.
SMALL = dict(rows=300, channels=8, times=64)


def _random_partitions(rows, count, seed):
    rng = np.random.default_rng(seed)
    partitions = [[rows], [1] * rows]
    while len(partitions) < count:
        sizes, remaining = [], rows
        while remaining:
            size = int(min(remaining, rng.integers(1, 150)))
            sizes.append(size)
            remaining -= size
        partitions.append(sizes)
    return partitions


@pytest.mark.parametrize('chunk_rows', [1, 3, 64, 100])
@pytest.mark.parametrize('eeg_dtype', ['float64', 'float32'])
def test_small_storage_chunks_and_field_types_are_layout_independent(tmp_path, chunk_rows, eeg_dtype):
    eeg, truth = _dataset(**SMALL)
    digests = set()
    for number, sizes in enumerate(_random_partitions(SMALL['rows'], 6, seed=chunk_rows)):
        path = _write(tmp_path / f'{number}.h5', eeg, truth, sizes, compress=True, chunk_rows=chunk_rows,
                      eeg_dtype=eeg_dtype, reverse_keys_after_first=number % 2 == 1)
        digests.add(_sha256(path))
    assert len(digests) == 1
    _assert_content(tmp_path / '3.h5', eeg, truth, eeg_dtype)
    with h5py.File(tmp_path / '3.h5', 'r') as handle:
        assert handle['eeg'].chunks[0] == chunk_rows


def test_storage_chunks_larger_than_the_cache_are_what_the_large_case_exercises(tmp_path):
    eeg, truth = _dataset(**LARGE)
    path = _write(tmp_path / 'data.h5', eeg, truth, [50], compress=True, chunk_rows=20)
    with h5py.File(path, 'r') as handle:
        chunk_bytes = int(np.prod(handle['eeg'].chunks)) * handle['eeg'].dtype.itemsize
        assert chunk_bytes > 8 * 2 ** 20
        assert handle['eeg'].chunks == (20, LARGE['channels'], LARGE['times'])
    _assert_content(path, eeg, truth)


@pytest.mark.parametrize('sizes', [[1], [19], [20], [21], [40], [41, 9]])
def test_every_row_is_stored_whatever_the_final_partial_block(tmp_path, sizes):
    eeg, truth = _dataset(**LARGE)
    count = sum(sizes)
    eeg = eeg[:count]
    truth = {key: _rows(value, 0, count) for key, value in truth.items()}
    path = _write(tmp_path / 'data.h5', eeg, truth, sizes, compress=True, chunk_rows=20)
    _assert_content(path, eeg, truth)
    with h5py.File(path, 'r') as handle:
        assert handle['eeg'].shape[0] == count
        assert handle['ground_truth/support/row_splits'].shape[0] == count + 1


def test_the_writer_does_not_keep_references_to_caller_buffers(tmp_path):
    """A caller may reuse its chunk buffer between appends."""
    eeg, truth = _dataset(**SMALL)
    buffer = np.empty((7,) + eeg.shape[1:])
    snr = np.empty(7)
    with ChunkedHDF5Writer(tmp_path / 'data.h5', compress=False, chunk_rows=20) as writer:
        for start in range(0, 294, 7):
            buffer[:] = eeg[start:start + 7]
            snr[:] = truth['snr_db'][start:start + 7]
            writer.append(buffer, {'snr_db': snr})
        writer.finalize({'n_samples': 294})
    with h5py.File(tmp_path / 'data.h5', 'r') as handle:
        assert np.array_equal(handle['eeg'][:], eeg[:294])
        assert np.array_equal(handle['ground_truth/snr_db'][:], truth['snr_db'][:294])


@pytest.mark.parametrize('first, later', [
    (np.array([1, 2, 3], dtype=np.int32), np.array([2 ** 40, 5, 6], dtype=np.int64)),
    (np.array([b'ab', b'cd', b'ef']), np.array([b'ghijkl', b'mn', b'op'])),
    (np.array([1.0, 2.0, 3.0]), np.array([1, 2, 3])),
])
def test_a_dense_field_cannot_change_its_dtype_between_appends(tmp_path, first, later):
    """Casting to the first dtype would cut byte strings and wrap integers."""
    eeg, _ = _dataset(rows=6, channels=4, times=8)
    path = tmp_path / 'data.h5'
    with pytest.raises(ValueError, match='dtype changed'):
        with ChunkedHDF5Writer(path, compress=False, chunk_rows=4) as writer:
            writer.append(eeg[:3], {'field': first})
            writer.append(eeg[3:], {'field': later})
    assert not path.exists()
    assert not list(tmp_path.iterdir())


def test_a_ragged_field_without_any_value_can_be_finalized(tmp_path):
    eeg, _ = _dataset(rows=6, channels=4, times=8)
    empty = RaggedArray(np.zeros(0, dtype=np.int64), np.zeros(7, dtype=np.int64))
    with ChunkedHDF5Writer(tmp_path / 'data.h5', compress=True, chunk_rows=4) as writer:
        writer.append(eeg, {'support': empty, 'snr_db': np.arange(6.0)})
        writer.finalize({'n_samples': 6})
    with h5py.File(tmp_path / 'data.h5', 'r') as handle:
        assert handle['ground_truth/support/values'].shape == (0,)
        assert np.array_equal(handle['ground_truth/support/row_splits'][:], np.zeros(7, dtype=np.int64))
        assert np.array_equal(handle['ground_truth/snr_db'][:], np.arange(6.0))


def test_an_empty_writer_still_finalizes(tmp_path):
    with ChunkedHDF5Writer(tmp_path / 'data.h5', compress=True, chunk_rows=20) as writer:
        writer.finalize({'n_samples': 0})
    with h5py.File(tmp_path / 'data.h5', 'r') as handle:
        assert 'eeg' not in handle
        assert 'metadata' in handle


def test_small_appends_join_each_dense_row_at_most_once(tmp_path, monkeypatch):
    # Two full storage blocks and a tail. Count bytes, independent of CPU time.
    rows, chunk_rows = 163, 73
    eeg = np.arange(rows * 12, dtype=np.float64).reshape(rows, 4, 3)
    truth = {'index': np.arange(rows, dtype=np.int64)}
    whole = _write(tmp_path / 'whole.h5', eeg, truth, [rows],
                   compress=True, chunk_rows=chunk_rows)
    copied = 0
    concatenate = np.concatenate

    def measured(*args, **kwargs):
        nonlocal copied
        result = concatenate(*args, **kwargs)
        copied += result.nbytes
        return result

    monkeypatch.setattr(np, 'concatenate', measured)
    single = _write(tmp_path / 'single.h5', eeg, truth, [1] * rows,
                    compress=True, chunk_rows=chunk_rows)
    assert _sha256(single) == _sha256(whole)
    assert copied <= eeg.nbytes + truth['index'].nbytes
