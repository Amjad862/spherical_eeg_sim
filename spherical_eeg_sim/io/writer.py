"""
I/O utilities: write generated datasets to HDF5 (default), NPZ, or MAT files,
along with structured metadata.

Two ways to write a split:

  1. ``save_split(...)`` -- one-shot writer that takes the full split tensors
     in memory and writes them to disk. Suitable when the split fits in RAM.

  2. ``open_split_writer(...)`` -- streaming writer that accepts the split
     in chunks via ``.append(eeg_chunk, gt_chunk)`` calls. For HDF5, this
     produces extendable datasets and keeps at most one storage block of
     each dataset in memory (see ``ChunkedHDF5Writer``). For NPZ and MAT
     (which cannot extend on disk), the writer accumulates chunks in memory
     and writes them at finalize() time.

The streaming form is what the pipeline uses for large default-scale
datasets where one full EEG array would exceed available RAM.
"""

from __future__ import annotations
import copy
import hashlib
import io
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union
import zipfile

import numpy as np

from .._compat import import_hdf5storage
from ..ragged import RaggedArray, concatenate_ragged


def canonical_config(cfg_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Return the scientific configuration stored in canonical artifacts.

    Operational controls affect where/how a run executes, not which samples it
    defines.  Keeping them out of dataset metadata is necessary for complete
    directory identity across chunk sizes, worker counts, and output paths.
    """
    out = copy.deepcopy(_jsonify(cfg_dict))
    out.pop('output_dir', None)
    output = out.get('output', {})
    for key in ('chunk_samples', 'n_workers', 'overwrite'):
        output.pop(key, None)
    return out


def _atomic_replace(temp_path: Path, final_path: Path) -> None:
    for attempt in range(6):
        try:
            os.replace(temp_path, final_path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.1 * (2 ** attempt))


def _canonicalize_mat_userblock(path: Union[str, Path]) -> None:
    """Replace hdf5storage's wall-clock MAT header date with a fixed value.

    MATLAB v7.3 files carry a 128-byte user block. ``hdf5storage`` writes the
    current time into its ``Created on`` field even when every HDF5 object is
    otherwise deterministic. Replacing only that fixed-width text field keeps
    the file MATLAB-compatible while removing operational time from the
    canonical artifact.
    """
    path = Path(path)
    prefix = b'Created on: '
    suffix = b' HDF5 schema'
    fixed = b'Thu Jan 01 00:00:00 1970'
    with path.open('r+b') as f:
        header = f.read(128)
        start = header.find(prefix)
        end = header.find(suffix, start + len(prefix))
        if start < 0 or end < 0:
            raise RuntimeError(
                "MATLAB v7.3 writer produced an unrecognized user block; "
                "cannot remove its creation timestamp deterministically"
            )
        value_start = start + len(prefix)
        if end - value_start != len(fixed):
            raise RuntimeError(
                "MATLAB v7.3 creation-date field has an unexpected width"
            )
        f.seek(value_start)
        f.write(fixed)


def _deterministic_savez(
    path: Union[str, Path],
    arrays: Dict[str, np.ndarray],
    compress: bool,
) -> None:
    """Write an NPZ with fixed member metadata and stable key order."""
    path = Path(path)
    compression = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    with zipfile.ZipFile(path, mode='w', compression=compression,
                         compresslevel=9 if compress else None) as zf:
        for key in arrays:
            payload = io.BytesIO()
            np.lib.format.write_array(payload, np.asanyarray(arrays[key]),
                                      allow_pickle=False)
            info = zipfile.ZipInfo(f'{key}.npy', date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = compression
            info.external_attr = 0o600 << 16
            zf.writestr(info, payload.getvalue(), compress_type=compression,
                        compresslevel=9 if compress else None)


# =============================================================================
# Mode-aware ground-truth schema
# =============================================================================

# Ground-truth fields every split must contain, by source mode. This is a
# defensive contract: the pipeline already populates these, and the check
# guarantees a written split is self-describing and loadable downstream. It is
# enforced on the first append so a misconfigured writer fails fast rather than
# producing a silently incomplete file.
_BASE_GT_KEYS = (
    'positions_xyz', 'orientations_world', 'magnitudes_nAm', 'population',
    'source_radius_mm', 'source_depth_mm',
    'snr_db', 'snr_db_raw', 'snr_db_referenced',
    'signal_topographic_concentration', 'eeg_normalization_scale_V',
    'band_idx', 'frequencies_hz', 'phases_rad', 'component_amplitudes',
    'n_components', 'source_time_course',
    'position_pool_idx',
)
_MODE_EXTRA_GT_KEYS = {
    'point': (),
    'multi_point': (
        'n_dipoles_per_sample', 'separation_violated', 'time_coupling',
        'source_time_courses', 'source_base_band_idx',
        'source_base_frequencies_hz', 'source_base_phases_rad',
        'source_base_component_amplitudes', 'source_base_n_components',
        'source_primary_blend_weight', 'source_independent_blend_weight',
        'source_blend_std',
    ),
    'patch': (
        'patch_diameter_mm', 'n_active_per_patch',
        'patch_active_position_idx', 'patch_active_moments_nAm',
    ),
}


def required_gt_keys(source_mode: str) -> set:
    """Return the set of ground-truth keys a split must contain for a mode."""
    if source_mode not in _MODE_EXTRA_GT_KEYS:
        raise ValueError(
            f"Unknown source_mode {source_mode!r}; expected one of "
            f"{sorted(_MODE_EXTRA_GT_KEYS)}"
        )
    return set(_BASE_GT_KEYS) | set(_MODE_EXTRA_GT_KEYS[source_mode])


def _check_required_gt_keys(gt: Dict[str, Any], source_mode: Optional[str]) -> None:
    """Raise if the ground-truth dict is missing any mode-required key."""
    if source_mode is None:
        return
    missing = required_gt_keys(source_mode) - set(gt.keys())
    if missing:
        raise ValueError(
            f"Ground-truth dict for source_mode {source_mode!r} is missing "
            f"required field(s): {sorted(missing)}. Present fields: "
            f"{sorted(gt.keys())}."
        )


def _validate_sample_axes(eeg: np.ndarray, gt: Dict[str, Any]) -> int:
    """Validate the leading sample dimension of EEG and every truth field."""
    eeg_arr = np.asarray(eeg)
    if eeg_arr.ndim != 3:
        raise ValueError(
            "EEG must have shape (n_samples, n_channels, n_times); "
            f"got {eeg_arr.shape}"
        )
    n_samples = int(eeg_arr.shape[0])
    for key, value in gt.items():
        if isinstance(value, RaggedArray):
            if value.n_rows != n_samples:
                raise ValueError(
                    f"Ragged ground-truth field {key!r} has {value.n_rows} "
                    f"rows but EEG has {n_samples} samples"
                )
            continue
        arr = np.asarray(value)
        if arr.ndim < 1:
            raise ValueError(
                f"Ground-truth field {key!r} is 0-dimensional; every field "
                "must have a leading sample axis"
            )
        if arr.shape[0] != n_samples:
            raise ValueError(
                f"Ground-truth field {key!r} has {arr.shape[0]} rows but EEG "
                f"has {n_samples} samples"
            )
    return n_samples


# =============================================================================
# One-shot save (in-memory)
# =============================================================================

def save_split(
    output_dir: Union[str, Path],
    split_name: str,
    eeg: np.ndarray,
    ground_truth: Dict[str, Any],
    metadata: Dict[str, Any],
    fmt: str = 'hdf5',
    compress: bool = True,
    source_mode: Optional[str] = None,
    chunk_rows: int = 64,
    eeg_dtype: Union[str, np.dtype] = 'float64',
) -> Path:
    """Save one split (train/val/test) to disk in one call.

    Convenience wrapper around the streaming writer for callers that
    already have the full split in memory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = open_split_writer(
        output_dir, split_name, fmt, compress,
        source_mode=source_mode, chunk_rows=chunk_rows, eeg_dtype=eeg_dtype,
    )
    if isinstance(writer, ChunkedHDF5Writer):
        with writer:
            writer.append(eeg, ground_truth)
            return writer.finalize(metadata)
    writer.append(eeg, ground_truth)
    return writer.finalize(metadata)


# =============================================================================
# Streaming writer factory
# =============================================================================

def open_split_writer(
    output_dir: Union[str, Path],
    split_name: str,
    fmt: str = 'hdf5',
    compress: bool = True,
    source_mode: Optional[str] = None,
    chunk_rows: int = 64,
    electrodes: Optional[Dict[str, Any]] = None,
    eeg_dtype: Union[str, np.dtype] = 'float64',
):
    """Open a streaming writer for one split.

    Use:
        w = open_split_writer(out_dir, 'train', fmt='hdf5')
        for eeg_chunk, gt_chunk in chunks:
            w.append(eeg_chunk, gt_chunk)
        path = w.finalize(metadata)

    The HDF5 writer is also a context manager. Inside a ``with`` block a
    failure in ``append`` or ``finalize`` closes the file and removes the
    temporary files; without one they stay open until the object is
    collected.

    Parameters
    ----------
    output_dir : str or Path
    split_name : str
    fmt : 'hdf5' | 'npz' | 'mat'
    compress : bool
    source_mode : str or None
        If given ('point' | 'multi_point' | 'patch'), the first append is
        checked against that mode's required ground-truth schema.
    chunk_rows : int
        Number of samples per on-disk chunk along the leading axis for native
        HDF5 output. Affects read performance and storage layout only, not the
        generated arrays. MATLAB and NPZ output is materialized at finalize.
    electrodes : dict or None
        Optional self-describing electrode block embedded in the output:
        keys 'names', 'cartesian_xyz', 'spherical_r_theta_phi',
        'reference_mode', and 'reference_info'. Makes the dataset reproducible
        without the original electrode CSV.
    eeg_dtype : str or numpy dtype
        Floating dtype used for /eeg and saved component arrays. Usually
        'float64' for validation or 'float32' for large ML datasets.

    Returns
    -------
    A writer object with ``.append(eeg, gt)`` and ``.finalize(metadata)``
    methods.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if fmt == 'hdf5':
        return ChunkedHDF5Writer(
            output_dir / f'{split_name}.h5', compress,
            source_mode=source_mode, chunk_rows=chunk_rows, electrodes=electrodes,
            eeg_dtype=eeg_dtype,
        )
    if fmt == 'npz':
        return _NPZAccumulator(
            output_dir / f'{split_name}.npz', compress,
            source_mode=source_mode, chunk_rows=chunk_rows, electrodes=electrodes,
            eeg_dtype=eeg_dtype,
        )
    if fmt == 'mat':
        return _MATAccumulator(
            output_dir / f'{split_name}.mat', compress,
            source_mode=source_mode, chunk_rows=chunk_rows, electrodes=electrodes,
            eeg_dtype=eeg_dtype,
        )
    raise ValueError(f"Unknown output format: {fmt!r}")


def _write_electrodes_group(parent, electrodes: Dict[str, Any]) -> None:
    """Write a self-describing /electrodes group into an open h5py file/group."""
    import h5py
    if electrodes is None:
        return
    grp = parent.create_group('electrodes')
    names = list(electrodes.get('names', []))
    grp.create_dataset(
        'names', data=np.array(names, dtype=h5py.string_dtype('utf-8')),
        track_times=False,
    )
    if 'cartesian_xyz' in electrodes and electrodes['cartesian_xyz'] is not None:
        grp.create_dataset(
            'cartesian_xyz', data=np.asarray(electrodes['cartesian_xyz']),
            track_times=False,
        )
    if 'spherical_r_theta_phi' in electrodes and electrodes['spherical_r_theta_phi'] is not None:
        grp.create_dataset(
            'spherical_r_theta_phi',
            data=np.asarray(electrodes['spherical_r_theta_phi']),
            track_times=False,
        )
    grp.attrs['reference_mode'] = str(electrodes.get('reference_mode', 'none'))
    grp.attrs['reference_info'] = json.dumps(_jsonify(electrodes.get('reference_info', {})))


# =============================================================================
# Chunked HDF5 writer (true streaming)
# =============================================================================

class ChunkedHDF5Writer:
    """Streaming HDF5 writer using extendable datasets.

    The first ``append`` call determines the trailing-dimension shapes and
    dtypes of every dataset; subsequent calls must supply chunks with
    matching trailing dims (the leading sample-count dim can vary).

    The file remains open between appends and is finalised by
    ``finalize(metadata)``. The on-disk chunk size along the leading axis is
    ``chunk_rows`` (default 64), balancing single-sample access against
    minibatch reads.

    Rows are stored in blocks of complete storage chunks, in one fixed order
    over the datasets. A block holds ``chunk_rows`` rows, or the smallest
    multiple of ``chunk_rows`` that reaches 64 rows when ``chunk_rows`` is
    smaller, so its size depends on ``chunk_rows`` alone. Rows that do not
    yet fill a block wait in memory, and ``finalize`` stores the last,
    possibly shorter, block. The sequence of HDF5 calls is therefore a
    function of the data and of ``chunk_rows`` alone, not of how many rows
    each ``append`` supplies, so the file does not depend on
    ``chunk_samples`` or ``n_workers``. Writing every generation chunk as it
    arrived, as version 1.0.0 did, lets the layout follow the sizes of the
    appends: the first append fixes the initial extent of each dataset, and
    partly filled storage chunks are stored and replaced later.

    A dense field keeps the dtype of its first append; a later append with
    another dtype is an error and is not cast.
    """

    # Smallest number of rows stored by one HDF5 call. With one call per
    # storage chunk, ``chunk_rows=1`` would cost one resize and one write per
    # window and dataset.
    _MIN_BLOCK_ROWS = 64

    def __init__(
        self,
        path: Path,
        compress: bool = True,
        source_mode: Optional[str] = None,
        chunk_rows: int = 64,
        electrodes: Optional[Dict[str, Any]] = None,
        eeg_dtype: Union[str, np.dtype] = 'float64',
    ):
        import h5py
        self.path = Path(path)
        self._temp_path = self.path.with_name(
            f'.{self.path.stem}.writing{self.path.suffix}'
        )
        if self._temp_path.exists():
            self._temp_path.unlink()
        self.compress = compress
        self._compression = 'gzip' if compress else None
        self._source_mode = source_mode
        self._chunk_rows = max(1, int(chunk_rows))
        # A whole number of storage chunks; a function of chunk_rows alone.
        self._block_rows = self._chunk_rows * -(-self._MIN_BLOCK_ROWS // self._chunk_rows)
        self._electrodes = electrodes
        self._eeg_dtype = np.dtype(eeg_dtype)
        if self._eeg_dtype not in (np.dtype('float32'), np.dtype('float64')):
            raise ValueError("eeg_dtype must be float32 or float64")
        self._f = h5py.File(self._temp_path, 'w')
        try:
            self._gt_group = self._f.create_group('ground_truth')
        except BaseException as error:
            _clean_writer_resource(self._f.close, error)
            raise
        self._eeg_dset = None
        self._gt_dsets: Dict[str, Any] = {}
        # Dense ground-truth fields in creation order; with the EEG dataset
        # they define the fixed order in which storage blocks are written.
        self._dense_keys: List[str] = []
        # Own each incoming fragment until the block is complete. Repeatedly
        # extending a contiguous array would recopy all previously held rows.
        self._pending: List[List[np.ndarray]] = []
        self._pending_rows = 0
        self._written = 0   # rows accepted by append()
        self._stored = 0    # rows stored in the dense HDF5 datasets
        self._closed = False

    def _dense_arrays(
        self, eeg_chunk: np.ndarray, gt_chunk: Dict[str, Any]
    ) -> List[np.ndarray]:
        """EEG and dense truth rows of one append, in storage order."""
        arrays = [np.asarray(eeg_chunk).astype(self._eeg_dtype, copy=False)]
        arrays.extend(np.asarray(gt_chunk[key]) for key in self._dense_keys)
        return arrays

    def _store_block(self, arrays: Iterable[np.ndarray], start: int, stop: int) -> None:
        """Store rows ``start:stop`` of every dense array as the next block."""
        new_stored = self._stored + (stop - start)
        datasets = [self._eeg_dset] + [self._gt_dsets[key] for key in self._dense_keys]
        for dataset, array in zip(datasets, arrays):
            dataset.resize(new_stored, axis=0)
            dataset[self._stored:new_stored] = array[start:stop]
        self._stored = new_stored

    def _flush_pending(self) -> None:
        if len(self._pending) == 1:
            self._store_block(self._pending[0], 0, self._pending_rows)
        else:
            # Join one field at a time, once per block, in dataset order.
            fields = (np.concatenate(parts) for parts in zip(*self._pending))
            self._store_block(fields, 0, self._pending_rows)
        self._pending.clear()
        self._pending_rows = 0

    def _store_complete_blocks(self, arrays: List[np.ndarray], n: int) -> None:
        """Store every complete block and keep the remainder in memory."""
        block = self._block_rows
        offset = 0
        if self._pending_rows:
            take = min(block - self._pending_rows, n)
            self._pending.append([array[:take].copy() for array in arrays])
            self._pending_rows += take
            offset = take
            if self._pending_rows == block:
                self._flush_pending()
        while n - offset >= block:
            self._store_block(arrays, offset, offset + block)
            offset += block
        if offset < n:
            # Reached only with an empty buffer. Copy: the caller may reuse
            # its arrays before the rows are stored.
            self._pending.append([array[offset:].copy() for array in arrays])
            self._pending_rows = n - offset

    def append(self, eeg_chunk: np.ndarray, gt_chunk: Dict[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("Writer has already been finalized")
        n = _validate_sample_axes(eeg_chunk, gt_chunk)
        if n == 0:
            return
        if self._eeg_dset is None:
            _check_required_gt_keys(gt_chunk, self._source_mode)
            eeg_dset = self._f.create_dataset(
                'eeg',
                shape=(0,) + tuple(eeg_chunk.shape[1:]),
                maxshape=(None,) + tuple(eeg_chunk.shape[1:]),
                chunks=(self._chunk_rows,) + tuple(eeg_chunk.shape[1:]),
                dtype=self._eeg_dtype,
                compression=self._compression,
                track_times=False,
            )
            self._eeg_dset = eeg_dset
            for k, v in gt_chunk.items():
                if isinstance(v, RaggedArray):
                    spool_path = self._temp_path.with_name(
                        f'{self._temp_path.name}.{k}.ragged'
                    )
                    if spool_path.exists():
                        spool_path.unlink()
                    descriptor = {
                        'kind': 'ragged',
                        'spool_path': spool_path,
                        'spool': spool_path.open('wb'),
                        'dtype': v.values.dtype,
                        'trailing_shape': tuple(v.values.shape[1:]),
                        'row_splits': [0],
                        'value_count': 0,
                        'rows_written': 0,
                    }
                    self._gt_dsets[k] = descriptor
                    self._append_ragged_spool(descriptor, v)
                    continue
                arr = np.asarray(v)
                chunk_shape = (self._chunk_rows,) + tuple(arr.shape[1:])
                self._gt_dsets[k] = self._gt_group.create_dataset(
                    k,
                    shape=(0,) + tuple(arr.shape[1:]),
                    maxshape=(None,) + tuple(arr.shape[1:]),
                    chunks=chunk_shape,
                    dtype=arr.dtype,
                    compression=self._compression,
                    track_times=False,
                )
                self._dense_keys.append(k)
        else:
            if eeg_chunk.shape[1:] != self._eeg_dset.shape[1:]:
                raise ValueError(
                    f"EEG chunk trailing shape {eeg_chunk.shape[1:]} does not match "
                    f"established shape {self._eeg_dset.shape[1:]}"
                )
            if set(gt_chunk) != set(self._gt_dsets):
                missing = set(self._gt_dsets) - set(gt_chunk)
                added = set(gt_chunk) - set(self._gt_dsets)
                raise ValueError(
                    "Ground-truth fields changed mid-stream; "
                    f"missing={sorted(missing)}, added={sorted(added)}"
                )
            for k, v in gt_chunk.items():
                descriptor = self._gt_dsets[k]
                if isinstance(descriptor, dict):
                    if not isinstance(v, RaggedArray):
                        raise ValueError(
                            f"Ground-truth field {k!r} changed from ragged to dense"
                        )
                    if v.values.shape[1:] != descriptor['trailing_shape']:
                        raise ValueError(
                            f"Ragged GT field {k!r} trailing shape changed"
                        )
                    if v.values.dtype != descriptor['dtype']:
                        raise ValueError(
                            f"Ragged GT field {k!r} dtype changed"
                        )
                elif isinstance(v, RaggedArray):
                    raise ValueError(
                        f"Ground-truth field {k!r} changed from dense to ragged"
                    )
                else:
                    arr = np.asarray(v)
                    if arr.shape[1:] != descriptor.shape[1:]:
                        raise ValueError(
                            f"GT field {k!r} trailing shape {arr.shape[1:]} does "
                            f"not match established shape {descriptor.shape[1:]}"
                        )
                    if arr.dtype != descriptor.dtype:
                        # Casting would change values silently (a longer byte
                        # string is cut, a wider integer wraps).
                        raise ValueError(
                            f"GT field {k!r} dtype changed from "
                            f"{descriptor.dtype} to {arr.dtype}"
                        )
            for k, v in gt_chunk.items():
                descriptor = self._gt_dsets[k]
                if isinstance(descriptor, dict):
                    self._append_ragged_spool(descriptor, v)
        self._store_complete_blocks(self._dense_arrays(eeg_chunk, gt_chunk), n)
        self._written += n

    @staticmethod
    def _append_ragged_spool(
        descriptor: Dict[str, Any], value: RaggedArray
    ) -> None:
        """Append ragged values to a raw spool without affecting HDF5 layout."""
        values = np.ascontiguousarray(value.values, dtype=descriptor['dtype'])
        descriptor['spool'].write(values.tobytes(order='C'))
        counts = np.diff(value.row_splits)
        base = int(descriptor['value_count'])
        descriptor['row_splits'].extend(
            (base + np.cumsum(counts, dtype=np.int64)).tolist()
        )
        descriptor['value_count'] = base + int(values.shape[0])
        descriptor['rows_written'] += value.n_rows

    def _materialize_ragged_hdf5(
        self, key: str, descriptor: Dict[str, Any]
    ) -> None:
        """Create one canonical CSR group from its completed raw spool."""
        descriptor['spool'].close()
        grp = self._gt_group.create_group(key, track_order=False)
        grp.attrs['encoding'] = 'csr_rows'

        n_values = int(descriptor['value_count'])
        trailing_shape = tuple(descriptor['trailing_shape'])
        block_rows = 65536
        values_dset = grp.create_dataset(
            'values',
            shape=(n_values,) + trailing_shape,
            # A chunked dataset without rows needs an extendable first axis.
            maxshape=((None,) + trailing_shape) if n_values == 0 else None,
            chunks=(max(1, min(block_rows, max(1, n_values))),) + trailing_shape,
            dtype=descriptor['dtype'],
            compression=self._compression,
            track_times=False,
        )
        row_width = int(np.prod(trailing_shape, dtype=np.int64)) or 1
        itemsize = int(np.dtype(descriptor['dtype']).itemsize)
        with descriptor['spool_path'].open('rb') as spool:
            for start in range(0, n_values, block_rows):
                stop = min(start + block_rows, n_values)
                byte_count = (stop - start) * row_width * itemsize
                payload = spool.read(byte_count)
                if len(payload) != byte_count:
                    raise OSError(
                        f"Incomplete ragged spool for ground-truth field {key!r}"
                    )
                values_dset[start:stop] = np.frombuffer(
                    payload, dtype=descriptor['dtype']
                ).reshape((stop - start,) + trailing_shape)
            if spool.read(1):
                raise OSError(
                    f"Unexpected trailing bytes in ragged spool for {key!r}"
                )
        grp.create_dataset(
            'row_splits',
            data=np.asarray(descriptor['row_splits'], dtype=np.int64),
            chunks=(max(1, min(block_rows, len(descriptor['row_splits']))),),
            dtype=np.int64,
            compression=self._compression,
            track_times=False,
        )
        descriptor['spool_path'].unlink()

    def finalize(self, metadata: Dict[str, Any]) -> Path:
        if self._closed:
            return self.path
        if self._pending_rows:
            self._flush_pending()
        if self._stored != self._written:
            raise RuntimeError("HDF5 writer lost rows between append and finalize")
        for key, descriptor in self._gt_dsets.items():
            if (
                isinstance(descriptor, dict)
                and descriptor['rows_written'] != self._written
            ):
                raise ValueError("Ragged ground-truth row count does not match EEG")
            if isinstance(descriptor, dict):
                self._materialize_ragged_hdf5(key, descriptor)
        meta_group = self._f.create_group('metadata')
        _write_metadata_to_h5(meta_group, metadata)
        if self._electrodes is not None:
            _write_electrodes_group(self._f, self._electrodes)
        self._f.close()
        _atomic_replace(self._temp_path, self.path)
        self._closed = True
        return self.path

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._closed:
            def discard_temp_files():
                for entry in self._gt_dsets.values():
                    if isinstance(entry, dict):
                        entry['spool_path'].unlink(missing_ok=True)
                self._temp_path.unlink(missing_ok=True)

            # Release every handle before attempting to unlink any file.
            cleanup_errors = []
            for descriptor in self._gt_dsets.values():
                if isinstance(descriptor, dict):
                    cleanup_errors.append(_clean_writer_resource(descriptor['spool'].close, exc_val))
            cleanup_errors.append(_clean_writer_resource(self._f.close, exc_val))
            cleanup_errors.append(_clean_writer_resource(discard_temp_files, exc_val))
            self._closed = True
            if exc_val is None:
                for error in cleanup_errors:
                    if error is not None:
                        raise error


def _clean_writer_resource(action, original_error):
    try:
        action()
    except Exception as error:
        if original_error is not None:
            message = f'HDF5 writer cleanup failed: {error}'
            logging.getLogger(__name__).warning(message, exc_info=True)
            if hasattr(original_error, 'add_note'):
                original_error.add_note(message)
        return error
    return None


# =============================================================================
# In-memory accumulators (NPZ, MAT)
# =============================================================================

class _ChunkAccumulator:
    """Base class: accumulate (eeg, gt) chunks; concrete subclass writes on finalize."""

    def __init__(
        self,
        path: Path,
        compress: bool = True,
        source_mode: Optional[str] = None,
        chunk_rows: int = 64,
        electrodes: Optional[Dict[str, Any]] = None,
        eeg_dtype: Union[str, np.dtype] = 'float64',
    ):
        self.path = Path(path)
        self.compress = compress
        self._source_mode = source_mode
        self._chunk_rows = max(1, int(chunk_rows))
        self._electrodes = electrodes
        self._eeg_dtype = np.dtype(eeg_dtype)
        if self._eeg_dtype not in (np.dtype('float32'), np.dtype('float64')):
            raise ValueError("eeg_dtype must be float32 or float64")
        self._eeg_chunks: List[np.ndarray] = []
        self._gt_chunks: Dict[str, List[Any]] = {}
        self._gt_keys: Optional[List[str]] = None
        self._closed = False

    def append(self, eeg_chunk: np.ndarray, gt_chunk: Dict[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("Writer has already been finalized")
        n_samples = _validate_sample_axes(eeg_chunk, gt_chunk)
        if n_samples == 0:
            return
        if self._gt_keys is None:
            _check_required_gt_keys(gt_chunk, self._source_mode)
            self._gt_keys = list(gt_chunk.keys())
            for k in self._gt_keys:
                self._gt_chunks[k] = []
        else:
            if set(gt_chunk.keys()) != set(self._gt_keys):
                raise ValueError(
                    f"GT field set drifted between chunks: "
                    f"previous {sorted(self._gt_keys)}, now {sorted(gt_chunk.keys())}"
                )
        self._eeg_chunks.append(np.asarray(eeg_chunk, dtype=self._eeg_dtype))
        for k in self._gt_keys:
            value = gt_chunk[k]
            self._gt_chunks[k].append(
                value if isinstance(value, RaggedArray) else np.asarray(value)
            )

    def _materialize(self):
        eeg = np.concatenate(self._eeg_chunks, axis=0) if self._eeg_chunks else np.empty((0,))
        gt: Dict[str, Any] = {}
        if self._gt_keys is not None:
            for k in self._gt_keys:
                parts = self._gt_chunks[k]
                if parts and isinstance(parts[0], RaggedArray):
                    gt[k] = concatenate_ragged(parts)
                else:
                    gt[k] = np.concatenate(parts, axis=0)
        return eeg, gt


class _NPZAccumulator(_ChunkAccumulator):
    def finalize(self, metadata: Dict[str, Any]) -> Path:
        eeg, gt = self._materialize()
        save_dict: Dict[str, np.ndarray] = {'eeg': eeg}
        for k, v in gt.items():
            if isinstance(v, RaggedArray):
                save_dict[f'gt_{k}_values'] = v.values
                save_dict[f'gt_{k}_row_splits'] = v.row_splits
            else:
                save_dict[f'gt_{k}'] = v
        save_dict['_metadata_json'] = np.array(
            json.dumps(_jsonify(metadata), sort_keys=True, separators=(',', ':'))
        )
        if self._electrodes is not None:
            el = self._electrodes
            save_dict['electrode_names'] = np.array(list(el.get('names', [])))
            if el.get('cartesian_xyz') is not None:
                save_dict['electrode_cartesian_xyz'] = np.asarray(el['cartesian_xyz'])
            if el.get('spherical_r_theta_phi') is not None:
                save_dict['electrode_spherical_r_theta_phi'] = np.asarray(el['spherical_r_theta_phi'])
            save_dict['_electrode_reference_json'] = np.array(json.dumps(_jsonify({
                'reference_mode': el.get('reference_mode', 'none'),
                'reference_info': el.get('reference_info', {}),
            }), sort_keys=True, separators=(',', ':')))
        temp = self.path.with_name(f'.{self.path.stem}.writing{self.path.suffix}')
        _deterministic_savez(temp, save_dict, self.compress)
        _atomic_replace(temp, self.path)
        self._closed = True
        return self.path


class _MATAccumulator(_ChunkAccumulator):
    def finalize(self, metadata: Dict[str, Any]) -> Path:
        try:
            hdf5storage = import_hdf5storage()
        except ImportError as exc:  # pragma: no cover - optional dependency path
            raise ImportError(
                "MATLAB v7.3 output requires hdf5storage; install "
                "spherical_eeg_sim[mat]."
            ) from exc
        eeg, gt = self._materialize()
        ground_truth: Dict[str, Any] = {}
        for key, value in gt.items():
            if isinstance(value, RaggedArray):
                ground_truth[key] = {
                    'values': value.values,
                    'row_splits': value.row_splits,
                    'encoding': 'csr_rows',
                }
            else:
                ground_truth[key] = value
        payload: Dict[str, Any] = {
            'eeg': eeg,
            'ground_truth': ground_truth,
            'metadata_json': json.dumps(_jsonify(metadata), sort_keys=True),
        }
        if self._electrodes is not None:
            payload['electrodes'] = {
                'names': np.asarray(list(self._electrodes.get('names', []))),
                'cartesian_xyz': np.asarray(
                    self._electrodes.get('cartesian_xyz')
                ),
                'spherical_r_theta_phi': np.asarray(
                    self._electrodes.get('spherical_r_theta_phi')
                ),
                'reference_json': json.dumps(_jsonify({
                    'reference_mode': self._electrodes.get('reference_mode', 'none'),
                    'reference_info': self._electrodes.get('reference_info', {}),
                }), sort_keys=True),
            }
        temp = self.path.with_name(f'.{self.path.stem}.writing{self.path.suffix}')
        hdf5storage.savemat(
            temp,
            payload,
            fmt='7.3',
            appendmat=False,
            store_python_metadata=False,
            matlab_compatible=True,
            compress=self.compress,
            truncate_existing=True,
        )
        _canonicalize_mat_userblock(temp)
        _atomic_replace(temp, self.path)
        self._closed = True
        return self.path


# =============================================================================
# Metadata helpers
# =============================================================================

def _write_metadata_to_h5(group, meta):
    """Serialize a metadata dict into HDF5 attributes / sub-groups."""
    for k, v in meta.items():
        if isinstance(v, dict):
            sub = group.create_group(k)
            _write_metadata_to_h5(sub, v)
        elif isinstance(v, (list, tuple)):
            try:
                arr = np.asarray(v)
                if arr.dtype.kind in ('U', 'O'):
                    group.attrs[k] = json.dumps(v)
                else:
                    group.create_dataset(k, data=arr, track_times=False)
            except (TypeError, ValueError):
                group.attrs[k] = json.dumps(v)
        elif isinstance(v, np.ndarray):
            group.create_dataset(k, data=v, track_times=False)
        elif isinstance(v, (str, int, float, bool, np.integer, np.floating)):
            group.attrs[k] = v
        elif v is None:
            group.attrs[k] = 'null'
        else:
            group.attrs[k] = str(v)


def _jsonify(obj):
    """Recursively convert numpy types to Python natives for JSON."""
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


# =============================================================================
# Run-level manifest and human-readable README
# =============================================================================

def write_leadfield_operator(
    output_dir: Union[str, Path],
    leadfield: np.ndarray,
    positions_xyz: np.ndarray,
    population: np.ndarray,
    split_label: np.ndarray,
    electrodes: Dict[str, Any],
    head: Dict[str, Any],
    fmt: str = 'hdf5',
    compress: bool = True,
    split_policy: str = 'position',
    split_membership: Optional[np.ndarray] = None,
) -> Path:
    """Write the position-pool basis lead-field operator as an auxiliary file.

    The operator is the forward map evaluated once at every source position:
    ``leadfield[p]`` is an ``(n_channels, 3)`` matrix whose columns are the
    scalp potentials for unit radial, theta, and phi dipole moments (local
    spherical frame) at position ``p``. Any dipole moment produces scalp
    potentials ``L @ [p_r, p_theta, p_phi]``. Units are volts per A*m.

    Parameters
    ----------
    output_dir : str or Path
    leadfield : ndarray (P, n_channels, 3)
    positions_xyz : ndarray (P, 3), metres
    population : ndarray (P,), 0=cortical 1=subcortical
    split_label : ndarray (P,), 0=train 1=val 2=test, -1=excluded, -2=shared
    electrodes : dict
        Self-describing electrode block (names, cartesian_xyz,
        spherical_r_theta_phi, reference_mode, reference_info).
    head : dict
        Head-model provenance (radii_m, conductivities_S_per_m, n_legendre).
    fmt : 'hdf5' | 'npz' | 'mat'
        Container format. Writes ``leadfield.h5``, ``leadfield.npz``, or a
        MATLAB v7.3 ``leadfield.mat``.
    compress : bool
    split_policy : str
    split_membership : ndarray (P, 3), optional
        Shared-policy eligibility in train, val, test order.

    Returns
    -------
    Path to the written file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    leadfield = np.asarray(leadfield)
    meta = {
        'description': (
            'Basis lead-field operator L (P, n_channels, 3); columns are '
            'scalp potentials for unit (radial, theta, phi) dipole moments in '
            'the local spherical frame at each source position. '
            'V = L @ [p_r, p_theta, p_phi]. Units: volts per A*m.'
        ),
        'units': 'V/(A*m)',
        'frame': 'local spherical (radial, theta, phi)',
        'reference_mode': str(electrodes.get('reference_mode', 'none')),
        'reference_description': _jsonify(electrodes.get('reference_info', {})),
        'n_positions': int(leadfield.shape[0]),
        'n_channels': int(leadfield.shape[1]),
        'split_codes': {'train': 0, 'val': 1, 'test': 2},
        'population_codes': {'cortical': 0, 'subcortical': 1},
        'head': head,
    }
    membership_data = _describe_split_membership(meta, split_policy, split_membership)

    if fmt == 'npz':
        path = output_dir / 'leadfield.npz'
        save_dict = {
            'leadfield': leadfield,
            'positions_xyz': np.asarray(positions_xyz),
            'population': np.asarray(population),
            'split': np.asarray(split_label),
            'electrode_names': np.array(list(electrodes.get('names', []))),
            'electrode_cartesian_xyz': np.asarray(electrodes.get('cartesian_xyz')),
            'electrode_spherical_r_theta_phi': np.asarray(
                electrodes.get('spherical_r_theta_phi')),
            '_metadata_json': np.array(json.dumps(_jsonify(meta))),
            **membership_data,
        }
        temp = output_dir / '.leadfield.writing.npz'
        _deterministic_savez(temp, save_dict, compress)
        _atomic_replace(temp, path)
        return path


    if fmt == 'mat':
        try:
            hdf5storage = import_hdf5storage()
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "MATLAB v7.3 output requires hdf5storage; install "
                "spherical_eeg_sim[mat]."
            ) from exc
        path = output_dir / 'leadfield.mat'
        temp = output_dir / '.leadfield.writing.mat'
        payload = {
            'leadfield': leadfield,
            'positions_xyz': np.asarray(positions_xyz),
            'population': np.asarray(population),
            'split': np.asarray(split_label),
            'electrodes': {
                'names': np.asarray(list(electrodes.get('names', []))),
                'cartesian_xyz': np.asarray(electrodes.get('cartesian_xyz')),
                'spherical_r_theta_phi': np.asarray(
                    electrodes.get('spherical_r_theta_phi')
                ),
            },
            'metadata_json': json.dumps(_jsonify(meta), sort_keys=True),
            **membership_data,
        }
        hdf5storage.savemat(
            temp, payload, fmt='7.3', appendmat=False,
            store_python_metadata=False, matlab_compatible=True,
            compress=compress, truncate_existing=True,
        )
        _canonicalize_mat_userblock(temp)
        _atomic_replace(temp, path)
        return path

    # Native HDF5 container.
    import h5py
    path = output_dir / 'leadfield.h5'
    temp = output_dir / '.leadfield.writing.h5'
    comp = 'gzip' if compress else None
    with h5py.File(temp, 'w') as f:
        f.create_dataset(
            'leadfield', data=leadfield, compression=comp, track_times=False,
        )
        f.create_dataset(
            'positions_xyz', data=np.asarray(positions_xyz), compression=comp,
            track_times=False,
        )
        f.create_dataset(
            'population', data=np.asarray(population), track_times=False,
        )
        f.create_dataset(
            'split', data=np.asarray(split_label), track_times=False,
        )
        for name, values in membership_data.items():
            f.create_dataset(name, data=values, track_times=False)
        _write_electrodes_group(f, electrodes)
        mg = f.create_group('metadata')
        _write_metadata_to_h5(mg, meta)
    _atomic_replace(temp, path)
    return path


def write_source_pool(
    output_dir: Union[str, Path],
    positions_xyz: np.ndarray,
    population: np.ndarray,
    split_label: np.ndarray,
    fmt: str = 'hdf5',
    compress: bool = True,
    split_policy: str = 'position',
    split_membership: Optional[np.ndarray] = None,
    patch_containment_radius_mm: Optional[np.ndarray] = None,
) -> Path:
    """Write the global source-position pool referenced by sample truth."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        'positions_xyz': np.asarray(positions_xyz, dtype=np.float64),
        'population': np.asarray(population, dtype=np.int8),
        'split': np.asarray(split_label, dtype=np.int8),
    }
    meta = {
        'description': (
            'Global source-position pool. position_pool_idx and patch active '
            'position indices in split ground truth index rows of positions_xyz.'
        ),
        'position_units': 'm',
        'population_codes': {'cortical': 0, 'subcortical': 1},
        'split_codes': {'train': 0, 'val': 1, 'test': 2},
    }
    payload.update(_describe_split_membership(meta, split_policy, split_membership))
    if patch_containment_radius_mm is not None:
        payload['patch_containment_radius_mm'] = np.asarray(
            patch_containment_radius_mm, dtype=np.float64,
        )
        meta['patch_containment_radius_mm'] = (
            'Largest patch radius in mm (geodesic on the position radius) that a patch '
            'centred at each position can have under the region split policy; NaN for '
            'positions excluded by the buffer.'
        )
    if fmt == 'npz':
        path = output_dir / 'source_pool.npz'
        data: Dict[str, Any] = dict(payload)
        data['_metadata_json'] = np.array(
            json.dumps(meta, sort_keys=True, separators=(',', ':'))
        )
        temp = output_dir / '.source_pool.writing.npz'
        _deterministic_savez(temp, data, compress)
        _atomic_replace(temp, path)
        return path
    if fmt == 'mat':
        try:
            hdf5storage = import_hdf5storage()
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "MATLAB v7.3 output requires hdf5storage; install "
                "spherical_eeg_sim[mat]."
            ) from exc
        path = output_dir / 'source_pool.mat'
        temp = output_dir / '.source_pool.writing.mat'
        data = dict(payload)
        data['metadata_json'] = json.dumps(meta, sort_keys=True)
        hdf5storage.savemat(
            temp, data, fmt='7.3', appendmat=False,
            store_python_metadata=False, matlab_compatible=True,
            compress=compress, truncate_existing=True,
        )
        _canonicalize_mat_userblock(temp)
        _atomic_replace(temp, path)
        return path

    import h5py
    path = output_dir / 'source_pool.h5'
    temp = output_dir / '.source_pool.writing.h5'
    comp = 'gzip' if compress else None
    with h5py.File(temp, 'w') as f:
        f.create_dataset(
            'positions_xyz', data=payload['positions_xyz'], compression=comp,
            track_times=False,
        )
        f.create_dataset('population', data=payload['population'], track_times=False)
        f.create_dataset('split', data=payload['split'], track_times=False)
        if 'split_membership' in payload:
            f.create_dataset('split_membership', data=payload['split_membership'], track_times=False)
        if 'patch_containment_radius_mm' in payload:
            f.create_dataset(
                'patch_containment_radius_mm', data=payload['patch_containment_radius_mm'],
                compression=comp, track_times=False,
            )
        mg = f.create_group('metadata')
        _write_metadata_to_h5(mg, meta)
    _atomic_replace(temp, path)
    return path


def _describe_split_membership(
    metadata: Dict[str, Any], policy: str, membership: Optional[np.ndarray],
) -> Dict[str, np.ndarray]:
    if policy == 'region':
        metadata['split_policy'] = policy
        metadata['split_codes']['excluded'] = -1
    elif policy == 'shared':
        metadata['split_policy'] = policy
        metadata['split_codes']['shared'] = -2
        metadata['split_membership_columns'] = ['train', 'val', 'test']
    if membership is None:
        return {}
    return {'split_membership': np.asarray(membership, dtype=np.uint8)}


def write_checksum_manifest(output_dir: Union[str, Path]) -> Path:
    """Hash every canonical artifact except the checksum file itself."""
    output_dir = Path(output_dir)
    path = output_dir / 'checksums.sha256'
    temp = output_dir / '.checksums.writing.sha256'
    lines: List[str] = []
    for file_path in sorted(p for p in output_dir.rglob('*') if p.is_file()):
        if file_path in (path, temp) or file_path.name.startswith('.'):
            continue
        hasher = hashlib.sha256()
        with file_path.open('rb') as source:
            for block in iter(lambda: source.read(1024 * 1024), b''):
                hasher.update(block)
        digest = hasher.hexdigest()
        relative = file_path.relative_to(output_dir).as_posix()
        lines.append(f"{digest}  {relative}")
    with temp.open('w', newline='\n') as f:
        f.write('\n'.join(lines))
        f.write('\n')
    _atomic_replace(temp, path)
    return path


def write_run_manifest(
    output_dir: Union[str, Path],
    cfg_dict: Dict[str, Any],
    extras: Dict[str, Any],
) -> Path:
    """Write the top-level metadata.json describing the run.

    Always written as plain JSON for easy reading without h5py / MAT tools.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        'config': canonical_config(cfg_dict),
        'extras': _jsonify(extras),
        'package': 'spherical_eeg_sim',
        'artifact_schema': '1.0',
    }
    path = output_dir / 'metadata.json'
    temp = output_dir / '.metadata.writing.json'
    with temp.open('w', newline='\n') as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write('\n')
    _atomic_replace(temp, path)
    return path


def write_run_readme(
    output_dir: Union[str, Path],
    cfg_dict: Dict[str, Any],
    summary: Dict[str, Any],
) -> Path:
    """Write a human-readable README describing the dataset.

    Suitable for users who download just the dataset without the generator.
    """
    output_dir = Path(output_dir)
    fs = cfg_dict.get('signal', {}).get('sampling_freq_hz', 1)
    win = cfg_dict.get('signal', {}).get('window_samples', 0)
    output_cfg = cfg_dict.get('output', {})
    fmt = output_cfg.get('format', 'hdf5')
    suffix = {'hdf5': 'h5', 'npz': 'npz', 'mat': 'mat'}.get(fmt, fmt)
    eeg_units = (
        'dimensionless (unit total-EEG RMS)'
        if output_cfg.get('eeg_normalization') == 'per_sample_rms' else 'volts'
    )
    magnitude_convention = (
        'rms'
        if cfg_dict.get('signal', {}).get('zero_mean_signal', True)
        else 'waveform_scale'
    )
    source_mode = cfg_dict.get('source_mode', 'unknown')
    lines = [
        '=' * 70,
        ' EEG SOURCE LOCALIZATION DATASET',
        '=' * 70,
        '',
        "Generated by spherical_eeg_sim using the configuration in metadata.json.",
        '',
        '-' * 70,
        ' Dataset summary',
        '-' * 70,
        f"  Source mode      : {source_mode}",
        *([f"  Split policy     : {cfg_dict['split']['policy']}"]
          if cfg_dict.get('split', {}).get('policy', 'position') != 'position' else []),
        f"  Output backend   : {fmt}",
        f"  Stored EEG units : {eeg_units}",
        f"  Magnitude convention: {magnitude_convention}",
        f"  Number of channels: {summary.get('n_channels', 'unknown')}",
        f"  Window length    : {win} samples ({win / max(fs, 1) * 1000:.1f} ms)",
        f"  Sampling rate    : {fs} Hz",
        f"  Splits           : train={summary.get('n_train', 'unknown')}, "
        f"val={summary.get('n_val', 'unknown')}, test={summary.get('n_test', 'unknown')}",
        '',
        '-' * 70,
        ' Files',
        '-' * 70,
        f'  train.{suffix} / val.{suffix} / test.{suffix} -- dataset splits',
        '  metadata.json               -- canonical scientific configuration',
        '  environment.json            -- software and platform provenance',
        '  validation_report.json      -- measured internal checks',
        '  checksums.sha256            -- SHA-256 hashes of canonical artifacts',
        '  diagnostic_plots/           -- plots, if enabled',
        '  leadfield.* / graph.*       -- optional auxiliary operators',
        '  source_pool.*               -- global source-position provenance',
        '',
        '-' * 70,
        ' Logical arrays (HDF5 paths shown; NPZ/MAT use backend conventions)',
        '-' * 70,
        '  /eeg                              -- (B, n_channels, n_timepoints) cfg.output.dtype',
        f'                                       Stored units: {eeg_units}.',
        '  /ground_truth/positions_xyz       -- (B, 3) or (B, max_K, 3) for multi-source',
        '                                       Cartesian dipole positions in meters.',
        '  /ground_truth/orientations_world  -- World-frame unit orientations.',
        '  /ground_truth/magnitudes_nAm      -- Dipole moments in nA*m; interpretation',
        f'                                       follows magnitude_convention={magnitude_convention}.',
        '  /ground_truth/population          -- 0=cortical, 1=subcortical.',
        '  /ground_truth/source_radius_mm    -- source radius from model origin.',
        '  /ground_truth/source_depth_mm     -- brain radius minus source radius.',
        '  /ground_truth/snr_db              -- per-sample SNR in dB on the',
        '                                       snr_definition montage. With',
        '                                       noise_model="snr_target" this is the',
        '                                       requested target SNR; with',
        '                                       "absolute_floor" (default) it is the',
        '                                       emergent SNR set by the fixed noise',
        '                                       floor and forward-projected signal.',
        '                                       (noise_model is recorded in /metadata.)',
        '  /ground_truth/snr_db_raw          -- achieved SNR on the raw montage.',
        '  /ground_truth/snr_db_referenced   -- achieved SNR on the referenced /eeg.',
        '  /ground_truth/eeg_normalization_scale_V',
        '                                    -- volts per stored unit; multiply /eeg',
        '                                       by this per-sample value to restore volts.',
        '  /ground_truth/signal_topographic_concentration',
        '                                    -- descriptive peak-to-RMS sensor-map ratio;',
        '                                       not an anatomical depth estimator.',
        '  /ground_truth/band_idx            -- source-signal index into metadata band_names.',
        '  /ground_truth/position_pool_idx   -- per-sample index into the position pool.',
        '  /ground_truth/frequencies_hz      -- sinusoid component frequencies.',
        '  /ground_truth/phases_rad          -- sinusoid component phases.',
        '  /ground_truth/component_amplitudes -- sinusoid component weights.',
        '  /ground_truth/n_components        -- active sinusoidal components per sample.',
        '  /ground_truth/source_time_course  -- final primary source course.',
        '  /electrodes/...                   -- names, positions, reference.',
        '  /metadata/...                     -- per-split metadata.',
        '',
        '-' * 70,
        ' Citation',
        '-' * 70,
        '  Please cite the originating paper if you use this dataset.',
        '  See README.md in the source repository for details.',
        '',
        '=' * 70,
    ]
    mode_lines: List[str] = []
    if source_mode == 'multi_point':
        mode_lines = [
            '',
            '  Multi-point additions: final padded source_time_courses plus',
            '  each base spectral construction, blend weights/std, source count,',
            '  and separation_violated.',
        ]
    elif source_mode == 'patch':
        mode_lines = [
            '',
            '  Patch additions: diameter, active count, and CSR-encoded exact',
            '  global source-pool indices plus signed radial moments.',
        ]
    if mode_lines:
        citation_divider = lines.index(' Citation') - 1
        lines[citation_divider:citation_divider] = mode_lines
    path = output_dir / 'README.txt'
    temp = output_dir / '.README.writing.txt'
    with temp.open('w', newline='\n') as f:
        f.write('\n'.join(lines))
        f.write('\n')
    _atomic_replace(temp, path)
    return path
