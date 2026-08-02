"""
MNE-Python export / compatibility utilities.

Convenience helpers that turn generated datasets (or in-memory arrays) into
MNE-Python objects so the output drops straight into MNE-based source-imaging
and machine-learning workflows. Physical-voltage datasets pass through
unchanged. RMS-normalized datasets are restored to volts from their per-sample
stored scale because MNE requires SI volts.

MNE is an OPTIONAL dependency. These helpers import it lazily and raise a clear
error if it is missing; the rest of the package does not require MNE. Install
with ``pip install mne`` or ``pip install spherical_eeg_sim[mne]``.

Typical use::

    from spherical_eeg_sim.io.mne_export import dataset_to_epochs
    epochs = dataset_to_epochs('output_eeg_dataset/train.h5')
    # epochs.metadata carries scalar/vector per-epoch ground truth (population,
    # snr_db, source_depth_mm, orientation, ...) when available.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np


def _validate_channel_names(names: Sequence[str]) -> List[str]:
    """Return channel names after strict public-API validation."""
    result = list(names)
    if (not result or not all(isinstance(name, str) and name for name in result)):
        raise ValueError("names must be a non-empty sequence of non-empty strings")
    if len(set(result)) != len(result):
        raise ValueError("names must be unique")
    return result


def _require_mne():
    try:
        import mne  # noqa: F401
        return mne
    except ImportError as e:  # pragma: no cover - exercised only without MNE
        raise ImportError(
            "MNE-Python is required for spherical_eeg_sim.io.mne_export. "
            "Install it with `pip install mne` or "
            "`pip install spherical_eeg_sim[mne]`."
        ) from e


def make_montage(names: Sequence[str], cartesian_xyz: np.ndarray):
    """Build an MNE :class:`~mne.channels.DigMontage` from electrode positions.

    Parameters
    ----------
    names : sequence of str
        Electrode names, in channel order.
    cartesian_xyz : ndarray (N, 3)
        Electrode Cartesian positions in metres (the unit MNE expects).

    Returns
    -------
    mne.channels.DigMontage
    """
    mne = _require_mne()
    checked_names = _validate_channel_names(names)
    xyz = np.asarray(cartesian_xyz, dtype=np.float64)
    if xyz.shape != (len(checked_names), 3):
        raise ValueError(
            f"cartesian_xyz must be ({len(checked_names)}, 3); got {xyz.shape}"
        )
    if not np.all(np.isfinite(xyz)):
        raise ValueError("cartesian_xyz must contain only finite coordinates")
    ch_pos = {name: xyz[i] for i, name in enumerate(checked_names)}
    return mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='head')


def make_info(
    names: Sequence[str],
    sfreq: float,
    cartesian_xyz: Optional[np.ndarray] = None,
):
    """Build an MNE :class:`~mne.Info` for an EEG montage.

    If ``cartesian_xyz`` is given, the montage is attached so channel
    positions travel with the object.
    """
    mne = _require_mne()
    checked_names = _validate_channel_names(names)
    if not np.isfinite(sfreq) or sfreq <= 0.0:
        raise ValueError("sfreq must be finite and positive")
    info = mne.create_info(
        ch_names=checked_names, sfreq=float(sfreq), ch_types='eeg',
    )
    if cartesian_xyz is not None:
        info.set_montage(make_montage(checked_names, cartesian_xyz), verbose=False)
    return info


def _ground_truth_metadata(ground_truth: Optional[Dict[str, np.ndarray]], n: int):
    """Build a per-epoch metadata DataFrame from 1-D ground-truth fields.

    Only fields with one value per epoch are included; padded per-dipole
    tensors (multi-point) and other multi-dimensional fields are skipped.
    Returns None if pandas is unavailable or no usable field is present.
    """
    if ground_truth is None:
        return None
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover
        return None

    cols: Dict[str, np.ndarray] = {}
    for key, val in ground_truth.items():
        arr = np.asarray(val)
        if arr.shape[:1] != (n,):
            continue
        if arr.ndim == 1:
            cols[key] = arr
        elif arr.ndim == 2 and arr.shape[1] == 3:
            if key == 'positions_xyz':
                cols['pos_x'] = arr[:, 0]
                cols['pos_y'] = arr[:, 1]
                cols['pos_z'] = arr[:, 2]
            elif key == 'orientations_world':
                cols['ori_x'] = arr[:, 0]
                cols['ori_y'] = arr[:, 1]
                cols['ori_z'] = arr[:, 2]
    if not cols:
        return None
    return pd.DataFrame(cols)


def to_epochs(
    eeg: np.ndarray,
    sfreq: float,
    names: Sequence[str],
    cartesian_xyz: Optional[np.ndarray] = None,
    tmin: float = 0.0,
    ground_truth: Optional[Dict[str, np.ndarray]] = None,
):
    """Wrap an EEG tensor as an MNE :class:`~mne.EpochsArray`.

    Parameters
    ----------
    eeg : ndarray (B, n_channels, n_times)
        Scalp potentials in volts. Use :func:`dataset_to_epochs` for generated
        files because it restores target-SNR datasets from stored normalized
        units before constructing the MNE object.
    sfreq : float
        Sampling frequency in Hz.
    names : sequence of str
        Electrode names, in channel order.
    cartesian_xyz : ndarray (n_channels, 3), optional
        Electrode positions in metres; attached as a montage when given.
    tmin : float
        Time of the first sample in seconds (default 0).
    ground_truth : dict, optional
        Per-sample ground truth; 1-D fields (and positions_xyz) are attached as
        ``epochs.metadata`` when pandas is available.

    Returns
    -------
    mne.EpochsArray
    """
    mne = _require_mne()
    checked_names = _validate_channel_names(names)
    data = np.asarray(eeg, dtype=np.float64)
    if data.ndim != 3:
        raise ValueError(f"eeg must be (B, n_channels, n_times); got {data.shape}")
    if data.shape[0] < 1 or data.shape[2] < 1:
        raise ValueError("eeg must contain at least one epoch and one time sample")
    if not np.all(np.isfinite(data)):
        raise ValueError("eeg must contain only finite values")
    if not np.isfinite(tmin):
        raise ValueError("tmin must be finite")
    if data.shape[1] != len(checked_names):
        raise ValueError(
            f"eeg has {data.shape[1]} channels but {len(checked_names)} names were given"
        )
    info = make_info(checked_names, sfreq, cartesian_xyz)
    metadata = _ground_truth_metadata(ground_truth, data.shape[0])
    return mne.EpochsArray(
        data, info, tmin=float(tmin), metadata=metadata, verbose=False,
    )


def to_raw(
    eeg_2d: np.ndarray,
    sfreq: float,
    names: Sequence[str],
    cartesian_xyz: Optional[np.ndarray] = None,
):
    """Wrap a single continuous (n_channels, n_times) trace as a RawArray.

    Useful when a single long recording is preferred over epochs; to turn a
    batch of windows into one continuous stream, concatenate along the time
    axis before calling this.
    """
    mne = _require_mne()
    checked_names = _validate_channel_names(names)
    data = np.asarray(eeg_2d, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f"eeg_2d must be (n_channels, n_times); got {data.shape}")
    if data.shape[1] < 1 or not np.all(np.isfinite(data)):
        raise ValueError("eeg_2d must contain finite values and at least one time sample")
    if data.shape[0] != len(checked_names):
        raise ValueError(
            f"eeg_2d has {data.shape[0]} channels but {len(checked_names)} names were given"
        )
    info = make_info(checked_names, sfreq, cartesian_xyz)
    return mne.io.RawArray(data, info, verbose=False)


def _read_split_file(
    h5_path: Union[str, Path],
    selection: Optional[slice] = None,
):
    """Read a generated HDF5 split file, optionally by epoch slice."""
    import h5py
    h5_path = Path(h5_path)
    with h5py.File(h5_path, 'r') as f:
        if selection is None:
            index = slice(None)
        elif isinstance(selection, slice):
            start, stop, step = selection.indices(f['eeg'].shape[0])
            if step <= 0:
                raise ValueError("selection must have a positive slice step")
            index = slice(start, stop, step)
        else:
            raise TypeError("selection must be a slice or None")
        eeg = f['eeg'][index]
        names = [
            n.decode('utf-8') if isinstance(n, bytes) else str(n)
            for n in f['electrodes/names'][:]
        ]
        xyz = f['electrodes/cartesian_xyz'][:] if 'electrodes/cartesian_xyz' in f else None
        sfreq = None
        eeg_units = 'volts'
        reference_mode = 'none'
        reference_info: Dict[str, Any] = {}
        if 'metadata' in f:
            sfreq = f['metadata'].attrs.get('sampling_freq_hz', None)
            eeg_units = f['metadata'].attrs.get('eeg_units', 'volts')
            if isinstance(eeg_units, bytes):
                eeg_units = eeg_units.decode('utf-8')
        gt = {}
        if 'ground_truth' in f:
            for k in f['ground_truth'].keys():
                node = f['ground_truth'][k]
                # Ragged CSR truth is intentionally omitted from MNE epoch
                # metadata; attaching variable-length patch supports to a
                # DataFrame would be both lossy and unnecessarily large.
                if isinstance(node, h5py.Dataset):
                    gt[k] = node[index]
        if 'electrodes' in f:
            reference_mode = f['electrodes'].attrs.get(
                'reference_mode', 'none'
            )
            reference_json = f['electrodes'].attrs.get('reference_info', '{}')
            if isinstance(reference_mode, bytes):
                reference_mode = reference_mode.decode('utf-8')
            if isinstance(reference_json, bytes):
                reference_json = reference_json.decode('utf-8')
            try:
                reference_info = json.loads(str(reference_json))
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"{h5_path} contains invalid electrodes/reference_info JSON"
                ) from exc
    if sfreq is None:
        raise ValueError(
            f"{h5_path} has no metadata/sampling_freq_hz attribute; pass sfreq "
            f"to to_epochs() directly."
        )
    return (
        eeg, float(sfreq), names, xyz, gt, str(eeg_units),
        str(reference_mode), reference_info,
    )


def _mark_stored_reference(epochs, mode: str, info: Dict[str, Any]):
    """Represent the already-applied dataset reference in MNE metadata."""
    if mode == 'average':
        # An inactive average-reference projector records the reference state
        # without applying a second numerical transform to already-referenced
        # samples.
        epochs.set_eeg_reference(
            ref_channels='average', projection=True, verbose=False,
        )
    elif mode in ('cz', 'custom'):
        # An empty channel list is MNE's public way to mark data as having an
        # externally applied custom EEG reference without changing the data.
        epochs.set_eeg_reference(
            ref_channels=[], projection=False, verbose=False,
        )
    elif mode != 'none':
        raise ValueError(f"Unsupported stored reference mode: {mode!r}")

    description = info.get('description', '') if isinstance(info, dict) else ''
    if mode == 'none':
        reference_note = (
            "Stored EEG has no discrete electrode reference; the analytic "
            "n>=1 sphere series uses the zero-spherical-mean potential gauge."
        )
    else:
        reference_note = f"Stored EEG reference: {mode}. {description}".strip()
    epochs.info['description'] = reference_note
    return epochs


def dataset_to_epochs(
    h5_path: Union[str, Path],
    with_metadata: bool = True,
    restore_physical_units: bool = True,
    selection: Optional[slice] = None,
):
    """Load a generated HDF5 split file directly as an MNE EpochsArray.

    Reads ``/eeg``, the embedded ``/electrodes`` group, and the sampling rate
    from ``/metadata``. When ``with_metadata`` is True, per-epoch ground truth
    is attached as ``epochs.metadata`` (requires pandas). ``selection`` may be
    a positive-step epoch slice; it is applied while reading so a subset does
    not materialize the full split in memory.

    Only HDF5 output is supported here; for NPZ/MAT, load the arrays yourself
    and call :func:`to_epochs`.
    """
    (eeg, sfreq, names, xyz, gt, eeg_units,
     reference_mode, reference_info) = _read_split_file(h5_path, selection)
    if eeg.shape[0] == 0:
        raise ValueError("selection produced no epochs")
    if eeg_units == 'dimensionless_rms_normalized':
        if not restore_physical_units:
            raise ValueError(
                "MNE objects require EEG in volts; normalized dataset input "
                "must be restored with restore_physical_units=True"
            )
        if 'eeg_normalization_scale_V' not in gt:
            raise ValueError(
                "Normalized dataset is missing ground_truth/"
                "eeg_normalization_scale_V"
            )
        scale = np.asarray(gt['eeg_normalization_scale_V'], dtype=np.float64)
        if (scale.shape != (eeg.shape[0],) or not np.all(np.isfinite(scale))
                or np.any(scale <= 0.0)):
            raise ValueError(
                "ground_truth/eeg_normalization_scale_V must contain one "
                "finite positive value per epoch"
            )
        eeg = eeg * scale[:, None, None]
    elif eeg_units != 'volts':
        raise ValueError(f"Unsupported stored EEG units: {eeg_units!r}")
    epochs = to_epochs(
        eeg, sfreq, names, cartesian_xyz=xyz,
        ground_truth=gt if with_metadata else None,
    )
    return _mark_stored_reference(epochs, reference_mode, reference_info)


__all__ = [
    'make_montage',
    'make_info',
    'to_epochs',
    'to_raw',
    'dataset_to_epochs',
]
