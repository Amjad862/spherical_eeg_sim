"""
Electrode-position loading and validation.

The generator accepts any electrode CSV with the columns:
    Name, x, y, z, R, theta, phi

with units:
    x, y, z, R : meters
    theta, phi : DEGREES

This module provides:
  - resolve_electrode_path(): map a filesystem path or bundled-montage
    name to a concrete CSV path.
  - load_electrodes(): read the CSV, validate, and return a structured array.
  - validate_electrodes(): sanity checks (on-sphere, no duplicates, etc.).
  - generate_standard_montage(): build standard layouts (10-20, 10-10,
    10-05) from the bundled projected-position CSVs, at any scalp radius.

A theta/phi convention auto-detection step accepts CSVs that use a swapped
convention where the polar/azimuthal column labels are interchanged: the
loader picks whichever interpretation makes the spherical columns agree with
the (authoritative) x,y,z columns. Internally everything is canonicalised to
standard math/physics conventions (theta = polar from +z, phi = azimuth from
+x) in radians.
"""

from __future__ import annotations
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np


# =============================================================================
# Bundled-montage path resolution
# =============================================================================

def resolve_electrode_path(path_or_name: str) -> Path:
    """Resolve a user-supplied electrode-file string to a concrete path.

    Resolution rules (first match wins):
      1. If ``path_or_name`` is an existing filesystem path, use it.
      2. For a bare filename or montage name, try the bundled montages shipped
         under ``spherical_eeg_sim.montages``. The ``.csv`` suffix is added if
         missing. A missing path that contains a directory component is never
         replaced by a same-basename bundled file.
      3. Return the original string as a Path so the caller can produce
         a descriptive FileNotFoundError.

    Examples
    --------
    >>> resolve_electrode_path('standard_1010_69.csv')          # bundled
    >>> resolve_electrode_path('standard_1010_69')              # bundled
    >>> resolve_electrode_path('config/standard_1010_69.csv')   # unresolved explicit path
    >>> resolve_electrode_path('/abs/path/custom_montage.csv')  # filesystem
    """
    if not isinstance(path_or_name, str) or not path_or_name:
        raise ValueError("electrode_file must be a non-empty string")

    p = Path(path_or_name)
    if p.is_file():
        return p

    if p.parent == Path('.'):
        base = p.name if p.suffix == '.csv' else f'{p.name}.csv'
        try:
            from importlib.resources import files
            candidate = files('spherical_eeg_sim.montages').joinpath(base)
            try:
                with candidate.open('rb'):
                    pass
                return Path(str(candidate))
            except (FileNotFoundError, OSError):
                pass
        except (ModuleNotFoundError, ImportError, AttributeError):
            pass

    # Last resort: hand back the original so downstream error messages
    # reference what the user actually wrote.
    return p


# =============================================================================
# ElectrodeArray dataclass
# =============================================================================

@dataclass
class ElectrodeArray:
    """Container for an electrode montage.

    Attributes
    ----------
    names : list of str, length N
        Electrode names (e.g. 'Fz', 'Cz', 'Pz').
    cartesian : ndarray, shape (N, 3)
        Cartesian positions (meters).
    spherical : ndarray, shape (N, 3)
        Spherical positions as (r, theta, phi) in (m, rad, rad).
        theta is the polar angle from +z axis; phi is the azimuthal
        angle in the xy-plane measured from +x axis. Stored in RADIANS
        even though the source CSV is in degrees.
    radius_m : float
        Common scalp radius if all electrodes share one; otherwise the mean.
    """
    names: List[str]
    cartesian: np.ndarray
    spherical: np.ndarray
    radius_m: float

    @property
    def n_electrodes(self) -> int:
        return len(self.names)

    @property
    def theta_rad(self) -> np.ndarray:
        return self.spherical[:, 1]

    @property
    def phi_rad(self) -> np.ndarray:
        return self.spherical[:, 2]


# =============================================================================
# CSV loader
# =============================================================================

def load_electrodes(
    csv_path: Union[str, Path],
    expected_radius_m: Optional[float] = None,
    radius_tolerance_mm: float = 1.0,
    target_radius_m: Optional[float] = None,
) -> ElectrodeArray:
    """Load electrode positions from a CSV file.

    Expected columns (case-insensitive header match):
        Name, x, y, z, R, theta, phi
    where x,y,z,R are in meters and theta,phi are in degrees.

    ``csv_path`` may be either a literal filesystem path or a bundled
    montage name resolvable by :func:`resolve_electrode_path`.

    Parameters
    ----------
    csv_path : str or Path
    expected_radius_m : float, optional
        If provided (and ``target_radius_m`` is not), all electrode radii must
        be within ``radius_tolerance_mm`` of this value or a ValueError is
        raised. This is the strict, non-rescaling validation path.
    radius_tolerance_mm : float
        Tolerance for the ``expected_radius_m`` check.
    target_radius_m : float, optional
        If provided, every electrode is projected radially onto a sphere of
        this radius, preserving its angular position (theta, phi). This makes
        the bundled montages -- and any custom CSV -- radius-independent: a
        montage stored at one scalp radius is placed correctly for whatever
        ``head.radii_m[-1]`` the model uses. A sanity check on the source radii
        (median must be a plausible head radius in metres) still catches unit
        mismatches. When set, ``expected_radius_m`` is ignored.

    Returns
    -------
    ElectrodeArray

    Raises
    ------
    FileNotFoundError, ValueError on parsing or validation failures.
    """
    csv_path = resolve_electrode_path(str(csv_path))
    if not csv_path.is_file():
        raise FileNotFoundError(f"Electrode CSV not found: {csv_path}")
    for label, value in (
        ('expected_radius_m', expected_radius_m),
        ('target_radius_m', target_radius_m),
    ):
        if value is not None and (not np.isfinite(value) or value <= 0):
            raise ValueError(f"{label} must be a finite value > 0")
    if not np.isfinite(radius_tolerance_mm) or radius_tolerance_mm < 0:
        raise ValueError("radius_tolerance_mm must be a finite value >= 0")

    names: List[str] = []
    rows: List[Tuple[float, float, float, float, float, float]] = []

    with csv_path.open('r', newline='') as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"Electrode CSV {csv_path} is empty") from exc
        header_lower = [h.strip().lower() for h in header]
        col_lookup = {}
        for canonical, candidates in [
            ('name', ['name', 'label']),
            ('x', ['x']),
            ('y', ['y']),
            ('z', ['z']),
            ('r', ['r', 'radius']),
            ('theta', ['theta', 'theta_deg', 'polar']),
            ('phi', ['phi', 'phi_deg', 'azimuth']),
        ]:
            for c in candidates:
                if c in header_lower:
                    col_lookup[canonical] = header_lower.index(c)
                    break
            if canonical not in col_lookup:
                raise ValueError(
                    f"Electrode CSV missing required column '{canonical}'. "
                    f"Found columns: {header}"
                )

        for line_no, line in enumerate(reader, start=2):
            if not line or all(c.strip() == '' for c in line):
                continue
            try:
                names.append(str(line[col_lookup['name']]).strip())
                rows.append((
                    float(line[col_lookup['x']]),
                    float(line[col_lookup['y']]),
                    float(line[col_lookup['z']]),
                    float(line[col_lookup['r']]),
                    float(line[col_lookup['theta']]),
                    float(line[col_lookup['phi']]),
                ))
            except (ValueError, IndexError) as e:
                raise ValueError(
                    f"Failed to parse electrode CSV at line {line_no}: {e}"
                ) from e

    if not rows:
        raise ValueError(f"Electrode CSV {csv_path} contains no data rows")

    arr = np.asarray(rows, dtype=np.float64)
    if any(not name for name in names):
        raise ValueError("Electrode names must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(
            f"Electrode CSV {csv_path} contains non-finite coordinates or angles"
        )
    cartesian = arr[:, 0:3]
    radii = arr[:, 3]
    theta_deg = arr[:, 4]
    phi_deg = arr[:, 5]

    spherical = np.column_stack([
        radii,
        np.deg2rad(theta_deg),
        np.deg2rad(phi_deg),
    ])

    # Validate consistency between Cartesian and spherical entries.
    expected_x = radii * np.sin(np.deg2rad(theta_deg)) * np.cos(np.deg2rad(phi_deg))
    expected_y = radii * np.sin(np.deg2rad(theta_deg)) * np.sin(np.deg2rad(phi_deg))
    expected_z = radii * np.cos(np.deg2rad(theta_deg))

    err_standard = np.max(np.abs(np.column_stack([
        cartesian[:, 0] - expected_x,
        cartesian[:, 1] - expected_y,
        cartesian[:, 2] - expected_z,
    ])))

    if err_standard > 1e-3:
        # Swapped-convention fallback: the column labelled phi is actually the
        # polar angle and the column labelled theta is the azimuth.
        expected_x_swapped = radii * np.sin(np.deg2rad(phi_deg)) * np.cos(np.deg2rad(theta_deg))
        expected_y_swapped = radii * np.sin(np.deg2rad(phi_deg)) * np.sin(np.deg2rad(theta_deg))
        expected_z_swapped = radii * np.cos(np.deg2rad(phi_deg))
        err_swapped = np.max(np.abs(np.column_stack([
            cartesian[:, 0] - expected_x_swapped,
            cartesian[:, 1] - expected_y_swapped,
            cartesian[:, 2] - expected_z_swapped,
        ])))
        if err_swapped <= 1e-3:
            # The swapped interpretation is consistent, so the CSV is accepted.
            # No spherical array is built here: Cartesian is authoritative and
            # the canonical spherical representation is recomputed from it
            # below, which would overwrite anything assigned at this point.
            pass
        else:
            raise ValueError(
                f"Electrode CSV {csv_path} has Cartesian/spherical mismatch "
                f"(max error {err_standard * 1000:.2f} mm). Check that x,y,z "
                f"columns are consistent with R, theta, phi columns under "
                f"either the standard or the swapped (theta<->phi) convention."
            )

    if target_radius_m is not None:
        # Radius-independent placement: project every electrode radially onto a
        # sphere of target_radius_m, preserving angular position (theta, phi).
        median_r = float(np.median(radii))
        if median_r < 0.01 or median_r > 0.30:
            raise ValueError(
                f"Electrode CSV {csv_path} has implausible source radii "
                f"(median {median_r * 1000:.1f} mm); positions are expected in "
                f"metres. Check the units of the x, y, z, R columns."
            )
        norms = np.linalg.norm(cartesian, axis=1)
        if np.any(norms < 1e-9):
            raise ValueError(
                f"Electrode CSV {csv_path} has a zero-length position vector; "
                f"cannot project onto the scalp sphere."
            )
        cartesian = cartesian * (target_radius_m / norms)[:, None]
    else:
        norms = np.linalg.norm(cartesian, axis=1)
        if expected_radius_m is not None:
            max_dev_mm = np.max(np.abs(norms - expected_radius_m)) * 1000
            if max_dev_mm > radius_tolerance_mm:
                raise ValueError(
                    f"Electrode radii deviate from expected scalp radius "
                    f"{expected_radius_m * 1000:.1f} mm by up to "
                    f"{max_dev_mm:.2f} mm (tolerance {radius_tolerance_mm:.1f} mm)"
                )

    # Cartesian coordinates are authoritative after the input convention has
    # been checked. Recompute the canonical spherical representation so a
    # tolerated CSV round-off mismatch cannot leave the two representations
    # divergent, and so radial projection updates both radii and angles.
    norms = np.linalg.norm(cartesian, axis=1)
    if np.any(norms <= 0.0):
        raise ValueError("Electrode Cartesian positions must be non-origin")
    theta = np.arccos(np.clip(cartesian[:, 2] / norms, -1.0, 1.0))
    phi = np.mod(np.arctan2(cartesian[:, 1], cartesian[:, 0]), 2 * np.pi)
    spherical = np.column_stack([norms, theta, phi])
    radius_m = float(np.mean(norms))

    array = ElectrodeArray(
        names=names,
        cartesian=cartesian,
        spherical=spherical,
        radius_m=radius_m,
    )
    validate_electrodes(array)
    return array


def validate_electrodes(arr: ElectrodeArray) -> None:
    """Run sanity checks on a loaded electrode array.

    Raises ValueError if any check fails.
    """
    if arr.n_electrodes < 4:
        raise ValueError(f"Need at least 4 electrodes; got {arr.n_electrodes}")
    if (arr.cartesian.shape != (arr.n_electrodes, 3)
            or arr.spherical.shape != (arr.n_electrodes, 3)):
        raise ValueError(
            "Electrode Cartesian and spherical arrays must both have shape (N, 3)"
        )
    if not np.all(np.isfinite(arr.cartesian)) or not np.all(np.isfinite(arr.spherical)):
        raise ValueError("Electrode positions must contain only finite values")
    if not np.isfinite(arr.radius_m) or arr.radius_m <= 0:
        raise ValueError("Electrode radius_m must be a finite value > 0")
    if any(not isinstance(name, str) or not name.strip() for name in arr.names):
        raise ValueError("Electrode names must be non-empty strings")

    # No duplicate names (case-insensitive).
    lower = [n.lower() for n in arr.names]
    if len(set(lower)) != len(lower):
        seen = set()
        dupes = []
        for n in lower:
            if n in seen:
                dupes.append(n)
            seen.add(n)
        raise ValueError(f"Duplicate electrode names (case-insensitive): {sorted(set(dupes))}")

    # No coincident positions.
    pairwise_dist = np.linalg.norm(
        arr.cartesian[:, None, :] - arr.cartesian[None, :, :],
        axis=-1
    )
    np.fill_diagonal(pairwise_dist, np.inf)
    min_dist = pairwise_dist.min()
    if min_dist < 1e-4:  # 0.1 mm
        raise ValueError(
            f"Two electrodes are within {min_dist * 1000:.4f} mm of each other"
        )

    # Radii must be positive. Their physical scale is controlled by the head
    # model, so this structural validator does not impose an adult-head bound.
    radii = arr.spherical[:, 0]
    if np.any(radii <= 0.0):
        raise ValueError("Electrode spherical radii must be positive")
    cartesian_radii = np.linalg.norm(arr.cartesian, axis=1)
    if not np.allclose(cartesian_radii, radii, rtol=0.0, atol=1e-9):
        raise ValueError("Electrode Cartesian and spherical radii are inconsistent")
    expected_cartesian = np.column_stack([
        radii * np.sin(arr.spherical[:, 1]) * np.cos(arr.spherical[:, 2]),
        radii * np.sin(arr.spherical[:, 1]) * np.sin(arr.spherical[:, 2]),
        radii * np.cos(arr.spherical[:, 1]),
    ])
    if not np.allclose(arr.cartesian, expected_cartesian, rtol=0.0, atol=1e-9):
        raise ValueError(
            "Electrode Cartesian and spherical coordinates are inconsistent"
        )

    # Theta in [0, pi].
    theta = arr.spherical[:, 1]
    if np.any(theta < -1e-9) or np.any(theta > np.pi + 1e-9):
        raise ValueError(
            f"Electrode polar angles out of [0, pi]: "
            f"min={np.degrees(theta.min()):.1f}, max={np.degrees(theta.max()):.1f}"
        )

    # Phi in [0, 2 pi) with small tolerance.
    phi = arr.spherical[:, 2]
    if np.any(phi < -1e-9) or np.any(phi > 2 * np.pi + 1e-9):
        raise ValueError(
            f"Electrode azimuthal angles out of [0, 2 pi): "
            f"min={np.degrees(phi.min()):.1f}, max={np.degrees(phi.max()):.1f}"
        )


def save_electrodes_csv(
    arr: ElectrodeArray,
    path: Union[str, Path],
) -> None:
    """Save an ElectrodeArray to CSV in the canonical format.

    Output columns: Name, x, y, z, R, theta, phi (theta, phi in degrees).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Name', 'x', 'y', 'z', 'R', 'theta', 'phi'])
        for i, name in enumerate(arr.names):
            x, y, z = arr.cartesian[i]
            r = arr.spherical[i, 0]
            theta = np.degrees(arr.spherical[i, 1])
            phi = np.degrees(arr.spherical[i, 2])
            writer.writerow([
                name,
                f'{x:.9g}', f'{y:.9g}', f'{z:.9g}',
                f'{r:.9g}', f'{theta:.9g}', f'{phi:.9g}',
            ])


# =============================================================================
# Standard montage generation
# =============================================================================


def generate_standard_montage(
    system: str = '10-10',
    radius_m: float = 0.092,
    n_select: Optional[int] = None,
    selection: Optional[List[str]] = None,
) -> ElectrodeArray:
    """Build a standard montage from the bundled projected positions.

    Positions are taken from the bundled montage CSVs (derived by radial
    projection of the BSD-licensed ``standard_1005`` template; see
    :mod:`spherical_eeg_sim.montages._generate`) and placed on a sphere of
    ``radius_m``. Coordinates are NOT individually MRI-warped; for
    MRI-realistic positions use MNE-Python's ``standard_1005`` montage on the
    relevant scalp surface.

    Parameters
    ----------
    system : str
        '10-20' (~20-channel subset), '10-10' (full ~69 recording channels;
        template fiducials excluded while recording electrodes such as Iz are
        retained), '10-05' (full high-density,
        ~336 channels), or 'custom' (use ``selection``).
    radius_m : float
        Scalp radius. Default 0.092 m (92 mm) matches the head model.
    n_select : int, optional
        If set, return only the first ``n_select`` electrodes.
    selection : list of str, optional
        Explicit list of electrode names to include (drawn from the 10-05 set).

    Returns
    -------
    ElectrodeArray
    """
    system = system.lower()
    montage_for_system = {
        '10-20': 'standard_1020_20.csv',
        '10-10': 'standard_1010_69.csv',
        '10-05': 'standard_1005_336.csv',
    }
    if system == 'custom':
        if selection is None:
            raise ValueError("system='custom' requires selection=[...]")
        base = 'standard_1005_336.csv'
    elif system in montage_for_system:
        base = montage_for_system[system]
    else:
        raise ValueError(f"Unknown montage system: {system!r}")

    arr = load_electrodes(base, target_radius_m=radius_m)

    keep: Optional[List[int]] = None
    if selection is not None:
        keep_set = set(selection)
        keep = [i for i, n in enumerate(arr.names) if n in keep_set]
        missing = keep_set - {arr.names[i] for i in keep}
        if missing:
            raise ValueError(f"Unknown electrode names in selection: {sorted(missing)}")
    elif n_select is not None:
        keep = list(range(min(n_select, arr.n_electrodes)))

    if keep is not None:
        if not keep:
            raise ValueError("No electrodes selected")
        arr = ElectrodeArray(
            names=[arr.names[i] for i in keep],
            cartesian=arr.cartesian[keep],
            spherical=arr.spherical[keep],
            radius_m=arr.radius_m,
        )
        validate_electrodes(arr)
    return arr


# =============================================================================
# EEG referencing
# =============================================================================

def resolve_reference(
    names: List[str],
    mode: str = 'none',
    channel: str = 'Cz',
    weights: Optional[List[float]] = None,
) -> dict:
    """Build a reference descriptor for :func:`apply_reference`.

    Resolving the descriptor once (it needs the electrode names) keeps the
    per-sample hot path free of name lookups and lets the chosen reference be
    recorded in metadata.

    Parameters
    ----------
    names : list of str
        Electrode names in channel order.
    mode : 'none' | 'average' | 'cz' | 'custom'
    channel : str
        Channel name used for 'cz' (default 'Cz') and for 'custom' when no
        ``weights`` are supplied.
    weights : list of float, optional
        For 'custom': per-channel weights of the reference combination. Must
        have one entry per channel; normalized to sum to 1.

    Returns
    -------
    dict
        One of:
          {'mode': 'none'}
          {'mode': 'average'}
          {'mode': 'single', 'idx': int, 'channel': str}
          {'mode': 'weights', 'w': ndarray}
        plus a JSON-friendly 'description' / 'channel'/'weights' for metadata.
    """
    mode = (mode or 'none').lower()
    n = len(names)
    if mode == 'none':
        return {
            'mode': 'none',
            'description': (
                'no discrete electrode reference; analytic n>=1 series uses '
                'the zero-spherical-mean potential gauge'
            ),
        }
    if mode == 'average':
        return {'mode': 'average', 'description': 'common average reference'}
    if mode in ('cz', 'custom'):
        if mode == 'custom' and weights is not None:
            w = np.asarray(weights, dtype=np.float64)
            if w.shape != (n,):
                raise ValueError(
                    f"reference_weights has length {w.shape[0] if w.ndim else 'scalar'} "
                    f"but the montage has {n} channels"
                )
            s = w.sum()
            if abs(s) < 1e-12:
                raise ValueError("reference_weights must not sum to zero")
            w = w / s
            return {
                'mode': 'weights', 'w': w,
                'description': 'custom weighted reference',
                'weights': w.tolist(),
            }
        # Single named reference channel ('cz', or 'custom' without weights).
        if channel not in names:
            raise ValueError(
                f"reference_channel {channel!r} is not in the montage "
                f"({n} channels). Available: {names}"
            )
        idx = names.index(channel)
        return {
            'mode': 'single', 'idx': idx, 'channel': channel,
            'description': f"single-channel reference ({channel})",
        }
    raise ValueError(f"Unknown reference mode {mode!r}")


def apply_reference(eeg: np.ndarray, ref: dict) -> np.ndarray:
    """Apply a reference descriptor to EEG of shape (..., N_channels, T).

    Referencing subtracts, at each time point, either a single channel's value,
    the across-channel mean, or a weighted channel combination. It is a linear
    operation on the channel axis (axis -2), so it commutes with signal+noise
    superposition.
    """
    m = ref['mode']
    if m == 'none':
        # Return a copy rather than the caller's array. Every other branch
        # produces a new array, and returning the input here would alias the
        # unreferenced signal into the referenced result, so a later in-place
        # operation on one would silently corrupt the other.
        return eeg.copy()
    if m == 'average':
        return eeg - eeg.mean(axis=-2, keepdims=True)
    if m == 'single':
        idx = ref['idx']
        return eeg - eeg[..., idx:idx + 1, :]
    if m == 'weights':
        w = ref['w']  # (N,)
        # Weighted sum over channel axis -> (..., T), broadcast back.
        ref_trace = np.tensordot(eeg, w, axes=([-2], [0]))
        return eeg - ref_trace[..., None, :]
    raise ValueError(f"Unknown reference descriptor mode {m!r}")
