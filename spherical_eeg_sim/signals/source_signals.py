"""
Source time-course generation.

Each source's time course is a sum of n_components band-selected sinusoids
with random frequencies, phases, and (optionally) per-component amplitudes.

The default band boundaries follow established EEG rhythm categories
(Buzsaki 2006; Babiloni et al. 2020). The categorical sampling weights are a
declared alpha-enriched benchmark distribution, not empirical proportions of
recorded EEG power.

Setting n_components=1 reduces to a single-sinusoid time course, as used in
many earlier ESI generators.
"""

from __future__ import annotations
import numbers
from typing import Dict, List, Optional, Sequence
import numpy as np


def _normalize_probabilities(prob_dict: Dict[str, float]) -> Dict[str, float]:
    if not prob_dict:
        raise ValueError("Band probabilities must not be empty")
    values: Dict[str, float] = {}
    for name, value in prob_dict.items():
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise ValueError(f"Band probability {name!r} must be a finite number")
        numeric = float(value)
        if not np.isfinite(numeric) or numeric < 0.0:
            raise ValueError(
                f"Band probability {name!r} must be finite and non-negative"
            )
        values[name] = numeric
    total = sum(values.values())
    if total <= 0:
        raise ValueError("Band probabilities must sum to a positive value")
    return {k: v / total for k, v in values.items()}


def _resolve_band_order(
    bands_hz: Dict[str, List[float]],
    band_probabilities: Dict[str, float],
    band_order: Optional[Sequence[str]],
) -> List[str]:
    """Return and validate the semantic order used by ``band_idx``."""
    names = list(bands_hz.keys()) if band_order is None else list(band_order)
    if not names or len(names) != len(set(names)):
        raise ValueError("band_order must contain unique band names")
    expected = set(bands_hz.keys())
    if set(band_probabilities.keys()) != expected or set(names) != expected:
        raise ValueError(
            "bands_hz, band_probabilities, and band_order must contain the "
            "same band names"
        )
    for name in names:
        bounds = bands_hz[name]
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError(f"Band {name!r} must be a two-value [low, high] pair")
        low, high = bounds
        if (isinstance(low, bool) or isinstance(high, bool)
                or not isinstance(low, numbers.Real)
                or not isinstance(high, numbers.Real)):
            raise ValueError(f"Band {name!r} bounds must be finite numbers")
        low_f, high_f = float(low), float(high)
        if (not np.isfinite(low_f) or not np.isfinite(high_f)
                or low_f < 0.0 or high_f <= low_f):
            raise ValueError(
                f"Band {name!r} must satisfy 0 <= low < high with finite bounds"
            )
    return names


def _validate_band_upper_edge(name: str, high_hz: float, sampling_freq_hz: float) -> None:
    nyquist = 0.5 * float(sampling_freq_hz)
    if float(high_hz) >= nyquist:
        raise ValueError(
            f"Band {name!r} high edge {float(high_hz):g} Hz reaches or exceeds "
            f"the Nyquist frequency {nyquist:g} Hz. The band must close strictly below it."
        )


def band_cycles_in_window(
    f_low_hz: float, window_samples: int, sampling_freq_hz: float
) -> float:
    """Number of cycles of a band's low edge that fit in the analysis window."""
    if (not np.isfinite(f_low_hz) or f_low_hz < 0.0
            or isinstance(window_samples, bool)
            or not isinstance(window_samples, numbers.Integral)
            or window_samples < 1
            or not np.isfinite(sampling_freq_hz)
            or sampling_freq_hz <= 0.0):
        raise ValueError(
            "f_low_hz must be finite and non-negative, window_samples must be "
            "a positive integer, and sampling_freq_hz must be finite and positive"
        )
    return float(f_low_hz) * window_samples / sampling_freq_hz


def resolve_effective_band_probabilities(
    bands_hz: Dict[str, List[float]],
    band_probabilities: Dict[str, float],
    window_samples: int,
    sampling_freq_hz: float,
    min_band_cycles: float = 1.0,
    policy: str = 'restrict',
    band_order: Optional[Sequence[str]] = None,
) -> "tuple[Dict[str, float], Dict[str, object]]":
    """Adjust band probabilities for window-vs-band resolvability.

    A band is "resolvable" if its low edge completes at least ``min_band_cycles``
    cycles inside the window (``f_low * window_samples / sampling_freq``). Bands
    below that threshold are under-resolved: within the window they are closer to
    a slowly-varying offset than a resolved oscillation.

    policy='restrict' (default): under-resolved bands get probability 0 and the
    mass is redistributed over the resolvable bands. A configuration in which
    every band is under-resolved is rejected because the requested restriction
    cannot be satisfied.

    policy='keep': probabilities are returned unchanged; the info records which
    band labels denote the dominant spectral component of the generating process
    rather than a resolved in-window oscillation.

    Returns
    -------
    (effective_probabilities, info)
        ``effective_probabilities`` is normalized to sum to 1.
        ``info`` is a JSON-friendly dict describing the decision (per-band cycle
        counts, which bands are under-resolved, which were dropped/kept, the
        policy, and a human-readable note) for the run manifest.
    """
    if policy not in ('restrict', 'keep'):
        raise ValueError("policy must be 'restrict' or 'keep'")
    if not np.isfinite(min_band_cycles) or min_band_cycles <= 0.0:
        raise ValueError("min_band_cycles must be finite and positive")
    names = _resolve_band_order(bands_hz, band_probabilities, band_order)
    cycles = {
        n: band_cycles_in_window(bands_hz[n][0], window_samples, sampling_freq_hz)
        for n in names
    }
    for name in names:
        _validate_band_upper_edge(name, bands_hz[name][1], sampling_freq_hz)
    under = [n for n in names if cycles[n] < min_band_cycles]

    base = _normalize_probabilities({n: band_probabilities[n] for n in names})

    if policy == 'restrict' and under and len(under) < len(names):
        kept = {n: base[n] for n in names if n not in under}
        # Drop under-resolved bands and renormalize the survivors.
        eff = _normalize_probabilities({n: kept.get(n, 0.0) for n in names})
        dropped = list(under)
        note = (
            f"policy=restrict: dropped under-resolved band(s) {dropped} "
            f"(< {min_band_cycles:g} cycles in a {window_samples}-sample window "
            f"at {sampling_freq_hz:g} Hz); probability redistributed over "
            f"{[n for n in names if n not in under]}."
        )
    elif policy == 'restrict' and under and len(under) == len(names):
        raise ValueError(
            "low_freq_band_policy='restrict' leaves no resolvable bands; "
            "lengthen window_samples, lower min_band_cycles, change the band "
            "bounds, or use policy='keep'"
        )
    else:  # policy == 'keep', or nothing under-resolved
        eff = base
        dropped = []
        if under:
            note = (
                f"policy=keep: under-resolved band(s) {under} retained; their "
                f"labels denote the DOMINANT SPECTRAL COMPONENT of the generating "
                f"process, not a resolved in-window oscillation."
            )
        else:
            note = "All bands resolvable in the configured window."

    info = {
        'policy': policy,
        'min_band_cycles': float(min_band_cycles),
        'window_samples': int(window_samples),
        'sampling_freq_hz': float(sampling_freq_hz),
        'band_low_edge_cycles_in_window': {n: float(cycles[n]) for n in names},
        'under_resolved_bands': list(under),
        'dropped_bands': dropped,
        'kept_bands': [n for n in names if eff[n] > 0.0],
        'effective_band_probabilities': {n: float(eff[n]) for n in names},
        'note': note,
    }
    return eff, info


def generate_signal_batch(
    n_samples: int,
    window_samples: int,
    sampling_freq_hz: float,
    bands_hz: Dict[str, List[float]],
    band_probabilities: Dict[str, float],
    n_components_min: int = 1,
    n_components_max: int = 3,
    component_amplitude_jitter: bool = True,
    component_dirichlet_concentration: float = 1.0,
    rng: Optional[np.random.Generator] = None,
    zero_mean: bool = False,
    band_order: Optional[Sequence[str]] = None,
) -> Dict[str, np.ndarray]:
    """Generate per-sample source time courses (mixture of sinusoids).

    Each sample is a 1-D time series of length window_samples. The signal
    is normalized so its standard deviation is exactly 1.0 -- subsequent
    SNR scaling and dipole-magnitude scaling can then be applied without
    needing to track per-sample variances.

    Parameters
    ----------
    n_samples : int
        Number of independent time courses to generate.
    window_samples : int
        Number of time samples per window.
    sampling_freq_hz : float
        Sampling rate in Hz.
    bands_hz : dict {name: [low, high]}
        Frequency bands. Keys must match band_probabilities.
    band_probabilities : dict {name: probability}
        Sampling probability for each band. Will be normalized to sum to 1.
    band_order : sequence of str, optional
        Semantic order of ``band_idx``. If omitted, ``bands_hz`` insertion
        order is used.
    n_components_min, n_components_max : int
        Number of sinusoidal components per sample is drawn uniformly from
        this inclusive range.
    component_amplitude_jitter : bool
        If True, components have random relative amplitudes (Dirichlet);
        otherwise all components have equal amplitude.
    component_dirichlet_concentration : float
        Positive symmetric Dirichlet concentration used when jitter is enabled.
    rng : numpy Generator

    Returns
    -------
    dict with keys:
        'signal' : ndarray (n_samples, window_samples)
            Standard-normalized source time courses.
        'band'   : ndarray (n_samples,)
            Band index per sample. Index follows ``band_order`` when supplied;
            otherwise it follows ``bands_hz`` insertion order.
        'band_names' : list of str
            Band names in the same order used for indexing.
        'frequencies_hz' : ndarray (n_samples, n_components_max)
            Component frequencies in Hz. Unused components are NaN.
        'phases_rad' : ndarray (n_samples, n_components_max)
            Component initial phases in radians. Unused components are NaN.
        'component_amplitudes' : ndarray (n_samples, n_components_max)
            Relative sinusoid amplitudes before temporal centering and
            standard-deviation normalization. Unused components are NaN.
        'n_components' : ndarray (n_samples,)
            Number of active sinusoidal components per sample.
    """
    if rng is None:
        rng = np.random.default_rng()

    for name, value in (
        ('n_samples', n_samples),
        ('window_samples', window_samples),
        ('n_components_min', n_components_min),
        ('n_components_max', n_components_max),
    ):
        if (isinstance(value, bool)
                or not isinstance(value, numbers.Integral)
                or value < 1):
            raise ValueError(f"{name} must be a positive integer")
    if n_components_max < n_components_min:
        raise ValueError("n_components_max must be >= n_components_min")
    if (isinstance(sampling_freq_hz, bool)
            or not isinstance(sampling_freq_hz, numbers.Real)
            or not np.isfinite(sampling_freq_hz)
            or sampling_freq_hz <= 0.0):
        raise ValueError("sampling_freq_hz must be a finite positive number")
    if (isinstance(component_dirichlet_concentration, bool)
            or not isinstance(component_dirichlet_concentration, numbers.Real)
            or not np.isfinite(component_dirichlet_concentration)
            or component_dirichlet_concentration <= 0.0):
        raise ValueError(
            "component_dirichlet_concentration must be finite and positive"
        )

    band_names = _resolve_band_order(bands_hz, band_probabilities, band_order)
    for name in band_names:
        _validate_band_upper_edge(name, bands_hz[name][1], sampling_freq_hz)
    n_bands = len(band_names)

    probs = _normalize_probabilities({k: band_probabilities[k] for k in band_names})
    p_arr = np.array([probs[k] for k in band_names])

    band_idx = rng.choice(n_bands, size=n_samples, p=p_arr)

    if n_components_min == n_components_max:
        n_comps = np.full(n_samples, n_components_min, dtype=np.int32)
    else:
        n_comps = rng.integers(n_components_min, n_components_max + 1, size=n_samples)

    t = np.arange(window_samples, dtype=np.float64) / sampling_freq_hz  # (T,)

    signal = np.zeros((n_samples, window_samples), dtype=np.float64)
    frequencies_hz = np.full((n_samples, n_components_max), np.nan, dtype=np.float64)
    phases_rad = np.full((n_samples, n_components_max), np.nan, dtype=np.float64)
    component_amplitudes = np.full(
        (n_samples, n_components_max), np.nan, dtype=np.float64
    )

    for b in range(n_bands):
        mask = (band_idx == b)
        if not mask.any():
            continue
        idx = np.where(mask)[0]
        f_low, f_high = bands_hz[band_names[b]]
        n_in = len(idx)
        freqs_block = rng.uniform(f_low, f_high, size=(n_in, n_components_max))
        phases_block = rng.uniform(0.0, 2 * np.pi, size=(n_in, n_components_max))

        if component_amplitude_jitter:
            amps_block = rng.dirichlet(
                np.full(n_components_max, component_dirichlet_concentration),
                size=n_in,
            )
        else:
            amps_block = None

        for k, sample_idx in enumerate(idx):
            n_c = int(n_comps[sample_idx])
            f_k = freqs_block[k, :n_c]
            ph_k = phases_block[k, :n_c]
            if component_amplitude_jitter:
                a_k = amps_block[k, :n_c]
                a_k = a_k / a_k.sum()
            else:
                a_k = np.full(n_c, 1.0 / n_c)

            # x(t) = sum_j a_j * sin(2*pi*f_j*t + phi_j)
            phase_grid = 2 * np.pi * f_k[:, None] * t[None, :] + ph_k[:, None]
            comp = a_k[:, None] * np.sin(phase_grid)
            signal[sample_idx] = comp.sum(axis=0)

            frequencies_hz[sample_idx, :n_c] = f_k
            phases_rad[sample_idx, :n_c] = ph_k
            component_amplitudes[sample_idx, :n_c] = a_k

    if zero_mean:
        # Remove the per-sample temporal mean so the window is a zero-mean
        # fluctuation, consistent with analyzing EEG after high-pass filtering.
        # This matters most for the lowest bands, whose short-window segments carry a
        # large DC term that survives unit-variance scaling.
        signal = signal - signal.mean(axis=1, keepdims=True)

    sd = signal.std(axis=1, keepdims=True)
    sd = np.where(sd > 0, sd, 1.0)
    signal = signal / sd

    return {
        'signal': signal,
        'band': band_idx,
        'band_names': band_names,
        'frequencies_hz': frequencies_hz,
        'phases_rad': phases_rad,
        'component_amplitudes': component_amplitudes,
        'n_components': n_comps,
    }
