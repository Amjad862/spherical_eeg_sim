"""
Three-component noise model for the EEG dataset generator.

Recorded EEG contains spatially correlated and uncorrelated variability. This
benchmark mixes three declared proxies:

  1. Source-projected background -- N background dipoles at random spherical
     positions, each with a 1/f time course, forward-projected through
     the head model. Naturally spatially correlated through volume
     conduction. Reference: de Munck et al. 2002, Bijma et al. 2003.

  2. Independent pink -- per-channel 1/f noise.

  3. Independent white -- per-channel flat-spectrum noise.

The source-projected component is a controlled proxy for correlated background
activity, not a physiological recording model. The independent components are
spectral/statistical terms and are not assigned to a particular instrument
mechanism. Square roots of normalized non-negative weights multiply the three
components. The weights are intended relative variance contributions;
finite-window cross-covariances mean they are not exact realized shares. The
mix is then scaled according to the configured noise model (see pipeline): either to
a fixed absolute RMS ('absolute_floor', the default) or to a per-sample target
SNR ('snr_target').
"""

from __future__ import annotations
import numbers
from typing import Optional, Sequence
import numpy as np


def _validate_noise_shape(
    n_samples: int, n_channels: int, n_timepoints: int,
) -> None:
    """Validate public noise-generator dimensions."""
    for name, value, minimum in (
        ('n_samples', n_samples, 1),
        ('n_channels', n_channels, 1),
        ('n_timepoints', n_timepoints, 2),
    ):
        if (isinstance(value, bool)
                or not isinstance(value, numbers.Integral)
                or value < minimum):
            raise ValueError(f"{name} must be an integer >= {minimum}")


def generate_pink_noise(
    n_samples: int,
    n_channels: int,
    n_timepoints: int,
    alpha: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Generate 1/f^alpha noise via FFT spectral shaping.

    Parameters
    ----------
    n_samples, n_channels, n_timepoints : int
        Output shape is (n_samples, n_channels, n_timepoints).
    alpha : float
        Spectral exponent; alpha=1 denotes pink noise and alpha=2 brown noise.
        Exponents for a target dataset should be measured or justified
        separately (Donoghue et al. 2020).
    rng : numpy Generator

    Returns
    -------
    noise : ndarray, shape (n_samples, n_channels, n_timepoints)
        Standard-normalized noise (per-sample, per-channel std = 1).
    """
    if rng is None:
        rng = np.random.default_rng()
    _validate_noise_shape(n_samples, n_channels, n_timepoints)
    if (isinstance(alpha, bool) or not isinstance(alpha, numbers.Real)
            or not np.isfinite(alpha) or alpha < 0.0):
        raise ValueError("alpha must be a finite non-negative number")

    white = rng.standard_normal(size=(n_samples, n_channels, n_timepoints))
    spec = np.fft.rfft(white, axis=-1)
    n_freq = spec.shape[-1]

    freqs = np.arange(n_freq, dtype=np.float64)
    scale = np.ones(n_freq, dtype=np.float64)
    scale[1:] = freqs[1:] ** (-alpha / 2.0)
    scale[0] = 0.0  # remove DC

    spec_shaped = spec * scale[None, None, :]
    shaped = np.fft.irfft(spec_shaped, n=n_timepoints, axis=-1)

    sd = shaped.std(axis=-1, keepdims=True)
    sd = np.where(sd > 0, sd, 1.0)
    return shaped / sd


def generate_white_noise(
    n_samples: int,
    n_channels: int,
    n_timepoints: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Generate flat-spectrum (white) noise, standard-normalized per-channel."""
    if rng is None:
        rng = np.random.default_rng()
    _validate_noise_shape(n_samples, n_channels, n_timepoints)
    noise = rng.standard_normal(size=(n_samples, n_channels, n_timepoints))
    sd = noise.std(axis=-1, keepdims=True)
    sd = np.where(sd > 0, sd, 1.0)
    return noise / sd


def generate_physiological_background(
    n_samples: int,
    n_channels: int,
    n_timepoints: int,
    background_leadfields: np.ndarray,
    n_background_sources: int,
    amplitude_concentration: float = 1.0,
    alpha: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Generate spatially-correlated background EEG noise.

    For each sample:
      1. Draw n_background_sources random rows from the precomputed pool of
         background lead-field projections (each row is a (channels,)
         vector representing one background dipole's scalp signature).
      2. Multiply by independent 1/f^alpha time courses to get spatially
         coherent but temporally distinct noise per source.
      3. Sum over sources.
      4. Standard-normalize per channel.

    Parameters
    ----------
    n_samples, n_channels, n_timepoints : int
    background_leadfields : ndarray, shape (P_bg, N) where N = n_channels
        Pool of background dipole projections to scalp. Typically computed
        once at dataset-generation time using random brain positions and
        random orientations.
    n_background_sources : int
        Number of background dipoles per sample.
    amplitude_concentration : float
        Positive symmetric Dirichlet concentration for relative source weights.
    alpha : float
        Spectral exponent of background time courses.
    rng : numpy Generator

    Returns
    -------
    noise : ndarray, shape (n_samples, n_channels, n_timepoints)
        Per-channel-standard-normalized.
    """
    if rng is None:
        rng = np.random.default_rng()

    _validate_noise_shape(n_samples, n_channels, n_timepoints)
    if (isinstance(n_background_sources, bool)
            or not isinstance(n_background_sources, numbers.Integral)
            or n_background_sources < 1):
        raise ValueError("n_background_sources must be a positive integer")
    if (isinstance(amplitude_concentration, bool)
            or not isinstance(amplitude_concentration, numbers.Real)
            or not np.isfinite(amplitude_concentration)
            or amplitude_concentration <= 0.0):
        raise ValueError("amplitude_concentration must be finite and positive")
    leadfields = np.asarray(background_leadfields, dtype=np.float64)
    if leadfields.ndim != 2:
        raise ValueError("background_leadfields must have shape (pool, channels)")
    if not np.all(np.isfinite(leadfields)):
        raise ValueError("background_leadfields must contain only finite values")

    if leadfields.shape[1] != n_channels:
        raise ValueError(
            f"background_leadfields has {leadfields.shape[1]} "
            f"channels but n_channels={n_channels}"
        )
    P_bg = leadfields.shape[0]
    if n_background_sources > P_bg:
        raise ValueError(
            f"n_background_sources={n_background_sources} > pool size {P_bg}"
        )

    bg_indices = np.empty((n_samples, n_background_sources), dtype=np.int64)
    for i in range(n_samples):
        bg_indices[i] = rng.choice(P_bg, size=n_background_sources, replace=False)

    n_total_courses = n_samples * n_background_sources
    courses = generate_pink_noise(
        n_samples=n_total_courses,
        n_channels=1,
        n_timepoints=n_timepoints,
        alpha=alpha,
        rng=rng,
    ).squeeze(1)
    courses = courses.reshape(n_samples, n_background_sources, n_timepoints)

    amps = rng.dirichlet(
        np.full(n_background_sources, amplitude_concentration), size=n_samples,
    )
    courses = courses * amps[:, :, None]

    noise = np.zeros((n_samples, n_channels, n_timepoints), dtype=np.float64)
    for i in range(n_samples):
        L_i = leadfields[bg_indices[i]]   # (n_bg, N)
        noise[i] = L_i.T @ courses[i]

    sd = noise.std(axis=-1, keepdims=True)
    sd = np.where(sd > 0, sd, 1.0)
    return noise / sd


def mix_noise_components(
    physiological: np.ndarray,
    pink: np.ndarray,
    white: np.ndarray,
    mix_physiological: float,
    mix_pink: float,
    mix_white: float,
) -> np.ndarray:
    """Mix three noise components by per-sample variance.

    Each input is assumed standard-normalized per-channel (std = 1 along
    the time axis). Square roots of the normalized weights multiply the
    components, so the weights are intended relative variance contributions.
    Finite-window component cross-covariances prevent exact realized shares;
    the combined output is then renormalized per channel.

    Parameters
    ----------
    physiological, pink, white : ndarray, shape (n_samples, n_channels, T)
    mix_physiological, mix_pink, mix_white : float
        Each weight must be non-negative. They will be normalized to sum to 1.

    Returns
    -------
    noise : ndarray, shape (n_samples, n_channels, T)
        Mixed noise, per-channel standard-normalized.
    """
    for label, w in (
        ('mix_physiological', mix_physiological),
        ('mix_pink', mix_pink),
        ('mix_white', mix_white),
    ):
        if (isinstance(w, bool) or not isinstance(w, numbers.Real)
                or not np.isfinite(w) or w < 0):
            raise ValueError(f"{label} must be finite and >= 0 (got {w})")
    arrays = [np.asarray(x, dtype=np.float64) for x in (physiological, pink, white)]
    if any(x.ndim != 3 for x in arrays) or len({x.shape for x in arrays}) != 1:
        raise ValueError("All noise components must have the same 3-D shape")
    if not all(np.all(np.isfinite(x)) for x in arrays):
        raise ValueError("Noise components must contain only finite values")
    physiological, pink, white = arrays
    total = mix_physiological + mix_pink + mix_white
    if total <= 0:
        raise ValueError("Noise mix proportions must sum to a positive value")
    a_phys = np.sqrt(mix_physiological / total)
    a_pink = np.sqrt(mix_pink / total)
    a_white = np.sqrt(mix_white / total)

    mixed = a_phys * physiological + a_pink * pink + a_white * white

    sd = mixed.std(axis=-1, keepdims=True)
    sd = np.where(sd > 0, sd, 1.0)
    return mixed / sd


def sample_snr_db(
    n_samples: int,
    snr_db_range: Sequence[float],
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Draw per-sample SNR values uniformly from snr_db_range."""
    if rng is None:
        rng = np.random.default_rng()
    if (isinstance(n_samples, bool) or not isinstance(n_samples, numbers.Integral)
            or n_samples < 1):
        raise ValueError("n_samples must be a positive integer")
    if not isinstance(snr_db_range, (list, tuple)) or len(snr_db_range) != 2:
        raise ValueError("snr_db_range must be a two-value (low, high) pair")
    low, high = snr_db_range
    if (isinstance(low, bool) or isinstance(high, bool)
            or not isinstance(low, numbers.Real)
            or not isinstance(high, numbers.Real)
            or not np.isfinite(low) or not np.isfinite(high)
            or low >= high):
        raise ValueError("snr_db_range must contain finite values with low < high")
    return rng.uniform(float(low), float(high), size=n_samples)
