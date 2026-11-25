"""
Unit tests for signal generation and noise components.
"""

import sys as _sys
from pathlib import Path as _Path
_repo_root = _Path(__file__).resolve().parent.parent
if str(_repo_root) not in _sys.path:
    _sys.path.insert(0, str(_repo_root))

import numpy as np
import pytest

from spherical_eeg_sim.signals.source_signals import generate_signal_batch
from spherical_eeg_sim.signals.noise import (
    generate_physiological_background, generate_pink_noise,
    generate_white_noise, mix_noise_components, sample_snr_db,
)


DEFAULT_BANDS = {
    'delta': [0.5, 4.0], 'theta': [4.0, 8.0], 'alpha': [8.0, 13.0],
    'beta': [14.0, 30.0], 'gamma': [30.0, 45.0],
}
DEFAULT_PROBS = {
    'delta': 0.20, 'theta': 0.15, 'alpha': 0.35,
    'beta': 0.20, 'gamma': 0.10,
}


def test_signal_batch_shape_and_normalization():
    rng = np.random.default_rng(0)
    out = generate_signal_batch(
        n_samples=200, window_samples=100, sampling_freq_hz=500.0,
        bands_hz=DEFAULT_BANDS, band_probabilities=DEFAULT_PROBS,
        n_components_min=3, n_components_max=3, rng=rng,
    )
    assert out['signal'].shape == (200, 100)
    # Each sample standard-normalized
    sd = out['signal'].std(axis=1)
    assert np.allclose(sd, 1.0, atol=1e-10)


def test_signal_band_probabilities_match():
    rng = np.random.default_rng(0)
    out = generate_signal_batch(
        n_samples=10_000, window_samples=100, sampling_freq_hz=500.0,
        bands_hz=DEFAULT_BANDS, band_probabilities=DEFAULT_PROBS,
        n_components_min=3, n_components_max=3, rng=rng,
    )
    counts = np.bincount(out['band'], minlength=5)
    fractions = counts / counts.sum()
    expected = np.array([0.20, 0.15, 0.35, 0.20, 0.10])
    # Allow 2% tolerance
    assert np.max(np.abs(fractions - expected)) < 0.02, fractions


def test_signal_n_components_variable():
    rng = np.random.default_rng(0)
    out = generate_signal_batch(
        n_samples=5000, window_samples=100, sampling_freq_hz=500.0,
        bands_hz=DEFAULT_BANDS, band_probabilities=DEFAULT_PROBS,
        n_components_min=1, n_components_max=5, rng=rng,
    )
    assert out['n_components'].min() == 1
    assert out['n_components'].max() == 5


def test_signal_frequency_in_band():
    rng = np.random.default_rng(0)
    out = generate_signal_batch(
        n_samples=5000, window_samples=100, sampling_freq_hz=500.0,
        bands_hz=DEFAULT_BANDS, band_probabilities=DEFAULT_PROBS,
        n_components_min=2, n_components_max=2, rng=rng,
    )
    band_names = out['band_names']
    for i in range(out['signal'].shape[0]):
        b = out['band'][i]
        f_low, f_high = DEFAULT_BANDS[band_names[b]]
        f = out['frequencies_hz'][i]
        f_used = f[~np.isnan(f)]
        assert (f_used >= f_low).all() and (f_used <= f_high).all()


def test_explicit_band_order_is_mapping_order_independent():
    order = list(DEFAULT_BANDS)
    reversed_bands = dict(reversed(list(DEFAULT_BANDS.items())))
    reversed_probs = dict(reversed(list(DEFAULT_PROBS.items())))
    kwargs = dict(
        n_samples=200, window_samples=100, sampling_freq_hz=500.0,
        n_components_min=3, n_components_max=3, band_order=order,
    )
    a = generate_signal_batch(
        bands_hz=DEFAULT_BANDS, band_probabilities=DEFAULT_PROBS,
        rng=np.random.default_rng(42), **kwargs,
    )
    b = generate_signal_batch(
        bands_hz=reversed_bands, band_probabilities=reversed_probs,
        rng=np.random.default_rng(42), **kwargs,
    )
    assert a['band_names'] == b['band_names'] == order
    for key in ('signal', 'band', 'frequencies_hz', 'phases_rad',
                'component_amplitudes', 'n_components'):
        assert np.array_equal(a[key], b[key], equal_nan=True)


@pytest.mark.parametrize('probs', [
    {**DEFAULT_PROBS, 'delta': -0.1},
    {**DEFAULT_PROBS, 'delta': np.nan},
    {**DEFAULT_PROBS, 'delta': np.inf},
])
def test_signal_rejects_invalid_direct_probabilities(probs):
    with pytest.raises(ValueError, match='probability'):
        generate_signal_batch(
            n_samples=2, window_samples=100, sampling_freq_hz=500.0,
            bands_hz=DEFAULT_BANDS, band_probabilities=probs,
            rng=np.random.default_rng(0),
        )


def test_signal_rejects_invalid_direct_dimensions_and_nyquist():
    kwargs = dict(
        n_samples=2, window_samples=100, sampling_freq_hz=500.0,
        bands_hz=DEFAULT_BANDS, band_probabilities=DEFAULT_PROBS,
        rng=np.random.default_rng(0),
    )
    with pytest.raises(ValueError, match='n_samples'):
        generate_signal_batch(**{**kwargs, 'n_samples': 0})
    with pytest.raises(ValueError, match='n_components_max'):
        generate_signal_batch(
            **kwargs, n_components_min=3, n_components_max=2,
        )
    with pytest.raises(ValueError, match='component_dirichlet_concentration'):
        generate_signal_batch(
            **kwargs, component_dirichlet_concentration=0.0,
        )
    for high in (250.0, 300.0):
        invalid_bands = {**DEFAULT_BANDS, 'gamma': [30.0, high]}
        with pytest.raises(ValueError, match='Nyquist'):
            generate_signal_batch(**{**kwargs, 'bands_hz': invalid_bands})
        with pytest.raises(ValueError, match='Nyquist'):
            resolve_effective_band_probabilities(invalid_bands, DEFAULT_PROBS, 100, 500.0)


@pytest.mark.parametrize('sampling_rate', [500.0, 501.0, 255.0])
def test_all_signal_entry_points_enforce_strict_nyquist(sampling_rate):
    from spherical_eeg_sim import Config

    boundary = sampling_rate / 2
    for high in (np.nextafter(boundary, -np.inf), boundary, np.nextafter(boundary, np.inf)):
        bands = {'wide': [30.0, high]}
        cfg = Config()
        cfg.signal.sampling_freq_hz = sampling_rate
        cfg.signal.bands_hz = bands
        cfg.signal.band_probabilities = {'wide': 1.0}
        cfg.signal.band_order = ['wide']

        def generate():
            generate_signal_batch(
                n_samples=2, window_samples=100, sampling_freq_hz=sampling_rate,
                bands_hz=bands, band_probabilities={'wide': 1.0}, rng=np.random.default_rng(0),
            )

        def resolve():
            resolve_effective_band_probabilities(bands, {'wide': 1.0}, 100, sampling_rate)

        for call in (cfg.validate, generate, resolve):
            if high < boundary:
                call()
            else:
                with pytest.raises(ValueError, match='Nyquist'):
                    call()


def test_pink_noise_normalized():
    rng = np.random.default_rng(0)
    n = generate_pink_noise(20, 32, 100, alpha=1.0, rng=rng)
    assert n.shape == (20, 32, 100)
    sd = n.std(axis=-1)
    assert np.allclose(sd, 1.0, atol=1e-10)


def test_white_noise_flat_spectrum():
    rng = np.random.default_rng(0)
    n = generate_white_noise(1, 1, 4000, rng)[0, 0]
    spec = np.abs(np.fft.rfft(n))**2
    # White noise: spectrum should be approximately flat (slope ~0)
    f = np.fft.rfftfreq(4000, d=1/500)
    mask = f > 5
    log_f = np.log(f[mask])
    log_p = np.log(spec[mask] + 1e-12)
    slope = np.polyfit(log_f, log_p, 1)[0]
    assert abs(slope) < 0.3, f"white slope = {slope:.2f}"


def test_sample_snr_db_in_range():
    rng = np.random.default_rng(0)
    s = sample_snr_db(10_000, (5.0, 25.0), rng)
    assert s.min() >= 5.0
    assert s.max() <= 25.0
    assert abs(s.mean() - 15.0) < 0.5  # uniform mean


def test_noise_generators_reject_invalid_direct_inputs():
    with pytest.raises(ValueError, match='n_timepoints'):
        generate_pink_noise(1, 2, 1)
    with pytest.raises(ValueError, match='alpha'):
        generate_pink_noise(1, 2, 10, alpha=np.nan)
    with pytest.raises(ValueError, match='finite'):
        generate_physiological_background(
            1, 2, 10, np.array([[1.0, np.nan]]), 1,
        )
    with pytest.raises(ValueError, match='positive integer'):
        generate_physiological_background(
            1, 2, 10, np.ones((2, 2)), 0,
        )
    with pytest.raises(ValueError, match='amplitude_concentration'):
        generate_physiological_background(
            1, 2, 10, np.ones((2, 2)), 1, amplitude_concentration=0.0,
        )
    with pytest.raises(ValueError, match='same 3-D shape'):
        mix_noise_components(
            np.ones((1, 2, 10)), np.ones((1, 3, 10)), np.ones((1, 2, 10)),
            0.5, 0.3, 0.2,
        )
    with pytest.raises(ValueError, match='low < high'):
        sample_snr_db(2, (5.0, 5.0))


if __name__ == '__main__':
    test_signal_batch_shape_and_normalization()
    test_signal_band_probabilities_match()
    test_signal_n_components_variable()
    test_signal_frequency_in_band()
    test_pink_noise_normalized()
    test_white_noise_flat_spectrum()
    test_sample_snr_db_in_range()
    print("All signal/noise tests passed!")


# ---------------------------------------------------------------------------
# Window-vs-band resolvability policy and zero-mean option
# ---------------------------------------------------------------------------

from spherical_eeg_sim.signals.source_signals import (
    resolve_effective_band_probabilities,
    band_cycles_in_window,
)


def test_band_cycles_in_window():
    # delta low edge 0.5 Hz over 100 samples @ 500 Hz -> 0.1 cycles
    assert abs(band_cycles_in_window(0.5, 100, 500.0) - 0.1) < 1e-12


def test_restrict_drops_underresolved_bands():
    eff, info = resolve_effective_band_probabilities(
        DEFAULT_BANDS, DEFAULT_PROBS, window_samples=100,
        sampling_freq_hz=500.0, min_band_cycles=1.0, policy='restrict',
    )
    # delta (0.1) and theta (0.8) are < 1 cycle and must be dropped.
    assert eff['delta'] == 0.0 and eff['theta'] == 0.0
    assert 'delta' in info['dropped_bands'] and 'theta' in info['dropped_bands']
    # Survivors renormalize to 1.
    assert abs(sum(eff.values()) - 1.0) < 1e-12
    # alpha keeps its relative share among survivors.
    assert eff['alpha'] > eff['beta'] > eff['gamma'] > 0.0


def test_keep_retains_all_bands_with_note():
    eff, info = resolve_effective_band_probabilities(
        DEFAULT_BANDS, DEFAULT_PROBS, window_samples=100,
        sampling_freq_hz=500.0, min_band_cycles=1.0, policy='keep',
    )
    assert eff['delta'] > 0.0 and eff['theta'] > 0.0
    assert info['dropped_bands'] == []
    assert 'delta' in info['under_resolved_bands']
    assert 'dominant spectral component' in info['note'].lower()


def test_band_resolution_rejects_invalid_direct_policy():
    with pytest.raises(ValueError, match='policy'):
        resolve_effective_band_probabilities(
            DEFAULT_BANDS, DEFAULT_PROBS, 100, 500.0, policy='silent_drop',
        )


def test_restrict_long_window_keeps_everything():
    # A 5 s window resolves all bands; nothing is dropped.
    eff, info = resolve_effective_band_probabilities(
        DEFAULT_BANDS, DEFAULT_PROBS, window_samples=2500,
        sampling_freq_hz=500.0, min_band_cycles=1.0, policy='restrict',
    )
    assert info['dropped_bands'] == []
    assert all(v > 0 for v in eff.values())


def test_restrict_all_underresolved_is_rejected():
    with pytest.raises(ValueError, match='no resolvable bands'):
        resolve_effective_band_probabilities(
            DEFAULT_BANDS, DEFAULT_PROBS, window_samples=2,
            sampling_freq_hz=500.0, min_band_cycles=1.0, policy='restrict',
        )


def test_restrict_empirical_no_delta():
    eff, _ = resolve_effective_band_probabilities(
        DEFAULT_BANDS, DEFAULT_PROBS, 100, 500.0, 1.0, 'restrict')
    out = generate_signal_batch(
        n_samples=8000, window_samples=100, sampling_freq_hz=500.0,
        bands_hz=DEFAULT_BANDS, band_probabilities=eff,
        rng=np.random.default_rng(0),
    )
    counts = np.bincount(out['band'], minlength=5)
    assert counts[0] == 0 and counts[1] == 0  # no delta, no theta


def test_zero_mean_removes_dc():
    out = generate_signal_batch(
        n_samples=500, window_samples=100, sampling_freq_hz=500.0,
        bands_hz=DEFAULT_BANDS, band_probabilities=DEFAULT_PROBS,
        rng=np.random.default_rng(0), zero_mean=True,
    )
    assert np.abs(out['signal'].mean(axis=1)).max() < 1e-9
    # Unit variance is preserved per sample.
    assert np.allclose(out['signal'].std(axis=1), 1.0, atol=1e-6)


def test_zero_mean_false_preserves_dc():
    # With delta-heavy short windows and zero_mean=False, some samples retain a
    # non-negligible temporal mean.
    probs = {'delta': 1.0, 'theta': 0.0, 'alpha': 0.0, 'beta': 0.0, 'gamma': 0.0}
    out = generate_signal_batch(
        n_samples=500, window_samples=100, sampling_freq_hz=500.0,
        bands_hz=DEFAULT_BANDS, band_probabilities=probs,
        rng=np.random.default_rng(0), zero_mean=False,
    )
    assert np.abs(out['signal'].mean(axis=1)).max() > 0.1
