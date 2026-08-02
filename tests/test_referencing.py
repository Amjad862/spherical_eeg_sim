"""
Unit tests for EEG referencing (resolve_reference / apply_reference).
"""

import sys as _sys
from pathlib import Path as _Path
_repo_root = _Path(__file__).resolve().parent.parent
if str(_repo_root) not in _sys.path:
    _sys.path.insert(0, str(_repo_root))

import numpy as np
import pytest

from spherical_eeg_sim.core.electrodes import resolve_reference, apply_reference
from spherical_eeg_sim.pipeline import _combine_snr_reference


NAMES = ['Fz', 'Cz', 'Pz', 'Oz']


def _eeg(n_ch=4, T=20, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((5, n_ch, T))


def test_reference_none_is_identity():
    eeg = _eeg()
    ref = resolve_reference(NAMES, 'none')
    out = apply_reference(eeg, ref)
    assert np.array_equal(out, eeg)


def test_average_reference_zero_channel_mean():
    eeg = _eeg()
    ref = resolve_reference(NAMES, 'average')
    out = apply_reference(eeg, ref)
    # Mean across channels at every (sample, time) must be ~0.
    assert np.abs(out.mean(axis=1)).max() < 1e-12


def test_single_channel_reference_zeroes_that_channel():
    eeg = _eeg()
    ref = resolve_reference(NAMES, 'cz', channel='Cz')
    out = apply_reference(eeg, ref)
    cz = NAMES.index('Cz')
    # The reference channel becomes exactly zero everywhere.
    assert np.abs(out[:, cz, :]).max() < 1e-12
    # Differences between channels are preserved (referencing is a shift).
    d_in = eeg[:, 0, :] - eeg[:, 2, :]
    d_out = out[:, 0, :] - out[:, 2, :]
    assert np.allclose(d_in, d_out)


def test_custom_weighted_reference():
    eeg = _eeg()
    w = [0.25, 0.25, 0.25, 0.25]  # equals average reference
    ref = resolve_reference(NAMES, 'custom', weights=w)
    out = apply_reference(eeg, ref)
    avg = apply_reference(eeg, resolve_reference(NAMES, 'average'))
    assert np.allclose(out, avg)


def test_custom_weights_normalized():
    # Unnormalized weights are normalized to sum 1.
    ref = resolve_reference(NAMES, 'custom', weights=[2.0, 2.0, 2.0, 2.0])
    assert np.allclose(ref['w'].sum(), 1.0)


def test_unknown_channel_raises():
    with pytest.raises(ValueError):
        resolve_reference(NAMES, 'cz', channel='T7')


def test_weights_wrong_length_raises():
    with pytest.raises(ValueError):
        resolve_reference(NAMES, 'custom', weights=[0.5, 0.5])


def test_referencing_is_linear_over_components():
    # ref(signal + noise) == ref(signal) + ref(noise) for every mode.
    sig = _eeg(seed=1)
    noise = _eeg(seed=2)
    for mode, kw in [('average', {}), ('cz', {'channel': 'Cz'}),
                     ('custom', {'weights': [0.1, 0.2, 0.3, 0.4]})]:
        ref = resolve_reference(NAMES, mode, **kw)
        lhs = apply_reference(sig + noise, ref)
        rhs = apply_reference(sig, ref) + apply_reference(noise, ref)
        assert np.allclose(lhs, rhs), mode


def test_snr_scaling_rejects_zero_or_nonfinite_physical_inputs():
    signal = np.zeros((1, 4, 20))
    noise = _eeg(seed=3)[:1]
    reference = resolve_reference(NAMES, 'average')
    with pytest.raises(ValueError, match='clean signal has zero RMS'):
        _combine_snr_reference(
            signal, noise, np.array([10.0]), reference, 'referenced',
            'snr_target', 1.0,
        )

    signal = _eeg(seed=4)[:1]
    noise = np.zeros_like(signal)
    with pytest.raises(ValueError, match='noise has zero RMS'):
        _combine_snr_reference(
            signal, noise, np.array([10.0]), reference, 'referenced',
            'snr_target', 1.0,
        )

    signal[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match='finite arrays'):
        _combine_snr_reference(
            signal, _eeg(seed=5)[:1], np.array([10.0]), reference,
            'referenced', 'snr_target', 1.0,
        )


if __name__ == '__main__':
    test_reference_none_is_identity()
    test_average_reference_zero_channel_mean()
    test_single_channel_reference_zeroes_that_channel()
    test_custom_weighted_reference()
    test_custom_weights_normalized()
    test_unknown_channel_raises()
    test_weights_wrong_length_raises()
    test_referencing_is_linear_over_components()
    print("All referencing tests passed.")
