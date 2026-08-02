"""Validation tests for the bundled electrode montages and radius handling."""

import sys as _sys
from pathlib import Path as _Path
_repo_root = _Path(__file__).resolve().parent.parent
if str(_repo_root) not in _sys.path:
    _sys.path.insert(0, str(_repo_root))

import csv
import tempfile
from importlib.resources import files

import numpy as np
import pytest

from spherical_eeg_sim.core.electrodes import (
    load_electrodes,
    generate_standard_montage,
)


def _bundled_csvs():
    d = files('spherical_eeg_sim.montages')
    return sorted(f.name for f in d.iterdir() if f.name.endswith('.csv'))


def _mirror_name(n):
    import re
    core, suf = (n[:-1], n[-1]) if n and n[-1] == 'h' else (n, '')
    m = re.match(r'^(.*?)(\d+)$', core)
    if not m:
        return None
    stem, num = m.group(1), int(m.group(2))
    return f'{stem}{num + 1}{suf}' if num % 2 == 1 else None


R = 0.092
STANDARD_FAMILY = {
    'standard_1020_19.csv': 19, 'standard_1020_20.csv': 20,
    'standard_1010_32.csv': 32, 'standard_1010_61.csv': 61,
    'standard_1010_69.csv': 69, 'standard_1005_336.csv': 336,
}
VENDOR_FAMILY = {
    'biosemi_16_spherical.csv': 16,
    'biosemi_32_spherical.csv': 32,
    'biosemi_64_spherical.csv': 64,
    'biosemi_128_spherical.csv': 128,
    'biosemi_256_spherical.csv': 256,
    # The HydroCel 32 template has 32 numbered sensors plus Cz, the recording
    # reference supplied in the manufacturer template.
    'gsn_hydrocel_32_spherical.csv': 33,
    'gsn_hydrocel_64_spherical.csv': 64,
    'gsn_hydrocel_128_spherical.csv': 128,
    'gsn_hydrocel_256_spherical.csv': 256,
    'gsn_hydrocel_257_spherical.csv': 257,
}
SYMMETRIC = list(STANDARD_FAMILY) + [
    'biosemi_16_spherical.csv', 'biosemi_32_spherical.csv', 'biosemi_64_spherical.csv',
]


def test_bundled_set_present():
    csvs = set(_bundled_csvs())
    # standard family + both vendor systems are all shipped
    assert STANDARD_FAMILY.keys() <= csvs
    for n in [16, 32, 64, 128, 256]:
        assert f'biosemi_{n}_spherical.csv' in csvs
    for n in [32, 64, 128, 256, 257]:
        assert f'gsn_hydrocel_{n}_spherical.csv' in csvs


def test_committed_montages_match_generator_bytes(tmp_path, monkeypatch):
    """The deterministic generator must reproduce every committed CSV byte."""
    from spherical_eeg_sim.montages import _generate

    monkeypatch.setattr(_generate, '_HERE', tmp_path)
    assert _generate.main() == 0
    bundled = files('spherical_eeg_sim.montages')
    for name in _bundled_csvs():
        assert (tmp_path / name).read_bytes() == bundled.joinpath(name).read_bytes()


@pytest.mark.parametrize('name', _bundled_csvs())
def test_every_montage_loads_and_is_on_sphere(name):
    """Loading validates no-collision + radii range; here we also check the
    positions lie exactly on a single sphere and Cartesian<->spherical agree."""
    arr = load_electrodes(name)
    radii = np.linalg.norm(arr.cartesian, axis=1)
    assert np.max(np.abs(radii - arr.radius_m)) < 1e-6  # single sphere
    r, th, ph = arr.spherical[:, 0], arr.spherical[:, 1], arr.spherical[:, 2]
    expected = np.column_stack([
        r * np.sin(th) * np.cos(ph), r * np.sin(th) * np.sin(ph), r * np.cos(th)])
    assert np.max(np.abs(arr.cartesian - expected)) < 1e-9  # xyz consistent


@pytest.mark.parametrize('name,count', STANDARD_FAMILY.items())
def test_standard_family_counts_and_vertex(name, count):
    arr = load_electrodes(name)
    assert arr.n_electrodes == count
    # Cz is present and sits exactly at the vertex (theta = 0)
    cz = arr.names.index('Cz')
    assert np.degrees(arr.spherical[cz, 1]) < 1e-6


@pytest.mark.parametrize('name,count', VENDOR_FAMILY.items())
def test_vendor_family_channel_counts(name, count):
    arr = load_electrodes(name)
    assert arr.n_electrodes == count
    if name == 'biosemi_64_spherical.csv':
        assert 'Iz' in arr.names


@pytest.mark.parametrize('name', SYMMETRIC)
def test_left_right_symmetry(name):
    arr = load_electrodes(name)
    pos = {n: arr.cartesian[i] for i, n in enumerate(arr.names)}
    checked = 0
    for n, p in pos.items():
        partner = _mirror_name(n)
        if partner in pos:
            q = pos[partner]
            assert abs(p[0] + q[0]) < 1e-9  # mirrored in x
            assert abs(p[1] - q[1]) < 1e-9  # same y
            assert abs(p[2] - q[2]) < 1e-9  # same z
            checked += 1
        elif n.endswith('z'):
            assert abs(p[0]) < 1e-9         # midline on x = 0
    assert checked > 0


@pytest.mark.parametrize('name', _bundled_csvs())
def test_no_coincident_electrodes(name):
    arr = load_electrodes(name)
    D = np.linalg.norm(arr.cartesian[:, None, :] - arr.cartesian[None, :, :], axis=-1)
    np.fill_diagonal(D, np.inf)
    assert D.min() > 1e-3  # at least 1 mm apart


def test_radius_independence():
    """A montage stored at 0.092 m is placed correctly at any scalp radius:
    angles preserved, positions scaled by the radius ratio."""
    a = load_electrodes('standard_1010_69.csv', target_radius_m=0.092)
    b = load_electrodes('standard_1010_69.csv', target_radius_m=0.105)
    assert abs(a.radius_m - 0.092) < 1e-12
    assert abs(b.radius_m - 0.105) < 1e-12
    assert np.allclose(a.spherical[:, 1:], b.spherical[:, 1:])          # theta, phi
    assert np.allclose(b.cartesian, a.cartesian * (0.105 / 0.092))       # scaled


def test_path_object_can_resolve_bundled_montage():
    assert load_electrodes(_Path('standard_1010_69.csv')).n_electrodes == 69


def test_missing_explicit_path_does_not_fall_back_to_bundled_basename(tmp_path):
    missing = tmp_path / 'standard_1010_69.csv'
    with pytest.raises(FileNotFoundError) as exc_info:
        load_electrodes(missing)
    assert str(missing) in str(exc_info.value)


def test_tolerated_csv_roundoff_is_canonicalized_from_cartesian(tmp_path):
    source = load_electrodes('standard_1020_20.csv')
    path = tmp_path / 'rounded_angles.csv'
    with path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Name', 'x', 'y', 'z', 'R', 'theta', 'phi'])
        for i, name in enumerate(source.names):
            r, theta, phi = source.spherical[i]
            writer.writerow([
                name, *source.cartesian[i], r,
                np.degrees(theta) + 0.1, np.degrees(phi),
            ])
    loaded = load_electrodes(path)
    r, theta, phi = loaded.spherical.T
    reconstructed = np.column_stack([
        r * np.sin(theta) * np.cos(phi),
        r * np.sin(theta) * np.sin(phi),
        r * np.cos(theta),
    ])
    assert np.allclose(reconstructed, loaded.cartesian, rtol=0.0, atol=1e-12)


def test_target_radius_is_not_limited_to_adult_head_scale():
    arr = load_electrodes('standard_1020_20.csv', target_radius_m=0.5)
    assert np.allclose(np.linalg.norm(arr.cartesian, axis=1), 0.5)


def test_invalid_electrode_values_rejected(tmp_path):
    p = tmp_path / 'nonfinite.csv'
    p.write_text(
        'Name,x,y,z,R,theta,phi\n'
        'Cz,0,0,nan,0.092,0,0\n'
        'Fz,0,0.092,0,0.092,90,90\n'
        'Pz,0,-0.092,0,0.092,90,270\n'
        'T7,-0.092,0,0,0.092,90,180\n'
    )
    with pytest.raises(ValueError, match='non-finite'):
        load_electrodes(p)
    with pytest.raises(ValueError, match='target_radius_m'):
        load_electrodes('standard_1020_20.csv', target_radius_m=np.nan)


def test_unit_mismatch_rejected():
    """Positions supplied in millimetres are caught by the sanity check."""
    p = tempfile.mktemp(suffix='.csv')
    rad_mm = 92.0
    specs = [('Cz', 0.0, 0.0), ('Fz', 54.0, 90.0), ('Pz', 54.0, 270.0),
             ('T7', 90.0, 180.0), ('T8', 90.0, 0.0)]
    with open(p, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Name', 'x', 'y', 'z', 'R', 'theta', 'phi'])
        for name, th, ph in specs:
            t, q = np.deg2rad(th), np.deg2rad(ph)
            x = rad_mm * np.sin(t) * np.cos(q)
            y = rad_mm * np.sin(t) * np.sin(q)
            z = rad_mm * np.cos(t)
            w.writerow([name, f'{x:.9g}', f'{y:.9g}', f'{z:.9g}', rad_mm, th, ph])
    with pytest.raises(ValueError, match='implausible source radii'):
        load_electrodes(p, target_radius_m=0.092)


def test_generate_standard_montage_systems():
    assert generate_standard_montage('10-20').n_electrodes == 20
    assert generate_standard_montage('10-10').n_electrodes == 69
    assert generate_standard_montage('10-05').n_electrodes == 336
    # custom selection may draw from the full 10-05 set (>= 4 electrodes)
    c = generate_standard_montage('custom', selection=['Cz', 'Fz', 'Pz', 'AFp1h'])
    assert set(c.names) == {'Cz', 'Fz', 'Pz', 'AFp1h'}
    # generation honours an arbitrary scalp radius
    a = generate_standard_montage('10-10', radius_m=0.10)
    assert abs(a.radius_m - 0.10) < 1e-12


def test_generate_standard_montage_unknown_selection():
    with pytest.raises(ValueError, match='Unknown electrode names'):
        generate_standard_montage('custom', selection=['NotAnElectrode'])
    with pytest.raises(ValueError, match='Unknown montage system'):
        generate_standard_montage('10-42')
