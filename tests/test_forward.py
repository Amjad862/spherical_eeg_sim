"""
Unit tests for the 4-layer spherical forward model.

Run with:
    cd <repo root>
    python -m pytest tests/test_forward.py -v

or simply:
    python tests/test_forward.py
"""

import sys as _sys
from pathlib import Path as _Path
_repo_root = _Path(__file__).resolve().parent.parent
if str(_repo_root) not in _sys.path:
    _sys.path.insert(0, str(_repo_root))

import numpy as np
import pytest
from scipy.special import eval_jacobi

from spherical_eeg_sim.core.forward import (
    _legendre_derivative,
    compute_transfer_coefficients,
    forward_potential,
    forward_potential_basis,
    forward_potential_batch,
    potentials_from_basis,
)
from spherical_eeg_sim.sources.sampling import (
    cartesian_to_spherical,
    world_orientation_to_spherical_frame,
)
from spherical_eeg_sim.core.electrodes import (
    load_electrodes,
    resolve_electrode_path,
)


# Standard head model for tests
RADII = (0.080, 0.081, 0.086, 0.092)
SIGMA = (0.330, 1.790, 0.0042, 0.330)


def _avg_ref_cosine(a, b):
    """Cosine similarity of two average-referenced sensor maps."""
    a = a - a.mean()
    b = b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_transfer_coefficients_shape():
    M = compute_transfer_coefficients(RADII, SIGMA, n_legendre=80)
    assert M.shape == (80,)
    assert np.all(np.isfinite(M))


def test_transfer_coefficient_n1_identity():
    """Guard the n=1 transfer coefficient against unintended code changes.

    This recomputes M_tilde[0] with the same closed-form 2x2 transfer-matrix
    algebra used inside compute_transfer_coefficients, so it is a
    self-consistency check, not an independent physical value: it
    catches an accidental edit to the recurrence but would not detect an error
    shared by both copies of the formula. Independent validation of the forward
    solution is provided by the MNE cross-check
    (test_forward_matches_mne_sphere_topography) and by
    scripts/validate_inverse_mne.py.
    """
    M = compute_transfer_coefficients(RADII, SIGMA, n_legendre=80)

    # Same closed-form n=1 algebra, evaluated independently of the loop.
    n = 1
    R1, R2, R3, R4 = RADII
    s1, s2, s3, s4 = SIGMA
    T12 = 1/((2*n+1)*s2) * np.array([
        [(n+1)*s2 + n*s1, (n+1)*(s2-s1)],
        [n*(s2-s1),       n*s2 + (n+1)*s1]
    ])
    T23 = 1/((2*n+1)*s3) * np.array([
        [(n+1)*s3 + n*s2, (n+1)*(s3-s2)],
        [n*(s3-s2),       n*s3 + (n+1)*s2]
    ])
    T34 = 1/((2*n+1)*s4) * np.array([
        [(n+1)*s4 + n*s3, (n+1)*(s4-s3)],
        [n*(s4-s3),       n*s4 + (n+1)*s3]
    ])
    S12 = np.diag([(R2/R1)**n, (R1/R2)**(n+1)])
    S23 = np.diag([(R3/R2)**n, (R2/R3)**(n+1)])
    S34 = np.diag([(R4/R3)**n, (R3/R4)**(n+1)])
    C_n = S34 @ T34 @ S23 @ T23 @ S12 @ T12
    alpha = n / (n + 1)
    Q = (alpha*C_n[0,1] - C_n[1,1]) / (C_n[1,0] - alpha*C_n[0,0])
    u4 = C_n[0,0]*Q + C_n[0,1]
    expected = (2*n + 1) / (n + 1) * u4 / (R1**(n+1) * 4*np.pi*s1)

    assert abs(M[0] - expected) < 1e-12


def test_basis_consistency():
    """L @ p (basis form) should equal forward_potential exactly."""
    M = compute_transfer_coefficients(RADII, SIGMA, n_legendre=60)
    rng = np.random.default_rng(0)
    elec_theta = rng.uniform(0.05, np.pi - 0.05, 32)
    elec_phi = rng.uniform(0, 2*np.pi, 32)
    for _ in range(20):
        r0 = rng.uniform(0.030, 0.078)
        theta0 = rng.uniform(0.05, np.pi - 0.05)
        phi0 = rng.uniform(0, 2*np.pi)
        p = rng.standard_normal(3) * 1e-9
        V_direct = forward_potential(r0, theta0, phi0, p[0], p[1], p[2],
                                      elec_theta, elec_phi, M,
                                      brain_radius_m=RADII[0])
        L = forward_potential_basis(
            r0, theta0, phi0, elec_theta, elec_phi, M,
            brain_radius_m=RADII[0],
        )
        V_basis = potentials_from_basis(L, p[0], p[1], p[2])
        assert np.max(np.abs(V_direct - V_basis)) < 1e-15


def test_tangential_at_aligned_electrode_is_zero():
    """Tangential dipole produces zero potential directly above it."""
    M = compute_transfer_coefficients(RADII, SIGMA, n_legendre=40)
    elec_t = np.array([np.pi/3])
    elec_p = np.array([np.pi/4])
    V = forward_potential(0.075, np.pi/3, np.pi/4, 0.0, 1e-9, 0.0,
                           elec_t, elec_p, M, brain_radius_m=RADII[0])
    assert abs(V[0]) < 1e-12


def test_legendre_derivative_is_accurate_at_and_near_endpoints():
    """The angular derivative remains accurate throughout [-1, 1]."""
    x = np.array([
        -1.0,
        -1.0 + 1e-14,
        -0.3,
        0.4,
        1.0 - 1e-14,
        1.0,
    ])
    for n in (2, 5, 20, 80):
        expected = (n + 1) / 2 * eval_jacobi(n - 1, 1, 1, x)
        actual = _legendre_derivative(n, x)
        np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=3e-12)


def test_nearly_aligned_tangential_potential_matches_jacobi_reference():
    """Near alignment must not suppress the finite tangential derivative."""
    M = compute_transfer_coefficients(RADII, SIGMA, n_legendre=80)
    r0 = 0.0795
    theta0 = 1.0
    phi0 = 0.7
    separation = 1e-7
    elec_theta = np.array([theta0 + separation])
    elec_phi = np.array([phi0])

    actual = forward_potential(
        r0, theta0, phi0, 0.0, 1e-9, 0.0,
        elec_theta, elec_phi, M, brain_radius_m=RADII[0],
    )[0]

    x = np.cos(separation)
    dcos_dtheta = np.sin(separation)
    expected = 0.0
    for n, coefficient in enumerate(M, start=1):
        derivative = (n + 1) / 2 * eval_jacobi(n - 1, 1, 1, x)
        expected += coefficient * r0 ** (n - 1) * derivative * dcos_dtheta * 1e-9

    np.testing.assert_allclose(actual, expected, rtol=2e-9, atol=1e-18)


def test_radial_dipole_amplitude_plausible():
    """30 nA*m radial cortical dipole gives ~1-3 uV peak at scalp."""
    M = compute_transfer_coefficients(RADII, SIGMA, n_legendre=80)
    elec_t = np.linspace(0.1, np.pi - 0.1, 73)
    elec_p = np.linspace(0, 2*np.pi, 73, endpoint=False)
    V = forward_potential(0.077, np.pi/3, np.pi/4, 30e-9, 0.0, 0.0,
                           elec_t, elec_p, M, brain_radius_m=RADII[0])
    peak_uV = np.max(np.abs(V)) * 1e6
    assert 0.5 < peak_uV < 5.0, f"30 nA*m cortical peak |V| = {peak_uV:.2f} uV"


def test_legendre_convergence():
    """Higher n_legendre should give converged values for shallow sources."""
    M40 = compute_transfer_coefficients(RADII, SIGMA, n_legendre=40)
    M80 = compute_transfer_coefficients(RADII, SIGMA, n_legendre=80)
    elec_t = np.array([np.pi/3])
    elec_p = np.array([np.pi/3])
    # Mid-depth source (not at the cortical surface)
    V40 = forward_potential(
        0.060, np.pi/4, 0.0, 1e-9, 0.0, 0.0, elec_t, elec_p, M40,
        brain_radius_m=RADII[0],
    )
    V80 = forward_potential(
        0.060, np.pi/4, 0.0, 1e-9, 0.0, 0.0, elec_t, elec_p, M80,
        brain_radius_m=RADII[0],
    )
    rel_diff = abs(V40[0] - V80[0]) / max(abs(V80[0]), 1e-30)
    assert rel_diff < 1e-3, f"n_legendre=40 not converged at mid-depth: rel diff {rel_diff:.3e}"


def _evaluate_world_dipole(position, moment, electrodes_xyz, M):
    rtp = cartesian_to_spherical(np.asarray(position)[None, :])[0]
    er = np.linalg.norm(electrodes_xyz, axis=1)
    etp = cartesian_to_spherical(electrodes_xyz)
    assert np.all(er > rtp[0])
    local = world_orientation_to_spherical_frame(moment, position)
    return forward_potential(
        rtp[0], rtp[1], rtp[2], local[0], local[1], local[2],
        etp[:, 1], etp[:, 2], M, brain_radius_m=RADII[0],
    )


def test_forward_rotational_covariance_for_mixed_orientation():
    """Rotating the complete physical configuration cannot change voltages."""
    M = compute_transfer_coefficients(RADII, SIGMA, n_legendre=80)
    rng = np.random.default_rng(73)
    position = np.array([0.041, -0.035, 0.047])
    moment = np.array([0.3, -0.8, 0.45]) * 1e-9
    directions = rng.normal(size=(41, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    electrodes = 0.092 * directions

    axis = np.array([0.2, 0.7, -0.3])
    axis /= np.linalg.norm(axis)
    angle = 1.137
    K = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    R = np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)

    v = _evaluate_world_dipole(position, moment, electrodes, M)
    v_rot = _evaluate_world_dipole(R @ position, R @ moment, electrodes @ R.T, M)
    assert np.allclose(v_rot, v, rtol=2e-12, atol=1e-16)


def test_phi_tangent_is_pole_safe_and_nonzero():
    """The +y tangent at the north pole has a finite, continuous field."""
    M = compute_transfer_coefficients(RADII, SIGMA, n_legendre=80)
    theta = np.array([0.35, 0.8, 1.4, 2.1])
    phi = np.array([0.2, 1.0, 2.5, 5.1])
    v_pole = forward_potential(
        0.075, 0.0, 0.0, 0.0, 0.0, 1e-9, theta, phi, M,
        brain_radius_m=RADII[0],
    )
    v_near = forward_potential(
        0.075, 1e-9, 0.0, 0.0, 0.0, 1e-9, theta, phi, M,
        brain_radius_m=RADII[0],
    )
    assert np.max(np.abs(v_pole)) > 1e-9
    assert np.allclose(v_pole, v_near, rtol=2e-8, atol=1e-14)

    L = forward_potential_batch(
        np.array([[0.075, 0.0, 0.0]]), theta, phi, M,
        brain_radius_m=RADII[0],
    )
    assert np.array_equal(L[0, :, 2], forward_potential_basis(
        0.075, 0.0, 0.0, theta, phi, M,
        brain_radius_m=RADII[0],
    )[:, 2])


def test_legendre_order_above_validated_limit_is_rejected():
    with pytest.raises(ValueError, match="120"):
        compute_transfer_coefficients(RADII, SIGMA, n_legendre=121)


def test_forward_matches_mne_sphere_topography():
    """Independent check: package scalp topographies agree with MNE's sphere model.

    The scalp fields are synthesized with this package's de Munck/Peters four-
    sphere series and compared, as average-referenced topographies, to MNE-
    Python's own multilayer-sphere EEG forward (the Berg-Scherg equivalent-source
    approximation) on the same geometry, conductivities, and electrodes. This is
    an independent-implementation cross-check of the forward physics -- covering
    the radial and both tangential moment directions -- rather than a
    self-consistency round trip. Both signed topography and absolute
    average-referenced field norm are checked.
    """
    mne = pytest.importorskip('mne')
    mne.set_log_level('ERROR')

    arr = load_electrodes(resolve_electrode_path('standard_1010_69.csv'))
    names = list(arr.names)
    xyz = np.asarray(arr.cartesian, dtype=float)
    xyz_sph = cartesian_to_spherical(xyz)
    M = compute_transfer_coefficients(RADII, SIGMA, n_legendre=80)

    info = mne.create_info(names, 1000.0, 'eeg')
    montage = mne.channels.make_dig_montage(
        ch_pos={n: p for n, p in zip(names, xyz)}, coord_frame='head',
    )
    info.set_montage(montage, verbose=False)
    sphere = mne.make_sphere_model(
        r0=(0.0, 0.0, 0.0), head_radius=float(RADII[-1]),
        relative_radii=tuple(np.asarray(RADII) / RADII[-1]), sigmas=SIGMA,
        verbose=False,
    )

    # One free-orientation forward solution evaluated at every test position.
    positions = np.array([
        [0.030, -0.025, 0.065],   # generic mid-depth
        [0.000, 0.000, 0.079],    # near-superficial, on the +z axis
        [-0.040, 0.030, 0.050],
        [0.055, 0.010, -0.030],
    ], dtype=float)
    nn = positions / np.linalg.norm(positions, axis=1, keepdims=True)
    src = mne.setup_volume_source_space(pos=dict(rr=positions, nn=nn), verbose=False)
    fwd = mne.make_forward_solution(
        info, trans=None, src=src, bem=sphere, eeg=True, meg=False, verbose=False,
    )
    fwd = mne.convert_forward_solution(fwd, force_fixed=False, verbose=False)
    order = [fwd['sol']['row_names'].index(n) for n in names]
    L_free = fwd['sol']['data'][order, :].reshape(len(names), positions.shape[0], 3)

    def pkg_field(pos, mom):
        rtp = cartesian_to_spherical(pos[None, :])[0]
        loc = world_orientation_to_spherical_frame(mom, pos)
        return forward_potential(
            rtp[0], rtp[1], rtp[2], loc[0], loc[1], loc[2],
            xyz_sph[:, 1], xyz_sph[:, 2], M,
            brain_radius_m=RADII[0],
        )

    rng = np.random.default_rng(4)
    worst = 1.0
    for j, pos in enumerate(positions):
        r = np.linalg.norm(pos)
        th, ph = cartesian_to_spherical(pos[None, :])[0][1:]
        e_r = pos / r
        e_th = np.array([np.cos(th) * np.cos(ph), np.cos(th) * np.sin(ph), -np.sin(th)])
        e_ph = np.array([-np.sin(ph), np.cos(ph), 0.0])
        rand = rng.normal(size=3)
        rand /= np.linalg.norm(rand)
        for mom in (e_r, e_th, e_ph, rand):
            v_pkg = pkg_field(pos, mom)
            v_mne = L_free[:, j, :] @ mom
            cos = _avg_ref_cosine(v_pkg, v_mne)
            v_pkg_ar = v_pkg - v_pkg.mean()
            v_mne_ar = v_mne - v_mne.mean()
            norm_ratio = np.linalg.norm(v_pkg_ar) / np.linalg.norm(v_mne_ar)
            worst = min(worst, cos)
            assert cos > 0.999, (
                f"position {j} moment {mom}: cosine {cos:.6f} below 0.999"
            )
            assert 0.98 < norm_ratio < 1.02, (
                f"position {j} moment {mom}: field-norm ratio "
                f"{norm_ratio:.6f} outside [0.98, 1.02]"
            )
    # Agreement is in fact very tight, not merely above the 0.999 floor.
    assert worst > 0.999


def test_superficial_solution_converges_before_stability_limit():
    """The default order is close to the validated order-120 reference."""
    theta = np.linspace(0.07, np.pi - 0.07, 97)
    phi = np.linspace(0.0, 2 * np.pi, 97, endpoint=False)
    m80 = compute_transfer_coefficients(RADII, SIGMA, 80)
    m120 = compute_transfer_coefficients(RADII, SIGMA, 120)
    v80 = forward_potential(
        0.079, 1.1, 2.0, 1e-9, 0.0, 0.0, theta, phi, m80,
        brain_radius_m=RADII[0],
    )
    v120 = forward_potential(
        0.079, 1.1, 2.0, 1e-9, 0.0, 0.0, theta, phi, m120,
        brain_radius_m=RADII[0],
    )
    rel = np.linalg.norm(v80 - v120) / np.linalg.norm(v120)
    assert rel < 1e-5


@pytest.mark.parametrize('call', [
    lambda m: forward_potential(
        np.nan, 0.5, 0.2, 1e-9, 0.0, 0.0,
        np.array([0.2]), np.array([0.1]), m, brain_radius_m=RADII[0],
    ),
    lambda m: forward_potential(
        np.array([0.05]), 0.5, 0.2, 1e-9, 0.0, 0.0,
        np.array([0.2]), np.array([0.1]), m, brain_radius_m=RADII[0],
    ),
    lambda m: forward_potential(
        0.05, 0.5, 0.2, np.array([1e-9]), 0.0, 0.0,
        np.array([0.2]), np.array([0.1]), m, brain_radius_m=RADII[0],
    ),
    lambda m: forward_potential_basis(
        0.05, -0.1, 0.2, np.array([0.2]), np.array([0.1]), m,
        brain_radius_m=RADII[0],
    ),
    lambda m: forward_potential_batch(
        np.ones((2, 2)), np.array([0.2]), np.array([0.1]), m,
        brain_radius_m=RADII[0],
    ),
    lambda m: forward_potential_batch(
        np.array([[0.05, 0.4, 0.2]]),
        np.array([0.2, 0.3]), np.array([0.1]), m,
        brain_radius_m=RADII[0],
    ),
    lambda m: forward_potential_batch(
        np.array([[0.05, 0.4, 0.2]]),
        np.array([0.2]), np.array([0.1]), m,
        brain_radius_m=RADII[0], progress_callback=3,
    ),
    lambda m: potentials_from_basis(np.ones((3, 2)), 1.0, 0.0, 0.0),
])
def test_public_forward_functions_reject_malformed_inputs(call):
    m = compute_transfer_coefficients(RADII, SIGMA, 40)
    with pytest.raises(ValueError):
        call(m)


@pytest.mark.parametrize("source_radius", [RADII[0], 0.081, 0.092])
def test_public_forward_functions_reject_sources_outside_the_brain(source_radius):
    m = compute_transfer_coefficients(RADII, SIGMA, 40)
    theta = np.array([0.3])
    phi = np.array([0.2])
    with pytest.raises(ValueError, match="brain_radius"):
        forward_potential(
            source_radius, 0.5, 0.2, 1e-9, 0.0, 0.0,
            theta, phi, m, brain_radius_m=RADII[0],
        )
    with pytest.raises(ValueError, match="brain_radius"):
        forward_potential_basis(
            source_radius, 0.5, 0.2, theta, phi, m,
            brain_radius_m=RADII[0],
        )
    with pytest.raises(ValueError, match="brain_radius"):
        forward_potential_batch(
            np.array([[source_radius, 0.5, 0.2]]),
            theta, phi, m, brain_radius_m=RADII[0],
        )


if __name__ == '__main__':
    test_transfer_coefficients_shape()
    test_transfer_coefficient_n1_identity()
    test_basis_consistency()
    test_tangential_at_aligned_electrode_is_zero()
    test_radial_dipole_amplitude_plausible()
    test_legendre_convergence()
    print("All forward-model tests passed!")
