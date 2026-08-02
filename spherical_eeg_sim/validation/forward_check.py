"""
Layer 1 validation: forward-model internal checks.

Verifies algebraic agreement between separately evaluated code paths,
rotational covariance, and finite/nonzero representative outputs. These are
self-consistency and numerical-sanity checks, not independent physical
validation; the independent comparison is implemented separately in
``scripts/validate_inverse_mne.py``.

These checks run automatically on every pipeline execution.
Failures are loud (raise RuntimeError); successes are silent except for
a one-line confirmation in the log.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np

from ..core.forward import (
    compute_transfer_coefficients,
    forward_potential,
    forward_potential_basis,
    potentials_from_basis,
)


@dataclass
class ForwardValidationResult:
    """Container for forward-model validation outcomes."""
    transfer_coefficient_match: bool
    transfer_coefficient_max_error: float
    basis_consistency: bool
    basis_max_error: float
    rotation_covariance: bool
    rotation_covariance_max_relative_error: float
    numerical_sanity: bool
    numerical_notes: List[str]
    n_test_dipoles: int

    @property
    def all_passed(self) -> bool:
        return (
            self.transfer_coefficient_match
            and self.basis_consistency
            and self.rotation_covariance
            and self.numerical_sanity
        )


def validate_forward_model(
    radii_m=(0.080, 0.081, 0.086, 0.092),
    conductivities_S_per_m=(0.330, 1.790, 0.0042, 0.330),
    n_legendre: int = 80,
    n_test_dipoles: int = 100,
    rng_seed: int = 1,
    tolerance_transfer: float = 1e-12,
    tolerance_basis: float = 1e-15,
    tolerance_rotation: float = 1e-10,
) -> ForwardValidationResult:
    """Run four forward-model internal checks.

    Check 1: Transfer-coefficient cross-check
        Compute M_tilde[0] (n=1) two ways: via the function and via direct
        2x2 transfer-matrix multiplication done by hand. They must match
        to within tolerance_transfer.

    Check 2: Basis-lead-field consistency
        For random dipole moments at random positions, verify that
        potentials_from_basis(L, p) equals forward_potential(...) to within
        machine epsilon (typical residual 1e-20 with our floating point).

    Check 3: Rotational covariance
        Jointly rotate the source position, world-frame moment, and electrode
        array and verify that the scalar potentials are unchanged.

    Check 4: Numerical sanity
        Verify that:
          - tangential dipole produces zero potential at the directly
            overhead electrode (within 1e-12)
          - representative radial cases produce finite, nonzero potentials
        The measured amplitudes are reported without assigning an empirical
        physiological acceptance interval. These are implementation sanity
        checks, not independent physical validation.

    Returns
    -------
    ForwardValidationResult
    """
    rng = np.random.default_rng(rng_seed)

    # ----- Transfer-coefficient check -----
    M_tilde = compute_transfer_coefficients(radii_m, conductivities_S_per_m, n_legendre)

    R1, R2, R3, R4 = radii_m
    s1, s2, s3, s4 = conductivities_S_per_m
    n = 1
    T12 = (1.0 / ((2 * n + 1) * s2)) * np.array([
        [(n + 1) * s2 + n * s1, (n + 1) * (s2 - s1)],
        [n * (s2 - s1), n * s2 + (n + 1) * s1],
    ])
    T23 = (1.0 / ((2 * n + 1) * s3)) * np.array([
        [(n + 1) * s3 + n * s2, (n + 1) * (s3 - s2)],
        [n * (s3 - s2), n * s3 + (n + 1) * s2],
    ])
    T34 = (1.0 / ((2 * n + 1) * s4)) * np.array([
        [(n + 1) * s4 + n * s3, (n + 1) * (s4 - s3)],
        [n * (s4 - s3), n * s4 + (n + 1) * s3],
    ])
    S12 = np.diag([(R2 / R1) ** n, (R1 / R2) ** (n + 1)])
    S23 = np.diag([(R3 / R2) ** n, (R2 / R3) ** (n + 1)])
    S34 = np.diag([(R4 / R3) ** n, (R3 / R4) ** (n + 1)])
    C_n = S34 @ T34 @ S23 @ T23 @ S12 @ T12
    alpha = n / (n + 1.0)
    Q = (alpha * C_n[0, 1] - C_n[1, 1]) / (C_n[1, 0] - alpha * C_n[0, 0])
    u4 = C_n[0, 0] * Q + C_n[0, 1]
    M_tilde_n1 = (2 * n + 1) / (n + 1.0) * u4 / (R1 ** (n + 1) * 4 * np.pi * s1)

    transfer_err = abs(M_tilde[0] - M_tilde_n1)
    transfer_match = transfer_err < tolerance_transfer

    # ----- Basis-lead-field consistency -----
    n_elec = 32
    elec_theta = rng.uniform(0.05, np.pi - 0.05, size=n_elec)
    elec_phi = rng.uniform(0, 2 * np.pi, size=n_elec)

    test_radii = rng.uniform(0.030, 0.078, size=n_test_dipoles)
    test_theta = rng.uniform(0.05, np.pi - 0.05, size=n_test_dipoles)
    test_phi = rng.uniform(0, 2 * np.pi, size=n_test_dipoles)
    moments = rng.standard_normal(size=(n_test_dipoles, 3)) * 1e-9

    max_diff = 0.0
    for i in range(n_test_dipoles):
        r0, t0, p0 = test_radii[i], test_theta[i], test_phi[i]
        pr, pt, pp = moments[i]
        V_direct = forward_potential(
            r0, t0, p0, pr, pt, pp, elec_theta, elec_phi, M_tilde,
            brain_radius_m=float(radii_m[0]),
        )
        L = forward_potential_basis(
            r0, t0, p0, elec_theta, elec_phi, M_tilde,
            brain_radius_m=float(radii_m[0]),
        )
        V_basis = potentials_from_basis(L, pr, pt, pp)
        diff = float(np.max(np.abs(V_direct - V_basis)))
        if diff > max_diff:
            max_diff = diff
    basis_consistent = max_diff < tolerance_basis

    # ----- Rotational covariance -----
    from ..sources.sampling import (
        cartesian_to_spherical,
        world_orientation_to_spherical_frame,
    )
    pos_world = np.array([0.041, -0.035, 0.047])
    moment_world = np.array([0.3, -0.8, 0.45]) * 1e-9
    e_xyz = np.column_stack([
        np.sin(elec_theta) * np.cos(elec_phi),
        np.sin(elec_theta) * np.sin(elec_phi),
        np.cos(elec_theta),
    ]) * float(radii_m[-1])
    axis = np.array([0.2, 0.7, -0.3])
    axis /= np.linalg.norm(axis)
    angle = 1.137
    K = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    rotation = (
        np.eye(3) + np.sin(angle) * K
        + (1.0 - np.cos(angle)) * (K @ K)
    )

    def evaluate_world(position, moment, electrodes_xyz):
        rtp = cartesian_to_spherical(position)
        ertp = cartesian_to_spherical(electrodes_xyz)
        local = world_orientation_to_spherical_frame(moment, position)
        return forward_potential(
            rtp[0], rtp[1], rtp[2],
            local[0], local[1], local[2],
            ertp[:, 1], ertp[:, 2], M_tilde,
            brain_radius_m=float(radii_m[0]),
        )

    v0 = evaluate_world(pos_world, moment_world, e_xyz)
    v1 = evaluate_world(
        rotation @ pos_world,
        rotation @ moment_world,
        e_xyz @ rotation.T,
    )
    rotation_error = float(
        np.max(np.abs(v1 - v0)) / max(np.max(np.abs(v0)), 1e-300)
    )
    rotation_covariant = rotation_error < tolerance_rotation

    # ----- Numerical sanity -----
    notes: List[str] = []
    numerical_sanity = True

    # Tangential dipole at directly-aligned electrode should be ~0
    elec_t_test = np.array([np.pi / 3])
    elec_p_test = np.array([np.pi / 4])
    V_tang = forward_potential(
        0.075, np.pi / 3, np.pi / 4, 0.0, 1e-9, 0.0,
        elec_t_test, elec_p_test, M_tilde,
        brain_radius_m=float(radii_m[0]),
    )
    if abs(V_tang[0]) > 1e-12:
        numerical_sanity = False
        notes.append(f"Tangential dipole at aligned electrode gave V={V_tang[0]:.3e} (expected ~0)")
    else:
        notes.append(f"Aligned tangential test: |V|={abs(V_tang[0]):.2e} V (expected ~0): OK")

    # Representative cortical radial case: finite and nonzero.
    elec_grid_t = np.linspace(0.1, np.pi - 0.1, 73)
    elec_grid_p = np.linspace(0, 2 * np.pi, 73, endpoint=False)
    V_cort = forward_potential(
        0.077, np.pi / 3, np.pi / 4, 30e-9, 0.0, 0.0,
        elec_grid_t, elec_grid_p, M_tilde,
        brain_radius_m=float(radii_m[0]),
    )
    cort_peak_uV = float(np.max(np.abs(V_cort)) * 1e6)
    if not (np.all(np.isfinite(V_cort)) and cort_peak_uV > 0.0):
        numerical_sanity = False
    notes.append(
        f"30 nA*m radial case at r=77mm: measured peak "
        f"|V|={cort_peak_uV:.2f} uV; finite/nonzero="
        f"{bool(np.all(np.isfinite(V_cort)) and cort_peak_uV > 0.0)}"
    )

    # Representative deep radial case: finite and nonzero.
    V_sub = forward_potential(
        0.050, np.pi / 3, np.pi / 4, 200e-9, 0.0, 0.0,
        elec_grid_t, elec_grid_p, M_tilde,
        brain_radius_m=float(radii_m[0]),
    )
    sub_peak_uV = float(np.max(np.abs(V_sub)) * 1e6)
    if not (np.all(np.isfinite(V_sub)) and sub_peak_uV > 0.0):
        numerical_sanity = False
    notes.append(
        f"200 nA*m radial case at r=50mm: measured peak "
        f"|V|={sub_peak_uV:.2f} uV; finite/nonzero="
        f"{bool(np.all(np.isfinite(V_sub)) and sub_peak_uV > 0.0)}"
    )

    return ForwardValidationResult(
        transfer_coefficient_match=transfer_match,
        transfer_coefficient_max_error=transfer_err,
        basis_consistency=basis_consistent,
        basis_max_error=max_diff,
        rotation_covariance=rotation_covariant,
        rotation_covariance_max_relative_error=rotation_error,
        numerical_sanity=numerical_sanity,
        numerical_notes=notes,
        n_test_dipoles=n_test_dipoles,
    )
