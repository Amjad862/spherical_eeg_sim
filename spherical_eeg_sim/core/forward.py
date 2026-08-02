"""
Four-layer spherical head forward model for EEG.

Computes scalp electric potentials due to a current dipole inside a
4-layer concentric-sphere head model (brain, CSF, skull, scalp).

The implementation has two parts:
  1. compute_transfer_coefficients(...) -- precomputes M_tilde_n once for
     a given head geometry. Cost is O(n_legendre); ~80 iterations.
  2. forward_potential(...) -- computes potentials at electrodes for a
     specific dipole. Vectorised across electrodes; vectorised across
     dipoles for batch use via forward_potential_batch(...).

Mathematical background
-----------------------
The scalp potential at angle gamma from the dipole position, for a current
dipole with radial moment p_r and tangential moment p_t in a 4-layer
spherical head, is:

    V(gamma) = sum_{n=1}^{Nmax} M_tilde_n * r0^(n-1) * (
                  p_r * n * P_n(cos gamma)
                  + p_theta * dP_n/dx |_{x=cos gamma} * (...)
                  + p_phi   * dP_n/dx |_{x=cos gamma} * (...)
              )

where P_n are Legendre polynomials and M_tilde_n encodes the layer geometry
and conductivities through a recursive transfer-matrix product.

References
----------
- de Munck, J. C. & Peters, M. J. (1993). A fast method to compute the
  potential in the multisphere model. IEEE Trans. Biomed. Eng. 40(11).
- Nunez, P. L. & Srinivasan, R. (2006). Electric Fields of the Brain:
  The Neurophysics of EEG (2nd ed.). Oxford University Press.
- Stok, C. J. (1986). The inverse problem in EEG and MEG with application
  to visual evoked responses. PhD Thesis, University of Twente.

The series is truncated at ``n_legendre`` terms; the radial term uses
P_n(cos gamma) directly while the tangential terms use its derivative.
Increasing ``n_legendre`` initially reduces truncation error for superficial
dipoles, but float64 round-off eventually grows enough that total accuracy is
not monotone. The precompute cost is O(n_legendre).
"""

from __future__ import annotations
import numbers
import numpy as np
import warnings
from typing import Sequence, Tuple
from scipy.special import lpmv  # Associated Legendre, m=0 gives P_n


# Upper bound on the expansion order accepted by the float64 recurrence.
#
# The unscaled transfer recurrence carries intermediate terms that span a
# dynamic range of roughly (R_scalp / R_brain) ** (2n), so individual transfer
# coefficients lose relative accuracy as the order grows: with the default
# geometry and skull conductivity they are round-off dominated well before this
# bound. Scalp potentials remain accurate because order n enters weighted by
# (r_source / R_scalp) ** n, which suppresses the corrupted tail exponentially,
# but the coefficients themselves must not be treated as a converged reference
# at high order. This cap bounds the recurrence rather than asserting that order
# 120 is exact; scripts/legendre_convergence.py measures the realised truncation
# and round-off error of the potential against an arbitrary-precision reference,
# and estimate_transfer_coefficient_accuracy() reports a per-order conditioning
# indicator for the coefficients themselves.
MAX_NUMERIC_LEGENDRE_ORDER = 120

# Expansion order beyond which added terms stop improving the scalp potential
# for the default geometry. Truncation error falls with order while round-off in
# the transfer recurrence grows, so the total error has an interior minimum
# rather than decreasing monotonically: scripts/legendre_convergence.py measures
# the superficial radial topography error against an arbitrary-precision
# reference at 7.197e-07 for order 100 and 5.017e-06 for order 120. Orders above
# this are accepted, because the crossover moves with the radii and
# conductivities, but they are not more accurate at the shipped defaults and
# compute_transfer_coefficients() says so.
ACCURACY_OPTIMAL_LEGENDRE_ORDER = 100

# Conditioning-indicator level above which direct use of a coefficient warrants
# a warning. The indicator is deliberately conservative near the threshold but
# is not a rigorous forward-error bound; at high order a coefficient can lose
# its sign or collapse to zero through cancellation even though the scalp
# potential it contributes to stays accurate.
COEFFICIENT_RELIABILITY_LIMIT = 1e-1


def _validate_electrode_angles(
    elec_theta: np.ndarray,
    elec_phi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and return one-dimensional electrode-angle arrays."""
    theta = np.asarray(elec_theta, dtype=np.float64)
    phi = np.asarray(elec_phi, dtype=np.float64)
    if theta.ndim != 1 or phi.ndim != 1 or theta.shape != phi.shape:
        raise ValueError(
            "elec_theta and elec_phi must be one-dimensional arrays with "
            "the same shape"
        )
    if theta.size == 0:
        raise ValueError("at least one electrode angle is required")
    if not np.all(np.isfinite(theta)) or not np.all(np.isfinite(phi)):
        raise ValueError("electrode angles must contain only finite values")
    if np.any(theta < 0.0) or np.any(theta > np.pi):
        raise ValueError("elec_theta must lie in [0, pi]")
    return theta, phi


def _validate_transfer_coefficients(M_tilde: np.ndarray) -> np.ndarray:
    """Validate and return a nonempty transfer-coefficient vector."""
    coefficients = np.asarray(M_tilde, dtype=np.float64)
    if coefficients.ndim != 1 or coefficients.size == 0:
        raise ValueError("M_tilde must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("M_tilde must contain only finite values")
    return coefficients


def _validate_spherical_position(r0: float, theta0: float, phi0: float) -> None:
    """Validate a source position expressed in spherical coordinates."""
    if any(np.asarray(value).ndim != 0 for value in (r0, theta0, phi0)):
        raise ValueError("r0, theta0, and phi0 must be scalar values")
    values = np.asarray([r0, theta0, phi0], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("r0, theta0, and phi0 must be finite")
    if r0 <= 0.0:
        raise ValueError("r0 must be positive")
    if theta0 < 0.0 or theta0 > np.pi:
        raise ValueError("theta0 must lie in [0, pi]")


# =============================================================================
# Legendre polynomials
# =============================================================================

def _legendre_polynomial(n: int, x: np.ndarray) -> np.ndarray:
    """Compute the Legendre polynomial P_n(x).

    Wraps scipy.special.lpmv with m=0 to yield ordinary Legendre polynomials.

    Parameters
    ----------
    n : int
        Polynomial order (n >= 0).
    x : ndarray
        Argument values, typically cos(gamma) in [-1, 1].

    Returns
    -------
    ndarray of same shape as x.
    """
    return lpmv(0, n, x)


def _legendre_derivative(n: int, x: np.ndarray) -> np.ndarray:
    """Compute dP_n(x)/dx with an endpoint-safe upward recurrence.

    The coupled identity

        dP_n/dx = n * P_{n-1}(x) + x * dP_{n-1}(x)

    avoids division by ``1 - x**2`` and remains well defined at both endpoints.
    """
    if n == 0:
        return np.zeros_like(x)
    if n == 1:
        return np.ones_like(x)

    P_nm2 = np.ones_like(x)
    P_nm1 = x.copy()
    dP_nm1 = np.ones_like(x)
    for order in range(2, n + 1):
        Pn = (
            (2 * order - 1) * x * P_nm1 - (order - 1) * P_nm2
        ) / order
        dPn = order * P_nm1 + x * dP_nm1
        P_nm2 = P_nm1
        P_nm1 = Pn
        dP_nm1 = dPn
    return dP_nm1


# =============================================================================
# Transfer coefficients M_tilde_n
# =============================================================================

def compute_transfer_coefficients(
    radii_m: Sequence[float],
    conductivities_S_per_m: Sequence[float],
    n_legendre: int,
) -> np.ndarray:
    """Compute the M_tilde_n transfer coefficients for a 4-layer sphere.

    These coefficients encode the effect of the layered geometry on the
    forward solution. They depend only on the head model parameters and
    n_legendre, so they should be computed ONCE and reused across all
    dipole evaluations.

    Parameters
    ----------
    radii_m : sequence of 4 floats
        Layer radii (innermost to outermost), in meters.
        E.g. (0.080, 0.081, 0.086, 0.092) for brain/CSF/skull/scalp.
    conductivities_S_per_m : sequence of 4 floats
        Conductivities of each layer, in S/m. Same order as radii_m.
    n_legendre : int
        Maximum order of Legendre expansion.

    Returns
    -------
    ndarray, shape (n_legendre,)
        M_tilde_n for n = 1, 2, ..., n_legendre. Index 0 corresponds to n=1.

    Warns
    -----
    RuntimeWarning
        If the conditioning indicator for any requested order crosses the
        warning level. The warning names the first such order. Scalp potentials
        can remain accurate when the source-domain tail decays sufficiently
        because order n reaches the scalp weighted by
        ``(r_source / R_scalp) ** n``; the warning concerns direct use of the
        coefficients. Use :func:`estimate_transfer_coefficient_accuracy` to
        inspect the per-order indicator.

    Notes
    -----
    The transfer matrices T_{ij} and scaling matrices S_{ij} are 2x2; the
    final M_tilde combines them via S34*T34 * S23*T23 * S12*T12 followed
    by application of the insulating-exterior boundary condition.
    """
    M_tilde, estimated_error = _transfer_coefficients_and_error(
        radii_m, conductivities_S_per_m, n_legendre,
    )
    if n_legendre > ACCURACY_OPTIMAL_LEGENDRE_ORDER:
        warnings.warn(
            f"n_legendre={n_legendre} is past the order at which added terms "
            f"stop improving the scalp potential for the shipped geometry. "
            f"Truncation error falls with order while round-off in the transfer "
            f"recurrence grows, so accuracy has an interior optimum near order "
            f"{ACCURACY_OPTIMAL_LEGENDRE_ORDER} rather than improving "
            f"monotonically; for the default radii and conductivities the "
            f"superficial radial topography error is about 7.2e-07 at order "
            f"{ACCURACY_OPTIMAL_LEGENDRE_ORDER} and about 5.0e-06 at order 120. "
            f"The crossover depends on the geometry, so this is guidance rather "
            f"than a limit: run scripts/legendre_convergence.py to locate it for "
            f"a non-default head model.",
            RuntimeWarning,
            stacklevel=2,
        )
    ill_conditioned = np.flatnonzero(
        estimated_error > COEFFICIENT_RELIABILITY_LIMIT
    )
    if ill_conditioned.size:
        first_order = int(ill_conditioned[0]) + 1
        warnings.warn(
            f"Transfer coefficients at Legendre order >= {first_order} have a "
            f"large float64 conditioning indicator for this geometry. "
            f"Cancellation in the transfer recurrence can remove significant "
            f"digits and, at sufficiently high order, can change a coefficient's "
            f"sign or reduce it to zero. Scalp potentials can remain accurate "
            f"when the source-domain tail decays sufficiently because order n "
            f"reaches the scalp "
            f"weighted by (r_source / R_scalp) ** n, which suppresses the "
            f"affected tail. Reduce n_legendre below {first_order}, or inspect "
            f"estimate_transfer_coefficient_accuracy(), if the coefficients "
            f"themselves are needed as reference values.",
            RuntimeWarning,
            stacklevel=2,
        )
    return M_tilde


def estimate_transfer_coefficient_accuracy(
    radii_m: Sequence[float],
    conductivities_S_per_m: Sequence[float],
    n_legendre: int,
) -> np.ndarray:
    """Estimate float64 conditioning for each transfer coefficient.

    The transfer recurrence closes the layered system by forming
    ``C[1, 0] - alpha * C[0, 0]``, whose two terms grow like
    ``(R_scalp / R_brain) ** (2n)`` while their difference does not. The
    returned indicator is machine epsilon times the larger of the cancellation
    ratio of that subtraction and the growth of the intermediate matrix entries
    relative to the closed result. Both ratios are available at no extra cost
    inside the recurrence. The indicator identifies ill-conditioned orders but
    is not a rigorous forward-error bound.

    Parameters
    ----------
    radii_m, conductivities_S_per_m, n_legendre
        As for :func:`compute_transfer_coefficients`.

    Returns
    -------
    ndarray, shape (n_legendre,)
        Heuristic conditioning indicator for ``M_tilde[n - 1]``. Larger values
        indicate greater susceptibility to cancellation; values at or above
        one warrant treating the coefficient as numerically unreliable.

    Notes
    -----
    The indicator describes the coefficients, not the scalp potential. Order n
    contributes to the potential weighted by ``(r_source / R_scalp) ** n``, so a
    coefficient whose own relative error is large can still contribute a
    negligible absolute error; ``scripts/legendre_convergence.py`` measures the
    realised topography error against an arbitrary-precision reference.
    """
    _, estimated_error = _transfer_coefficients_and_error(
        radii_m, conductivities_S_per_m, n_legendre,
    )
    return estimated_error


def _transfer_coefficients_and_error(
    radii_m: Sequence[float],
    conductivities_S_per_m: Sequence[float],
    n_legendre: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate the transfer recurrence and its conditioning indicator."""
    radii = np.asarray(radii_m, dtype=np.float64)
    sigma = np.asarray(conductivities_S_per_m, dtype=np.float64)
    _validate_head_model(radii, sigma, n_legendre)

    R1, R2, R3, R4 = radii
    s1, s2, s3, s4 = sigma

    M_tilde = np.zeros(n_legendre, dtype=np.float64)
    estimated_error = np.zeros(n_legendre, dtype=np.float64)
    eps = float(np.finfo(np.float64).eps)

    for idx in range(n_legendre):
        n = idx + 1  # M_tilde[0] corresponds to Legendre order n=1 (1-indexed).

        T12 = (1.0 / ((2 * n + 1) * s2)) * np.array([
            [(n + 1) * s2 + n * s1,       (n + 1) * (s2 - s1)],
            [n * (s2 - s1),               n * s2 + (n + 1) * s1]
        ])
        T23 = (1.0 / ((2 * n + 1) * s3)) * np.array([
            [(n + 1) * s3 + n * s2,       (n + 1) * (s3 - s2)],
            [n * (s3 - s2),               n * s3 + (n + 1) * s2]
        ])
        T34 = (1.0 / ((2 * n + 1) * s4)) * np.array([
            [(n + 1) * s4 + n * s3,       (n + 1) * (s4 - s3)],
            [n * (s4 - s3),               n * s4 + (n + 1) * s3]
        ])

        S12 = np.diag([(R2 / R1) ** n, (R1 / R2) ** (n + 1)])
        S23 = np.diag([(R3 / R2) ** n, (R2 / R3) ** (n + 1)])
        S34 = np.diag([(R4 / R3) ** n, (R3 / R4) ** (n + 1)])

        C_n = S34 @ T34 @ S23 @ T23 @ S12 @ T12

        # Insulating-exterior boundary condition (zero radial current at
        # scalp -> outside). alpha = n / (n + 1) closes the system.
        alpha = n / (n + 1.0)
        denom = C_n[1, 0] - alpha * C_n[0, 0]
        Q = (alpha * C_n[0, 1] - C_n[1, 1]) / denom
        u4 = C_n[0, 0] * Q + C_n[0, 1]

        M_tilde[idx] = (
            (2 * n + 1) / (n + 1.0) * u4
            / (R1 ** (n + 1) * 4 * np.pi * s1)
        )

        # Cancellation in the closure, and growth of the intermediate entries
        # relative to the closed result, form a conservative conditioning
        # indicator. It is diagnostic rather than a rigorous error bound.
        with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
            closure_ratio = (
                max(abs(C_n[1, 0]), abs(alpha * C_n[0, 0])) / abs(denom)
                if denom != 0.0 else np.inf
            )
            growth_ratio = (
                np.max(np.abs(C_n)) / abs(u4) if u4 != 0.0 else np.inf
            )
        indicator = eps * max(closure_ratio, growth_ratio)
        estimated_error[idx] = (
            indicator if np.isfinite(indicator) else np.inf
        )

    return M_tilde, estimated_error


def _validate_head_model(
    radii: np.ndarray, sigma: np.ndarray, n_legendre: int,
) -> None:
    """Validate four-layer geometry, conductivities, and expansion order."""
    if radii.shape != (4,):
        raise ValueError(f"radii_m must have length 4; got {radii.shape}")
    if sigma.shape != (4,):
        raise ValueError(f"conductivities must have length 4; got {sigma.shape}")
    if not np.all(np.isfinite(radii)) or not np.all(radii > 0):
        raise ValueError("radii_m must contain four finite, positive values")
    if not np.all(np.diff(radii) > 0):
        raise ValueError("radii_m must be strictly increasing")
    if not np.all(np.isfinite(sigma)) or not np.all(sigma > 0):
        raise ValueError(
            "conductivities_S_per_m must contain four finite, positive values"
        )
    if isinstance(n_legendre, (bool, np.bool_)) or not isinstance(
        n_legendre, (int, np.integer)
    ):
        raise ValueError("n_legendre must be an integer")
    if n_legendre < 1 or n_legendre > MAX_NUMERIC_LEGENDRE_ORDER:
        raise ValueError(
            f"n_legendre must be in [1, {MAX_NUMERIC_LEGENDRE_ORDER}]; "
            f"got {n_legendre}"
        )


def _validate_brain_radius(brain_radius_m: float) -> float:
    """Validate and return the outer radius of the source compartment."""
    if (
        isinstance(brain_radius_m, (bool, np.bool_))
        or not isinstance(brain_radius_m, numbers.Real)
    ):
        raise ValueError("brain_radius_m must be a finite positive scalar")
    value = float(brain_radius_m)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("brain_radius_m must be a finite positive scalar")
    return value


def _validate_source_within_brain(r0: float, brain_radius_m: float) -> None:
    """Require a source to lie strictly inside the brain compartment."""
    brain_radius_m = _validate_brain_radius(brain_radius_m)
    if r0 >= brain_radius_m:
        raise ValueError(
            f"source radius r0={r0} m must be smaller than "
            f"brain_radius_m={brain_radius_m} m"
        )


# =============================================================================
# Forward potential for a single dipole
# =============================================================================

def forward_potential(
    r0: float,
    theta0: float,
    phi0: float,
    p_r: float,
    p_theta: float,
    p_phi: float,
    elec_theta: np.ndarray,
    elec_phi: np.ndarray,
    M_tilde: np.ndarray,
    *,
    brain_radius_m: float,
) -> np.ndarray:
    """Scalp potentials at electrodes for a single dipole.

    Parameters
    ----------
    r0, theta0, phi0 : float
        Dipole position in spherical coordinates (m, rad, rad).
    p_r, p_theta, p_phi : float
        Dipole moment components in the local spherical frame (A*m).
        p_r along the radial direction (away from origin),
        p_theta along the local theta direction,
        p_phi along the local phi direction.
    elec_theta, elec_phi : ndarray, shape (N,)
        Electrode polar and azimuthal angles (rad).
    M_tilde : ndarray, shape (n_legendre,)
        Precomputed transfer coefficients (see compute_transfer_coefficients).
    brain_radius_m : float
        Radius of the innermost compartment. The source must lie strictly
        inside this radius.

    Returns
    -------
    V : ndarray, shape (N,)
        Scalp potential at each electrode (volts).
    """
    _validate_spherical_position(r0, theta0, phi0)
    _validate_source_within_brain(r0, brain_radius_m)
    moment = np.asarray([p_r, p_theta, p_phi], dtype=np.float64)
    if moment.shape != (3,) or not np.all(np.isfinite(moment)):
        raise ValueError("dipole-moment components must be finite scalars")
    elec_theta, elec_phi = _validate_electrode_angles(elec_theta, elec_phi)
    M_tilde = _validate_transfer_coefficients(M_tilde)
    n_electrodes = elec_theta.shape[0]
    n_legendre = M_tilde.shape[0]

    cos_gamma = (
        np.cos(theta0) * np.cos(elec_theta)
        + np.sin(theta0) * np.sin(elec_theta) * np.cos(phi0 - elec_phi)
    )
    cos_gamma = np.clip(cos_gamma, -1.0, 1.0)

    dcos_dtheta = (
        -np.sin(theta0) * np.cos(elec_theta)
        + np.cos(theta0) * np.sin(elec_theta) * np.cos(phi0 - elec_phi)
    )
    # The azimuthal term in the spherical gradient is
    # (1 / sin(theta0)) * d(cos(gamma))/d(phi0).  Evaluating the quotient
    # directly avoids both a removable singularity at the source poles and a
    # loss of accuracy near them.  The minus sign follows from differentiating
    # cos(phi0 - phi_e) with respect to phi0.
    dcos_dphi_over_sin = -np.sin(elec_theta) * np.sin(phi0 - elec_phi)

    V = np.zeros(n_electrodes, dtype=np.float64)
    # Evaluate the Legendre polynomials by the upward Bonnet recurrence
    #   P_n(x) = ((2n-1) x P_{n-1}(x) - (n-1) P_{n-2}(x)) / n,
    # which is numerically stable on [-1, 1] and costs O(1) per order, so the
    # whole sum is O(n_legendre) instead of O(n_legendre^2) (one lpmv call per
    # order). The derivative reuses P_{n-1} via the standard identity. r0_power
    # accumulates r0^(n-1) incrementally to avoid a pow() each order.
    P_nm1 = np.ones_like(cos_gamma)   # P_0(x) = 1
    P_nm2 = np.zeros_like(cos_gamma)  # P_{-1} placeholder (unused at n=1)
    dP_nm1 = np.zeros_like(cos_gamma)  # P'_0(x) = 0
    r0_power = 1.0                    # r0^(n-1); 1.0 at n=1
    for idx in range(n_legendre):
        n = idx + 1
        if n == 1:
            Pn = cos_gamma
            dPn = np.ones_like(cos_gamma)
        else:
            Pn = ((2 * n - 1) * cos_gamma * P_nm1 - (n - 1) * P_nm2) / n
            dPn = n * P_nm1 + cos_gamma * dP_nm1

        Mr = M_tilde[idx] * r0_power
        V += p_r * (Mr * n * Pn) + Mr * dPn * (
            p_theta * dcos_dtheta + p_phi * dcos_dphi_over_sin
        )

        P_nm2 = P_nm1
        P_nm1 = Pn
        dP_nm1 = dPn
        r0_power *= r0

    return V


# =============================================================================
# Vectorised batch forward
# =============================================================================

def forward_potential_basis(
    r0: float,
    theta0: float,
    phi0: float,
    elec_theta: np.ndarray,
    elec_phi: np.ndarray,
    M_tilde: np.ndarray,
    *,
    brain_radius_m: float,
) -> np.ndarray:
    """Compute the 3-basis lead field at one position.

    For a position (r0, theta0, phi0), returns three lead-field columns:
      [:, 0] -- scalp potentials for unit radial moment p_r = 1, p_t = 0
      [:, 1] -- scalp potentials for unit p_theta = 1, others = 0
      [:, 2] -- scalp potentials for unit p_phi   = 1, others = 0

    Any dipole moment p = (p_r, p_theta, p_phi) at this position produces
    scalp potentials V = L @ p where L is this 3-basis lead-field matrix.
    This is the foundation of the storage-efficient 'lead-field tensor'
    representation: store L per position (shape Nx3), synthesise any
    orientation/magnitude combination at training time as L @ p.

    Parameters
    ----------
    r0, theta0, phi0 : float
        Dipole position in spherical coordinates.
    elec_theta, elec_phi : ndarray, shape (N,)
        Electrode positions.
    M_tilde : ndarray, shape (n_legendre,)
    brain_radius_m : float
        Radius of the innermost compartment.

    Returns
    -------
    L : ndarray, shape (N, 3)
        Lead-field matrix in the local spherical frame at the dipole position.
        Columns are (radial, theta, phi) basis lead fields.
    """
    _validate_spherical_position(r0, theta0, phi0)
    _validate_source_within_brain(r0, brain_radius_m)
    elec_theta, elec_phi = _validate_electrode_angles(elec_theta, elec_phi)
    M_tilde = _validate_transfer_coefficients(M_tilde)
    L = _leadfield_basis_block(
        np.asarray([[r0, theta0, phi0]], dtype=np.float64),
        elec_theta, elec_phi, M_tilde,
    )
    return L[0]


def _leadfield_basis_block(
    positions_spherical: np.ndarray,   # (B, 3)
    elec_theta: np.ndarray,            # (N,)
    elec_phi: np.ndarray,              # (N,)
    M_tilde: np.ndarray,               # (n_legendre,)
) -> np.ndarray:
    """Vectorised 3-basis lead field for a block of positions -> (B, N, 3).

    This is the single shared implementation behind both ``forward_potential_basis``
    (one position) and ``forward_potential_batch`` (many positions). Every
    (position, electrode) pair is evaluated with the same Bonnet recurrence and
    the same operation order, so the per-position and batched paths are
    numerically identical. The Legendre sum is O(n_legendre) per pair.
    """
    pos = np.asarray(positions_spherical, dtype=np.float64)
    elec_theta = np.asarray(elec_theta, dtype=np.float64)
    elec_phi = np.asarray(elec_phi, dtype=np.float64)
    B = pos.shape[0]
    N = elec_theta.shape[0]
    n_legendre = M_tilde.shape[0]

    r0 = pos[:, 0][:, None]      # (B, 1)
    theta0 = pos[:, 1][:, None]  # (B, 1)
    phi0 = pos[:, 2][:, None]    # (B, 1)
    et = elec_theta[None, :]     # (1, N)
    ep = elec_phi[None, :]       # (1, N)

    cos_gamma = (
        np.cos(theta0) * np.cos(et)
        + np.sin(theta0) * np.sin(et) * np.cos(phi0 - ep)
    )
    cos_gamma = np.clip(cos_gamma, -1.0, 1.0)                  # (B, N)
    dcos_dtheta = (
        -np.sin(theta0) * np.cos(et)
        + np.cos(theta0) * np.sin(et) * np.cos(phi0 - ep)
    )
    # Pole-safe azimuthal part of the spherical gradient; see
    # forward_potential() for the derivative and sign convention.
    dcos_dphi_over_sin = -np.sin(et) * np.sin(phi0 - ep)
    L = np.zeros((B, N, 3), dtype=np.float64)
    # See forward_potential() for the coupled polynomial/derivative recurrence.
    P_nm1 = np.ones_like(cos_gamma)   # P_0
    P_nm2 = np.zeros_like(cos_gamma)  # P_{-1} placeholder
    dP_nm1 = np.zeros_like(cos_gamma)  # P'_0
    r0_power = np.ones((B, 1), dtype=np.float64)   # r0^(n-1)
    for idx in range(n_legendre):
        n = idx + 1
        if n == 1:
            Pn = cos_gamma
            dPn = np.ones_like(cos_gamma)
        else:
            Pn = ((2 * n - 1) * cos_gamma * P_nm1 - (n - 1) * P_nm2) / n
            dPn = n * P_nm1 + cos_gamma * dP_nm1

        Mr = M_tilde[idx] * r0_power           # (B, 1)
        L[:, :, 0] += Mr * (n * Pn)
        L[:, :, 1] += Mr * (dPn * dcos_dtheta)
        L[:, :, 2] += Mr * (dPn * dcos_dphi_over_sin)

        P_nm2 = P_nm1
        P_nm1 = Pn
        dP_nm1 = dPn
        r0_power = r0_power * r0

    return L


def forward_potential_batch(
    positions_spherical: np.ndarray,
    elec_theta: np.ndarray,
    elec_phi: np.ndarray,
    M_tilde: np.ndarray,
    *,
    brain_radius_m: float,
    progress_callback=None,
) -> np.ndarray:
    """Compute basis lead fields for a batch of dipole positions.

    Parameters
    ----------
    positions_spherical : ndarray, shape (P, 3)
        Dipole positions as (r, theta, phi) in (m, rad, rad).
    elec_theta, elec_phi : ndarray, shape (N,)
        Electrode positions.
    M_tilde : ndarray, shape (n_legendre,)
    brain_radius_m : float
        Radius of the innermost compartment. Every source must lie strictly
        inside this radius.
    progress_callback : callable, optional
        Called periodically with (current, total) to report progress.

    Returns
    -------
    L_batch : ndarray, shape (P, N, 3)
        Stack of basis lead-field matrices. L_batch[p, :, :] is the
        lead-field matrix for position p, with columns
        (radial, theta, phi) in the local spherical frame.
    """
    positions_spherical = np.asarray(positions_spherical, dtype=np.float64)
    if positions_spherical.ndim != 2 or positions_spherical.shape[1:] != (3,):
        raise ValueError(
            "positions_spherical must have shape (n_positions, 3)"
        )
    if not np.all(np.isfinite(positions_spherical)):
        raise ValueError("positions_spherical must contain only finite values")
    if np.any(positions_spherical[:, 0] <= 0.0):
        raise ValueError("source radii must be positive")
    brain_radius_m = _validate_brain_radius(brain_radius_m)
    if np.any(positions_spherical[:, 0] >= brain_radius_m):
        raise ValueError(
            "every source radius must be smaller than brain_radius_m"
        )
    if (np.any(positions_spherical[:, 1] < 0.0)
            or np.any(positions_spherical[:, 1] > np.pi)):
        raise ValueError("source polar angles must lie in [0, pi]")
    elec_theta, elec_phi = _validate_electrode_angles(elec_theta, elec_phi)
    M_tilde = _validate_transfer_coefficients(M_tilde)
    if progress_callback is not None and not callable(progress_callback):
        raise ValueError("progress_callback must be callable or None")
    P = positions_spherical.shape[0]
    N = elec_theta.shape[0]
    L_batch = np.zeros((P, N, 3), dtype=np.float64)
    if P == 0:
        if progress_callback is not None:
            progress_callback(0, 0)
        return L_batch

    # Process positions in blocks: the per-pair work is vectorised within a
    # block (O(n_legendre) passes over (block, N) arrays), while blocking
    # bounds peak memory and lets us report progress on large pools.
    block = 4096
    for start in range(0, P, block):
        end = min(start + block, P)
        L_batch[start:end] = _leadfield_basis_block(
            positions_spherical[start:end], elec_theta, elec_phi, M_tilde,
        )
        if progress_callback is not None:
            progress_callback(end, P)

    return L_batch


# =============================================================================
# Convenience: synthesize potentials from a basis lead field
# =============================================================================

def potentials_from_basis(
    L: np.ndarray,
    p_r: float,
    p_theta: float,
    p_phi: float,
) -> np.ndarray:
    """Synthesize scalp potentials from a 3-basis lead field and a moment.

    For an L of shape (N, 3) and a moment vector (p_r, p_theta, p_phi)
    expressed in the local spherical frame at the dipole's position, this
    returns L @ [p_r, p_theta, p_phi] (shape (N,)).
    """
    L = np.asarray(L, dtype=np.float64)
    if L.ndim != 2 or L.shape[1:] != (3,):
        raise ValueError("L must have shape (n_electrodes, 3)")
    if L.shape[0] == 0 or not np.all(np.isfinite(L)):
        raise ValueError("L must be nonempty and contain only finite values")
    moment = np.asarray([p_r, p_theta, p_phi], dtype=np.float64)
    if moment.shape != (3,) or not np.all(np.isfinite(moment)):
        raise ValueError("dipole-moment components must be finite scalars")
    return L @ moment
