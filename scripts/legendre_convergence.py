"""
Legendre truncation and round-off audit for the four-layer forward model.

The scalp potential is a Legendre series that must be truncated at
``cfg.head.n_legendre``. This script measures how much accuracy that truncation
costs on a real montage, at superficial, mid-depth, and deep source radii, for
both radial and tangential moments.

------------------------------------------------------------------------------
WHY THE REFERENCE IS COMPUTED IN ARBITRARY PRECISION
------------------------------------------------------------------------------
Comparing one float64 order against another float64 order cannot separate the
two error sources that matter here, and cannot bound either of them:

  * Truncation error falls as the order rises, because order n enters the scalp
    potential weighted by (r_source / R_scalp) ** n.
  * Round-off error rises as the order grows, because the unscaled transfer
    recurrence carries intermediate terms spanning a dynamic range of roughly
    (R_scalp / R_brain) ** (2n). With the default geometry and skull
    conductivity the individual transfer coefficients are already round-off
    dominated at the upper end of the accepted order range.

A high float64 order is therefore not a converged reference: it is the point
where the second effect takes over from the first. This script instead builds
the reference with mpmath at a precision high enough that neither effect is
visible, sums far beyond convergence, and requires the reference at half order
to agree with the full reference before reporting errors.

------------------------------------------------------------------------------
USAGE
------------------------------------------------------------------------------
    python scripts/legendre_convergence.py
    python scripts/legendre_convergence.py --config my_config.json
    python scripts/legendre_convergence.py --write-canonical

Without an explicit path, the summary goes to a new validation_runs/ directory.

Requires mpmath for the reference:
    pip install "spherical_eeg_sim[validation]"      # or: pip install mpmath
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._validation_outputs import ValidationOutputs, add_output_options
from spherical_eeg_sim.config import Config
from spherical_eeg_sim.core.electrodes import load_electrodes
from spherical_eeg_sim.core.forward import (
    MAX_NUMERIC_LEGENDRE_ORDER,
    compute_transfer_coefficients,
    forward_potential_basis,
    potentials_from_basis,
)
from spherical_eeg_sim.sources.sampling import cartesian_to_spherical
from spherical_eeg_sim.provenance import environment_snapshot

# The accepted range plus two below-range diagnostics that show the onset of
# convergence.
ORDERS = (20, 30, 40, 50, 60, 80, 100, 120)

# Reference settings. A run is reported only when the half-order and full-order
# arbitrary-precision topographies agree within the declared tolerance.
REFERENCE_ORDER = 400
REFERENCE_DPS = 120
REFERENCE_SELF_CONVERGENCE_TOLERANCE = 1e-9

# Preferred locations as fractions of the brain radius. Each is projected onto
# the configured source domain, so the shipped defaults retain the archived
# 79.5, 55, and 35 mm cases while custom configurations remain valid.
PREFERRED_MID_RADIUS_FRACTION = 0.6875
PREFERRED_DEEP_RADIUS_FRACTION = 0.4375


def _require_mpmath():
    try:
        import mpmath
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "mpmath is required to build the arbitrary-precision reference.\n"
            "Install it with:  pip install mpmath"
        ) from exc
    return mpmath


def _mp_interface(sa, sb, nn, mp):
    return mp.matrix([
        [(nn + 1) * sb + nn * sa, (nn + 1) * (sb - sa)],
        [nn * (sb - sa), nn * sb + (nn + 1) * sa],
    ]) / ((2 * nn + 1) * sb)


def _mp_propagate(Ra, Rb, nn, mp):
    return mp.matrix([
        [(Rb / Ra) ** nn, 0],
        [0, (Ra / Rb) ** (nn + 1)],
    ])


def reference_transfer_coefficients(radii, sigma, n_max, mp):
    """Transfer coefficients M_tilde_n evaluated in arbitrary precision."""
    R1, R2, R3, R4 = [mp.mpf(str(r)) for r in radii]
    s1, s2, s3, s4 = [mp.mpf(str(s)) for s in sigma]
    out = []
    for n in range(1, n_max + 1):
        nn = mp.mpf(n)
        C = (
            _mp_propagate(R3, R4, nn, mp) * _mp_interface(s3, s4, nn, mp)
            * _mp_propagate(R2, R3, nn, mp) * _mp_interface(s2, s3, nn, mp)
            * _mp_propagate(R1, R2, nn, mp) * _mp_interface(s1, s2, nn, mp)
        )
        alpha = nn / (nn + 1)
        Q = (alpha * C[0, 1] - C[1, 1]) / (C[1, 0] - alpha * C[0, 0])
        u4 = C[0, 0] * Q + C[0, 1]
        out.append((2 * nn + 1) / (nn + 1) * u4
                   / (R1 ** (nn + 1) * 4 * mp.pi * s1))
    return out


def _active_source_intervals_mm(cfg):
    intervals = []
    if cfg.sources.n_cortical_positions > 0:
        intervals.append(tuple(map(float, cfg.sources.cortical_radius_range_mm)))
    if (cfg.source_mode != 'patch'
            and cfg.sources.n_subcortical_positions > 0):
        intervals.append(
            tuple(map(float, cfg.sources.subcortical_radius_range_mm))
        )
    intervals.append(tuple(map(float, cfg.noise.background_radius_range_mm)))
    return intervals


def _project_to_intervals(value, intervals):
    candidates: list[float] = []
    for low, high in intervals:
        if low <= value <= high:
            return float(value)
        candidates.extend((float(low), float(high)))
    return min(candidates, key=lambda candidate: abs(candidate - value))


def _radius_label(radius_mm):
    value = f'{radius_mm:g}'.replace('.', '_')
    return f'{value}mm'


def derive_source_cases(cfg):
    """Build valid superficial, middle, and deep cases from the configuration."""
    intervals = _active_source_intervals_mm(cfg)
    if not intervals:
        raise ValueError('the configuration has no active source interval')
    brain_radius_mm = float(cfg.head.radii_m[0]) * 1000.0
    superficial = max(high for _, high in intervals)
    middle = _project_to_intervals(
        PREFERRED_MID_RADIUS_FRACTION * brain_radius_mm,
        intervals,
    )
    deep = _project_to_intervals(
        PREFERRED_DEEP_RADIUS_FRACTION * brain_radius_mm,
        intervals,
    )
    return (
        (
            f'superficial_radial_{_radius_label(superficial)}',
            superficial,
            (1.0, 0.0, 0.0),
        ),
        (
            f'superficial_tangential_{_radius_label(superficial)}',
            superficial,
            (0.0, 1.0, 0.0),
        ),
        (
            f'mid_depth_radial_{_radius_label(middle)}',
            middle,
            (1.0, 0.0, 0.0),
        ),
        (
            f'deep_radial_{_radius_label(deep)}',
            deep,
            (1.0, 0.0, 0.0),
        ),
    )


def validate_for_convergence_audit(cfg):
    """Validate a configuration while allowing this script to inspect its tail.

    A configuration that fails only the geometric-tail publication gate is
    still a legitimate convergence-audit target. All other validation remains
    enforced, and the declared tolerance is restored before the configuration
    is recorded.
    """
    try:
        cfg.validate()
        return False
    except ValueError as error:
        if "geometric-tail indicator" not in str(error):
            raise

    declared_tolerance = cfg.head.truncation_tail_tolerance
    cfg.head.truncation_tail_tolerance = float(np.finfo(np.float64).max)
    try:
        cfg.validate()
    finally:
        cfg.head.truncation_tail_tolerance = declared_tolerance
    return True


def reference_topography(r_mm, moment, elec_theta, elec_phi, coefficients, mp):
    """Scalp topography of a source on the +z axis, in arbitrary precision.

    The source sits at theta0 = 0, so cos(gamma) reduces to cos(theta_e) and the
    two tangential angular factors reduce to closed forms that stay finite at the
    pole. This mirrors the pole-safe formulation used by the float64 path.
    """
    p_r, p_theta, p_phi = [mp.mpf(str(v)) for v in moment]
    r0 = mp.mpf(str(r_mm)) / 1000
    topography = []
    for theta_e, phi_e in zip(elec_theta, elec_phi):
        te = mp.mpf(str(float(theta_e)))
        pe = mp.mpf(str(float(phi_e)))
        x = mp.cos(te)
        d_cos_d_theta = mp.sin(te) * mp.cos(pe)
        d_cos_d_phi_over_sin = mp.sin(te) * mp.sin(pe)
        one_minus_x2 = 1 - x * x

        P_prev, P_prev2 = mp.mpf(1), mp.mpf(0)
        r_power, total = mp.mpf(1), mp.mpf(0)
        for index in range(len(coefficients)):
            n = index + 1
            if n == 1:
                P_n, dP_n = x, mp.mpf(1)
            else:
                P_n = ((2 * n - 1) * x * P_prev - (n - 1) * P_prev2) / n
                dP_n = (n * (P_prev - x * P_n) / one_minus_x2
                        if one_minus_x2 != 0 else mp.mpf(0))
            c = coefficients[index] * r_power
            total += (p_r * c * n * P_n
                      + c * dP_n * (p_theta * d_cos_d_theta
                                    + p_phi * d_cos_d_phi_over_sin))
            P_prev2, P_prev, r_power = P_prev, P_n, r_power * r0
        topography.append(float(total))
    return np.asarray(topography, dtype=np.float64)


def float64_topography(r_mm, moment, elec_theta, elec_phi, radii, sigma, order):
    """Scalp topography from the shipped float64 forward path.

    The sweep deliberately visits orders whose individual transfer coefficients
    are round-off dominated, because locating that crossover is the point of the
    measurement. The coefficient-reliability warning is therefore expected here
    and is silenced; what the sweep reports is the error in the topography those
    coefficients produce, which the ``(r_source / R_scalp) ** n`` weighting keeps
    far smaller than the error in the coefficients themselves.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        M = compute_transfer_coefficients(radii, sigma, order)
    position = np.array([0.0, 0.0, r_mm * 1e-3])
    r, theta, phi = cartesian_to_spherical(position)
    basis = forward_potential_basis(
        r, theta, phi, elec_theta, elec_phi, M,
        brain_radius_m=float(radii[0]),
    )
    return potentials_from_basis(basis, *moment)


def relative_error(value, reference):
    denominator = float(np.linalg.norm(reference))
    if denominator == 0.0:
        raise ValueError('reference topography has zero norm')
    return float(np.linalg.norm(value - reference) / denominator)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('--montage', default=None,
                    help='montage CSV (default: the configured montage)')
    ap.add_argument('--config', default=None,
                    help='JSON configuration (default: shipped defaults)')
    ap.add_argument('--json', default=None,
                    help='write the measurements to this path')
    add_output_options(ap)
    args = ap.parse_args()
    outputs = ValidationOutputs(
        ap, args, name='legendre-convergence',
        canonical={'json': 'legendre_convergence.json'},
        settings={'config': None, 'montage': None},
        explicit={'json': args.json} if args.json is not None else None,
    )

    mp = _require_mpmath()
    mp.mp.dps = REFERENCE_DPS

    cfg = Config.from_json(args.config) if args.config else Config()
    tail_gate_would_reject = validate_for_convergence_audit(cfg)
    radii = list(cfg.head.radii_m)
    sigma = list(cfg.head.conductivities_S_per_m)
    montage = args.montage or cfg.electrode_file
    electrodes = load_electrodes(montage, target_radius_m=radii[-1])
    theta, phi = electrodes.theta_rad, electrodes.phi_rad
    cases = derive_source_cases(cfg)

    print('Legendre truncation and round-off audit')
    print(f'  montage           : {montage} ({electrodes.n_electrodes} channels)')
    print(f'  radii (m)         : {radii}')
    print(f'  conductivities    : {sigma}')
    print(f'  configured order  : {cfg.head.n_legendre}')
    print(f'  accepted range    : 40 to {MAX_NUMERIC_LEGENDRE_ORDER}')
    if tail_gate_would_reject:
        print('  tail gate          : would reject generation at the declared '
              'tolerance; audit continues')
    print(f'  source cases      : '
          f'{", ".join(f"{label}={radius:g} mm" for label, radius, _ in cases)}')
    print(f'  reference         : mpmath, {REFERENCE_DPS} digits, '
          f'order {REFERENCE_ORDER}')

    print(f'\n  building the arbitrary-precision reference '
          f'({REFERENCE_ORDER} coefficients) ...')
    ref_coefficients = reference_transfer_coefficients(
        radii, sigma, REFERENCE_ORDER, mp
    )

    reference_topographies = {}
    reference_self_error = {}
    half_order = REFERENCE_ORDER // 2
    for label, r_mm, moment in cases:
        full = reference_topography(
            r_mm, moment, theta, phi, ref_coefficients, mp
        )
        half = reference_topography(
            r_mm, moment, theta, phi, ref_coefficients[:half_order], mp
        )
        reference_topographies[label] = full
        reference_self_error[label] = relative_error(half, full)

    maximum_reference_error = max(reference_self_error.values())
    print(f'  reference check   : order {half_order} vs {REFERENCE_ORDER}, '
          f'max relative difference {maximum_reference_error:.3e}')
    if maximum_reference_error > REFERENCE_SELF_CONVERGENCE_TOLERANCE:
        raise RuntimeError(
            'arbitrary-precision reference did not converge: order '
            f'{half_order} vs {REFERENCE_ORDER} differs by '
            f'{maximum_reference_error:.3e}, exceeding '
            f'{REFERENCE_SELF_CONVERGENCE_TOLERANCE:.1e}'
        )

    results = {}
    for label, r_mm, moment in cases:
        reference = reference_topographies[label]
        errors = {}
        for order in ORDERS:
            value = float64_topography(
                r_mm, moment, theta, phi, radii, sigma, order
            )
            errors[order] = relative_error(value, reference)
        results[label] = errors

    width = max(len(label) for label in results)
    print('\nRelative topography error of the float64 path '
          '(2-norm over channels):\n')
    header = f'  {"case".ljust(width)} ' + ''.join(f'{o:>12d}' for o in ORDERS)
    print(header)
    print('  ' + '-' * (len(header) - 2))
    for label, errors in results.items():
        row = f'  {label.ljust(width)} '
        row += ''.join(f'{errors[o]:12.3e}' for o in ORDERS)
        print(row)

    configured = cfg.head.n_legendre
    if configured in ORDERS:
        worst = max(results[label][configured] for label in results)
        print(f'\n  worst case at the configured order {configured}: {worst:.3e}')
    floor = max(min(errors.values()) for errors in results.values())
    print(f'  best attainable within the accepted range: {floor:.3e}')
    print('  (that floor is set by float64 round-off in the transfer '
          'recurrence, not by truncation)')

    with outputs.stage() as paths:
        payload = {
            'software_environment': environment_snapshot(
                extra_distributions=('mpmath',)
            ),
            'config': cfg.to_dict(),
            'tail_gate_would_reject_generation': bool(
                tail_gate_would_reject
            ),
            'montage': montage,
            'n_electrodes': int(electrodes.n_electrodes),
            'radii_m': radii,
            'conductivities_S_per_m': sigma,
            'configured_n_legendre': int(configured),
            'accepted_order_range': [40, int(MAX_NUMERIC_LEGENDRE_ORDER)],
            'reference': {
                'method': 'mpmath arbitrary precision',
                'decimal_digits': int(REFERENCE_DPS),
                'order': int(REFERENCE_ORDER),
                'self_convergence': {
                    'comparison_order': int(half_order),
                    'relative_error_tolerance': float(
                        REFERENCE_SELF_CONVERGENCE_TOLERANCE
                    ),
                    'maximum_relative_error': float(
                        maximum_reference_error
                    ),
                    'relative_error_by_case': reference_self_error,
                    'passed': True,
                },
            },
            'source_cases': [
                {
                    'label': label,
                    'radius_mm': float(radius_mm),
                    'moment_local': list(moment),
                }
                for label, radius_mm, moment in cases
            ],
            'orders': list(ORDERS),
            'relative_error': {
                label: {str(o): errors[o] for o in ORDERS}
                for label, errors in results.items()
            },
        }
        with paths['json'].open('w', newline='\n') as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write('\n')
    print(f"\n  measurements written to {outputs.paths['json']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
