"""
Independent verification of the four-layer transfer coefficients.

The forward model reaches the scalp potential through a transfer-matrix
recursion: interface matrices relate the expansion coefficients across each
conductivity jump, propagation matrices carry them across each shell, and a
closure condition imposes the insulating exterior. That recursion is compact and
fast, but it is also the step where an algebraic slip would be least visible,
because every downstream check in the package reuses the same coefficients.

This script recomputes the same quantity from first principles by a completely
different route, so agreement is evidence about the mathematics rather than
about the implementation reproducing itself.

------------------------------------------------------------------------------
THE INDEPENDENT ROUTE
------------------------------------------------------------------------------
For Legendre order n the potential in shell j is written directly as

    phi_j(r) = A_j r**n + B_j r**(-(n+1))

with the source contributing a known r**(-(n+1)) term in the innermost shell.
Imposing the physical boundary conditions

    continuity of phi                  at R1, R2, R3
    continuity of sigma dphi/dr        at R1, R2, R3
    sigma dphi/dr = 0                  at R4   (insulating exterior)

gives a 7x7 linear system in (A1, A2, B2, A3, B3, A4, B4). Solving it and
evaluating phi at R4 yields the same surface coefficient the recursion produces.
No interface matrix, no propagation matrix, and no closure algebra is shared
between the two routes.

Both routes run in arbitrary precision so the comparison reflects the algebra
alone and not float64 round-off. The linear system is written in a scaled basis,
(r/R4)**n and (R4/r)**(n+1), which keeps it well conditioned at high order.

------------------------------------------------------------------------------
USAGE
------------------------------------------------------------------------------
    python scripts/validate_transfer_coefficients.py
    python scripts/validate_transfer_coefficients.py --write-canonical

Without an explicit path, the summary goes to a new validation_runs/ directory.

Requires mpmath:
    pip install "spherical_eeg_sim[validation]"      # or: pip install mpmath
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._validation_outputs import ValidationOutputs, add_output_options
from spherical_eeg_sim.config import Config
from spherical_eeg_sim.provenance import environment_snapshot

DPS = 60
ORDERS = (1, 2, 3, 5, 8, 13, 21, 34, 55, 80, 100, 120)

# Geometries spanning the reported brain-to-skull conductivity range, plus the
# degenerate cases that exercise the interface algebra from both sides.
def build_cases(cfg):
    radii = tuple(cfg.head.radii_m)
    s_brain, s_csf, _, s_scalp = cfg.head.conductivities_S_per_m
    return (
        ('default_skull_ratio_79', radii,
         tuple(cfg.head.conductivities_S_per_m)),
        ('skull_ratio_8', radii, (s_brain, s_csf, s_brain / 8.0, s_scalp)),
        ('skull_ratio_15', radii, (s_brain, s_csf, s_brain / 15.0, s_scalp)),
        ('three_shell_csf_collapsed', radii,
         (s_brain, s_brain, s_brain / 80.0, s_scalp)),
        ('homogeneous', radii, (s_brain, s_brain, s_brain, s_brain)),
    )


def _require_mpmath():
    try:
        import mpmath
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "mpmath is required for the arbitrary-precision comparison.\n"
            "Install it with:  pip install mpmath"
        ) from exc
    return mpmath


def transfer_matrix_route(radii, sigma, n, mp):
    """Surface coefficient from the package's transfer-matrix formulation."""
    R1, R2, R3, R4 = [mp.mpf(str(r)) for r in radii]
    s1, s2, s3, s4 = [mp.mpf(str(s)) for s in sigma]
    nn = mp.mpf(n)

    def interface(sa, sb):
        return mp.matrix([
            [(nn + 1) * sb + nn * sa, (nn + 1) * (sb - sa)],
            [nn * (sb - sa), nn * sb + (nn + 1) * sa],
        ]) / ((2 * nn + 1) * sb)

    def propagate(Ra, Rb):
        return mp.matrix([[(Rb / Ra) ** nn, 0], [0, (Ra / Rb) ** (nn + 1)]])

    C = (propagate(R3, R4) * interface(s3, s4)
         * propagate(R2, R3) * interface(s2, s3)
         * propagate(R1, R2) * interface(s1, s2))
    alpha = nn / (nn + 1)
    Q = (alpha * C[0, 1] - C[1, 1]) / (C[1, 0] - alpha * C[0, 0])
    u4 = C[0, 0] * Q + C[0, 1]
    return (2 * nn + 1) / (nn + 1) * u4 / (R1 ** (nn + 1) * 4 * mp.pi * s1)


def boundary_value_route(radii, sigma, n, mp):
    """Surface coefficient from a direct solve of the layered boundary problem."""
    R = [mp.mpf(str(r)) for r in radii]
    S = [mp.mpf(str(s)) for s in sigma]
    R1, R2, R3, R4 = R
    nn = mp.mpf(n)

    # Scaled radial basis: growing (r/R4)**n and decaying (R4/r)**(n+1).
    grow = lambda r: (r / R4) ** nn
    decay = lambda r: (R4 / r) ** (nn + 1)
    inv = lambda r: R4 / r

    # Unit source strength in the innermost shell, expressed in the same basis.
    source = 1 / (4 * mp.pi * S[0]) * R4 ** (-(nn + 1))

    slot = {'a1': 0, 'a2': 1, 'b2': 2, 'a3': 3, 'b3': 4, 'a4': 5, 'b4': 6}
    M = mp.zeros(7, 7)
    rhs = mp.zeros(7, 1)

    def potential_row(row, r, grow_key, decay_key, sign):
        if grow_key:
            M[row, slot[grow_key]] += sign * grow(r)
        if decay_key:
            M[row, slot[decay_key]] += sign * decay(r)

    def current_row(row, r, grow_key, decay_key, s, sign):
        if grow_key:
            M[row, slot[grow_key]] += sign * s * inv(r) * nn * grow(r)
        if decay_key:
            M[row, slot[decay_key]] -= sign * s * inv(r) * (nn + 1) * decay(r)

    # R1: potential and normal current continuity, source on the inner side.
    potential_row(0, R1, 'a1', None, 1)
    potential_row(0, R1, 'a2', 'b2', -1)
    rhs[0] = -source * decay(R1)
    current_row(1, R1, 'a1', None, S[0], 1)
    current_row(1, R1, 'a2', 'b2', S[1], -1)
    rhs[1] = S[0] * inv(R1) * (nn + 1) * decay(R1) * source

    # R2 and R3: plain continuity.
    potential_row(2, R2, 'a2', 'b2', 1)
    potential_row(2, R2, 'a3', 'b3', -1)
    current_row(3, R2, 'a2', 'b2', S[1], 1)
    current_row(3, R2, 'a3', 'b3', S[2], -1)
    potential_row(4, R3, 'a3', 'b3', 1)
    potential_row(4, R3, 'a4', 'b4', -1)
    current_row(5, R3, 'a3', 'b3', S[2], 1)
    current_row(5, R3, 'a4', 'b4', S[3], -1)

    # R4: insulating exterior.
    current_row(6, R4, 'a4', 'b4', S[3], 1)

    solution = mp.lu_solve(M, rhs)
    return solution[slot['a4']] + solution[slot['b4']]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('--json', default=None,
                    help='write the measurements to this path')
    add_output_options(ap)
    args = ap.parse_args()
    outputs = ValidationOutputs(
        ap, args, name='transfer-coefficients',
        canonical={'json': 'transfer_coefficients.json'},
        explicit={'json': args.json} if args.json is not None else None,
    )

    mp = _require_mpmath()
    mp.mp.dps = DPS
    cfg = Config()
    cases = build_cases(cfg)

    print('Transfer-coefficient verification')
    print(f'  precision : {DPS} decimal digits')
    print(f'  orders    : {", ".join(str(o) for o in ORDERS)}')
    print('  routes    : transfer-matrix recursion  vs  direct 7x7 boundary '
          'solve\n')

    results = {}
    for label, radii, sigma in cases:
        worst = mp.mpf(0)
        worst_order = None
        for n in ORDERS:
            a = transfer_matrix_route(radii, sigma, n, mp)
            b = boundary_value_route(radii, sigma, n, mp)
            rel = abs((a - b) / b)
            if rel > worst:
                worst, worst_order = rel, n
        results[label] = {
            'conductivities_S_per_m': list(sigma),
            'max_relative_difference': mp.nstr(worst, 8),
            'at_order': worst_order,
        }
        print(f'  {label:28s} sigma_skull={sigma[2]:.5f}  '
              f'max rel diff = {mp.nstr(worst, 6)}  (order {worst_order})')

    # Both routes accumulate round-off across the recursion and the LU
    # solve, so the criterion allows headroom below the working precision
    # while remaining far stricter than any physically meaningful margin.
    tolerance = mp.mpf(10) ** (-(DPS - 20))
    agreed = all(mp.mpf(results[k]['max_relative_difference']) < tolerance
                 for k in results)
    print(f'\n  agreement tolerance : {mp.nstr(tolerance, 3)}')
    print(f'  verdict             : '
          f'{"the two routes agree" if agreed else "MISMATCH"}')

    if args.write_canonical and not agreed:
        return 1
    with outputs.stage() as paths:
        payload = {
            'software_environment': environment_snapshot(
                extra_distributions=('mpmath',)
            ),
            'decimal_digits': DPS,
            'orders': list(ORDERS),
            'radii_m': list(cfg.head.radii_m),
            'tolerance': mp.nstr(tolerance, 3),
            'agreement': bool(agreed),
            'cases': results,
        }
        with paths['json'].open('w', newline='\n') as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write('\n')
    print(f"\n  measurements written to {outputs.paths['json']}")
    return 0 if agreed else 1


if __name__ == '__main__':
    raise SystemExit(main())
