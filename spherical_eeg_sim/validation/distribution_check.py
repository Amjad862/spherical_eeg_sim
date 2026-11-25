"""
Layer 2 validation: distribution checks on the generated dataset.

Verifies that:
  - The realized cortical/subcortical fraction is statistically consistent
  - Orientation norms and low-order moments match the configured mode
  - Positions and magnitudes remain inside configured bounds
  - Target-SNR values remain inside the configured range
  - Serialized positions satisfy the selected split eligibility policy
  - Clean scalp-potential summaries are finite and nonzero

Returns a structured DistributionValidationResult that can be saved to JSON
in the output directory.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence
import numpy as np
from scipy.stats import kstest


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str
    statistics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DiagnosticResult:
    """Measured distribution evidence that never gates publication."""

    name: str
    detail: str
    statistics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DistributionValidationResult:
    checks: List[CheckResult]
    diagnostics: List[DiagnosticResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'all_passed': self.all_passed,
            'checks': [
                {
                    'name': c.name,
                    'passed': c.passed,
                    'detail': c.detail,
                    'statistics': _jsonify(c.statistics),
                }
                for c in self.checks
            ],
            'diagnostics': [
                {
                    'name': d.name,
                    'detail': d.detail,
                    'statistics': _jsonify(d.statistics),
                    'gating': False,
                }
                for d in self.diagnostics
            ],
        }


def _jsonify(obj):
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def _append_magnitude_ks(
    diagnostics: List[DiagnosticResult],
    population: str,
    magnitudes: np.ndarray,
    low: float,
    high: float,
    distribution: str,
) -> None:
    """Record a non-gating KS diagnostic for a configured magnitude law."""
    if magnitudes.size < 20 or distribution not in ('log_uniform', 'uniform'):
        return
    if distribution == 'log_uniform':
        uniformized = (
            (np.log(magnitudes) - np.log(low))
            / (np.log(high) - np.log(low))
        )
    else:
        uniformized = (magnitudes - low) / (high - low)
    ks = kstest(uniformized, 'uniform')
    diagnostics.append(DiagnosticResult(
        name=f'{population}_magnitude_uniformized_ks',
        detail=(
            f"Uniformized magnitude KS D={ks.statistic:.4f}, "
            f"p={ks.pvalue:.3g}"
        ),
        statistics={
            'ks_D': float(ks.statistic),
            'ks_p': float(ks.pvalue),
            'n': int(magnitudes.size),
            'distribution': distribution,
        },
    ))


def validate_distributions(
    positions_xyz: np.ndarray,        # (B, 3) or (B, K, 3)
    populations: np.ndarray,          # (B,) or (B, K)
    magnitudes_nAm: np.ndarray,       # (B,) or (B, K)
    snr_db: np.ndarray,               # (B,)
    cortical_radius_range_mm: Sequence[float],
    subcortical_radius_range_mm: Sequence[float],
    cortical_magnitude_range_nAm: Sequence[float],
    subcortical_magnitude_range_nAm: Sequence[float],
    snr_db_range: Sequence[float],
    expected_subcortical_fraction: float,
    orientation_mode: str,
    magnitude_distribution: str = 'log_uniform',
    orientations_world: Optional[np.ndarray] = None,  # (B, 3) or (B, K, 3)
    train_position_indices: Optional[np.ndarray] = None,
    val_position_indices: Optional[np.ndarray] = None,
    test_position_indices: Optional[np.ndarray] = None,
    assigned_position_indices: Optional[Dict[str, np.ndarray]] = None,
    serialized_position_indices: Optional[Dict[str, np.ndarray]] = None,
    source_mode: str = 'point',
    noise_model: str = 'snr_target',
    eeg_signal: Optional[np.ndarray] = None,   # (B, N, T)
    peak_uV: Optional[np.ndarray] = None,      # (B,) precomputed peak |V| in uV
    split_policy: str = 'position',
) -> DistributionValidationResult:
    """Run all distribution checks on a generated dataset.

    Each check has a passed/fail boolean and a human-readable detail string.
    Checks are not aborted on failure; all are run and returned.

    The inputs are expected to cover the WHOLE split being validated (the
    pipeline accumulates lightweight ground truth across all chunks), so the
    fractions and ranges reflect the entire dataset rather than one chunk.

    The scalp-amplitude check needs only a per-sample peak amplitude. Provide
    it directly via ``peak_uV`` (so the full EEG tensor need not be held in
    memory) or, for convenience, pass ``eeg_signal`` and let the peak be
    computed here. If neither is given, the amplitude check is skipped.

    Note: in 'patch' mode, sources are cortex-only by design, so the
    subcortical_fraction check is skipped.
    """
    checks: List[CheckResult] = []
    diagnostics: List[DiagnosticResult] = []

    if positions_xyz.ndim == 3:
        valid_mask = ~np.isnan(positions_xyz[..., 0])
        flat_positions = positions_xyz[valid_mask]
        flat_pops = populations[valid_mask]
        flat_mags = magnitudes_nAm[valid_mask]
    else:
        flat_positions = positions_xyz
        flat_pops = populations
        flat_mags = magnitudes_nAm

    # ----- Check: subcortical fraction (skipped in patch mode) -----
    if source_mode == 'patch':
        checks.append(CheckResult(
            name='subcortical_fraction',
            passed=True,
            detail="Skipped (patch mode is cortex-only by design)",
            statistics={'skipped': True},
        ))
    else:
        n_total = flat_pops.size
        n_sub = int((flat_pops == 1).sum())
        measured_frac = n_sub / max(n_total, 1)
        frac_diff = abs(measured_frac - expected_subcortical_fraction)
        if expected_subcortical_fraction in (0.0, 1.0):
            adaptive_tol = 0.0
        else:
            binomial_se = np.sqrt(
                expected_subcortical_fraction
                * (1.0 - expected_subcortical_fraction)
                / max(n_total, 1)
            )
            # A five-standard-error acceptance band makes stochastic false
            # failures rare while shrinking with the actual validation size.
            adaptive_tol = max(5.0 * binomial_se, 1.0 / max(n_total, 1))
        checks.append(CheckResult(
            name='subcortical_fraction',
            passed=frac_diff <= adaptive_tol,
            detail=(
                f"Expected fraction {expected_subcortical_fraction:.3f}, "
                f"got {measured_frac:.3f} (n_sub={n_sub}/{n_total}, "
                f"tolerance={adaptive_tol:.3f})"
            ),
            statistics={
                'expected': expected_subcortical_fraction,
                'measured': measured_frac,
                'n_sub': n_sub,
                'n_total': n_total,
                'tolerance_used': adaptive_tol,
                'tolerance_basis': 'five binomial standard errors',
            },
        ))

    # ----- Check: cortical positions within radial range -----
    cort_pos = flat_positions[flat_pops == 0]
    if cort_pos.size > 0:
        cort_radii = np.linalg.norm(cort_pos, axis=1) * 1000  # mm
        clow, chigh = cortical_radius_range_mm
        eps = 1e-9
        in_range = ((cort_radii >= clow - eps) & (cort_radii <= chigh + eps)).all()
        checks.append(CheckResult(
            name='cortical_radial_range',
            passed=bool(in_range),
            detail=(
                f"Cortical radii: min={cort_radii.min():.2f} mm, "
                f"max={cort_radii.max():.2f} mm, expected [{clow}, {chigh}] mm"
            ),
            statistics={
                'min_mm': float(cort_radii.min()),
                'max_mm': float(cort_radii.max()),
                'mean_mm': float(cort_radii.mean()),
                'expected_range_mm': [clow, chigh],
            },
        ))

    # ----- Check: subcortical positions within radial range -----
    sub_pos = flat_positions[flat_pops == 1]
    if sub_pos.size > 0:
        sub_radii = np.linalg.norm(sub_pos, axis=1) * 1000
        slow, shigh = subcortical_radius_range_mm
        eps = 1e-9
        in_range = ((sub_radii >= slow - eps) & (sub_radii <= shigh + eps)).all()
        checks.append(CheckResult(
            name='subcortical_radial_range',
            passed=bool(in_range),
            detail=(
                f"Subcortical radii: min={sub_radii.min():.2f} mm, "
                f"max={sub_radii.max():.2f} mm, expected [{slow}, {shigh}] mm"
            ),
            statistics={
                'min_mm': float(sub_radii.min()),
                'max_mm': float(sub_radii.max()),
                'mean_mm': float(sub_radii.mean()),
                'expected_range_mm': [slow, shigh],
            },
        ))

    # ----- Check: cortical magnitudes within range -----
    cort_mags = flat_mags[flat_pops == 0]
    if cort_mags.size > 0:
        clow, chigh = cortical_magnitude_range_nAm
        eps = max(1e-12, 1e-12 * chigh)
        in_range = (
            (cort_mags.min() >= clow - eps)
            and (cort_mags.max() <= chigh + eps)
        )
        checks.append(CheckResult(
            name='cortical_magnitude_range',
            passed=bool(in_range),
            detail=(
                f"Cortical magnitudes: [{cort_mags.min():.2f}, "
                f"{cort_mags.max():.2f}] nA*m, expected [{clow}, {chigh}]"
            ),
            statistics={
                'min_nAm': float(cort_mags.min()),
                'max_nAm': float(cort_mags.max()),
                'mean_nAm': float(cort_mags.mean()),
                'expected_range_nAm': [clow, chigh],
            },
        ))
        _append_magnitude_ks(
            diagnostics, 'cortical', cort_mags, clow, chigh,
            magnitude_distribution,
        )

    # ----- Check: subcortical magnitudes within range -----
    sub_mags = flat_mags[flat_pops == 1]
    if sub_mags.size > 0:
        slow, shigh = subcortical_magnitude_range_nAm
        eps = max(1e-12, 1e-12 * shigh)
        in_range = (
            (sub_mags.min() >= slow - eps)
            and (sub_mags.max() <= shigh + eps)
        )
        checks.append(CheckResult(
            name='subcortical_magnitude_range',
            passed=bool(in_range),
            detail=(
                f"Subcortical magnitudes: [{sub_mags.min():.2f}, "
                f"{sub_mags.max():.2f}] nA*m, expected [{slow}, {shigh}]"
            ),
            statistics={
                'min_nAm': float(sub_mags.min()),
                'max_nAm': float(sub_mags.max()),
                'mean_nAm': float(sub_mags.mean()),
                'expected_range_nAm': [slow, shigh],
            },
        ))
        _append_magnitude_ks(
            diagnostics, 'subcortical', sub_mags, slow, shigh,
            magnitude_distribution,
        )

    # ----- Check: SNR distribution -----
    if noise_model == 'absolute_floor':
        # SNR is not targeted; it emerges from source amplitude and depth. There
        # is no expected band, so we only require the emergent SNRs to be finite
        # and report their distribution for inspection.
        snr_finite = bool(np.all(np.isfinite(snr_db)))
        checks.append(CheckResult(
            name='snr_emergent',
            passed=snr_finite,
            detail=(
                f"emergent SNR: [{snr_db.min():.2f}, {snr_db.max():.2f}] dB, "
                f"mean {snr_db.mean():.2f} (absolute_floor; not targeted)"
            ),
            statistics={
                'min_dB': float(snr_db.min()),
                'max_dB': float(snr_db.max()),
                'mean_dB': float(snr_db.mean()),
                'noise_model': 'absolute_floor',
            },
        ))
    else:
        snr_lo, snr_hi = snr_db_range
        eps = max(1e-10, 1e-12 * max(abs(snr_lo), abs(snr_hi), 1.0))
        snr_in_range = (snr_db.min() >= snr_lo - eps) and (snr_db.max() <= snr_hi + eps)
        checks.append(CheckResult(
            name='snr_range',
            passed=bool(snr_in_range),
            detail=(
                f"SNR: [{snr_db.min():.2f}, {snr_db.max():.2f}] dB, "
                f"expected [{snr_lo}, {snr_hi}]"
            ),
            statistics={
                'min_dB': float(snr_db.min()),
                'max_dB': float(snr_db.max()),
                'mean_dB': float(snr_db.mean()),
                'expected_range_dB': [snr_lo, snr_hi],
            },
        ))
        if snr_db.size >= 20:
            uniformized = (snr_db - snr_lo) / (snr_hi - snr_lo)
            ks = kstest(uniformized, 'uniform')
            diagnostics.append(DiagnosticResult(
                name='target_snr_uniformity_ks',
                detail=(
                    f"Target-SNR uniformity KS D={ks.statistic:.4f}, "
                    f"p={ks.pvalue:.3g}"
                ),
                statistics={
                    'ks_D': float(ks.statistic),
                    'ks_p': float(ks.pvalue),
                    'n': int(snr_db.size),
                },
            ))

    # ----- Checks: orientation norms and low-order S2 moments -----
    if orientations_world is not None:
        if orientations_world.ndim == 3:
            valid = ~np.isnan(orientations_world[..., 0])
            flat_orient = orientations_world[valid]
        else:
            flat_orient = orientations_world

        if flat_orient.size > 0:
            norms = np.linalg.norm(flat_orient, axis=1)
            norm_error = float(np.max(np.abs(norms - 1.0)))
            norm_passed = bool(np.all(np.isfinite(norms)) and norm_error < 1e-12)
        else:
            norm_error = 0.0
            norm_passed = True

        checks.append(CheckResult(
            name='orientation_unit_norm',
            passed=norm_passed,
            detail=(
                f"Maximum |norm-1|={norm_error:.3e} over "
                f"{flat_orient.shape[0]} active orientations"
            ),
            statistics={
                'max_abs_norm_error': norm_error,
                'n': int(flat_orient.shape[0]),
            },
        ))

    # A patch records the signed radial normal of its centre. Under the region
    # policy the centres of a split cover only its own sectors, so their
    # directions are not uniform over the sphere and the isotropy moments do
    # not apply; the radial alignment itself is checked instead.
    region_patch = source_mode == 'patch' and split_policy == 'region'
    if orientations_world is not None and region_patch and flat_orient.size > 0:
        centre_unit = positions_xyz / np.linalg.norm(positions_xyz, axis=1, keepdims=True)
        alignment = np.abs(np.sum(flat_orient * centre_unit, axis=1))
        radial_error = float(np.max(np.abs(alignment - 1.0)))
        checks.append(CheckResult(
            name='patch_orientation_radial',
            passed=bool(radial_error < 1e-12),
            detail=(
                f"Maximum | |orientation . centre normal| - 1 | = {radial_error:.3e} "
                f"over {flat_orient.shape[0]} patches"
            ),
            statistics={
                'max_abs_alignment_error': radial_error,
                'n': int(flat_orient.shape[0]),
                'scope': (
                    'region policy: patch centres cover only the sectors of their split, '
                    'so the uniform-sphere moment check is not applicable'
                ),
            },
        ))

    if (orientations_world is not None and orientation_mode == 'uniform_s2'
            and flat_orient.size > 0 and not region_patch):
        n = flat_orient.shape[0]
        means = flat_orient.mean(axis=0)
        stds = flat_orient.std(axis=0)
        target_std = np.sqrt(1.0 / 3.0)
        # Every Cartesian component of a uniform S2 vector is Uniform[-1, 1].
        # Its mean has SE sqrt(1/(3n)); applying the delta method to its sample
        # standard deviation gives SE sqrt(1/(15n)). For off-diagonal products,
        # E[x*y]=0 and E[x^2*y^2]=1/15, giving the same SE sqrt(1/(15n)).
        mean_tol = 5.0 * np.sqrt(1.0 / (3.0 * n))
        std_tol = 5.0 * np.sqrt(1.0 / (15.0 * n))
        cross = np.array([
            np.mean(flat_orient[:, 0] * flat_orient[:, 1]),
            np.mean(flat_orient[:, 0] * flat_orient[:, 2]),
            np.mean(flat_orient[:, 1] * flat_orient[:, 2]),
        ])
        passed = bool(
            np.all(np.abs(means) <= mean_tol)
            and np.all(np.abs(stds - target_std) <= std_tol)
            and np.all(np.abs(cross) <= std_tol)
        )

        checks.append(CheckResult(
            name='orientation_s2_moments',
            passed=passed,
            detail=(
                f"Cartesian means={means.tolist()}, stds={stds.tolist()}, "
                f"cross-moments={cross.tolist()}; five-SE tolerances "
                f"mean={mean_tol:.4f}, std/cross={std_tol:.4f}, n={n}"
            ),
            statistics={
                'means_xyz': means,
                'stds_xyz': stds,
                'cross_moments_xy_xz_yz': cross,
                'target_std': float(target_std),
                'five_se_mean': float(mean_tol),
                'five_se_std_cross': float(std_tol),
                'n': int(n),
                'scope': 'low-order moment check, not a complete uniformity test',
            },
        ))
        polar_u = (flat_orient[:, 2] + 1.0) / 2.0
        azimuth_u = (
            np.arctan2(flat_orient[:, 1], flat_orient[:, 0]) % (2.0 * np.pi)
        ) / (2.0 * np.pi)
        polar_ks = kstest(polar_u, 'uniform')
        azimuth_ks = kstest(azimuth_u, 'uniform')
        diagnostics.append(DiagnosticResult(
            name='orientation_s2_uniformity_ks',
            detail=(
                f"S2 polar KS p={polar_ks.pvalue:.3g}; azimuth KS "
                f"p={azimuth_ks.pvalue:.3g}"
            ),
            statistics={
                'polar_ks_D': float(polar_ks.statistic),
                'polar_ks_p': float(polar_ks.pvalue),
                'azimuth_ks_D': float(azimuth_ks.statistic),
                'azimuth_ks_p': float(azimuth_ks.pvalue),
                'n': int(n),
            },
        ))

    # ----- Check: exact position-index disjointness -----
    # Assigned pools can be checked independently. When serialized indices are
    # supplied, the pipeline also audits the indices read back from output
    # files instead of only rechecking the split constructor.
    if (
        split_policy != 'shared'
        and train_position_indices is not None
        and val_position_indices is not None
        and test_position_indices is not None
    ):
        tr = set(np.unique(train_position_indices).tolist())
        va = set(np.unique(val_position_indices).tolist())
        te = set(np.unique(test_position_indices).tolist())
        tr_va = tr & va
        tr_te = tr & te
        va_te = va & te
        leakage = len(tr_va) + len(tr_te) + len(va_te)
        checks.append(CheckResult(
            name='split_position_disjoint',
            passed=(leakage == 0),
            detail=(
                f"Position-level overlap: train&val={len(tr_va)}, "
                f"train&test={len(tr_te)}, val&test={len(va_te)}"
            ),
            statistics={
                'n_train_positions': len(tr),
                'n_val_positions': len(va),
                'n_test_positions': len(te),
                'overlap_train_val': len(tr_va),
                'overlap_train_test': len(tr_te),
                'overlap_val_test': len(va_te),
            },
        ))

    if assigned_position_indices is not None or serialized_position_indices is not None:
        expected_names = ('train', 'val', 'test')
        if assigned_position_indices is None or serialized_position_indices is None:
            raise ValueError(
                "assigned_position_indices and serialized_position_indices "
                "must be provided together"
            )
        if (set(assigned_position_indices) != set(expected_names)
                or set(serialized_position_indices) != set(expected_names)):
            raise ValueError(
                "serialized split audit requires train, val, and test mappings"
            )

        assigned = {
            name: set(np.asarray(assigned_position_indices[name]).ravel().tolist())
            for name in expected_names
        }
        written = {
            name: set(
                np.asarray(serialized_position_indices[name]).ravel()[
                    np.asarray(serialized_position_indices[name]).ravel() >= 0
                ].tolist()
            )
            for name in expected_names
        }
        overlaps = {
            'train_val': written['train'] & written['val'],
            'train_test': written['train'] & written['test'],
            'val_test': written['val'] & written['test'],
        }
        outside = {
            name: written[name] - assigned[name] for name in expected_names
        }
        nonempty = all(written[name] for name in expected_names)
        passed = (
            nonempty
            and (split_policy == 'shared' or all(not values for values in overlaps.values()))
            and all(len(values) == 0 for values in outside.values())
        )
        checks.append(CheckResult(
            name='serialized_split_position_audit',
            passed=passed,
            detail=(
                "Serialized used-position overlap: "
                f"train&val={len(overlaps['train_val'])}, "
                f"train&test={len(overlaps['train_test'])}, "
                f"val&test={len(overlaps['val_test'])}; "
                "outside assigned pool: "
                + ', '.join(
                    f"{name}={len(outside[name])}" for name in expected_names
                )
                + f"; all splits nonempty={nonempty}"
            ),
            statistics={
                'serialized_unique_positions': {
                    name: len(written[name]) for name in expected_names
                },
                'overlap_train_val': len(overlaps['train_val']),
                'overlap_train_test': len(overlaps['train_test']),
                'overlap_val_test': len(overlaps['val_test']),
                'outside_assigned_pool': {
                    name: len(outside[name]) for name in expected_names
                },
                'all_splits_nonempty': nonempty,
            },
        ))

    # ----- Check: clean scalp-potential summary is numerically valid -----
    if peak_uV is None and eeg_signal is not None:
        peak_uV = np.max(np.abs(eeg_signal), axis=(-1, -2)) * 1e6
    if peak_uV is not None and np.asarray(peak_uV).size > 0:
        peak_uV = np.asarray(peak_uV)
        median_peak = float(np.median(peak_uV))
        peaks_valid = bool(
            np.all(np.isfinite(peak_uV)) and np.all(peak_uV > 0.0)
        )
        checks.append(CheckResult(
            name='scalp_potential_finite_nonzero',
            passed=peaks_valid,
            detail=(
                f"Clean-signal peak |V|: median={median_peak:.2f} uV, "
                f"finite={bool(np.all(np.isfinite(peak_uV)))}, "
                f"all_nonzero={bool(np.all(peak_uV > 0.0))}"
            ),
            statistics={
                'peak_uV_p10': float(np.percentile(peak_uV, 10)),
                'peak_uV_p50': float(np.percentile(peak_uV, 50)),
                'peak_uV_p90': float(np.percentile(peak_uV, 90)),
                'n': int(peak_uV.size),
            },
        ))

    return DistributionValidationResult(checks=checks, diagnostics=diagnostics)
