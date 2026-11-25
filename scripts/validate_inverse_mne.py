"""
Cross-implementation source recovery under matched spherical assumptions.

End-to-end validation of the generator: known single-dipole sources are round-
tripped through independent, community-standard localization pipelines. The
script generates a point-mode dataset with spherical_eeg_sim, writes it to HDF5
and reloads it through the package's own MNE export path (dataset_to_epochs),
then localizes every reloaded sample and compares the recovered position to the
ground-truth position, signed orientation, and instantaneous current moment
stored on disk.

The primary validation is an equivalent-current-dipole (ECD) fit on a matching
four-layer sphere -- a focal, continuous-position solver that is the natural
match to the single-dipole ground truth. The
optional ``--distributed`` panel is exploratory and adds a distributed linear
inverse (sLORETA) on a whole-brain volumetric grid. No claim about its bias or
resolution is made until a particular run is archived and reported.

------------------------------------------------------------------------------
MATCHED-SPHERE CROSS-IMPLEMENTATION CHECK
------------------------------------------------------------------------------
The scalp data are synthesized with the package's exact four-sphere series
(de Munck & Peters 1993). MNE-Python inverts them using its own multilayer-sphere
EEG forward, which fits the Berg-Scherg three-dipole equivalent-source
approximation (Berg & Scherg 1994) -- an independent formulation. Close recovery
therefore tests the generated scalp fields and labels against a different
implementation under matched spherical geometry and conductivity assumptions.
The scope is recovery within that model class, including end-to-end coordinate
handling.

Two design choices make the check robust and unambiguous:
  * The equivalent-current-dipole fit is the natural match to the single-dipole
    ground truth. It returns a continuous 3-D position (no source-grid
    discretization floor) and independently fits orientation and amplitude, so
    all three recovered quantities can be checked against serialized labels.
  * Everything stays in one origin-centred head frame: the sphere centre is
    pinned to the origin and the montage is origin-centred, so no head<->MRI
    transform enters the comparison.

------------------------------------------------------------------------------
USAGE
------------------------------------------------------------------------------
    python scripts/validate_inverse_mne.py --quick --no-sweep
    python scripts/validate_inverse_mne.py
    python scripts/validate_inverse_mne.py --n-test 300 --snr-db 35 --figure recovery.png
    python scripts/validate_inverse_mne.py --distributed --figure recovery.png
    python scripts/validate_inverse_mne.py --write-canonical

Ordinary runs save a new bundle under validation_runs/. The quick command
writes a 60-dipole CSV and summary there. The archived CSV, summary and figure
are regenerated together only with --write-canonical and publication settings.

Requires MNE-Python and h5py:
    pip install "spherical_eeg_sim[all]"      # or: pip install mne h5py

Outputs a source-radius-stratified recovery table, a per-sample metrics CSV,
and an SNR-degradation sweep. The CSV includes true and estimated positions,
true signed and estimated orientation vectors, true and estimated instantaneous
moments, and the resulting errors. With ``--distributed`` it adds an sLORETA
recovery table (peak error, grid-resolution floor, a chance baseline, and the
systematic bias) and appends the per-sample sLORETA metrics to the same CSV.
With ``--figure`` it writes error-vs-depth and SNR-degradation panels, plus an
sLORETA-vs-depth panel when the distributed panel is on.
"""

import argparse
import contextlib
import csv
import hashlib
import io
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts._validation_outputs import (
    ValidationOutputs, add_output_options, relative_reference,
)
from spherical_eeg_sim.provenance import environment_snapshot

# Validation defaults. These are experiment settings for the recovery study,
# not head-model parameters (the head model comes from Config; see docs).
DEFAULT_MONTAGE = 'standard_1010_69.csv'   # package default recording montage
# A high-SNR diagnostic isolates forward/inverse agreement from noise-limited
# localization. Controlled-SNR simulation follows Krol et al. (2018) and Sun et
# al. (2022); 35 dB is the declared validation operating point, not a clinical
# SNR estimate. The generated target is drawn within +/-0.5 dB of this centre.
DEFAULT_SNR_DB = 35.0
DEFAULT_N_TEST = 300
DEFAULT_SWEEP_DB = [5, 10, 15, 20, 25, 30, 35]
DEFAULT_SWEEP_N = 150
DEFAULT_POOL_CORTICAL = 4000
DEFAULT_POOL_SUB = 800
MIN_DIST_MM = 0.5   # allows the full cortical shell (inner sphere - 0.5 mm)

# Distributed-inverse (sLORETA) panel settings. The grid spacing sets the
# discretization floor (roughly half the spacing); r_safe keeps grid vertices a
# safe margin inside the innermost sphere, where the sphere forward is finite and
# well conditioned (the boundary itself is singular); lambda2 is the standard
# regularization for an assumed SNR of 3.
DIST_GRID_MM = 6.0
DIST_RSAFE_M = 0.0795
DIST_LAMBDA2 = 1.0 / 9.0


def generate_test_split(snr_db, n_test, montage, seed, pool_cortical, pool_sub,
                        workdir, tag=None, reference_mode='none', snr_range=None, head=None):
    """Generate one point-mode dataset via the full pipeline; return the test file.

    A narrow SNR band centred on ``snr_db`` is used (the configuration validator
    requires a non-degenerate range). Only the test split is localized, so the
    train/val splits are kept minimal. When ``tag`` is given the dataset is
    written to a directory keyed by that tag rather than by SNR, so a run whose
    SNR coincides with a sweep point cannot be overwritten by the sweep.
    ``snr_range`` replaces the narrow band by a range in dB (``snr_db`` is then
    unused and ``tag`` is required), and ``head`` replaces the default head model
    (radii in m, conductivities in S/m). Returns (test_h5_path, radii_m, sigmas).
    """
    from spherical_eeg_sim import Config, run_pipeline

    if snr_range is not None and tag is None:
        raise ValueError('a dataset with an SNR range needs a tag')
    cfg = Config()
    if head is not None:
        cfg.head.radii_m, cfg.head.conductivities_S_per_m = list(head[0]), list(head[1])
    cfg.source_mode = 'point'
    cfg.noise.noise_model = 'snr_target'
    cfg.noise.snr_db_range = ([snr_db - 0.5, snr_db + 0.5] if snr_range is None
                              else [float(snr_range[0]), float(snr_range[1])])
    cfg.sources.n_cortical_positions = pool_cortical
    cfg.sources.n_subcortical_positions = pool_sub
    cfg.dataset.n_train = max(40, n_test // 4)
    cfg.dataset.n_val = 20
    cfg.dataset.n_test = n_test
    cfg.electrode_file = montage
    cfg.output.reference_mode = reference_mode
    cfg.output.eeg_normalization = 'none'
    subdir = f'ds_{tag}' if tag is not None else f'ds_snr{snr_db:g}'
    cfg.output_dir = str(Path(workdir) / subdir)
    cfg.master_seed = seed
    cfg.output.diagnostic_plots = False

    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        run_pipeline(cfg)

    return (
        Path(cfg.output_dir) / 'test.h5',
        np.asarray(cfg.head.radii_m, dtype=float),
        np.asarray(cfg.head.conductivities_S_per_m, dtype=float),
    )


def recover(h5_path, radii, sigmas, min_dist_mm=MIN_DIST_MM, tol=None):
    """Localize every epoch with an MNE ECD fit on a matching origin-centred sphere.

    The dataset is reloaded through the package's own ``dataset_to_epochs`` path,
    so the montage (origin-centred, metres) and the ground-truth position travel
    with the data. Each epoch is average-referenced and the equivalent-current
    dipole is fitted at the peak global-field-power time sample (the point-source
    topography is rank-1 in time, so any high-power sample carries the full
    spatial pattern). ``tol`` is the final trust-region radius of the dipole fit in
    metres; without it the default of MNE-Python applies. Returns a dict of
    per-sample arrays.
    """
    import mne
    mne.set_log_level('ERROR')
    from spherical_eeg_sim.io.mne_export import dataset_to_epochs

    epochs = dataset_to_epochs(str(h5_path))
    info = epochs.info
    data = epochs.get_data(copy=True)

    sphere = mne.make_sphere_model(
        r0=(0.0, 0.0, 0.0),
        head_radius=float(radii[-1]),
        relative_radii=tuple(radii / radii[-1]),
        sigmas=tuple(sigmas),
        verbose=False,
    )
    cov = mne.make_ad_hoc_cov(info, verbose=False)
    tolerance = {} if tol is None else dict(tol=float(tol))

    import h5py
    with h5py.File(h5_path, 'r') as f:
        true_xyz = f['ground_truth/positions_xyz'][:]
        true_orientation = f['ground_truth/orientations_world'][:]
        source_courses = f['ground_truth/source_time_course'][:]
        magnitudes_nAm = f['ground_truth/magnitudes_nAm'][:]
        snr_db = f['ground_truth/snr_db'][:]
        population = f['ground_truth/population'][:]
    n = data.shape[0]
    err_mm = np.empty(n)
    gof = np.empty(n)
    orientation_error_deg = np.empty(n)
    estimated_orientation = np.empty((n, 3))
    instantaneous_true_orientation = np.empty((n, 3))
    estimated_xyz = np.empty((n, 3))
    fit_sample = np.empty(n, dtype=np.int64)
    true_moment_nAm = np.empty(n)
    estimated_moment_nAm = np.empty(n)
    for i in range(n):
        ev = mne.EvokedArray(data[i], info, tmin=0.0, verbose=False)
        ev.set_eeg_reference('average', projection=True, verbose=False)
        t_peak = int(np.argmax(ev.data.std(axis=0)))
        ev.crop(ev.times[t_peak], ev.times[t_peak])
        dip = mne.fit_dipole(ev, cov, sphere, min_dist=min_dist_mm, verbose=False, **tolerance)[0]
        fit_sample[i] = t_peak
        estimated_xyz[i] = dip.pos[0]
        err_mm[i] = float(np.linalg.norm(dip.pos[0] - true_xyz[i]) * 1000.0)
        gof[i] = float(dip.gof[0])
        estimated_orientation[i] = dip.ori[0]
        true_moment_nAm[i] = (
            magnitudes_nAm[i] * abs(source_courses[i, t_peak])
        )
        estimated_moment_nAm[i] = float(dip.amplitude[0] * 1e9)
        instantaneous_true = true_orientation[i] * np.sign(
            source_courses[i, t_peak]
        )
        instantaneous_true_orientation[i] = instantaneous_true
        dot = float(np.clip(
            np.dot(estimated_orientation[i], instantaneous_true), -1.0, 1.0
        ))
        orientation_error_deg[i] = np.degrees(np.arccos(dot))

    source_radius_mm = np.linalg.norm(true_xyz, axis=1) * 1000.0
    return {
        'source_radius_mm': source_radius_mm,
        'source_depth_mm': float(radii[0] * 1000.0) - source_radius_mm,
        'err_mm': err_mm,
        'gof': gof,
        'orientation_error_deg': orientation_error_deg,
        'true_xyz': true_xyz,
        'estimated_xyz': estimated_xyz,
        'true_orientation_at_fit': instantaneous_true_orientation,
        'estimated_orientation': estimated_orientation,
        'fit_sample': fit_sample,
        'true_moment_nAm': true_moment_nAm,
        'estimated_moment_nAm': estimated_moment_nAm,
        'moment_absolute_error_nAm': np.abs(
            estimated_moment_nAm - true_moment_nAm
        ),
        'moment_relative_error': np.abs(
            estimated_moment_nAm - true_moment_nAm
        ) / true_moment_nAm,
        'snr_db': snr_db,
        'population': population,
    }


def build_sloreta_operator(info, radii, sigmas, grid_mm=DIST_GRID_MM,
                           r_safe_m=DIST_RSAFE_M):
    """Build an sLORETA inverse operator on an interior volumetric grid.

    The source space is a regular Cartesian lattice confined to the brain
    interior (``0.005 m <= r <= r_safe_m``), a margin inside the innermost sphere
    where the sphere forward stays finite and well conditioned. No depth weighting
    is applied: sLORETA supplies its own standardization. The orientation is free
    and the sphere is origin-centred, matching the origin-centred montage that
    travels with the reloaded data, so the operator and the data live in one head
    frame and no head-to-MRI transform is needed. Returns
    (inverse_operator, source_positions).
    """
    import mne
    mne.set_log_level('ERROR')

    step = grid_mm / 1000.0
    axis = np.arange(-r_safe_m, r_safe_m + step, step)
    gx, gy, gz = np.meshgrid(axis, axis, axis, indexing='ij')
    pts = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    radius = np.linalg.norm(pts, axis=1)
    pts = pts[(radius >= 0.005) & (radius <= r_safe_m)]

    src = mne.setup_volume_source_space(
        pos=dict(rr=pts.astype(float), nn=np.tile([0.0, 0.0, 1.0], (len(pts), 1))),
        verbose=False,
    )
    sphere = mne.make_sphere_model(
        r0=(0.0, 0.0, 0.0),
        head_radius=float(radii[-1]),
        relative_radii=tuple(radii / radii[-1]),
        sigmas=tuple(sigmas),
        verbose=False,
    )
    fwd = mne.make_forward_solution(
        info, trans=None, src=src, bem=sphere, eeg=True, meg=False, verbose=False,
    )
    cov = mne.make_ad_hoc_cov(info, verbose=False)
    inv = mne.minimum_norm.make_inverse_operator(
        info, fwd, cov, loose=1.0, depth=None, verbose=False,
    )
    return inv, fwd['source_rr']


def recover_distributed(h5_path, radii, sigmas, grid_mm=DIST_GRID_MM,
                        lambda2=DIST_LAMBDA2):
    """Localize every epoch with a distributed sLORETA inverse and score the peak.

    The dataset is reloaded through the package's own ``dataset_to_epochs`` path
    and average-referenced up front, so the forward operator, the covariance, and
    every evoked share one reference convention. Each epoch is localized at its
    peak global-field-power sample (the point-source topography is rank-1 in
    time), and the grid vertex carrying the largest absolute current amplitude is
    taken as the position estimate. Alongside the per-sample peak error the
    function returns the grid-resolution floor (distance to the nearest grid
    vertex), a random-vertex chance baseline, and the systematic bias, which
    together make the peak error interpretable. Returns a dict of per-sample
    arrays and scalar context.
    """
    import mne
    mne.set_log_level('ERROR')
    from scipy.spatial import cKDTree
    from spherical_eeg_sim.io.mne_export import dataset_to_epochs

    epochs = dataset_to_epochs(str(h5_path))
    epochs.set_eeg_reference('average', projection=True, verbose=False)
    info = epochs.info
    data = epochs.get_data(copy=True)
    import h5py
    with h5py.File(h5_path, 'r') as f:
        true_xyz = f['ground_truth/positions_xyz'][:]

    inv, src_rr = build_sloreta_operator(info, radii, sigmas, grid_mm)

    n = data.shape[0]
    est = np.empty((n, 3))
    for i in range(n):
        ev = mne.EvokedArray(data[i], info, tmin=0.0, verbose=False)
        ev.apply_proj(verbose=False)
        t_peak = int(np.argmax(ev.data.std(axis=0)))
        ev.crop(ev.times[t_peak], ev.times[t_peak])
        stc = mne.minimum_norm.apply_inverse(
            ev, inv, lambda2=lambda2, method='sLORETA', verbose=False,
        )
        est[i] = src_rr[int(np.argmax(np.abs(stc.data[:, 0])))]

    err_mm = np.linalg.norm(est - true_xyz, axis=1) * 1000.0
    floor_mm = cKDTree(src_rr).query(true_xyz)[0] * 1000.0
    rng = np.random.default_rng(0)
    chance_mm = np.linalg.norm(
        src_rr[rng.integers(len(src_rr), size=n)] - true_xyz, axis=1) * 1000.0
    bias_mm = float(np.linalg.norm((est - true_xyz).mean(axis=0)) * 1000.0)

    source_radius_mm = np.linalg.norm(true_xyz, axis=1) * 1000.0
    return {
        'source_radius_mm': source_radius_mm,
        'source_depth_mm': float(radii[0] * 1000.0) - source_radius_mm,
        'err_mm': err_mm,
        'floor_mm': floor_mm,
        'chance_mm': chance_mm,
        'bias_mm': bias_mm,
        'n_grid': int(len(src_rr)),
    }


def print_radius_table(res, edges=(30, 45, 55, 65, 72, 76, 80)):
    """Source-radius-stratified recovery summary."""
    d = res['source_radius_mm']
    print(f"\n{'radius band (mm)':>16} {'n':>4} {'median_mm':>10} "
          f"{'p90_mm':>8} {'max_mm':>8} {'median_gof%':>12}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (d >= lo) & (d < hi)
        if not m.any():
            continue
        e = res['err_mm'][m]
        print(f"{f'[{lo}, {hi})':>16} {int(m.sum()):4d} {np.median(e):10.2f} "
              f"{np.percentile(e, 90):8.2f} {e.max():8.2f} "
              f"{np.median(res['gof'][m]):12.2f}")


def print_distributed_summary(dres, edges=(30, 45, 55, 65, 72, 76, 80)):
    """Distributed-inverse (sLORETA) recovery summary with resolution context."""
    e = dres['err_mm']
    floor = dres['floor_mm']
    med_chance = float(np.median(dres['chance_mm']))
    med_err = float(np.median(e))
    ratio = med_chance / med_err if med_err > 0 else float('inf')
    print(f"\nDistributed sLORETA recovery over {len(e)} dipoles "
          f"({dres['n_grid']} interior grid vertices):")
    print(f"  peak error:  median = {med_err:.2f} mm   "
          f"p90 = {np.percentile(e, 90):.2f} mm   max = {e.max():.2f} mm")
    print(f"  grid floor (nearest vertex):  median = {np.median(floor):.2f} mm")
    print(f"  chance (random vertex):  median = {med_chance:.2f} mm   "
          f"-> {ratio:.1f}x better than chance")
    print(f"  systematic bias |mean(estimate - truth)| = {dres['bias_mm']:.2f} mm")

    d = dres['source_radius_mm']
    print(f"\n{'radius band (mm)':>16} {'n':>4} {'median_mm':>10} "
          f"{'p90_mm':>8} {'max_mm':>8} {'floor_mm':>9}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (d >= lo) & (d < hi)
        if not m.any():
            continue
        eb = e[m]
        print(f"{f'[{lo}, {hi})':>16} {int(m.sum()):4d} {np.median(eb):10.2f} "
              f"{np.percentile(eb, 90):8.2f} {eb.max():8.2f} "
              f"{np.median(floor[m]):9.2f}")


def write_csv(res, path, dres=None):
    """Per-sample metrics CSV.

    The base columns describe the ECD recovery. When distributed results are
    supplied, two sLORETA columns are appended; both panels localize the same
    reloaded dipoles in the same order, so each row describes one dipole under
    both methods. Without ``dres`` the output is identical to the ECD-only run.
    """
    if dres is not None and len(dres['err_mm']) != len(res['err_mm']):
        raise ValueError('ECD and distributed panels localized different numbers '
                         'of dipoles; per-row correspondence would be invalid.')
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        header = [
            'sample_index', 'fit_sample', 'source_radius_mm', 'source_depth_mm',
            'true_x_m', 'true_y_m', 'true_z_m',
            'estimated_x_m', 'estimated_y_m', 'estimated_z_m',
            'err_mm', 'gof_pct', 'orientation_error_deg',
            'true_moment_nAm', 'estimated_moment_nAm',
            'moment_absolute_error_nAm', 'moment_relative_error',
            'true_orientation_x', 'true_orientation_y', 'true_orientation_z',
            'estimated_orientation_x', 'estimated_orientation_y',
            'estimated_orientation_z', 'snr_db', 'population',
        ]
        if dres is not None:
            header += ['sloreta_err_mm', 'sloreta_floor_mm']
        w.writerow(header)
        pop_name = {0: 'cortical', 1: 'subcortical'}
        for i in range(len(res['err_mm'])):
            row = [
                i, int(res['fit_sample'][i]),
                f"{res['source_radius_mm'][i]:.3f}",
                f"{res['source_depth_mm'][i]:.3f}",
                f"{res['true_xyz'][i, 0]:.9f}",
                f"{res['true_xyz'][i, 1]:.9f}",
                f"{res['true_xyz'][i, 2]:.9f}",
                f"{res['estimated_xyz'][i, 0]:.9f}",
                f"{res['estimated_xyz'][i, 1]:.9f}",
                f"{res['estimated_xyz'][i, 2]:.9f}",
                f"{res['err_mm'][i]:.4f}",
                f"{res['gof'][i]:.4f}",
                f"{res['orientation_error_deg'][i]:.5f}",
                f"{res['true_moment_nAm'][i]:.8f}",
                f"{res['estimated_moment_nAm'][i]:.8f}",
                f"{res['moment_absolute_error_nAm'][i]:.8f}",
                f"{res['moment_relative_error'][i]:.8f}",
                f"{res['true_orientation_at_fit'][i, 0]:.8f}",
                f"{res['true_orientation_at_fit'][i, 1]:.8f}",
                f"{res['true_orientation_at_fit'][i, 2]:.8f}",
                f"{res['estimated_orientation'][i, 0]:.8f}",
                f"{res['estimated_orientation'][i, 1]:.8f}",
                f"{res['estimated_orientation'][i, 2]:.8f}",
                f"{res['snr_db'][i]:.3f}",
                pop_name.get(int(res['population'][i]), int(res['population'][i])),
            ]
            if dres is not None:
                row += [f"{dres['err_mm'][i]:.4f}", f"{dres['floor_mm'][i]:.4f}"]
            w.writerow(row)


def make_figure(res, sweep_db, sweep_median_mm, path, dres=None):
    """ECD error-vs-depth and SNR-degradation panels, plus an optional sLORETA panel."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping figure.")
        return False

    edges = np.array([30, 45, 55, 65, 72, 76, 80], dtype=float)

    def binned_median(depth, err):
        centres, med = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (depth >= lo) & (depth < hi)
            if m.any():
                centres.append((lo + hi) / 2.0)
                med.append(np.median(err[m]))
        return centres, med

    ncols = 3 if dres is not None else 2
    fig, ax = plt.subplots(1, ncols, figsize=(5.5 * ncols, 4.2))

    d, e = res['source_radius_mm'], res['err_mm']
    ax[0].scatter(d, e, s=14, alpha=0.5, edgecolor='none')
    centres, med = binned_median(d, e)
    ax[0].plot(centres, med, 'o-', color='crimson', label='binned median')
    ax[0].set_xlabel('source radius (mm)')
    ax[0].set_ylabel('localization error (mm)')
    ax[0].set_title('ECD recovery vs source depth')
    ax[0].legend(fontsize=8)

    if sweep_db:
        ax[1].plot(sweep_db, sweep_median_mm, 's-', color='navy')
        ax[1].set_yscale('log')
        ax[1].set_xlabel('SNR (dB)')
        ax[1].set_ylabel('median localization error (mm)')
        ax[1].set_title('ECD recovery vs SNR')
        ax[1].grid(True, which='both', alpha=0.3)
    else:
        ax[1].axis('off')

    if dres is not None:
        dd, de = dres['source_radius_mm'], dres['err_mm']
        ax[2].scatter(dd, de, s=14, alpha=0.5, edgecolor='none', color='seagreen')
        centres_s, med_s = binned_median(dd, de)
        ax[2].plot(centres_s, med_s, 'o-', color='darkorange', label='binned median')
        ax[2].axhline(np.median(dres['floor_mm']), ls='--', color='grey', lw=1,
                      label='grid-floor median')
        ax[2].set_xlabel('source radius (mm)')
        ax[2].set_ylabel('localization error (mm)')
        ax[2].set_title('sLORETA recovery vs source depth')
        ax[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote figure -> {path}")
    return True


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('--montage', default=DEFAULT_MONTAGE,
                    help=f'electrode montage (default {DEFAULT_MONTAGE})')
    ap.add_argument('--snr-db', type=float, default=DEFAULT_SNR_DB,
                    help=f'headline SNR in dB (default {DEFAULT_SNR_DB})')
    ap.add_argument('--n-test', type=int, default=DEFAULT_N_TEST,
                    help=f'number of test dipoles for the headline run (default {DEFAULT_N_TEST})')
    ap.add_argument('--seed', type=int, default=20260703, help='master seed')
    ap.add_argument('--pool-cortical', type=int, default=DEFAULT_POOL_CORTICAL,
                    help='cortical position-pool size (depth coverage)')
    ap.add_argument('--pool-sub', type=int, default=DEFAULT_POOL_SUB,
                    help='subcortical position-pool size')
    ap.add_argument('--min-dist', type=float, default=MIN_DIST_MM,
                    help='ECD min distance from inner-sphere surface (mm)')
    ap.add_argument(
        '--reference-mode', choices=('none', 'average', 'cz'), default='none',
        help='stored dataset reference before MNE import (default none)',
    )
    ap.add_argument('--sweep', type=float, nargs='*', default=DEFAULT_SWEEP_DB,
                    help='SNR values (dB) for the degradation sweep')
    ap.add_argument('--sweep-n', type=int, default=DEFAULT_SWEEP_N,
                    help='samples per SNR-sweep point')
    ap.add_argument('--no-sweep', action='store_true', help='skip the SNR sweep')
    ap.add_argument('--quick', action='store_true',
                    help='reduced configuration for a smoke test')
    ap.add_argument('--distributed', action='store_true',
                    help='also run the distributed sLORETA inverse panel on the '
                         'headline dataset')
    ap.add_argument('--dist-grid-mm', type=float, default=DIST_GRID_MM,
                    help=f'sLORETA source-grid spacing in mm (default {DIST_GRID_MM:g})')
    ap.add_argument('--figure', default=None, help='path to save the figure')
    ap.add_argument('--csv', default=None, help='path for the per-sample metrics CSV')
    ap.add_argument('--summary-json', default=None,
                    help='path for the summary with the metrics CSV hash')
    add_output_options(ap)
    args = ap.parse_args()

    explicit = {
        name: getattr(args, name) for name in ('csv', 'summary_json', 'figure')
        if getattr(args, name) is not None
    }
    ordinary = {
        'csv': 'inverse_recovery_metrics.csv',
        'summary_json': 'inverse_recovery_summary.json',
    }
    if explicit:
        directory = Path(next(iter(explicit.values()))).expanduser().parent
        for key, name in ordinary.items():
            explicit.setdefault(key, directory / name)
    outputs = ValidationOutputs(
        ap, args, name='inverse',
        canonical={
            'csv': 'ecd_69ch_35db_per_sample.csv',
            'summary_json': 'ecd_69ch_35db_summary.json',
            'figure': 'ecd_69ch_35db.png',
        },
        settings={
            'montage': DEFAULT_MONTAGE, 'snr_db': DEFAULT_SNR_DB,
            'n_test': DEFAULT_N_TEST, 'seed': 20260703,
            'pool_cortical': DEFAULT_POOL_CORTICAL, 'pool_sub': DEFAULT_POOL_SUB,
            'min_dist': MIN_DIST_MM, 'reference_mode': 'none',
            'sweep': DEFAULT_SWEEP_DB, 'sweep_n': DEFAULT_SWEEP_N,
            'no_sweep': False, 'quick': False, 'distributed': False,
            'dist_grid_mm': DIST_GRID_MM,
        },
        explicit=explicit,
        ordinary=ordinary,
    )

    if args.quick:
        args.n_test = min(args.n_test, 60)
        args.sweep = [10, 20, 35]
        args.sweep_n = 60
        args.pool_cortical = min(args.pool_cortical, 1500)
        args.pool_sub = min(args.pool_sub, 300)

    with outputs.stage() as paths:
        _run(args, paths, outputs.paths)
    for path in outputs.paths.values():
        print(f"Wrote {path}")


def _run(args, paths, final_paths):
    with tempfile.TemporaryDirectory(prefix='.inverse-data-', dir=paths['csv'].parent.parent) as workdir:
        print("Closed-loop inverse recovery: spherical_eeg_sim -> MNE ECD fit")
        print(f"  montage={args.montage}  headline SNR={args.snr_db:g} dB  "
              f"n_test={args.n_test}  seed={args.seed}")
        print("  generating headline dataset and localizing ...")

        h5, radii, sigmas = generate_test_split(
            args.snr_db, args.n_test, args.montage, args.seed,
            args.pool_cortical, args.pool_sub, workdir, tag='headline',
            reference_mode=args.reference_mode,
        )
        res = recover(h5, radii, sigmas, args.min_dist)

        e = res['err_mm']
        print(f"\nHeadline recovery over {len(e)} dipoles at ~{args.snr_db:g} dB SNR:")
        print(f"  median = {np.median(e):.2f} mm   p90 = {np.percentile(e, 90):.2f} mm   "
              f"max = {e.max():.2f} mm")
        print(f"  median goodness-of-fit = {np.median(res['gof']):.2f}%")
        oe = res['orientation_error_deg']
        print(f"  orientation error: median = {np.median(oe):.3f} deg   "
              f"p90 = {np.percentile(oe, 90):.3f} deg   "
              f"max = {oe.max():.3f} deg")
        mre = res['moment_relative_error']
        print(f"  moment relative error: median = {np.median(mre):.4%}   "
              f"p90 = {np.percentile(mre, 90):.4%}   "
              f"max = {mre.max():.4%}")
        print_radius_table(res)

        dres = None
        if args.distributed:
            print("\n  building distributed sLORETA operator and localizing ...")
            dres = recover_distributed(h5, radii, sigmas, args.dist_grid_mm)
            print_distributed_summary(dres)

        write_csv(res, paths['csv'], dres)

        sweep_db, sweep_median, sweep_results = [], [], []
        if not args.no_sweep and args.sweep:
            print(f"\nSNR-degradation sweep ({args.sweep_n} dipoles per point):")
            print(f"{'SNR_dB':>7} {'median_mm':>10} {'p90_mm':>8} {'median_gof%':>12}")
            for s in sorted(args.sweep):
                h5s, _, _ = generate_test_split(
                    s, args.sweep_n, args.montage, args.seed,
                    args.pool_cortical, args.pool_sub, workdir,
                    reference_mode=args.reference_mode,
                )
                rs = recover(h5s, radii, sigmas, args.min_dist)
                sweep_db.append(s)
                sweep_result = {
                    'snr_db': float(s),
                    'n_test': int(len(rs['err_mm'])),
                    'position_error_median_mm': float(
                        np.median(rs['err_mm'])
                    ),
                    'position_error_p90_mm': float(
                        np.percentile(rs['err_mm'], 90)
                    ),
                    'goodness_of_fit_median_percent': float(
                        np.median(rs['gof'])
                    ),
                }
                sweep_results.append(sweep_result)
                sweep_median.append(
                    sweep_result['position_error_median_mm']
                )
                print(f"{s:7.1f} {np.median(rs['err_mm']):10.2f} "
                      f"{np.percentile(rs['err_mm'], 90):8.2f} "
                      f"{np.median(rs['gof']):12.2f}")

        figure_written = False
        if 'figure' in paths:
            figure_written = make_figure(
                res, sweep_db, sweep_median, paths['figure'], dres
            )

        if 'summary_json' in paths:
            summary = {
                'method': (
                    'MNE equivalent-current-dipole fit with independent '
                    'Berg-Scherg multilayer-sphere forward'
                ),
                'montage': args.montage,
                'n_test': int(len(e)),
                'seed': int(args.seed),
                # The Berg-Scherg parameter fit and the dipole fit of
                # MNE-Python are iterative, so the metrics below depend on
                # the platform as well as on the numerical stack recorded
                # here (see scripts/run_sphere_fit_convergence.py).
                'software_environment': environment_snapshot(),
                'target_snr_db': float(args.snr_db),
                'target_snr_db_range': [
                    float(args.snr_db - 0.5), float(args.snr_db + 0.5),
                ],
                'noise_model': 'snr_target',
                'eeg_normalization': 'none',
                'reference_mode': args.reference_mode,
                'configuration': {
                    'cortical_pool_size': args.pool_cortical,
                    'subcortical_pool_size': args.pool_sub,
                    'minimum_distance_mm': args.min_dist,
                    'sweep_snr_db': list(sweep_db),
                    'sweep_samples_per_snr': args.sweep_n if sweep_db else 0,
                    'distributed': args.distributed,
                    'distributed_grid_mm': args.dist_grid_mm if args.distributed else None,
                },
                'position_error_mm': {
                    'median': float(np.median(e)),
                    'p90': float(np.percentile(e, 90)),
                    'maximum': float(e.max()),
                },
                'orientation_error_deg': {
                    'median': float(np.median(oe)),
                    'p90': float(np.percentile(oe, 90)),
                    'maximum': float(oe.max()),
                },
                'moment_amplitude': {
                    'units': 'nA*m at fitted time sample',
                    'absolute_error_nAm': {
                        'median': float(np.median(
                            res['moment_absolute_error_nAm']
                        )),
                        'p90': float(np.percentile(
                            res['moment_absolute_error_nAm'], 90
                        )),
                        'maximum': float(
                            res['moment_absolute_error_nAm'].max()
                        ),
                    },
                    'relative_error': {
                        'median': float(np.median(mre)),
                        'p90': float(np.percentile(mre, 90)),
                        'maximum': float(mre.max()),
                    },
                    'estimated_to_true_ratio': {
                        'median': float(np.median(
                            res['estimated_moment_nAm']
                            / res['true_moment_nAm']
                        )),
                    },
                },
                'goodness_of_fit_percent': {
                    'median': float(np.median(res['gof'])),
                },
                'snr_degradation_sweep': sweep_results,
                'metrics_csv': relative_reference(
                    final_paths['csv'], final_paths['summary_json'],
                ),
                'metrics_csv_sha256': hashlib.sha256(
                    paths['csv'].read_bytes()
                ).hexdigest(),
            }
            if figure_written:
                figure_path = paths['figure']
                summary['figure'] = relative_reference(
                    final_paths['figure'], final_paths['summary_json'],
                )
                summary['figure_sha256'] = hashlib.sha256(
                    figure_path.read_bytes()
                ).hexdigest()
            summary_path = paths['summary_json']
            with summary_path.open('w', newline='\n') as f:
                json.dump(summary, f, indent=2, sort_keys=True)
                f.write('\n')

        print("\nThe Berg-Scherg implementation and the de Munck series use "
              "matched\nconcentric-sphere assumptions. The reported recovery "
              "tests implementation,\nserialization, labels, and coordinates "
              "within that model class.")

        if dres is not None:
            print("\nThe optional distributed panel reports its measured grid floor, "
                  "chance baseline, bias, and error without assigning those values "
                  "to all distributed inverse methods.")


if __name__ == '__main__':
    main()
