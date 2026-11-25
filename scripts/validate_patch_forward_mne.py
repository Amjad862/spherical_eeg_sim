"""
Cross-implementation check of patch fields under matched spherical assumptions.

Point-mode ground truth is validated by equivalent-current-dipole recovery in
scripts/validate_inverse_mne.py. Patch mode carries a different and larger
serialized target: a variable-length set of active source-pool indices, a signed
radial moment for each, and the scalp field of their weighted superposition.
None of that is exercised by a single-dipole fit, so this script checks it
directly against an independent implementation.

------------------------------------------------------------------------------
WHAT IS COMPARED
------------------------------------------------------------------------------
A patch dataset is generated with the clean signal component retained. For every
sample the script reads back, from the finalized file alone:

  * the CSR row of active global source-pool indices,
  * the CSR row of signed radial moments in nA*m,
  * the source-pool coordinates those indices refer to,
  * the stored source time course.

Each active dipole is then rebuilt as a discrete MNE source with a radial
normal, its scalp field is computed by MNE's own multilayer-sphere EEG forward,
the fields are superposed with the stored moments, referenced, and compared with
the clean EEG the package wrote. Agreement tests whether the serialized
support, signed radial moments, and time course reproduce the scalp field under
matched spherical geometry and conductivity assumptions.

------------------------------------------------------------------------------
WHY A FORWARD CHECK RATHER THAN A DISTRIBUTED INVERSE
------------------------------------------------------------------------------
A distributed linear inverse applied to patch data would mostly measure its own
regularization: the spatial spread of a minimum-norm or sLORETA solution is set
by the prior, not by the simulator, so the recovered extent would say little
about whether the generated field is correct. The forward comparison has no such
free parameter. It asks the one question that can be answered unambiguously --
does an independent implementation of the same physics produce the same scalp
field from the same serialized sources.

------------------------------------------------------------------------------
THE POINT-SOURCE CONTROL
------------------------------------------------------------------------------
MNE fits the Berg-Scherg three-dipole equivalent-source approximation to the
multilayer sphere, so it does not reproduce the exact series to machine
precision. The residual belongs to that approximation as MNE-Python fits it.
For the head model of the package, the parameter fit of MNE-Python 1.12.1 with
SciPy 1.17.1 ends at SciPy's limit of 1,000 function evaluations before it has
converged, so the residual is larger than that of a converged fit and depends
on the numerical library (scripts/run_sphere_fit_convergence.py records how
the fit ends and how the comparison reacts). The script therefore runs the
identical comparison on point-mode data.
That control establishes the level of agreement to expect, and the patch result
is interpreted only relative to it. Comparable summary distributions indicate
that no patch-specific excess is apparent above that level in this experiment.

------------------------------------------------------------------------------
USAGE
------------------------------------------------------------------------------
    python scripts/validate_patch_forward_mne.py
    python scripts/validate_patch_forward_mne.py --n-samples 60 --json out.json
    python scripts/validate_patch_forward_mne.py --write-canonical

Without an explicit path, the summary goes to a new validation_runs/ directory.

Requires MNE-Python and h5py:
    pip install "spherical_eeg_sim[all]"      # or: pip install mne h5py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._validation_outputs import ValidationOutputs, add_output_options
from spherical_eeg_sim.config import Config
from spherical_eeg_sim.pipeline import run_pipeline
from spherical_eeg_sim.provenance import environment_snapshot


def _require(module_name):
    try:
        return __import__(module_name)
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            f"{module_name} is required by this script.\n"
            f"Install it with:  pip install \"spherical_eeg_sim[all]\""
        ) from exc


def build_sphere_and_info(electrode_xyz, names, radii, sigma, sfreq, mne):
    """Origin-centred montage and matching MNE multilayer sphere."""
    info = mne.create_info(list(names), float(sfreq), 'eeg')
    montage = mne.channels.make_dig_montage(
        ch_pos={n: electrode_xyz[i] for i, n in enumerate(names)},
        coord_frame='head',
    )
    info.set_montage(montage)
    scalp = radii[-1]
    sphere = mne.make_sphere_model(
        r0=(0.0, 0.0, 0.0),
        head_radius=scalp,
        relative_radii=tuple(r / scalp for r in radii),
        sigmas=tuple(sigma),
        verbose=False,
    )
    return info, sphere


def mne_superposed_field(positions_m, normals, moments_am, info, sphere, mne):
    """Scalp topography of a set of oriented dipoles, via MNE's own forward."""
    src = mne.setup_volume_source_space(
        pos=dict(rr=np.ascontiguousarray(positions_m, dtype=np.float64),
                 nn=np.ascontiguousarray(normals, dtype=np.float64)),
        sphere_units='m',
        verbose=False,
    )
    fwd = mne.make_forward_solution(
        info, trans=None, src=src, bem=sphere, eeg=True, meg=False, verbose=False,
    )
    fwd = mne.convert_forward_solution(
        fwd, force_fixed=True, use_cps=False, verbose=False,
    )
    return fwd['sol']['data'] @ np.asarray(moments_am, dtype=np.float64)


def generate(source_mode, n_samples, out_dir, seed, head=None):
    """Generate the dataset of the comparison; ``head`` replaces the default (radii in m, conductivities in S/m)."""
    cfg = Config()
    if head is not None:
        cfg.head.radii_m, cfg.head.conductivities_S_per_m = list(head[0]), list(head[1])
    cfg.source_mode = source_mode
    cfg.master_seed = seed
    cfg.dataset.n_train = n_samples
    cfg.dataset.n_val = max(2, n_samples // 10)
    cfg.dataset.n_test = max(2, n_samples // 10)
    cfg.sources.n_cortical_positions = 35000
    cfg.sources.n_subcortical_positions = 0
    cfg.sources.subcortical_fraction = 0.0
    cfg.output_dir = str(out_dir)
    cfg.output.format = 'hdf5'
    cfg.output.save_components = True
    cfg.output.reference_mode = 'average'
    run_pipeline(cfg, log_level=logging.ERROR)
    return cfg


def relative_errors_patch(h5py, mne, root, cfg, limit):
    with h5py.File(Path(root) / 'train.h5', 'r') as f, \
         h5py.File(Path(root) / 'source_pool.h5', 'r') as pool:
        gt = f['ground_truth']
        idx_values = gt['patch_active_position_idx/values'][:]
        idx_splits = gt['patch_active_position_idx/row_splits'][:]
        mom_values = gt['patch_active_moments_nAm/values'][:]
        mom_splits = gt['patch_active_moments_nAm/row_splits'][:]
        pool_xyz = pool['positions_xyz'][:]
        signal = gt['eeg_signal_only'][:]
        courses = gt['source_time_course'][:]
        diameters = gt['patch_diameter_mm'][:]
        electrodes = f['electrodes/cartesian_xyz'][:]
        names = [n.decode() if isinstance(n, bytes) else n
                 for n in f['electrodes/names'][:]]
        sfreq = cfg.signal.sampling_freq_hz

    info, sphere = build_sphere_and_info(
        electrodes, names, list(cfg.head.radii_m),
        list(cfg.head.conductivities_S_per_m), sfreq, mne,
    )
    errors, active_counts = [], []
    n = min(limit, signal.shape[0])
    for b in range(n):
        a, z = idx_splits[b], idx_splits[b + 1]
        assert (mom_splits[b], mom_splits[b + 1]) == (a, z), \
            'patch index and moment rows must be aligned'
        rr = pool_xyz[idx_values[a:z]]
        moments = mom_values[a:z] * 1e-9
        normals = rr / np.linalg.norm(rr, axis=1, keepdims=True)
        topography = mne_superposed_field(rr, normals, moments, info, sphere, mne)
        predicted = np.outer(topography, courses[b])
        predicted -= predicted.mean(axis=0, keepdims=True)
        reference = signal[b]
        errors.append(float(np.linalg.norm(predicted - reference)
                            / np.linalg.norm(reference)))
        active_counts.append(int(z - a))
    return np.asarray(errors), np.asarray(active_counts), diameters[:n]


def relative_errors_point(h5py, mne, root, cfg, limit):
    with h5py.File(Path(root) / 'train.h5', 'r') as f:
        gt = f['ground_truth']
        signal = gt['eeg_signal_only'][:]
        courses = gt['source_time_course'][:]
        positions = gt['positions_xyz'][:]
        orientations = gt['orientations_world'][:]
        magnitudes = gt['magnitudes_nAm'][:]
        electrodes = f['electrodes/cartesian_xyz'][:]
        names = [n.decode() if isinstance(n, bytes) else n
                 for n in f['electrodes/names'][:]]
        sfreq = cfg.signal.sampling_freq_hz

    info, sphere = build_sphere_and_info(
        electrodes, names, list(cfg.head.radii_m),
        list(cfg.head.conductivities_S_per_m), sfreq, mne,
    )
    errors = []
    n = min(limit, signal.shape[0])
    for b in range(n):
        topography = mne_superposed_field(
            positions[b][None, :], orientations[b][None, :],
            np.array([magnitudes[b] * 1e-9]), info, sphere, mne,
        )
        predicted = np.outer(topography, courses[b])
        predicted -= predicted.mean(axis=0, keepdims=True)
        reference = signal[b]
        errors.append(float(np.linalg.norm(predicted - reference)
                            / np.linalg.norm(reference)))
    return np.asarray(errors)


def summarize(errors):
    return {
        'n': int(errors.size),
        'median': float(np.median(errors)),
        'p90': float(np.percentile(errors, 90)),
        'max': float(errors.max()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('--n-samples', type=int, default=60,
                    help='samples generated and compared per mode')
    ap.add_argument('--seed', type=int, default=20260703)
    ap.add_argument('--json', default=None,
                    help='write the summary to this path')
    add_output_options(ap)
    args = ap.parse_args()
    outputs = ValidationOutputs(
        ap, args, name='patch-forward',
        canonical={'json': 'patch_forward_mne_summary.json'},
        settings={'n_samples': 60, 'seed': 20260703},
        explicit={'json': args.json} if args.json is not None else None,
    )
    with outputs.stage() as paths:
        _run(args, paths['json'])
    print(f"\n  summary written to {outputs.paths['json']}")
    return 0


def _run(args, summary_path):
    mne = _require('mne')
    h5py = _require('h5py')
    mne.set_log_level('ERROR')

    with tempfile.TemporaryDirectory(prefix='.patch-data-', dir=summary_path.parent.parent) as tmp:
        patch_dir = Path(tmp) / 'patch'
        point_dir = Path(tmp) / 'point'

        print('Distributed-patch forward cross-check against MNE-Python')
        print(f'  MNE-Python : {mne.__version__}')
        print(f'  samples    : {args.n_samples} per mode\n')

        print('  generating patch-mode data ...')
        cfg_patch = generate('patch', args.n_samples, patch_dir, args.seed)
        patch_err, active, diameters = relative_errors_patch(
            h5py, mne, patch_dir, cfg_patch, args.n_samples
        )

        print('  generating point-mode control ...')
        cfg_point = generate('point', args.n_samples, point_dir, args.seed)
        point_err = relative_errors_point(
            h5py, mne, point_dir, cfg_point, args.n_samples
        )

    patch_stats = summarize(patch_err)
    point_stats = summarize(point_err)

    print('\n  Relative field difference (2-norm over channels and time):\n')
    print(f'    {"":26s}{"median":>12s}{"p90":>12s}{"max":>12s}')
    print(f'    {"point-source control":26s}{point_stats["median"]:12.3e}'
          f'{point_stats["p90"]:12.3e}{point_stats["max"]:12.3e}')
    print(f'    {"distributed patch":26s}{patch_stats["median"]:12.3e}'
          f'{patch_stats["p90"]:12.3e}{patch_stats["max"]:12.3e}')

    ratio = patch_stats['median'] / point_stats['median']
    print(f'\n    active dipoles per patch : {int(active.min())} to '
          f'{int(active.max())} (median {int(np.median(active))})')
    print(f'    patch diameter (mm)      : {diameters.min():.1f} to '
          f'{diameters.max():.1f}')
    print(f'    patch / control ratio    : {ratio:.2f}')
    print('\n  The control residual is that of the Berg-Scherg fit of MNE-Python. '
          'The\n  comparison checks whether the patch residual is elevated '
          'above it\n  at the reported summary-statistic level.')

    payload = {
        'software_environment': environment_snapshot(),
        'n_samples': int(args.n_samples),
        'seed': int(args.seed),
        'radii_m': list(cfg_patch.head.radii_m),
        'conductivities_S_per_m': list(cfg_patch.head.conductivities_S_per_m),
        'n_cortical_positions': int(cfg_patch.sources.n_cortical_positions),
        'point_source_control': point_stats,
        'distributed_patch': patch_stats,
        'patch_to_control_median_ratio': float(ratio),
        'active_dipoles_per_patch': {
            'min': int(active.min()),
            'median': int(np.median(active)),
            'max': int(active.max()),
        },
        'patch_diameter_mm': {
            'min': float(diameters.min()),
            'max': float(diameters.max()),
        },
    }
    with summary_path.open('w', newline='\n') as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write('\n')


if __name__ == '__main__':
    raise SystemExit(main())
