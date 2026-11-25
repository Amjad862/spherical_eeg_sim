"""Repeat the sphere-model dipole fits of the experiments with a converged parameter fit of the sphere model.

The split study, the transfer experiment and the adaptation experiment compare
their networks with dipole fits, most of which use the sphere model of
MNE-Python. For the head models of these experiments, MNE-Python 1.12.1 with
SciPy 1.17.1 ends the fit of the Berg-Scherg parameters of that model at
SciPy's limit of 1,000 function evaluations before it has converged
(scripts/run_sphere_fit_convergence.py). This script repeats the sphere-model
dipole fits that the paper reports, on the same windows and with the same
calls, once with that limit and once with a limit that lets the optimizer stop
by itself:

  1. the matched-sphere fit of run_split_study.py, on the dipole-fit subset of
     every dataset and split policy;
  2. the matched-sphere fit of run_head_transfer.py, on its subset of sphere
     windows;
  3. the fits of run_head_adaptation.py, in which each sphere is fitted to the
     same boundary-element windows.

The fits at the limit of 1,000 evaluations must be those of the recorded
experiments. With --recorded the script compares them with the per-window
files of those experiments and stops if a fitted position differs.

Not repeated: the fit with the boundary-element model, which uses no sphere
model, and two sphere-model fits that the experiments record and the paper
does not report (the automatically fitted sphere of run_head_transfer.py and
the fit on the shared test windows of run_split_study.py).

    python scripts/run_dipole_comparators_converged.py --split-run validation_runs/split-recorded \\
        --subjects-dir path/to/templates --recorded path/to/evidence --output-dir validation_runs/comparators

It needs the datasets of a recorded run of run_split_study.py and the
fsaverage template (mne.datasets.fetch_fsaverage). With --smoke it takes a
smoke run of run_split_study.py, the datasets of the first dataset seed and
two spheres, with six windows of the split study and four of the other two
experiments. The dipole
fits run with one thread per library, as in the experiments.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _run_provenance import check_new_output, collect_provenance, hash_file

MNE_LIMIT, RAISED_LIMIT = 1000, 100_000
FITS = (('mne_limit', MNE_LIMIT), ('raised_limit', RAISED_LIMIT))
POLICIES = ('position', 'region')
HEADER = ['experiment', 'dataset_seed', 'case', 'window_id',
          *[f'{quantity}_{name}' for name, _ in FITS
            for quantity in ('position_error_mm', 'fitted_x_mm', 'fitted_y_mm', 'fitted_z_mm')]]


def subset_rows(seed, stream, available, count):
    """The windows of a dipole-fit subset, as the experiments draw them."""
    return np.sort(np.random.default_rng([seed, stream]).choice(available, size=min(count, available), replace=False))


def metrics(errors):
    errors = np.asarray(errors, dtype=float)
    return dict(windows=int(errors.size), median_mm=float(np.median(errors)), mean_mm=float(np.mean(errors)),
                p90_mm=float(np.percentile(errors, 90)))


def group_summary(fits, errors, fitted):
    """The record of one group of windows: how both parameter fits ended, and the errors with each."""
    moved = np.linalg.norm(fitted['raised_limit'] - fitted['mne_limit'], axis=1)
    return dict(
        parameter_fit={name: fits[name] for name, _ in FITS},
        **{name: metrics(errors[name]) for name, _ in FITS},
        median_of_raised_limit_minus_median_of_mne_limit_mm=float(
            np.median(errors['raised_limit']) - np.median(errors['mne_limit'])),
        fitted_position_change_mm=dict(median=float(np.median(moved)), p90=float(np.percentile(moved, 90)),
                                       maximum=float(moved.max())))


def group_rows(experiment, seed, case, window_ids, errors, fitted):
    return [[experiment, seed, case, int(window),
             *[cell for name, _ in FITS
               for cell in (repr(float(errors[name][index])), *[repr(float(value)) for value in fitted[name][index]])]]
            for index, window in enumerate(window_ids)]


def recorded_positions(path, prefix, id_column=None):
    """Fitted positions in mm, and the window numbers if the file has them, of a recorded per-window file."""
    with Path(path).open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    positions = np.array([[float(row[f'{prefix}_{axis}_mm']) for axis in 'xyz'] for row in rows])
    windows = None if id_column is None else [int(row[id_column]) for row in rows]
    return positions, windows


def check_recorded(label, fitted, window_ids, positions, windows):
    """Stop unless the fits at the limit of SciPy are the recorded ones. Returns the largest difference in mm."""
    if len(positions) != len(fitted) or (windows is not None and windows != [int(item) for item in window_ids]):
        raise ValueError(f'{label}: the recorded experiment fitted other windows')
    difference = float(np.abs(positions - fitted).max())
    if not difference <= 1e-9:      # also stops on a position that is not finite
        raise ValueError(f'{label}: the fit at the limit of {MNE_LIMIT:,} evaluations differs from the recorded fit '
                         f'by up to {difference:.3g} mm')
    return difference


def both_fits(windows, electrodes, names, radii, sigmas, sampling_freq):
    """Fit the same windows with the sphere model of each evaluation limit.

    Returns how the two parameter fits ended and the fitted positions in mm.
    Each sphere model is built inside the instrumented block, so that its
    parameter fit runs with the limit of its arm, and exactly one parameter
    fit must be logged per arm. The arm with the limit of SciPy makes the
    calls of the experiments (run_split_study.dipole_fit).
    """
    import mne

    from run_sphere_fit_convergence import instrumented_fit
    from run_split_study import dipole_fit
    from validate_patch_forward_mne import build_sphere_and_info

    fits, fitted = {}, {}
    for name, limit in FITS:
        log = []
        with instrumented_fit(limit, 1.0, log):
            info, sphere = build_sphere_and_info(electrodes, names, radii, sigmas, sampling_freq, mne)
            fitted[name] = dipole_fit(windows, info, sphere)
        if len(log) != 1:
            raise RuntimeError('expected one parameter fit for one sphere model')
        fits[name] = dict(evaluations=log[0]['evaluations'], stopped_by_itself=log[0]['stopped_by_itself'],
                          objective=log[0]['objective'])
    return fits, fitted


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--split-run', required=True, type=Path, help='completed run_split_study.py output directory')
    parser.add_argument('--subjects-dir', required=True, type=Path, help='directory that contains fsaverage/bem')
    parser.add_argument('--output-dir', required=True, type=Path, help='new result directory')
    parser.add_argument('--recorded', type=Path,
                        help='evidence folder with split-study-v11, head-transfer-v11 and head-adaptation-v11; the '
                             'fits at the limit of SciPy are compared with their per-window files')
    parser.add_argument('--ecd-count', type=int, default=500, help='windows in each dipole-fit subset (default: 500)')
    parser.add_argument('--smoke', action='store_true',
                        help='a smoke split run, the datasets of the first dataset seed, two spheres, six windows of '
                             'the split study and four of the other experiments; relaxed provenance')
    args = parser.parse_args(argv)
    if args.ecd_count < 1:
        parser.error('--ecd-count must be positive')
    if args.smoke and args.recorded is not None:
        parser.error('--recorded belongs to a full run')
    try:
        output = check_new_output(args.output_dir, ROOT)
        provenance = collect_provenance(
            ROOT, [Path(__file__), ROOT / 'scripts/run_sphere_fit_convergence.py',
                   ROOT / 'scripts/run_split_study.py', ROOT / 'scripts/run_head_transfer.py',
                   ROOT / 'scripts/run_head_adaptation.py', ROOT / 'scripts/train_baseline.py',
                   ROOT / 'scripts/validate_patch_forward_mne.py', ROOT / 'validation-requirements.txt',
                   ROOT / 'validation-windows-lock.txt'],
            list(sys.argv[1:] if argv is None else argv), require_clean=not args.smoke)
    except ValueError as error:
        parser.error(str(error))

    import mne
    from mne._fiff.constants import FIFF
    from mne.transforms import invert_transform
    from threadpoolctl import threadpool_limits

    from run_head_adaptation import sphere_variants, template_geometry
    from run_head_transfer import (Registration, bem_features, forward_columns, gate_dataset, pick_bem_surfaces,
                                   registration_gate)
    from run_split_study import DATA_SEEDS, load_windows, record_output_hashes, record_provenance, sensor_info, write_json
    from spherical_eeg_sim import Config

    if mne.__version__ != '1.12.1':
        parser.error('the recorded experiments pin MNE-Python 1.12.1')
    mne.set_log_level('ERROR')
    template = args.subjects_dir / 'fsaverage/bem'
    head_file, trans_file = template / 'fsaverage-5120-5120-5120-bem.fif', template / 'fsaverage-trans.fif'
    seeds = DATA_SEEDS[:1] if args.smoke else DATA_SEEDS
    count = 4 if args.smoke else args.ecd_count
    first = args.split_run / str(seeds[0]) / 'position'
    # The record of the threads was taken at the start of the program; the fits below run with one thread.
    provenance.update(smoke=args.smoke, inputs={}, threads_per_library_during_the_fits=1,
                      template_sha256={file.name: hash_file(file) for file in (head_file, trans_file)})
    for seed in seeds:
        for policy in POLICIES:
            path = args.split_run / str(seed) / policy
            provenance['inputs'][f'{seed}/{policy}'] = {
                name: hash_file(path / name) for name in ('config.json', 'dataset/checksums.sha256')}
    output.mkdir(parents=True, exist_ok=False)
    record_provenance(output, provenance)

    def record_group(experiment, seed, case, window_ids, truth, fits, fitted, recorded=None):
        errors = {name: np.linalg.norm(fitted[name] - truth, axis=1) for name, _ in FITS}
        entry = group_summary(fits, errors, fitted)
        entry['largest_difference_from_the_recorded_fit_mm'] = None if recorded is None else check_recorded(
            f'{experiment}, dataset {seed}, {case}', fitted['mne_limit'], window_ids, *recorded)
        rows.extend(group_rows(experiment, seed, case, window_ids, errors, fitted))
        summary[experiment].setdefault(str(seed), {})[case] = entry
        print(f"{experiment} {seed} {case}: median {entry['mne_limit']['median_mm']:.3f} mm with the limit of SciPy, "
              f"{entry['raised_limit']['median_mm']:.3f} mm with the converged fit "
              f"({fits['raised_limit']['evaluations']} evaluations)", flush=True)

    rows, summary = [], dict(split_study={}, transfer={}, adaptation={})
    with threadpool_limits(limits=1), tempfile.TemporaryDirectory(prefix='.comparators-', dir=output) as scratch:
        # 1. The matched-sphere fit of the split study.
        for seed in seeds:
            for policy in POLICIES:
                path = args.split_run / str(seed) / policy
                cfg = Config.from_json(path / 'config.json')
                test = load_windows(path / 'dataset/test.h5')
                electrodes, names = sensor_info(path / 'dataset/test.h5')
                chosen = subset_rows(seed, 601, len(test.position), 6 if args.smoke else args.ecd_count)
                fits, fitted = both_fits(test.topography[chosen], electrodes, names, cfg.head.radii_m,
                                         cfg.head.conductivities_S_per_m, cfg.signal.sampling_freq_hz)
                recorded = None
                if args.recorded is not None:
                    subset = args.recorded / 'split-study-v11' / str(seed) / policy / 'ecd_subset'
                    recorded = (recorded_positions(subset / 'predictions.csv', 'dipole')[0],
                                json.loads((subset / 'test_rows.json').read_text(encoding='utf-8')))
                record_group('split_study', seed, policy, chosen, test.position[chosen], fits, fitted, recorded)

        # 2 and 3. The first dataset on the sphere and on the boundary-element model of the template head.
        seed = seeds[0]
        cfg = Config.from_json(first / 'config.json')
        trans = invert_transform(mne.read_trans(trans_file, verbose=False))
        reg = Registration.from_recording_electrodes()
        surfaces = pick_bem_surfaces(args.subjects_dir, cfg)
        gates = Path(scratch) / 'gates'
        gates.mkdir()
        state = gate_dataset(first, gates, reg, trans, surfaces, args.smoke)
        test, eligible = state['test'], state['rows']
        electrodes, names = sensor_info(first / 'dataset/test.h5')
        chosen = subset_rows(seed, 602, len(eligible), count)
        windows, truth = eligible[chosen], test.position[eligible[chosen]]
        radii, sigmas = cfg.head.radii_m, cfg.head.conductivities_S_per_m
        fits, fitted = both_fits(state['original'][chosen], electrodes, names, radii, sigmas,
                                 cfg.signal.sampling_freq_hz)
        recorded = None if args.recorded is None else recorded_positions(
            args.recorded / 'head-transfer-v11' / str(seed) / 'ecd_subset.csv', 'ecd_package_sphere_matched',
            'window_id')
        record_group('transfer', seed, 'package_sphere_matched', windows, truth, fits, fitted, recorded)

        scalp = next(surface for surface in surfaces if surface['id'] == FIFF.FIFFV_BEM_SURF_ID_HEAD)
        _, registration = registration_gate(reg, electrodes, names, trans, scalp, cfg.signal.sampling_freq_hz)
        geometry = template_geometry(reg, registration, surfaces)
        variants = sphere_variants(geometry)
        if args.recorded is not None:
            archived = json.loads((args.recorded / 'head-adaptation-v11/geometry.json').read_text(encoding='utf-8'))
            if archived['variants'] != variants:
                raise ValueError('the spheres differ from those of the recorded adaptation experiment')
            provenance['inputs']['geometry.json'] = hash_file(args.recorded / 'head-adaptation-v11/geometry.json')
            record_provenance(output, provenance)
        bem = mne.make_bem_solution(surfaces, verbose=False)
        field = forward_columns(reg.to_mri(state['unique_xyz']), state['bem_info'], bem, trans)
        features, _ = bem_features(state['dataset'], eligible, field[state['window_to_source']], reg, trans)
        spheres = {}
        for name in list(variants)[:2] if args.smoke else list(variants):
            variant = spheres[name] = variants[name]
            scaled = electrodes * (variant['radii_m'][-1] / np.linalg.norm(electrodes, axis=1, keepdims=True))
            fits, fitted = both_fits(features[chosen], scaled, names, variant['radii_m'],
                                     variant['conductivities_S_per_m'], cfg.signal.sampling_freq_hz)
            recorded = None if args.recorded is None else recorded_positions(
                args.recorded / 'head-adaptation-v11' / str(seed) / 'dipole_subset.csv', f'dipole_{name}', 'window_id')
            record_group('adaptation', seed, f'{name}_sphere_on_bem', windows, truth, fits, fitted, recorded)

    with (output / 'comparators.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(HEADER)
        writer.writerows(rows)
    write_json(output / 'summary.json', dict(
        smoke=args.smoke, evaluation_limits=dict(FITS), windows_per_subset=count,
        compared_with_the_recorded_experiments=args.recorded is not None,
        spheres=spheres, **summary))
    record_output_hashes(output)
    print(f'wrote {output}')


if __name__ == '__main__':
    main()
