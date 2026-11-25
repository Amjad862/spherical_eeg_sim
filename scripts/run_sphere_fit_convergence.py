"""Record how the Berg-Scherg parameter fit of MNE-Python ends, and what follows from it.

The two cross-implementation checks of the package, validate_inverse_mne.py and
validate_patch_forward_mne.py, compare the series solution with the sphere
model of MNE-Python. MNE-Python represents the multilayer sphere by three
Berg-Scherg terms and finds their parameters with SciPy's COBYLA optimizer. It
passes no evaluation limit, so SciPy's default of 1,000 function evaluations
applies. For the head model of the package, MNE-Python 1.12.1 with SciPy 1.17.1
reaches that limit before the optimizer has converged. On the datasets of the
two checks this script records

  1. how the parameter fit ends at that limit, and how it ends when the limit
     is raised so far that the optimizer stops by itself, also for the sphere
     model that MNE-Python builds when no head model is given;
  2. the patch comparison, the dipole recovery at 35 dB and the SNR sweep with
     both fits;
  3. the patch comparison and the dipole recovery at 35 dB with the objective
     of the parameter fit multiplied by 1 + j * eps for j = 1, 2, ..., where
     eps is the machine precision, which changes its value in the last bits
     only;
  4. the dipole recovery with the objective of the dipole fits multiplied by
     the same factors, with either sphere model, and how far the fitted
     position of single dipoles moves in every trial of parts 3 and 4;
  5. the dipole recovery with the dipole fits ended at a trust-region radius
     of 1e-7 m in place of the 5e-5 m of MNE-Python, for the unchanged
     objectives and for some of the factors of parts 3 and 4;
  6. parts 1 and 2 without the SNR sweep, and part 5 for the unchanged
     objectives, in child processes that use another kernel of the
     linear-algebra library (OpenBLAS selects its processor-specific routines
     from the environment variable OPENBLAS_CORETYPE), and the parameter fit
     with the objective at its second trial point replaced by the value of
     the other kernel;
  7. with --geometry, the parameter fit, the point-source comparison and a
     dipole recovery at 10 to 25 dB for every sphere of
     run_head_adaptation.py.

Parts 1 and 2 show whether the residual of the comparison belongs to the
Berg-Scherg approximation or to an unfinished fit. Parts 3, 4 and 6 show that
the end point of the unfinished fit, and the dipole statistics with either
fit, depend on details of the arithmetic, and part 5 shows what remains when
the parameter fit has converged and the dipole fits end at a much smaller
radius. The runs are examples from one machine:
eight changes of one kind and a few library kernels. Their ranges are not
bounds for other platforms. The SNR sweep is run with the two unchanged fits
only.

The optimizer starts at mu = 0 with a first step of 0.5. At each of its first
four trial points at least two of the three Berg-Scherg terms coincide, so the
least-squares problem inside the objective is rank-deficient and the value
that MNE-Python computes depends on singular vectors that the problem does not
determine. At three of these points the degenerate columns of the model matrix
are exactly zero. At the second, mu = (0.5, 0, 0), two columns are equal and
nonzero, and the value differs between library kernels.

    python scripts/run_sphere_fit_convergence.py --output-dir validation_runs/sphere-fit
    python scripts/run_sphere_fit_convergence.py --output-dir validation_runs/sphere-fit-smoke --smoke

Requires MNE-Python and h5py:  pip install "spherical_eeg_sim[all]"
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _run_provenance import check_new_output, collect_provenance, hash_file
from spherical_eeg_sim.provenance import environment_snapshot

SEED = 20260703
MNE_LIMIT = 1000        # SciPy's default for fmin_cobyla; MNE-Python passes no limit of its own
RAISED_LIMIT = 100_000
FITS = (('mne_limit', MNE_LIMIT), ('raised_limit', RAISED_LIMIT))
FIRST_EVALUATIONS = 4   # recorded for every parameter fit; the second is the point (0.5, 0, 0)
KERNEL_VARIABLE = 'OPENBLAS_CORETYPE'
KERNELS = ('Haswell', 'Sandybridge', 'Nehalem')
COMPARATOR_SNR_DB = (10.0, 25.0)   # the SNR range of the experiments that use the dipole fit as a comparator
MOVED_MM = 0.1
SMALL_TOLERANCE_M = 1e-7           # final trust-region radius of the dipole fits in part 5
SMALL_TOLERANCE_TRIALS = 4         # changed objectives of each kind in part 5


def _cobyla(func, x0, cons, consargs, scale, *, replace=None, trace=None, **options):
    """The call that scipy.optimize.fmin_cobyla makes, with the full result.

    ``scale`` multiplies the objective. ``replace`` maps the number of an
    evaluation, counted from one, to a value that the optimizer receives in
    its place. ``trace`` receives the point and the value of every evaluation.
    """
    from scipy.optimize import NonlinearConstraint, minimize

    if scale == 1.0 and not replace and trace is None:
        objective = func
    else:
        count = 0

        def objective(x):
            nonlocal count
            count += 1
            value = func(x) if scale == 1.0 else scale * func(x)
            if replace and count in replace:
                value = replace[count]
            if trace is not None:
                trace.append(([float(item) for item in x], float(value)))
            return value

    constraints = [NonlinearConstraint(lambda x, constraint=constraint: constraint(x, *consargs), 0, np.inf)
                   for constraint in cons]
    return minimize(objective, x0, method='COBYLA', constraints=constraints, options=options)


@contextmanager
def instrumented_fit(limit, scale=1.0, log=None, *, replace=None):
    """Replace the optimizer call of MNE-Python's Berg-Scherg fit inside the block.

    The replacement makes the same COBYLA call, which also reports how the
    optimizer stopped. ``limit`` is the number of function evaluations
    allowed, ``scale`` and ``replace`` change the objective as in ``_cobyla``,
    and every fit made inside the block appends a record to ``log``. The
    record holds the arguments that MNE-Python passed. The replacement takes
    no evaluation limit from its caller, so a call with one would fail.
    """
    import mne.bem

    def fit(func, x0, cons, rhobeg=1.0, rhoend=1e-4, catol=2e-4):
        trace = []
        result = _cobyla(func, x0, cons, (), scale, replace=replace, trace=trace,
                         rhobeg=rhobeg, tol=rhoend, maxiter=limit, catol=catol)
        if log is not None:
            log.append(dict(evaluations=int(result.nfev), stopped_by_itself=bool(result.success),
                            status=int(result.status), message=str(result.message),
                            objective=float(func(result.x)),
                            call=dict(start=[float(item) for item in x0], constraints=len(cons),
                                      rhobeg=float(rhobeg), rhoend=float(rhoend), catol=float(catol)),
                            first_evaluations=[dict(mu=point, objective=value)
                                               for point, value in trace[:FIRST_EVALUATIONS]]))
        return result.x

    original = mne.bem.fmin_cobyla
    mne.bem.fmin_cobyla = fit
    try:
        yield
    finally:
        mne.bem.fmin_cobyla = original


@contextmanager
def instrumented_dipole_fit(scale=1.0, log=None):
    """Replace the optimizer call of MNE-Python's dipole fit inside the block.

    The evaluation limit stays at SciPy's default. ``scale`` multiplies the
    objective, and every dipole fit appends a record to ``log``.
    """
    import mne.dipole

    def fit(func, x0, cons, consargs=(), rhobeg=1.0, rhoend=1e-4, catol=2e-4, disp=None):
        result = _cobyla(func, x0, cons, tuple(consargs), scale, rhobeg=rhobeg, tol=rhoend, maxiter=MNE_LIMIT,
                         catol=catol)
        if log is not None:
            log.append(dict(evaluations=int(result.nfev), stopped_by_itself=bool(result.success),
                            rhoend=float(rhoend)))
        return result.x

    original = mne.dipole.fmin_cobyla
    mne.dipole.fmin_cobyla = fit
    try:
        yield
    finally:
        mne.dipole.fmin_cobyla = original


def sphere_model(radii, sigmas):
    """The sphere model as the two validation scripts build it."""
    import mne

    radii = np.asarray(radii, dtype=float)
    return mne.make_sphere_model(r0=(0.0, 0.0, 0.0), head_radius=float(radii[-1]),
                                 relative_radii=tuple(radii / radii[-1]), sigmas=tuple(sigmas), verbose=False)


def default_sphere_of_mne_python():
    """How the parameter fit ends for the sphere model that MNE-Python builds when no head model is given."""
    import mne

    record = {}
    for name, limit in FITS:
        log = []
        with instrumented_fit(limit, 1.0, log):
            model = mne.make_sphere_model(verbose=False)
        record[name] = dict(log[0], evaluation_limit=limit, mu=[float(value) for value in model['mu']])
    record['layers'] = [dict(radius_m=float(layer['rad']), conductivity_S_per_m=float(layer['sigma']))
                        for layer in model['layers']]
    return record


def spread(values):
    values = np.asarray(values, dtype=float)
    return dict(n=int(values.size), median=float(np.median(values)),
                p90=float(np.percentile(values, 90)), maximum=float(values.max()))


def dipole_fit_record(log):
    """How the dipole fits of one run ended."""
    return dict(fits=len(log), stopped_by_themselves=sum(entry['stopped_by_itself'] for entry in log),
                evaluations=spread([entry['evaluations'] for entry in log]))


def position_change(estimated, reference):
    """Distance in mm between the fitted positions of two runs on the same dipoles."""
    moved = np.linalg.norm(np.asarray(estimated, dtype=float) - np.asarray(reference, dtype=float), axis=1) * 1000.0
    return dict(spread(moved), share_above_0p1_mm=float(np.mean(moved > MOVED_MM)))


def array_digest(path, name):
    import h5py

    with h5py.File(path, 'r') as file:
        return hashlib.sha256(np.ascontiguousarray(file[name][()]).tobytes()).hexdigest()


def comparison_config(radii, sigmas, sampling_freq_hz):
    """The settings that the field comparisons of validate_patch_forward_mne.py read from a configuration."""
    return SimpleNamespace(signal=SimpleNamespace(sampling_freq_hz=float(sampling_freq_hz)),
                           head=SimpleNamespace(radii_m=[float(value) for value in radii],
                                                conductivities_S_per_m=[float(value) for value in sigmas]))


def evaluate(limit, scale, data, *, fields=True, with_sweep=False, dipole_scale=None, dipole_log=None,
             dipole_tolerance=None):
    """Fit the sphere model and run the comparisons of the two validation scripts with it.

    ``data`` holds the paths and settings of the datasets as plain values, so
    that a child process can receive it. ``dipole_scale`` selects the
    instrumented dipole fit; without it the dipole fit of MNE-Python runs
    unchanged. ``dipole_tolerance`` is the final trust-region radius of the
    dipole fits in metres; without it MNE-Python's default applies.
    """
    import h5py
    import mne

    import validate_inverse_mne as inverse
    import validate_patch_forward_mne as patch

    @contextmanager
    def dipole_fit():
        if dipole_scale is None:
            yield
        else:
            with instrumented_dipole_fit(dipole_scale, dipole_log):
                yield

    radii, sigmas = np.asarray(data['radii'], dtype=float), np.asarray(data['sigmas'], dtype=float)
    config = comparison_config(radii, sigmas, data['sampling_freq_hz'])
    log = []
    patch_errors = point_errors = active = diameters = None
    with instrumented_fit(limit, scale, log), dipole_fit():
        model = sphere_model(radii, sigmas)
        if fields:
            patch_errors, active, diameters = patch.relative_errors_patch(
                h5py, mne, data['patch_dir'], config, data['samples'])
            point_errors = patch.relative_errors_point(h5py, mne, data['point_dir'], config, data['samples'])
        recovery = inverse.recover(data['headline'], radii, sigmas, inverse.MIN_DIST_MM, tol=dipole_tolerance)
        sweep = [(level, inverse.recover(path, radii, sigmas, inverse.MIN_DIST_MM, tol=dipole_tolerance))
                 for level, path in data['sweep']] if with_sweep else []
    if any(entry != log[0] for entry in log):
        raise RuntimeError('the parameter fit did not end the same way for every sphere model of one arm')
    epsilon = float(np.finfo(float).eps)
    summary = dict(
        fit=dict(log[0], evaluation_limit=limit, objective_scale_minus_one_in_eps=round((scale - 1.0) / epsilon),
                 mu=[float(value) for value in model['mu']], lambda_=[float(value) for value in model['lambda']]),
        dipole_recovery=dict(position_error_mm=spread(recovery['err_mm']),
                             orientation_error_deg=spread(recovery['orientation_error_deg']),
                             moment_relative_error=spread(recovery['moment_relative_error']),
                             goodness_of_fit_percent=spread(recovery['gof'])),
        sweep=[dict(snr_db=float(level), position_error_mm=spread(result['err_mm'])) for level, result in sweep])
    if fields:
        summary['patch_comparison'] = dict(
            patch=spread(patch_errors), point=spread(point_errors),
            patch_to_point_median_ratio=float(np.median(patch_errors) / np.median(point_errors)))
    arrays = dict(patch=patch_errors, point=point_errors, active=active, diameters=diameters,
                  recovery=recovery, sweep=sweep)
    return summary, arrays


def write_rows(path, header, rows):
    with path.open('w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerow(header)
        writer.writerows(rows)


def write_per_window_files(output, arrays):
    names = [name for name, _ in FITS]
    first = arrays[names[0]]
    recovery = {name: arrays[name]['recovery'] for name in names}
    write_rows(output / 'dipole_recovery_35db.csv',
               ['sample_index', 'source_radius_mm', 'snr_db'] +
               [f'{quantity}_{name}' for name in names
                for quantity in ('position_error_mm', 'orientation_error_deg', 'moment_relative_error',
                                 'goodness_of_fit_percent')],
               [[index, repr(float(recovery[names[0]]['source_radius_mm'][index])),
                 repr(float(recovery[names[0]]['snr_db'][index]))] +
                [repr(float(recovery[name][key][index])) for name in names
                 for key in ('err_mm', 'orientation_error_deg', 'moment_relative_error', 'gof')]
                for index in range(len(recovery[names[0]]['err_mm']))])
    rows = []
    for kind in ('patch', 'point'):
        for index in range(len(first[kind])):
            rows.append([kind, index,
                         int(first['active'][index]) if kind == 'patch' else '',
                         repr(float(first['diameters'][index])) if kind == 'patch' else ''] +
                        [repr(float(arrays[name][kind][index])) for name in names])
    write_rows(output / 'field_differences.csv',
               ['source', 'sample_index', 'active_dipoles', 'patch_diameter_mm'] +
               [f'relative_field_difference_{name}' for name in names], rows)
    rows = []
    for position, (level, _) in enumerate(first['sweep']):
        errors = {name: arrays[name]['sweep'][position][1]['err_mm'] for name in names}
        rows += [[repr(float(level)), index] + [repr(float(errors[name][index])) for name in names]
                 for index in range(len(errors[names[0]]))]
    write_rows(output / 'snr_sweep.csv',
               ['snr_db', 'sample_index'] + [f'position_error_mm_{name}' for name in names], rows)


PERTURBATION_HEADER = ['perturbed_fit', 'sphere_fit', 'objective_scale_minus_one_in_eps',
                       'parameter_fit_evaluations', 'parameter_fit_stopped_by_itself', 'parameter_fit_objective',
                       'mu_1', 'mu_2', 'mu_3',
                       'dipole_fits', 'dipole_fits_stopped_by_themselves', 'dipole_fit_evaluations_median',
                       'dipole_fit_evaluations_maximum',
                       'patch_median_relative_field_difference', 'point_median_relative_field_difference',
                       'position_error_median_mm', 'position_error_p90_mm', 'position_error_maximum_mm',
                       'moment_relative_error_median',
                       'fitted_position_change_median_mm', 'fitted_position_change_p90_mm',
                       'fitted_position_change_maximum_mm', 'fitted_positions_changed_by_more_than_0p1_mm']
STATISTICS = PERTURBATION_HEADER[13:]
DIPOLE_VALUES = ['fitted_x_mm', 'fitted_y_mm', 'fitted_z_mm', 'position_error_mm', 'orientation_error_deg',
                 'moment_relative_error', 'goodness_of_fit_percent']
DIPOLE_HEADER = ['perturbed_fit', 'sphere_fit', 'objective_scale_minus_one_in_eps', 'sample_index', *DIPOLE_VALUES]
FIELD_HEADER = ['sphere_fit', 'objective_scale_minus_one_in_eps', 'source', 'sample_index',
                'relative_field_difference']
# Part 5. The kernel is that of the process that made the run; a change is taken from the unchanged run of
# the same sphere fit in the main process.
SMALL_TOLERANCE_HEADER = ['kernel', 'changed_objective', 'sphere_fit', 'objective_scale_minus_one_in_eps',
                          'parameter_fit_evaluations', 'parameter_fit_stopped_by_itself',
                          'dipole_fits', 'dipole_fits_stopped_by_themselves', 'dipole_fit_evaluations_median',
                          'dipole_fit_evaluations_maximum',
                          'position_error_median_mm', 'position_error_p90_mm', 'position_error_maximum_mm',
                          'fitted_position_change_median_mm', 'fitted_position_change_p90_mm',
                          'fitted_position_change_maximum_mm']
SMALL_TOLERANCE_DIPOLE_HEADER = ['kernel', 'changed_objective', 'sphere_fit', 'objective_scale_minus_one_in_eps',
                                 'sample_index', 'fitted_x_mm', 'fitted_y_mm', 'fitted_z_mm', 'position_error_mm']


def perturbation_row(perturbed, name, step, summary, change, dipole_fits=None):
    """One row of perturbation.csv. Fields are empty where a quantity was not computed.

    The three columns of the parameter fit describe the sphere model of the
    row; in a row of the dipole fits that model was fitted with the unchanged
    objective. ``change`` compares the fitted positions with those of the
    unchanged run of the same sphere fit, and ``dipole_fits`` is the record of
    ``dipole_fit_record`` where the dipole fits were instrumented.
    """
    fit, recovery = summary['fit'], summary['dipole_recovery']
    fields = summary.get('patch_comparison')
    dipoles = ['', '', '', ''] if dipole_fits is None else [
        dipole_fits['fits'], dipole_fits['stopped_by_themselves'],
        repr(dipole_fits['evaluations']['median']), repr(dipole_fits['evaluations']['maximum'])]
    return [perturbed, name, step, fit['evaluations'], int(fit['stopped_by_itself']), repr(fit['objective']),
            *[repr(value) for value in fit['mu']], *dipoles,
            repr(fields['patch']['median']) if fields else '', repr(fields['point']['median']) if fields else '',
            repr(recovery['position_error_mm']['median']), repr(recovery['position_error_mm']['p90']),
            repr(recovery['position_error_mm']['maximum']), repr(recovery['moment_relative_error']['median']),
            repr(change['median']), repr(change['p90']), repr(change['maximum']),
            int(round(change['share_above_0p1_mm'] * change['n']))]


def dipole_values(recovery, index):
    """The cells of one dipole of one run: fitted position in mm and the four errors."""
    fitted = np.asarray(recovery['estimated_xyz'][index], dtype=float) * 1000.0
    return [*[repr(float(value)) for value in fitted], repr(float(recovery['err_mm'][index])),
            repr(float(recovery['orientation_error_deg'][index])),
            repr(float(recovery['moment_relative_error'][index])), repr(float(recovery['gof'][index]))]


def dipole_rows(perturbed, name, step, recovery):
    """Every dipole of one run, for perturbation_dipoles.csv."""
    return [[perturbed, name, step, index, *dipole_values(recovery, index)]
            for index in range(len(recovery['err_mm']))]


def column_range(rows, name):
    values = [float(row[PERTURBATION_HEADER.index(name)]) for row in rows
              if row[PERTURBATION_HEADER.index(name)] != '']
    return [min(values), max(values)] if values else None


def perturbation_ranges(rows):
    """Smallest and largest value of every quantity, by perturbed fit and sphere fit."""
    ranges = {}
    stopped = PERTURBATION_HEADER.index('parameter_fit_stopped_by_itself')
    for perturbed, name in dict.fromkeys((row[0], row[1]) for row in rows):
        selected = [row for row in rows if (row[0], row[1]) == (perturbed, name)]
        entry = {column: column_range(selected, column)
                 for column in ('parameter_fit_evaluations', 'parameter_fit_objective', 'mu_1', 'mu_2', 'mu_3',
                                'dipole_fits', 'dipole_fits_stopped_by_themselves',
                                'dipole_fit_evaluations_median', 'dipole_fit_evaluations_maximum', *STATISTICS)}
        entry = {column: value for column, value in entry.items() if value is not None}
        entry.update(trials=len(selected),
                     trials_in_which_the_parameter_fit_stopped_by_itself=sum(int(row[stopped]) for row in selected))
        ranges.setdefault(perturbed, {})[name] = entry
    return ranges


def small_tolerance_row(kernel, changed, name, step, fit, recovery, fits, change):
    """One row of small_tolerance.csv: a dipole recovery with the dipole fits ended at the small radius."""
    position = spread(recovery['err_mm'])
    moved = ['', '', ''] if change is None else [repr(change['median']), repr(change['p90']), repr(change['maximum'])]
    return [kernel, changed, name, step, fit['evaluations'], int(fit['stopped_by_itself']),
            fits['fits'], fits['stopped_by_themselves'], repr(fits['evaluations']['median']),
            repr(fits['evaluations']['maximum']),
            repr(position['median']), repr(position['p90']), repr(position['maximum']), *moved]


def small_tolerance_dipoles(kernel, changed, name, step, fitted_m, errors):
    fitted = np.asarray(fitted_m, dtype=float) * 1000.0
    return [[kernel, changed, name, step, index, *[repr(float(value)) for value in fitted[index]],
             repr(float(errors[index]))] for index in range(len(fitted))]


def small_tolerance_ranges(rows):
    """Smallest and largest value of the three position statistics over the runs of each sphere fit."""
    ranges = {}
    for name in dict.fromkeys(row[2] for row in rows):
        selected = [row for row in rows if row[2] == name]
        entry = dict(runs=len(selected), kernels=sorted({row[0] for row in selected}),
                     runs_in_which_the_parameter_fit_stopped_by_itself=sum(int(row[5]) for row in selected),
                     dipole_fits=sum(int(row[6]) for row in selected),
                     dipole_fits_stopped_by_themselves=sum(int(row[7]) for row in selected))
        for column in SMALL_TOLERANCE_HEADER[10:]:
            values = [float(row[SMALL_TOLERANCE_HEADER.index(column)]) for row in selected
                      if row[SMALL_TOLERANCE_HEADER.index(column)] != '']
            entry[column] = [min(values), max(values)] if values else None
        ranges[name] = entry
    return ranges


def library_kernels():
    """The kernel that every loaded linear-algebra library reports."""
    from threadpoolctl import threadpool_info

    return [dict(library=str(entry.get('internal_api')), version=str(entry.get('version')),
                 kernel=str(entry.get('architecture')))
            for entry in threadpool_info() if entry.get('user_api') == 'blas']


def kernel_job(job_path):
    """Child process of part 6 under the kernel of this process.

    Parts 1 and 2 without the SNR sweep, and the unchanged run of part 5.
    """
    import mne

    mne.set_log_level('ERROR')
    job = json.loads(Path(job_path).read_text(encoding='utf-8'))
    record, stored = dict(libraries=library_kernels(), arms={}), {}
    for name, limit in FITS:
        summary, arrays = evaluate(limit, 1.0, job['data'])
        record['arms'][name] = summary
        recovery = arrays['recovery']
        stored.update({f'{name}_patch': arrays['patch'], f'{name}_point': arrays['point'],
                       f'{name}_fitted_m': recovery['estimated_xyz'],
                       f'{name}_position_error_mm': recovery['err_mm'],
                       f'{name}_orientation_error_deg': recovery['orientation_error_deg'],
                       f'{name}_moment_relative_error': recovery['moment_relative_error'],
                       f'{name}_goodness_of_fit_percent': recovery['gof']})
    record['default_sphere_model_of_mne_python'] = default_sphere_of_mne_python()
    record['small_tolerance'] = {}
    for name, limit in FITS:
        log = []
        summary, arrays = evaluate(limit, 1.0, job['data'], fields=False, dipole_scale=1.0, dipole_log=log,
                                   dipole_tolerance=SMALL_TOLERANCE_M)
        record['small_tolerance'][name] = dict(fit=summary['fit'], dipole_fits=dipole_fit_record(log),
                                               position_error_mm=summary['dipole_recovery']['position_error_mm'])
        stored.update({f'small_{name}_fitted_m': arrays['recovery']['estimated_xyz'],
                       f'small_{name}_position_error_mm': arrays['recovery']['err_mm']})
    np.savez(job['arrays'], **stored)
    Path(job['result']).write_text(json.dumps(record), encoding='utf-8')


def run_kernel(kernel, data, work):
    """Run ``kernel_job`` in a child process whose library is asked for ``kernel``."""
    job = dict(data=data, result=str(work / f'kernel_{kernel}.json'), arrays=str(work / f'kernel_{kernel}.npz'))
    job_path = work / f'kernel_{kernel}_job.json'
    job_path.write_text(json.dumps(job), encoding='utf-8')
    done = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--kernel-job', str(job_path)],
                          env=dict(os.environ, **{KERNEL_VARIABLE: kernel}), capture_output=True, text=True)
    if done.returncode != 0 or not Path(job['result']).is_file():
        return None, None, f'the child process ended with exit code {done.returncode}'
    record = json.loads(Path(job['result']).read_text(encoding='utf-8'))
    with np.load(job['arrays']) as stored:
        arrays = {key: stored[key] for key in stored.files}
    return record, arrays, None


def kernel_rows(kernel, arrays):
    """Per-window rows of one kernel for kernels_dipole_recovery_35db.csv and kernels_field_differences.csv."""
    dipoles, fields = [], []
    for name, _ in FITS:
        recovery = dict(estimated_xyz=arrays[f'{name}_fitted_m'], err_mm=arrays[f'{name}_position_error_mm'],
                        orientation_error_deg=arrays[f'{name}_orientation_error_deg'],
                        moment_relative_error=arrays[f'{name}_moment_relative_error'],
                        gof=arrays[f'{name}_goodness_of_fit_percent'])
        dipoles += [[kernel, name, index, *dipole_values(recovery, index)]
                    for index in range(len(recovery['err_mm']))]
        for kind in ('patch', 'point'):
            fields += [[kernel, name, kind, index, repr(float(value))]
                       for index, value in enumerate(arrays[f'{name}_{kind}'])]
    return dipoles, fields


def sphere_datasets(name, variant, sizes, work):
    """The point-source datasets of one sphere: clean fields, and noisy windows at 10 to 25 dB."""
    import validate_inverse_mne as inverse
    import validate_patch_forward_mne as patch

    head = (variant['radii_m'], variant['conductivities_S_per_m'])
    field_dir = work / f'sphere_{name}_fields'
    config = patch.generate('point', sizes['samples'], field_dir, SEED, head=head)
    windows, radii, sigmas = inverse.generate_test_split(
        None, sizes['dipoles'], inverse.DEFAULT_MONTAGE, SEED, sizes['cortical'], sizes['subcortical'], work,
        tag=f'sphere_{name}', snr_range=COMPARATOR_SNR_DB, head=head)
    return dict(field_dir=str(field_dir), windows=str(windows), radii=[float(value) for value in radii],
                sigmas=[float(value) for value in sigmas], sampling_freq_hz=float(config.signal.sampling_freq_hz),
                samples=sizes['samples'],
                digests=dict(fields=array_digest(field_dir / 'train.h5', 'ground_truth/eeg_signal_only'),
                             windows=array_digest(windows, 'eeg')))


def evaluate_sphere(limit, data):
    """Part 7 for one sphere and one evaluation limit."""
    import h5py
    import mne

    import validate_inverse_mne as inverse
    import validate_patch_forward_mne as patch

    radii, sigmas = np.asarray(data['radii'], dtype=float), np.asarray(data['sigmas'], dtype=float)
    config = comparison_config(radii, sigmas, data['sampling_freq_hz'])
    log = []
    with instrumented_fit(limit, 1.0, log):
        model = sphere_model(radii, sigmas)
        fields = patch.relative_errors_point(h5py, mne, data['field_dir'], config, data['samples'])
        recovery = inverse.recover(data['windows'], radii, sigmas, inverse.MIN_DIST_MM)
    if any(entry != log[0] for entry in log):
        raise RuntimeError('the parameter fit did not end the same way for every sphere model of one arm')
    summary = dict(fit=dict(log[0], evaluation_limit=limit, mu=[float(value) for value in model['mu']],
                            lambda_=[float(value) for value in model['lambda']]),
                   point_field_difference=spread(fields), position_error_mm=spread(recovery['err_mm']))
    return summary, dict(fields=fields, recovery=recovery)


def spheres_part(geometry_path, sizes, smoke, output, work):
    """Part 7: every sphere of a run of run_head_adaptation.py."""
    variants = json.loads(geometry_path.read_text(encoding='utf-8'))['variants']
    names = list(variants)[:2] if smoke else list(variants)
    record, field_rows, recovery_rows = {}, [], []
    for name in names:
        data = sphere_datasets(name, variants[name], sizes, work)
        entry = dict(radii_m=data['radii'], conductivities_S_per_m=data['sigmas'],
                     dataset_array_sha256=data['digests'])
        arrays = {}
        for fit_name, limit in FITS:
            entry[fit_name], arrays[fit_name] = evaluate_sphere(limit, data)
        recovery = {fit_name: arrays[fit_name]['recovery'] for fit_name, _ in FITS}
        entry['position_error_change_mm'] = dict(
            median_of_raised_limit_minus_median_of_mne_limit=float(
                np.median(recovery['raised_limit']['err_mm']) - np.median(recovery['mne_limit']['err_mm'])),
            fitted_position=position_change(recovery['raised_limit']['estimated_xyz'],
                                            recovery['mne_limit']['estimated_xyz']))
        record[name] = entry
        first = recovery['mne_limit']
        field_rows += [[name, index] + [repr(float(arrays[fit_name]['fields'][index])) for fit_name, _ in FITS]
                       for index in range(len(arrays['mne_limit']['fields']))]
        recovery_rows += [[name, index, repr(float(first['source_radius_mm'][index])),
                           repr(float(first['snr_db'][index]))]
                          + [cell for fit_name, _ in FITS
                             for cell in (repr(float(recovery[fit_name]['err_mm'][index])),
                                          *[repr(float(value) * 1000.0)
                                            for value in recovery[fit_name]['estimated_xyz'][index]])]
                          for index in range(len(first['err_mm']))]
        print(f"sphere {name}: " + '; '.join(
            f"{fit_name} {entry[fit_name]['fit']['evaluations']} evaluations, field difference "
            f"{entry[fit_name]['point_field_difference']['median']:.2e}, position error median "
            f"{entry[fit_name]['position_error_mm']['median']:.3f} mm" for fit_name, _ in FITS), flush=True)
    write_rows(output / 'spheres_field_differences.csv',
               ['sphere', 'sample_index'] + [f'relative_field_difference_{fit_name}' for fit_name, _ in FITS],
               field_rows)
    write_rows(output / 'spheres_dipole_recovery.csv',
               ['sphere', 'sample_index', 'source_radius_mm', 'snr_db']
               + [f'{quantity}_{fit_name}' for fit_name, _ in FITS
                  for quantity in ('position_error_mm', 'fitted_x_mm', 'fitted_y_mm', 'fitted_z_mm')],
               recovery_rows)
    return dict(geometry_sha256=hash_file(geometry_path), snr_db_range=list(COMPARATOR_SNR_DB),
                spheres_of_the_geometry_record=list(variants), variants=record)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output-dir', type=Path, help='new result directory')
    parser.add_argument('--perturbations', type=int, default=8,
                        help='number of scaled objectives per fit in parts 3 and 4 (default: 8)')
    parser.add_argument('--kernels', nargs='*', default=list(KERNELS), metavar='NAME',
                        help=f'values of {KERNEL_VARIABLE} for part 6; kernels that the processor does not support, '
                             f'that the library does not have, or that equal the kernel in use are recorded as not '
                             f'available (default: {" ".join(KERNELS)})')
    parser.add_argument('--geometry', type=Path,
                        help='geometry.json of a run of run_head_adaptation.py; adds part 7 for its spheres')
    parser.add_argument('--smoke', action='store_true',
                        help='a few sources per dataset, one perturbation, one kernel and two spheres; '
                             'relaxed provenance')
    parser.add_argument('--kernel-job', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.kernel_job is not None:
        kernel_job(args.kernel_job)
        return
    if args.output_dir is None:
        parser.error('--output-dir is required')
    if args.perturbations < 0:
        parser.error('--perturbations must not be negative')
    if args.geometry is not None and not args.geometry.is_file():
        parser.error(f'--geometry: no such file: {args.geometry}')
    try:
        output = check_new_output(args.output_dir, ROOT)
        provenance = collect_provenance(
            ROOT, [Path(__file__), ROOT / 'scripts/validate_inverse_mne.py',
                   ROOT / 'scripts/validate_patch_forward_mne.py', ROOT / 'scripts/_validation_outputs.py',
                   ROOT / 'scripts/run_split_study.py',
                   ROOT / 'validation-requirements.txt', ROOT / 'validation-windows-lock.txt'],
            list(sys.argv[1:] if argv is None else argv), require_clean=not args.smoke)
    except ValueError as error:
        parser.error(str(error))
    provenance['inputs'] = {} if args.geometry is None else {'geometry.json': hash_file(args.geometry)}
    provenance['library_kernels'] = library_kernels()

    import inspect

    import mne

    import mne.bem
    import mne.dipole

    import validate_inverse_mne as inverse
    import validate_patch_forward_mne as patch
    from run_split_study import record_output_hashes, record_provenance, write_json

    if not (hasattr(mne.bem, 'fmin_cobyla') and hasattr(mne.dipole, 'fmin_cobyla')):
        # MNE-Python 1.13 imports the optimizer inside the functions that make the fits.
        parser.error(f'this script replaces the optimizer call inside MNE-Python 1.12; MNE-Python {mne.__version__} '
                     'looks the optimizer up elsewhere. scripts/check_installed_sphere_fit.py runs with any version.')
    mne.set_log_level('ERROR')
    sizes = dict(dipoles=8, sweep_levels=[20], sweep_dipoles=6, samples=3, cortical=1500, subcortical=300,
                 perturbations=min(args.perturbations, 1)) if args.smoke else dict(
        dipoles=inverse.DEFAULT_N_TEST, sweep_levels=list(inverse.DEFAULT_SWEEP_DB),
        sweep_dipoles=inverse.DEFAULT_SWEEP_N, samples=60, cortical=inverse.DEFAULT_POOL_CORTICAL,
        subcortical=inverse.DEFAULT_POOL_SUB, perturbations=args.perturbations)
    output.mkdir(parents=True, exist_ok=False)
    record_provenance(output, provenance)
    epsilon = float(np.finfo(float).eps)
    steps = range(1, sizes['perturbations'] + 1)
    small_steps = range(1, min(sizes['perturbations'], SMALL_TOLERANCE_TRIALS) + 1)
    own_kernels = library_kernels()
    own_name = '+'.join(sorted({entry['kernel'] for entry in own_kernels})) or 'unknown'

    with tempfile.TemporaryDirectory(prefix='.sphere-fit-data-', dir=output) as work:
        work = Path(work)
        print('generating the datasets of the two validation scripts ...', flush=True)
        pools = (inverse.DEFAULT_MONTAGE, SEED, sizes['cortical'], sizes['subcortical'])
        headline, radii, sigmas = inverse.generate_test_split(
            inverse.DEFAULT_SNR_DB, sizes['dipoles'], *pools, work, tag='headline')
        sweep = [(level, inverse.generate_test_split(level, sizes['sweep_dipoles'], *pools, work)[0])
                 for level in sorted(sizes['sweep_levels'])]
        patch_config = patch.generate('patch', sizes['samples'], work / 'patch', SEED)
        patch.generate('point', sizes['samples'], work / 'point', SEED)
        data = dict(headline=str(headline), sweep=[[float(level), str(path)] for level, path in sweep],
                    radii=[float(value) for value in radii], sigmas=[float(value) for value in sigmas],
                    samples=sizes['samples'], patch_dir=str(work / 'patch'), point_dir=str(work / 'point'),
                    sampling_freq_hz=float(patch_config.signal.sampling_freq_hz))
        digests = dict(
            dipole_recovery_35db=array_digest(headline, 'eeg'),
            snr_sweep={f'{level:g}': array_digest(path, 'eeg') for level, path in sweep},
            patch=array_digest(work / 'patch/train.h5', 'ground_truth/eeg_signal_only'),
            point=array_digest(work / 'point/train.h5', 'ground_truth/eeg_signal_only'))

        # Parts 1 and 2: the fit at the limit of SciPy and with the limit raised.
        unmodified = sphere_model(radii, sigmas)
        arms, arrays = {}, {}
        for name, limit in FITS:
            arms[name], arrays[name] = evaluate(limit, 1.0, data, with_sweep=True)
            fit, position = arms[name]['fit'], arms[name]['dipole_recovery']['position_error_mm']
            print(f"{name}: {fit['evaluations']} evaluations, stopped by itself: {fit['stopped_by_itself']}, "
                  f"objective {fit['objective']:.3e}; field difference patch "
                  f"{arms[name]['patch_comparison']['patch']['median']:.2e} / point "
                  f"{arms[name]['patch_comparison']['point']['median']:.2e}; position error median "
                  f"{position['median']:.3f} mm, p90 {position['p90']:.3f}, maximum {position['maximum']:.3f}",
                  flush=True)
        same = bool(np.array_equal(unmodified['mu'], arms['mne_limit']['fit']['mu'])
                    and np.array_equal(unmodified['lambda'], arms['mne_limit']['fit']['lambda_']))
        if not same:
            raise RuntimeError('the instrumented call at the limit of 1,000 evaluations does not reproduce '
                               'the fit of the installed MNE-Python')
        write_per_window_files(output, arrays)
        unchanged = {name: arrays[name]['recovery'] for name, _ in FITS}
        rows = []
        per_dipole = [row for name, _ in FITS for row in dipole_rows('none', name, 0, unchanged[name])]

        # Part 3: the objective of the parameter fit scaled by 1 + j * eps.
        per_field = []
        for step in steps:
            for name, limit in FITS:
                summary, trial = evaluate(limit, 1.0 + step * epsilon, data)
                change = position_change(trial['recovery']['estimated_xyz'], unchanged[name]['estimated_xyz'])
                rows.append(perturbation_row('sphere_fit', name, step, summary, change))
                per_dipole += dipole_rows('sphere_fit', name, step, trial['recovery'])
                per_field += [[name, step, kind, index, repr(float(value))]
                              for kind in ('patch', 'point') for index, value in enumerate(trial[kind])]
                print(f"parameter fit {name}, objective times 1 + {step} eps: {summary['fit']['evaluations']} "
                      f"evaluations; patch {summary['patch_comparison']['patch']['median']:.2e}; position error "
                      f"median {summary['dipole_recovery']['position_error_mm']['median']:.3f} mm, p90 "
                      f"{summary['dipole_recovery']['position_error_mm']['p90']:.3f}", flush=True)

        # Part 4: the objective of the dipole fits scaled by the same factors, with either sphere model.
        dipole_fits, dipole_same = {}, True
        for name, limit in FITS:
            log = []
            _, repeated = evaluate(limit, 1.0, data, fields=False, dipole_scale=1.0, dipole_log=log)
            dipole_same = dipole_same and bool(np.array_equal(repeated['recovery']['estimated_xyz'],
                                                              unchanged[name]['estimated_xyz']))
            dipole_fits[name] = dict(dipole_fit_record(log),
                                     final_radius_m=sorted({entry['rhoend'] for entry in log}))
            print(f"dipole fits with {name}: {dipole_fits[name]['stopped_by_themselves']} of "
                  f"{dipole_fits[name]['fits']} stopped by themselves, median "
                  f"{dipole_fits[name]['evaluations']['median']:.0f} evaluations", flush=True)
        if not dipole_same:
            raise RuntimeError('the instrumented dipole fit does not reproduce the dipole fit of the installed '
                               'MNE-Python')
        for step in steps:
            for name, limit in FITS:
                log = []
                summary, trial = evaluate(limit, 1.0, data, fields=False, dipole_scale=1.0 + step * epsilon,
                                          dipole_log=log)
                change = position_change(trial['recovery']['estimated_xyz'], unchanged[name]['estimated_xyz'])
                rows.append(perturbation_row('dipole_fit', name, step, summary, change, dipole_fit_record(log)))
                per_dipole += dipole_rows('dipole_fit', name, step, trial['recovery'])
                print(f"dipole fits with {name}, objective times 1 + {step} eps: position error median "
                      f"{summary['dipole_recovery']['position_error_mm']['median']:.3f} mm, p90 "
                      f"{summary['dipole_recovery']['position_error_mm']['p90']:.3f}; fitted positions moved by a "
                      f"median of {change['median']:.3f} mm, at most {change['maximum']:.3f} mm", flush=True)
        write_rows(output / 'perturbation.csv', PERTURBATION_HEADER, rows)
        write_rows(output / 'perturbation_dipoles.csv', DIPOLE_HEADER, per_dipole)
        write_rows(output / 'perturbation_fields.csv', FIELD_HEADER, per_field)

        # Part 5: the dipole fits ended at a small trust-region radius.
        small_rows, small_dipoles, small_unchanged = [], [], {}

        def small_run(changed, name, limit, step, scale, dipole_scale):
            log = []
            summary, trial = evaluate(limit, scale, data, fields=False, dipole_scale=dipole_scale, dipole_log=log,
                                      dipole_tolerance=SMALL_TOLERANCE_M)
            if {entry['rhoend'] for entry in log} != {SMALL_TOLERANCE_M}:
                raise RuntimeError('the dipole fits did not receive the small final radius')
            recovery = trial['recovery']
            change = None if changed == 'none' else position_change(recovery['estimated_xyz'],
                                                                     small_unchanged[name]['estimated_xyz'])
            if changed == 'none':
                small_unchanged[name] = recovery
            small_rows.append(small_tolerance_row(own_name, changed, name, step, summary['fit'], recovery,
                                                  dipole_fit_record(log), change))
            small_dipoles.extend(small_tolerance_dipoles(own_name, changed, name, step, recovery['estimated_xyz'],
                                                         recovery['err_mm']))
            position = summary['dipole_recovery']['position_error_mm']
            print(f"small final radius, {name}, {changed} {step}: position error median {position['median']:.4f} "
                  f"mm, p90 {position['p90']:.4f}, maximum {position['maximum']:.4f}", flush=True)

        for name, limit in FITS:
            small_run('none', name, limit, 0, 1.0, 1.0)
        for step in small_steps:
            for name, limit in FITS:
                small_run('parameter_fit', name, limit, step, 1.0 + step * epsilon, 1.0)
                small_run('dipole_fits', name, limit, step, 1.0, 1.0 + step * epsilon)

        # Part 6: other kernels of the linear-algebra library, and the value at the second trial point.
        own_names = {entry['kernel'].lower() for entry in own_kernels}
        kernels, not_available, kernel_dipoles, kernel_fields = {}, {}, [], []
        requested = list(args.kernels)[:2] if args.smoke else list(args.kernels)
        for kernel in requested:
            if args.smoke and kernels:
                break
            if kernel.lower() in own_names:
                not_available[kernel] = 'the kernel of the parent process'
                continue
            record, stored, problem = run_kernel(kernel, data, work)
            if record is None:
                not_available[kernel] = problem
                continue
            reported = {entry['kernel'].lower() for entry in record['libraries']}
            if reported & own_names or not record['libraries']:
                not_available[kernel] = 'the library kept the kernel of the parent process'
                continue
            if reported != {kernel.lower()}:
                not_available[kernel] = f"the library reports the kernel {' and '.join(sorted(reported))}"
                continue
            second = record['arms']['mne_limit']['fit']['first_evaluations'][1]
            record['parent_fit_with_the_second_value_of_this_kernel'] = None
            if second['mu'] == arms['mne_limit']['fit']['first_evaluations'][1]['mu']:
                exchanged = []
                with instrumented_fit(MNE_LIMIT, 1.0, exchanged, replace={2: second['objective']}):
                    model = sphere_model(radii, sigmas)
                record['parent_fit_with_the_second_value_of_this_kernel'] = dict(
                    replaced_evaluation=2, mu_of_that_evaluation=second['mu'], value=second['objective'],
                    evaluations=exchanged[0]['evaluations'], stopped_by_itself=exchanged[0]['stopped_by_itself'],
                    objective=exchanged[0]['objective'], mu=[float(value) for value in model['mu']])
            record['fitted_position_change_from_the_parent_mm'] = {
                name: position_change(stored[f'{name}_fitted_m'], unchanged[name]['estimated_xyz'])
                for name, _ in FITS}
            for name, _ in FITS:
                entry = record['small_tolerance'][name]
                recovery = dict(estimated_xyz=stored[f'small_{name}_fitted_m'],
                                err_mm=stored[f'small_{name}_position_error_mm'])
                change = position_change(recovery['estimated_xyz'], small_unchanged[name]['estimated_xyz'])
                entry['fitted_position_change_from_the_parent_mm'] = change
                small_rows.append(small_tolerance_row(kernel, 'none', name, 0, entry['fit'], recovery,
                                                      entry['dipole_fits'], change))
                small_dipoles.extend(small_tolerance_dipoles(kernel, 'none', name, 0, recovery['estimated_xyz'],
                                                             recovery['err_mm']))
            kernels[kernel] = record
            dipoles, fields = kernel_rows(kernel, stored)
            kernel_dipoles += dipoles
            kernel_fields += fields
            for name, _ in FITS:
                arm = record['arms'][name]
                print(f"kernel {kernel}, {name}: {arm['fit']['evaluations']} evaluations, objective "
                      f"{arm['fit']['objective']:.3e}; field difference patch "
                      f"{arm['patch_comparison']['patch']['median']:.2e} / point "
                      f"{arm['patch_comparison']['point']['median']:.2e}; position error median "
                      f"{arm['dipole_recovery']['position_error_mm']['median']:.3f} mm, p90 "
                      f"{arm['dipole_recovery']['position_error_mm']['p90']:.3f}; with the small final radius "
                      f"{record['small_tolerance'][name]['position_error_mm']['median']:.4f} mm", flush=True)
        write_rows(output / 'kernels_dipole_recovery_35db.csv',
                   ['kernel', 'sphere_fit', 'sample_index', *DIPOLE_VALUES], kernel_dipoles)
        write_rows(output / 'kernels_field_differences.csv',
                   ['kernel', 'sphere_fit', 'source', 'sample_index', 'relative_field_difference'], kernel_fields)
        write_rows(output / 'small_tolerance.csv', SMALL_TOLERANCE_HEADER, small_rows)
        write_rows(output / 'small_tolerance_dipoles.csv', SMALL_TOLERANCE_DIPOLE_HEADER, small_dipoles)

        # Part 7: the spheres of run_head_adaptation.py.
        spheres = None if args.geometry is None else spheres_part(args.geometry, sizes, args.smoke, output, work)

    write_json(output / 'summary.json', dict(
        smoke=args.smoke,
        head_model=dict(radii_m=[float(value) for value in radii],
                        conductivities_S_per_m=[float(value) for value in sigmas]),
        optimizer_call=dict(function='scipy.optimize.fmin_cobyla, called by mne.bem._fwd_eeg_fit_berg_scherg',
                            arguments=arms['mne_limit']['fit']['call'],
                            evaluation_limit='none passed; the default of SciPy applies'),
        evaluation_limits=dict(FITS),
        instrumented_call_reproduces_installed_mne_python=same,
        instrumented_dipole_fit_reproduces_installed_mne_python=dipole_same,
        sizes=sizes, seed=SEED, dataset_array_sha256=digests,
        arms=arms, default_sphere_model_of_mne_python=default_sphere_of_mne_python(), dipole_fits=dipole_fits,
        perturbation=dict(objective_factor='1 + j * eps, eps = 2**-52', steps=list(steps),
                          fitted_position_change='distance from the fitted position of the same dipole in the '
                                                 'unchanged run of the same sphere fit',
                          ranges=perturbation_ranges(rows)),
        small_tolerance=dict(
            final_radius_m=SMALL_TOLERANCE_M,
            final_radius_of_mne_python_m=float(inspect.signature(mne.fit_dipole).parameters['tol'].default),
            steps=list(small_steps), kernel_of_the_main_process=own_name,
            fitted_position_change='distance from the fitted position of the same dipole in the unchanged run of '
                                   'the same sphere fit in the main process',
            ranges=small_tolerance_ranges(small_rows)),
        library_kernels=dict(variable=KERNEL_VARIABLE, of_the_parent_process=own_kernels, requested=requested,
                             arms=kernels, not_available=not_available),
        spheres=spheres,
        software_environment=environment_snapshot()))
    record_output_hashes(output)
    print(f'wrote {output}')


if __name__ == '__main__':
    main()
