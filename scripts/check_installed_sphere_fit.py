#!/usr/bin/env python
"""How the Berg-Scherg parameter fit of the installed MNE-Python ends for the spheres of the paper.

    python scripts/check_installed_sphere_fit.py --output-dir validation_runs/sphere-fit-installed \\
        --geometry path/to/evidence/head-adaptation-v11/geometry.json

run_sphere_fit_convergence.py replaces the optimizer call inside MNE-Python
1.12.1 and belongs to that version. This script changes nothing in the fit. It
counts the function evaluations of SciPy's COBYLA while the installed
MNE-Python builds its sphere model, once with SciPy's default limit of 1,000
evaluations, which is what MNE-Python uses, and once with a limit of 100,000.
It therefore runs with any version of MNE-Python: version 1.13 imports the
optimizer inside the function that makes the fit and keeps the result of a
fit, where version 1.12 imported the optimizer into its module.

It records, for the default head model of the package, for the sphere that
MNE-Python builds when no head model is given, and with --geometry for every
sphere of run_head_adaptation.py: the arguments of the optimizer call, the
number of evaluations, whether the optimizer stopped before the limit, the
objective at the end and the fitted parameters. It stops if the fit that it
counted at the default limit is not the fit of the installed MNE-Python. The
same is recorded in child processes that use another kernel of OpenBLAS,
selected with the environment variable OPENBLAS_CORETYPE. No dipole is fitted
and no dataset is generated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _run_provenance import check_new_output, collect_provenance, hash_file

MNE_LIMIT, RAISED_LIMIT = 1000, 100_000
FITS = (('mne_limit', MNE_LIMIT), ('raised_limit', RAISED_LIMIT))
KERNEL_VARIABLE = 'OPENBLAS_CORETYPE'
KERNELS = ('Haswell', 'Sandybridge', 'Nehalem')


@contextmanager
def counted_fits(limit, log):
    """Count the evaluations of every COBYLA call that MNE-Python makes inside the block.

    The call is passed on to SciPy as MNE-Python made it, with the objective
    wrapped in a counter and with ``limit`` as the evaluation limit. A call
    that brings a limit of its own is refused, because the limit would no
    longer be the one of this script. Every call appends a record to ``log``.
    MNE-Python 1.13 keeps the result of a fit, so what it has kept is dropped
    before the block and after it.
    """
    import mne.bem
    import scipy.optimize

    original = scipy.optimize.fmin_cobyla

    def fmin_cobyla(func, x0, cons, *args, **options):
        if args or 'maxfun' in options:
            raise TypeError('the fit of MNE-Python passed an evaluation limit or positional options')
        evaluations = 0

        def counted(x, *extra):
            nonlocal evaluations
            evaluations += 1
            return func(x, *extra)

        result = original(counted, x0, cons, maxfun=limit, **options)
        log.append(dict(evaluations=evaluations, stopped_before_the_limit=evaluations < limit,
                        objective=float(func(result)),
                        call=dict(start=[float(item) for item in x0], constraints=len(cons),
                                  **{name: float(value) for name, value in sorted(options.items())
                                     if isinstance(value, (int, float)) and not isinstance(value, bool)})))
        return result

    # MNE-Python 1.12 looks the optimizer up in its own module, later versions in scipy.optimize.
    homes = [scipy.optimize] + ([mne.bem] if hasattr(mne.bem, 'fmin_cobyla') else [])
    kept = getattr(mne.bem, '_fit_berg_scherg_cached', None)

    def forget():
        if kept is not None and hasattr(kept, 'cache_clear'):
            kept.cache_clear()

    saved = [(home, home.fmin_cobyla) for home in homes]
    for home, _ in saved:
        home.fmin_cobyla = fmin_cobyla
    forget()
    try:
        yield
    finally:
        for home, function in saved:
            home.fmin_cobyla = function
        forget()


def package_sphere(radii, sigmas):
    """The sphere model as the validation scripts of the package build it."""
    import mne

    radii = np.asarray(radii, dtype=float)
    return mne.make_sphere_model(r0=(0.0, 0.0, 0.0), head_radius=float(radii[-1]),
                                 relative_radii=tuple(radii / radii[-1]), sigmas=tuple(sigmas), verbose=False)


def default_sphere():
    """The sphere model that MNE-Python builds when no head model is given."""
    import mne

    return mne.make_sphere_model(verbose=False)


def fit_record(build):
    """How the parameter fit of one sphere ends with either limit, and whether the counted fit is the installed one."""
    installed = build()
    record = {}
    for name, limit in FITS:
        log = []
        with counted_fits(limit, log):
            model = build()
        if len(log) != 1:
            raise RuntimeError(f'MNE-Python made {len(log)} optimizer calls for one sphere model')
        record[name] = dict(log[0], evaluation_limit=limit, mu=[float(value) for value in model['mu']],
                            lambda_=[float(value) for value in model['lambda']])
        if name == 'mne_limit' and not (np.array_equal(model['mu'], installed['mu'])
                                        and np.array_equal(model['lambda'], installed['lambda'])):
            raise RuntimeError('the counted fit at the default limit is not the fit of the installed MNE-Python')
    record['layers'] = [dict(radius_m=float(layer['rad']), conductivity_S_per_m=float(layer['sigma']))
                        for layer in installed['layers']]
    return record


def library_kernels():
    """The kernel that every loaded linear-algebra library reports."""
    from threadpoolctl import threadpool_info

    return [dict(library=str(entry.get('internal_api')), version=str(entry.get('version')),
                 kernel=str(entry.get('architecture')))
            for entry in threadpool_info() if entry.get('user_api') == 'blas']


def all_fits(spheres):
    """The record of every sphere under the kernel of this process."""
    return dict(libraries=library_kernels(),
                spheres={name: fit_record(lambda head=head: package_sphere(head['radii_m'],
                                                                           head['conductivities_S_per_m']))
                         for name, head in spheres.items()},
                default_sphere_model_of_mne_python=fit_record(default_sphere))


def under_kernel(kernel, spheres):
    """The same record from a child process that uses another kernel of OpenBLAS; None with the reason if it fails."""
    completed = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()), '--child'],
                               input=json.dumps(spheres), capture_output=True, text=True, encoding='utf-8',
                               env={**os.environ, KERNEL_VARIABLE: kernel})
    if completed.returncode != 0:
        lines = completed.stderr.strip().splitlines()
        return None, f"the child process failed: {lines[-1] if lines else 'no message'}"
    return json.loads(completed.stdout), None


def versions():
    import platform

    import mne
    import scipy

    return dict(python=platform.python_version(), mne=mne.__version__, scipy=scipy.__version__,
                numpy=np.__version__)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output-dir', type=Path, help='new result directory')
    parser.add_argument('--geometry', type=Path,
                        help='geometry.json of a run of run_head_adaptation.py; adds its spheres')
    parser.add_argument('--kernels', nargs='*', default=list(KERNELS), metavar='NAME',
                        help=f'values of {KERNEL_VARIABLE} for the child processes; kernels that the processor does '
                             f'not support, that the library does not have, or that equal the kernel in use are '
                             f'recorded as not available (default: {" ".join(KERNELS)})')
    parser.add_argument('--smoke', action='store_true',
                        help='the default head model only, one other kernel; relaxed provenance')
    parser.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.child:
        import mne

        mne.set_log_level('ERROR')
        print(json.dumps(all_fits(json.loads(sys.stdin.read()))))
        return
    if args.output_dir is None:
        parser.error('--output-dir is required')
    if args.geometry is not None and not args.geometry.is_file():
        parser.error(f'--geometry: no such file: {args.geometry}')
    try:
        output = check_new_output(args.output_dir, ROOT)
        provenance = collect_provenance(ROOT, [Path(__file__)], list(sys.argv[1:] if argv is None else argv),
                                        require_clean=not args.smoke)
    except ValueError as error:
        parser.error(str(error))
    provenance['inputs'] = {} if args.geometry is None else {'geometry.json': hash_file(args.geometry)}
    provenance['smoke'] = args.smoke

    import mne

    from spherical_eeg_sim import Config

    mne.set_log_level('ERROR')
    head = Config().head
    spheres = dict(default=dict(radii_m=[float(value) for value in head.radii_m],
                                conductivities_S_per_m=[float(value) for value in head.conductivities_S_per_m]))
    if args.geometry is not None and not args.smoke:
        for name, variant in json.loads(args.geometry.read_text(encoding='utf-8'))['variants'].items():
            sphere = dict(radii_m=[float(value) for value in variant['radii_m']],
                          conductivities_S_per_m=[float(value) for value in variant['conductivities_S_per_m']])
            if name == 'default' and sphere != spheres['default']:
                parser.error('--geometry: its default sphere is not the default head model of the package')
            spheres[name] = sphere
    own = all_fits(spheres)
    own_names = {entry['kernel'].lower() for entry in own['libraries']}
    # The experiment script names the kernel of the main process by this rule.
    own_name = '+'.join(sorted({entry['kernel'] for entry in own['libraries']})) or 'unknown'
    others, not_available = {}, {}
    for kernel in list(args.kernels)[:2] if args.smoke else list(args.kernels):
        if args.smoke and others:
            break
        if kernel.lower() in own_names:
            not_available[kernel] = 'the kernel of the parent process'
            continue
        record, problem = under_kernel(kernel, spheres)
        if record is None:
            not_available[kernel] = problem
            continue
        reported = {entry['kernel'].lower() for entry in record['libraries']}
        if reported & own_names or not record['libraries']:
            not_available[kernel] = 'the library kept the kernel of the parent process'
        elif reported != {kernel.lower()}:
            not_available[kernel] = f"the library reports the kernel {' and '.join(sorted(reported))}"
        else:
            others[kernel] = record

    output.mkdir(parents=True)
    write_json(output / 'provenance.json', provenance)
    (output / 'environment.txt').write_text(provenance['resolved_packages'], encoding='utf-8')
    write_json(output / 'summary.json', dict(
        smoke=args.smoke, versions=versions(), evaluation_limits=dict(FITS),
        optimizer=dict(function='scipy.optimize.fmin_cobyla',
                       imported_into_the_module_of_mne_python=hasattr(mne.bem, 'fmin_cobyla'),
                       mne_python_keeps_the_result_of_a_fit=hasattr(mne.bem, '_fit_berg_scherg_cached')),
        counted_fit_at_the_default_limit_is_the_installed_fit=True,
        kernel_of_the_main_process=own_name, kernel_variable=KERNEL_VARIABLE, requested_kernels=list(args.kernels),
        **own, other_kernels=others, not_available=not_available))
    write_json(output / 'output_sha256.json',
               {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(output.iterdir())})
    for name, record in {**own['spheres'], 'sphere of MNE-Python': own['default_sphere_model_of_mne_python']}.items():
        first, second = record['mne_limit'], record['raised_limit']
        print(f"{name}: {first['evaluations']} evaluations with the default limit "
              f"({'stopped by itself' if first['stopped_before_the_limit'] else 'reached the limit'}, objective "
              f"{first['objective']:.3e}); {second['evaluations']} with the raised limit (objective "
              f"{second['objective']:.3e})")
    print(f'wrote {output}')


if __name__ == '__main__':
    main()
