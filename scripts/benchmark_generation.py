"""Measure dataset generation time, peak process-tree RSS, and output size.

Each case runs the generator in a child process and is repeated, so the report
carries a median and a range. Suites:

  modes          point, multi-point and patch sources, compressed HDF5, one worker
  workers        point sources with each requested worker count
  formats        uncompressed HDF5, NPZ and MATLAB v7.3, which hold a split in memory
  default-scale  the unmodified default configuration (360,000 windows), one run per worker count

Run it on an otherwise idle machine. Datasets are written to a scratch
directory and deleted after each measurement.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spherical_eeg_sim import Config
from spherical_eeg_sim.core.electrodes import load_electrodes
from spherical_eeg_sim.provenance import environment_snapshot
from _run_provenance import check_new_output, collect_provenance, hash_file

SUITES = ('modes', 'workers', 'formats', 'default-scale')
RSS_PERIOD_SECONDS = 0.05


def cpu_model():
    if sys.platform == 'win32':
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
                return winreg.QueryValueEx(key, 'ProcessorNameString')[0].strip()
        except OSError:
            pass
    return platform.processor()


def peak_process_tree_rss(pid):
    try:
        parent = psutil.Process(pid)
        processes = [parent, *parent.children(recursive=True)]
    except psutil.NoSuchProcess:
        return 0
    rss = 0
    for process in processes:
        try:
            rss += process.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    return rss


def case_config(case):
    """The generator configuration of one benchmark case (without an output directory)."""
    cfg = Config()
    cfg.source_mode = case['mode']
    if case['n_train'] is not None:
        cfg.dataset.n_train = case['n_train']
        cfg.dataset.n_val = cfg.dataset.n_test = max(2, case['n_train'] // 10)
    cfg.output.n_workers = case['workers']
    cfg.output.format = case['format']
    cfg.output.compress = case['compress']
    cfg.output.diagnostic_plots = False
    return cfg


def measure(run_dir, scratch_dir, case, repetition):
    cfg = case_config(case)
    label = f"{case['case']}-r{repetition}"
    with tempfile.TemporaryDirectory(prefix=f'{label}-', dir=scratch_dir) as scratch:
        dataset = Path(scratch) / 'dataset'
        cfg.output_dir = str(dataset)
        config = run_dir / f'{label}.json'
        cfg.to_json(config)
        stages = run_dir / f'{label}.stages.json'
        command = [sys.executable, '-B', str(ROOT / 'scripts/_benchmark_child.py'), str(config), str(stages)]
        available_before = psutil.virtual_memory().available
        cpu_before = psutil.cpu_percent(interval=1.0)
        with (run_dir / f'{label}.log').open('w', encoding='utf-8') as log:
            start = time.perf_counter()
            with subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT) as child:
                peak = 0
                samples = 0
                minimum_available = available_before
                while child.poll() is None:
                    peak = max(peak, peak_process_tree_rss(child.pid))
                    samples += 1
                    minimum_available = min(minimum_available, psutil.virtual_memory().available)
                    time.sleep(RSS_PERIOD_SECONDS)
                elapsed = time.perf_counter() - start
                if child.returncode:
                    raise RuntimeError(f'{label} failed. See {log.name}.')
        validation = json.loads((dataset / 'validation_report.json').read_text(encoding='utf-8'))
        size = sum(p.stat().st_size for p in dataset.rglob('*') if p.is_file())
    phases = json.loads(stages.read_text(encoding='utf-8'))
    windows = cfg.dataset.n_train + cfg.dataset.n_val + cfg.dataset.n_test
    result = dict(repetition=repetition, seconds=elapsed, windows_per_second=windows / elapsed,
                  peak_process_tree_rss_bytes=peak, output_bytes=size,
                  available_ram_before_bytes=available_before,
                  minimum_available_ram_during_bytes=minimum_available,
                  host_cpu_percent_before=cpu_before, rss_samples=samples, phase_seconds=phases,
                  setup_seconds=phases.get('basis_forward', 0.0) + phases.get('background_pool', 0.0))
    return result, validation.get('patch_overlap')


def spread(values):
    return dict(median=float(statistics.median(values)), minimum=float(min(values)), maximum=float(max(values)))


def build_cases(args):
    cases = []

    def add(label, mode, workers, n_train, file_format, compress, repeats):
        cases.append(dict(case=label, mode=mode, workers=workers, n_train=n_train,
                          format=file_format, compress=compress, repeats=repeats))

    if 'modes' in args.suite:
        add('point-1', 'point', 1, args.n_train, 'hdf5', True, args.repeats)
        add('multi-point-1', 'multi_point', 1, args.n_train, 'hdf5', True, args.repeats)
        add('patch-1', 'patch', 1, args.n_train, 'hdf5', True, args.repeats)
    if 'workers' in args.suite:
        if 'modes' not in args.suite:
            add('point-1', 'point', 1, args.n_train, 'hdf5', True, args.repeats)
        for workers in args.workers:
            add(f'point-{workers}', 'point', workers, args.n_train, 'hdf5', True, args.repeats)
    if 'formats' in args.suite:
        if not any(case['case'] == 'point-1' for case in cases):
            add('point-1', 'point', 1, args.n_train, 'hdf5', True, args.repeats)
        add('hdf5-uncompressed-1', 'point', 1, args.n_train, 'hdf5', False, args.repeats)
        add('npz-1', 'point', 1, args.n_train, 'npz', True, args.repeats)
        add('mat-1', 'point', 1, args.n_train, 'mat', True, args.repeats)
    if 'default-scale' in args.suite:
        for workers in (1, *args.workers):
            add(f'default-scale-{workers}', 'point', workers, None, 'hdf5', True, 1)
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output-dir', required=True, type=Path, help='new folder for the report and logs')
    parser.add_argument('--suite', nargs='+', choices=(*SUITES, 'all'), default=['modes', 'workers'],
                        help='one or more suites (default: modes workers)')
    parser.add_argument('--n-train', default=10_000, type=int,
                        help='training windows of the fixed-size cases; validation and test get one tenth each')
    parser.add_argument('--workers', type=int, nargs='+', default=[6],
                        help='worker counts of the parallel cases (default: 6)')
    parser.add_argument('--repeats', type=int, default=3,
                        help='repetitions of every fixed-size case (default: 3)')
    parser.add_argument('--scratch-dir', type=Path,
                        help='where temporary datasets are written (default: the output folder)')
    parser.add_argument('--minimum-available-gib', type=float, default=2.0,
                        help='refuse to start a case with less free memory than this (default: 2)')
    args = parser.parse_args()
    if 'all' in args.suite:
        args.suite = list(SUITES)
    if args.n_train < 2 or args.repeats < 1 or args.minimum_available_gib < 0:
        parser.error('--n-train must be at least 2, --repeats at least 1 and the memory floor nonnegative')
    if any(workers < 2 for workers in args.workers):
        parser.error('--workers lists parallel worker counts of at least 2')
    try:
        output = check_new_output(args.output_dir, ROOT)
        provenance = collect_provenance(ROOT, [Path(__file__), ROOT / 'scripts/_benchmark_child.py',
                                        ROOT / 'validation-windows-lock.txt'], sys.argv[1:], require_clean=True)
    except ValueError as error:
        parser.error(str(error))
    output.mkdir(parents=True, exist_ok=False)
    scratch_dir = output if args.scratch_dir is None else args.scratch_dir.resolve()
    scratch_dir.mkdir(parents=True, exist_ok=True)
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
    battery = psutil.sensors_battery()
    default = Config()
    channels = load_electrodes(default.electrode_file, target_radius_m=default.head.radii_m[-1]).n_electrodes
    report = dict(environment=environment_snapshot(), platform=platform.platform(),
                  cpu=cpu_model(), logical_cpus=psutil.cpu_count(),
                  physical_cpus=psutil.cpu_count(logical=False), ram_bytes=psutil.virtual_memory().total,
                  available_ram_bytes=psutil.virtual_memory().available,
                  power_plugged=None if battery is None else battery.power_plugged,
                  channels=channels, window_samples=default.signal.window_samples,
                  sampling_freq_hz=default.signal.sampling_freq_hz, eeg_dtype=default.output.dtype,
                  rss_sample_period_seconds=RSS_PERIOD_SECONDS,
                  rss_metric=('Peak sampled sum of resident memory over the generator process and its '
                              'children. Shared pages may be counted in several processes; samples can '
                              'miss short peaks.'),
                  timing_scope=('Wall time of the whole generator process, from interpreter start to exit, '
                                'including validation and publication. Stage times are measured inside it; '
                                'split generation and writing includes the HDF5 append and finalize times.'),
                  threads={key: os.getenv(key) for key in (
                      'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')},
                  repeats=args.repeats, cases=[])
    for case in build_cases(args):
        cfg = case_config(case)
        windows = cfg.dataset.n_train + cfg.dataset.n_val + cfg.dataset.n_test
        needed = windows * channels * default.signal.window_samples * 8 * 1.3
        entry = dict(case, train=cfg.dataset.n_train, validation=cfg.dataset.n_val, test=cfg.dataset.n_test,
                     windows=windows, repetitions=[])
        del entry['n_train'], entry['repeats']
        for repetition in range(1, case['repeats'] + 1):
            if psutil.virtual_memory().available < args.minimum_available_gib * 2**30:
                parser.error('Available memory is below --minimum-available-gib; '
                             'completed cases are retained in the run folder')
            if shutil.disk_usage(scratch_dir).free < needed:
                parser.error(f'{case["case"]} needs about {needed / 2**30:.1f} GiB of free scratch space')
            result, overlap = measure(output, scratch_dir, case, repetition)
            entry['repetitions'].append(result)
            if overlap is not None and 'patch_overlap' not in entry:
                entry['patch_overlap'] = overlap
            print(f"{case['case']} repetition {repetition}: {result['seconds']:.1f} s, "
                  f"{result['windows_per_second']:.1f} windows/s", flush=True)
        runs = entry['repetitions']
        sizes = {run['output_bytes'] for run in runs}
        if len(sizes) != 1:
            raise RuntimeError(f'{case["case"]}: output size changed between repetitions: {sorted(sizes)}')
        middle = sorted(runs, key=lambda run: run['seconds'])[len(runs) // 2]
        entry['summary'] = dict(
            seconds=spread([run['seconds'] for run in runs]),
            windows_per_second=spread([run['windows_per_second'] for run in runs]),
            peak_process_tree_rss_bytes=spread([run['peak_process_tree_rss_bytes'] for run in runs]),
            output_bytes=runs[0]['output_bytes'],
            median_run_phase_seconds=middle['phase_seconds'],
            median_run_setup_seconds=middle['setup_seconds'],
        )
        report['cases'].append(entry)
        (output / 'generation_benchmark.json').write_text(
            json.dumps(report, indent=2) + '\n', encoding='utf-8')
    inventory = {path.name: hash_file(path) for path in sorted(output.iterdir()) if path.is_file()}
    (output / 'output_sha256.json').write_text(json.dumps(inventory, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
