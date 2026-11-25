"""Measure coordinate reuse when test locations are redrawn from a shared pool.

Generates the seed-862 supervised-study dataset under the ``shared`` split
policy for each requested fraction of the position pool and reports, from the
dataset's own validation report, how many test windows fall on a location that
training also used. A fraction of 1.0 is a full independent redraw of test
locations from the training pool.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spherical_eeg_sim import run_pipeline
from run_split_study import DATA_SEEDS, config_for, record_output_hashes, record_provenance, write_json
from _run_provenance import check_new_output, collect_provenance

KEPT_DATASET_FILES = ('checksums.sha256', 'validation_report.json', 'metadata.json')


def run_fraction(fraction, folder, smoke, keep):
    dataset = folder / 'dataset'
    cfg = config_for(DATA_SEEDS[0], 'position', dataset, smoke)
    cfg.split.policy = 'shared'
    cfg.split.shared_position_fraction = fraction
    cfg.output.save_components = False
    cfg.output.save_leadfield = False
    cfg.to_json(folder / 'config.json')
    run_pipeline(cfg, log_level=logging.WARNING)
    shared = json.loads((dataset / 'validation_report.json').read_text(encoding='utf-8'))['source_split']['shared']
    if not keep:
        for path in dataset.iterdir():
            if path.name not in KEPT_DATASET_FILES:
                shutil.rmtree(path) if path.is_dir() else path.unlink()
    reuse = shared['observed_reuse']
    return dict(
        requested_fraction=fraction, realized_fraction=shared['realized_common_pool_fraction'],
        shared_positions_by_population=shared['common_positions_by_population'],
        eligible_positions=shared['eligible_positions'], used_positions=shared['used_unique_positions'],
        positions_used_by_training_and_test=reuse['test_to_train']['used_position_overlap'],
        test_windows=reuse['test_to_train']['query_windows'],
        test_windows_at_a_training_location=reuse['test_to_train'][
            'query_windows_with_any_position_seen_in_reference'],
        test_window_reuse_fraction=reuse['test_to_train']['query_window_reuse_fraction'],
        validation_window_reuse_fraction=reuse['val_to_train']['query_window_reuse_fraction'],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output-dir', type=Path, required=True, help='new result directory')
    parser.add_argument('--fractions', type=float, nargs='+', default=[0.01, 1.0],
                        help='shared fractions of the position pool (default: 0.01 1.0)')
    parser.add_argument('--smoke', action='store_true', help='tiny dataset, relaxed provenance')
    parser.add_argument('--keep-datasets', action='store_true', help='keep the generated datasets')
    args = parser.parse_args()
    if any(not 0 < fraction <= 1 for fraction in args.fractions):
        parser.error('fractions must lie in (0, 1]')
    with threadpool_limits(limits=1):
        try:
            output = check_new_output(args.output_dir, ROOT)
            provenance = collect_provenance(
                ROOT, [Path(__file__), ROOT / 'scripts/run_split_study.py',
                       ROOT / 'validation-requirements.txt', ROOT / 'validation-windows-lock.txt'],
                sys.argv[1:], require_clean=not args.smoke)
        except ValueError as error:
            parser.error(str(error))
        output.mkdir(parents=True, exist_ok=False)
        record_provenance(output, provenance)
        results = []
        for fraction in args.fractions:
            folder = output / f'fraction-{fraction:g}'
            folder.mkdir()
            results.append(run_fraction(fraction, folder, args.smoke, args.keep_datasets))
            write_json(output / 'summary.json', results)
            print(f"shared fraction {fraction:g}: {results[-1]['test_windows_at_a_training_location']} of "
                  f"{results[-1]['test_windows']} test windows "
                  f"({100 * results[-1]['test_window_reuse_fraction']:.1f}%) at a training location", flush=True)
        record_output_hashes(output)


if __name__ == '__main__':
    main()
