"""Control for the number of training locations in the split comparison.

Region holdout leaves fewer distinct training locations than exact-position
holdout. This control refits the exact-position network after reducing its
training set to the number of distinct locations the regional training set
used, cell by cell (source population and radius band), and scores it on the
same test windows. It shows how much of the regional error increase could be
due to the smaller number of training locations alone.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_split_study import (
    DATA_SEEDS,
    DEPTH_BANDS,
    MODEL_SEEDS,
    cluster_interval,
    describe,
    load_windows,
    predict,
    record_output_hashes,
    record_provenance,
    train_mlp,
    write_json,
)
from _run_provenance import check_new_output, collect_provenance, hash_file


def matched_training_rows(train, regional_cells, rng):
    """Rows of the exact-position training set at a count-matched set of locations."""
    radius = np.linalg.norm(train.position, axis=1)
    keep = np.zeros(len(radius), dtype=bool)
    cells = []
    for (population, low, high), regional in zip(DEPTH_BANDS, regional_cells):
        if regional['population'] != population or regional['radius_mm'] != [low, high]:
            raise ValueError('regional composition does not follow the study radius bands')
        mask = (train.population == population) & (radius >= low) & (radius < high)
        available = np.unique(train.pool_id[mask])
        count = min(len(available), regional['distinct_positions'])
        chosen = rng.choice(available, size=count, replace=False)
        selected = mask & np.isin(train.pool_id, chosen)
        keep |= selected
        cells.append(dict(population=population, radius_mm=[low, high],
                          exact_positions=int(len(available)), regional_positions=regional['distinct_positions'],
                          matched_positions=int(count), exact_windows=int(mask.sum()),
                          regional_windows=regional['windows'], matched_windows=int(selected.sum())))
    return np.flatnonzero(keep), cells


def write_errors(path, position_id, exact_mm, count_matched_mm):
    """Per-window errors of both networks, with every value as a plain decimal number."""
    with Path(path).open('w', encoding='utf-8', newline='') as file:
        file.write('window,position_id,exact_error_mm,count_matched_error_mm\n')
        for window, (position, exact, matched) in enumerate(zip(position_id, exact_mm, count_matched_mm)):
            file.write(f'{window},{int(position)},{float(exact)!r},{float(matched)!r}\n')


def evaluate(seed, split_run, output, smoke):
    draws = 40 if smoke else 2000
    exact = split_run / str(seed) / 'position'
    region_summary = json.loads((split_run / str(seed) / 'region/summary.json').read_text(encoding='utf-8'))
    train, validation, test = (load_windows(exact / f'dataset/{name}.h5') for name in ('train', 'val', 'test'))
    rows, cells = matched_training_rows(
        train, region_summary['observed_split_composition']['train'], np.random.default_rng([seed, 101]))
    model_seed = MODEL_SEEDS[0]
    matched = train_mlp(train.subset(rows), validation, model_seed, 3 if smoke else 300)
    joblib.dump(matched, output / f'count_matched_mlp_{model_seed}.joblib')
    original = joblib.load(exact / f'mlp_{model_seed}.joblib')
    error = {name: np.linalg.norm(predict(fit, test.topography) - test.position, axis=1)
             for name, fit in (('exact', original), ('count_matched', matched))}
    summary = dict(
        model_seed=model_seed, cells=cells,
        training_positions=dict(exact=int(len(np.unique(train.pool_id))),
                                count_matched=int(len(np.unique(train.pool_id[rows]))),
                                regional=sum(cell['regional_positions'] for cell in cells)),
        training_windows=dict(exact=int(len(train.pool_id)), count_matched=int(len(rows)),
                              regional=sum(cell['regional_windows'] for cell in cells)),
        fit=dict(best_epoch=matched['best_epoch'], epochs_run=matched['epochs_run']),
        all_test_windows={name: describe(value, test.pool_id, seed, draws) for name, value in error.items()},
    )
    summary['all_test_windows']['count_matched_minus_exact'] = dict(
        median_difference_mm=float(np.median(error['count_matched']) - np.median(error['exact'])),
        ci95_mm=cluster_interval(error['count_matched'], test.pool_id, np.random.default_rng([seed, 102]),
                                 error['exact'], draws))
    shared_rows = split_run / str(seed) / 'same_test_windows/exact_position_test_rows.json'
    if shared_rows.is_file():
        index = np.array(json.loads(shared_rows.read_text(encoding='utf-8')))
        with np.load(split_run / str(seed) / 'same_test_windows/paired_mlp.npz') as paired:
            regional = paired['region'][0]
            if not np.allclose(paired['position_only'][0], error['exact'][index], rtol=0, atol=1e-9):
                raise ValueError('the recorded same-window errors do not match the exact-position model')
        groups = test.pool_id[index]
        subset = dict(exact=error['exact'][index], count_matched=error['count_matched'][index], regional=regional)
        block = {name: describe(value, groups, seed, draws) for name, value in subset.items()}
        for name, reference in (('count_matched_minus_exact', 'exact'), ('regional_minus_count_matched', None)):
            a, b = ((subset['count_matched'], subset['exact']) if reference == 'exact'
                    else (subset['regional'], subset['count_matched']))
            block[name] = dict(median_difference_mm=float(np.median(a) - np.median(b)),
                               ci95_mm=cluster_interval(a, groups, np.random.default_rng([seed, 103]), b, draws))
        summary['same_test_windows'] = block
    write_json(output / 'summary.json', summary)
    write_errors(output / 'errors.csv', test.pool_id, error['exact'], error['count_matched'])
    print(f"{seed}: exact {np.median(error['exact']):.3f} mm, count-matched "
          f"{np.median(error['count_matched']):.3f} mm "
          f"({summary['training_positions']['count_matched']} of "
          f"{summary['training_positions']['exact']} training locations)", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--split-run', required=True, type=Path,
                        help='completed run_split_study.py output directory')
    parser.add_argument('--output-dir', required=True, type=Path, help='new result directory')
    parser.add_argument('--smoke', action='store_true', help='use a smoke split run, seed 862 only')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        try:
            output = check_new_output(args.output_dir, ROOT)
            provenance = collect_provenance(
                ROOT, [Path(__file__), ROOT / 'scripts/run_split_study.py', ROOT / 'scripts/train_baseline.py',
                       ROOT / 'validation-requirements.txt', ROOT / 'validation-windows-lock.txt'],
                sys.argv[1:], require_clean=not args.smoke)
        except ValueError as error:
            parser.error(str(error))
        output.mkdir(parents=True, exist_ok=False)
        seeds = DATA_SEEDS[:1] if args.smoke else DATA_SEEDS
        provenance['inputs'] = {
            seed: {name: hash_file(args.split_run / str(seed) / name)
                   for name in ('position/config.json', 'position/dataset/checksums.sha256',
                                f'position/mlp_{MODEL_SEEDS[0]}.joblib', 'region/summary.json')}
            for seed in seeds}
        record_provenance(output, provenance)
        reports = {}
        for seed in seeds:
            folder = output / str(seed)
            folder.mkdir()
            reports[seed] = evaluate(seed, args.split_run, folder, args.smoke)
        headline = {}
        for panel in ('all_test_windows', 'same_test_windows'):
            if panel not in reports[seeds[0]]:
                continue
            headline[panel] = {}
            for name, entry in reports[seeds[0]][panel].items():
                key = 'median_mm' if 'median_mm' in entry else 'median_difference_mm'
                values = [report[panel][name][key] for report in reports.values()]
                headline[panel][name] = dict(statistic=key, dataset_seed_values=values,
                                             mean=float(np.mean(values)), range=[min(values), max(values)])
        write_json(output / 'headline.json', headline)
        record_output_hashes(output)


if __name__ == '__main__':
    main()
