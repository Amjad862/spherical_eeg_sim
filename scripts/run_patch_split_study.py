"""Compare the default and the contained region split for distributed patches.

For each dataset seed the script generates one patch dataset under the default
exact-position policy and one under the region policy with support
containment, at equal support-mesh density. It records the patch-overlap and
containment audits of both and scores a centre regressor, nearest-training-
patch lookups and the training-mean centre on each test split.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import sys
import warnings
from pathlib import Path

import h5py
import joblib
import numpy as np
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spherical_eeg_sim import Config, plan_position_split, run_pipeline
from spherical_eeg_sim.sources.partition import patch_capacity
from run_split_study import (
    DATA_SEEDS,
    MODEL_SEEDS,
    cluster_interval,
    describe,
    load_windows,
    predict,
    record_output_hashes,
    record_provenance,
    train_mlp,
    unit_rows,
    write_json,
)
from _run_provenance import check_new_output, collect_provenance

# Positions in the support mesh of each split. The default policy divides its
# pool into three such meshes; the region policy keeps the pool density inside
# every region, so its pool has the size of one mesh.
MESH_POSITIONS = 11_667
REGION_COUNT = 8
BUFFER_MM = 15.0
MIN_CENTRES_FOR_LARGEST_PATCH = 50
MIN_TRAINING_RETENTION = 0.5
DIAMETER_BANDS_MM = ((10.0, 25.0), (25.0, 40.0), (40.0, 60.001))
SNR_BANDS_DB = ((10.0, 15.0), (15.0, 20.0), (20.0, 25.001))
KEPT_DATASET_FILES = ('checksums.sha256', 'validation_report.json', 'metadata.json')


def config_for(seed, policy, output, smoke=False):
    cfg = Config()
    cfg.source_mode = 'patch'
    cfg.master_seed = cfg.position_seed = cfg.sample_seed = seed
    cfg.dataset.n_train, cfg.dataset.n_val, cfg.dataset.n_test = (
        (240, 60, 60) if smoke else (20_000, 2_000, 3_000))
    mesh = 3_000 if smoke else MESH_POSITIONS
    cfg.sources.n_cortical_positions = mesh if policy == 'region' else 3 * mesh
    cfg.head.n_legendre = 60
    cfg.noise.noise_model = 'snr_target'
    cfg.noise.snr_db_range = [10., 25.]
    cfg.output.reference_mode = 'average'
    cfg.output.eeg_normalization = 'per_sample_rms'
    cfg.output.diagnostic_plots = False
    cfg.split.policy = policy
    if policy == 'region':
        cfg.split.region_count = 6 if smoke else REGION_COUNT
        cfg.split.buffer_mm = 10.0 if smoke else BUFFER_MM
        cfg.split.region_rotation_seed = 24_000 + seed - 862
    cfg.output_dir = str(output)
    return cfg


def adequacy(cfg):
    """Geometry-only checks of the region assignment, made before any data exist."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        plan = plan_position_split(cfg)
    labels, limit = plan.split_labels, plan.patch_containment_radius_mm
    low, high = cfg.patch.diameter_range_mm
    pre = np.bincount(plan.region.pre_buffer_split_labels, minlength=3)
    post = np.bincount(labels[labels >= 0], minlength=3)
    largest = patch_capacity(limit, labels, high)
    report = dict(
        positions=int(len(labels)), pre_buffer_positions=pre.tolist(), eligible_positions=post.tolist(),
        excluded_positions=int(np.sum(labels < 0)),
        training_retention=float(post[0] / pre[0]),
        centres_admitting_smallest_diameter=patch_capacity(limit, labels, low),
        centres_admitting_largest_diameter=largest,
        largest_admissible_diameter_mm={
            name: 2.0 * float(np.nanmax(limit[labels == i])) for i, name in enumerate(('train', 'val', 'test'))},
    )
    report['checks'] = dict(
        largest_patch_fits_at_enough_centres=all(
            count >= MIN_CENTRES_FOR_LARGEST_PATCH for count in largest.values()),
        training_region_retained=report['training_retention'] >= MIN_TRAINING_RETENTION,
    )
    return report


def patch_diameters(path):
    with h5py.File(path, 'r') as file:
        return file['ground_truth/patch_diameter_mm'][:].ravel()


def neighbours(train, test_topography):
    """Centre of the best-correlated training patch, and the mean of the ten best."""
    reference = unit_rows(train.topography)
    one, ten = [], []
    for start in range(0, len(test_topography), 128):
        correlation = abs(unit_rows(test_topography[start:start + 128]) @ reference.T)
        one.append(train.position[correlation.argmax(axis=1)])
        k = min(10, len(train.position))
        ten.append(train.position[np.argpartition(correlation, -k, axis=1)[:, -k:]].mean(axis=1))
    return np.concatenate(one), np.concatenate(ten)


def strata(diameter, snr):
    for lo, hi in DIAMETER_BANDS_MM:
        yield f'diameter_{lo:g}_{hi:g}', (diameter >= lo) & (diameter < hi)
    for lo, hi in SNR_BANDS_DB:
        yield f'snr_{lo:g}_{hi:g}', (snr >= lo) & (snr < hi)


def audit_summary(report):
    """The parts of a dataset validation report that describe split separation."""
    pairs = {}
    for name, pair in report['patch_overlap']['pairs'].items():
        pairs[name] = dict(
            query_patches=pair['query_patches'], scored_patches=pair['iou_query_patches'],
            best_iou=pair['best_cap_iou'], nearest_centre_euclidean_mm=pair['nearest_centre_euclidean_mm'],
            nearest_centre_geodesic_mm=pair['nearest_centre_geodesic_mm'],
            minimum_footprint_gap_mm=pair['minimum_footprint_gap_mm'],
        )
    summary = dict(pairs=pairs)
    region = report['source_split'].get('region')
    if region is not None:
        summary['minimum_support_clearance_mm'] = region['minimum_serialized_used_clearance_mm']
        summary['patch_containment'] = region['patch_containment']['splits']
    return summary


def prune_dataset(dataset):
    for path in dataset.iterdir():
        if path.name not in KEPT_DATASET_FILES:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()


def evaluate(seed, policy, folder, args):
    smoke = args.smoke
    draws = 40 if smoke else 2000
    dataset = folder / 'dataset'
    cfg = config_for(seed, policy, dataset, smoke)
    if policy == 'region':
        check = adequacy(cfg)
        write_json(folder / 'split_adequacy.json', check)
        if not smoke and not all(check['checks'].values()):
            raise RuntimeError(f'predeclared patch region assignment failed adequacy for seed {seed}')
    cfg.to_json(folder / 'config.json')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        run_pipeline(cfg, log_level=logging.WARNING)
    report = json.loads((dataset / 'validation_report.json').read_text(encoding='utf-8'))
    train, validation, test = (load_windows(dataset / f'{name}.h5') for name in ('train', 'val', 'test'))
    diameter = patch_diameters(dataset / 'test.h5')
    predictions = {}
    predictions['nearest_training_patch'], predictions['ten_nearest_training_patches'] = neighbours(
        train, test.topography)
    predictions['training_mean'] = np.broadcast_to(train.position.mean(axis=0), test.position.shape)
    fits = {}
    for model_seed in MODEL_SEEDS:
        fit = train_mlp(train, validation, model_seed, 3 if smoke else 300)
        fits[f'mlp_{model_seed}'] = fit
        joblib.dump(fit, folder / f'mlp_{model_seed}.joblib')
        predictions[f'mlp_{model_seed}'] = predict(fit, test.topography)
    error = {name: np.linalg.norm(value - test.position, axis=1) for name, value in predictions.items()}
    error['mlp_pooled'] = np.stack([error[name] for name in fits])
    nearest_centre = cKDTree(np.unique(train.position, axis=0)).query(test.position)[0]
    with (folder / 'predictions.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        names = list(predictions)
        writer.writerow(['window', 'centre_position_id', 'diameter_mm', 'snr_db', 'nearest_training_centre_mm',
                         'x_mm', 'y_mm', 'z_mm'] + [f'{name}_{axis}_mm' for name in names for axis in 'xyz'])
        for i in range(len(test.position)):
            writer.writerow([i, int(test.pool_id[i]), diameter[i], test.snr[i], nearest_centre[i],
                             *test.position[i], *(value for name in names for value in predictions[name][i])])
    results = {}
    for name, value in error.items():
        results[name] = describe(value, test.pool_id, 720, draws)
        results[name]['strata'] = {
            label: describe(value[..., mask], test.pool_id[mask], 721, draws)
            for label, mask in strata(diameter, test.snr) if np.any(mask)}
    results['paired_mlp_minus_lookup'] = {
        name: dict(median_difference_mm=float(np.median(error['mlp_pooled']) - np.median(error[name])),
                   ci95_mm=cluster_interval(error['mlp_pooled'], test.pool_id, np.random.default_rng(722),
                                            error[name], draws))
        for name in ('nearest_training_patch', 'ten_nearest_training_patches')}
    summary = dict(
        methods=results,
        fits={key: dict(best_epoch=fit['best_epoch'], epochs_run=fit['epochs_run']) for key, fit in fits.items()},
        audit=audit_summary(report),
        support_positions={name: int(report['source_split']['eligible_position_nearest_neighbour_mm']
                                     [pair]['all']['n'])
                           for name, pair in (('test', 'test_to_train'), ('val', 'val_to_train'))},
        distinct_training_centres=int(len(np.unique(train.pool_id))),
        nearest_training_centre_mm=dict(median=float(np.median(nearest_centre)),
                                        minimum=float(nearest_centre.min())),
        mlp_error_below_nearest_training_centre=float(np.mean(error['mlp_pooled'] < nearest_centre)),
    )
    write_json(folder / 'summary.json', summary)
    if not args.keep_datasets:
        prune_dataset(dataset)
    print(f'{seed} {policy}: MLP {results["mlp_pooled"]["median_mm"]:.3f} mm, nearest training patch '
          f'{results["nearest_training_patch"]["median_mm"]:.3f} mm, median best overlap '
          f'{summary["audit"]["pairs"]["test_to_train"]["best_iou"]["median"]:.3f}', flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output-dir', type=Path, required=True, help='new result directory')
    parser.add_argument('--smoke', action='store_true', help='one seed, tiny datasets, relaxed provenance')
    parser.add_argument('--keep-datasets', action='store_true', help='keep the generated datasets')
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
        record_provenance(output, provenance)
        seeds = DATA_SEEDS[:1] if args.smoke else DATA_SEEDS
        if not args.smoke:
            preflight = {seed: adequacy(config_for(seed, 'region', output / str(seed) / 'region/dataset'))
                         for seed in seeds}
            write_json(output / 'region_adequacy_preflight.json', preflight)
            if not all(all(item['checks'].values()) for item in preflight.values()):
                raise RuntimeError('a predeclared patch region assignment failed adequacy; see the preflight report')
        reports = {}
        for seed in seeds:
            reports[seed] = {}
            for policy in ('position', 'region'):
                folder = output / str(seed) / policy
                folder.mkdir(parents=True)
                reports[seed][policy] = evaluate(seed, policy, folder, args)
                write_json(output / 'summary.json', reports)
        headline = {}
        for policy in ('position', 'region'):
            headline[policy] = {}
            for method in reports[seeds[0]][policy]['methods']:
                if 'median_mm' not in reports[seeds[0]][policy]['methods'][method]:
                    continue
                medians = {seed: report[policy]['methods'][method]['median_mm'] for seed, report in reports.items()}
                values = list(medians.values())
                headline[policy][method] = dict(dataset_seed_medians_mm=medians,
                                                mean_median_mm=float(np.mean(values)),
                                                range_mm=[min(values), max(values)])
            overlap = [report[policy]['audit']['pairs']['test_to_train']['best_iou']['median']
                       for report in reports.values()]
            headline[policy]['median_best_test_to_train_overlap'] = dict(
                dataset_seed_values=overlap, mean=float(np.mean(overlap)), range=[min(overlap), max(overlap)])
        write_json(output / 'headline.json', headline)
        record_output_hashes(output)


if __name__ == '__main__':
    main()
