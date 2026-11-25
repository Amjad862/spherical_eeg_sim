"""Run the supervised position-split comparison in a new output directory."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import joblib
import numpy as np
from scipy.spatial import cKDTree
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spherical_eeg_sim import Config, run_pipeline
from train_baseline import peak_gfp_topography
from validate_patch_forward_mne import build_sphere_and_info
from _run_provenance import check_new_output, collect_provenance

DATA_SEEDS = (862, 863, 864)
MODEL_SEEDS = (862, 863)
DEPTH_BANDS = ((1, 30, 45), (1, 45, 55), (1, 55, 65.001),
               (0, 70.5, 75), (0, 75, 79.501))


@dataclass
class Windows:
    topography: np.ndarray
    position: np.ndarray
    pool_id: np.ndarray
    population: np.ndarray
    snr: np.ndarray

    def subset(self, mask):
        return Windows(self.topography[mask], self.position[mask], self.pool_id[mask],
                       self.population[mask], self.snr[mask])


def load_windows(path):
    with h5py.File(path, 'r') as file:
        gt = file['ground_truth']
        eeg = file['eeg']
        topography = np.concatenate([peak_gfp_topography(eeg[i:i + 256])
                                     for i in range(0, len(eeg), 256)])
        return Windows(topography, gt['positions_xyz'][:] * 1000, gt['position_pool_idx'][:],
                       gt['population'][:], gt['snr_db_referenced'][:])


def config_for(seed, policy, output, smoke=False):
    cfg = Config()
    cfg.master_seed = cfg.position_seed = cfg.sample_seed = seed
    cfg.dataset.n_train, cfg.dataset.n_val, cfg.dataset.n_test = (
        (160, 16, 24) if smoke else (20_000, 2_000, 3_000))
    cfg.sources.n_cortical_positions = 8_000
    cfg.sources.n_subcortical_positions = 1_500
    cfg.head.n_legendre = 60
    cfg.noise.noise_model = 'snr_target'
    cfg.noise.snr_db_range = [10., 25.]
    cfg.output.reference_mode = 'average'
    cfg.output.eeg_normalization = 'per_sample_rms'
    cfg.output.save_components = policy == 'position'
    cfg.output.save_leadfield = True
    cfg.output.diagnostic_plots = False
    cfg.split.policy = policy
    if policy == 'region':
        cfg.split.region_count = 24
        cfg.split.buffer_mm = 15.0
        cfg.split.region_rotation_seed = 24_000 + seed - 862
    cfg.output_dir = str(output)
    return cfg


def adequacy(plan, cfg):
    before = plan.region.pre_buffer_split_labels
    after = plan.split_labels
    radius = np.linalg.norm(plan.positions_xyz, axis=1) * 1000
    sizes = np.array([cfg.dataset.n_train, cfg.dataset.n_val, cfg.dataset.n_test])
    shares = np.bincount(before, minlength=3) / len(before)
    c1 = bool(np.all(abs(shares[1:] - (sizes / sizes.sum())[1:]) <= .03))
    report = {'pre_buffer_shares': shares.tolist(), 'population': [], 'radius_bands': []}

    def counts(mask):
        a = np.bincount(before[mask], minlength=3)
        b = np.bincount(after[mask & (after >= 0)], minlength=3)
        return dict(pre=a.tolist(), retained=b.tolist(),
                    fraction=np.divide(b, a, out=np.zeros(3), where=a > 0).tolist())

    for population in (0, 1):
        report['population'].append(dict(population=population,
                                         **counts(plan.population == population)))
    for pop, lo, hi in DEPTH_BANDS:
        mask = (plan.population == pop) & (radius >= lo) & (radius < hi)
        report['radius_bands'].append(dict(population=pop, radius_mm=[lo, hi], **counts(mask)))
    c2 = all(item['fraction'][0] >= .60 for item in report['population'])
    c3 = all(item['fraction'][0] >= .50 and item['retained'][2] >= 30
             for item in report['radius_bands'])
    c4 = (all(item['fraction'][1] >= .60 for item in report['population']) and
          all(item['fraction'][1] >= .50 and item['retained'][1] > 0
              for item in report['radius_bands']))
    report['checks'] = dict(C1=c1, C2=c2, C3=c3, C4=c4)
    return report


def train_mlp(train, validation, seed, epochs):
    x_scaler = StandardScaler().fit(train.topography)
    y_scaler = StandardScaler().fit(train.position)
    x = x_scaler.transform(train.topography)
    y = y_scaler.transform(train.position)
    vx = x_scaler.transform(validation.topography)
    model = MLPRegressor(hidden_layer_sizes=(256, 128), activation='relu',
                         solver='adam', early_stopping=False, max_iter=1, random_state=seed)
    best_loss, best_epoch = np.inf, 0
    for epoch in range(1, epochs + 1):
        model.partial_fit(x, y)
        loss = np.mean((y_scaler.inverse_transform(model.predict(vx)) - validation.position) ** 2)
        if loss < best_loss - 1e-8:
            best_loss, best_epoch = loss, epoch
            best = copy.deepcopy((model.coefs_, model.intercepts_))
        if epoch - best_epoch >= 15:
            break
    if not best_epoch:
        raise RuntimeError('MLP validation error is non-finite')
    model.coefs_, model.intercepts_ = best
    return dict(model=model, x_scaler=x_scaler, y_scaler=y_scaler,
                best_epoch=best_epoch, epochs_run=epoch, training_seed=seed)


def predict(fit, topography):
    return fit['y_scaler'].inverse_transform(fit['model'].predict(fit['x_scaler'].transform(topography)))


def unit_rows(array):
    norm = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norm, np.finfo(float).tiny)


def dictionary_and_neighbors(train, test, leadfield_path):
    positions = np.unique(train.pool_id)
    with h5py.File(leadfield_path, 'r') as field:
        xyz = field['positions_xyz'][positions] * 1000
        leadfield = field['leadfield'][positions]
    leadfield -= leadfield.mean(axis=1, keepdims=True)
    u, singular, _ = np.linalg.svd(leadfield, full_matrices=False)
    u *= (singular > 1e-12 * singular[:, :1])[:, None, :]
    basis = u.transpose(1, 0, 2).reshape(train.topography.shape[1], -1)
    normal_train = unit_rows(train.topography)
    results = {name: [] for name in ('dictionary', 'knn1', 'knn10')}
    for i in range(0, len(test), 128):
        topography = unit_rows(test[i:i + 128])
        scores = ((topography @ basis).reshape(len(topography), len(xyz), 3) ** 2).sum(axis=2)
        results['dictionary'].append(xyz[np.argmax(scores, axis=1)])
        correlation = abs(topography @ normal_train.T)
        results['knn1'].append(train.position[correlation.argmax(axis=1)])
        k = min(10, len(train.position))
        neighbors = np.argpartition(correlation, -k, axis=1)[:, -k:]
        results['knn10'].append(train.position[neighbors].mean(axis=1))
    return {name: np.concatenate(batches) for name, batches in results.items()}


def sensor_info(path):
    with h5py.File(path, 'r') as file:
        group = file['electrodes']
        names = [name.decode() if isinstance(name, bytes) else str(name) for name in group['names'][:]]
        return group['cartesian_xyz'][:], names


def dipole_fit(topography, info, conductor, trans=None, min_dist=.5):
    import mne
    evoked = mne.EvokedArray(topography.T, info, verbose=False)
    evoked.set_eeg_reference('average', projection=True, verbose=False)
    covariance = mne.make_ad_hoc_cov(info, verbose=False)
    dipoles, _ = mne.fit_dipole(evoked, covariance, conductor, trans=trans,
                                min_dist=min_dist, n_jobs=1, verbose=False)
    return dipoles.pos * 1000


def errors(prediction, test):
    return np.linalg.norm(prediction - test.position, axis=-1)


def cluster_interval(error, ids, rng, other=None, draws=2000):
    blocks = position_clusters(ids)
    if len(blocks) < 30:
        return None
    stats = np.empty(draws)
    for i in range(draws):
        rows = np.concatenate([blocks[j] for j in rng.integers(len(blocks), size=len(blocks))])
        stats[i] = np.median(error[..., rows])
        if other is not None:
            stats[i] -= np.median(other[..., rows])
    return np.quantile(stats, [.025, .975]).tolist()


def position_clusters(ids):
    order = np.argsort(ids, kind='stable')
    return np.split(order, np.flatnonzero(np.diff(ids[order])) + 1)


def pooled_shared_intervals(output, seeds, draws):
    groups = {}
    for seed in seeds:
        path = output / str(seed) / 'same_test_windows' / 'paired_mlp.npz'
        if not path.is_file():
            continue
        with np.load(path) as source:
            test = Windows(np.empty((0, 0)), source['position'], source['pool_id'],
                           source['population'], source['snr'])
            strata_masks = [('all', np.ones(len(test.position), dtype=bool)),
                            *strata(test, source['exact_distance'], source['coverage'])]
            for label, mask in strata_masks:
                if np.any(mask):
                    groups.setdefault(label, []).append((source['region'][:, mask],
                                                          source['position_only'][:, mask],
                                                          position_clusters(test.pool_id[mask])))
    rng = np.random.default_rng(714)
    results = {}
    for label, values in groups.items():
        region = np.concatenate([item[0] for item in values], axis=1)
        position = np.concatenate([item[1] for item in values], axis=1)
        count = sum(len(item[2]) for item in values)
        interval = None
        if count >= 30:
            stats = []
            for _ in range(draws):
                samples = [np.concatenate([blocks[i] for i in rng.integers(len(blocks), size=len(blocks))])
                           for _, _, blocks in values]
                r = np.concatenate([item[0][:, rows] for item, rows in zip(values, samples)], axis=1)
                p = np.concatenate([item[1][:, rows] for item, rows in zip(values, samples)], axis=1)
                stats.append(np.median(r) - np.median(p))
            interval = np.quantile(stats, [.025, .975]).tolist()
        results[label] = dict(windows=region.shape[1], seed_position_clusters=count,
                              region_minus_position_mm=float(np.median(region) - np.median(position)),
                              conditional_ci95_mm=interval)
    return dict(weighting='equal weight per window and fit, with fixed dataset seed groups',
                distance_strata='nearest exact-position training window', strata=results)


def describe(error, ids, seed, draws):
    return dict(windows=len(ids), positions=len(np.unique(ids)),
                median_mm=float(np.median(error)), mean_mm=float(np.mean(error)),
                p90_mm=float(np.quantile(error, .90)),
                median_ci95_mm=cluster_interval(error, ids, np.random.default_rng(seed), draws=draws))


def strata(test, distance, coverage):
    radius = np.linalg.norm(test.position, axis=1)
    for pop, lo, hi in DEPTH_BANDS:
        yield f'radius_{pop}_{lo:g}_{hi:g}', (test.population == pop) & (radius >= lo) & (radius < hi)
    for pop in (0, 1):
        yield f'population_{pop}', test.population == pop
    for name, values, bounds in (('snr', test.snr, [10, 15, 20, 25.001]),
                                 ('distance', distance, [0, 5, 10, 20, 30, np.inf])):
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            yield f'{name}_{lo:g}_{hi:g}', (values >= lo) & (values < hi)
    yield 'below_montage', coverage
    yield 'within_montage', ~coverage


def save_results(output, test, predictions, train, electrodes, draws, fit_names=()):
    distance = cKDTree(np.unique(train.position, axis=0)).query(test.position)[0]
    coverage = (test.position[:, 2] / np.linalg.norm(test.position, axis=1) <
                np.min(electrodes[:, 2] / np.linalg.norm(electrodes, axis=1)))
    names = list(predictions)
    with (output / 'predictions.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['window', 'position_id', 'population', 'snr_db', 'nearest_train_mm',
                         'below_montage', 'x_mm', 'y_mm', 'z_mm'] +
                        [f'{name}_{axis}_mm' for name in names for axis in 'xyz'])
        for i in range(len(test.position)):
            writer.writerow([i, int(test.pool_id[i]), int(test.population[i]), test.snr[i],
                             distance[i], int(coverage[i]), *test.position[i],
                             *(v for name in names for v in predictions[name][i])])
    results = {}
    metrics = {name: errors(prediction, test) for name, prediction in predictions.items()}
    if fit_names:
        metrics['mlp_pooled'] = np.stack([metrics[name] for name in fit_names])
    for name, error in metrics.items():
        results[name] = describe(error, test.pool_id, 710, draws)
        results[name]['strata'] = {
            label: describe(error[..., mask], test.pool_id[mask], 711, draws)
            for label, mask in strata(test, distance, coverage) if np.any(mask)
        }
    if fit_names:
        mlp_error = metrics['mlp_pooled']
        results['paired_mlp_minus_baseline'] = {
            name: dict(median_difference_mm=float(np.median(mlp_error) - np.median(metrics[name])),
                       ci95_mm=cluster_interval(mlp_error, test.pool_id, np.random.default_rng(713),
                                                metrics[name], draws))
            for name in ('dictionary', 'knn1', 'knn10', 'dipole') if name in metrics
        }
    return results


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def record_provenance(output, record):
    write_json(output / 'provenance.json', record)
    (output / 'environment.txt').write_text(record['resolved_packages'], encoding='utf-8')


def study_provenance(smoke):
    files = [Path(__file__),
             ROOT / 'scripts/train_baseline.py', ROOT / 'scripts/validate_patch_forward_mne.py',
             ROOT / 'scripts/_validation_outputs.py', ROOT / 'docs/revision_experiments.md',
             ROOT / 'validation-requirements.txt', ROOT / 'validation-windows-lock.txt']
    return collect_provenance(ROOT, files, sys.argv[1:], require_clean=not smoke)


def record_output_hashes(output):
    inventory = {}
    for path in sorted(output.rglob('*')):
        relative = path.relative_to(output)
        if path.is_file() and ('dataset' not in relative.parts or path.name == 'checksums.sha256'):
            inventory[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(output / 'output_sha256.json', inventory)


def evaluate_seed(seed, run, smoke, ecd_count):
    from spherical_eeg_sim import plan_position_split

    runs = {}
    draws = 40 if smoke else 2000
    for policy in ('position', 'region'):
        folder = run / policy
        folder.mkdir()
        dataset = folder / 'dataset'
        cfg = config_for(seed, policy, dataset, smoke)
        if policy == 'region':
            plan = plan_position_split(cfg)
            check = adequacy(plan, cfg)
            write_json(folder / 'split_adequacy.json', check)
            if not smoke and not all(check['checks'].values()):
                raise RuntimeError(f'predeclared region partition failed adequacy for seed {seed}')
        cfg.to_json(folder / 'config.json')
        run_pipeline(cfg, log_level=logging.WARNING)
        if policy == 'region':
            with h5py.File(dataset / 'source_pool.h5', 'r') as file:
                np.testing.assert_array_equal(file['split'][:], plan.split_labels)
                np.testing.assert_array_equal(file['positions_xyz'][:], plan.positions_xyz)
        train, validation, test = (load_windows(dataset / f'{name}.h5') for name in ('train', 'val', 'test'))
        electrodes, names = sensor_info(dataset / 'test.h5')
        ecd_rows = np.sort(np.random.default_rng([seed, 601]).choice(
            len(test.position), size=min(ecd_count, len(test.position)), replace=False))
        fits = {}
        for model_seed in MODEL_SEEDS:
            fit = train_mlp(train, validation, model_seed, 3 if smoke else 300)
            key = f'mlp_{model_seed}'
            fits[key] = fit
            joblib.dump(fit, folder / f'{key}.joblib')
        predictions = dictionary_and_neighbors(train, test.topography, dataset / 'leadfield.h5')
        predictions['training_mean'] = np.broadcast_to(train.position.mean(axis=0), test.position.shape)
        predictions.update({key: predict(fit, test.topography) for key, fit in fits.items()})
        full = save_results(folder, test, predictions, train, electrodes, draws, tuple(fits))
        summary = dict(full=full, fits={key: {field: fit[field] for field in ('best_epoch', 'epochs_run')}
                                       for key, fit in fits.items()})
        summary['observed_split_composition'] = {}
        for name, windows in (('train', train), ('validation', validation), ('test', test)):
            radii = np.linalg.norm(windows.position, axis=1)
            summary['observed_split_composition'][name] = [
                dict(population=pop, radius_mm=[lo, hi], windows=int(mask.sum()),
                     distinct_positions=len(np.unique(windows.pool_id[mask])))
                for pop, lo, hi in DEPTH_BANDS
                for mask in [(windows.population == pop) & (radii >= lo) & (radii < hi)]
            ]
        if ecd_count:
            import mne
            info, sphere = build_sphere_and_info(electrodes, names, cfg.head.radii_m,
                                                 cfg.head.conductivities_S_per_m,
                                                 cfg.signal.sampling_freq_hz, mne)
            subset = {key: value[ecd_rows] for key, value in predictions.items()}
            subset['dipole'] = dipole_fit(test.topography[ecd_rows], info, sphere)
            ecd_folder = folder / 'ecd_subset'
            ecd_folder.mkdir()
            write_json(ecd_folder / 'test_rows.json', ecd_rows.tolist())
            summary['ecd_subset'] = save_results(ecd_folder, test.subset(ecd_rows), subset,
                                                 train, electrodes, draws, tuple(fits))
        write_json(folder / 'summary.json', summary)
        runs[policy] = (dataset, train, test, fits, predictions, electrodes)
        print(f'{seed} {policy} pooled MLP median: {full["mlp_pooled"]["median_mm"]:.4f} mm', flush=True)

    exact_path, exact_train, test, exact_fits, predictions, electrodes = runs['position']
    region_path, region_train, _, region_fits, _, _ = runs['region']
    with h5py.File(exact_path / 'source_pool.h5', 'r') as file, \
            h5py.File(region_path / 'source_pool.h5', 'r') as region:
        np.testing.assert_array_equal(file['positions_xyz'][:], region['positions_xyz'][:])
        rows = np.flatnonzero(region['split'][:][test.pool_id] == 2)
    if len(rows):
        folder = run / 'same_test_windows'
        folder.mkdir()
        write_json(folder / 'exact_position_test_rows.json', rows.tolist())
        common = test.subset(rows)
        a = {key: value[rows] for key, value in predictions.items()}
        b = dictionary_and_neighbors(region_train, common.topography, region_path / 'leadfield.h5')
        b['training_mean'] = np.broadcast_to(region_train.position.mean(axis=0), common.position.shape)
        b.update({key: predict(fit, common.topography) for key, fit in region_fits.items()})
        if ecd_count:
            import mne
            names = sensor_info(exact_path / 'test.h5')[1]
            info, sphere = build_sphere_and_info(electrodes, names, cfg.head.radii_m,
                                                 cfg.head.conductivities_S_per_m,
                                                 cfg.signal.sampling_freq_hz, mne)
            a['dipole'] = b['dipole'] = dipole_fit(common.topography, info, sphere)
        for name, pred, tr in (('position', a, exact_train), ('region', b, region_train)):
            target = folder / name
            target.mkdir()
            write_json(target / 'summary.json', save_results(target, common, pred, tr, electrodes,
                                                               draws, tuple(exact_fits)))
        ae = np.stack([errors(a[key], common) for key in exact_fits])
        be = np.stack([errors(b[key], common) for key in exact_fits])
        distance = cKDTree(np.unique(exact_train.position, axis=0)).query(common.position)[0]
        below = (common.position[:, 2] / np.linalg.norm(common.position, axis=1) <
                 np.min(electrodes[:, 2] / np.linalg.norm(electrodes, axis=1)))
        np.savez(folder / 'paired_mlp.npz', region=be, position_only=ae, position=common.position,
                 pool_id=common.pool_id, population=common.population, snr=common.snr,
                 exact_distance=distance, coverage=below)
        write_json(folder / 'paired_mlp.json', dict(region_minus_position_mm=float(np.median(be) - np.median(ae)),
                   ci95_mm=cluster_interval(be, common.pool_id, np.random.default_rng(712), ae, draws),
                   windows=len(rows), distinct_positions=len(np.unique(common.pool_id))))
    return {policy: json.loads((run / policy / 'summary.json').read_text(encoding='utf-8'))
            for policy in runs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--ecd-count', type=int, default=500)
    args = parser.parse_args()
    if args.ecd_count < 0:
        parser.error('--ecd-count must be nonnegative')
    with threadpool_limits(limits=1):
        try:
            output = check_new_output(args.output_dir, ROOT)
            provenance = study_provenance(args.smoke)
        except ValueError as error:
            parser.error(str(error))
        output.mkdir(parents=True, exist_ok=False)
        record_provenance(output, provenance)
        seeds = DATA_SEEDS[:1] if args.smoke else DATA_SEEDS
        if not args.smoke:
            from spherical_eeg_sim import plan_position_split

            preliminary = {}
            for seed in seeds:
                cfg = config_for(seed, 'region', output / str(seed) / 'region/dataset')
                preliminary[seed] = adequacy(plan_position_split(cfg), cfg)
            write_json(output / 'region_adequacy_preflight.json', preliminary)
            if not all(all(result['checks'].values()) for result in preliminary.values()):
                raise RuntimeError('a predeclared region partition failed adequacy; see preflight report')
        reports = {}
        for seed in seeds:
            run = output / str(seed)
            run.mkdir()
            reports[seed] = evaluate_seed(seed, run, args.smoke,
                                           min(6, args.ecd_count) if args.smoke else args.ecd_count)
            write_json(output / 'summary.json', reports)
        headline = {}
        for policy in ('position', 'region'):
            headline[policy] = {}
            for panel in ('full', 'ecd_subset'):
                if panel not in reports[DATA_SEEDS[0]][policy]:
                    continue
                panels = {seed: report[policy][panel] for seed, report in reports.items()}
                headline[policy][panel] = {}
                for method, statistic in panels[DATA_SEEDS[0]].items():
                    if 'median_mm' not in statistic:
                        continue
                    medians = {seed: report[method]['median_mm'] for seed, report in panels.items()}
                    values = list(medians.values())
                    headline[policy][panel][method] = dict(
                        dataset_seed_medians_mm=medians, mean_median_mm=float(np.mean(values)),
                        range_mm=[min(values), max(values)])
        write_json(output / 'headline.json', headline)
        write_json(output / 'paired_split_summary.json',
                   pooled_shared_intervals(output, reports, 40 if args.smoke else 2000))
        record_output_hashes(output)


if __name__ == '__main__':
    main()
