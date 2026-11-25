"""Adapt the sphere to the template head and score every arm on the paired BEM windows.

Companion to ``run_head_transfer.py``. All arms are scored on the same eligible
fsaverage boundary-element test windows as the recorded transfer experiment:

* sphere variants: networks trained on spheres whose radii or compartments are
  matched to the template, with source pools and per-window draws unchanged;
* dipole fits that assume each sphere while the data come from the BEM;
* a reference trained on BEM windows, and fine-tuning of the sphere-trained
  network on small numbers of BEM windows.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import shutil
import sys
from pathlib import Path

import joblib
import mne
import numpy as np
from mne._fiff.constants import FIFF
from mne.transforms import invert_transform
from threadpoolctl import threadpool_limits

from run_head_transfer import (
    MARGIN_MM,
    Registration,
    bem_features,
    forward_columns,
    gate_dataset,
    pick_bem_surfaces,
    populations_and_bands,
    registration_gate,
    require,
    source_eligibility,
    write_csv,
    write_json,
)
from run_split_study import (
    DATA_SEEDS,
    MODEL_SEEDS,
    Windows,
    cluster_interval,
    config_for,
    describe,
    dipole_fit,
    load_windows,
    predict,
    record_output_hashes,
    sensor_info,
    train_mlp,
)
from spherical_eeg_sim import Config, run_pipeline
from validate_patch_forward_mne import build_sphere_and_info
from _run_provenance import check_new_output, collect_provenance, hash_file as digest


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RADII_M = (0.080, 0.081, 0.086, 0.092)
FOUR_SHELL_S_PER_M = (0.330, 1.790, 0.0042, 0.330)
THREE_SHELL_S_PER_M = (0.330, 0.330, 0.0042, 0.330)
FINETUNE_SIZES = (100, 300, 1000, 3000, None)
FORWARD_CHUNK = 1000
KEPT_DATASET_FILES = ('checksums.sha256', 'validation_report.json', 'metadata.json')


def template_geometry(reg, registration, surfaces):
    """Head size and shell radii of the template about the registered centre.

    ``reg`` maps package coordinates to MRI coordinates; ``registration`` is the
    record written by ``registration_gate``; ``surfaces`` are the BEM surfaces
    in MRI coordinates (metres).
    """
    centre = np.asarray(registration['centre_m'])
    projected = np.array([item['projected_mri_m'] for item in registration['electrodes']])
    electrode_radius = np.linalg.norm(projected - centre, axis=1)
    head_radius = float(np.median(electrode_radius))
    shells = {}
    for name, surface_id in (('inner_skull', FIFF.FIFFV_BEM_SURF_ID_BRAIN),
                             ('outer_skull', FIFF.FIFFV_BEM_SURF_ID_SKULL),
                             ('scalp', FIFF.FIFFV_BEM_SURF_ID_HEAD)):
        surface = next(item for item in surfaces if item['id'] == surface_id)
        local = reg.from_mri(np.asarray(surface['rr']))
        distance = np.linalg.norm(local, axis=1)
        upper = local[:, 2] >= 0
        shells[name] = dict(
            vertices=int(len(distance)), upper_vertices=int(upper.sum()),
            median_all_m=float(np.median(distance)),
            median_upper_m=float(np.median(distance[upper])),
            minimum_m=float(distance.min()), maximum_m=float(distance.max()),
        )
    return dict(
        definition=('Head radius: median distance of the template electrodes, projected to the BEM scalp, '
                    'from the registered centre. Shell radii: median vertex distance in the upper half '
                    '(z >= 0) of the package frame.'),
        electrode_radius_m=dict(median=head_radius, mean=float(electrode_radius.mean()),
                                minimum=float(electrode_radius.min()),
                                maximum=float(electrode_radius.max())),
        head_radius_m=head_radius, scale=head_radius / DEFAULT_RADII_M[-1], shells=shells,
    )


def sphere_variants(geometry):
    """Radii (metres) and conductivities (S/m) of every sphere, keyed by name."""
    scale = geometry['scale']
    scaled = tuple(radius * scale for radius in DEFAULT_RADII_M)
    shells = geometry['shells']
    inner = shells['inner_skull']['median_upper_m']
    fitted = (inner, inner + 0.001, shells['outer_skull']['median_upper_m'],
              shells['scalp']['median_upper_m'])
    variants = {
        'default': (DEFAULT_RADII_M, FOUR_SHELL_S_PER_M),
        'three_shell': (DEFAULT_RADII_M, THREE_SHELL_S_PER_M),
        'size_matched': (scaled, FOUR_SHELL_S_PER_M),
        'size_matched_three_shell': (scaled, THREE_SHELL_S_PER_M),
        'surface_fitted_three_shell': (fitted, THREE_SHELL_S_PER_M),
    }
    for name, (radii, _) in variants.items():
        require(all(a < b for a, b in zip(radii[:-1], radii[1:])),
                f'{name}: shell radii must increase strictly')
    return {name: dict(radii_m=[float(r) for r in radii],
                       conductivities_S_per_m=[float(s) for s in sigma])
            for name, (radii, sigma) in variants.items()}


def variant_config(seed, variant, dataset, smoke):
    cfg = config_for(seed, 'position', dataset, smoke)
    cfg.head.radii_m = list(variant['radii_m'])
    cfg.head.conductivities_S_per_m = list(variant['conductivities_S_per_m'])
    cfg.output.save_components = False
    cfg.output.save_leadfield = False
    return cfg


def prune_dataset(dataset):
    """Keep the small provenance files of a generated dataset and delete the rest."""
    for path in dataset.iterdir():
        if path.name not in KEPT_DATASET_FILES:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()


def finetune_mlp(fit, train, validation, epochs):
    """Continue training a fitted network on new windows, keeping its scalers.

    The starting network is kept when no epoch lowers the validation error.
    """
    model = copy.deepcopy(fit['model'])
    x = fit['x_scaler'].transform(train.topography)
    y = fit['y_scaler'].transform(train.position)
    vx = fit['x_scaler'].transform(validation.topography)

    def loss():
        return float(np.mean((fit['y_scaler'].inverse_transform(model.predict(vx)) -
                              validation.position) ** 2))

    best_loss, best_epoch = loss(), 0
    best = copy.deepcopy((model.coefs_, model.intercepts_))
    epoch = 0
    for epoch in range(1, epochs + 1):
        model.partial_fit(x, y)
        current = loss()
        if current < best_loss - 1e-8:
            best_loss, best_epoch = current, epoch
            best = copy.deepcopy((model.coefs_, model.intercepts_))
        if epoch - best_epoch >= 15:
            break
    model.coefs_, model.intercepts_ = best
    return dict(model=model, x_scaler=fit['x_scaler'], y_scaler=fit['y_scaler'],
                best_epoch=best_epoch, epochs_run=epoch, training_seed=fit['training_seed'])


def bem_split(dataset, split, reg, trans, bem, bem_info, inner_skull):
    """Eligible windows of one split as BEM features, with their eligibility record."""
    windows = load_windows(dataset / f'{split}.h5')
    _, first, inverse = np.unique(windows.pool_id, return_index=True, return_inverse=True)
    positions_m = windows.position * .001
    unique_xyz = positions_m[first]
    require(np.array_equal(unique_xyz[inverse], positions_m), f'inconsistent {split} source coordinates')
    clearance = source_eligibility(reg.to_mri(unique_xyz), inner_skull)
    keep = clearance >= MARGIN_MM
    rows = np.flatnonzero(keep[inverse])
    selected, window_to_source = np.unique(inverse[rows], return_inverse=True)
    points = reg.to_mri(unique_xyz[selected])
    field = np.concatenate([
        forward_columns(points[start:start + FORWARD_CHUNK], bem_info, bem, trans)
        for start in range(0, len(points), FORWARD_CHUNK)
    ])
    features, alpha = bem_features(dataset, rows, field[window_to_source], reg, trans, split=split)
    record = dict(windows=int(len(windows.position)), eligible_windows=int(len(rows)),
                  positions=int(len(unique_xyz)), eligible_positions=int(len(selected)),
                  alpha_median=float(np.median(alpha)), alpha_maximum=float(np.max(alpha)))
    subset = windows.subset(rows)
    return Windows(features, subset.position, subset.pool_id, subset.population, subset.snr), record


def take(windows, index):
    return Windows(windows.topography[index], windows.position[index], windows.pool_id[index],
                   windows.population[index], windows.snr[index])


def pooled(errors, names):
    return np.stack([errors[name] for name in names])


def summarize_arms(errors, groups, arms, seed, draws, reference):
    """Describe each arm (model seeds pooled) and pair it with the reference arm."""
    result = {}
    base = pooled(errors, [f'{reference}_mlp_{ms}' for ms in MODEL_SEEDS])
    for arm in arms:
        stacked = pooled(errors, [f'{arm}_mlp_{ms}' for ms in MODEL_SEEDS])
        entry = describe(stacked, groups, seed, draws)
        if arm != reference:
            entry['median_difference_from_default_mm'] = float(np.median(stacked) - np.median(base))
            entry['median_difference_ci95_mm'] = cluster_interval(
                stacked, groups, np.random.default_rng([seed, 606]), other=base, draws=draws)
        result[arm] = entry
    return result


def evaluate_seed(args, seed, output, reg, trans, surfaces, bem, variants, recorded):
    smoke = args.smoke
    draws = 50 if smoke else 2000
    epochs = 3 if smoke else 300
    path = args.split_run / str(seed) / 'position'
    state = gate_dataset(path, output, reg, trans, surfaces, smoke)
    cfg = state['cfg']
    require(tuple(cfg.head.radii_m) == DEFAULT_RADII_M and
            tuple(cfg.head.conductivities_S_per_m) == FOUR_SHELL_S_PER_M,
            'the split run must use the default four-shell sphere')
    rows, test = state['rows'], state['test']
    positions, groups = test.position[rows], test.pool_id[rows]
    field = forward_columns(reg.to_mri(state['unique_xyz']), state['bem_info'], bem, trans)
    bem_test, alpha = bem_features(state['dataset'], rows, field[state['window_to_source']], reg, trans)
    inner_skull = next(surf for surf in surfaces if surf['id'] == FIFF.FIFFV_BEM_SURF_ID_BRAIN)

    errors, predictions, fits_record = {}, {}, {}

    def score(name, fit, features=bem_test, truth=positions):
        prediction = predict(fit, features)
        predictions[name] = prediction
        errors[name] = np.linalg.norm(prediction - truth, axis=1)

    frozen = {}
    for model_seed in MODEL_SEEDS:
        fit = joblib.load(path / f'mlp_{model_seed}.joblib')
        require(fit['training_seed'] == model_seed, 'frozen model has an unexpected training seed')
        frozen[model_seed] = fit
        score(f'default_mlp_{model_seed}', fit)
        score(f'default_own_sphere_mlp_{model_seed}', fit, state['original'])
    default_median = float(np.median(pooled(errors, [f'default_mlp_{ms}' for ms in MODEL_SEEDS])))
    if recorded is not None:
        saved = recorded['by_dataset_seed'][str(seed)]['paired_pooled_model_seeds']['all']
        require(saved['windows'] == len(rows) and abs(saved['bem_median_mm'] - default_median) <= 1e-6,
                f'seed {seed}: the recorded transfer result was not reproduced '
                f'({saved["bem_median_mm"]} recorded, {default_median} now)')

    own_sphere = {}
    names = [name for name in variants if name != 'default']
    if smoke:
        names = names[1:2]
    for name in names:
        folder = output / name
        folder.mkdir()
        dataset = folder / 'dataset'
        variant_cfg = variant_config(seed, variants[name], dataset, smoke)
        variant_cfg.to_json(folder / 'config.json')
        run_pipeline(variant_cfg, log_level=logging.WARNING)
        train, validation, own_test = (load_windows(dataset / f'{split}.h5')
                                       for split in ('train', 'val', 'test'))
        require(np.array_equal(own_test.pool_id, test.pool_id) and
                np.allclose(own_test.position, test.position, rtol=0, atol=1e-9),
                f'{name}: test sources differ from the default dataset')
        own_sphere[name] = {}
        for model_seed in MODEL_SEEDS:
            fit = train_mlp(train, validation, model_seed, epochs)
            joblib.dump(fit, folder / f'mlp_{model_seed}.joblib')
            fits_record[f'{name}_mlp_{model_seed}'] = dict(best_epoch=fit['best_epoch'],
                                                           epochs_run=fit['epochs_run'])
            score(f'{name}_mlp_{model_seed}', fit)
            score(f'{name}_own_sphere_mlp_{model_seed}', fit, own_test.topography[rows])
            all_error = np.linalg.norm(predict(fit, own_test.topography) - own_test.position, axis=1)
            own_sphere[name][f'mlp_{model_seed}_all_windows'] = describe(all_error, own_test.pool_id, seed, draws)
        if not args.keep_datasets:
            prune_dataset(dataset)
        print(f'{seed} {name}: BEM median '
              f'{np.median(pooled(errors, [f"{name}_mlp_{ms}" for ms in MODEL_SEEDS])):.3f} mm', flush=True)

    finetune_rows = []
    eligibility = {}
    arms = ['default', *names]
    if not args.skip_finetune:
        bem_train, eligibility['train'] = bem_split(state['dataset'], 'train', reg, trans, bem,
                                                     state['bem_info'], inner_skull)
        bem_val, eligibility['validation'] = bem_split(state['dataset'], 'val', reg, trans, bem,
                                                        state['bem_info'], inner_skull)
        write_json(output / 'train_eligibility.json', eligibility)
        sizes = (8, None) if smoke else FINETUNE_SIZES
        for size in sizes:
            label = 'all' if size is None else str(size)
            for model_seed in MODEL_SEEDS:
                rng = np.random.default_rng([seed, model_seed, 0 if size is None else size, 605])
                if size is None:
                    train, validation = bem_train, bem_val
                else:
                    require(size <= len(bem_train.position), 'not enough eligible BEM training windows')
                    count = min(len(bem_val.position), max(25, size // 4))
                    train = take(bem_train, np.sort(rng.choice(len(bem_train.position), size=size, replace=False)))
                    validation = take(bem_val, np.sort(rng.choice(len(bem_val.position), size=count, replace=False)))
                for arm, fit in (('finetuned', finetune_mlp(frozen[model_seed], train, validation, epochs)),
                                 ('scratch', train_mlp(train, validation, model_seed, epochs))):
                    key = f'{arm}_{label}_mlp_{model_seed}'
                    score(key, fit)
                    finetune_rows.append(dict(
                        dataset_seed=seed, model_seed=model_seed, bem_training_windows=len(train.position),
                        bem_validation_windows=len(validation.position), size=label, arm=arm,
                        best_epoch=fit['best_epoch'], epochs_run=fit['epochs_run'],
                        median_mm=float(np.median(errors[key])), mean_mm=float(np.mean(errors[key])),
                        p90_mm=float(np.quantile(errors[key], .9))))
            arms.extend([f'finetuned_{label}', f'scratch_{label}'])
            print(f'{seed} BEM training windows {label}: fine-tuned '
                  f'{np.median(pooled(errors, [f"finetuned_{label}_mlp_{ms}" for ms in MODEL_SEEDS])):.3f} mm, '
                  f'from scratch '
                  f'{np.median(pooled(errors, [f"scratch_{label}_mlp_{ms}" for ms in MODEL_SEEDS])):.3f} mm',
                  flush=True)
        write_csv(output / 'finetune.csv', finetune_rows)

    radius = np.linalg.norm(positions, axis=1)
    strata = {}
    for label, mask in populations_and_bands(test.population[rows], radius):
        if np.any(mask):
            subset = {key: value[mask] for key, value in errors.items()}
            strata[label] = summarize_arms(subset, groups[mask], arms, seed, draws, 'default')
    own = {name: dict(eligible_windows=describe(
        pooled(errors, [f'{name}_own_sphere_mlp_{ms}' for ms in MODEL_SEEDS]), groups, seed, draws),
        **own_sphere.get(name, {})) for name in ['default', *names]}
    write_csv(output / 'adaptation_predictions.csv', [
        dict(window_id=int(row), position_pool_id=int(groups[i]), population=int(test.population[row]),
             source_radius_mm=float(radius[i]), snr_db_referenced=float(test.snr[row]),
             alpha_bem=float(alpha[i]),
             **{f'{key}_error_mm': float(value[i]) for key, value in errors.items()})
        for i, row in enumerate(rows)
    ])
    np.savez_compressed(output / 'predictions.npz', window_id=rows, true_position_mm=positions,
                        **{key: value for key, value in predictions.items()})
    summary = dict(bem_windows=int(len(rows)), positions=int(len(np.unique(groups))),
                   arms=strata['all'], strata=strata, own_sphere=own, fits=fits_record,
                   bem_training_eligibility=eligibility, finetune=finetune_rows)
    write_json(output / 'summary.json', summary)
    return summary, dict(state=state, bem_test=bem_test, rows=rows, errors=errors)


def dipole_arms(args, output, seed, kept, variants, transfer_run):
    """Fit the BEM windows with each sphere and its own montage (package frame)."""
    state, features, rows = kept['state'], kept['bem_test'], kept['rows']
    test, cfg = state['test'], state['cfg']
    count = 4 if args.smoke else args.ecd_count
    draws = 50 if args.smoke else 2000
    indices = np.sort(np.random.default_rng([seed, 602]).choice(
        len(rows), size=min(count, len(rows)), replace=False))
    truth = test.position[rows[indices]]
    groups = test.pool_id[rows[indices]]
    electrodes, names = sensor_info(state['dataset'] / 'test.h5')
    errors, estimates = {}, {}
    for name, variant in variants.items():
        if args.smoke and name not in ('default', 'size_matched'):
            continue
        radii = variant['radii_m']
        scaled = electrodes * (radii[-1] / np.linalg.norm(electrodes, axis=1, keepdims=True))
        info, sphere = build_sphere_and_info(scaled, names, radii, variant['conductivities_S_per_m'],
                                             cfg.signal.sampling_freq_hz, mne)
        estimates[name] = dipole_fit(features[indices], info, sphere)
        errors[f'dipole_{name}_sphere_on_bem'] = np.linalg.norm(estimates[name] - truth, axis=1)
    if transfer_run is not None and not args.smoke:
        import csv
        with (transfer_run / str(seed) / 'ecd_subset.csv').open(encoding='utf-8', newline='') as stream:
            recorded = list(csv.DictReader(stream))
        require([int(item['window_id']) for item in recorded] == [int(row) for row in rows[indices]],
                'the recorded dipole subset uses different windows')
        for column, key in (('ecd_package_sphere_matched_error_mm', 'dipole_default_sphere_on_sphere_recorded'),
                            ('ecd_bem_matched_error_mm', 'dipole_bem_on_bem_recorded')):
            errors[key] = np.array([float(item[column]) for item in recorded])
    for key in list(kept['errors']):
        if key.startswith(('default_mlp_', 'size_matched_three_shell_mlp_', 'surface_fitted_three_shell_mlp_')):
            errors[key] = kept['errors'][key][indices]
    summary = {key: describe(value, groups, seed, draws) for key, value in errors.items()}
    reference = errors.get('dipole_bem_on_bem_recorded')
    if reference is not None:
        for key, value in errors.items():
            if key.startswith('dipole_') and key.endswith('_sphere_on_bem'):
                summary[key]['median_difference_from_bem_matched_mm'] = float(np.median(value) - np.median(reference))
                summary[key]['median_difference_ci95_mm'] = cluster_interval(
                    value, groups, np.random.default_rng([seed, 607]), other=reference, draws=draws)
    write_csv(output / 'dipole_subset.csv', [
        dict(window_id=int(rows[index]), position_pool_id=int(groups[i]),
             **{f'{key}_error_mm': float(value[i]) for key, value in errors.items()},
             **{f'dipole_{name}_{axis}_mm': float(value[i, j]) for name, value in estimates.items()
                for j, axis in enumerate('xyz')})
        for i, index in enumerate(indices)
    ])
    return summary


def run(args):
    require(mne.__version__ == '1.12.1', 'this protocol pins MNE-Python 1.12.1')
    mne.set_log_level('WARNING')
    args.output = check_new_output(args.output, ROOT)
    template = args.subjects_dir / 'fsaverage/bem'
    head_file = template / 'fsaverage-5120-5120-5120-bem.fif'
    trans_file = template / 'fsaverage-trans.fif'
    trans = invert_transform(mne.read_trans(trans_file, verbose=False))
    reg = Registration.from_recording_electrodes()
    source_files = [Path(__file__), ROOT / 'scripts/run_head_transfer.py', ROOT / 'scripts/run_split_study.py',
                    ROOT / 'scripts/train_baseline.py', ROOT / 'scripts/validate_patch_forward_mne.py',
                    ROOT / 'validation-requirements.txt', ROOT / 'validation-windows-lock.txt']
    provenance = collect_provenance(ROOT, source_files, sys.argv[1:], require_clean=not args.smoke)
    provenance.update(dict(template_sha256={file.name: digest(file) for file in (head_file, trans_file)},
                           inputs={}, smoke=args.smoke))
    recorded = None
    if args.transfer_run is not None:
        recorded = json.loads((args.transfer_run / 'summary.json').read_text(encoding='utf-8'))
        provenance['recorded_transfer_summary_sha256'] = digest(args.transfer_run / 'summary.json')
    args.output.mkdir(parents=True)
    (args.output / 'environment.txt').write_text(provenance['resolved_packages'], encoding='utf-8')
    seeds = DATA_SEEDS[:1] if args.smoke else DATA_SEEDS
    cfg = Config.from_json(args.split_run / str(seeds[0]) / 'position/config.json')
    surfaces = pick_bem_surfaces(args.subjects_dir, cfg)
    bem = mne.make_bem_solution(surfaces, verbose=False)
    # The registration record depends on the montage and template only, not on the dataset seed.
    electrodes, names = sensor_info(args.split_run / str(seeds[0]) / 'position/dataset/test.h5')
    scalp = next(surf for surf in surfaces if surf['id'] == FIFF.FIFFV_BEM_SURF_ID_HEAD)
    _, registration = registration_gate(reg, electrodes, names, trans, scalp, cfg.signal.sampling_freq_hz)
    geometry = template_geometry(reg, registration, surfaces)
    variants = sphere_variants(geometry)
    geometry['variants'] = variants
    write_json(args.output / 'geometry.json', geometry)
    summaries, kept = {}, None
    for seed in seeds:
        path = args.split_run / str(seed) / 'position'
        paths = [path / 'config.json', path / 'dataset/checksums.sha256',
                 *(path / f'mlp_{model_seed}.joblib' for model_seed in MODEL_SEEDS)]
        provenance['inputs'][seed] = {p.relative_to(path).as_posix(): digest(p) for p in paths}
        write_json(args.output / 'provenance.json', provenance)
        output = args.output / str(seed)
        output.mkdir()
        summaries[seed], state = evaluate_seed(args, seed, output, reg, trans, surfaces, bem, variants, recorded)
        if seed == DATA_SEEDS[0]:
            kept = state
            dipoles = dipole_arms(args, output, seed, kept, variants, args.transfer_run)
            write_json(output / 'dipole_summary.json', dipoles)
        write_json(args.output / 'summary.json', dict(smoke=args.smoke, by_dataset_seed=summaries))
    headline = {}
    for arm in summaries[seeds[0]]['arms']:
        values = [summary['arms'][arm]['median_mm'] for summary in summaries.values()]
        headline[arm] = dict(dataset_seed_medians_mm=values, mean_of_dataset_seed_medians=float(np.mean(values)),
                             range=[float(np.min(values)), float(np.max(values))])
        differences = [summary['arms'][arm].get('median_difference_from_default_mm')
                       for summary in summaries.values()]
        if all(value is not None for value in differences):
            headline[arm]['mean_difference_from_default_mm'] = float(np.mean(differences))
    own = {}
    for name in summaries[seeds[0]]['own_sphere']:
        values = [summary['own_sphere'][name]['eligible_windows']['median_mm'] for summary in summaries.values()]
        own[name] = dict(mean_of_dataset_seed_medians=float(np.mean(values)),
                         range=[float(np.min(values)), float(np.max(values))])
    write_json(args.output / 'summary.json', dict(
        smoke=args.smoke, geometry=geometry, bem_windows_headline=headline, own_sphere_eligible_headline=own,
        dipole_subset=dipoles, by_dataset_seed=summaries))
    record_output_hashes(args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--split-run', required=True, type=Path,
                        help='completed run_split_study.py output directory')
    parser.add_argument('--subjects-dir', required=True, type=Path,
                        help='directory that contains fsaverage/bem')
    parser.add_argument('--output', required=True, type=Path, help='new result directory')
    parser.add_argument('--transfer-run', type=Path,
                        help='recorded run_head_transfer.py output; checked for agreement and used to pair dipole fits')
    parser.add_argument('--ecd-count', type=int, default=500, help='windows in the dipole-fit subset')
    parser.add_argument('--skip-finetune', action='store_true',
                        help='omit the BEM-trained reference and the fine-tuning curve')
    parser.add_argument('--keep-datasets', action='store_true', help='keep the generated variant datasets')
    parser.add_argument('--smoke', action='store_true', help='use a small smoke split run, seed 862 only')
    args = parser.parse_args()
    if args.ecd_count < 1:
        parser.error('--ecd-count must be positive')
    with threadpool_limits(limits=1):
        try:
            run(args)
        except ValueError as error:
            parser.error(str(error))


if __name__ == '__main__':
    main()
