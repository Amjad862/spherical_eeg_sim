"""Evaluate frozen sphere-trained point-source models on paired fsaverage EEG."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import joblib
import mne
import numpy as np
from mne._fiff.constants import FIFF
from mne.surface import _CheckInside, _project_onto_surface
from mne.transforms import apply_trans, invert_transform
from threadpoolctl import threadpool_limits

from run_split_study import (
    DATA_SEEDS,
    DEPTH_BANDS,
    MODEL_SEEDS,
    cluster_interval,
    describe,
    dipole_fit,
    load_windows,
    peak_gfp_topography,
    predict,
    record_output_hashes,
    sensor_info,
)
from spherical_eeg_sim import Config
from spherical_eeg_sim.montages import _generate as montage_generator
from spherical_eeg_sim.sources.sampling import world_orientation_to_spherical_frame
from validate_patch_forward_mne import build_sphere_and_info
from _run_provenance import check_new_output, collect_provenance, hash_file as digest


ROOT = Path(__file__).resolve().parents[1]
MARGIN_MM = 5.0
EXACTNESS_LIMIT = 1e-6
REGISTRATION_DEG = 1.5
REPLAY_LIMIT = 1e-10
SNR_LIMIT_DB = 1e-9
PHYSICAL_MEDIAN_LIMIT = .005
PHYSICAL_WINDOW_LIMIT = .05
MODEL_LIMIT_MM = .5


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def normalize_rows(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


@dataclass(frozen=True)
class Registration:
    """Rigid package-sphere to native MRI map. All coordinates here use metres."""

    centre: np.ndarray
    axes: np.ndarray

    @classmethod
    def from_recording_electrodes(cls):
        points = montage_generator._parse_elc(
            montage_generator._TEMPLATES / 'standard_1005.elc'
        )
        rec = [xyz for name, xyz in points.items()
               if name not in montage_generator.FIDUCIALS
               and name not in montage_generator.DEPRECATED]
        require(len(rec) == 336, 'unexpected template recording-channel count')
        centre_mm = montage_generator._fit_sphere(np.array(rec))
        up = normalize_rows(points['Cz'] - centre_mm)
        anterior = points['Fpz'] - centre_mm
        front = normalize_rows(anterior - np.dot(anterior, up) * up)
        right = normalize_rows(np.cross(front, up))
        return cls(centre_mm * .001, np.stack([right, front, up]))

    def to_mri(self, points):
        return points @ self.axes + self.centre

    def from_mri(self, points):
        return (points - self.centre) @ self.axes.T


def info_for(electrodes_head, names, sampling_freq):
    info = mne.create_info(names, sampling_freq, 'eeg', verbose=False)
    info.set_montage(mne.channels.make_dig_montage(
        ch_pos=dict(zip(names, electrodes_head)), coord_frame='head'
    ), verbose=False)
    return info


def average_reference(array):
    return array - array.mean(axis=-2, keepdims=True)


def forward_columns(positions_mri, info, conductor, trans=None):
    """Return (position, channel, x/y/z in head frame) fields in V/(A m)."""
    normals = np.zeros_like(positions_mri)
    normals[:, 2] = 1.
    src = mne.setup_volume_source_space(
        pos=dict(rr=np.ascontiguousarray(positions_mri), nn=normals),
        sphere_units='m', verbose=False,
    )
    fwd = mne.make_forward_solution(
        info, trans, src, conductor, eeg=True, meg=False, mindist=0.,
        n_jobs=1, verbose=False,
    )
    positions_head = positions_mri if trans is None else apply_trans(trans, positions_mri)
    require(fwd['nsource'] == len(positions_mri) and
            np.allclose(fwd['source_rr'], positions_head, rtol=0, atol=1e-10),
            'MNE omitted or reordered requested sources')
    require(fwd['sol']['row_names'] == info.ch_names,
            'MNE changed electrode order')
    basis = np.tile(np.eye(3), (len(positions_mri), 1))
    require(np.array_equal(fwd['source_nn'], basis),
            'MNE forward columns are not head-frame Cartesian components')
    return fwd['sol']['data'].reshape(len(info.ch_names), len(positions_mri), 3).transpose(1, 0, 2)


def norm(array):
    return np.linalg.norm(array.reshape(len(array), -1), axis=1)


def relative_error(observed, expected):
    denominator = norm(expected)
    require(np.all(np.isfinite(denominator) & (denominator > 0.)),
            'non-finite or zero reference norm')
    result = norm(observed - expected) / denominator
    require(np.all(np.isfinite(result)), 'non-finite relative error')
    return result


def stats(values):
    if not len(values):
        return None
    return dict(median=float(np.median(values)), p90=float(np.quantile(values, .9)),
                p99=float(np.quantile(values, .99)), maximum=float(np.max(values)))


def registration_gate(reg, package_electrodes, names, trans, scalp, sampling_freq):
    """Check unprojected electrode directions using the BEM montage path."""
    require(np.allclose(reg.axes @ reg.axes.T, np.eye(3), atol=1e-12, rtol=0) and
            np.isclose(np.linalg.det(reg.axes), 1., atol=1e-12, rtol=0),
            'registration axes must be a proper rotation')
    montage = mne.channels.make_standard_montage('standard_1005', head_size=None)
    positions = montage.get_positions()
    require(positions['coord_frame'] == 'mri', 'unexpected standard_1005 frame')
    native_mri = np.stack([positions['ch_pos'][name] for name in names])
    info = info_for(apply_trans(trans, native_mri), names, sampling_freq)
    info_electrodes = np.array([channel['loc'][:3] for channel in info['chs']])
    mri_round_trip = apply_trans(invert_transform(trans), info_electrodes)
    require(np.allclose(native_mri, mri_round_trip, rtol=0, atol=1e-9),
            'native template electrode round trip failed')
    aligned = reg.to_mri(package_electrodes)
    a = normalize_rows(aligned - reg.centre)
    b = normalize_rows(mri_round_trip - reg.centre)
    angle = np.rad2deg(np.arccos(np.clip(np.sum(a * b, axis=-1), -1., 1.)))
    require(np.max(angle) <= REGISTRATION_DEG,
            f'unprojected registration discrepancy {np.max(angle):.4g}°')
    # MNE uses this same triangular projection when integrating scalp electrodes.
    _, _, projected = _project_onto_surface(mri_round_trip, scalp, project_rrs=True)
    return info, dict(
        centre_m=reg.centre.tolist(), axes=reg.axes.tolist(),
        determinant=float(np.linalg.det(reg.axes)), max_angle_deg=float(np.max(angle)),
        mri_to_head=trans['trans'].tolist(),
        electrodes=[dict(name=name, angle_deg=float(deg), native_mri_m=raw.tolist(),
                         projected_mri_m=prj.tolist(),
                         projection_displacement_mm=float(np.linalg.norm(prj - raw) * 1e3),
                         package_to_native_mm=float(np.linalg.norm(mapped - raw) * 1e3))
                    for name, deg, raw, prj, mapped in
                    zip(names, angle, mri_round_trip, projected, aligned)],
    )


def source_eligibility(positions_mri, inner_skull):
    """Measure clearance to the triangular inner-skull surface in MRI space."""
    inside = _CheckInside(inner_skull)(positions_mri, n_jobs=1)
    _, _, nearest = _project_onto_surface(positions_mri, inner_skull, project_rrs=True)
    distance_mm = np.linalg.norm(positions_mri - nearest, axis=1) * 1000
    return np.where(inside, distance_mm, -distance_mm)


def sphere_fields(positions, electrodes, names, cfg, reg):
    radii = cfg.head.radii_m
    conductivities = cfg.head.conductivities_S_per_m
    sampling_freq = cfg.signal.sampling_freq_hz
    info, sphere = build_sphere_and_info(electrodes, names, radii, conductivities,
                                         sampling_freq, mne)
    original = average_reference(forward_columns(positions, info, sphere))
    rotated_info = info_for(reg.to_mri(electrodes), names, sampling_freq)
    rotated_sphere = mne.make_sphere_model(
        r0=reg.centre, head_radius=radii[-1],
        relative_radii=np.asarray(radii) / radii[-1],
        sigmas=conductivities, verbose=False,
    )
    rotated = forward_columns(reg.to_mri(positions), rotated_info, rotated_sphere)
    residual = relative_error(average_reference(rotated @ reg.axes.T), original)
    require(np.max(residual) <= EXACTNESS_LIMIT,
            f'rigidly transformed sphere relative residual {np.max(residual):.4g}')
    return original, info, sphere, stats(residual)


def signal_from_truth(fields, orientation, magnitude_nAm, course):
    """Construct an average-referenced signal window in physical volts."""
    topography = np.einsum('nck,nk->nc', fields, orientation) * (magnitude_nAm[:, None] * 1e-9)
    return average_reference(topography[:, :, None] * course[:, None, :])


def rms(array):
    value = np.sqrt(np.mean(array ** 2, axis=(1, 2)))
    require(np.all(np.isfinite(value) & (value > 0.)), 'non-finite or zero window RMS')
    return value


def replay_window(signal, original_signal, noise, snr_db):
    """Preserve the saved noise pattern and referenced SNR."""
    amplitude_ratio = rms(signal) / rms(original_signal)
    new_noise = amplitude_ratio[:, None, None] * noise
    achieved = 20. * np.log10(rms(signal) / rms(new_noise))
    require(np.max(abs(achieved - snr_db)) <= SNR_LIMIT_DB,
            'rebuilt window did not preserve referenced SNR')
    total = signal + new_noise
    return total / rms(total)[:, None, None], amplitude_ratio


def pick_bem_surfaces(subjects_dir, cfg):
    brain, _, skull, scalp = cfg.head.conductivities_S_per_m
    path = subjects_dir / 'fsaverage/bem/fsaverage-5120-5120-5120-bem.fif'
    surfaces = mne.read_bem_surfaces(path, verbose=False)
    conductivity = {FIFF.FIFFV_BEM_SURF_ID_BRAIN: brain,
                    FIFF.FIFFV_BEM_SURF_ID_SKULL: skull,
                    FIFF.FIFFV_BEM_SURF_ID_HEAD: scalp}
    require({surface['id'] for surface in surfaces} == set(conductivity),
            'fsaverage BEM must contain brain, skull, and scalp surfaces')
    for surface in surfaces:
        surface['sigma'] = conductivity[surface['id']]
    return surfaces


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def populations_and_bands(population, radius_mm):
    yield 'all', np.ones(len(population), dtype=bool)
    for pop in (0, 1):
        yield f'population_{pop}', population == pop
    for pop, lo, hi in DEPTH_BANDS:
        yield f'radius_{pop}_{lo:g}_{hi:g}', (
            (population == pop) & (radius_mm >= lo) & (radius_mm < hi)
        )


def geometry_counts(positions, pool_id, population, eligible):
    radius = np.linalg.norm(positions, axis=1) * 1000
    strata = list(populations_and_bands(population, radius))
    for axis, label in enumerate(('right', 'anterior', 'superior')):
        strata.append((label, positions[:, axis] >= 0))
        strata.append((f'negative_{label}', positions[:, axis] < 0))
    return {name: dict(windows=int(np.sum(mask)), eligible_windows=int(np.sum(mask & eligible)),
                       positions=len(np.unique(pool_id[mask])),
                       eligible_positions=len(np.unique(pool_id[mask & eligible])))
            for name, mask in strata}


def truth_batch(gt, rows):
    return (gt['orientations_world'][rows], gt['magnitudes_nAm'][rows],
            gt['source_time_course'][rows])


def reconstruction(dataset, rows, positions, columns, sphere_columns):
    sphere_features, diagnostic_features, relative, norm_ratio, correlation = [], [], [], [], []
    replay_checks = np.zeros(3)
    with h5py.File(dataset / 'test.h5', 'r') as file:
        gt = file['ground_truth']
        require(file['eeg'].dtype == np.dtype('float64'),
                'head transfer requires float64 for exact physical replay')
        for start in range(0, len(rows), 128):
            subset = slice(start, start + 128)
            idx = rows[subset]
            original = file['eeg'][idx]
            saved_signal = gt['eeg_signal_only'][idx]
            saved_noise = gt['eeg_noise_only'][idx]
            require(np.array_equal(saved_signal + saved_noise, original),
                    'stored EEG does not exactly equal its two saved components')
            require(np.max(abs(rms(original) - 1.)) <= REPLAY_LIMIT,
                    'stored windows must have unit RMS')
            divisor = gt['eeg_normalization_scale_V'][idx][:, None, None]
            require(np.all(np.isfinite(divisor) & (divisor > 0)),
                    'invalid stored volt-per-unit normalization scale')
            signal, noise = saved_signal * divisor, saved_noise * divisor
            snr = gt['snr_db_referenced'][idx]
            orientation, amplitude, course = truth_batch(gt, idx)
            local = np.stack([world_orientation_to_spherical_frame(o, p)
                              for o, p in zip(orientation, positions[subset])])
            analytic = signal_from_truth(columns[subset], local, amplitude, course)
            rebuilt, alpha = replay_window(analytic, signal, noise, snr)
            batch_checks = np.array([np.max(relative_error(analytic, signal)),
                                     np.max(abs(alpha - 1.)),
                                     np.max(relative_error(rebuilt, original))])
            replay_checks = np.maximum(replay_checks, batch_checks)
            require(np.max(replay_checks) <= REPLAY_LIMIT,
                    f'physical replay failed (signal, alpha, total): {replay_checks}')
            diagnostic = signal_from_truth(sphere_columns[subset], orientation, amplitude, course)
            rebuilt_mne, alpha = replay_window(diagnostic, signal, noise, snr)
            relative.extend(relative_error(diagnostic, signal))
            norm_ratio.extend(alpha)
            correlation.extend(np.sum(diagnostic * signal, axis=(1, 2)) /
                               (norm(diagnostic) * norm(signal)))
            sphere_features.append(peak_gfp_topography(original))
            diagnostic_features.append(peak_gfp_topography(rebuilt_mne))
    return (np.concatenate(sphere_features), np.concatenate(diagnostic_features),
            np.asarray(relative), np.asarray(norm_ratio), np.asarray(correlation),
            replay_checks.tolist())


def bem_features(dataset, rows, columns, reg, trans, split='test'):
    features, alpha_bem = [], []
    with h5py.File(dataset / f'{split}.h5', 'r') as file:
        gt = file['ground_truth']
        for start in range(0, len(rows), 128):
            subset = slice(start, start + 128)
            idx = rows[subset]
            divisor = gt['eeg_normalization_scale_V'][idx][:, None, None]
            signal = gt['eeg_signal_only'][idx] * divisor
            noise = gt['eeg_noise_only'][idx] * divisor
            orientation, amplitude, course = truth_batch(gt, idx)
            orientation_head = apply_trans(trans, orientation @ reg.axes, move=False)
            diagnostic = signal_from_truth(columns[subset], orientation_head, amplitude, course)
            replay, alpha = replay_window(diagnostic, signal, noise, gt['snr_db_referenced'][idx])
            features.append(peak_gfp_topography(replay))
            alpha_bem.extend(alpha)
    return np.concatenate(features), np.asarray(alpha_bem)


def write_csv(path, records):
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def gate_dataset(path, output, reg, trans, surfaces, smoke):
    dataset = path / 'dataset'
    cfg = Config.from_json(path / 'config.json')
    require(cfg.source_mode == 'point' and cfg.split.policy == 'position' and
            cfg.output.reference_mode == 'average' and cfg.output.save_components and
            cfg.output.eeg_normalization == 'per_sample_rms' and
            cfg.noise.noise_model == 'snr_target' and cfg.noise.snr_definition == 'referenced',
            'need the declared exact-position, average-referenced point dataset with saved components')
    electrodes, names = sensor_info(dataset / 'test.h5')
    require(len(names) == 69, 'the registered transfer uses the 69-channel montage')
    sphere_test = load_windows(dataset / 'test.h5')
    pool_ids, first, inverse = np.unique(sphere_test.pool_id, return_index=True, return_inverse=True)
    positions_m = sphere_test.position * .001
    unique_xyz = positions_m[first]
    require(np.array_equal(unique_xyz[inverse], positions_m), 'inconsistent test source coordinates')

    scalp = next(surf for surf in surfaces if surf['id'] == FIFF.FIFFV_BEM_SURF_ID_HEAD)
    skull = next(surf for surf in surfaces if surf['id'] == FIFF.FIFFV_BEM_SURF_ID_BRAIN)
    bem_info, registration = registration_gate(reg, electrodes, names, trans, scalp,
                                               cfg.signal.sampling_freq_hz)
    clearance = source_eligibility(reg.to_mri(unique_xyz), skull)[inverse]
    eligible = clearance >= MARGIN_MM
    rows = np.flatnonzero(eligible)
    write_json(output / 'registration.json', registration)
    write_json(output / 'eligibility.json', dict(
        minimum_inner_skull_clearance_mm=MARGIN_MM,
        strata=geometry_counts(positions_m, sphere_test.pool_id, sphere_test.population, eligible),
    ))
    write_csv(output / 'eligibility.csv', [
        dict(window_id=i, position_pool_id=int(pool_id), population=int(pop),
             x_mm=xyz[0], y_mm=xyz[1], z_mm=xyz[2], inner_skull_clearance_mm=float(mm),
             eligible=bool(keep))
        for i, (pool_id, pop, xyz, mm, keep) in enumerate(zip(
            sphere_test.pool_id, sphere_test.population, sphere_test.position, clearance, eligible
        ))
    ])
    require(len(rows) > 0, 'no test windows meet the inner-skull eligibility rule')
    selected, window_to_source = np.unique(inverse[rows], return_inverse=True)
    sphere_field, sphere_info, sphere, exactness = sphere_fields(
        unique_xyz[selected], electrodes, names, cfg, reg
    )
    with h5py.File(dataset / 'leadfield.h5', 'r') as file:
        stored_pos = file['positions_xyz'][pool_ids[selected]]
        require(np.allclose(stored_pos, unique_xyz[selected], rtol=0, atol=1e-15),
                'lead-field artifact does not match the test coordinates')
        analytic = file['leadfield'][pool_ids[selected]]
    original, mne_sphere, relative, ratios, correlation, replay = reconstruction(
        dataset, rows, positions_m[rows], analytic[window_to_source],
        sphere_field[window_to_source]
    )
    radius_mm = np.linalg.norm(sphere_test.position[rows], axis=1)
    physical = {
        label: dict(windows=int(np.sum(mask)), relative=stats(relative[mask]),
                    norm_ratio=stats(ratios[mask]), signed_correlation=stats(correlation[mask]))
        for label, mask in populations_and_bands(sphere_test.population[rows], radius_mm)
    }
    physical_pass = (np.median(relative) <= PHYSICAL_MEDIAN_LIMIT and
                     np.max(relative) <= PHYSICAL_WINDOW_LIMIT)
    results = dict(
        scope='smoke' if smoke else 'recorded protocol', check1_exactness_relative=exactness,
        check2_registration_max_angle_deg=registration['max_angle_deg'],
        check3_replay_maxima=dict(zip(('signal_relative', 'alpha_minus_one_abs', 'total_relative'),
                                     replay)),
        check4_physical=physical, check4_pass=bool(physical_pass),
    )
    write_json(output / 'adapter_gates.json', results)
    require(physical_pass, f'physical sphere agreement failed: {physical["all"]}')
    return dict(cfg=cfg, dataset=dataset, test=sphere_test, rows=rows, unique_xyz=unique_xyz[selected],
                window_to_source=window_to_source, original=original, mne_sphere=mne_sphere,
                residual=relative, ratio=ratios, correlation=correlation, gates=results,
                bem_info=bem_info, sphere_info=sphere_info, sphere=sphere)


def model_gate(path, output, state, seed, draws):
    rows = state['rows']
    test = state['test']
    positions = test.position[rows]
    pool_ids = test.pool_id[rows]
    fitted = {}
    diagnostics = {}
    for training_seed in MODEL_SEEDS:
        key = f'mlp_{training_seed}'
        fit = joblib.load(path / f'{key}.joblib')
        require(fit['training_seed'] == training_seed, f'{key} artifact has incorrect training seed')
        original = predict(fit, state['original'])
        independent = predict(fit, state['mne_sphere'])
        original_error = np.linalg.norm(original - positions, axis=1)
        independent_error = np.linalg.norm(independent - positions, axis=1)
        delta = float(np.median(independent_error) - np.median(original_error))
        all_error = np.linalg.norm(predict(fit, test.topography) - test.position, axis=1)
        diagnostics[key] = dict(
            package_sphere_all=describe(all_error, test.pool_id, seed, draws),
            package_sphere_eligible=describe(original_error, pool_ids, seed, draws),
            independent_sphere_eligible=describe(independent_error, pool_ids, seed, draws),
            median_difference_mm=delta,
            per_window_prediction_shift_mm=stats(np.linalg.norm(independent - original, axis=1)),
            passes=abs(delta) <= MODEL_LIMIT_MM,
        )
        fitted[key] = (fit, original, independent)
    state['gates']['check5_model'] = diagnostics
    write_json(output / 'adapter_gates.json', state['gates'])
    return fitted, all(item['passes'] for item in diagnostics.values())


def fit_comparators(state, reg, trans, bem, seed, count, draw_count, errors, predictions, output):
    test = state['test']
    rows = state['rows']
    indices = np.sort(np.random.default_rng([seed, 602]).choice(
        len(rows), size=min(count, len(rows)), replace=False
    ))
    ground_truth = test.position[rows[indices]]
    groups = test.pool_id[rows[indices]]
    fitted_sphere = mne.make_sphere_model(
        r0='auto', head_radius='auto', info=state['bem_info'], verbose=False,
    )
    write_json(output / 'fitted_sphere_model.json', dict(
        description='Automatically fitted sphere with MNE default relative radii and conductivities.',
        coordinate_frame='MNE head', centre_m=np.asarray(fitted_sphere['r0']).tolist(),
        layer_radii_m=[float(layer['rad']) for layer in fitted_sphere['layers']],
        conductivities_S_per_m=[float(layer['sigma']) for layer in fitted_sphere['layers']],
        fitting_points=len(mne.bem.get_fitting_dig(state['bem_info'], verbose=False)),
    ))
    original_ecd = dipole_fit(state['original'][indices], state['sphere_info'], state['sphere'])
    ecd_bem_head_mm = dipole_fit(state['bem'][indices], state['bem_info'], bem, trans=trans)
    ecd_sphere_head_mm = dipole_fit(state['bem'][indices], state['bem_info'], fitted_sphere)
    head_to_mri = invert_transform(trans)
    ecd_bem = reg.from_mri(apply_trans(head_to_mri, ecd_bem_head_mm * .001)) * 1000.
    ecd_sphere = reg.from_mri(apply_trans(head_to_mri, ecd_sphere_head_mm * .001)) * 1000.
    comparator = {name: np.linalg.norm(pred - ground_truth, axis=1)
                  for name, pred in (('ecd_package_sphere_matched', original_ecd),
                                     ('ecd_bem_matched', ecd_bem),
                                     ('ecd_bem_fitted_sphere', ecd_sphere))}
    comparator.update({key: value[indices] for key, value in errors.items()})
    summary = {key: describe(value, groups, seed, draw_count) for key, value in comparator.items()}
    a, b = comparator['ecd_bem_matched'], comparator['ecd_package_sphere_matched']
    summary['ecd_matched_bem_minus_sphere'] = dict(
        median_difference_mm=float(np.median(a) - np.median(b)),
        median_difference_ci95_mm=cluster_interval(a, groups, np.random.default_rng([seed, 603]),
                                                    other=b, draws=draw_count),
    )
    write_csv(output / 'ecd_subset.csv', [
        dict(window_id=int(rows[index]), position_pool_id=int(groups[i]),
             **{f'{key}_error_mm': float(value[i]) for key, value in comparator.items()},
             **{f'{name}_{axis}_mm': float(value[i, j])
                for name, value in (('ecd_package_sphere_matched', original_ecd),
                                    ('ecd_bem_matched', ecd_bem), ('ecd_bem_fitted_sphere', ecd_sphere))
                for j, axis in enumerate('xyz')})
        for i, index in enumerate(indices)
    ])
    return summary


def evaluate(path, output, state, fitted, reg, trans, bem, seed, smoke):
    rows = state['rows']
    test = state['test']
    positions = test.position[rows]
    groups = test.pool_id[rows]
    field = forward_columns(reg.to_mri(state['unique_xyz']), state['bem_info'], bem, trans)
    state['bem'], alpha = bem_features(state['dataset'], rows, field[state['window_to_source']],
                                       reg, trans)
    train = load_windows(state['dataset'] / 'train.h5')
    predictions = {'constant': np.broadcast_to(train.position.mean(axis=0), positions.shape)}
    paired = {}
    draws = 50 if smoke else 2000
    for key, (fit, original, independent) in fitted.items():
        predictions[f'{key}_package_sphere'] = original
        predictions[f'{key}_independent_sphere'] = independent
        predictions[f'{key}_bem'] = predict(fit, state['bem'])
    errors = {key: np.linalg.norm(value - positions, axis=1)
              for key, value in predictions.items()}
    strata = {}
    for label, mask in populations_and_bands(test.population[rows], np.linalg.norm(positions, axis=1)):
        if not np.any(mask):
            continue
        ids = groups[mask]
        strata[label] = {key: describe(value[mask], ids, seed, draws)
                         for key, value in errors.items()}
        a = np.stack([errors[f'{key}_bem'][mask] for key in fitted])
        b = np.stack([errors[f'{key}_package_sphere'][mask] for key in fitted])
        paired[label] = dict(
            position_clusters=len(np.unique(ids)), windows=int(np.sum(mask)),
            bem_median_mm=float(np.median(a)), package_sphere_median_mm=float(np.median(b)),
            median_difference_mm=float(np.median(a) - np.median(b)),
            median_difference_ci95_mm=cluster_interval(a, ids, np.random.default_rng([seed, 604]),
                                                        other=b, draws=draws),
        )
    write_csv(output / 'paired_predictions.csv', [
        dict(window_id=int(row), position_pool_id=int(groups[i]),
             population=int(test.population[row]), source_radius_mm=float(np.linalg.norm(positions[i])),
             snr_db_referenced=float(test.snr[row]), alpha_bem=float(alpha[i]),
             mne_sphere_relative_residual=float(state['residual'][i]),
             mne_sphere_norm_ratio=float(state['ratio'][i]),
             mne_sphere_signed_correlation=float(state['correlation'][i]),
             **{f'{key}_error_mm': float(value[i]) for key, value in errors.items()},
             **{f'{key}_{axis}_mm': float(value[i, j]) for key, value in predictions.items()
                for j, axis in enumerate('xyz')})
        for i, row in enumerate(rows)
    ])
    comparator = (fit_comparators(state, reg, trans, bem, seed, 4 if smoke else 500, draws,
                                  errors, predictions, output)
                  if seed == DATA_SEEDS[0] else None)
    return dict(strata=strata, paired_pooled_model_seeds=paired,
                alpha_bem=stats(alpha), ecd_subset=comparator)


def run(args):
    require(mne.__version__ == '1.12.1', 'this protocol pins MNE-Python 1.12.1')
    args.output = check_new_output(args.output, ROOT)
    fsaverage = args.subjects_dir / 'fsaverage/bem'
    head_file = fsaverage / 'fsaverage-5120-5120-5120-bem.fif'
    trans_file = fsaverage / 'fsaverage-trans.fif'
    trans = invert_transform(mne.read_trans(trans_file, verbose=False))
    require(trans['from'] == FIFF.FIFFV_COORD_MRI and trans['to'] == FIFF.FIFFV_COORD_HEAD,
            'unexpected fsaverage transform direction')
    reg = Registration.from_recording_electrodes()
    source_files = [Path(__file__), ROOT / 'scripts/run_split_study.py',
                    ROOT / 'scripts/train_baseline.py', ROOT / 'scripts/validate_patch_forward_mne.py',
                    ROOT / 'scripts/_validation_outputs.py', ROOT / 'docs/revision_experiments.md',
                    ROOT / 'validation-requirements.txt', ROOT / 'validation-windows-lock.txt']
    provenance = collect_provenance(ROOT, source_files, sys.argv[1:], require_clean=not args.smoke)
    provenance.update(dict(
        script_sha256={file.relative_to(ROOT).as_posix(): digest(file) for file in source_files},
        template_sha256={file.name: digest(file) for file in (head_file, trans_file)},
        inputs={},
        smoke=args.smoke, gates_only=args.gates_only,
        model_diagnostic_note=(args.model_diagnostic_note.read_text(encoding='utf-8')
                               if args.model_diagnostic_note else None),
    ))
    args.output.mkdir(parents=True)
    (args.output / 'environment.txt').write_text(provenance['resolved_packages'], encoding='utf-8')
    seeds = DATA_SEEDS[:1] if args.smoke else DATA_SEEDS
    cfg = Config.from_json(args.split_run / str(seeds[0]) / 'position/config.json')
    surfaces = pick_bem_surfaces(args.subjects_dir, cfg)
    provenance['bem_conductivities_S_per_m'] = {
        int(surface['id']): float(surface['sigma']) for surface in surfaces
    }
    states = {}
    all_diagnostics_pass = True
    for seed in seeds:
        path = args.split_run / str(seed) / 'position'
        output = args.output / str(seed)
        output.mkdir()
        paths = [path / 'config.json', path / 'dataset/checksums.sha256',
                 *(path / f'mlp_{training_seed}.joblib' for training_seed in MODEL_SEEDS)]
        provenance['inputs'][seed] = {p.relative_to(path).as_posix(): digest(p) for p in paths}
        write_json(args.output / 'provenance.json', provenance)
        state = gate_dataset(path, output, reg, trans, surfaces, args.smoke)
        require(state['cfg'].head.radii_m == cfg.head.radii_m and
                state['cfg'].head.conductivities_S_per_m == cfg.head.conductivities_S_per_m,
                'all transfer datasets must use the declared conductor')
        fitted, passes = model_gate(path, output, state, seed, 50 if args.smoke else 2000)
        states[seed] = (path, output, state, fitted)
        all_diagnostics_pass &= passes
        print(f'{seed}: checks 1–4 pass, check 5 {"passes" if passes else "exceeds limit"}, '
              f'{len(state["rows"])} eligible windows', flush=True)
    if args.gates_only:
        return
    require(all_diagnostics_pass or (provenance['model_diagnostic_note'] or '').strip(),
            'investigate the model diagnostic and document the finding before proceeding under D14')

    bem = mne.make_bem_solution(surfaces, verbose=False)
    summaries = {}
    for seed, (path, output, state, fitted) in states.items():
        summaries[seed] = evaluate(path, output, state, fitted, reg, trans, bem, seed, args.smoke)
        write_json(output / 'summary.json', summaries[seed])
    headline = {}
    for key in ('bem_median_mm', 'package_sphere_median_mm', 'median_difference_mm'):
        values = [result['paired_pooled_model_seeds']['all'][key] for result in summaries.values()]
        headline[key] = dict(mean_of_dataset_seed_medians=float(np.mean(values)),
                             range=[float(np.min(values)), float(np.max(values))])
    write_json(args.output / 'summary.json', dict(
        smoke=args.smoke, check5_all_pass=all_diagnostics_pass, paired_headline=headline,
        by_dataset_seed=summaries,
    ))
    record_output_hashes(args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split-run', required=True, type=Path)
    parser.add_argument('--subjects-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--gates-only', action='store_true')
    parser.add_argument('--smoke', action='store_true', help='use the small seed-862 split run')
    parser.add_argument('--model-diagnostic-note', type=Path,
                        help='document investigation if proceeding after a check-5 discrepancy')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        try:
            run(args)
        except ValueError as error:
            parser.error(str(error))


if __name__ == '__main__':
    main()
