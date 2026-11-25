"""A miniature evidence folder for the table driver.

``build`` writes the eleven experiment folders that ``scripts/make_revision_tables.py`` reads, filled with
seeded random numbers, and returns the values that a reader should find in the tables. Every recorded summary
is computed here with a few lines of NumPy and without the driver, so a test that runs the driver on the folder
compares two independent computations. Only the constants SEEDS, FITS and POLICIES come from the driver.

The folder is small: 72 test windows per dataset and 12 dipoles in the comparisons with MNE-Python. The data
are chosen so that a wrong rule changes a result:

* true source positions lie 30 to 79.5 mm from the origin, and no two errors of a sample are equal;
* the two fitted networks have errors of different size, so the pooled median differs from the median of
  either network and from the mean of the two;
* a position can have two windows, and strata have exactly 29 or exactly 30 distinct positions in a dataset;
* every method and each fitted network has its strata in both panels of the split study and in the patch
  study; a stratum of the dipole-fit subset can hold a single window, lacks the strata without a window of the
  subset, and in one dataset holds two windows of one position;
* each fine-tuned network and each network trained on the boundary-element windows alone has the statistics
  of its own errors, which differ from those of the other network of its arm and from the pooled ones;
* single windows sit on a threshold: a clearance equal to the eligibility threshold, a source on the plane
  y = 0 and one on z = 0, an error equal to the distance to the nearest training location, and a window in
  which both head models give the same prediction;
* among the dipole recoveries with a small final radius there is a run in which a single dipole moves, and
  among the dipole comparators a group whose median error grows with the converged parameter fit.

The experiments share windows as the real ones do. The exact-position test set of a dataset seed is the test
set of the split study, of the count control and of the transfer experiment; its eligible windows are the
windows of the adaptation experiment. The run at the evaluation limit of the sphere-fit record is the Windows
run of the validation rerun. The dipole fits of the split study, of the transfer experiment and of the
adaptation experiment are the fits at the evaluation limit of the record of the dipole comparators, whose
parameter fits are those of the spheres of the sphere-fit record. Every one of these fits has the same number
of windows, and its errors are the distances from the true source positions in the files of its experiment.
The adaptation experiment wrote the eligibility file of the transfer experiment once more: the miniature holds
it for the one dataset whose dipole fits the driver reads, and the archive for every dataset.

File names, column names and keys are those of the archived evidence. A file holds what the driver reads and
a little of the rest; intervals, counts of training data and fit records, which the driver passes through,
are stand-ins. Building the folder takes about a second.
"""

import csv
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from make_revision_tables import FITS, POLICIES, SEEDS

MINIMUM_POSITIONS = 30      # a stratum counts for a dataset with at least this many distinct positions
THRESHOLD_MM = 5.0          # clearance from the inner skull that makes a source eligible in the template head
TEST_WINDOWS = 72           # test windows of every dataset
ECD_WINDOWS = 12            # dipole-fit subset of every dataset of the split study
# The dipole-fit subset of the transfer and adaptation experiments has the same size: the record of the dipole
# comparators, which repeats the fits of all three experiments, names one number of windows for every fit.
DIPOLE_WINDOWS = ECD_WINDOWS
MODEL_SEED = 862            # the network of the count control

# Positions of the test sets: (population, eligible in the template head, positions, positions with two
# windows). Among the eligible positions of the exact-position sets, population 0 has exactly 30 in dataset
# 863 and population 1 exactly 29 in dataset 864, each with more windows than positions.
EXACT_GROUPS = {
    '862': (('0', True, 32, 2), ('0', False, 5, 1), ('1', True, 30, 1), ('1', False, 1, 0)),
    '863': (('0', True, 30, 1), ('0', False, 6, 0), ('1', True, 33, 2), ('1', False, 0, 0)),
    '864': (('0', True, 31, 2), ('0', False, 4, 1), ('1', True, 29, 2), ('1', False, 2, 1))}
REGION_GROUPS = {
    '862': (('0', True, 34, 2), ('1', True, 33, 3)),
    '863': (('0', True, 31, 3), ('1', True, 36, 2)),
    '864': (('0', True, 30, 2), ('1', True, 38, 2))}
# Distinct positions below the montage: 29 and 30 occur under both policies.
BELOW_MONTAGE = {'position': {'862': 30, '863': 34, '864': 29}, 'region': {'862': 29, '863': 31, '864': 30}}

CORTICAL_BANDS = ((70.5, 75.0), (75.0, 79.5))
SUBCORTICAL_BANDS = ((30.0, 45.0), (45.0, 55.0), (55.0, 65.0))
# Strata as the experiment scripts label them: population, lower edge, upper edge (half open).
RADIUS_STRATA = (('1', 30, 45), ('1', 45, 55), ('1', 55, 65.001), ('0', 70.5, 75), ('0', 75, 79.501))
CENSUS_RADIUS = (('radius_30_45', 30, 45), ('radius_45_55', 45, 55), ('radius_55_65', 55, 65.001),
                 ('radius_70.5_75', 70.5, 75), ('radius_75_79.5', 75, 79.501))
POINT_METHODS = ('dictionary', 'knn1', 'knn10', 'training_mean', 'mlp_862', 'mlp_863')
POINT_SCALES = dict(dictionary=1.5, knn1=4.0, knn10=2.6, mlp_862=1.9, mlp_863=2.7)
PATCH_METHODS = ('nearest_training_patch', 'ten_nearest_training_patches', 'training_mean', 'mlp_862', 'mlp_863')
PATCH_SCALES = dict(nearest_training_patch=2.4, ten_nearest_training_patches=1.8, mlp_862=1.6, mlp_863=2.3)
HEADS = ('package_sphere', 'independent_sphere', 'bem')
VARIANTS = ('default', 'three_shell', 'size_matched', 'size_matched_three_shell', 'surface_fitted_three_shell')
SIZES = ('100', '1000', 'all')
ARMS = (*VARIANTS, *[f'{kind}_{size}' for size in SIZES for kind in ('finetuned', 'scratch')])
ARM_SCALES = dict(three_shell=3.4, size_matched=3.0, size_matched_three_shell=2.8, surface_fitted_three_shell=2.6,
                  finetuned_100=2.5, scratch_100=9.0, finetuned_1000=2.0, scratch_1000=4.4, finetuned_all=1.6,
                  scratch_all=2.1)
SPHERE_FITS = ('mne_limit', 'raised_limit')
EVALUATION_LIMITS = dict(mne_limit=1000, raised_limit=100000)


# --------------------------------------------------------------------------- files

def write_json(path, data):
    """Write a record and return its bytes."""
    content = (json.dumps(data, indent=1, allow_nan=False, default=lambda value: value.tolist()) + '\n').encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return content


def write_csv(path, columns, ending='\r\n'):
    """Write equally long columns as a CSV file and return its bytes. A float keeps every bit.

    Most per-window files of the archived evidence end their lines with CR LF, which is the default here.
    """
    def cell(value):
        if isinstance(value, (bool, np.bool_)):
            return str(bool(value))
        return repr(float(value)) if isinstance(value, (float, np.floating)) else str(value)

    if len({len(values) for values in columns.values()}) != 1:
        raise ValueError(f'{path.name}: the columns differ in length')
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator=ending)
    writer.writerow(list(columns))
    writer.writerows(zip(*[[cell(value) for value in values] for values in columns.values()]))
    content = buffer.getvalue().encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return content


def write_csv_lf(path, columns):
    """A CSV file whose lines end with LF, as the count control and the sphere-fit experiment write them."""
    return write_csv(path, columns, ending='\n')


def coordinates(prefix, positions):
    return {f'{prefix}{axis}_mm': positions[:, index] for index, axis in enumerate('xyz')}


def take(table, keep):
    """The chosen windows of every array of a table."""
    return {name: values[keep] for name, values in table.items()}


# --------------------------------------------------------------------------- statistics

def interval(values):
    """A stand-in for a recorded bootstrap interval. The driver passes intervals through."""
    return [float(np.percentile(values, 40)), float(np.percentile(values, 60))]


def around(value):
    """A stand-in for the recorded interval of a difference."""
    return [float(value) - 0.31, float(value) + 0.37]


def error_summary(errors, positions):
    """The recorded summary of a sample of localization errors.

    ``errors`` has one row per fitted network and one column per window; the statistics pool every value,
    and the number of windows is the number of columns.
    """
    errors = np.atleast_2d(errors)
    return dict(windows=int(errors.shape[1]), positions=len(set(np.asarray(positions).tolist())),
                median_mm=float(np.median(errors)), mean_mm=float(np.mean(errors)),
                p90_mm=float(np.percentile(errors, 90)), median_ci95_mm=interval(errors))


def three_statistics(errors):
    """Median, mean and 90th percentile of the errors of one network, under the keys of the records."""
    return dict(median_mm=float(np.median(errors)), mean_mm=float(np.mean(errors)),
                p90_mm=float(np.percentile(errors, 90)))


def stratum_statistics(errors, positions):
    """Windows, distinct positions and the three statistics of the errors of one network in a stratum."""
    return dict(windows=int(errors.size), positions=len(set(np.asarray(positions).tolist())),
                **three_statistics(errors))


def spread(values):
    """Number of values, median, 90th percentile and maximum of a sample."""
    values = np.asarray(values, dtype=float)
    return dict(n=int(values.size), median=float(np.median(values)), p90=float(np.percentile(values, 90)),
                maximum=float(np.max(values)))


def over_seeds(by_seed):
    """Mean and range of one value per dataset seed."""
    values = [float(by_seed[seed]) for seed in SEEDS]
    return float(np.mean(values)), [min(values), max(values)]


def in_band(values, low, high):
    return (values >= low) & (values < high)


# --------------------------------------------------------------------------- sources and predictions

def directions(rng, count):
    """Unit vectors with a uniform distribution of directions."""
    vectors = rng.normal(size=(count, 3))
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def scattered(rng, truth, scale):
    """Positions around ``truth``: a uniform direction and a skewed distance of the given scale in mm."""
    return truth + directions(rng, len(truth)) * rng.gamma(2.0, scale, (len(truth), 1))


def point_test_set(rng, groups, below, nearest_range):
    """The windows of a point-source test set; every returned array has one entry per window.

    The radii cycle through the bands of a population, so that no band is empty. The first position has two
    windows, lies below the montage and has a clearance equal to the eligibility threshold; the second lies on
    the plane y = 0 and the third on z = 0.
    """
    population, eligible, edges, twice = [], [], [], []
    for group_population, group_eligible, positions, two_windows in groups:
        bands = SUBCORTICAL_BANDS if group_population == '1' else CORTICAL_BANDS
        population += [group_population] * positions
        eligible += [group_eligible] * positions
        edges += [bands[index % len(bands)] for index in range(positions)]
        twice += [index < two_windows for index in range(positions)]
    count = len(population)
    population, eligible, edges, twice = (np.array(item) for item in (population, eligible, edges, twice))
    radius = rng.uniform(edges[:, 0], edges[:, 1])
    direction = directions(rng, count)
    direction[1] = (0.6, 0.0, 0.8)
    direction[2] = (-0.8, 0.6, 0.0)
    # Three of five positions get a clearance of 10 to 20 mm, so that this band holds 30 eligible positions.
    wide = np.array([(5.0, 10.0), (10.0, 20.0), (10.0, 20.0), (10.0, 20.0), (20.0, 40.0)])[np.arange(count) % 5]
    clearance = np.where(eligible, rng.uniform(wide[:, 0], wide[:, 1]), rng.uniform(0.5, 4.9, count))
    clearance[0] = THRESHOLD_MM
    # Ineligible cortical sources: every other one lies outside the inner skull, and one lies on it.
    shallow = np.flatnonzero(~eligible & (population == '0'))
    clearance[shallow[::2]] = rng.uniform(-6.0, -0.5, len(shallow[::2]))
    clearance[shallow[1:2]] = 0.0
    flagged = np.zeros(count, dtype=int)
    flagged[np.r_[0, 1 + rng.choice(count - 1, below - 1, replace=False)]] = 1
    nearest = rng.uniform(*nearest_range, count)
    identity = 100 + rng.choice(9000, count, replace=False)
    order = rng.permutation(np.r_[np.arange(count), np.flatnonzero(twice)])     # the position of every window
    truth = (radius[:, None] * direction)[order]
    # Three of five windows have an SNR of 15 to 20 dB, so that this band holds 30 positions as well.
    levels = np.array([(15.0, 20.0), (15.0, 20.0), (10.0, 15.0), (15.0, 20.0), (20.0, 25.0)])[np.arange(len(order)) % 5]
    return dict(position=identity[order], population=population[order], truth=truth,
                radius=np.linalg.norm(truth, axis=1), snr=rng.uniform(levels[:, 0], levels[:, 1]),
                nearest=nearest[order], below=flagged[order], clearance=clearance[order], eligible=eligible[order])


def point_predictions(rng, truth, factor):
    """Predicted positions of every method. The training mean is one position for all windows."""
    return {name: scattered(rng, truth, factor * POINT_SCALES[name]) if name in POINT_SCALES
            else np.tile(rng.normal(0.0, 3.0 * factor, 3), (len(truth), 1)) for name in POINT_METHODS}


def distances(predictions, truth):
    """Localization errors: one row per fitted network and one column per window.

    The entry ``mlp_pooled`` holds the rows of both networks.
    """
    errors = {name: np.linalg.norm(values - truth, axis=1)[None, :] for name, values in predictions.items()}
    errors['mlp_pooled'] = np.concatenate([errors[f'mlp_{fit}'] for fit in FITS])
    return errors


def touch_the_floor(windows, predictions):
    """Put the nearest-neighbour prediction of four windows at the nearest training location.

    An error counts as below the recorded distance to that location if it is smaller by more than 1e-8 mm.
    The error of the first window equals the distance, that of the second is smaller by 5e-9 mm and that of
    the third by the margin itself: none of them counts. The error of the fourth is smaller by 5e-8 mm, which
    counts.
    """
    single = [window for window in range(len(windows['position']))
              if np.sum(windows['position'] == windows['position'][window]) == 1]
    for window, gap in zip(single[:4], (0.0, 5e-9, 1e-8, 5e-8)):
        truth = windows['truth'][window]
        predictions['knn1'][window] = truth + windows['nearest'][window] * np.array([0.48, -0.6, 0.64])
        error = np.linalg.norm(predictions['knn1'][window] - truth)
        nearest = error + gap
        if gap == 1e-8:
            # Move the distance by single steps of the floating-point grid until it less the margin is the error.
            while nearest - gap < error:
                nearest = np.nextafter(nearest, np.inf)
            while nearest - gap > error:
                nearest = np.nextafter(nearest, -np.inf)
        windows['nearest'][window] = nearest


def split_strata(windows):
    """Masks of the strata of a point-source test set, by the labels of the split study. Empty strata are left out."""
    population, radius = windows['population'], windows['radius']
    strata = {f'radius_{group}_{low:g}_{high:g}': (population == group) & in_band(radius, low, high)
              for group, low, high in RADIUS_STRATA}
    strata.update({f'population_{group}': population == group for group in '01'})
    strata.update({f'snr_{low:g}_{high:g}': in_band(windows['snr'], low, high)
                   for low, high in ((10, 15), (15, 20), (20, 25.001))})
    strata.update({f'distance_{low:g}_{high:g}': in_band(windows['nearest'], low, high)
                   for low, high in ((0, 5), (5, 10), (10, 20), (20, 30), (30, float('inf')))})
    strata.update(below_montage=windows['below'] == 1, within_montage=windows['below'] == 0)
    return {label: mask for label, mask in strata.items() if mask.any()}


def population_strata(windows):
    """Masks of the strata of the transfer and adaptation summaries."""
    population, radius = windows['population'], windows['radius']
    strata = dict(all=np.ones(len(radius), dtype=bool), population_0=population == '0', population_1=population == '1')
    strata.update({f'radius_{group}_{low:g}_{high:g}': (population == group) & in_band(radius, low, high)
                   for group, low, high in RADIUS_STRATA})
    return strata


def prediction_columns(windows, predictions):
    """The columns of a prediction file of the split study."""
    columns = {'window': np.arange(len(windows['position'])), 'position_id': windows['position'],
               'population': windows['population'], 'snr_db': windows['snr'], 'nearest_train_mm': windows['nearest'],
               'below_montage': windows['below'], **coordinates('', windows['truth'])}
    for name, values in predictions.items():
        columns.update(coordinates(f'{name}_', values))
    return columns


def dataset_config(seed, policy, mode):
    """The settings of a generated dataset that the tables print."""
    region = policy == 'region'
    point = mode == 'point'
    return dict(
        master_seed=int(seed), source_mode=mode, electrode_file='standard_1010_69.csv',
        dataset=dict(n_train=480, n_val=48, n_test=TEST_WINDOWS), head=dict(n_legendre=60),
        signal=dict(window_samples=100, sampling_freq_hz=500.0),
        noise=dict(noise_model='snr_target', snr_db_range=[10.0, 25.0]),
        output=dict(reference_mode='average', eeg_normalization='per_sample_rms'),
        sources=dict(n_cortical_positions=800 if point else 1160 if region else 3500,
                     n_subcortical_positions=150 if point else 650),
        split=dict(policy=policy, region_count=(24 if point else 8) if region else None,
                   buffer_mm=(15.0 if point else 12.0) if region else None,
                   region_rotation_seed=(12000 if point else 24000) + SEEDS.index(seed) if region else None),
        patch=dict(diameter_range_mm=[10.0, 60.0]))


def neighbour_distances(rng, low, high):
    """A stand-in for the generator's report of the distances between the positions of two splits."""
    def one():
        values = np.sort(rng.uniform(low, high, 6))
        return dict(min=values[0], p10=values[1], median=values[2], p90=values[4], max=values[5],
                    n=int(rng.integers(100, 200)))

    return {pair: {group: one() for group in ('all', 'cortical', 'subcortical')}
            for pair in ('test_to_train', 'test_to_val', 'val_to_train')}


# --------------------------------------------------------------------------- split study and count control

def panel_summary(windows, errors, comparators):
    """One panel of a dataset summary: every method with its strata, and the paired contrasts.

    Every method and each fitted network has the strata of the panel, under the same labels in the same
    order: the strata that hold a window of the panel.
    """
    strata = split_strata(windows)
    panel = {}
    for method, values in errors.items():
        panel[method] = error_summary(values, windows['position'])
        panel[method]['strata'] = {label: error_summary(values[:, mask], windows['position'][mask])
                                   for label, mask in strata.items()}
    panel['paired_mlp_minus_baseline'] = {}
    for method in comparators:
        difference = float(np.median(errors['mlp_pooled']) - np.median(errors[method]))
        panel['paired_mlp_minus_baseline'][method] = dict(median_difference_mm=difference, ci95_mm=around(difference))
    return panel


def split_study(base, rng, expected):
    """split-study-v11: both split policies for three dataset seeds, and the two regimes on the same windows."""
    sets, predictions_of, summaries, common, dipole_fits = {}, {}, {}, {}, {}
    medians = {policy: dict(full={}, ecd_subset={}) for policy in POLICIES}
    out = dict(training_positions={}, below_floor={}, mean_offset={policy: {} for policy in POLICIES}, per_fit={},
               per_fit_strata={}, strata_sizes={}, distance={})
    for seed in SEEDS:
        summaries[seed] = {}
        for policy in POLICIES:
            exact = policy == 'position'
            windows = point_test_set(rng, (EXACT_GROUPS if exact else REGION_GROUPS)[seed], BELOW_MONTAGE[policy][seed],
                                     (0.6, 9.5) if exact else (10.5, 42.0))
            predictions = point_predictions(rng, windows['truth'], 1.0 if exact else 2.4)
            if exact and seed == SEEDS[0]:
                # Only this dataset has exact-position test windows farther than 10 mm from the training data.
                windows['nearest'][windows['position'] == windows['position'][0]] = 11.7
                touch_the_floor(windows, predictions)
            errors = distances(predictions, windows['truth'])
            keep = np.sort(rng.choice(TEST_WINDOWS, ECD_WINDOWS, replace=False))
            subset = take(windows, keep)
            subset_predictions = dict(take(predictions, keep), dipole=scattered(rng, subset['truth'], 0.7))
            subset_errors = distances(subset_predictions, subset['truth'])
            for panel in (errors, subset_errors):       # the pooled networks come last, as in the records
                panel['mlp_pooled'] = panel.pop('mlp_pooled')
            composition = {split: [dict(population=int(group), radius_mm=[low, high], distinct_positions=int(positions),
                                        windows=int(3 * positions + rng.integers(0, 9)))
                                   for (group, low, high), positions in zip(RADIUS_STRATA, rng.integers(20, 90, 5))]
                           for split in ('train', 'validation', 'test')}
            summary = dict(
                full=panel_summary(windows, errors, ('dictionary', 'knn1', 'knn10')),
                fits={f'mlp_{fit}': dict(best_epoch=int(epoch), epochs_run=int(epoch) + 15)
                      for fit, epoch in zip(FITS, rng.integers(20, 90, len(FITS)))},
                observed_split_composition=composition,
                ecd_subset=panel_summary(subset, subset_errors, ('dictionary', 'knn1', 'knn10', 'dipole')))
            report = dict(source_split=dict(
                policy=policy,
                eligible_position_nearest_neighbour_mm=neighbour_distances(rng, 1.0, 9.0),
                serialized_used_position_nearest_neighbour_mm=neighbour_distances(rng, 1.0, 9.0)))
            folder = base / seed / policy
            write_json(folder / 'summary.json', summary)
            write_csv(folder / 'predictions.csv', prediction_columns(windows, predictions))
            write_csv(folder / 'ecd_subset/predictions.csv', prediction_columns(subset, subset_predictions))
            write_json(folder / 'ecd_subset/test_rows.json', keep)      # the test windows of the subset
            write_json(folder / 'dataset/validation_report.json', report)
            write_json(folder / 'dataset/metadata.json', dict(config=dataset_config(seed, policy, 'point')))
            sets[seed, policy], predictions_of[seed, policy], summaries[seed][policy] = windows, predictions, summary
            dipole_fits[seed, policy] = dict(windows=keep, truth=subset['truth'], fitted=subset_predictions['dipole'])
            for panel, panel_errors in (('full', errors), ('ecd_subset', subset_errors)):
                for method, values in panel_errors.items():
                    medians[policy][panel].setdefault(method, {})[seed] = float(np.median(values))
            out['training_positions'][seed, policy] = sum(cell['distinct_positions'] for cell in composition['train'])
            out['below_floor'][seed, policy] = {
                method: float(np.mean(errors[method] < windows['nearest'] - 1e-8))
                for method in ('mlp_pooled', 'dictionary', 'knn1', 'knn10', 'training_mean')}
            out['mean_offset'][policy][seed] = float(np.linalg.norm(predictions['training_mean'][0]))
            out['per_fit'][seed, policy] = {fit: (summary['fits'][f'mlp_{fit}']['best_epoch'],
                                                  float(np.median(errors[f'mlp_{fit}']))) for fit in FITS}
            # The strata of each fitted network on all windows, which the tables hold next to its record, and
            # the windows and positions of the strata of each panel: a stratum without a window is left out.
            out['per_fit_strata'][seed, policy] = {
                fit: {label: stratum_statistics(errors[f'mlp_{fit}'][0, mask], windows['position'][mask])
                      for label, mask in split_strata(windows).items()} for fit in FITS}
            out['strata_sizes'][seed, policy] = {
                panel: {label: (int(mask.sum()), len(set(members['position'][mask].tolist())))
                        for label, mask in split_strata(members).items()}
                for panel, members in (('full', windows), ('ecd_subset', subset))}
            out['distance'][seed, policy] = tuple(
                float(report['source_split'][f'{kind}_position_nearest_neighbour_mm']['test_to_train']['all']['min'])
                for kind in ('eligible', 'serialized_used'))
    out['median_test_radius'] = float(np.median(sets[SEEDS[0], 'position']['radius']))
    first = sets[SEEDS[0], 'position']
    nearest_neighbour = np.linalg.norm(predictions_of[SEEDS[0], 'position']['knn1'] - first['truth'], axis=1)
    out['floor_cases'] = [int(np.sum(nearest_neighbour == first['nearest'] - gap)) for gap in (0.0, 1e-8)]

    # The study-level headline: one median per dataset, their mean and their range.
    headline = {policy: {panel: {} for panel in ('full', 'ecd_subset')} for policy in POLICIES}
    for policy in POLICIES:
        for panel, by_method in medians[policy].items():
            for method, by_seed in by_method.items():
                mean, limits = over_seeds(by_seed)
                headline[policy][panel][method] = dict(dataset_seed_medians_mm=by_seed, mean_median_mm=mean,
                                                       range_mm=limits)
    # A record may list the datasets in another order than that of their seeds.
    entry = headline['region']['full']['knn1']
    entry['dataset_seed_medians_mm'] = {seed: entry['dataset_seed_medians_mm'][seed] for seed in ('864', '862', '863')}
    out['headline'] = headline

    # Network and dictionary by stratum: the mean over the datasets with at least 30 positions in the stratum.
    def strata_of(seed, policy, method):
        return summaries[seed][policy]['full'][method]['strata']

    labels = list(dict.fromkeys(label for policy in POLICIES for seed in SEEDS
                                for label in strata_of(seed, policy, 'mlp_pooled')))
    out['strata'] = []
    for label in labels:
        row = dict(stratum=label)
        for policy in POLICIES:
            seeds = [seed for seed in SEEDS if label in strata_of(seed, policy, 'mlp_pooled')
                     and strata_of(seed, policy, 'mlp_pooled')[label]['positions'] >= MINIMUM_POSITIONS]
            recorded = {seed: strata_of(seed, policy, 'mlp_pooled').get(label, {}) for seed in SEEDS}
            row[policy] = dict(
                seeds=seeds, positions={seed: recorded[seed].get('positions') for seed in SEEDS},
                windows={seed: recorded[seed].get('windows') for seed in SEEDS},
                network=float(np.mean([recorded[seed]['median_mm'] for seed in seeds])) if seeds else None,
                dictionary=float(np.mean([strata_of(seed, policy, 'dictionary')[label]['median_mm'] for seed in seeds]))
                if seeds else None)
        out['strata'].append(row)

    # The two training regimes on the exact-position test windows that lie in regional test sectors.
    out['paired'], pooled, cortical, clusters = {}, {policy: [] for policy in POLICIES}, [], []
    for seed, size in zip(SEEDS, (16, 14, 15)):
        windows = sets[seed, 'position']
        # Both windows of the position at the eligibility threshold are common: more windows than positions.
        both = np.flatnonzero(windows['clearance'] == THRESHOLD_MM)
        others = np.setdiff1d(np.arange(TEST_WINDOWS), both)
        keep = np.sort(np.r_[both, rng.choice(others, size - len(both), replace=False)])
        shared = take(windows, keep)
        regional = dict(shared, nearest=rng.uniform(10.5, 42.0, size))
        regional_predictions = {name: values for name, values in point_predictions(rng, shared['truth'], 2.2).items()
                                if name != 'training_mean'}
        by_policy = dict(
            position=distances(take(predictions_of[seed, 'position'], keep), shared['truth'])['mlp_pooled'],
            region=distances(regional_predictions, shared['truth'])['mlp_pooled'])
        contrast = float(np.median(by_policy['region']) - np.median(by_policy['position']))
        record = dict(region_minus_position_mm=contrast, ci95_mm=around(contrast), windows=size,
                      distinct_positions=len(set(shared['position'].tolist())))
        folder = base / seed / 'same_test_windows'
        write_json(folder / 'paired_mlp.json', record)
        write_json(folder / 'exact_position_test_rows.json', keep)
        write_csv(folder / 'position/predictions.csv',
                  prediction_columns(shared, take(predictions_of[seed, 'position'], keep)))
        write_csv(folder / 'region/predictions.csv', prediction_columns(regional, regional_predictions))
        for policy in POLICIES:
            pooled[policy].append(by_policy[policy])
        cortical.append(shared['population'] == '0')
        clusters.append(shared['position'])
        common[seed] = dict(keep=keep, predictions=regional_predictions)
        out['paired'][seed] = record

    def pooled_contrast(masks):
        """The contrast over the windows of the three datasets together, for the chosen windows of each."""
        both = {policy: np.concatenate([values[:, mask] for values, mask in zip(pooled[policy], masks)], axis=1)
                for policy in POLICIES}
        contrast = float(np.median(both['region']) - np.median(both['position']))
        return dict(windows=int(sum(mask.sum() for mask in masks)),
                    seed_position_clusters=sum(len(set(ids[mask].tolist())) for ids, mask in zip(clusters, masks)),
                    region_minus_position_mm=contrast, conditional_ci95_mm=around(contrast))

    out['paired']['pooled'] = pooled_contrast([np.ones(len(mask), dtype=bool) for mask in cortical])
    write_json(base / 'paired_split_summary.json', dict(
        weighting='equal weight per window and fit, with fixed dataset seed groups',
        strata=dict(all=out['paired']['pooled'], population_0=pooled_contrast(cortical),
                    population_1=pooled_contrast([~mask for mask in cortical]))))
    write_json(base / 'headline.json', headline)
    write_json(base / 'summary.json', summaries)
    write_json(base / 'region_adequacy_preflight.json', {seed: dict(checks=dict(C1=True, C2=True)) for seed in SEEDS})
    write_json(base / 'provenance.json', dict(commit='1' * 40, arguments=['--output-dir', 'split-study-v11']))
    expected['split'] = out
    return dict(sets=sets, predictions=predictions_of, common=common, dipole_fits=dipole_fits)


def count_control(base, rng, study, expected):
    """training-count-control-v11: one network per training regime on the exact-position test windows."""
    values = {'all_test_windows': {}, 'same_test_windows': {}}
    for seed in SEEDS:
        windows, common = study['sets'][seed, 'position'], study['common'][seed]
        truth, keep = windows['truth'], common['keep']
        exact = np.linalg.norm(study['predictions'][seed, 'position'][f'mlp_{MODEL_SEED}'] - truth, axis=1)
        matched = np.linalg.norm(scattered(rng, truth, 2.3) - truth, axis=1)
        regional = np.linalg.norm(common['predictions'][f'mlp_{MODEL_SEED}'] - truth[keep], axis=1)
        samples = {'all_test_windows': (windows['position'], dict(exact=exact, count_matched=matched)),
                   'same_test_windows': (windows['position'][keep],
                                         dict(exact=exact[keep], count_matched=matched[keep], regional=regional))}
        summary = dict(model_seed=MODEL_SEED,
                       training_positions=dict(exact=int(rng.integers(300, 400)), count_matched=260, regional=260),
                       fit=dict(best_epoch=int(rng.integers(20, 90)), epochs_run=95))
        for panel, (positions, by_name) in samples.items():
            summary[panel] = {name: error_summary(sample, positions) for name, sample in by_name.items()}
            contrasts = [('count_matched_minus_exact', 'count_matched', 'exact')]
            if panel == 'same_test_windows':
                contrasts.append(('regional_minus_count_matched', 'regional', 'count_matched'))
            for name, first, second in contrasts:
                difference = float(np.median(by_name[first]) - np.median(by_name[second]))
                summary[panel][name] = dict(median_difference_mm=difference, ci95_mm=around(difference))
            for name, entry in summary[panel].items():
                values[panel].setdefault(name, {})[seed] = entry.get('median_mm', entry.get('median_difference_mm'))
        write_json(base / seed / 'summary.json', summary)
        write_csv_lf(base / seed / 'errors.csv', dict(window=np.arange(TEST_WINDOWS), position_id=windows['position'],
                                                      exact_error_mm=exact, count_matched_error_mm=matched))
    headline = {}
    for panel, by_name in values.items():
        headline[panel] = {}
        for name, by_seed in by_name.items():
            mean, limits = over_seeds(by_seed)
            headline[panel][name] = dict(
                statistic='median_difference_mm' if '_minus_' in name else 'median_mm',
                dataset_seed_values=[by_seed[seed] for seed in SEEDS], mean=mean, range=limits)
    write_json(base / 'headline.json', headline)
    write_json(base / 'provenance.json', dict(commit='2' * 40, package_tree='a' * 40,
                                              arguments=['--output-dir', 'training-count-control-v11']))
    expected['count_control'] = headline


# --------------------------------------------------------------------------- patch study

def patch_study(base, rng, expected):
    """patch-split-study-v11: extended sources under both split policies."""
    summaries, medians = {}, {policy: {} for policy in POLICIES}
    out = dict(audits={}, below_centre={policy: {} for policy in POLICIES})
    bands = ((10.0, 25.0), (25.0, 40.0), (40.0, 60.0))
    for seed in SEEDS:
        for policy in POLICIES:
            exact = policy == 'position'
            edges = np.array([bands[index % 3] for index in range(TEST_WINDOWS)])
            diameter, snr = rng.uniform(edges[:, 0], edges[:, 1]), rng.uniform(10.0, 25.0, TEST_WINDOWS)
            truth = rng.uniform(70.5, 79.5, (TEST_WINDOWS, 1)) * directions(rng, TEST_WINDOWS)
            factor = 1.0 if exact else 9.0
            predictions = {name: scattered(rng, truth, factor * PATCH_SCALES[name]) if name in PATCH_SCALES
                           else np.tile(rng.normal(0.0, 3.0 * factor, 3), (TEST_WINDOWS, 1)) for name in PATCH_METHODS}
            errors = distances(predictions, truth)
            nearest = rng.uniform(1.2, 6.0, TEST_WINDOWS) if exact else rng.uniform(20.0, 60.0, TEST_WINDOWS)
            nearest[5] = errors['mlp_862'][0, 5]        # an error equal to the distance is not below it
            centres = 100 + rng.choice(9000, TEST_WINDOWS, replace=False)
            strata = {f'diameter_{low:g}_{high:g}': in_band(diameter, low, high)
                      for low, high in ((10, 25), (25, 40), (40, 60.001))}
            strata.update({f'snr_{low:g}_{high:g}': in_band(snr, low, high)
                           for low, high in ((10, 15), (15, 20), (20, 25.001))})
            # Every method and each fitted network, over all windows and by stratum.
            methods = {name: dict(error_summary(values, centres),
                                  strata={label: error_summary(values[:, mask], centres[mask])
                                          for label, mask in strata.items()}) for name, values in errors.items()}
            methods['paired_mlp_minus_lookup'] = {}
            for name in ('nearest_training_patch', 'ten_nearest_training_patches'):
                difference = float(np.median(errors['mlp_pooled']) - np.median(errors[name]))
                methods['paired_mlp_minus_lookup'][name] = dict(median_difference_mm=difference,
                                                                ci95_mm=around(difference))
            # The distances between used positions of different splits: the smallest is not in the first pair.
            used = neighbour_distances(rng, 1.4, 6.0) if exact else neighbour_distances(rng, 12.6, 40.0)
            used['val_to_train']['all']['min'] = float(rng.uniform(1.2, 1.3) if exact else rng.uniform(12.0, 12.5))
            smallest = min(pair['all']['min'] for pair in used.values())
            overlap = np.sort(rng.uniform(0.5, 0.95, 6)) if exact else np.zeros(6)
            pair = dict(query_patches=TEST_WINDOWS, scored_patches=TEST_WINDOWS - (0 if exact else 2),
                        minimum_footprint_gap_mm=None if exact else float(rng.uniform(8.0, 12.0)),
                        best_iou=dict(min=overlap[0], p10=overlap[1], median=overlap[2], p90=overlap[4], max=overlap[5],
                                      n=TEST_WINDOWS),
                        nearest_centre_euclidean_mm=dict(min=float(nearest.min()), median=float(np.median(nearest)),
                                                         max=float(nearest.max()), n=TEST_WINDOWS))
            audit = dict(pairs=dict(test_to_train=pair))
            if not exact:
                # The study computed its own clearance: one record differs from the dataset report by rounding.
                audit.update(minimum_support_clearance_mm=smallest + (3e-7 if seed == SEEDS[0] else 0.0),
                             patch_containment=dict(test=dict(clipped_patches=0, patches=TEST_WINDOWS)))
            summary = dict(
                methods=methods,
                fits={f'mlp_{fit}': dict(best_epoch=int(epoch), epochs_run=int(epoch) + 15)
                      for fit, epoch in zip(FITS, rng.integers(5, 60, len(FITS)))},
                audit=audit, distinct_training_centres=int(rng.integers(300, 480)),
                mlp_error_below_nearest_training_centre=float(np.mean(errors['mlp_pooled'] < nearest)))
            folder = base / seed / policy
            write_json(folder / 'summary.json', summary)
            write_csv(folder / 'predictions.csv', {
                'window': np.arange(TEST_WINDOWS), 'centre_position_id': centres, 'diameter_mm': diameter,
                'snr_db': snr, 'nearest_training_centre_mm': nearest, **coordinates('', truth),
                **{key: values for name, positions in predictions.items()
                   for key, values in coordinates(f'{name}_', positions).items()}})
            write_json(folder / 'dataset/validation_report.json',
                       dict(source_split=dict(policy=policy, serialized_used_position_nearest_neighbour_mm=used)))
            write_json(folder / 'dataset/metadata.json', dict(config=dataset_config(seed, policy, 'patch')))
            summaries[seed, policy] = summary
            for name, values in errors.items():
                medians[policy].setdefault(name, {})[seed] = float(np.median(values))
            out['audits'][seed, policy] = dict(
                overlap=float(overlap[2]), gap=pair['minimum_footprint_gap_mm'], used_distance=float(smallest),
                centres=summary['distinct_training_centres'], scored=pair['scored_patches'],
                clearance=audit.get('minimum_support_clearance_mm'))
            out['below_centre'][policy][seed] = summary['mlp_error_below_nearest_training_centre']
    headline = {policy: {} for policy in POLICIES}
    out['strata'] = {}
    for policy in POLICIES:
        for name, by_seed in medians[policy].items():
            mean, limits = over_seeds(by_seed)
            headline[policy][name] = dict(dataset_seed_medians_mm=by_seed, mean_median_mm=mean, range_mm=limits)
        overlaps = {seed: out['audits'][seed, policy]['overlap'] for seed in SEEDS}
        mean, limits = over_seeds(overlaps)
        headline[policy]['median_best_test_to_train_overlap'] = dict(
            dataset_seed_values=[overlaps[seed] for seed in SEEDS], mean=mean, range=limits)
        for label in summaries[SEEDS[0], policy]['methods']['mlp_pooled']['strata']:
            by_seed = {seed: summaries[seed, policy]['methods']['mlp_pooled']['strata'][label]['median_mm']
                       for seed in SEEDS}
            out['strata'].setdefault(label, {})[policy] = (by_seed, *over_seeds(by_seed))
        out['below_centre'][policy] = (out['below_centre'][policy], *over_seeds(out['below_centre'][policy]))
    out['headline'] = headline
    write_json(base / 'headline.json', headline)
    write_json(base / 'region_adequacy_preflight.json',
               {seed: dict(checks=dict(largest_patch_fits_at_enough_centres=True)) for seed in SEEDS})
    write_json(base / 'provenance.json', dict(commit='3' * 40, arguments=['--output-dir', 'patch-split-study-v11']))
    expected['patch'] = out


# --------------------------------------------------------------------------- transfer to the template head

def census_masks(windows):
    """Strata of the eligibility census. Anterior is y >= 0 and superior is z >= 0 in the package frame."""
    cortical = windows['population'] == '0'
    y, z = windows['truth'][:, 1], windows['truth'][:, 2]
    masks = dict(all=np.ones(len(cortical), dtype=bool), cortical=cortical, subcortical=windows['population'] == '1')
    masks.update({name: in_band(windows['radius'], low, high) for name, low, high in CENSUS_RADIUS})
    for vertical, superior in (('superior', z >= 0), ('inferior', z < 0)):
        for horizontal, anterior in (('posterior', y < 0), ('anterior', y >= 0)):
            masks[f'cortical_{vertical}_{horizontal}'] = cortical & superior & anterior
    return masks


def eligibility_columns(windows):
    """The columns of an eligibility file: every test window with its true source position and its clearance."""
    return {'window_id': np.arange(TEST_WINDOWS), 'position_pool_id': windows['position'],
            'population': windows['population'], **coordinates('', windows['truth']),
            'inner_skull_clearance_mm': windows['clearance'], 'eligible': windows['eligible']}


def window_strata_masks(windows):
    """Strata of the transfer errors: population, radius, SNR and clearance from the inner skull."""
    masks = dict(population_0=windows['population'] == '0', population_1=windows['population'] == '1')
    masks.update({name: in_band(windows['radius'], low, high) for name, low, high in CENSUS_RADIUS})
    masks.update(snr_10_15=in_band(windows['snr'], 10, 15), snr_15_20=in_band(windows['snr'], 15, 20),
                 snr_20_25=in_band(windows['snr'], 20, 25.001),
                 clearance_5_10=in_band(windows['clearance'], 5, 10),
                 clearance_10_20=in_band(windows['clearance'], 10, 20), clearance_20_inf=windows['clearance'] >= 20)
    return masks


def head_transfer(base, rng, study, expected):
    """head-transfer-v11: the networks of the exact-position split on windows of a template head."""
    by_seed, records, paired_of, heads_of = {}, {}, {}, {}
    out = dict(independent={}, directional={}, window_statistics={}, census_counts={}, outside={})
    for seed in SEEDS:
        windows, predictions = study['sets'][seed, 'position'], study['predictions'][seed, 'position']
        keep = np.flatnonzero(windows['eligible'])
        paired = dict(take(windows, keep), window=keep)
        truth, count = paired['truth'], len(keep)
        heads = {'constant': predictions['training_mean'][keep]}
        for fit, scale in zip(FITS, (3.0, 3.9)):
            sphere = predictions[f'mlp_{fit}'][keep]
            heads[f'mlp_{fit}_package_sphere'] = sphere
            heads[f'mlp_{fit}_independent_sphere'] = sphere + directions(rng, count) * rng.gamma(2.0, 0.05, (count, 1))
            heads[f'mlp_{fit}_bem'] = scattered(rng, truth, scale)
        # In one window the first network predicts the same position with both head models.
        heads[f'mlp_{FITS[0]}_bem'][7] = heads[f'mlp_{FITS[0]}_package_sphere'][7]
        errors = {name: np.linalg.norm(values - truth, axis=1) for name, values in heads.items()}
        pooled = {head: np.stack([errors[f'mlp_{fit}_{head}'] for fit in FITS]) for head in HEADS}

        record = dict(strata={}, paired_pooled_model_seeds={}, ecd_subset=None,
                      alpha_bem=dict(median=1.01, p90=1.18, p99=1.4, maximum=1.7))
        for label, mask in population_strata(paired).items():
            if label in ('all', 'population_0', 'radius_1_45_55'):      # every method, for one stratum of each kind
                record['strata'][label] = {name: error_summary(values[mask], paired['position'][mask])
                                           for name, values in errors.items()}
            sphere, bem = (float(np.median(pooled[head][:, mask])) for head in ('package_sphere', 'bem'))
            record['paired_pooled_model_seeds'][label] = dict(
                position_clusters=len(set(paired['position'][mask].tolist())), windows=int(mask.sum()),
                bem_median_mm=bem, package_sphere_median_mm=sphere, median_difference_mm=bem - sphere,
                median_difference_ci95_mm=around(bem - sphere))
        columns = {
            'window_id': keep, 'position_pool_id': paired['position'], 'population': paired['population'],
            'source_radius_mm': paired['radius'], 'snr_db_referenced': paired['snr'],
            'alpha_bem': rng.uniform(0.85, 1.4, count), 'mne_sphere_relative_residual': rng.uniform(1e-3, 4e-3, count),
            **{f'{name}_error_mm': values for name, values in errors.items()},
            **{key: values for name, positions in heads.items()
               for key, values in coordinates(f'{name}_', positions).items()}}
        if seed == SEEDS[0]:
            # Dipole fits were made for the first dataset only, on a subset of its eligible windows.
            chosen = np.sort(rng.choice(count, DIPOLE_WINDOWS, replace=False))
            fits = dict(ecd_package_sphere_matched=scattered(rng, truth[chosen], 0.9),
                        ecd_bem_matched=scattered(rng, truth[chosen], 1.0),
                        ecd_bem_fitted_sphere=scattered(rng, truth[chosen], 4.5))
            subset_errors = {name: np.linalg.norm(values - truth[chosen], axis=1) for name, values in fits.items()}
            subset_errors.update({name: values[chosen] for name, values in errors.items()})
            record['ecd_subset'] = {name: error_summary(values, paired['position'][chosen])
                                    for name, values in subset_errors.items()}
            difference = float(np.median(subset_errors['ecd_bem_matched']) -
                               np.median(subset_errors['ecd_package_sphere_matched']))
            record['ecd_subset']['ecd_matched_bem_minus_sphere'] = dict(median_difference_mm=difference,
                                                                        median_difference_ci95_mm=around(difference))
            write_csv(base / seed / 'ecd_subset.csv', {
                'window_id': keep[chosen], 'position_pool_id': paired['position'][chosen],
                **{f'{name}_error_mm': values for name, values in subset_errors.items()},
                **{key: values for name, positions in fits.items()
                   for key, values in coordinates(f'{name}_', positions).items()}})
            out['dipole_subset'] = dict(chosen=chosen, errors=subset_errors, difference=difference)
            dipole_subset = dict(windows=keep[chosen], truth=truth[chosen], fitted=fits['ecd_package_sphere_matched'])

        # The pairing with the template head: each network on the package sphere and on the independent one.
        gates = dict(scope='recorded protocol', check4_pass=True, check5_model={})
        for fit in FITS:
            everywhere = np.linalg.norm(predictions[f'mlp_{fit}'] - windows['truth'], axis=1)
            shift = np.linalg.norm(heads[f'mlp_{fit}_independent_sphere'] - heads[f'mlp_{fit}_package_sphere'], axis=1)
            gates['check5_model'][f'mlp_{fit}'] = dict(
                package_sphere_all=error_summary(everywhere, windows['position']),
                package_sphere_eligible=error_summary(errors[f'mlp_{fit}_package_sphere'], paired['position']),
                independent_sphere_eligible=error_summary(errors[f'mlp_{fit}_independent_sphere'], paired['position']),
                median_difference_mm=float(np.median(errors[f'mlp_{fit}_independent_sphere']) -
                                           np.median(errors[f'mlp_{fit}_package_sphere'])),
                per_window_prediction_shift_mm=dict(median=float(np.median(shift)), p90=float(np.percentile(shift, 90)),
                                                    p99=float(np.percentile(shift, 99)), maximum=float(shift.max())),
                passes=True)
        # Populations, radius bands, and the halves of the head (a coordinate of zero or more, or below zero).
        strata = dict(population_strata(windows))
        for axis, name in enumerate(('right', 'anterior', 'superior')):
            strata[name] = windows['truth'][:, axis] >= 0
            strata[f'negative_{name}'] = windows['truth'][:, axis] < 0
        eligibility = dict(minimum_inner_skull_clearance_mm=THRESHOLD_MM, strata={
            label: dict(windows=int(mask.sum()), eligible_windows=int((mask & windows['eligible']).sum()),
                        positions=len(set(windows['position'][mask].tolist())),
                        eligible_positions=len(set(windows['position'][mask & windows['eligible']].tolist())))
            for label, mask in strata.items()})
        write_csv(base / seed / 'eligibility.csv', eligibility_columns(windows))
        write_csv(base / seed / 'paired_predictions.csv', columns)
        write_json(base / seed / 'eligibility.json', eligibility)
        write_json(base / seed / 'adapter_gates.json', gates)
        by_seed[seed], records[seed], paired_of[seed], heads_of[seed] = record, (pooled, errors), paired, heads

        # Values that the driver computes without a record to compare with.
        out['independent'][seed] = float(np.median(pooled['independent_sphere']))
        displacement = np.stack([np.linalg.norm(heads[f'mlp_{fit}_bem'] - heads[f'mlp_{fit}_package_sphere'], axis=1)
                                 for fit in FITS])
        out['window_statistics'][seed] = dict(
            windows=count, larger=float(np.mean(pooled['bem'] > pooled['package_sphere'])),
            displacement=float(np.median(displacement)),
            difference=float(np.median(pooled['bem'] - pooled['package_sphere'])))
        out['census_counts'][seed] = {name: (int((mask & windows['eligible']).sum()), int(mask.sum()))
                                      for name, mask in census_masks(windows).items()}
        cortical = windows['population'] == '0'
        out['outside'][seed] = (int((cortical & (windows['clearance'] < 0)).sum()), int(cortical.sum()))
        y, z = windows['truth'][:, 1], windows['truth'][:, 2]
        for direction, side in (('anterior', y >= 0), ('posterior', y < 0), ('superior', z >= 0), ('inferior', z < 0)):
            chosen = (side & cortical)[keep]
            out['directional'][seed, direction] = dict(
                everything=(int((side & windows['eligible']).sum()), int(side.sum())),
                cortical=(int((side & cortical & windows['eligible']).sum()), int((side & cortical).sum())),
                difference=float(np.median(pooled['bem'][:, chosen]) - np.median(pooled['package_sphere'][:, chosen])))

    headline = {}
    for key in ('bem_median_mm', 'package_sphere_median_mm', 'median_difference_mm'):
        mean, limits = over_seeds({seed: by_seed[seed]['paired_pooled_model_seeds']['all'][key] for seed in SEEDS})
        headline[key] = dict(mean_of_dataset_seed_medians=mean, range=limits)
    write_json(base / 'summary.json', dict(smoke=False, check5_all_pass=True, paired_headline=headline,
                                           by_dataset_seed=by_seed))
    template = {'fsaverage-trans.fif': 'd' * 64, 'fsaverage-5120-5120-5120-bem.fif': 'c' * 64}
    write_json(base / 'provenance.json', dict(commit='5' * 40, smoke=False, template_sha256=template,
                                              arguments=['--output-dir', 'head-transfer-v11']))
    out['headline'] = headline

    # Strata of the summary: the mean over the datasets with at least 30 positions in the stratum.
    out['strata'] = []
    for label in by_seed[SEEDS[0]]['paired_pooled_model_seeds']:
        entries = {seed: by_seed[seed]['paired_pooled_model_seeds'][label] for seed in SEEDS}
        seeds = [seed for seed in SEEDS if entries[seed]['position_clusters'] >= MINIMUM_POSITIONS]
        row = dict(stratum=label, seeds=seeds, windows=sum(entries[seed]['windows'] for seed in seeds),
                   positions={seed: entries[seed]['position_clusters'] for seed in SEEDS},
                   window_counts={seed: entries[seed]['windows'] for seed in SEEDS})
        for name, key in (('sphere', 'package_sphere_median_mm'), ('bem', 'bem_median_mm'),
                          ('difference', 'median_difference_mm')):
            row[name] = float(np.mean([entries[seed][key] for seed in seeds])) if seeds else None
        out['strata'].append(row)

    # Strata of the windows: averaged only if every dataset has at least 30 positions in the stratum.
    out['window_strata'] = []
    for name in window_strata_masks(paired_of[SEEDS[0]]):
        masks = {seed: window_strata_masks(paired_of[seed])[name] for seed in SEEDS}
        positions = [len(set(paired_of[seed]['position'][masks[seed]].tolist())) for seed in SEEDS]
        row = dict(stratum=name, windows=int(sum(mask.sum() for mask in masks.values())), positions=min(positions),
                   sphere=None, bem=None, difference=None)
        if min(positions) >= MINIMUM_POSITIONS:
            sphere, bem = ([float(np.median(records[seed][0][head][:, masks[seed]])) for seed in SEEDS]
                           for head in ('package_sphere', 'bem'))
            row.update(sphere=float(np.mean(sphere)), bem=float(np.mean(bem)),
                       difference=float(np.mean(np.subtract(bem, sphere))))
        out['window_strata'].append(row)
    expected['transfer'] = out
    return dict(paired=paired_of, heads=heads_of, template=sorted(template), dipole_subset=dipole_subset)


# --------------------------------------------------------------------------- adaptation of the head model

def head_adaptation(base, rng, study, transfer, expected):
    """head-adaptation-v11: sphere variants and fine-tuning on the boundary-element windows."""
    geometry = dict(
        definition='Head radius: median distance of the template electrodes from the registered centre.',
        head_radius_m=0.1004, scale=1.0915,
        variants={'default': dict(radii_m=[0.08, 0.081, 0.086, 0.092],
                                  conductivities_S_per_m=[0.33, 1.79, 0.0042, 0.33]),
                  'three_shell': dict(radii_m=[0.08, 0.081, 0.086, 0.092],
                                      conductivities_S_per_m=[0.33, 0.33, 0.0042, 0.33]),
                  'size_matched': dict(radii_m=[0.0873, 0.0884, 0.0939, 0.1004],
                                       conductivities_S_per_m=[0.33, 1.79, 0.0042, 0.33]),
                  'size_matched_three_shell': dict(radii_m=[0.0873, 0.0884, 0.0939, 0.1004],
                                                   conductivities_S_per_m=[0.33, 0.33, 0.0042, 0.33]),
                  'surface_fitted_three_shell': dict(radii_m=[0.0849, 0.0859, 0.0976, 0.1008],
                                                     conductivities_S_per_m=[0.33, 0.33, 0.0042, 0.33])})
    by_seed, medians, differences, single_networks = {}, {arm: {} for arm in ARMS}, {}, {}
    own_medians = {variant: {} for variant in VARIANTS}
    training_windows = dict(zip(SEEDS, (310, 305, 298)))
    for seed in SEEDS:
        paired, heads = transfer['paired'][seed], transfer['heads'][seed]
        truth, count = paired['truth'], len(paired['position'])
        positions, own = {}, {}
        for arm in ARMS:
            for number, fit in enumerate(FITS):
                positions[f'{arm}_mlp_{fit}'] = heads[f'mlp_{fit}_bem'] if arm == 'default' else scattered(
                    rng, truth, ARM_SCALES[arm] * (1.0, 1.25)[number])
                if arm in VARIANTS:
                    own[f'{arm}_own_sphere_mlp_{fit}'] = heads[f'mlp_{fit}_package_sphere'] if arm == 'default' \
                        else scattered(rng, truth, (1.9, 2.5)[number])
        errors = {arm: np.stack([np.linalg.norm(positions[f'{arm}_mlp_{fit}'] - truth, axis=1) for fit in FITS])
                  for arm in ARMS}
        own_errors = {variant: np.stack([np.linalg.norm(own[f'{variant}_own_sphere_mlp_{fit}'] - truth, axis=1)
                                         for fit in FITS]) for variant in VARIANTS}

        def arm_summary(arm, mask):
            entry = error_summary(errors[arm][:, mask], paired['position'][mask])
            if arm != 'default':
                difference = float(np.median(errors[arm][:, mask]) - np.median(errors['default'][:, mask]))
                entry.update(median_difference_from_default_mm=difference, median_difference_ci95_mm=around(difference))
            return entry

        strata = population_strata(paired)
        record = dict(
            bem_windows=count, positions=len(set(paired['position'].tolist())),
            arms={arm: arm_summary(arm, strata['all']) for arm in ARMS},
            strata={label: {arm: arm_summary(arm, strata[label]) for arm in ARMS}
                    for label in ('all', 'population_1', 'radius_0_75_79.501')},
            own_sphere={variant: dict(eligible_windows=error_summary(own_errors[variant], paired['position']))
                        for variant in VARIANTS},
            fits={f'{variant}_mlp_{fit}': dict(best_epoch=int(rng.integers(20, 90)), epochs_run=100)
                  for variant in VARIANTS[1:] for fit in FITS},
            bem_training_eligibility=dict(train=dict(windows=480, eligible_windows=training_windows[seed]),
                                          validation=dict(windows=48, eligible_windows=30)),
            # One row per fine-tuned network and per network trained on the boundary-element windows alone: the
            # numbers of its training windows, and the statistics of its own errors in the prediction file.
            finetune=[dict(dataset_seed=int(seed), model_seed=int(fit), size=size, arm=kind,
                           bem_training_windows=training_windows[seed] if size == 'all' else int(size),
                           bem_validation_windows=30 if size == 'all' else int(size) // 4,
                           **three_statistics(errors[f'{kind}_{size}'][number]))
                      for size in SIZES for number, fit in enumerate(FITS) for kind in ('finetuned', 'scratch')])
        by_seed[seed] = record
        single_networks[seed] = {(row['arm'], row['size'], str(row['model_seed'])): row['median_mm']
                                 for row in record['finetune']}
        for arm in ARMS:
            medians[arm][seed] = record['arms'][arm]['median_mm']
            if arm != 'default':
                differences.setdefault(arm, {})[seed] = record['arms'][arm]['median_difference_from_default_mm']
        for variant in VARIANTS:
            own_medians[variant][seed] = record['own_sphere'][variant]['eligible_windows']['median_mm']
        columns = {'window_id': paired['window'], 'position_pool_id': paired['position'],
                   'population': paired['population'], 'source_radius_mm': paired['radius'],
                   'snr_db_referenced': paired['snr'], 'alpha_bem': rng.uniform(0.85, 1.4, count)}
        for arm in ARMS:
            for number, fit in enumerate(FITS):
                columns[f'{arm}_mlp_{fit}_error_mm'] = errors[arm][number]
                if arm in VARIANTS:
                    columns[f'{arm}_own_sphere_mlp_{fit}_error_mm'] = own_errors[arm][number]
        write_csv(base / seed / 'adaptation_predictions.csv', columns)
        np.savez_compressed(base / seed / 'predictions.npz', window_id=paired['window'], true_position_mm=truth,
                            **positions, **own)
        if seed == SEEDS[0]:
            chosen = expected['transfer']['dipole_subset']['chosen']
            recorded = expected['transfer']['dipole_subset']['errors']
            fits = {f'dipole_{variant}': scattered(rng, truth[chosen], scale)
                    for variant, scale in zip(VARIANTS, (3.6, 4.4, 3.1, 3.0, 2.9))}
            subset = {f'{name}_sphere_on_bem': np.linalg.norm(values - truth[chosen], axis=1)
                      for name, values in fits.items()}
            subset.update(dipole_default_sphere_on_sphere_recorded=recorded['ecd_package_sphere_matched'],
                          dipole_bem_on_bem_recorded=recorded['ecd_bem_matched'])
            subset.update({f'{arm}_mlp_{fit}': errors[arm][number, chosen]
                           for arm in ('default', 'size_matched_three_shell', 'surface_fitted_three_shell')
                           for number, fit in enumerate(FITS)})
            dipole_subset = {}
            for name, values in subset.items():
                dipole_subset[name] = error_summary(values, paired['position'][chosen])
                if name.endswith('_sphere_on_bem'):
                    difference = float(np.median(values) - np.median(subset['dipole_bem_on_bem_recorded']))
                    dipole_subset[name].update(median_difference_from_bem_matched_mm=difference,
                                               median_difference_ci95_mm=around(difference))
            write_csv(base / seed / 'dipole_subset.csv', {
                'window_id': paired['window'][chosen], 'position_pool_id': paired['position'][chosen],
                **{f'{name}_error_mm': values for name, values in subset.items()},
                **{key: values for name, where in fits.items()
                   for key, values in coordinates(f'{name}_', where).items()}})
            # The experiment wrote the eligibility file of its test windows once more: the same windows, and so
            # the same bytes, as in the transfer experiment. It holds the true source positions of the fits. In
            # the archive every dataset of the experiment has the file; the miniature holds it for the one
            # dataset whose dipole fits the driver reads.
            write_csv(base / seed / 'eligibility.csv', eligibility_columns(study['sets'][seed, 'position']))
            sphere_fits = {variant: fits[f'dipole_{variant}'] for variant in VARIANTS}

    bem_headline, own_headline = {}, {}
    for arm in ARMS:
        mean, limits = over_seeds(medians[arm])
        bem_headline[arm] = dict(dataset_seed_medians_mm=[medians[arm][seed] for seed in SEEDS],
                                 mean_of_dataset_seed_medians=mean, range=limits)
        if arm in differences:
            bem_headline[arm]['mean_difference_from_default_mm'] = over_seeds(differences[arm])[0]
    for variant in VARIANTS:
        mean, limits = over_seeds(own_medians[variant])
        own_headline[variant] = dict(mean_of_dataset_seed_medians=mean, range=limits)
    write_json(base / 'summary.json', dict(smoke=False, geometry=geometry, bem_windows_headline=bem_headline,
                                           own_sphere_eligible_headline=own_headline, dipole_subset=dipole_subset,
                                           by_dataset_seed=by_seed))
    geometry_bytes = write_json(base / 'geometry.json', geometry)
    write_json(base / 'provenance.json', dict(commit='6' * 40, package_tree='b' * 40,
                                              arguments=['--output-dir', 'head-adaptation-v11']))
    expected['adaptation'] = dict(
        medians=medians, differences=differences, own_sphere=own_headline, dipole_subset=dipole_subset,
        dipole_windows=DIPOLE_WINDOWS, bem_windows={seed: by_seed[seed]['bem_windows'] for seed in SEEDS},
        single_networks=single_networks,
        finetuning=[(size, sorted(set(training_windows.values())) if size == 'all' else [int(size)]) for size in SIZES])
    return geometry, hashlib.sha256(geometry_bytes).hexdigest(), sphere_fits


# --------------------------------------------------------------------------- benchmark and small records

def benchmark(base, rng, expected):
    """generation-benchmark-v11: timed generator runs; the summary of a case is computed from its runs."""
    cases, rows = [], []
    for name, mode, form, compress, workers, windows, repeats, seconds, size in (
            ('point-1', 'point', 'hdf5', True, 1, 12000, 3, 76.0, 53900),
            ('patch-1', 'patch', 'hdf5', True, 1, 12000, 3, 79.0, 55700),
            ('hdf5-uncompressed-1', 'point', 'hdf5', False, 1, 12000, 3, 48.0, 56800),
            ('npz-1', 'point', 'npz', True, 1, 12000, 3, 73.0, 50400),
            ('default-scale-6', 'point', 'hdf5', True, 6, 360000, 1, 1175.0, 53800)):
        runs = []
        for number in range(repeats):
            phases = dict(basis_forward=float(rng.uniform(12.0, 15.0)), background_pool=float(rng.uniform(4.6, 5.0)))
            if form == 'hdf5':
                phases.update(hdf5_parent_append=float(rng.uniform(1.0, 24.0)),
                              hdf5_finalize=float(rng.uniform(0.1, 2.4)))
            phases['split_generation_and_writing'] = float(rng.uniform(0.5, 0.7) * seconds)
            duration = float(seconds * rng.uniform(0.97, 1.05))
            runs.append(dict(repetition=number + 1, seconds=duration, windows_per_second=windows / duration,
                             peak_process_tree_rss_bytes=int(rng.integers(500, 3000)) * 2 ** 20,
                             output_bytes=windows * size, phase_seconds=phases,
                             setup_seconds=phases['basis_forward'] + phases['background_pool']))
        middle = sorted(runs, key=lambda run: run['seconds'])[repeats // 2]
        summary = {key: dict(median=float(np.median([run[key] for run in runs])),
                             minimum=float(np.min([run[key] for run in runs])),
                             maximum=float(np.max([run[key] for run in runs])))
                   for key in ('seconds', 'windows_per_second', 'peak_process_tree_rss_bytes')}
        summary.update(output_bytes=runs[0]['output_bytes'], median_run_phase_seconds=middle['phase_seconds'],
                       median_run_setup_seconds=middle['setup_seconds'])
        case = dict(case=name, mode=mode, workers=workers, format=form, compress=compress, windows=windows,
                    repetitions=runs, summary=summary)
        if mode == 'patch':
            case['patch_overlap'] = dict(algorithm='exact small-cap lens and size pruning')
        cases.append(case)
        setup = [run['setup_seconds'] for run in runs]
        phases = middle['phase_seconds']
        rows.append(dict(
            case=name, form=form, compressed=compress, workers=workers, windows=windows, repetitions=repeats,
            seconds=summary['seconds'], rate=summary['windows_per_second'],
            memory_gib={key: value / 1024 ** 3 for key, value in summary['peak_process_tree_rss_bytes'].items()},
            output_gib=summary['output_bytes'] / 1024 ** 3, setup_range=(min(setup), max(setup)),
            setup=middle['setup_seconds'], generation=phases['split_generation_and_writing'],
            writing=phases['hdf5_parent_append'] + phases['hdf5_finalize'] if form == 'hdf5' else None))
    host = dict(platform='Windows-11', cpu='six cores', logical_cpus=12, physical_cpus=6, ram_bytes=16 * 2 ** 30,
                available_ram_bytes=4 * 2 ** 30, threads=dict(OMP_NUM_THREADS=None), channels=69, window_samples=100,
                environment=dict(package='spherical_eeg_sim', python_version='3.12'))
    write_json(base / 'generation_benchmark.json', dict(host, sampling_freq_hz=500.0, cases=cases))
    write_json(base / 'provenance.json', dict(commit='7' * 40, arguments=['--output-dir', 'generation-benchmark-v11']))
    expected['benchmark'] = dict(rows=rows, host=host)


def small_records(root, expected):
    """continuity and shared-position-v11: records that the driver passes through."""
    for name, version in (('v100-continuity', '1.0.0'), ('v11-continuity', '1.1.0')):
        write_json(root / 'continuity' / f'{name}.json', dict(
            figure=f'{name}.png', seed=862, localization_error_mm=dict(mean=5.9, median=5.35, p90=10.2),
            software_environment=dict(software_versions=dict(spherical_eeg_sim=version))))
    shared = [dict(requested_fraction=0.01, realized_fraction=0.0102, test_windows=TEST_WINDOWS,
                   test_windows_at_a_training_location=6, test_window_reuse_fraction=6 / TEST_WINDOWS),
              dict(requested_fraction=1.0, realized_fraction=1.0, test_windows=TEST_WINDOWS,
                   test_windows_at_a_training_location=63, test_window_reuse_fraction=63 / TEST_WINDOWS)]
    write_json(root / 'shared-position-v11/summary.json', shared)
    write_json(root / 'shared-position-v11/provenance.json',
               dict(commit='4' * 40, arguments=['--output-dir', 'shared-position-v11']))
    expected['shared'] = shared


# --------------------------------------------------------------------------- comparisons with MNE-Python

def fit_record(rng, name, evaluations):
    """How a parameter fit of the sphere model ended; with the raised limit it needs ``evaluations`` (a range)."""
    stopped = name == 'raised_limit'
    return dict(
        evaluations=int(rng.integers(*evaluations)) if stopped else EVALUATION_LIMITS[name], stopped_by_itself=stopped,
        status=0 if stopped else 3, message='the trust region reached its lower bound' if stopped else 'limit reached',
        objective=float(rng.uniform(3.85e-6, 3.87e-6) if stopped else rng.uniform(1e-4, 5e-4)),
        first_evaluations=[dict(mu=[0.0, 0.0, 0.0], objective=1.0366),
                           dict(mu=[0.5, 0.0, 0.0], objective=float(rng.uniform(0.40, 0.44))),
                           dict(mu=[0.0, 0.5, 0.0], objective=0.7711), dict(mu=[0.0, 0.0, 0.5], objective=0.9102)],
        evaluation_limit=EVALUATION_LIMITS[name], mu=rng.uniform(0.3, 0.98, 3), lambda_=rng.uniform(0.1, 1.2, 3))


def field_samples(rng, name, samples, patch=1.0, point=1.0):
    """Relative field differences of patches and of point sources with one sphere fit.

    The two factors shift the samples of a run, so that each kind of run holds the smallest or the largest
    median of some comparison.
    """
    scale = 2.4e-3 if name == 'mne_limit' else 5e-4
    return dict(patch=rng.uniform(0.8, 1.25, samples) * scale * patch,
                point=rng.uniform(0.8, 1.25, samples) * scale * point)


def field_comparison(fields):
    return dict(patch=spread(fields['patch']), point=spread(fields['point']),
                patch_to_point_median_ratio=float(np.median(fields['patch']) / np.median(fields['point'])))


def per_sample_file(sources, run):
    """The columns of a per-dipole file of the dipole recovery, with the decimals of the validation script."""
    estimated_moment = sources['moment'] * (1.0 + run['moment_relative_error'])
    columns = {'sample_index': np.arange(len(sources['moment'])), 'fit_sample': sources['fit_sample'],
               'source_radius_mm': [f'{value:.3f}' for value in np.linalg.norm(sources['true'], axis=1)],
               'source_depth_mm': [f'{92.0 - value:.3f}' for value in np.linalg.norm(sources['true'], axis=1)]}
    for label, positions in (('true', sources['true']), ('estimated', run['fitted'])):
        columns.update({f'{label}_{axis}_m': [f'{value / 1000.0:.9f}' for value in positions[:, index]]
                        for index, axis in enumerate('xyz')})
    columns.update(
        err_mm=[f'{value:.4f}' for value in run['position_error_mm']],
        gof_pct=[f'{value:.4f}' for value in run['goodness_of_fit_percent']],
        orientation_error_deg=[f'{value:.5f}' for value in run['orientation_error_deg']],
        true_moment_nAm=[f'{value:.8f}' for value in sources['moment']],
        estimated_moment_nAm=[f'{value:.8f}' for value in estimated_moment],
        moment_absolute_error_nAm=[f'{value:.8f}' for value in estimated_moment - sources['moment']],
        moment_relative_error=[f'{value:.8f}' for value in run['moment_relative_error']])
    columns.update({f'true_orientation_{axis}': [f'{value:.8f}' for value in sources['orientation'][:, index]]
                    for index, axis in enumerate('xyz')})
    columns.update(snr_db=[f'{value:.3f}' for value in sources['snr']], population=sources['population'])
    return columns


def three(values):
    """Median, 90th percentile and maximum, as the summary of the dipole recovery stores them."""
    return dict(median=float(np.median(values)), p90=float(np.percentile(values, 90)), maximum=float(np.max(values)))


def comparisons_with_mne(root, rng, geometry, geometry_sha256, expected):
    """validation-rerun-v11 and sphere-fit-convergence-v11.

    The same dipoles are fitted on two platforms, with two releases, with both sphere fits, with objectives
    changed in the last bits, with dipole fits that end at a small trust-region radius, with other kernels of
    the linear-algebra library and on the spheres of the adaptation experiment.
    """
    rerun, base = root / 'validation-rerun-v11', root / 'sphere-fit-convergence-v11'
    count, levels, sweep_count, samples, steps = 12, (5.0, 20.0, 35.0), 6, 5, (1, 2, 3)
    kernels = ('Sandybridge', 'Nehalem')
    true = np.r_[rng.uniform(30.0, 65.0, 4), rng.uniform(70.5, 79.5, count - 4)][:, None] * directions(rng, count)
    sources = dict(true=true, fit_sample=rng.integers(0, 100, count), moment=rng.uniform(5.0, 300.0, count),
                   orientation=directions(rng, count), snr=rng.uniform(34.6, 35.4, count),
                   population=['subcortical'] * 4 + ['cortical'] * (count - 4))

    def run_of(fitted):
        """The per-dipole results of a run with the given fitted positions in mm."""
        return dict(fitted=fitted, position_error_mm=np.linalg.norm(fitted - true, axis=1),
                    orientation_error_deg=rng.gamma(2.0, 0.15, count),
                    moment_relative_error=rng.gamma(2.0, 0.002, count),
                    goodness_of_fit_percent=rng.uniform(99.985, 99.998, count))

    def first_fit(scale):
        """The fitted positions of an unchanged run. Its worst dipole lies twice as far from the truth as drawn."""
        fitted = scattered(rng, true, scale)
        worst = np.argmax(np.linalg.norm(fitted - true, axis=1))
        fitted[worst] += fitted[worst] - true[worst]
        return fitted

    def nudged(name, scale):
        """The fitted positions of an unchanged run, moved a little: some by more than 0.1 mm, most by less.

        The dipole with the largest error moves towards its true position, so that the unchanged run keeps
        the largest error of all runs of its sphere fit.
        """
        reference = unchanged[name]['fitted']
        moved = directions(rng, count) * rng.gamma(2.0, scale, (count, 1))
        worst = np.argmax(unchanged[name]['position_error_mm'])
        inward = true[worst] - reference[worst]
        moved[worst] = inward / np.linalg.norm(inward) * np.linalg.norm(moved[worst])
        return reference + moved

    quantities = ('position_error_mm', 'orientation_error_deg', 'moment_relative_error', 'goodness_of_fit_percent')
    unchanged = {name: run_of(first_fit(scale)) for name, scale in zip(SPHERE_FITS, (0.17, 0.14))}
    # Among the runs of one sphere fit the unchanged run has the smallest patch median and the largest point
    # median, a trial the smallest point median and a kernel the largest patch median. With the raised limit
    # the unchanged run needs the fewest evaluations and a trial the most; at the limit of SciPy one kernel
    # needs fewer than the limit.
    fields = {name: field_samples(rng, name, samples, patch=0.6, point=2.4) for name in SPHERE_FITS}
    sweep = {name: {level: rng.gamma(2.0, scale, sweep_count) for level, scale in zip(levels, (5.0, 0.9, 0.17))}
             for name in SPHERE_FITS}
    arms = {name: dict(
        fit=dict(fit_record(rng, name, (1500, 1700)), objective_scale_minus_one_in_eps=0),
        dipole_recovery={quantity: spread(unchanged[name][quantity]) for quantity in quantities},
        sweep=[dict(snr_db=level, position_error_mm=spread(sweep[name][level])) for level in levels],
        patch_comparison=field_comparison(fields[name])) for name in SPHERE_FITS}
    write_csv_lf(base / 'dipole_recovery_35db.csv', {
        'sample_index': np.arange(count), 'source_radius_mm': np.linalg.norm(true, axis=1), 'snr_db': sources['snr'],
        **{f'{quantity}_{name}': unchanged[name][quantity] for name in SPHERE_FITS for quantity in quantities}})
    write_csv_lf(base / 'field_differences.csv', {
        'source': ['patch'] * samples + ['point'] * samples, 'sample_index': [*range(samples)] * 2,
        'active_dipoles': [*rng.integers(40, 400, samples), *[''] * samples],
        'patch_diameter_mm': [*rng.uniform(10.0, 60.0, samples), *[''] * samples],
        **{f'relative_field_difference_{name}': np.r_[fields[name]['patch'], fields[name]['point']]
           for name in SPHERE_FITS}})
    write_csv_lf(base / 'snr_sweep.csv', {
        'snr_db': np.repeat(levels, sweep_count), 'sample_index': np.tile(np.arange(sweep_count), len(levels)),
        **{f'position_error_mm_{name}': np.concatenate([sweep[name][level] for level in levels])
           for name in SPHERE_FITS}})

    # Trials with a changed objective: of the parameter fit, then of the dipole fits.
    def dipole_rows(labels, run):
        """The rows of one run in a per-dipole file: its labels, the fitted position in mm and the four errors."""
        return [[*labels, index, *run['fitted'][index], *[run[quantity][index] for quantity in quantities]]
                for index in range(count)]

    trials, per_dipole, per_field, trial_fits = [], [], [], {}
    for name in SPHERE_FITS:
        per_dipole += dipole_rows(('none', name, 0), unchanged[name])
    for kind in ('sphere_fit', 'dipole_fit'):
        for step in steps:
            for name in SPHERE_FITS:
                run = run_of(nudged(name, 0.035))
                errors = run['position_error_mm']
                moved = np.linalg.norm(run['fitted'] - unchanged[name]['fitted'], axis=1)
                fit = fit_record(rng, name, (2500, 3500)) if kind == 'sphere_fit' else arms[name]['fit']
                trial_fits[kind, name, step] = fit
                trial_fields = field_samples(rng, name, samples, point=0.6) if kind == 'sphere_fit' else None
                refits = kind == 'dipole_fit'
                trials.append(dict(
                    perturbed_fit=kind, sphere_fit=name, objective_scale_minus_one_in_eps=step,
                    parameter_fit_evaluations=fit['evaluations'],
                    parameter_fit_stopped_by_itself=int(fit['stopped_by_itself']),
                    parameter_fit_objective=fit['objective'], mu_1=fit['mu'][0], mu_2=fit['mu'][1], mu_3=fit['mu'][2],
                    dipole_fits=count if refits else '', dipole_fits_stopped_by_themselves=count if refits else '',
                    dipole_fit_evaluations_median=float(rng.integers(40, 45)) if refits else '',
                    dipole_fit_evaluations_maximum=float(rng.integers(95, 130)) if refits else '',
                    patch_median_relative_field_difference=np.median(trial_fields['patch']) if trial_fields else '',
                    point_median_relative_field_difference=np.median(trial_fields['point']) if trial_fields else '',
                    position_error_median_mm=np.median(errors), position_error_p90_mm=np.percentile(errors, 90),
                    position_error_maximum_mm=errors.max(),
                    moment_relative_error_median=np.median(run['moment_relative_error']),
                    fitted_position_change_median_mm=np.median(moved),
                    fitted_position_change_p90_mm=np.percentile(moved, 90),
                    fitted_position_change_maximum_mm=moved.max(),
                    fitted_positions_changed_by_more_than_0p1_mm=int(np.sum(moved > 0.1))))
                per_dipole += dipole_rows((kind, name, step), run)
                if trial_fields:
                    per_field += [[name, step, source, index, value] for source in ('patch', 'point')
                                  for index, value in enumerate(trial_fields[source])]
    write_csv_lf(base / 'perturbation.csv', {column: [trial[column] for trial in trials] for column in trials[0]})
    dipole_columns = ('sample_index', 'fitted_x_mm', 'fitted_y_mm', 'fitted_z_mm', *quantities)
    write_csv_lf(base / 'perturbation_dipoles.csv', dict(zip(
        ('perturbed_fit', 'sphere_fit', 'objective_scale_minus_one_in_eps', *dipole_columns), zip(*per_dipole))))
    write_csv_lf(base / 'perturbation_fields.csv', dict(zip(
        ('sphere_fit', 'objective_scale_minus_one_in_eps', 'source', 'sample_index', 'relative_field_difference'),
        zip(*per_field))))
    # Smallest and largest value of every quantity over the trials of one kind with one sphere fit.
    labels = ('perturbed_fit', 'sphere_fit', 'objective_scale_minus_one_in_eps', 'parameter_fit_stopped_by_itself')
    ranges = {}
    for kind in ('sphere_fit', 'dipole_fit'):
        ranges[kind] = {}
        for name in SPHERE_FITS:
            group = [trial for trial in trials if (trial['perturbed_fit'], trial['sphere_fit']) == (kind, name)]
            entry = {column: [float(end(trial[column] for trial in group)) for end in (min, max)]
                     for column in trials[0] if column not in labels and group[0][column] != ''}
            entry.update(trials=len(group), trials_in_which_the_parameter_fit_stopped_by_itself=sum(
                trial['parameter_fit_stopped_by_itself'] for trial in group))
            ranges[kind][name] = entry

    # Other kernels of the linear-algebra library, each with both sphere fits.
    default_sphere = {name: dict(evaluations=87, stopped_by_itself=True, evaluation_limit=EVALUATION_LIMITS[name])
                      for name in SPHERE_FITS}
    kernel_record, kernel_dipoles, kernel_fields = {}, [], []
    for kernel in kernels:
        entry = dict(libraries=[dict(library='openblas', version='0.3.31', kernel=kernel)], arms={},
                     default_sphere_model_of_mne_python=default_sphere, small_tolerance={},
                     fitted_position_change_from_the_parent_mm={})
        for name in SPHERE_FITS:
            run = run_of(nudged(name, 0.05))
            moved = np.linalg.norm(run['fitted'] - unchanged[name]['fitted'], axis=1)
            own_fields = field_samples(rng, name, samples, patch=1.5, point=1.5)
            fit = fit_record(rng, name, (1800, 2400))
            if (kernel, name) == (kernels[-1], 'mne_limit'):
                # Under the last kernel the fit stops by itself before it reaches the limit of SciPy.
                fit.update(evaluations=950, stopped_by_itself=True, status=0)
            entry['arms'][name] = dict(
                fit=dict(fit, objective_scale_minus_one_in_eps=0),
                dipole_recovery={quantity: spread(run[quantity]) for quantity in quantities}, sweep=[],
                patch_comparison=field_comparison(own_fields))
            entry['fitted_position_change_from_the_parent_mm'][name] = dict(
                spread(moved), share_above_0p1_mm=float(np.mean(moved > 0.1)))
            kernel_dipoles += dipole_rows((kernel, name), run)
            kernel_fields += [[kernel, name, source, index, value] for source in ('patch', 'point')
                              for index, value in enumerate(own_fields[source])]
        kernel_record[kernel] = entry
    write_csv_lf(base / 'kernels_dipole_recovery_35db.csv', dict(zip(
        ('kernel', 'sphere_fit', *dipole_columns), zip(*kernel_dipoles))))
    write_csv_lf(base / 'kernels_field_differences.csv', dict(zip(
        ('kernel', 'sphere_fit', 'source', 'sample_index', 'relative_field_difference'), zip(*kernel_fields))))

    # Dipole recoveries in which the dipole fits end at a small trust-region radius: the unchanged run of the
    # main process, runs with a changed objective of the parameter fit or of the dipole fits, and the unchanged
    # run under every other kernel. A run records how far its fitted positions lie from those of the unchanged
    # run of the same sphere fit in the main process. At the limit of SciPy a changed parameter fit and another
    # kernel move the positions by hundredths of a millimetre; every other change moves them far less.
    own_kernel, small_steps, wide, narrow = 'Haswell', steps[:2], (0.01, 0.06), (5e-5, 3e-4)
    small_runs, small_dipoles = [], []

    def shifted(reference, low, high):
        """Positions that lie between ``low`` and ``high`` mm from those of ``reference``."""
        return reference + directions(rng, count) * rng.uniform(low, high, (count, 1))

    def small_run(kernel, changed, name, step, fit, fitted, stopped=count):
        """Record a run with the small final radius; returns its errors and how far its positions moved."""
        errors = np.linalg.norm(fitted - true, axis=1)
        first = (kernel, changed) == (own_kernel, 'none')
        moved = None if first else np.linalg.norm(fitted - small_unchanged[name], axis=1)
        small_runs.append(dict(
            kernel=kernel, changed_objective=changed, sphere_fit=name, objective_scale_minus_one_in_eps=step,
            parameter_fit_evaluations=fit['evaluations'],
            parameter_fit_stopped_by_itself=int(fit['stopped_by_itself']), dipole_fits=count,
            dipole_fits_stopped_by_themselves=stopped, dipole_fit_evaluations_median=float(rng.integers(70, 82)),
            dipole_fit_evaluations_maximum=float(rng.integers(85, 100)),
            position_error_median_mm=np.median(errors), position_error_p90_mm=np.percentile(errors, 90),
            position_error_maximum_mm=errors.max(),
            fitted_position_change_median_mm='' if first else np.median(moved),
            fitted_position_change_p90_mm='' if first else np.percentile(moved, 90),
            fitted_position_change_maximum_mm='' if first else moved.max()))
        small_dipoles.extend([kernel, changed, name, step, index, *fitted[index], errors[index]]
                             for index in range(count))
        return errors, moved

    small_unchanged = {name: shifted(unchanged[name]['fitted'], 0.005, 0.03) for name in SPHERE_FITS}
    for name in SPHERE_FITS:
        small_run(own_kernel, 'none', name, 0, arms[name]['fit'], small_unchanged[name])
    for step in small_steps:
        for name, reach in zip(SPHERE_FITS, (wide, narrow)):
            # The changed parameter fit is that of the trial above. In one run a dipole fit ends at its limit.
            small_run(own_kernel, 'parameter_fit', name, step, trial_fits['sphere_fit', name, step],
                      shifted(small_unchanged[name], *reach),
                      stopped=count - 1 if (name, step) == (SPHERE_FITS[0], small_steps[0]) else count)
            fitted = shifted(small_unchanged[name], *narrow)
            if (name, step) == (SPHERE_FITS[1], small_steps[-1]):
                # In one run a single dipole moves: the median and the 90th percentile of the changes are zero.
                fitted = small_unchanged[name].copy()
                fitted[3] += (2e-4, -1e-4, 2e-4)
            small_run(own_kernel, 'dipole_fits', name, step, arms[name]['fit'], fitted)
    for kernel in kernels:
        for name, reach in zip(SPHERE_FITS, (wide, narrow)):
            fitted = shifted(small_unchanged[name], *reach)
            if (kernel, name) == (kernels[0], SPHERE_FITS[0]):
                # Under the first kernel one dipole moves by more than 0.1 mm.
                fitted[5] = small_unchanged[name][5] + (0.09, -0.08, 0.0)
            fit = kernel_record[kernel]['arms'][name]['fit']
            errors, moved = small_run(kernel, 'none', name, 0, fit, fitted)
            kernel_record[kernel]['small_tolerance'][name] = dict(
                fit=fit, dipole_fits=dict(fits=count, stopped_by_themselves=count,
                                          evaluations=spread(rng.integers(60, 100, count))),
                position_error_mm=spread(errors),
                fitted_position_change_from_the_parent_mm=dict(spread(moved),
                                                               share_above_0p1_mm=float(np.mean(moved > 0.1))))
    write_csv_lf(base / 'small_tolerance.csv', {column: [run[column] for run in small_runs]
                                                for column in small_runs[0]})
    write_csv_lf(base / 'small_tolerance_dipoles.csv', dict(zip(
        ('kernel', 'changed_objective', 'sphere_fit', 'objective_scale_minus_one_in_eps', 'sample_index',
         'fitted_x_mm', 'fitted_y_mm', 'fitted_z_mm', 'position_error_mm'), zip(*small_dipoles))))
    # Over the runs of one sphere fit: counts, and the smallest and largest value of the position statistics.
    small_ranges = {}
    for name in SPHERE_FITS:
        group = [run for run in small_runs if run['sphere_fit'] == name]
        small_ranges[name] = dict(
            runs=len(group), kernels=sorted({run['kernel'] for run in group}),
            runs_in_which_the_parameter_fit_stopped_by_itself=sum(run['parameter_fit_stopped_by_itself']
                                                                  for run in group),
            dipole_fits=sum(run['dipole_fits'] for run in group),
            dipole_fits_stopped_by_themselves=sum(run['dipole_fits_stopped_by_themselves'] for run in group),
            **{column: [float(end(run[column] for run in group if run[column] != '')) for end in (min, max)]
               for column in small_runs[0] if column.startswith(('position_error', 'fitted_position_change'))})

    # The spheres of the adaptation experiment: point sources, and a dipole recovery at 10 to 25 dB. The fitted
    # positions with the converged parameter fit lie near those with the fit at the limit of SciPy.
    sphere_record, sphere_dipoles, sphere_fields = {}, [], []
    for sphere, model in geometry['variants'].items():
        fitted = dict(mne_limit=scattered(rng, true, 1.2))
        fitted['raised_limit'] = fitted['mne_limit'] + directions(rng, count) * rng.gamma(2.0, 0.05, (count, 1))
        errors = {name: np.linalg.norm(fitted[name] - true, axis=1) for name in SPHERE_FITS}
        own_fields = {name: field_samples(rng, name, samples)['point'] for name in SPHERE_FITS}
        entry = dict(radii_m=model['radii_m'], conductivities_S_per_m=model['conductivities_S_per_m'],
                     dataset_array_sha256=dict(fields=hashlib.sha256(f'fields {sphere}'.encode()).hexdigest(),
                                               windows=hashlib.sha256(f'windows {sphere}'.encode()).hexdigest()))
        for name in SPHERE_FITS:
            entry[name] = dict(fit=fit_record(rng, name, (1700, 6500)), point_field_difference=spread(own_fields[name]),
                               position_error_mm=spread(errors[name]))
        moved = np.linalg.norm(fitted['raised_limit'] - fitted['mne_limit'], axis=1)
        entry['position_error_change_mm'] = dict(
            median_of_raised_limit_minus_median_of_mne_limit=float(np.median(errors['raised_limit']) -
                                                                   np.median(errors['mne_limit'])),
            fitted_position=dict(spread(moved), share_above_0p1_mm=float(np.mean(moved > 0.1))))
        sphere_record[sphere] = entry
        radius, snr = np.linalg.norm(true, axis=1), rng.uniform(10.0, 25.0, count)
        sphere_dipoles += [[sphere, index, radius[index], snr[index],
                            *[value for name in SPHERE_FITS for value in (errors[name][index], *fitted[name][index])]]
                           for index in range(count)]
        sphere_fields += [[sphere, index, *[own_fields[name][index] for name in SPHERE_FITS]]
                          for index in range(samples)]
    fit_columns = [f'{quantity}_{name}' for name in SPHERE_FITS
                   for quantity in ('position_error_mm', 'fitted_x_mm', 'fitted_y_mm', 'fitted_z_mm')]
    write_csv_lf(base / 'spheres_dipole_recovery.csv', dict(zip(
        ('sphere', 'sample_index', 'source_radius_mm', 'snr_db', *fit_columns), zip(*sphere_dipoles))))
    write_csv_lf(base / 'spheres_field_differences.csv', dict(zip(
        ('sphere', 'sample_index', *[f'relative_field_difference_{name}' for name in SPHERE_FITS]),
        zip(*sphere_fields))))

    # The arrays of the validation datasets, as each release generated them: equal digests.
    def digest(*parts):
        return dict(dtype='float64', shape=[count, 69, 100],
                    sha256=hashlib.sha256('/'.join(parts).encode()).hexdigest())

    datasets = ['inverse_headline_35dB', *[f'inverse_sweep_{level:g}dB' for level in levels],
                'patch_comparison_patch', 'patch_comparison_point']
    arrays = {dataset: {'source_pool.h5': {'positions_xyz': digest(dataset, 'pool')},
                        'test.h5': {'eeg': digest(dataset, 'test', 'eeg'),
                                    'ground_truth/position_pool_idx': digest(dataset, 'test', 'index')},
                        'train.h5': {'eeg': digest(dataset, 'train', 'eeg'),
                                     'ground_truth/eeg_signal_only': digest(dataset, 'train', 'signal')}}
              for dataset in datasets}
    for release in ('1.0.0', '1.1.0'):
        write_json(rerun / f'generated_arrays_release-{release}.json', dict(package_version=release, datasets=arrays))

    environment = dict(package='spherical_eeg_sim', platform='Windows-11',
                       software_versions=dict(spherical_eeg_sim='1.1.0', numpy='2.4.4', scipy='1.17.1', mne='1.12.1'))
    optimizer_call = dict(function='scipy.optimize.fmin_cobyla, called by mne.bem._fwd_eeg_fit_berg_scherg',
                          arguments=dict(start=[0.0, 0.0, 0.0], constraints=1, rhobeg=0.5, rhoend=1e-05, catol=1e-06),
                          evaluation_limit='none passed; the default of SciPy applies')
    # The record explains its measure of the changes in a sentence, which is no part of the tables.
    small_tolerance = dict(final_radius_m=1e-7, final_radius_of_mne_python_m=5e-5, steps=list(small_steps),
                           kernel_of_the_main_process=own_kernel,
                           fitted_position_change='distance from the fitted position of the same dipole in the '
                                                  'unchanged run of the same sphere fit in the main process',
                           ranges=small_ranges)
    write_json(base / 'summary.json', dict(
        smoke=False, head_model=geometry['variants']['default'], optimizer_call=optimizer_call,
        evaluation_limits=EVALUATION_LIMITS, instrumented_call_reproduces_installed_mne_python=True,
        instrumented_dipole_fit_reproduces_installed_mne_python=True,
        sizes=dict(dipoles=count, sweep_levels=[int(level) for level in levels], sweep_dipoles=sweep_count,
                   samples=samples, cortical=400, subcortical=80, perturbations=len(steps)),
        seed=20260703,
        dataset_array_sha256=dict(
            dipole_recovery_35db=arrays['inverse_headline_35dB']['test.h5']['eeg']['sha256'],
            snr_sweep={f'{level:g}': arrays[f'inverse_sweep_{level:g}dB']['test.h5']['eeg']['sha256']
                       for level in levels},
            patch=arrays['patch_comparison_patch']['train.h5']['ground_truth/eeg_signal_only']['sha256'],
            point=arrays['patch_comparison_point']['train.h5']['ground_truth/eeg_signal_only']['sha256']),
        arms=arms, default_sphere_model_of_mne_python=default_sphere,
        dipole_fits={name: dict(fits=count, stopped_by_themselves=count,
                                evaluations=spread(rng.integers(30, 110, count))) for name in SPHERE_FITS},
        perturbation=dict(objective_factor='1 + j * eps, eps = 2**-52', steps=list(steps), ranges=ranges),
        small_tolerance=small_tolerance,
        library_kernels=dict(variable='OPENBLAS_CORETYPE',
                             of_the_parent_process=[dict(library='openblas', version='0.3.31', kernel=own_kernel),
                                                    dict(library='openblas', version='0.3.30', kernel=own_kernel)],
                             requested=[own_kernel, *kernels], arms=kernel_record,
                             not_available={own_kernel: 'the kernel of the parent process'}),
        spheres=dict(geometry_sha256=geometry_sha256, snr_db_range=[10.0, 25.0],
                     spheres_of_the_geometry_record=list(geometry['variants']), variants=sphere_record),
        software_environment=environment))
    write_json(base / 'provenance.json', dict(commit='8' * 40, package_tree='e' * 40, working_tree_dirty=False,
                                              arguments=['--output-dir', 'sphere-fit', '--geometry', 'geometry.json']))

    # The validation rerun. The Windows run is the run at the evaluation limit above; both releases give it.
    linux = run_of(nudged('mne_limit', 0.035))
    linux_fields = field_samples(rng, 'mne_limit', samples)
    linux_sweep = {level: rng.gamma(2.0, scale, sweep_count) for level, scale in zip(levels, (5.0, 0.9, 0.17))}
    coefficients = dict(agreement=True, decimal_digits=50, tolerance=1e-12, orders=[1, 2, 3, 5, 8],
                        cases=dict(homogeneous=dict(at_order=8, max_relative_difference=2.5e-15)))
    orders = ('20', '40', '80', '100', '120')
    legendre = {'windows': {case: {order: float(value) for order, value in zip(orders, rng.uniform(1e-9, 1e-3, 5))}
                            for case in ('deep_radial_35mm', 'mid_depth_radial_55mm', 'superficial_radial_79_5mm',
                                         'superficial_tangential_79_5mm')}}
    # The relative change of every series error between the platforms. Each number of the table comes from
    # another case and order: 2e-8, 3 %, 5 % and 8 %; the 12 % belong to no number of the table.
    shift = {case: dict.fromkeys(orders, 1e-8) for case in legendre['windows']}
    shift['superficial_radial_79_5mm'].update({'20': 0.12, '100': 0.02, '120': 0.03})
    shift['superficial_tangential_79_5mm'].update({'40': 2e-8, '100': 0.025})
    shift['deep_radial_35mm']['120'] = 0.05
    shift['mid_depth_radial_55mm']['20'] = 0.08
    legendre['linux'] = {case: {order: value / (1.0 + shift[case][order]) for order, value in by_order.items()}
                         for case, by_order in legendre['windows'].items()}
    platforms = {}
    for folder, run, patch, levels_errors, nested, release in (
            ('release-1.0.0-linux-archived', linux, linux_fields, linux_sweep, False, '1.0.0'),
            ('release-1.1.0-windows', unchanged['mne_limit'], fields['mne_limit'], sweep['mne_limit'], True, '1.1.0'),
            ('release-1.0.0-windows', unchanged['mne_limit'], fields['mne_limit'], sweep['mne_limit'], True, '1.0.0')):
        software = dict(package='spherical_eeg_sim', platform='Linux' if 'linux' in folder else 'Windows-11',
                        software_versions=dict(spherical_eeg_sim=release))
        content = write_csv(rerun / folder / ('inverse/' if nested else '') / 'ecd_69ch_35db_per_sample.csv',
                            per_sample_file(sources, run))
        inverse = dict(
            method='single-dipole fit', n_test=count, seed=20260703, target_snr_db=35.0,
            metrics_csv='ecd_69ch_35db_per_sample.csv', metrics_csv_sha256=hashlib.sha256(content).hexdigest(),
            goodness_of_fit_percent=dict(median=float(np.median(run['goodness_of_fit_percent']))),
            moment_amplitude=dict(relative_error=three(run['moment_relative_error']),
                                  units='nA*m at fitted time sample'),
            orientation_error_deg=three(run['orientation_error_deg']),
            position_error_mm=three(run['position_error_mm']),
            snr_degradation_sweep=[dict(snr_db=level, n_test=sweep_count,
                                        position_error_median_mm=float(np.median(levels_errors[level])),
                                        position_error_p90_mm=float(np.percentile(levels_errors[level], 90)))
                                   for level in levels],
            software_environment=software)
        if release == '1.1.0':
            inverse['configuration'] = dict(sweep_snr_db=[int(level) for level in levels],
                                            sweep_samples_per_snr=sweep_count)
        patch_summary = dict(
            n_samples=samples, seed=20260703,
            distributed_patch=dict(n=samples, median=float(np.median(patch['patch'])),
                                   p90=float(np.percentile(patch['patch'], 90)), max=float(patch['patch'].max())),
            point_source_control=dict(n=samples, median=float(np.median(patch['point'])),
                                      p90=float(np.percentile(patch['point'], 90)), max=float(patch['point'].max())),
            patch_to_control_median_ratio=float(np.median(patch['patch']) / np.median(patch['point'])),
            software_environment=software)
        relative = legendre['linux' if 'linux' in folder else 'windows']
        convergence = dict(orders=[int(order) for order in orders], relative_error=relative,
                           config=dict(release=release), software_environment=software)
        write_json(rerun / folder / ('inverse/' if nested else '') / 'ecd_69ch_35db_summary.json', inverse)
        write_json(rerun / folder / ('patch/' if nested else '') / 'patch_forward_mne_summary.json', patch_summary)
        write_json(rerun / folder / ('coefficients/' if nested else '') / 'transfer_coefficients.json',
                   dict(coefficients, software_environment=software))
        write_json(rerun / folder / ('legendre/' if nested else '') / 'legendre_convergence.json', convergence)
        platforms[folder] = dict(inverse=inverse, patch=patch_summary, software=software)

    # Values of the tables. The fitted positions are compared as the per-dipole files give them, in metres
    # with nine decimals.
    def rounded(fitted):
        return np.array([[float(f'{value / 1000.0:.9f}') for value in row] for row in fitted]) * 1000.0

    moved = np.linalg.norm(rounded(unchanged['mne_limit']['fitted']) - rounded(linux['fitted']), axis=1)
    first, second = platforms['release-1.0.0-linux-archived'], platforms['release-1.1.0-windows']
    change = {case: {order: abs(legendre['windows'][case][order] / legendre['linux'][case][order] - 1.0)
                     for order in orders} for case in legendre['linux']}
    superficial = [case for case in change if case.startswith('superficial')]
    expected['rerun'] = dict(
        datasets=len(datasets), arrays=5 * len(datasets), dipoles=count,
        moved=dict(spread(moved), share_above_0p1_mm=float(np.mean(moved > 0.1))),
        position={key: (first['inverse']['position_error_mm'][key], second['inverse']['position_error_mm'][key])
                  for key in ('median', 'p90', 'maximum')},
        orientation=(first['inverse']['orientation_error_deg']['median'],
                     second['inverse']['orientation_error_deg']['median']),
        moment=(first['inverse']['moment_amplitude']['relative_error']['median'],
                second['inverse']['moment_amplitude']['relative_error']['median']),
        goodness=(first['inverse']['goodness_of_fit_percent']['median'],
                  second['inverse']['goodness_of_fit_percent']['median']),
        sweep=[(level, float(np.median(linux_sweep[level])), float(np.median(sweep['mne_limit'][level])))
               for level in levels],
        patch=(first['patch']['distributed_patch']['median'], second['patch']['distributed_patch']['median']),
        point=(first['patch']['point_source_control']['median'], second['patch']['point_source_control']['median']),
        ratio=(first['patch']['patch_to_control_median_ratio'], second['patch']['patch_to_control_median_ratio']),
        environments=(first['software'], second['software']),
        legendre=dict(
            cases=sorted(change), superficial=sorted(superficial),
            superficial_40_80=max(change[case][order] for case in superficial for order in ('40', '80')),
            superficial_100_120=max(change[case][order] for case in superficial for order in ('100', '120')),
            any_100_120=max(change[case][order] for case in change for order in ('100', '120')),
            others=max(change[case][order] for case in change if case not in superficial for order in orders),
            radial_100=(legendre['linux']['superficial_radial_79_5mm']['100'],
                        legendre['windows']['superficial_radial_79_5mm']['100'])))

    # Ranges over every run of one sphere fit: the unchanged run, the trials and the kernels.
    all_runs = {}
    for name in SPHERE_FITS:
        of_kernels = [kernel_record[kernel]['arms'][name] for kernel in kernels]
        own_trials = [trial for trial in trials if trial['sphere_fit'] == name]
        refits = [trial for trial in own_trials if trial['perturbed_fit'] == 'sphere_fit']

        def limits(values):
            return [float(min(values)), float(max(values))]

        position = {key: limits([arms[name]['dipole_recovery']['position_error_mm'][key],
                                 *[trial[f'position_error_{key}_mm'] for trial in own_trials],
                                 *[item['dipole_recovery']['position_error_mm'][key] for item in of_kernels]])
                    for key in ('median', 'p90', 'maximum')}
        field = {source: limits([arms[name]['patch_comparison'][source]['median'],
                                 *[trial[f'{source}_median_relative_field_difference'] for trial in refits],
                                 *[item['patch_comparison'][source]['median'] for item in of_kernels]])
                 for source in ('patch', 'point')}
        ends = [arms[name]['fit'], *[item['fit'] for item in of_kernels]]
        all_runs[name] = dict(
            recoveries=1 + len(own_trials) + len(kernels), comparisons=1 + len(refits) + len(kernels),
            position=position, field=field,
            evaluations=[min([*[fit['evaluations'] for fit in ends],
                              *[trial['parameter_fit_evaluations'] for trial in refits]]),
                         max([*[fit['evaluations'] for fit in ends],
                              *[trial['parameter_fit_evaluations'] for trial in refits]])],
            fits=1 + len(refits) + len(kernels),
            stopped=sum(fit['stopped_by_itself'] for fit in ends) + sum(
                trial['parameter_fit_stopped_by_itself'] for trial in refits))
    expected['sphere_fit'] = dict(arms=arms, ranges=ranges, kernels=kernel_record, spheres=sphere_record,
                                  all_runs=all_runs, steps=list(steps), sizes=dict(dipoles=count, samples=samples),
                                  trials=trials, optimizer_call=optimizer_call,
                                  small=dict(small_tolerance, runs=small_runs))


# --------------------------------------------------------------------------- dipole comparators, converged fit

def dipole_comparators(base, rng, study, transfer, geometry, sphere_fits, expected):
    """dipole-comparators-converged-v11: the dipole fits of three experiments with both parameter fits.

    A group is a set of windows that were fitted with one sphere: the dipole-fit subset of every dataset of
    the split study, the subset of the transfer experiment, and the same subset for every sphere of the
    adaptation experiment. These are the groups of the record and it holds no others; every group has the
    same number of windows. The fits at the limit of SciPy are the fits that those experiments recorded, in
    position and in error. The converged parameter fit moves every fitted position a little: towards the true
    position in most groups, and away from it in one group of the split study and for one sphere, so that the
    median error changes in either direction. An error is the distance of the fitted position from the true
    source position that the experiment's own file gives for the window. Each parameter fit ended as the
    sphere-fit record says for its sphere: at the limit of SciPy, or by itself with the raised limit.
    """
    spheres = expected['sphere_fit']['spheres']
    groups = [('split_study', seed, policy, 'default', study['dipole_fits'][seed, policy])
              for seed in SEEDS for policy in POLICIES]
    shared = transfer['dipole_subset']
    groups.append(('transfer', SEEDS[0], 'package_sphere_matched', 'default', shared))
    # The adaptation experiment fitted the windows of the transfer subset with every sphere.
    groups += [('adaptation', SEEDS[0], f'{sphere}_sphere_on_bem', sphere, dict(shared, fitted=fitted))
               for sphere, fitted in sphere_fits.items()]
    farther = (('split_study', SEEDS[1], 'region'), ('adaptation', SEEDS[0], 'size_matched_sphere_on_bem'))
    summary, rows, out = dict(split_study={}, transfer={}, adaptation={}), [], []
    for experiment, seed, case, sphere, subset in groups:
        truth, windows = subset['truth'], len(subset['windows'])
        outward = 0.03 if (experiment, seed, case) in farther else -0.04
        fitted = dict(mne_limit=subset['fitted'])
        fitted['raised_limit'] = subset['fitted'] + outward * (subset['fitted'] - truth) \
            + directions(rng, windows) * rng.gamma(2.0, 0.008, (windows, 1))
        errors = {name: np.linalg.norm(fitted[name] - truth, axis=1) for name in SPHERE_FITS}
        moved = np.linalg.norm(fitted['raised_limit'] - fitted['mne_limit'], axis=1)
        entry = dict(
            parameter_fit={name: {key: spheres[sphere][name]['fit'][key]
                                  for key in ('evaluations', 'stopped_by_itself', 'objective')}
                           for name in SPHERE_FITS},
            **{name: dict(windows=windows, median_mm=float(np.median(errors[name])),
                          mean_mm=float(np.mean(errors[name])), p90_mm=float(np.percentile(errors[name], 90)))
               for name in SPHERE_FITS},
            median_of_raised_limit_minus_median_of_mne_limit_mm=float(np.median(errors['raised_limit']) -
                                                                      np.median(errors['mne_limit'])),
            fitted_position_change_mm=three(moved), largest_difference_from_the_recorded_fit_mm=0.0)
        summary[experiment].setdefault(seed, {})[case] = entry
        rows += [[experiment, seed, case, window,
                  *[value for name in SPHERE_FITS for value in (errors[name][index], *fitted[name][index])]]
                 for index, window in enumerate(subset['windows'])]
        out.append(dict(experiment=experiment, dataset_seed=seed, case=case, sphere=sphere, **entry))
    write_csv_lf(base / 'comparators.csv', dict(zip(
        ('experiment', 'dataset_seed', 'case', 'window_id',
         *[f'{quantity}_{name}' for name in SPHERE_FITS
           for quantity in ('position_error_mm', 'fitted_x_mm', 'fitted_y_mm', 'fitted_z_mm')]), zip(*rows))))
    # The record names the number of windows of a fit once: every subset of the three experiments has it.
    write_json(base / 'summary.json', dict(
        smoke=False, evaluation_limits=EVALUATION_LIMITS, windows_per_subset=DIPOLE_WINDOWS,
        compared_with_the_recorded_experiments=True, spheres=geometry['variants'], **summary))
    write_json(base / 'provenance.json', dict(commit='9' * 40, package_tree='f' * 40, working_tree_dirty=False,
                                              arguments=['--output-dir', 'dipole-comparators', '--recorded', '.']))
    # The matched-sphere fit of the split study: mean and range of the dataset medians under each policy.
    split_means = {}
    for policy in POLICIES:
        split_means[policy] = {}
        for name in SPHERE_FITS:
            medians = [group[name]['median_mm'] for group in out
                       if (group['experiment'], group['case']) == ('split_study', policy)]
            split_means[policy][name] = (float(np.mean(medians)), [min(medians), max(medians)])
    changes = [group['median_of_raised_limit_minus_median_of_mne_limit_mm'] for group in out]
    expected['comparators'] = dict(groups=out, split_means=split_means, windows_per_subset=DIPOLE_WINDOWS,
                                   largest_change=max(max(changes), -min(changes)))


def build(root, seed=862026):
    """Write the miniature evidence folder into ``root`` and return the values that the tables must show."""
    root = Path(root)
    rng = np.random.default_rng(seed)
    expected = {}
    study = split_study(root / 'split-study-v11', rng, expected)
    count_control(root / 'training-count-control-v11', rng, study, expected)
    patch_study(root / 'patch-split-study-v11', rng, expected)
    transfer = head_transfer(root / 'head-transfer-v11', rng, study, expected)
    geometry, geometry_sha256, sphere_fits = head_adaptation(root / 'head-adaptation-v11', rng, study, transfer,
                                                             expected)
    benchmark(root / 'generation-benchmark-v11', rng, expected)
    small_records(root, expected)
    comparisons_with_mne(root, rng, geometry, geometry_sha256, expected)
    dipole_comparators(root / 'dipole-comparators-converged-v11', rng, study, transfer, geometry, sphere_fits,
                       expected)
    expected['settings'] = dict(
        dataset_seeds=[int(seed) for seed in SEEDS], model_seeds=[int(fit) for fit in FITS],
        windows=dict(train=480, validation=48, test=TEST_WINDOWS), n_legendre=60, window_samples=100,
        sampling_freq_hz=500.0, snr_db_range=[10.0, 25.0], montage_file='standard_1010_69.csv',
        noise_model='snr_target', reference_mode='average', eeg_normalization='per_sample_rms',
        point=dict(n_cortical_positions=800, n_subcortical_positions=150, region_count=24, buffer_mm=15.0,
                   region_rotation_seeds=[12000, 12001, 12002]),
        patch=dict(n_cortical_positions_exact=3500, n_cortical_positions_region=1160, region_count=8, buffer_mm=12.0,
                   region_rotation_seeds=[24000, 24001, 24002], diameter_range_mm=[10.0, 60.0]),
        transfer=dict(minimum_inner_skull_clearance_mm=THRESHOLD_MM, template_files=transfer['template']),
        dipole_fit_subset_windows=dict(split_study_per_dataset=ECD_WINDOWS, transfer_and_adaptation=DIPOLE_WINDOWS))
    expected['commits'] = {folder: str(number) * 40 for number, folder in enumerate(
        ('split-study-v11', 'training-count-control-v11', 'patch-split-study-v11', 'shared-position-v11',
         'head-transfer-v11', 'head-adaptation-v11', 'generation-benchmark-v11', 'sphere-fit-convergence-v11',
         'dipole-comparators-converged-v11'), 1)}
    expected['files'] = sorted(path.relative_to(root).as_posix() for path in root.rglob('*') if path.is_file())
    return expected
