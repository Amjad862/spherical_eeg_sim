"""Reproduce the paper's experiment tables and inline statistics from the archived evidence.

No model is loaded or fitted and no EEG is generated: per-window files,
recorded summaries and dataset reports are sufficient. The driver recomputes
the statistics of the localization errors from the per-window files and
compares each with the recorded summary: the median, mean and 90th percentile
of every method, fitted network, arm and stratum that the summary of a dataset
or of its dipole-fit subset records, where the archive holds the per-window
errors; the differences of medians between methods and between head models;
and the means and ranges over the datasets that the tables print. Of the two
training regimes on their common test windows it recomputes the contrasts
from the per-window files; the summaries beside those files
(same_test_windows/<policy>/summary.json), which the paper does not use, are
not read. For the comparisons with
MNE-Python it recomputes the statistics of every run, trial, library kernel
and sphere from the per-dipole and per-source files, and for the dipole
comparators that were repeated with a converged parameter fit it also
requires that the fits with the limit of SciPy are those of the recorded
experiments, window by window. It stops at the first difference, and it stops
if the records contradict a statement of the paper about them (arrays or
results that differ between the two releases, records that do not belong
together).

Read from the records without recomputation:

* statistics of the generated datasets (distances between splits, patch
  overlap, eligibility of sources in the template head), which the generator
  and the experiment scripts wrote into their reports;
* the settings of the experiments and the geometry of the template head and
  of the spheres, which must be the same in the records that share them;
* bootstrap intervals, whose paired position-cluster resampling preserves
  windows and training seeds;
* counts and fit records (windows, positions, training locations, epochs,
  optimizer evaluations and objective values) and the checks of the pairing
  with the template head other than the error statistics;
* the timings, memory and output sizes of the benchmark;
* the archived Linux results of release 1.0.0 for which the archive holds no
  per-window file: the medians of the SNR sweep, the field differences of the
  patch comparison and the two numerical checks. Of the Windows rerun, the
  medians of the sweep and the field differences must equal the sphere-fit
  record, whose per-window files are recomputed; the results of the two
  numerical checks are read and must be equal under both releases;
* the records of the earlier release in the folder ``continuity``.

``revision_tables.json`` holds the values of the experiment tables of the paper
and of its supplementary material, and ``Revision_Tables.md`` prints the main
ones. Two rows of the settings table of the supplement, the regressor and its
input, describe the experiment scripts and are not part of the records.

The evidence folder holds one subfolder per experiment: split-study-v11,
training-count-control-v11, patch-split-study-v11, shared-position-v11,
head-transfer-v11, head-adaptation-v11, generation-benchmark-v11, continuity,
validation-rerun-v11, sphere-fit-convergence-v11 and
dipole-comparators-converged-v11.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np


SEEDS = ('862', '863', '864')
FITS = ('862', '863')
POLICIES = ('position', 'region')
METHODS = ('mlp_pooled', 'dictionary', 'knn1', 'knn10', 'dipole', 'training_mean')
PANELS = ('full', 'ecd_subset')
MINIMUM_POSITIONS = 30      # a stratum with fewer distinct positions in a dataset is not averaged


def panel_methods(panel):
    """The scored methods of a panel of the split study: the dipole fit exists on the subset only."""
    return [method for method in METHODS if panel == 'ecd_subset' or method != 'dipole'] \
        + [f'mlp_{fit}' for fit in FITS]


class Evidence:
    """Reads files of the evidence folder and records the SHA-256 of each one."""

    def __init__(self, root):
        self.root = root
        self.hashes = {}
        self.checked = 0        # comparisons of a recomputed number with a recorded one that were made and passed

    def bytes(self, relative):
        content = (self.root / relative).read_bytes()
        self.hashes[Path(relative).as_posix()] = hashlib.sha256(content).hexdigest()
        return content

    def read(self, relative):
        return self.bytes(relative).decode('utf-8-sig')

    def json(self, relative):
        return json.loads(self.read(relative))

    def rows(self, relative):
        return list(csv.DictReader(self.read(relative).splitlines()))

    def arrays(self, relative):
        return np.load(io.BytesIO(self.bytes(relative)))

    def check(self, value, recorded, what, tolerance=1e-10):
        """Stop unless a recomputed number equals the recorded one."""
        if not isinstance(recorded, (int, float, np.integer, np.floating)) or isinstance(recorded, bool):
            raise ValueError(f'{what}: the record holds no number ({recorded!r})')
        if not np.isclose(value, recorded, rtol=0, atol=tolerance):
            raise ValueError(f'{what}: recomputed {value}, recorded {recorded}')
        self.checked += 1
        return float(value)

    def check_metrics(self, values, record, what):
        """Compare the median, mean and 90th percentile of an error sample with a recorded summary.

        ``values`` has one column per window. The record must hold the three
        statistics and the number of windows.
        """
        values = np.asarray(values, dtype=float)
        missing = [key for key in ('median_mm', 'mean_mm', 'p90_mm', 'windows') if key not in record]
        if missing:
            raise ValueError(f"{what}: the record lacks {', '.join(missing)}")
        for key, statistic in (('median_mm', np.median), ('mean_mm', np.mean),
                               ('p90_mm', lambda sample: np.percentile(sample, 90))):
            self.check(statistic(values), record[key], f'{what}, {key}')
        if record['windows'] != values.shape[-1]:
            raise ValueError(f"{what}: {values.shape[-1]} windows, recorded {record['windows']}")

    def check_seed_values(self, values, record, what, *, values_key, mean_key, range_key):
        """Compare one recomputed value per dataset seed, their mean and their range with a recorded headline.

        ``values`` maps the dataset seeds to the recomputed values. The record
        holds the values as a list in the order of the seeds or as a mapping,
        or not at all (``values_key`` is None).
        """
        ordered = [float(values[seed]) for seed in SEEDS]
        if values_key is not None:
            recorded = record[values_key]
            if isinstance(recorded, dict):
                if set(recorded) != set(SEEDS):
                    raise ValueError(f'{what}: values recorded for the datasets {sorted(recorded)}')
                recorded = [recorded[seed] for seed in SEEDS]
            if len(recorded) != len(ordered):
                raise ValueError(f'{what}: {len(recorded)} dataset values recorded, {len(ordered)} expected')
            for seed, value, item in zip(SEEDS, ordered, recorded):
                self.check(value, item, f'{what}, dataset {seed}')
        self.check(np.mean(ordered), record[mean_key], f'{what}, mean over the datasets')
        low, high = record[range_key]
        self.check(min(ordered), low, f'{what}, smallest dataset value')
        self.check(max(ordered), high, f'{what}, largest dataset value')


def columns(rows, names):
    """The named columns of some rows as numbers: one row per input row, also when there is none."""
    return np.array([[float(row[name]) for name in names] for row in rows], dtype=float).reshape(len(rows), len(names))


def xyz(rows, prefix=''):
    return columns(rows, [f'{prefix}{axis}_mm' for axis in 'xyz'])


def errors(rows, method):
    """Localization errors of a method: one row per fit, one column per window."""
    truth = xyz(rows)
    if method == 'mlp_pooled':
        return np.stack([np.linalg.norm(xyz(rows, f'mlp_{fit}_') - truth, axis=1)
                         for fit in FITS])
    return np.linalg.norm(xyz(rows, f'{method}_') - truth, axis=1)[None, :]


def select(rows, mask):
    return [row for row, keep in zip(rows, mask) if keep]


def band(values, low, high):
    """Members of the half-open band [low, high)."""
    values = np.asarray(values, dtype=float)
    return (values >= float(low)) & (values < float(high))


def split_stratum_mask(label, rows):
    """Windows of a split-study prediction file that belong to a stratum, by the label of the run script.

    Labels: ``all``; ``radius_<population>_<low>_<high>``; ``population_<p>``;
    ``snr_<low>_<high>``; ``distance_<low>_<high>`` (to the nearest training
    position); ``below_montage`` and ``within_montage``. Bands are half open.
    """
    if label == 'all':
        return np.ones(len(rows), dtype=bool)
    if label in ('below_montage', 'within_montage'):
        below = np.array([int(row['below_montage']) == 1 for row in rows])
        return below if label == 'below_montage' else ~below
    kind, *parts = label.split('_')
    population = np.array([row['population'] for row in rows])
    if kind == 'population' and len(parts) == 1:
        return population == parts[0]
    if kind == 'radius' and len(parts) == 3:
        return (population == parts[0]) & band(np.linalg.norm(xyz(rows), axis=1), parts[1], parts[2])
    if kind in ('snr', 'distance') and len(parts) == 2:
        name = 'snr_db' if kind == 'snr' else 'nearest_train_mm'
        return band(columns(rows, [name]).ravel(), parts[0], parts[1])
    raise ValueError(f'Unknown stratum label: {label}')


def patch_stratum_mask(label, rows):
    """Windows of a patch prediction file that belong to ``diameter_<low>_<high>`` or ``snr_<low>_<high>``."""
    kind, *parts = label.split('_')
    if kind not in ('diameter', 'snr') or len(parts) != 2:
        raise ValueError(f'Unknown stratum label: {label}')
    name = 'diameter_mm' if kind == 'diameter' else 'snr_db'
    return band(columns(rows, [name]).ravel(), parts[0], parts[1])


def population_radius_mask(label, rows):
    """Windows of a transfer or adaptation file in ``all``, ``population_<p>`` or ``radius_<p>_<low>_<high>``.

    The files give the population as 0 (cortical) or 1 (subcortical) and the
    source radius in millimetres; the band is half open.
    """
    if label == 'all':
        return np.ones(len(rows), dtype=bool)
    kind, *parts = label.split('_')
    population = np.array([row['population'] for row in rows])
    if kind == 'population' and len(parts) == 1:
        return population == parts[0]
    if kind == 'radius' and len(parts) == 3:
        return (population == parts[0]) & band(columns(rows, ['source_radius_mm']).ravel(), parts[1], parts[2])
    raise ValueError(f'Unknown stratum label: {label}')


HALVES = {'right': 'x', 'anterior': 'y', 'superior': 'z'}     # package frame; the half holds the coordinate zero


def census_stratum_mask(label, rows):
    """Windows of an eligibility file in a stratum of its record.

    Besides the labels of ``population_radius_mask`` the record has the halves
    of the head: ``right``, ``anterior`` and ``superior`` for a coordinate of
    zero or more, and the same names with ``negative_`` for the other half.
    """
    name = label[len('negative_'):] if label.startswith('negative_') else label
    if name in HALVES:
        positive = np.array([float(row[f'{HALVES[name]}_mm']) >= 0 for row in rows], dtype=bool)
        return ~positive if label != name else positive
    return population_radius_mask(label, rows)


def difference_of_medians(first, second):
    return float(np.median(first) - np.median(second))


def split_tables(evidence):
    base = Path('split-study-v11')
    headline = evidence.json(base / 'headline.json')
    # Figure 5 is drawn from the study-level summary, which must hold the dataset summaries.
    study = evidence.json(base / 'summary.json')
    summaries = {}
    training_counts, lookup_floor, paired_comparators, per_fit, distances = [], [], [], [], []
    continuity_geometry = None
    mean_offsets = {policy: {} for policy in POLICIES}
    medians = {policy: {panel: {} for panel in PANELS} for policy in POLICIES}
    windows = {}
    for seed in SEEDS:
        summaries[seed] = {}
        for policy in POLICIES:
            folder = base / seed / policy
            what = f'split study, dataset {seed}, {policy}'
            summary = evidence.json(folder / 'summary.json')
            summaries[seed][policy] = summary
            if sorted(summary['fits']) != sorted(f'mlp_{fit}' for fit in FITS):
                raise ValueError(f'{what}: the fitted networks are not those of the model seeds {FITS}')
            rows = evidence.rows(folder / 'predictions.csv')
            floor = columns(rows, ['nearest_train_mm']).ravel()
            for method in METHODS:
                if method == 'dipole':
                    continue
                values = errors(rows, method)
                lookup_floor.append(dict(dataset_seed=seed, policy=policy, method=method,
                                         below_nearest_training_fraction=float(np.mean(values < floor - 1e-8)),
                                         windows=len(rows), fits=len(values)))
            panels = dict(full=rows, ecd_subset=evidence.rows(folder / 'ecd_subset/predictions.csv'))
            windows[seed, policy] = panels
            for panel in PANELS:
                subset = panels[panel]
                # Every method of the panel and each fitted network on its own; the dipole fit has the subset only.
                for method in panel_methods(panel):
                    values = errors(subset, method)
                    evidence.check_metrics(values, summary[panel][method], f'{what}, {panel}, {method}')
                    medians[policy][panel].setdefault(method, {})[seed] = float(np.median(values))
                for method, record in summary[panel]['paired_mlp_minus_baseline'].items():
                    evidence.check(difference_of_medians(errors(subset, 'mlp_pooled'), errors(subset, method)),
                                   record['median_difference_mm'],
                                   f'{what}, {panel}, network minus {method}')
                    paired_comparators.append(dict(
                        dataset_seed=seed, policy=policy, panel=panel, comparator=method, **record,
                        interval_scope='conditional on the sampled sectors and fitted models'))
            # Every method of a panel by stratum, each fitted network on its own included. The panel of all
            # windows keeps the wording of its messages; the dipole-fit subset is named in its own.
            for panel in PANELS:
                where = what if panel == 'full' else f'{what}, {panel}'
                labels = list(summary[panel]['mlp_pooled']['strata'])
                for method in panel_methods(panel):
                    if list(summary[panel][method]['strata']) != labels:
                        raise ValueError(f'{where}, {method}: the strata are not those of the pooled network, in '
                                         'the same order')
                for label in labels:
                    members = select(panels[panel], split_stratum_mask(label, panels[panel]))
                    positions = len({row['position_id'] for row in members})
                    for method in panel_methods(panel):
                        entry = summary[panel][method]['strata'][label]
                        if positions != entry['positions']:
                            raise ValueError(f'{where}, stratum {label}: the number of distinct positions differs '
                                             'from the record')
                        evidence.check_metrics(errors(members, method), entry, f'{where}, stratum {label}, {method}')
            composition = summary['observed_split_composition']
            training_counts.append(dict(dataset_seed=seed, policy=policy,
                                        distinct_training_positions=sum(row['distinct_positions']
                                                                        for row in composition['train']),
                                        cells=composition))
            audit = evidence.json(folder / 'dataset/validation_report.json')['source_split']
            distances.append(dict(dataset_seed=seed, policy=policy,
                                  eligible=audit['eligible_position_nearest_neighbour_mm']['test_to_train'],
                                  used=audit['serialized_used_position_nearest_neighbour_mm']['test_to_train']))
            for fit in FITS:
                per_fit.append(dict(dataset_seed=seed, policy=policy, fit_seed=fit,
                                    fit=summary['fits'][f'mlp_{fit}'],
                                    metrics=summary['full'][f'mlp_{fit}']))
            training_mean = xyz(rows, 'training_mean_')
            if not np.allclose(training_mean, training_mean[0], rtol=0, atol=0):
                raise ValueError('Training-mean comparator is not constant')
            # Distance of the mean training position from the sphere centre.
            mean_offsets[policy][seed] = float(np.linalg.norm(training_mean[0]))
            if seed == '862' and policy == 'position':
                continuity_geometry = dict(training_mean_offset_mm=float(np.linalg.norm(training_mean[0])),
                                           median_test_radius_mm=float(np.median(np.linalg.norm(xyz(rows), axis=1))))
    # A dataset records every stratum of the study for which it has windows. The strata of the study are those
    # that any of its datasets records.
    known = list(dict.fromkeys(label for seed in SEEDS for policy in POLICIES for panel in PANELS
                               for label in summaries[seed][policy][panel]['mlp_pooled']['strata']))
    for (seed, policy), panels in windows.items():
        for panel in PANELS:
            recorded = summaries[seed][policy][panel]['mlp_pooled']['strata']
            missing = [label for label in known
                       if label not in recorded and split_stratum_mask(label, panels[panel]).any()]
            if missing:
                where = f'split study, dataset {seed}, {policy}' + ('' if panel == 'full' else f', {panel}')
                raise ValueError(f"{where}: the record has no stratum under a label that another record of the "
                                 "study uses and for which the dataset has windows: "
                                 f"{', '.join(missing)}")
    # Every number of a dataset summary has been compared or is read from it; the study-level summary holds the same.
    if study != summaries:
        raise ValueError('split study: the study-level summary differs from the dataset summaries')
    # The study-level headline: one median per dataset, their mean and their range.
    for policy in POLICIES:
        for panel in PANELS:
            if set(headline[policy][panel]) != set(medians[policy][panel]):
                raise ValueError(f'split study headline, {policy}, {panel}: the methods differ from the dataset '
                                 'records')
            for method, entry in headline[policy][panel].items():
                evidence.check_seed_values(medians[policy][panel][method], entry,
                                           f'split study headline, {policy}, {panel}, {method}',
                                           values_key='dataset_seed_medians_mm', mean_key='mean_median_mm',
                                           range_key='range_mm')
    strata = []
    # A stratum can exist under one policy only: no exact-position test window
    # lies 20 mm from the training data. Keep every label, in first-seen order.
    labels = list(dict.fromkeys(label for policy in POLICIES for seed in SEEDS
                                for label in summaries[seed][policy]['full']['mlp_pooled']['strata']))
    for label in labels:
        record = dict(stratum=label)
        for policy in POLICIES:
            entries = [(seed, summaries[seed][policy]['full']['mlp_pooled']['strata'].get(label))
                       for seed in SEEDS]
            valid = [(seed, item) for seed, item in entries
                     if item is not None and item['positions'] >= MINIMUM_POSITIONS]
            # The dictionary is scored on the same windows, so the same dataset seeds qualify.
            dictionary = [summaries[seed][policy]['full']['dictionary']['strata'][label] for seed, _ in valid]
            record[policy] = dict(qualifying_dataset_seeds=[seed for seed, _ in valid],
                                  mean_of_seed_medians_mm=float(np.mean([item['median_mm'] for _, item in valid]))
                                  if valid else None,
                                  dictionary_mean_of_seed_medians_mm=float(np.mean([item['median_mm']
                                                                                    for item in dictionary]))
                                  if valid else None)
        strata.append(record)
    paired_by_seed, paired_summary = paired_split_tables(evidence, base)
    return dict(table11=headline, per_fit=per_fit, distance=distances, strata=strata,
                observed_composition=training_counts, below_training_location_floor=lookup_floor,
                paired_comparators=paired_comparators, continuity_geometry=continuity_geometry,
                paired_split=paired_summary,
                paired_split_by_dataset_seed=paired_by_seed,
                training_mean_offset_mm=mean_offsets,
                adequacy=evidence.json(base / 'region_adequacy_preflight.json'))


def paired_split_tables(evidence, base):
    """The two training regimes on the exact-position test windows that lie in regional test sectors.

    The contrast is the median error of the region-trained networks minus that
    of the networks trained under exact-position holdout, both pooled over the
    model seeds, per dataset seed and over the three datasets together.
    """
    paired_by_seed = {}
    pooled = {'position': [], 'region': []}
    exact_rows = []
    for seed in SEEDS:
        record = evidence.json(base / seed / 'same_test_windows/paired_mlp.json')
        regimes = {policy: evidence.rows(base / seed / f'same_test_windows/{policy}/predictions.csv')
                   for policy in POLICIES}
        if [row['window'] for row in regimes['position']] != [row['window'] for row in regimes['region']]:
            raise ValueError('The two training regimes were scored on different windows')
        values = {policy: errors(rows, 'mlp_pooled') for policy, rows in regimes.items()}
        what = f'paired split contrast, dataset {seed}'
        evidence.check(difference_of_medians(values['region'], values['position']),
                       record['region_minus_position_mm'], what)
        if record['windows'] != len(regimes['position']) or record['distinct_positions'] != len(
                {row['position_id'] for row in regimes['position']}):
            raise ValueError(f'{what}: windows or positions differ from the record')
        for policy in POLICIES:
            pooled[policy].append(values[policy])
        exact_rows.append(regimes['position'])
        paired_by_seed[seed] = record
    summary = evidence.json(base / 'paired_split_summary.json')
    # Every stratum of the pooled contrast. A window belongs to a stratum by its exact-position record, and
    # positions are counted within each dataset.
    for label, entry in summary['strata'].items():
        masks = [split_stratum_mask(label, rows) for rows in exact_rows]
        region, position = (np.concatenate([block[:, mask] for block, mask in zip(pooled[policy], masks)], axis=1)
                            for policy in ('region', 'position'))
        what = 'paired split contrast, three datasets' + ('' if label == 'all' else f', stratum {label}')
        if entry['windows'] != position.shape[1]:
            raise ValueError(f"paired split contrast{'' if label == 'all' else f', stratum {label}'}: "
                             'the number of windows differs from the record')
        if entry['seed_position_clusters'] != sum(len({row['position_id'] for row in select(rows, mask)})
                                                  for rows, mask in zip(exact_rows, masks)):
            raise ValueError(f"paired split contrast{'' if label == 'all' else f', stratum {label}'}: "
                             'the number of positions differs from the record')
        evidence.check(difference_of_medians(region, position), entry['region_minus_position_mm'], what)
    if 'all' not in summary['strata']:
        raise ValueError('paired split contrast: the record lacks the contrast over all windows')
    return paired_by_seed, summary


def directional_count(rows, axis, positive, cortical_only):
    selected = [row for row in rows if (float(row[f'{axis}_mm']) >= 0) == positive
                and (not cortical_only or row['population'] == '0')]
    kept = sum(row['eligible'].lower() == 'true' for row in selected)
    return dict(eligible_windows=kept, total_windows=len(selected),
                fraction=kept / len(selected) if selected else None)


def _in_band(name, low, high):
    return lambda row, clearance: low <= float(row[name]) < high


# Strata of the transfer errors: name, family, and test on a paired-prediction
# row and its clearance from the inner skull in millimetres. Every window must
# fall into exactly one stratum of each family.
TRANSFER_STRATA = (
    ('population_0', 'population', lambda row, clearance: row['population'] == '0'),
    ('population_1', 'population', lambda row, clearance: row['population'] == '1'),
    ('radius_30_45', 'radius', _in_band('source_radius_mm', 30, 45)),
    ('radius_45_55', 'radius', _in_band('source_radius_mm', 45, 55)),
    ('radius_55_65', 'radius', _in_band('source_radius_mm', 55, 65.001)),
    ('radius_70.5_75', 'radius', _in_band('source_radius_mm', 70.5, 75)),
    ('radius_75_79.5', 'radius', _in_band('source_radius_mm', 75, 79.501)),
    ('snr_10_15', 'snr', _in_band('snr_db_referenced', 10, 15)),
    ('snr_15_20', 'snr', _in_band('snr_db_referenced', 15, 20)),
    ('snr_20_25', 'snr', _in_band('snr_db_referenced', 20, 25.001)),
    ('clearance_5_10', 'clearance', lambda row, clearance: 5 <= clearance < 10),
    ('clearance_10_20', 'clearance', lambda row, clearance: 10 <= clearance < 20),
    ('clearance_20_inf', 'clearance', lambda row, clearance: clearance >= 20),
)


def census_radius(row):
    return float(np.linalg.norm([float(row[f'{axis}_mm']) for axis in 'xyz']))


def is_cortical(row):
    return row['population'] == '0'


def _radius_band(low, high):
    return lambda row: low <= census_radius(row) < high


def _quadrant(superior, anterior):
    return lambda row: (is_cortical(row) and (float(row['z_mm']) >= 0) == superior
                        and (float(row['y_mm']) >= 0) == anterior)


# Rows of the eligibility census: name, family, and test on a row of
# eligibility.csv. Package frame: anterior is y >= 0, superior is z >= 0. The
# strata of the population and radius families must each hold every window
# once; the four quadrants partition the cortical windows by construction.
CENSUS_STRATA = (
    ('all', 'all', lambda row: True),
    ('cortical', 'population', is_cortical),
    ('subcortical', 'population', lambda row: row['population'] == '1'),
    ('radius_30_45', 'radius', _radius_band(30, 45)),
    ('radius_45_55', 'radius', _radius_band(45, 55)),
    ('radius_55_65', 'radius', _radius_band(55, 65.001)),
    ('radius_70.5_75', 'radius', _radius_band(70.5, 75)),
    ('radius_75_79.5', 'radius', _radius_band(75, 79.501)),
    ('cortical_superior_posterior', 'quadrant', _quadrant(True, False)),
    ('cortical_superior_anterior', 'quadrant', _quadrant(True, True)),
    ('cortical_inferior_posterior', 'quadrant', _quadrant(False, False)),
    ('cortical_inferior_anterior', 'quadrant', _quadrant(False, True)),
)


def eligibility_census(census_by_seed):
    """Eligible windows over all windows by stratum, per dataset seed and pooled."""
    rows = []
    family_totals = {}
    for name, family, test in CENSUS_STRATA:
        by_seed, kept_all, total_all = {}, 0, 0
        for seed in SEEDS:
            members = [row for row in census_by_seed[seed] if test(row)]
            kept = sum(row['eligible'].lower() == 'true' for row in members)
            by_seed[seed] = dict(eligible_windows=kept, total_windows=len(members))
            kept_all += kept
            total_all += len(members)
            family_totals[family, seed] = family_totals.get((family, seed), 0) + len(members)
        rows.append(dict(stratum=name, by_dataset_seed=by_seed, eligible_windows=kept_all,
                         total_windows=total_all, fraction=kept_all / total_all if total_all else None))
    # Cortical sources that lie outside the inner skull (negative clearance).
    by_seed, outside_all, total_all = {}, 0, 0
    for seed in SEEDS:
        members = [row for row in census_by_seed[seed] if is_cortical(row)]
        for family in ('population', 'radius'):
            if family_totals[family, seed] != len(census_by_seed[seed]):
                raise ValueError(f'Eligibility census, dataset {seed}: the {family} strata do not hold every window '
                                 'once')
        outside = sum(float(row['inner_skull_clearance_mm']) < 0 for row in members)
        by_seed[seed] = dict(outside_windows=outside, total_windows=len(members))
        outside_all += outside
        total_all += len(members)
    outside_record = dict(by_dataset_seed=by_seed, outside_windows=outside_all, total_windows=total_all,
                          fraction=outside_all / total_all if total_all else None)
    return rows, outside_record


def transfer_window_strata(paired_by_seed, clearance_by_seed):
    """Sphere and boundary-element medians of the frozen networks by stratum.

    Model seeds are pooled within a dataset seed; the table averages the three
    dataset-seed medians. A stratum is averaged only if every dataset seed has
    at least 30 distinct source positions in it; otherwise it is reported with
    empty values. Every window must fall into exactly one stratum of each
    family.
    """
    strata = []
    family_totals = {}
    for name, family, test in TRANSFER_STRATA:
        sphere, bem, windows, positions = [], [], 0, []
        for seed in SEEDS:
            members = [row for row in paired_by_seed[seed] if test(row, clearance_by_seed[seed][row['window_id']])]
            windows += len(members)
            family_totals[family, seed] = family_totals.get((family, seed), 0) + len(members)
            positions.append(len({row['position_pool_id'] for row in members}))
            if members:
                sphere.append(float(np.median(columns(members, [f'mlp_{fit}_package_sphere_error_mm'
                                                                for fit in FITS]))))
                bem.append(float(np.median(columns(members, [f'mlp_{fit}_bem_error_mm' for fit in FITS]))))
        averaged = min(positions) >= MINIMUM_POSITIONS
        strata.append(dict(
            stratum=name, windows=windows, smallest_number_of_positions_in_a_dataset=min(positions),
            package_sphere_mean_of_seed_medians_mm=float(np.mean(sphere)) if averaged else None,
            bem_mean_of_seed_medians_mm=float(np.mean(bem)) if averaged else None,
            mean_difference_mm=float(np.mean(np.array(bem) - np.array(sphere))) if averaged else None))
    for (family, seed), total in family_totals.items():
        if total != len(paired_by_seed[seed]):
            raise ValueError(f'Transfer strata, dataset {seed}: the {family} strata hold {total} of '
                             f'{len(paired_by_seed[seed])} windows')
    return strata


def transfer_tables(evidence):
    base = Path('head-transfer-v11')
    summary = evidence.json(base / 'summary.json')
    directional, shifts, all_eligibility, independent = [], [], [], []
    headline_values = {name: {} for name in ('package_sphere', 'bem', 'difference')}
    eligibility, gates = {}, {}
    census_by_seed, paired_by_seed, clearance_by_seed = {}, {}, {}
    for seed in SEEDS:
        census = evidence.rows(base / seed / 'eligibility.csv')
        rows = evidence.rows(base / seed / 'paired_predictions.csv')
        by_window = {row['window_id']: row for row in census}
        if any(by_window[row['window_id']]['eligible'].lower() != 'true' for row in rows):
            raise ValueError('An ineligible transfer window was evaluated')
        census_by_seed[seed], paired_by_seed[seed] = census, rows
        clearance_by_seed[seed] = {row['window_id']: float(row['inner_skull_clearance_mm']) for row in census}
        all_eligibility.extend(census)
        eligibility[seed] = evidence.json(base / seed / 'eligibility.json')
        threshold = eligibility[seed]['minimum_inner_skull_clearance_mm']
        if any((float(row['inner_skull_clearance_mm']) >= threshold) != (row['eligible'].lower() == 'true')
               for row in census):
            raise ValueError('Eligibility does not follow the recorded clearance threshold')
        gates[seed] = evidence.json(base / seed / 'adapter_gates.json')
        record = summary['by_dataset_seed'][seed]
        # Sphere and boundary-element medians of the pooled networks, and their difference, by stratum.
        for label, entry in record['paired_pooled_model_seeds'].items():
            members = select(rows, population_radius_mask(label, rows))
            what = f'transfer, dataset {seed}, stratum {label}'
            pooled_medians = {head: evidence.check(
                np.median(columns(members, [f'mlp_{fit}_{head}_error_mm' for fit in FITS])),
                entry[f'{head}_median_mm'], f'{what}, {head}') for head in ('package_sphere', 'bem')}
            difference = evidence.check(pooled_medians['bem'] - pooled_medians['package_sphere'],
                                        entry['median_difference_mm'], f'{what}, difference of the medians')
            if entry['windows'] != len(members) or entry['position_clusters'] != len(
                    {row['position_pool_id'] for row in members}):
                raise ValueError(f'{what}: windows or positions differ from the record')
            if label == 'all':
                for name, value in (*pooled_medians.items(), ('difference', difference)):
                    headline_values[name][seed] = value
        # Every scored method by stratum, and the dipole-fit subset.
        for label, by_method in record['strata'].items():
            members = select(rows, population_radius_mask(label, rows))
            for method, entry in by_method.items():
                evidence.check_metrics(columns(members, [f'{method}_error_mm']).ravel(), entry,
                                       f'transfer, dataset {seed}, stratum {label}, {method}')
        # The dipole fits were made for one dataset only; the others record no subset.
        subset = evidence.rows(base / seed / 'ecd_subset.csv') if record.get('ecd_subset') else []
        for method, entry in (record.get('ecd_subset') or {}).items():
            what = f'transfer, dataset {seed}, dipole-fit subset, {method}'
            if method == 'ecd_matched_bem_minus_sphere':
                # Dipole fits with the model that generated the windows: boundary-element model minus sphere.
                evidence.check(difference_of_medians(columns(subset, ['ecd_bem_matched_error_mm']),
                                                     columns(subset, ['ecd_package_sphere_matched_error_mm'])),
                               entry['median_difference_mm'], what)
            else:
                evidence.check_metrics(columns(subset, [f'{method}_error_mm']).ravel(), entry, what)
        # Check of the pairing: each network on the package sphere and on the independent sphere implementation.
        exact = evidence.rows(Path('split-study-v11') / seed / 'position/predictions.csv')
        for fit in FITS:
            gate = gates[seed]['check5_model'][f'mlp_{fit}']
            what = f'transfer, dataset {seed}, pairing check of network {fit}'
            evidence.check_metrics(errors(exact, f'mlp_{fit}'), gate['package_sphere_all'], f'{what}, all windows')
            medians = {}
            for head in ('package_sphere', 'independent_sphere'):
                values = columns(rows, [f'mlp_{fit}_{head}_error_mm']).ravel()
                evidence.check_metrics(values, gate[f'{head}_eligible'], f'{what}, {head}')
                medians[head] = float(np.median(values))
            evidence.check(medians['independent_sphere'] - medians['package_sphere'], gate['median_difference_mm'],
                           f'{what}, difference of the medians')
            shift = np.linalg.norm(xyz(rows, f'mlp_{fit}_independent_sphere_')
                                   - xyz(rows, f'mlp_{fit}_package_sphere_'), axis=1)
            for key, value in (('median', np.median(shift)), ('p90', np.percentile(shift, 90)),
                               ('p99', np.percentile(shift, 99)), ('maximum', shift.max())):
                evidence.check(value, gate['per_window_prediction_shift_mm'][key], f'{what}, prediction shift, {key}')
        independent.append(float(np.median(columns(rows, [f'mlp_{fit}_independent_sphere_error_mm'
                                                         for fit in FITS]))))
        sphere = columns(rows, [f'mlp_{fit}_package_sphere_error_mm' for fit in FITS])
        bem = columns(rows, [f'mlp_{fit}_bem_error_mm' for fit in FITS])
        displacement = np.stack([np.linalg.norm(xyz(rows, f'mlp_{fit}_bem_') -
                                                xyz(rows, f'mlp_{fit}_package_sphere_'), axis=1)
                                 for fit in FITS], axis=1)
        shifts.append(dict(dataset_seed=seed, windows=len(rows), fits=len(FITS),
                           fraction_bem_error_larger=float(np.mean(bem > sphere)),
                           median_prediction_displacement_mm=float(np.median(displacement)),
                           median_of_window_error_differences_mm=float(np.median(bem - sphere))))
        for axis, positive, direction in (('y', True, 'anterior'), ('y', False, 'posterior'),
                                          ('z', True, 'superior'), ('z', False, 'inferior')):
            mask = np.array([(float(by_window[row['window_id']][f'{axis}_mm']) >= 0) == positive
                             and row['population'] == '0' for row in rows])
            directional.append(dict(dataset_seed=seed, direction=direction,
                                    all=directional_count(census, axis, positive, False),
                                    cortical=directional_count(census, axis, positive, True),
                                    eligible_cortical_difference_of_medians_mm=float(np.median(bem[mask]) -
                                                                                   np.median(sphere[mask]))))
    pooled = {}
    for axis, positive, direction in (('y', True, 'anterior'), ('y', False, 'posterior'),
                                      ('z', True, 'superior'), ('z', False, 'inferior')):
        pooled[direction] = dict(all=directional_count(all_eligibility, axis, positive, False),
                                 cortical=directional_count(all_eligibility, axis, positive, True))
    strata = []
    for key in summary['by_dataset_seed']['862']['paired_pooled_model_seeds']:
        entries = [(seed, summary['by_dataset_seed'][seed]['paired_pooled_model_seeds'].get(key))
                   for seed in SEEDS]
        valid = [(seed, item) for seed, item in entries
                 if item is not None and item['position_clusters'] >= MINIMUM_POSITIONS]
        strata.append(dict(stratum=key, qualifying_dataset_seeds=[seed for seed, _ in valid],
                           windows=sum(item['windows'] for _, item in valid),
                           **{name: float(np.mean([item[name] for _, item in valid])) if valid else None
                              for name in ('package_sphere_median_mm', 'bem_median_mm', 'median_difference_mm')}))
    census_rows, outside = eligibility_census(census_by_seed)
    for name, key in (('package_sphere', 'package_sphere_median_mm'), ('bem', 'bem_median_mm'),
                      ('difference', 'median_difference_mm')):
        evidence.check_seed_values(headline_values[name], summary['paired_headline'][key], f'transfer headline, {key}',
                                   values_key=None, mean_key='mean_of_dataset_seed_medians', range_key='range')
    # The eligibility record of every dataset must agree with a count of its windows, stratum by stratum.
    for seed in SEEDS:
        census = [dict(row, source_radius_mm=census_radius(row)) for row in census_by_seed[seed]]
        for label, recorded in eligibility[seed]['strata'].items():
            members = select(census, census_stratum_mask(label, census))
            eligible = [row for row in members if row['eligible'].lower() == 'true']
            counted = dict(windows=len(members), eligible_windows=len(eligible),
                           positions=len({row['position_pool_id'] for row in members}),
                           eligible_positions=len({row['position_pool_id'] for row in eligible}))
            if any(recorded[key] != value for key, value in counted.items()):
                raise ValueError(f'transfer, dataset {seed}: the eligibility record of {label} differs from its '
                                 'windows')
    return dict(summary=summary, eligibility=eligibility, gates=gates, strata=strata,
                window_strata=transfer_window_strata(paired_by_seed, clearance_by_seed),
                eligibility_census=census_rows, cortical_outside_inner_skull=outside,
                independent_sphere_mean_seed_median_mm=float(np.mean(independent)),
                independent_sphere_seed_medians_mm=dict(zip(SEEDS, independent)),
                directional_eligibility=directional, pooled_directional_eligibility=pooled,
                paired_window_statistics=shifts,
                direction_frame='package coordinates, anterior y>=0 and superior z>=0',
                estimand='difference between medians on the same windows, pooled over two model seeds within '
                         'dataset seed')


def seed_spread(values):
    return dict(dataset_seed_values=[float(value) for value in values], mean=float(np.mean(values)),
                range=[float(np.min(values)), float(np.max(values))])


def common_window_errors(rows, common_windows, region_rows, model_seed):
    """Errors of the count control on the windows that both training regimes of the split study scored.

    ``rows`` are the rows of errors.csv, one per exact-position test window;
    ``common_windows`` the indices of the windows in regional test sectors;
    ``region_rows`` the predictions of the region-trained networks on those
    windows, in the same order.
    """
    by_window = {int(row['window']): row for row in rows}
    wanted = [int(window) for window in common_windows]
    if len(set(wanted)) != len(wanted) or any(window not in by_window for window in wanted):
        raise ValueError('The count control does not hold every common test window once')
    members = [by_window[window] for window in wanted]
    if [row['position_id'] for row in members] != [row['position_id'] for row in region_rows]:
        raise ValueError('The count control and the split study disagree on the common test windows')
    regional = np.linalg.norm(xyz(region_rows, f'mlp_{model_seed}_') - xyz(region_rows), axis=1)
    return dict(exact=columns(members, ['exact_error_mm']).ravel(),
                count_matched=columns(members, ['count_matched_error_mm']).ravel(), regional=regional)


def count_control_tables(evidence):
    base = Path('training-count-control-v11')
    per_seed = {}
    by_panel = {'all_test_windows': {}, 'same_test_windows': {}}
    contrasts = {'all_test_windows': (('count_matched_minus_exact', 'count_matched', 'exact'),),
                 'same_test_windows': (('count_matched_minus_exact', 'count_matched', 'exact'),
                                       ('regional_minus_count_matched', 'regional', 'count_matched'))}
    for seed in SEEDS:
        summary = evidence.json(base / seed / 'summary.json')
        rows = evidence.rows(base / seed / 'errors.csv')
        what = f'count control, dataset {seed}'
        split = Path('split-study-v11') / seed / 'same_test_windows'
        samples = {
            'all_test_windows': {name: columns(rows, [f'{name}_error_mm']).ravel()
                                 for name in ('exact', 'count_matched')},
            'same_test_windows': common_window_errors(
                rows, evidence.json(split / 'exact_position_test_rows.json'),
                evidence.rows(split / 'region/predictions.csv'), summary['model_seed'])}
        for panel, by_name in samples.items():
            for name, values in by_name.items():
                evidence.check_metrics(values, summary[panel][name], f'{what}, {panel}, {name}')
                by_panel[panel].setdefault(name, {})[seed] = float(np.median(values))
            for name, first, second in contrasts[panel]:
                by_panel[panel].setdefault(name, {})[seed] = evidence.check(
                    difference_of_medians(by_name[first], by_name[second]),
                    summary[panel][name]['median_difference_mm'], f'{what}, {panel}, {name}')
        per_seed[seed] = summary
    headline = evidence.json(base / 'headline.json')
    if set(headline) != set(by_panel):
        raise ValueError('count control headline: the panels differ from the dataset records')
    for panel, entries in headline.items():
        if set(entries) != set(by_panel[panel]):
            raise ValueError(f'count control headline, {panel}: the entries differ from the dataset records')
        for name, entry in entries.items():
            evidence.check_seed_values(by_panel[panel][name], entry, f'count control headline, {panel}, {name}',
                                       values_key='dataset_seed_values', mean_key='mean', range_key='range')
    return dict(headline=headline, by_dataset_seed=per_seed)


def patch_tables(evidence):
    base = Path('patch-split-study-v11')
    methods = ('mlp_pooled', 'nearest_training_patch', 'ten_nearest_training_patches', 'training_mean',
               *[f'mlp_{fit}' for fit in FITS])
    per_seed, audits = {}, []
    medians = {policy: {} for policy in POLICIES}
    for seed in SEEDS:
        per_seed[seed] = {}
        for policy in POLICIES:
            folder = base / seed / policy
            what = f'patch study, dataset {seed}, {policy}'
            summary = evidence.json(folder / 'summary.json')
            if sorted(summary['fits']) != sorted(f'mlp_{fit}' for fit in FITS):
                raise ValueError(f'{what}: the fitted networks are not those of the model seeds {FITS}')
            rows = evidence.rows(folder / 'predictions.csv')
            for method in methods:
                values = errors(rows, method)
                evidence.check_metrics(values, summary['methods'][method], f'{what}, {method}')
                medians[policy].setdefault(method, {})[seed] = float(np.median(values))
            for method, record in summary['methods']['paired_mlp_minus_lookup'].items():
                evidence.check(difference_of_medians(errors(rows, 'mlp_pooled'), errors(rows, method)),
                               record['median_difference_mm'], f'{what}, network minus {method}')
            labels = list(summary['methods']['mlp_pooled']['strata'])
            for method in methods:
                if list(summary['methods'][method]['strata']) != labels:
                    raise ValueError(f'{what}, {method}: the strata are not those of the pooled network, in the '
                                     'same order')
                for label, entry in summary['methods'][method]['strata'].items():
                    evidence.check_metrics(errors(select(rows, patch_stratum_mask(label, rows)), method),
                                           entry, f'{what}, stratum {label}, {method}')
            floor = columns(rows, ['nearest_training_centre_mm']).ravel()
            below = float(np.mean(errors(rows, 'mlp_pooled') < floor))
            if not np.isclose(below, summary['mlp_error_below_nearest_training_centre'], rtol=0, atol=1e-12):
                raise ValueError('Patch share below the nearest training centre differs from the record')
            per_seed[seed][policy] = summary
            pair = summary['audit']['pairs']['test_to_train']
            # Smallest distance between positions that different splits used,
            # from the dataset's own report; defined under both policies.
            used = evidence.json(folder / 'dataset/validation_report.json')['source_split'][
                'serialized_used_position_nearest_neighbour_mm']
            used_distance = min(entry['all']['min'] for entry in used.values())
            recorded = summary['audit'].get('minimum_support_clearance_mm')
            if recorded is not None and not np.isclose(recorded, used_distance, rtol=0, atol=1e-6):
                raise ValueError('The study record and the dataset report disagree on the support clearance')
            audits.append(dict(
                dataset_seed=seed, policy=policy, scored_test_patches=pair['scored_patches'],
                best_overlap=pair['best_iou'], nearest_centre_euclidean_mm=pair['nearest_centre_euclidean_mm'],
                minimum_footprint_gap_mm=pair['minimum_footprint_gap_mm'],
                minimum_support_clearance_mm=recorded,
                minimum_used_position_distance_mm=float(used_distance),
                containment=summary['audit'].get('patch_containment'),
                distinct_training_centres=summary['distinct_training_centres']))
    # The table of the strata has one row per stratum and one value per dataset: every dataset has every stratum.
    for seed in SEEDS:
        for policy in POLICIES:
            if list(per_seed[seed][policy]['methods']['mlp_pooled']['strata']) != list(
                    per_seed['862']['position']['methods']['mlp_pooled']['strata']):
                raise ValueError(f'patch study, dataset {seed}, {policy}: the strata are not those of the first '
                                 'dataset, in the same order')
    strata = []
    for label in per_seed['862']['position']['methods']['mlp_pooled']['strata']:
        record = dict(stratum=label)
        for policy in POLICIES:
            record[policy] = seed_spread([per_seed[seed][policy]['methods']['mlp_pooled']['strata'][label]['median_mm']
                                          for seed in SEEDS])
        strata.append(record)
    # The study-level headline: one median per dataset, their mean and their range. The overlap of the test
    # patches with the training patches is a statistic of the datasets; it must agree with the dataset records.
    headline = evidence.json(base / 'headline.json')
    overlap_key = 'median_best_test_to_train_overlap'
    for policy in POLICIES:
        if set(headline[policy]) != {*medians[policy], overlap_key}:
            raise ValueError(f'patch study headline, {policy}: the methods differ from the dataset records')
        for method, by_seed in medians[policy].items():
            evidence.check_seed_values(by_seed, headline[policy][method], f'patch study headline, {policy}, {method}',
                                       values_key='dataset_seed_medians_mm', mean_key='mean_median_mm',
                                       range_key='range_mm')
        recorded = [per_seed[seed][policy]['audit']['pairs']['test_to_train']['best_iou']['median'] for seed in SEEDS]
        overlap = headline[policy][overlap_key]
        if not np.allclose([*overlap['dataset_seed_values'], overlap['mean'], *overlap['range']],
                           [*recorded, np.mean(recorded), min(recorded), max(recorded)], rtol=0, atol=1e-12):
            raise ValueError(f'patch study headline, {policy}: the overlap differs from the dataset records')
    return dict(headline=headline, audits=audits, strata=strata,
                adequacy=evidence.json(base / 'region_adequacy_preflight.json'),
                below_nearest_training_centre={
                    policy: seed_spread([per_seed[seed][policy]['mlp_error_below_nearest_training_centre']
                                         for seed in SEEDS]) for policy in POLICIES},
                paired_mlp_minus_lookup={seed: {policy: per_seed[seed][policy]['methods']['paired_mlp_minus_lookup']
                                                for policy in POLICIES} for seed in SEEDS},
                fits={seed: {policy: per_seed[seed][policy]['fits'] for policy in POLICIES} for seed in SEEDS})


def adaptation_tables(evidence):
    base = Path('head-adaptation-v11')
    summary = evidence.json(base / 'summary.json')
    by_seed = summary['by_dataset_seed']
    arms = list(by_seed['862']['arms'])
    if any(sorted(by_seed[seed]['arms']) != sorted(arms) for seed in SEEDS):
        raise ValueError('adaptation: the arms differ between the datasets')
    own_sphere = {variant: {} for variant in summary['own_sphere_eligible_headline']}
    if any(set(by_seed[seed]['own_sphere']) != set(own_sphere) for seed in SEEDS):
        raise ValueError('adaptation headline: the sphere variants differ from the dataset records')
    if evidence.json(base / 'geometry.json') != summary['geometry']:
        raise ValueError('adaptation: the geometry of the summary differs from the geometry record')
    arm_medians, arm_differences = {arm: {} for arm in arms}, {}

    def arm_errors(members, arm):
        return columns(members, [f'{arm}_mlp_{fit}_error_mm' for fit in FITS]).T

    def check_arm(members, arm, entry, what):
        """Statistics of an arm, and its difference of medians from the default sphere on the same windows."""
        values = arm_errors(members, arm)
        evidence.check_metrics(values, entry, what)
        if 'median_difference_from_default_mm' not in entry:
            return float(np.median(values)), None
        return float(np.median(values)), evidence.check(
            difference_of_medians(values, arm_errors(members, 'default')),
            entry['median_difference_from_default_mm'], f'{what}, difference from the default sphere')

    for seed in SEEDS:
        rows = evidence.rows(base / seed / 'adaptation_predictions.csv')
        if len(rows) != by_seed[seed]['bem_windows']:
            raise ValueError('Adaptation prediction file has an unexpected number of windows')
        for arm in arms:
            arm_medians[arm][seed], difference = check_arm(rows, arm, by_seed[seed]['arms'][arm],
                                                           f'adaptation, dataset {seed}, {arm}')
            if difference is not None:
                arm_differences.setdefault(arm, {})[seed] = difference
        for label, by_arm in by_seed[seed]['strata'].items():
            members = select(rows, population_radius_mask(label, rows))
            for arm, entry in by_arm.items():
                check_arm(members, arm, entry, f'adaptation, dataset {seed}, stratum {label}, {arm}')
        # Each fine-tuned network and each network trained on the realistic head alone, on its own.
        networks = [(str(entry['arm']), str(entry['size']), str(entry['model_seed']))
                    for entry in by_seed[seed]['finetune']]
        expected = [(arm.split('_', 1)[0], arm.split('_', 1)[1], fit) for arm in arms
                    if arm.startswith(('finetuned_', 'scratch_')) for fit in FITS]
        if sorted(networks) != sorted(expected):
            raise ValueError(f'adaptation, dataset {seed}: the single networks are not those of the arms')
        if any(str(entry.get('dataset_seed')) != seed for entry in by_seed[seed]['finetune']):
            raise ValueError(f'adaptation, dataset {seed}: the dataset seed of a row of a single network is not '
                             f'written as {seed}')
        for entry in by_seed[seed]['finetune']:
            name = f"{entry['arm']}_{entry['size']}"
            values = columns(rows, [f"{name}_mlp_{entry['model_seed']}_error_mm"]).ravel()
            for key, statistic in (('median_mm', np.median), ('mean_mm', np.mean),
                                   ('p90_mm', lambda sample: np.percentile(sample, 90))):
                evidence.check(statistic(values), entry[key],
                               f"adaptation, dataset {seed}, {name}, network {entry['model_seed']}, {key}")
        # Each sphere variant on windows of its own sphere, for the sources eligible in the template.
        predictions = evidence.arrays(base / seed / 'predictions.npz')
        truth = predictions['true_position_mm']
        for variant in own_sphere:
            values = np.stack([np.linalg.norm(predictions[f'{variant}_own_sphere_mlp_{fit}'] - truth, axis=1)
                               for fit in FITS])
            record = by_seed[seed]['own_sphere'][variant]['eligible_windows']
            evidence.check_metrics(values, record, f'adaptation, dataset {seed}, {variant} on its own sphere')
            own_sphere[variant][seed] = float(np.median(values))
    # The study-level headlines: one median per dataset, their mean and their range.
    for variant, medians in own_sphere.items():
        evidence.check_seed_values(medians, summary['own_sphere_eligible_headline'][variant],
                                   f'adaptation headline, {variant} on its own sphere', values_key=None,
                                   mean_key='mean_of_dataset_seed_medians', range_key='range')
    if set(summary['bem_windows_headline']) != set(arms):
        raise ValueError('adaptation headline: the arms differ from the dataset records')
    for arm in arms:
        entry = summary['bem_windows_headline'][arm]
        evidence.check_seed_values(arm_medians[arm], entry, f'adaptation headline, {arm}',
                                   values_key='dataset_seed_medians_mm', mean_key='mean_of_dataset_seed_medians',
                                   range_key='range')
        if ('mean_difference_from_default_mm' in entry) != (arm in arm_differences):
            raise ValueError(f'adaptation headline, {arm}: a difference from the default sphere is missing')
        if arm in arm_differences:
            evidence.check(np.mean([arm_differences[arm][seed] for seed in SEEDS]),
                           entry['mean_difference_from_default_mm'],
                           f'adaptation headline, {arm}, mean difference from the default sphere')
    headline = {}
    for arm in arms:
        entries = [by_seed[seed]['arms'][arm] for seed in SEEDS]
        headline[arm] = seed_spread([entry['median_mm'] for entry in entries])
        if 'median_difference_from_default_mm' in entries[0]:
            headline[arm]['difference_from_default_mm'] = seed_spread(
                [entry['median_difference_from_default_mm'] for entry in entries])
            headline[arm]['difference_ci95_mm_by_seed'] = [entry['median_difference_ci95_mm'] for entry in entries]
    dipole_rows = evidence.rows(base / '862' / 'dipole_subset.csv')
    dipole = summary['dipole_subset']
    for key, entry in dipole.items():
        values = columns(dipole_rows, [f'{key}_error_mm']).ravel()
        evidence.check_metrics(values, entry, f'adaptation, dipole fits, {key}')
        if 'median_difference_from_bem_matched_mm' in entry:
            # The reference is the dipole fit with the boundary-element model that generated the windows.
            reference = columns(dipole_rows, ['dipole_bem_on_bem_recorded_error_mm']).ravel()
            evidence.check(difference_of_medians(values, reference), entry['median_difference_from_bem_matched_mm'],
                           f'adaptation, dipole fits, {key}, difference from the matched model')
    sizes = []
    for arm in arms:
        if arm.startswith('finetuned_'):
            label = arm.split('_', 1)[1]
            windows = [int(row['bem_training_windows']) for seed in SEEDS for row in by_seed[seed]['finetune']
                       if str(row['size']) == label]
            sizes.append(dict(size=label, bem_training_windows=sorted(set(windows)),
                              finetuned=headline[arm], scratch=headline[f'scratch_{label}']))
    return dict(geometry=summary['geometry'], bem_windows=headline,
                own_sphere_eligible_windows=summary['own_sphere_eligible_headline'],
                strata={seed: by_seed[seed]['strata'] for seed in SEEDS},
                dipole_subset=dipole, dipole_windows=len(dipole_rows), finetuning=sizes,
                bem_training_eligibility={seed: by_seed[seed]['bem_training_eligibility'] for seed in SEEDS},
                fits={seed: by_seed[seed]['fits'] for seed in SEEDS})


def benchmark_table(evidence):
    report = evidence.json('generation-benchmark-v11/generation_benchmark.json')
    rows = []
    for case in report['cases']:
        summary, phases = case['summary'], case['summary']['median_run_phase_seconds']
        repetitions = case['repetitions']
        for key in ('seconds', 'windows_per_second', 'peak_process_tree_rss_bytes'):
            values = [repetition[key] for repetition in repetitions]
            for statistic, function in (('median', np.median), ('minimum', np.min), ('maximum', np.max)):
                if not np.isclose(function(values), summary[key][statistic], rtol=1e-12, atol=0):
                    raise ValueError(f"Benchmark case {case['case']}: {statistic} of {key} differs from its runs")
        # The stage times are those of the run with the median total time.
        median_runs = [repetition for repetition in repetitions
                       if repetition['seconds'] == summary['seconds']['median']]
        if len(median_runs) != 1 or median_runs[0]['setup_seconds'] != summary['median_run_setup_seconds'] \
                or median_runs[0]['phase_seconds'] != phases:
            raise ValueError(f"Benchmark case {case['case']}: the stage times are not those of the median run")
        if any(repetition['output_bytes'] != summary['output_bytes'] for repetition in repetitions):
            raise ValueError(f"Benchmark case {case['case']}: the output size differs between the runs")
        writing = phases.get('hdf5_parent_append', 0.0) + phases.get('hdf5_finalize', 0.0)
        setup = [repetition['setup_seconds'] for repetition in repetitions]
        rows.append(dict(
            case=case['case'], mode=case['mode'], format=case['format'], compressed=case['compress'],
            workers=case['workers'], windows=case['windows'], repetitions=len(repetitions),
            seconds=summary['seconds'], windows_per_second=summary['windows_per_second'],
            peak_rss_gib={key: value / 2**30 for key, value in summary['peak_process_tree_rss_bytes'].items()},
            output_gib=summary['output_bytes'] / 2**30,
            setup_seconds_of_all_runs=dict(minimum=float(min(setup)), maximum=float(max(setup))),
            median_run=dict(setup_seconds=summary['median_run_setup_seconds'],
                            split_generation_and_writing_seconds=phases.get('split_generation_and_writing'),
                            hdf5_writing_seconds=writing if writing else None)))
    host = {key: report[key] for key in ('platform', 'cpu', 'logical_cpus', 'physical_cpus', 'ram_bytes',
                                         'available_ram_bytes', 'threads', 'channels', 'window_samples',
                                         'environment')}
    patch = next((case['patch_overlap'] for case in report['cases'] if 'patch_overlap' in case), None)
    return dict(host=host, rows=rows, patch_overlap_of_patch_case=patch, report=report)


def common_setting(values, what):
    """The one value that every dataset of an experiment recorded."""
    values = list(values)
    if any(value != values[0] for value in values):
        raise ValueError(f'The datasets disagree on {what}: {values}')
    return values[0]


def experiment_settings(evidence):
    """Fixed settings of the supervised experiments, read from the records of every dataset."""
    def config(study, seed, policy):
        return evidence.json(Path(study) / seed / policy / 'dataset/metadata.json')['config']

    configs = {study: {(seed, policy): config(study, seed, policy) for seed in SEEDS for policy in POLICIES}
               for study in ('split-study-v11', 'patch-split-study-v11')}
    everything = [item for study in configs.values() for item in study.values()]
    point, patch = configs['split-study-v11'], configs['patch-split-study-v11']
    # The folder of a dataset is named after its seed; the model seeds are checked with the fitted networks.
    for study, by_dataset in configs.items():
        for (seed, policy), item in by_dataset.items():
            if str(item['master_seed']) != seed:
                raise ValueError(f"{study}/{seed}/{policy}: the dataset records seed {item['master_seed']}")

    def shared(path, records=everything):
        def value(item):
            for key in path:
                item = item[key]
            return item
        return common_setting((value(item) for item in records), '.'.join(path))

    def by_policy(study, policy, path):
        return shared(path, [item for (seed, item_policy), item in study.items() if item_policy == policy])

    def rotation(study):
        return [study[seed, 'region']['split']['region_rotation_seed'] for seed in SEEDS]

    dipole_subset = common_setting(
        (evidence.json(Path('split-study-v11') / seed / policy / 'summary.json')['ecd_subset']['dipole']['windows']
         for seed in SEEDS for policy in POLICIES), 'the size of the dipole-fit subset')
    clearance = common_setting(
        (evidence.json(Path('head-transfer-v11') / seed / 'eligibility.json')['minimum_inner_skull_clearance_mm']
         for seed in SEEDS), 'the clearance threshold of the transfer experiment')
    template = sorted(evidence.json('head-transfer-v11/provenance.json')['template_sha256'])
    return dict(
        dataset_seeds=[int(seed) for seed in SEEDS], model_seeds=[int(fit) for fit in FITS],
        windows=dict(train=shared(('dataset', 'n_train')), validation=shared(('dataset', 'n_val')),
                     test=shared(('dataset', 'n_test'))),
        n_legendre=shared(('head', 'n_legendre')),
        window_samples=shared(('signal', 'window_samples')), sampling_freq_hz=shared(('signal', 'sampling_freq_hz')),
        snr_db_range=shared(('noise', 'snr_db_range')),
        montage_file=shared(('electrode_file',)), noise_model=shared(('noise', 'noise_model')),
        reference_mode=shared(('output', 'reference_mode')),
        eeg_normalization=shared(('output', 'eeg_normalization')),
        point=dict(n_cortical_positions=shared(('sources', 'n_cortical_positions'), list(point.values())),
                   n_subcortical_positions=shared(('sources', 'n_subcortical_positions'), list(point.values())),
                   region_count=by_policy(point, 'region', ('split', 'region_count')),
                   buffer_mm=by_policy(point, 'region', ('split', 'buffer_mm')),
                   region_rotation_seeds=rotation(point)),
        patch=dict(n_cortical_positions_exact=by_policy(patch, 'position', ('sources', 'n_cortical_positions')),
                   n_cortical_positions_region=by_policy(patch, 'region', ('sources', 'n_cortical_positions')),
                   region_count=by_policy(patch, 'region', ('split', 'region_count')),
                   buffer_mm=by_policy(patch, 'region', ('split', 'buffer_mm')),
                   region_rotation_seeds=rotation(patch),
                   diameter_range_mm=shared(('patch', 'diameter_range_mm'), list(patch.values()))),
        transfer=dict(minimum_inner_skull_clearance_mm=clearance, template_files=template),
        dipole_fit_subset_windows=dict(split_study_per_dataset=dipole_subset))


# --------------------------------------------------------------------------- rerun of the numerical validation

def compare_array_digests(first, second):
    """Compare two records of array digests, as written per release for the validation datasets."""
    if set(first['datasets']) != set(second['datasets']):
        raise ValueError('The two releases generated different sets of datasets')
    arrays = equal = 0
    only_first, only_second = [], []
    for dataset in sorted(second['datasets']):
        files = sorted(set(first['datasets'][dataset]) | set(second['datasets'][dataset]))
        for file in files:
            one, other = first['datasets'][dataset].get(file, {}), second['datasets'][dataset].get(file, {})
            only_first += [f'{dataset}/{file}/{name}' for name in sorted(set(one) - set(other))]
            only_second += [f'{dataset}/{file}/{name}' for name in sorted(set(other) - set(one))]
            for name in sorted(set(one) & set(other)):
                arrays += 1
                equal += one[name] == other[name]
    return dict(datasets=len(second['datasets']), arrays_in_both=arrays, arrays_with_equal_digest=equal,
                arrays_only_in_first=only_first, arrays_only_in_second=only_second)


def without(record, *keys):
    return {key: value for key, value in record.items() if key not in keys}


def largest_relative_change(first, second, cases, orders):
    if any(first[case][order] == 0 for case in cases for order in orders):
        raise ValueError('A relative change from zero is not defined')
    return float(max(abs(second[case][order] / first[case][order] - 1) for case in cases for order in orders))


def validation_rerun_tables(evidence):
    """The numerical validation rerun on a second platform, with release 1.0.0 and with release 1.1.0."""
    base = Path('validation-rerun-v11')
    archived, new, old = base / 'release-1.0.0-linux-archived', base / 'release-1.1.0-windows', \
        base / 'release-1.0.0-windows'
    digests = compare_array_digests(evidence.json(base / 'generated_arrays_release-1.0.0.json'),
                                    evidence.json(base / 'generated_arrays_release-1.1.0.json'))
    if digests['arrays_with_equal_digest'] != digests['arrays_in_both'] or digests['arrays_only_in_first'] \
            or digests['arrays_only_in_second']:
        raise ValueError(f"validation rerun: the two releases generated different arrays "
                         f"({digests['arrays_with_equal_digest']} of {digests['arrays_in_both']} equal)")

    environment = ('software_environment',)
    results = {}
    for name, folder in (('linux_release_1_0_0', archived), ('windows_release_1_1_0', new),
                         ('windows_release_1_0_0', old)):
        nested = folder != archived
        inverse = evidence.json(folder / ('inverse/' if nested else '') / 'ecd_69ch_35db_summary.json')
        rows = evidence.rows(folder / ('inverse/' if nested else '') / 'ecd_69ch_35db_per_sample.csv')
        position = columns(rows, ['err_mm']).ravel()
        # The fitted and the true positions of the per-dipole file must give its position errors.
        distance = np.linalg.norm(columns(rows, [f'estimated_{axis}_m' for axis in 'xyz'])
                                  - columns(rows, [f'true_{axis}_m' for axis in 'xyz']), axis=1) * 1000.0
        if not np.allclose(distance, position, rtol=0, atol=1e-4):
            raise ValueError(f'validation rerun, {name}: the fitted positions do not give the recorded position errors')
        # The per-dipole file holds rounded values; the summary was computed from the full values.
        for key, statistic in (('median', np.median), ('p90', lambda sample: np.percentile(sample, 90)),
                               ('maximum', np.max)):
            evidence.check(statistic(position), inverse['position_error_mm'][key],
                           f'validation rerun, {name}, position error {key}', tolerance=1e-4)
        for column, recorded, what in (
                ('orientation_error_deg', inverse['orientation_error_deg']['median'], 'orientation error'),
                ('moment_relative_error', inverse['moment_amplitude']['relative_error']['median'], 'moment error'),
                ('gof_pct', inverse['goodness_of_fit_percent']['median'], 'goodness of fit')):
            evidence.check(np.median(columns(rows, [column])), recorded,
                           f'validation rerun, {name}, median {what}', tolerance=1e-4)
        results[name] = dict(
            inverse=inverse, rows=rows,
            patch=evidence.json(folder / ('patch/' if nested else '') / 'patch_forward_mne_summary.json'),
            coefficients=evidence.json(folder / ('coefficients/' if nested else '') / 'transfer_coefficients.json'),
            legendre=evidence.json(folder / ('legendre/' if nested else '') / 'legendre_convergence.json'))
    linux, windows, windows_old = (results[name] for name in ('linux_release_1_0_0', 'windows_release_1_1_0',
                                                              'windows_release_1_0_0'))

    def csv_hash(folder):
        return evidence.hashes[(folder / 'inverse/ecd_69ch_35db_per_sample.csv').as_posix()]

    # Fields that only one release writes, or that name files, are left out of the comparison.
    inverse_ignored = (*environment, 'configuration')
    releases = dict(
        per_dipole_file_identical=csv_hash(new) == csv_hash(old),
        dipole_recovery_and_sweep_equal=without(windows['inverse'], *inverse_ignored)
        == without(windows_old['inverse'], *inverse_ignored),
        patch_comparison_equal=without(windows['patch'], *environment) == without(windows_old['patch'], *environment),
        transfer_coefficients_equal=without(windows['coefficients'], *environment)
        == without(windows_old['coefficients'], *environment),
        # The recorded configuration gained fields in release 1.1.0; the results are compared.
        legendre_convergence_equal=without(windows['legendre'], *environment, 'config')
        == without(windows_old['legendre'], *environment, 'config'))
    unequal = [name for name, equal in releases.items() if not equal]
    if unequal:
        raise ValueError(f"validation rerun: the two releases differ on one machine: {', '.join(unequal)}")

    labels = [name for name in windows['rows'][0] if name.startswith('true_') or name in (
        'sample_index', 'fit_sample', 'source_radius_mm', 'source_depth_mm', 'snr_db', 'population')]
    same_sources = len(windows['rows']) == len(linux['rows']) and all(
        a[name] == b[name] for a, b in zip(windows['rows'], linux['rows']) for name in labels)
    if not same_sources:
        raise ValueError('validation rerun: the two platforms fitted different sources')
    fitted = {name: columns(record['rows'], [f'estimated_{axis}_m' for axis in 'xyz']) * 1000.0
              for name, record in (('linux', linux), ('windows', windows))}
    moved = np.linalg.norm(fitted['windows'] - fitted['linux'], axis=1)

    def pair(path):
        values = []
        for record in (linux, windows):
            item = record
            for key in path:
                item = item[key]
            values.append(float(item))
        return dict(linux=values[0], windows=values[1])

    levels = {name: [float(level['snr_db']) for level in record['inverse']['snr_degradation_sweep']]
              for name, record in (('linux', linux), ('windows', windows))}
    if levels['linux'] != levels['windows'] or len(set(levels['linux'])) != len(levels['linux']):
        raise ValueError('validation rerun: the sweep levels of the two platforms differ')
    sweep = [dict(snr_db=float(a['snr_db']), linux=float(a['position_error_median_mm']),
                  windows=float(b['position_error_median_mm']))
             for a, b in zip(linux['inverse']['snr_degradation_sweep'], windows['inverse']['snr_degradation_sweep'])]
    relative = {name: record['legendre']['relative_error'] for name, record in (('linux', linux),
                                                                                ('windows', windows))}
    cases = sorted(relative['linux'])
    superficial = [case for case in cases if case.startswith('superficial')]
    platforms = dict(
        environments=dict(linux=linux['inverse']['software_environment'],
                          windows=windows['inverse']['software_environment']),
        dipoles=len(windows['rows']), same_sources_and_realized_snr=bool(same_sources),
        fitted_position_difference_mm=dict(sample_spread(moved), share_above_0p1_mm=float(np.mean(moved > 0.1))),
        position_error_mm={key: pair(('inverse', 'position_error_mm', key)) for key in ('median', 'p90', 'maximum')},
        orientation_error_median_deg=pair(('inverse', 'orientation_error_deg', 'median')),
        moment_relative_error_median=pair(('inverse', 'moment_amplitude', 'relative_error', 'median')),
        goodness_of_fit_median_percent=pair(('inverse', 'goodness_of_fit_percent', 'median')),
        sweep_median_position_error_mm=sweep,
        largest_change_of_a_sweep_median_mm=float(max(abs(row['windows'] - row['linux']) for row in sweep)),
        patch_median_relative_field_difference=pair(('patch', 'distributed_patch', 'median')),
        point_median_relative_field_difference=pair(('patch', 'point_source_control', 'median')),
        patch_to_point_median_ratio=pair(('patch', 'patch_to_control_median_ratio')),
        transfer_coefficients_equal=without(windows['coefficients'], *environment)
        == without(linux['coefficients'], *environment),
        legendre=dict(
            cases=cases, superficial_cases=superficial,
            largest_relative_change_superficial_orders_40_and_80=largest_relative_change(
                relative['linux'], relative['windows'], superficial, ('40', '80')),
            largest_relative_change_superficial_orders_100_and_120=largest_relative_change(
                relative['linux'], relative['windows'], superficial, ('100', '120')),
            largest_relative_change_any_case_orders_100_and_120=largest_relative_change(
                relative['linux'], relative['windows'], cases, ('100', '120')),
            largest_relative_change_other_cases=largest_relative_change(
                relative['linux'], relative['windows'], [case for case in cases if case not in superficial],
                sorted(relative['linux'][cases[0]], key=int)),
            superficial_radial_order_100=dict(
                linux=float(relative['linux']['superficial_radial_79_5mm']['100']),
                windows=float(relative['windows']['superficial_radial_79_5mm']['100']))))
    if not platforms['transfer_coefficients_equal']:
        raise ValueError('validation rerun: the transfer coefficients differ between the platforms')
    return dict(array_digests_of_the_two_releases=digests, two_releases_on_one_machine=releases,
                two_platforms=platforms), windows


# --------------------------------------------------------------------------- parameter fit of MNE-Python's sphere model

def sample_spread(values):
    values = np.asarray(values, dtype=float)
    return dict(n=int(values.size), median=float(np.median(values)),
                p90=float(np.percentile(values, 90)), maximum=float(values.max()))


def check_spread(evidence, values, record, what):
    recomputed = sample_spread(values)
    if recomputed['n'] != record['n']:
        raise ValueError(f"{what}: {recomputed['n']} values, recorded {record['n']}")
    for key in ('median', 'p90', 'maximum'):
        evidence.check(recomputed[key], record[key], f'{what}, {key}', tolerance=1e-12)


def perturbation_ranges(rows):
    """Smallest and largest value of every quantity of perturbation.csv, by perturbed fit and sphere fit."""
    labels = ('perturbed_fit', 'sphere_fit', 'objective_scale_minus_one_in_eps', 'parameter_fit_stopped_by_itself')
    ranges = {}
    for perturbed, name in dict.fromkeys((row['perturbed_fit'], row['sphere_fit']) for row in rows):
        selected = [row for row in rows if (row['perturbed_fit'], row['sphere_fit']) == (perturbed, name)]
        entry = {}
        for column in selected[0]:
            values = [float(row[column]) for row in selected if row[column] != ''] if column not in labels else []
            if values:
                entry[column] = [min(values), max(values)]
        entry.update(trials=len(selected), trials_in_which_the_parameter_fit_stopped_by_itself=sum(
            int(row['parameter_fit_stopped_by_itself']) for row in selected))
        ranges.setdefault(perturbed, {})[name] = entry
    return ranges


def small_run_key(row):
    return row['kernel'], row['changed_objective'], row['sphere_fit'], row['objective_scale_minus_one_in_eps']


def small_tolerance_ranges(rows):
    """Smallest and largest value of the position statistics over the runs of each sphere fit with the small radius."""
    statistics = ('position_error_median_mm', 'position_error_p90_mm', 'position_error_maximum_mm',
                  'fitted_position_change_median_mm', 'fitted_position_change_p90_mm',
                  'fitted_position_change_maximum_mm')
    ranges = {}
    for name in dict.fromkeys(row['sphere_fit'] for row in rows):
        selected = [row for row in rows if row['sphere_fit'] == name]
        entry = dict(runs=len(selected), kernels=sorted({row['kernel'] for row in selected}),
                     runs_in_which_the_parameter_fit_stopped_by_itself=sum(
                         int(row['parameter_fit_stopped_by_itself']) for row in selected),
                     dipole_fits=sum(int(row['dipole_fits']) for row in selected),
                     dipole_fits_stopped_by_themselves=sum(
                         int(row['dipole_fits_stopped_by_themselves']) for row in selected))
        for column in statistics:
            values = [float(row[column]) for row in selected if row[column] != '']
            entry[column] = [min(values), max(values)] if values else None
        ranges[name] = entry
    return ranges


def position_change(fitted, reference):
    """Distances in mm between the fitted positions of two runs on the same dipoles."""
    return np.linalg.norm(np.asarray(fitted, dtype=float) - np.asarray(reference, dtype=float), axis=1)


def check_position_change(evidence, moved, record, what):
    """Compare the distances between two sets of fitted positions with a recorded summary of them."""
    moved = np.asarray(moved, dtype=float)
    check_spread(evidence, moved, record, what)
    # A distance within rounding of 0.1 mm may fall on either side.
    low, high = float(np.mean(moved > 0.1 + 1e-9)), float(np.mean(moved > 0.1 - 1e-9))
    if not low - 1e-12 <= record['share_above_0p1_mm'] <= high + 1e-12:
        raise ValueError(f'{what}: the share of positions that moved by more than 0.1 mm differs from the record')


def merged_range(ranges):
    """The smallest lower end and the largest upper end of several [low, high] ranges."""
    return [float(min(low for low, _ in ranges)), float(max(high for _, high in ranges))]


def sphere_fit_tables(evidence, rerun_digests, rerun):
    """How the Berg-Scherg parameter fit of MNE-Python ends, and the comparisons with both fits.

    Every statistic of a run is recomputed from the per-window files of the
    record: the two arms, each trial with a changed objective, each run with
    the small final radius of the dipole fits, each kernel of the
    linear-algebra library, and each sphere of the adaptation experiment.
    ``rerun_digests`` are the array digests of the validation datasets of
    release 1.1.0 and ``rerun`` the results of the validation rerun on the
    same machine, which the arm at the limit of SciPy must reproduce.
    """
    base = Path('sphere-fit-convergence-v11')
    summary = evidence.json(base / 'summary.json')
    if summary['smoke'] or not summary['instrumented_call_reproduces_installed_mne_python'] \
            or not summary['instrumented_dipole_fit_reproduces_installed_mne_python']:
        raise ValueError('The sphere-fit record is a smoke run or did not reproduce the fits of MNE-Python')
    names = list(summary['arms'])
    if names != ['mne_limit', 'raised_limit']:
        raise ValueError(f'sphere fit: unexpected arms {names}')
    recovery = evidence.rows(base / 'dipole_recovery_35db.csv')
    fields = evidence.rows(base / 'field_differences.csv')
    sweep = evidence.rows(base / 'snr_sweep.csv')
    for name in names:
        arm = summary['arms'][name]
        what = f'sphere fit, {name}'
        for quantity in ('position_error_mm', 'orientation_error_deg', 'moment_relative_error',
                         'goodness_of_fit_percent'):
            check_spread(evidence, columns(recovery, [f'{quantity}_{name}']).ravel(),
                         arm['dipole_recovery'][quantity], f'{what}, {quantity}')
        for source in ('patch', 'point'):
            check_spread(evidence, columns([row for row in fields if row['source'] == source],
                                           [f'relative_field_difference_{name}']).ravel(),
                         arm['patch_comparison'][source], f'{what}, {source} field difference')
        evidence.check(arm['patch_comparison']['patch']['median'] / arm['patch_comparison']['point']['median'],
                       arm['patch_comparison']['patch_to_point_median_ratio'],
                       f'{what}, ratio of the two field differences', tolerance=1e-12)
        if [level['snr_db'] for level in arm['sweep']] != sorted({float(row['snr_db']) for row in sweep}):
            raise ValueError(f'{what}: the sweep levels differ from those of the per-dipole file')
        for level in arm['sweep']:
            check_spread(evidence, columns([row for row in sweep if float(row['snr_db']) == level['snr_db']],
                                           [f'position_error_mm_{name}']).ravel(),
                         level['position_error_mm'], f"{what}, sweep at {level['snr_db']:g} dB")

    # The trials with a changed objective: every statistic of a trial from its per-dipole and per-source rows.
    trials = evidence.rows(base / 'perturbation.csv')
    by_run, by_field = {}, {}
    for row in evidence.rows(base / 'perturbation_dipoles.csv'):
        by_run.setdefault((row['perturbed_fit'], row['sphere_fit'], row['objective_scale_minus_one_in_eps']),
                          []).append(row)
    for row in evidence.rows(base / 'perturbation_fields.csv'):
        by_field.setdefault((row['sphere_fit'], row['objective_scale_minus_one_in_eps'], row['source']),
                            []).append(float(row['relative_field_difference']))

    def fitted(rows):
        if [int(row['sample_index']) for row in rows] != list(range(len(recovery))):
            raise ValueError('sphere fit: a run does not hold every dipole once, in order')
        return columns(rows, [f'fitted_{axis}_mm' for axis in 'xyz'])

    unchanged = {name: fitted(by_run['none', name, '0']) for name in names}
    quantities = ('position_error_mm', 'orientation_error_deg', 'moment_relative_error', 'goodness_of_fit_percent')
    for name in names:
        # The unchanged run of the trial file is the run of the per-window file.
        if not np.array_equal(columns(by_run['none', name, '0'], quantities),
                              columns(recovery, [f'{quantity}_{name}' for quantity in quantities])):
            raise ValueError(f'sphere fit, {name}: the two per-dipole files disagree on the unchanged run')
    if len(by_run) != len(trials) + len(names):
        raise ValueError('sphere fit: the per-dipole file and the list of trials hold different runs')
    for trial in trials:
        perturbed, name, step = trial['perturbed_fit'], trial['sphere_fit'], trial['objective_scale_minus_one_in_eps']
        what = f'sphere fit, {perturbed} trial {step} with {name}'
        members = by_run[perturbed, name, step]
        position = columns(members, ['position_error_mm']).ravel()
        for column, value in (('position_error_median_mm', np.median(position)),
                              ('position_error_p90_mm', np.percentile(position, 90)),
                              ('position_error_maximum_mm', position.max())):
            evidence.check(value, float(trial[column]), f'{what}, {column}', tolerance=1e-12)
        evidence.check(np.median(columns(members, ['moment_relative_error'])),
                       float(trial['moment_relative_error_median']), f'{what}, moment_relative_error_median',
                       tolerance=1e-12)
        moved = position_change(fitted(members), unchanged[name])
        check_position_change(evidence, moved, dict(
            n=len(moved), median=float(trial['fitted_position_change_median_mm']),
            p90=float(trial['fitted_position_change_p90_mm']),
            maximum=float(trial['fitted_position_change_maximum_mm']),
            share_above_0p1_mm=int(trial['fitted_positions_changed_by_more_than_0p1_mm']) / len(moved)),
            f'{what}, change of the fitted positions')
        for source in ('patch', 'point'):
            column = f'{source}_median_relative_field_difference'
            if (trial[column] == '') != (perturbed != 'sphere_fit'):
                raise ValueError(f'{what}: only the trials of the parameter fit repeat the field comparison')
            if trial[column] != '':
                evidence.check(np.median(by_field[name, step, source]), float(trial[column]), f'{what}, {column}',
                               tolerance=1e-12)
    ranges = perturbation_ranges(trials)
    if ranges != summary['perturbation']['ranges']:
        raise ValueError('sphere fit: the ranges over the trials differ from the record')

    # Other kernels of the linear-algebra library.
    kernels = summary['library_kernels']
    kernel_dipoles = evidence.rows(base / 'kernels_dipole_recovery_35db.csv')
    kernel_fields = evidence.rows(base / 'kernels_field_differences.csv')
    if {row['kernel'] for row in kernel_dipoles} != set(kernels['arms']):
        raise ValueError('sphere fit: the per-dipole file of the kernels and the record hold different kernels')
    for kernel, entry in kernels['arms'].items():
        for name in names:
            what = f'sphere fit, kernel {kernel}, {name}'
            members = [row for row in kernel_dipoles if (row['kernel'], row['sphere_fit']) == (kernel, name)]
            check_spread(evidence, columns(members, ['position_error_mm']).ravel(),
                         entry['arms'][name]['dipole_recovery']['position_error_mm'], f'{what}, position error')
            for quantity in quantities[1:]:
                check_spread(evidence, columns(members, [quantity]).ravel(),
                             entry['arms'][name]['dipole_recovery'][quantity], f'{what}, {quantity}')
            check_position_change(evidence, position_change(fitted(members), unchanged[name]),
                                  entry['fitted_position_change_from_the_parent_mm'][name],
                                  f'{what}, change of the fitted positions')
            comparison = entry['arms'][name]['patch_comparison']
            for source in ('patch', 'point'):
                check_spread(evidence, [float(row['relative_field_difference']) for row in kernel_fields
                                        if (row['kernel'], row['sphere_fit'], row['source']) == (kernel, name, source)],
                             comparison[source], f'{what}, {source} field difference')
            evidence.check(comparison['patch']['median'] / comparison['point']['median'],
                           comparison['patch_to_point_median_ratio'],
                           f'{what}, ratio of the two field differences', tolerance=1e-12)

    # The spheres of the adaptation experiment.
    spheres = summary['spheres']
    if spheres is None:
        raise ValueError('sphere fit: the record lacks the spheres of the adaptation experiment (--geometry)')
    geometry = evidence.json('head-adaptation-v11/geometry.json')['variants']
    if spheres['geometry_sha256'] != evidence.hashes['head-adaptation-v11/geometry.json'] \
            or list(spheres['variants']) != list(geometry):
        raise ValueError('sphere fit: the spheres are not those of the archived adaptation experiment')
    if summary['head_model'] != {key: spheres['variants']['default'][key]
                                 for key in ('radii_m', 'conductivities_S_per_m')}:
        raise ValueError('sphere fit: the head model of the two arms is not the default sphere of the adaptation '
                         'experiment')
    sphere_dipoles = evidence.rows(base / 'spheres_dipole_recovery.csv')
    sphere_fields = evidence.rows(base / 'spheres_field_differences.csv')
    for sphere, entry in spheres['variants'].items():
        if entry['radii_m'] != geometry[sphere]['radii_m'] \
                or entry['conductivities_S_per_m'] != geometry[sphere]['conductivities_S_per_m']:
            raise ValueError(f'sphere fit, sphere {sphere}: the head model differs from the adaptation experiment')
        members = [row for row in sphere_dipoles if row['sphere'] == sphere]
        medians = {}
        for name in names:
            position = columns(members, [f'position_error_mm_{name}']).ravel()
            check_spread(evidence, position, entry[name]['position_error_mm'],
                         f'sphere fit, sphere {sphere}, {name}, position error')
            medians[name] = float(np.median(position))
            check_spread(evidence, columns([row for row in sphere_fields if row['sphere'] == sphere],
                                           [f'relative_field_difference_{name}']).ravel(),
                         entry[name]['point_field_difference'],
                         f'sphere fit, sphere {sphere}, {name}, field difference')
        evidence.check(medians['raised_limit'] - medians['mne_limit'],
                       entry['position_error_change_mm']['median_of_raised_limit_minus_median_of_mne_limit'],
                       f'sphere fit, sphere {sphere}, change of the median position error', tolerance=1e-12)
        positions = {name: columns(members, [f'fitted_{axis}_mm_{name}' for axis in 'xyz']) for name in names}
        check_position_change(evidence, position_change(positions['raised_limit'], positions['mne_limit']),
                              entry['position_error_change_mm']['fitted_position'],
                              f'sphere fit, sphere {sphere}, change of the fitted positions')

    # The dipole fits ended at a small trust-region radius: every run from its per-dipole rows.
    small = summary['small_tolerance']
    small_runs = evidence.rows(base / 'small_tolerance.csv')
    small_by_run = {}
    for row in evidence.rows(base / 'small_tolerance_dipoles.csv'):
        small_by_run.setdefault(small_run_key(row), []).append(row)
    if len(small_by_run) != len(small_runs) or {small_run_key(run) for run in small_runs} != set(small_by_run):
        raise ValueError('sphere fit: the two files of the runs with the small final radius hold different runs')
    main_kernel = small['kernel_of_the_main_process']
    # The experiment script joins the kernels of the loaded libraries with '+' and writes 'unknown' for none.
    own_kernels = {str(entry['kernel']) for entry in kernels['of_the_parent_process']}
    if ('+'.join(sorted(own_kernels)) or 'unknown') != main_kernel:
        raise ValueError('sphere fit: the runs with the small final radius and the record of the kernels name '
                         'different kernels of the main process')
    missing = [name for name in names if (main_kernel, 'none', name, '0') not in small_by_run]
    if missing:
        raise ValueError('sphere fit: the runs with the small final radius lack the unchanged run of the main '
                         f"process with {', '.join(missing)}")
    # The trials that the record names are those of the runs.
    steps = sorted({int(run['objective_scale_minus_one_in_eps']) for run in small_runs
                    if run['changed_objective'] != 'none'})
    if steps != small['steps']:
        raise ValueError('sphere fit: the trials of the runs with the small final radius are not those that the '
                         'record names')
    # With each sphere fit: the unchanged run and every trial of the main process, and the unchanged run under
    # every other kernel of the record. An unchanged run has the factor 0.
    expected = {(main_kernel, changed, name, str(step)) for name in names
                for changed, numbers in (('none', [0]), ('parameter_fit', steps), ('dipole_fits', steps))
                for step in numbers}
    expected |= {(kernel, 'none', name, '0') for kernel in kernels['arms'] for name in names}
    if set(small_by_run) != expected:
        def listed(runs):
            return '; '.join(f'kernel {kernel}, {changed} {step} with {name}'
                             for kernel, changed, name, step in sorted(runs)) or 'none'
        raise ValueError('sphere fit: the runs with the small final radius are not the unchanged run and the trials '
                         'of the main process and the unchanged run under every other kernel. Missing: '
                         f'{listed(expected - set(small_by_run))}. Not expected: '
                         f'{listed(set(small_by_run) - expected)}')
    reference = {name: fitted(small_by_run[main_kernel, 'none', name, '0']) for name in names}
    changes = ('fitted_position_change_median_mm', 'fitted_position_change_p90_mm',
               'fitted_position_change_maximum_mm')
    for run in small_runs:
        kernel, changed, name, step = small_run_key(run)
        what = f'sphere fit, small final radius, kernel {kernel}, {changed} {step} with {name}'
        members = small_by_run[kernel, changed, name, step]
        position = columns(members, ['position_error_mm']).ravel()
        for column, value in (('position_error_median_mm', np.median(position)),
                              ('position_error_p90_mm', np.percentile(position, 90)),
                              ('position_error_maximum_mm', position.max())):
            evidence.check(value, float(run[column]), f'{what}, {column}', tolerance=1e-12)
        is_reference = (kernel, changed) == (main_kernel, 'none')
        if any((run[column] == '') != is_reference for column in changes):
            raise ValueError(f'{what}: every run except the unchanged run of the main process records how far '
                             'the fitted positions moved, and that run does not')
        if not is_reference:
            moved = position_change(fitted(members), reference[name])
            for column, value in zip(changes, (np.median(moved), np.percentile(moved, 90), moved.max())):
                evidence.check(value, float(run[column]), f'{what}, {column}', tolerance=1e-9)
    small_ranges = small_tolerance_ranges(small_runs)
    if small_ranges != small['ranges']:
        raise ValueError('sphere fit: the ranges over the runs with the small final radius differ from the record')
    for kernel, entry in kernels['arms'].items():
        for name in names:
            what = f'sphere fit, small final radius, kernel {kernel}, {name}'
            members = small_by_run[kernel, 'none', name, '0']
            check_spread(evidence, columns(members, ['position_error_mm']).ravel(),
                         entry['small_tolerance'][name]['position_error_mm'], f'{what}, position error')
            check_position_change(evidence, position_change(fitted(members), reference[name]),
                                  entry['small_tolerance'][name]['fitted_position_change_from_the_parent_mm'],
                                  f'{what}, change of the fitted positions')

    # Ranges over every run of one sphere fit: the unchanged run, the trials and the kernels.
    all_runs = {}
    for name in names:
        arm = summary['arms'][name]
        kernel_arms = [entry['arms'][name] for entry in kernels['arms'].values()]
        position = {}
        for key, column in (('median', 'position_error_median_mm'), ('p90', 'position_error_p90_mm'),
                            ('maximum', 'position_error_maximum_mm')):
            unchanged_value = arm['dipole_recovery']['position_error_mm'][key]
            others = [item['dipole_recovery']['position_error_mm'][key] for item in kernel_arms]
            position[key] = merged_range([[unchanged_value, unchanged_value], ranges['sphere_fit'][name][column],
                                          ranges['dipole_fit'][name][column], *[[value, value] for value in others]])
        field = {}
        for source in ('patch', 'point'):
            unchanged_value = arm['patch_comparison'][source]['median']
            others = [item['patch_comparison'][source]['median'] for item in kernel_arms]
            field[source] = merged_range([[unchanged_value, unchanged_value],
                                          ranges['sphere_fit'][name][f'{source}_median_relative_field_difference'],
                                          *[[value, value] for value in others]])
        evaluations = [arm['fit']['evaluations'], *ranges['sphere_fit'][name]['parameter_fit_evaluations'],
                       *[item['fit']['evaluations'] for item in kernel_arms]]
        trials_of_one_kind = ranges['sphere_fit'][name]['trials']
        all_runs[name] = dict(
            runs_with_a_dipole_recovery=(1 + trials_of_one_kind + ranges['dipole_fit'][name]['trials']
                                         + len(kernel_arms)),
            runs_with_a_field_comparison=1 + trials_of_one_kind + len(kernel_arms),
            position_error_mm=position, median_relative_field_difference=field,
            parameter_fit_evaluations=[int(min(evaluations)), int(max(evaluations))],
            parameter_fits=1 + trials_of_one_kind + len(kernel_arms),
            parameter_fits_that_stopped_by_themselves=int(
                arm['fit']['stopped_by_itself']
                + ranges['sphere_fit'][name]['trials_in_which_the_parameter_fit_stopped_by_itself']
                + sum(item['fit']['stopped_by_itself'] for item in kernel_arms)))

    digests = summary['dataset_array_sha256']
    expected = dict(
        dipole_recovery_35db=rerun_digests['inverse_headline_35dB']['test.h5']['eeg']['sha256'],
        patch=rerun_digests['patch_comparison_patch']['train.h5']['ground_truth/eeg_signal_only']['sha256'],
        point=rerun_digests['patch_comparison_point']['train.h5']['ground_truth/eeg_signal_only']['sha256'])
    # At the limit of SciPy the experiment must reproduce the validation run of the same machine: its per-dipole
    # file, which holds four decimals, and every statistic of its summaries.
    first = columns(recovery, ['position_error_mm_mne_limit']).ravel()
    limit = summary['arms']['mne_limit']
    pairs = [(limit['dipole_recovery']['position_error_mm'][key], rerun['inverse']['position_error_mm'][key])
             for key in ('median', 'p90', 'maximum')]
    pairs += [(limit['dipole_recovery']['orientation_error_deg']['median'],
               rerun['inverse']['orientation_error_deg']['median']),
              (limit['dipole_recovery']['moment_relative_error']['median'],
               rerun['inverse']['moment_amplitude']['relative_error']['median']),
              (limit['dipole_recovery']['goodness_of_fit_percent']['median'],
               rerun['inverse']['goodness_of_fit_percent']['median']),
              (limit['patch_comparison']['patch']['median'], rerun['patch']['distributed_patch']['median']),
              (limit['patch_comparison']['point']['median'], rerun['patch']['point_source_control']['median'])]
    recorded_sweep = rerun['inverse']['snr_degradation_sweep']
    pairs += [(level['position_error_mm']['median'], float(item['position_error_median_mm']))
              for level, item in zip(limit['sweep'], recorded_sweep)]
    links = dict(
        same_datasets_as_the_validation_rerun=bool(
            all(digests[key] == value for key, value in expected.items())
            and all(digests['snr_sweep'][level] == rerun_digests[f'inverse_sweep_{level}dB']['test.h5']['eeg']['sha256']
                    for level in digests['snr_sweep'])),
        mne_limit_arm_equals_the_validation_rerun=bool(
            len(first) == len(rerun['rows'])
            and np.allclose(first, columns(rerun['rows'], ['err_mm']).ravel(), rtol=0, atol=5.1e-5)
            and [level['snr_db'] for level in limit['sweep']] == [float(item['snr_db']) for item in recorded_sweep]
            and all(np.isclose(value, recorded, rtol=1e-9, atol=0) for value, recorded in pairs)))
    broken = [name for name, holds in links.items() if not holds]
    if broken:
        raise ValueError(f"sphere fit: the record is not linked to the validation rerun: {', '.join(broken)}")
    return dict(head_model=summary['head_model'], optimizer_call=summary['optimizer_call'],
                evaluation_limits=summary['evaluation_limits'], sizes=summary['sizes'], arms=summary['arms'],
                default_sphere_model_of_mne_python=summary['default_sphere_model_of_mne_python'],
                dipole_fits=summary['dipole_fits'],
                small_tolerance=dict(final_radius_m=small['final_radius_m'],
                                     final_radius_of_mne_python_m=small['final_radius_of_mne_python_m'],
                                     steps=small['steps'], kernel_of_the_main_process=main_kernel,
                                     ranges=small_ranges,
                                     runs=small_runs),
                perturbation=dict(steps=summary['perturbation']['steps'],
                                  objective_factor=summary['perturbation']['objective_factor'], ranges=ranges,
                                  trials=trials),
                library_kernels=kernels, spheres=spheres, all_runs=all_runs,
                links=links, software_environment=summary['software_environment'])


# --------------------------------------------------------------------------- dipole comparators, converged fit

def recorded_comparator(evidence, experiment, seed, case):
    """A sphere-model dipole fit as its experiment recorded it.

    Returns the fitted positions and the true source positions in mm, the
    position errors in mm and the window numbers, in the order of the
    per-window file of the experiment. The split study records positions
    only; its errors are their distances.
    """
    if experiment == 'split_study':
        folder = Path('split-study-v11') / seed / case / 'ecd_subset'
        rows = evidence.rows(folder / 'predictions.csv')
        windows = [int(window) for window in evidence.json(folder / 'test_rows.json')]
        if len(windows) != len(rows):
            raise ValueError(f'split study, dataset {seed}, {case}: the dipole-fit subset holds {len(rows)} rows '
                             f'and {len(windows)} window numbers')
        fitted, truth = xyz(rows, 'dipole_'), xyz(rows)
        return fitted, truth, np.linalg.norm(fitted - truth, axis=1), windows
    if experiment == 'transfer':
        base = Path('head-transfer-v11') / seed
        rows, prefix, error = evidence.rows(base / 'ecd_subset.csv'), f'ecd_{case}_', f'ecd_{case}_error_mm'
    elif experiment == 'adaptation':
        base = Path('head-adaptation-v11') / seed
        rows, error = evidence.rows(base / 'dipole_subset.csv'), f'dipole_{case}_error_mm'
        prefix = f"dipole_{case.removesuffix('_sphere_on_bem')}_"
    else:
        raise ValueError(f'Unknown experiment of a dipole comparator: {experiment}')
    windows = [int(row['window_id']) for row in rows]
    eligible = evidence.rows(base / 'eligibility.csv')
    sources = {int(row['window_id']): row for row in eligible}
    if len(sources) != len(eligible):
        raise ValueError(f'{experiment}, dataset {seed}: the record names a window twice')
    if not set(windows) <= set(sources):
        raise ValueError(f'{experiment}, dataset {seed}: the record lacks the source position of a fitted window')
    return xyz(rows, prefix), xyz([sources[window] for window in windows]), columns(rows, [error]).ravel(), windows


def comparator_tables(evidence, sphere_fit, split):
    """The sphere-model dipole fits of three experiments, repeated with a converged parameter fit.

    The record must hold the fits that the paper reports and no others. The
    fits with the limit of SciPy must be those of the recorded experiments,
    window by window, in position and in error; both error columns must be
    the distances of the fitted positions from the true source positions. The
    parameter fits must end as the paper says and as in the sphere-fit
    record, which fitted the same spheres.
    """
    base = Path('dipole-comparators-converged-v11')
    summary = evidence.json(base / 'summary.json')
    if summary['smoke'] or not summary['compared_with_the_recorded_experiments']:
        raise ValueError('The record of the dipole comparators is a smoke run or was not compared with the '
                         'recorded experiments')
    if summary['evaluation_limits'] != sphere_fit['evaluation_limits']:
        raise ValueError('dipole comparators: the evaluation limits differ from those of the sphere-fit record')
    names = ['mne_limit', 'raised_limit']
    if sorted(summary['evaluation_limits']) != names:
        raise ValueError('dipole comparators: the arms are not mne_limit and raised_limit: '
                         f"{sorted(summary['evaluation_limits'])}")
    by_group = {}
    for row in evidence.rows(base / 'comparators.csv'):
        by_group.setdefault((row['experiment'], row['dataset_seed'], row['case']), []).append(row)
    recorded_groups = [(experiment, seed, case) for experiment in ('split_study', 'transfer', 'adaptation')
                       for seed, cases in summary[experiment].items() for case in cases]
    if set(recorded_groups) != set(by_group) or len(recorded_groups) != len(by_group):
        raise ValueError('dipole comparators: the per-window file and the summary hold different groups')
    spheres = sphere_fit['spheres']['variants']
    # The fits that the paper reports: the matched sphere of every dataset and policy of the split study, the
    # matched sphere on the sphere windows of the transfer experiment, and every sphere of the adaptation experiment.
    expected = [('split_study', seed, policy) for seed in SEEDS for policy in POLICIES]
    expected += [('transfer', SEEDS[0], 'package_sphere_matched')]
    expected += [('adaptation', SEEDS[0], f'{sphere}_sphere_on_bem') for sphere in spheres]
    if sorted(recorded_groups) != sorted(expected):
        raise ValueError('dipole comparators: the record does not hold the sphere-model fits that the paper reports')
    groups = []
    for experiment, seed, case in recorded_groups:
        what = f'dipole comparators, {experiment}, dataset {seed}, {case}'
        entry, members = summary[experiment][seed][case], by_group[experiment, seed, case]
        errors = {name: columns(members, [f'position_error_mm_{name}']).ravel() for name in names}
        fitted = {name: columns(members, [f'fitted_{axis}_mm_{name}' for axis in 'xyz']) for name in names}
        for name in names:
            evidence.check_metrics(errors[name], entry[name], f'{what}, {name}')
            if entry[name]['windows'] != summary['windows_per_subset']:
                raise ValueError(f"{what}: {entry[name]['windows']} windows, the record names "
                                 f"{summary['windows_per_subset']} per fit")
        change = evidence.check(difference_of_medians(errors['raised_limit'], errors['mne_limit']),
                                entry['median_of_raised_limit_minus_median_of_mne_limit_mm'],
                                f'{what}, change of the median')
        moved = position_change(fitted['raised_limit'], fitted['mne_limit'])
        for key, value in (('median', np.median(moved)), ('p90', np.percentile(moved, 90)),
                           ('maximum', moved.max())):
            evidence.check(value, entry['fitted_position_change_mm'][key], f'{what}, fitted position change, {key}',
                           tolerance=1e-9)
        # The fit with the limit of SciPy is the fit of the recorded experiment, in position and in error.
        positions, truth, recorded, windows = recorded_comparator(evidence, experiment, seed, case)
        if windows != [int(row['window_id']) for row in members]:
            raise ValueError(f'{what}: the recorded experiment fitted other windows')
        if not np.allclose(positions, fitted['mne_limit'], rtol=0, atol=1e-9):
            raise ValueError(f'{what}: the fit with the limit of SciPy differs from the recorded fit')
        if not np.allclose(recorded, errors['mne_limit'], rtol=0, atol=1e-9):
            raise ValueError(f'{what}: the position errors with the limit of SciPy differ from the recorded ones')
        # Every error is the distance of the fitted position from the true source position.
        for name in names:
            if not np.allclose(np.linalg.norm(fitted[name] - truth, axis=1), errors[name], rtol=0, atol=1e-9):
                raise ValueError(f'{what}: the position errors with {name} are not those of the fitted positions')
        # The parameter fits are those of the same sphere in the sphere-fit record.
        sphere = case.removesuffix('_sphere_on_bem') if experiment == 'adaptation' else 'default'
        for name in names:
            fit, other = entry['parameter_fit'][name], spheres[sphere][name]['fit']
            if (fit['evaluations'], fit['stopped_by_itself']) != (other['evaluations'], other['stopped_by_itself']) \
                    or not np.isclose(fit['objective'], other['objective'], rtol=1e-9, atol=0):
                raise ValueError(f'{what}: the parameter fit with {name} did not end as in the sphere-fit record')
        # The fit as MNE-Python runs it ends at the limit; the other one stops by itself.
        unfinished, converged = entry['parameter_fit']['mne_limit'], entry['parameter_fit']['raised_limit']
        if unfinished['stopped_by_itself'] or unfinished['evaluations'] != summary['evaluation_limits']['mne_limit'] \
                or not converged['stopped_by_itself']:
            raise ValueError(f'{what}: the parameter fits did not end as the paper says')
        groups.append(dict(experiment=experiment, dataset_seed=seed, case=case, sphere=sphere,
                           **{name: entry[name] for name in names}, parameter_fit=entry['parameter_fit'],
                           change_of_the_median_mm=change,
                           fitted_position_change_mm=entry['fitted_position_change_mm']))
    # The matched-sphere fit of the split study as Table 3 reports it: mean of the dataset medians.
    split_means = {}
    for policy in POLICIES:
        medians = {name: [group[name]['median_mm'] for group in groups
                          if (group['experiment'], group['case']) == ('split_study', policy)] for name in names}
        evidence.check(np.mean(medians['mne_limit']),
                       split['table11'][policy]['ecd_subset']['dipole']['mean_median_mm'],
                       f'dipole comparators, split study, {policy}, mean of the dataset medians')
        split_means[policy] = {name: dict(mean_of_dataset_medians_mm=float(np.mean(medians[name])),
                                          range_mm=[float(min(medians[name])), float(max(medians[name]))])
                               for name in names}
    return dict(windows_per_subset=summary['windows_per_subset'], evaluation_limits=summary['evaluation_limits'],
                default_limit_fits_are_those_of_the_recorded_experiments=True,
                groups=groups, split_study_means=split_means,
                largest_change_of_a_median_mm=float(max(abs(group['change_of_the_median_mm']) for group in groups)))


def interval(spread, digits=1):
    return f"{spread['median']:.{digits}f} ({spread['minimum']:.{digits}f}–{spread['maximum']:.{digits}f})"


def seed_cell(spread):
    return f"{spread['mean']:.2f} ({spread['range'][0]:.2f}–{spread['range'][1]:.2f})"


def write_markdown(report, destination):
    lines = ['# Reproduced experiment statistics', '',
             'Computed from the bundled CSVs and JSON records. Errors are in millimetres.', '',
             'Intervals in recorded summaries condition on the sampled sectors, datasets, and fitted models.', '',
             '## Localization comparison', '',
             '| Method | Exact mean (seed range) | Region mean (seed range) | Panel |',
             '| --- | --- | --- | --- |']
    headline = report['split']['table11']
    for panel in ('full', 'ecd_subset'):
        for method in METHODS:
            if method not in headline['position'][panel]:
                continue
            cells = []
            for policy in POLICIES:
                entry = headline[policy][panel][method]
                cells.append(f"{entry['mean_median_mm']:.2f} ({entry['range_mm'][0]:.2f}–{entry['range_mm'][1]:.2f})")
            lines.append(f'| {method} | {cells[0]} | {cells[1]} | {panel} |')
    lines += ['', '## Observed training locations', '', '| Dataset seed | Policy | Distinct locations |',
              '| --- | --- | --- |']
    for row in report['split']['observed_composition']:
        lines.append(f"| {row['dataset_seed']} | {row['policy']} | {row['distinct_training_positions']} |")

    def optional(value, digits=2):
        return '—' if value is None else f'{value:.{digits}f}'

    lines += ['', '## Network and lead-field dictionary by stratum', '',
              'Mean of the dataset-seed medians over the seeds with at least 30 test positions in the stratum.', '',
              '| Stratum | Exact: network | Exact: dictionary | Seeds | Region: network | Region: dictionary | Seeds |',
              '| --- | --- | --- | --- | --- | --- | --- |']
    for row in report['split']['strata']:
        cells = []
        for policy in POLICIES:
            entry = row[policy]
            cells += [optional(entry['mean_of_seed_medians_mm']),
                      optional(entry['dictionary_mean_of_seed_medians_mm']),
                      str(len(entry['qualifying_dataset_seeds']))]
        lines.append(f"| {row['stratum']} | " + ' | '.join(cells) + ' |')
    lines += ['', '## Regional contrast on the same test windows', '',
              '| Dataset seed | Windows | Positions | Region minus exact | 95% interval |',
              '| --- | --- | --- | --- | --- |']
    for seed, record in report['split']['paired_split_by_dataset_seed'].items():
        lines.append(f"| {seed} | {record['windows']} | {record['distinct_positions']} | "
                     f"{record['region_minus_position_mm']:.2f} | "
                     f"{record['ci95_mm'][0]:.2f} to {record['ci95_mm'][1]:.2f} |")
    pooled = report['split']['paired_split']['strata']['all']
    lines.append(f"| pooled | {pooled['windows']} | {pooled['seed_position_clusters']} | "
                 f"{pooled['region_minus_position_mm']:.2f} | "
                 f"{pooled['conditional_ci95_mm'][0]:.2f} to {pooled['conditional_ci95_mm'][1]:.2f} |")
    lines += ['', '## Mean training position: distance from the sphere centre (mm)', '',
              '| Policy | ' + ' | '.join(SEEDS) + ' |', '| --- |' + ' --- |' * len(SEEDS)]
    for policy in POLICIES:
        offsets = report['split']['training_mean_offset_mm'][policy]
        lines.append(f'| {policy} | ' + ' | '.join(f'{offsets[seed]:.2f}' for seed in SEEDS) + ' |')
    lines += ['', '## Transfer eligibility census', '',
              '| Stratum | ' + ' | '.join(SEEDS) + ' | Pooled |', '| --- |' + ' --- |' * (len(SEEDS) + 1)]
    for row in report['transfer']['eligibility_census']:
        cells = [f"{row['by_dataset_seed'][seed]['eligible_windows']}/{row['by_dataset_seed'][seed]['total_windows']}"
                 for seed in SEEDS]
        share = '' if row['fraction'] is None else f" ({row['fraction']:.1%})"
        lines.append(f"| {row['stratum']} | " + ' | '.join(cells) +
                     f" | {row['eligible_windows']}/{row['total_windows']}{share} |")
    outside = report['transfer']['cortical_outside_inner_skull']
    cells = [f"{outside['by_dataset_seed'][seed]['outside_windows']}/"
             f"{outside['by_dataset_seed'][seed]['total_windows']}" for seed in SEEDS]
    share = '' if outside['fraction'] is None else f" ({outside['fraction']:.1%})"
    lines.append('| cortical_outside_inner_skull | ' + ' | '.join(cells) +
                 f" | {outside['outside_windows']}/{outside['total_windows']}{share} |")
    lines += ['', '## Transfer errors by stratum', '',
              '| Stratum | Windows | Sphere | Boundary-element model | Difference |', '| --- | --- | --- | --- | --- |']
    for row in report['transfer']['window_strata']:
        lines.append(f"| {row['stratum']} | {row['windows']} | "
                     f"{optional(row['package_sphere_mean_of_seed_medians_mm'])} | "
                     f"{optional(row['bem_mean_of_seed_medians_mm'])} | {optional(row['mean_difference_mm'])} |")
    independent = report['transfer']['independent_sphere_seed_medians_mm']
    lines += ['', '## Independent sphere implementation, median error by dataset seed (mm)', '',
              '| ' + ' | '.join(SEEDS) + ' |', '|' + ' --- |' * len(SEEDS),
              '| ' + ' | '.join(f'{independent[seed]:.2f}' for seed in SEEDS) + ' |']
    lines += ['', '## Transfer eligibility by direction', '',
              '| Package-frame half | All eligible / total | Cortical eligible / total |',
              '| --- | --- | --- |']
    for name, row in report['transfer']['pooled_directional_eligibility'].items():
        cells = [f"{row[key]['eligible_windows']}/{row[key]['total_windows']} ({row[key]['fraction']:.1%})"
                 for key in ('all', 'cortical')]
        lines.append(f'| {name} | {cells[0]} | {cells[1]} |')
    lines += ['', '## Training-location count control', '',
              '| Test windows | Exact | Count-matched | Count-matched minus exact |', '| --- | --- | --- | --- |']
    for panel, entry in report['count_control']['headline'].items():
        lines.append(f"| {panel} | {seed_cell(entry['exact'])} | {seed_cell(entry['count_matched'])} | "
                     f"{seed_cell(entry['count_matched_minus_exact'])} |")
    lines += ['', '## Patch split comparison', '',
              '| Method | Exact mean (seed range) | Region mean (seed range) |', '| --- | --- | --- |']
    patch = report['patch']['headline']
    for method in ('mlp_pooled', 'nearest_training_patch', 'ten_nearest_training_patches', 'training_mean'):
        cells = [f"{patch[policy][method]['mean_median_mm']:.2f} ({patch[policy][method]['range_mm'][0]:.2f}–"
                 f"{patch[policy][method]['range_mm'][1]:.2f})" for policy in POLICIES]
        lines.append(f'| {method} | {cells[0]} | {cells[1]} |')
    lines += ['', '| Dataset seed | Policy | Median best overlap | Minimum footprint gap (mm) | '
              'Smallest distance between used positions of different splits (mm) |',
              '| --- | --- | --- | --- | --- |']
    for row in report['patch']['audits']:
        gap = '—' if row['minimum_footprint_gap_mm'] is None else f"{row['minimum_footprint_gap_mm']:.2f}"
        lines.append(f"| {row['dataset_seed']} | {row['policy']} | {row['best_overlap']['median']:.3f} | "
                     f"{gap} | {row['minimum_used_position_distance_mm']:.2f} |")
    lines += ['', '## Coordinate reuse under shared positions', '',
              '| Shared fraction | Test windows at a training location |', '| --- | --- |']
    for row in report['shared']:
        lines.append(f"| {row['requested_fraction']:g} | {row['test_windows_at_a_training_location']} of "
                     f"{row['test_windows']} ({100 * row['test_window_reuse_fraction']:.1f}%) |")
    lines += ['', '## Head-model adaptation on the boundary-element windows', '',
              '| Arm | Mean of seed medians (range) | Difference from default (range) |', '| --- | --- | --- |']
    for arm, entry in report['adaptation']['bem_windows'].items():
        difference = seed_cell(entry['difference_from_default_mm']) if 'difference_from_default_mm' in entry else '—'
        lines.append(f'| {arm} | {seed_cell(entry)} | {difference} |')
    lines += ['', '| Dipole-fit arm (seed 862 subset) | Median (mm) | 95% interval |', '| --- | --- | --- |']
    for key, entry in report['adaptation']['dipole_subset'].items():
        ci = entry.get('median_ci95_mm')
        lines.append(f"| {key} | {entry['median_mm']:.2f} | "
                     + ('—' if ci is None else f'{ci[0]:.2f} to {ci[1]:.2f}') + ' |')
    lines += ['', '## Generation benchmark', '',
              '| Case | Format | Workers | Windows | Time, s | Windows/s | Peak memory, GiB | Output, GiB |',
              '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for row in report['benchmark']['rows']:
        form = row['format'] + ('' if row['compressed'] else ' (uncompressed)')
        lines.append(f"| {row['case']} | {form} | {row['workers']} | {row['windows']:,} | "
                     f"{interval(row['seconds'])} | {interval(row['windows_per_second'])} | "
                     f"{interval(row['peak_rss_gib'], 2)} | {row['output_gib']:.2f} |")

    rerun, fit = report['validation_rerun'], report['sphere_fit']
    platforms, digests = rerun['two_platforms'], rerun['array_digests_of_the_two_releases']
    lines += ['', '## Numerical validation on two platforms and with two releases', '',
              f"Arrays of the {digests['datasets']} validation datasets with equal digests under releases 1.0.0 and "
              f"1.1.0: {digests['arrays_with_equal_digest']} of {digests['arrays_in_both']}.", '',
              '| Quantity | Linux, release 1.0.0 | Windows, release 1.1.0 |', '| --- | --- | --- |']
    for label, entry, digits in (
            ('Position error, median (mm)', platforms['position_error_mm']['median'], 3),
            ('Position error, 90th percentile (mm)', platforms['position_error_mm']['p90'], 3),
            ('Position error, maximum (mm)', platforms['position_error_mm']['maximum'], 3),
            ('Orientation error, median (degrees)', platforms['orientation_error_median_deg'], 3),
            ('Moment-amplitude error, median', platforms['moment_relative_error_median'], 5)):
        lines.append(f"| {label} | {entry['linux']:.{digits}f} | {entry['windows']:.{digits}f} |")
    for row in platforms['sweep_median_position_error_mm']:
        lines.append(f"| Sweep at {row['snr_db']:g} dB, median position error (mm) | {row['linux']:.3f} | "
                     f"{row['windows']:.3f} |")
    for label, key in (('Patch comparison, median relative field difference', 'patch_median_relative_field_difference'),
                       ('Point-source control, median relative field difference',
                        'point_median_relative_field_difference')):
        lines.append(f"| {label} | {platforms[key]['linux']:.2e} | {platforms[key]['windows']:.2e} |")
    moved = platforms['fitted_position_difference_mm']
    lines += ['', f"The fitted positions of the same {moved['n']} dipoles differ between the platforms by a median of "
              f"{moved['median']:.3f} mm, by more than 0.1 mm for {moved['share_above_0p1_mm']:.1%} of them, and by "
              f"at most {moved['maximum']:.2f} mm."]
    arms = fit['arms']
    lines += ['', '## Parameter fit of the sphere model in MNE-Python', '',
              '| Quantity | ' + ' | '.join(arms) + ' |', '| --- |' + ' --- |' * len(arms)]
    for label, value in (
            ('Function evaluations', lambda arm: str(arm['fit']['evaluations'])),
            ('Optimizer stopped by itself', lambda arm: 'yes' if arm['fit']['stopped_by_itself'] else 'no'),
            ('Objective at the end', lambda arm: f"{arm['fit']['objective']:.3e}"),
            ('Patch, median relative field difference',
             lambda arm: f"{arm['patch_comparison']['patch']['median']:.2e}"),
            ('Point sources, median relative field difference',
             lambda arm: f"{arm['patch_comparison']['point']['median']:.2e}"),
            ('Position error at 35 dB, median (mm)',
             lambda arm: f"{arm['dipole_recovery']['position_error_mm']['median']:.3f}"),
            ('Position error at 35 dB, 90th percentile (mm)',
             lambda arm: f"{arm['dipole_recovery']['position_error_mm']['p90']:.3f}"),
            ('Position error at 35 dB, maximum (mm)',
             lambda arm: f"{arm['dipole_recovery']['position_error_mm']['maximum']:.3f}")):
        lines.append(f'| {label} | ' + ' | '.join(value(arm) for arm in arms.values()) + ' |')

    def span(values, digits=3, scale=1.0):
        return f'{values[0] * scale:.{digits}f}–{values[1] * scale:.{digits}f}'

    lines += ['', 'Ranges over every run of one sphere fit (the unchanged run, the trials with a changed objective and',
              'the other kernels of the linear-algebra library):', '',
              '| Quantity | ' + ' | '.join(arms) + ' |', '| --- |' + ' --- |' * len(arms)]
    runs = fit['all_runs']
    for label, value in (
            ('Runs with a dipole recovery', lambda item: str(item['runs_with_a_dipole_recovery'])),
            ('Parameter fits that stopped by themselves',
             lambda item: f"{item['parameter_fits_that_stopped_by_themselves']} of {item['parameter_fits']}"),
            ('Patch, median relative field difference (10⁻³)',
             lambda item: span(item['median_relative_field_difference']['patch'], 2, 1e3)),
            ('Point sources, median relative field difference (10⁻³)',
             lambda item: span(item['median_relative_field_difference']['point'], 2, 1e3)),
            ('Position error at 35 dB, median (mm)', lambda item: span(item['position_error_mm']['median'])),
            ('Position error at 35 dB, 90th percentile (mm)', lambda item: span(item['position_error_mm']['p90'])),
            ('Position error at 35 dB, maximum (mm)', lambda item: span(item['position_error_mm']['maximum']))):
        lines.append(f'| {label} | ' + ' | '.join(value(runs[name]) for name in arms) + ' |')
    kernels = fit['library_kernels']
    own = fit['small_tolerance']['kernel_of_the_main_process']
    lines += ['', f'Other kernels of the linear-algebra library (kernel of the main run: {own}):', '',
              '| Kernel | Objective at the second trial point | Default limit: objective at the end | '
              'Default limit: patch | Default limit: median position error (mm) | Raised limit: evaluations | '
              'Raised limit: patch | Raised limit: median position error (mm) |',
              '| --- | --- | --- | --- | --- | --- | --- | --- |']
    second = arms['mne_limit']['fit']['first_evaluations'][1]['objective']
    lines.append(f"| {own} (main run) | {second:.5f} | {arms['mne_limit']['fit']['objective']:.3e} | "
                 f"{arms['mne_limit']['patch_comparison']['patch']['median']:.2e} | "
                 f"{arms['mne_limit']['dipole_recovery']['position_error_mm']['median']:.3f} | "
                 f"{arms['raised_limit']['fit']['evaluations']} | "
                 f"{arms['raised_limit']['patch_comparison']['patch']['median']:.2e} | "
                 f"{arms['raised_limit']['dipole_recovery']['position_error_mm']['median']:.3f} |")
    for kernel, entry in kernels['arms'].items():
        limit, raised = entry['arms']['mne_limit'], entry['arms']['raised_limit']
        lines.append(f"| {kernel} | {limit['fit']['first_evaluations'][1]['objective']:.5f} | "
                     f"{limit['fit']['objective']:.3e} | {limit['patch_comparison']['patch']['median']:.2e} | "
                     f"{limit['dipole_recovery']['position_error_mm']['median']:.3f} | "
                     f"{raised['fit']['evaluations']} | "
                     f"{raised['patch_comparison']['patch']['median']:.2e} | "
                     f"{raised['dipole_recovery']['position_error_mm']['median']:.3f} |")
    lines += ['', f"Spheres of the adaptation experiment (point sources; dipole recovery at "
              f"{fit['spheres']['snr_db_range'][0]:g} to {fit['spheres']['snr_db_range'][1]:g} dB):", '',
              '| Sphere | Default limit: evaluations | Raised limit: evaluations | Field difference, default limit | '
              'Field difference, raised limit | Median position error, default limit (mm) | '
              'Median position error, raised limit (mm) |', '| --- | --- | --- | --- | --- | --- | --- |']
    for sphere, entry in fit['spheres']['variants'].items():
        limit, raised = entry['mne_limit'], entry['raised_limit']
        lines.append(f"| {sphere} | {limit['fit']['evaluations']} | {raised['fit']['evaluations']} | "
                     f"{limit['point_field_difference']['median']:.2e} | "
                     f"{raised['point_field_difference']['median']:.2e} | "
                     f"{limit['position_error_mm']['median']:.3f} | {raised['position_error_mm']['median']:.3f} |")
    small = fit['small_tolerance']
    lines += ['', f"Dipole fits ended at a trust-region radius of {small['final_radius_m']:g} m (MNE-Python: "
              f"{small['final_radius_of_mne_python_m']:g} m), ranges over the runs of each sphere fit:", '',
              '| Quantity | ' + ' | '.join(arms) + ' |', '| --- |' + ' --- |' * len(arms)]
    for label, value in (
            ('Runs', lambda item: str(item['runs'])),
            ('Kernels', lambda item: ', '.join(item['kernels'])),
            ('Parameter fits that stopped by themselves',
             lambda item: f"{item['runs_in_which_the_parameter_fit_stopped_by_itself']} of {item['runs']}"),
            ('Dipole fits that stopped by themselves',
             lambda item: f"{item['dipole_fits_stopped_by_themselves']} of {item['dipole_fits']}"),
            ('Position error at 35 dB, median (mm)', lambda item: span(item['position_error_median_mm'], 4)),
            ('Position error at 35 dB, 90th percentile (mm)', lambda item: span(item['position_error_p90_mm'], 4)),
            ('Position error at 35 dB, maximum (mm)', lambda item: span(item['position_error_maximum_mm'], 4))):
        lines.append(f'| {label} | ' + ' | '.join(value(small['ranges'][name]) for name in arms) + ' |')
    lines += ['', '## Dipole-fit comparators with a converged parameter fit', '',
              '| Experiment | Dataset seed | Case | Windows | Median, limit of SciPy (mm) | '
              'Median, converged fit (mm) | Change (mm) |', '| --- | --- | --- | --- | --- | --- | --- |']
    for group in report['comparators']['groups']:
        lines.append(f"| {group['experiment']} | {group['dataset_seed']} | {group['case']} | "
                     f"{group['mne_limit']['windows']} | {group['mne_limit']['median_mm']:.3f} | "
                     f"{group['raised_limit']['median_mm']:.3f} | {group['change_of_the_median_mm']:+.3f} |")
    lines += ['', 'The machine-readable report holds every part of the experiment tables: besides the numbers above,',
              'the distances between splits, the records of each fitted network, the paired comparator intervals,',
              'the patch strata, the checks of the pairing with the template head, the paired-window statistics,',
              'the stage times of the benchmark, the trials of the parameter fit and of the dipole fits, the fixed',
              'settings of the experiments, and the hashes of the files that were read.', '']
    with destination.open('w', encoding='utf-8', newline='\n') as file:
        file.write('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--evidence', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('choose a new output directory')
    evidence = Evidence(args.evidence.resolve())
    report = dict(split=split_tables(evidence), count_control=count_control_tables(evidence),
                  patch=patch_tables(evidence), transfer=transfer_tables(evidence),
                  adaptation=adaptation_tables(evidence),
                  continuity={name: evidence.json(f'continuity/{name}.json')
                              for name in ('v100-continuity', 'v11-continuity')},
                  shared=evidence.json('shared-position-v11/summary.json'),
                  benchmark=benchmark_table(evidence),
                  settings=experiment_settings(evidence))
    report['validation_rerun'], rerun_results = validation_rerun_tables(evidence)
    report['settings']['dipole_fit_subset_windows']['transfer_and_adaptation'] = report['adaptation']['dipole_windows']
    rerun = Path('validation-rerun-v11')
    report['sphere_fit'] = sphere_fit_tables(
        evidence, evidence.json(rerun / 'generated_arrays_release-1.1.0.json')['datasets'], rerun_results)
    report['comparators'] = comparator_tables(evidence, report['sphere_fit'], report['split'])
    experiments = {}
    for folder in ('split-study-v11', 'training-count-control-v11', 'patch-split-study-v11',
                   'shared-position-v11', 'head-transfer-v11', 'head-adaptation-v11',
                   'generation-benchmark-v11', 'sphere-fit-convergence-v11', 'dipole-comparators-converged-v11'):
        record = evidence.json(f'{folder}/provenance.json')
        experiments[folder] = dict(commit=record.get('commit'), package_tree=record.get('package_tree'),
                                   arguments=record.get('arguments'))
    report['provenance'] = dict(evidence_sha256=dict(sorted(evidence.hashes.items())),
                                experiments=experiments,
                                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                comparisons_of_recomputed_numbers_with_the_records=evidence.checked,
                                bootstrap='recorded position-cluster intervals, no re-estimation',
                                pooling='model seeds pooled within dataset seed, then unweighted mean of dataset '
                                        'medians')
    args.output_dir.mkdir(parents=True)
    with (args.output_dir / 'revision_tables.json').open('w', encoding='utf-8', newline='\n') as file:
        json.dump(report, file, indent=2, allow_nan=False)
        file.write('\n')
    write_markdown(report, args.output_dir / 'Revision_Tables.md')
    print(f"Reproduced the tables from {len(evidence.hashes)} evidence files; {evidence.checked} comparisons of a "
          f"recomputed number with a recorded one were made, and all agree.")


if __name__ == '__main__':
    main()
