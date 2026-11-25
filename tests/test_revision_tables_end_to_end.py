"""The table driver on a miniature evidence folder: every table, every record check, and the counter of checks.

``mini_evidence.build`` writes the folder and returns the values that the tables must show, computed without
the driver. The tests run the driver's ``main`` on the folder and compare. Then they change one value of the
folder at a time and expect the driver to stop with a message that names the quantity; a change within the
tolerance of a comparison must not stop it.
"""

import contextlib
import csv
import hashlib
import io
import json
import re
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import make_revision_tables as tables
import mini_evidence

SEEDS, FITS, POLICIES = tables.SEEDS, tables.FITS, tables.POLICIES
DATASETS = [(seed, policy) for seed in SEEDS for policy in POLICIES]
TABLE_FUNCTIONS = ('split_tables', 'count_control_tables', 'patch_tables', 'transfer_tables', 'adaptation_tables',
                   'benchmark_table', 'experiment_settings', 'validation_rerun_tables', 'sphere_fit_tables',
                   'comparator_tables')
# Comparisons of a recomputed number with a recorded one that the driver makes on the miniature folder.
COMPARISONS = dict(split_tables=4338, count_control_tables=102, patch_tables=840, transfer_tables=385,
                   adaptation_tables=804, benchmark_table=0, experiment_settings=0, validation_rerun_tables=18,
                   sphere_fit_tables=422, comparator_tables=122)
# The methods of the two panels of the split study and of the patch study, each fitted network on its own
# included, and the statistics of a summary: written out once more for the tests.
FULL_METHODS = ('mlp_pooled', 'dictionary', 'knn1', 'knn10', 'training_mean', 'mlp_862', 'mlp_863')
SUBSET_METHODS = ('mlp_pooled', 'dictionary', 'knn1', 'knn10', 'dipole', 'training_mean', 'mlp_862', 'mlp_863')
PATCH_METHODS = ('mlp_pooled', 'nearest_training_patch', 'ten_nearest_training_patches', 'training_mean', 'mlp_862',
                 'mlp_863')
STATISTICS = ('median_mm', 'mean_mm', 'p90_mm')
# The single networks of the adaptation experiment: fine-tuned or trained on the boundary-element windows alone.
SINGLE_NETWORKS = [(kind, size, fit) for size in mini_evidence.SIZES for fit in FITS
                   for kind in ('finetuned', 'scratch')]


# --------------------------------------------------------------------------- running the driver

def run_driver(evidence, output, known=None, changed=()):
    """Run the driver's main() on an evidence folder.

    The result lists the comparisons that passed and holds, for every table function, the files that it read,
    what it returned and how many comparisons it made. With ``known``, the record of the unchanged folder,
    and the ``changed`` files, a table function that read none of the changed files is not run again: it
    returns its recorded result, which is what it would compute.
    """
    passed, reads, functions = [], [], {}
    check, read = tables.Evidence.check, tables.Evidence.bytes

    def counted(self, *arguments, **options):
        result = check(self, *arguments, **options)
        passed.append(arguments[2])
        return result

    def logged(self, relative):
        reads.append(Path(relative).as_posix())
        return read(self, relative)

    def recorded(name, function):
        def call(*arguments):
            first_read, first_check = len(reads), len(passed)
            result = function(*arguments)
            functions[name] = SimpleNamespace(files=set(reads[first_read:]), result=result,
                                              checks=len(passed) - first_check)
            return result
        return call

    printed = io.StringIO()
    with pytest.MonkeyPatch.context() as patch, contextlib.redirect_stdout(printed):
        patch.setattr(sys, 'argv', ['make_revision_tables.py', '--evidence', str(evidence),
                                    '--output-dir', str(output)])
        patch.setattr(tables.Evidence, 'check', counted)
        patch.setattr(tables.Evidence, 'bytes', logged)
        for name in TABLE_FUNCTIONS:
            if known is not None and known[name].files.isdisjoint(changed):
                patch.setattr(tables, name, lambda *arguments, result=known[name].result: result)
            else:
                patch.setattr(tables, name, recorded(name, getattr(tables, name)))
        tables.main()
    if known is None:
        # The sphere-fit tables take the results of the rerun, so they depend on its files as well. The tables
        # of the dipole comparators take the sphere-fit tables and the tables of the split study.
        functions['sphere_fit_tables'].files |= functions['validation_rerun_tables'].files
        functions['comparator_tables'].files |= functions['sphere_fit_tables'].files | functions['split_tables'].files
    return SimpleNamespace(passed=passed, functions=functions, printed=printed.getvalue())


def file_hashes(folder):
    return {path.relative_to(folder).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(folder.rglob('*')) if path.is_file()}


@pytest.fixture(scope='module')
def miniature(tmp_path_factory):
    """The miniature folder, the values it must give, and what the driver made of it."""
    base = tmp_path_factory.mktemp('miniature')
    expected = mini_evidence.build(base / 'evidence')
    hashes = file_hashes(base / 'evidence')
    output = base / 'new' / 'tables'            # the driver makes the folder and the folder above it
    run = run_driver(base / 'evidence', output)
    report = json.loads((output / 'revision_tables.json').read_text(encoding='utf-8'))
    lines = (output / 'Revision_Tables.md').read_text(encoding='utf-8').splitlines()
    yield SimpleNamespace(evidence=base / 'evidence', output=output, expected=expected, run=run, report=report,
                          lines=lines, hashes=hashes)
    assert file_hashes(base / 'evidence') == hashes, 'a test changed the unchanged folder'


def close(value):
    """A number that the builder computed: equal to within rounding."""
    return pytest.approx(value, rel=0, abs=1e-9)


def same(actual, wanted):
    """Equal to within rounding, where None stands for a cell that the tables leave empty."""
    if actual is None or wanted is None:
        return actual is wanted
    return actual == close(wanted)


def seed_values(by_seed):
    return [by_seed[seed] for seed in SEEDS]


# --------------------------------------------------------------------------- what the driver read and counted

def test_driver_reads_every_file_of_the_folder_and_records_its_hash(miniature):
    provenance = miniature.report['provenance']
    assert list(provenance['evidence_sha256']) == miniature.expected['files']
    assert provenance['evidence_sha256'] == miniature.hashes
    assert len(miniature.hashes) == 154
    assert provenance['script_sha256'] == hashlib.sha256(Path(tables.__file__).read_bytes()).hexdigest()
    for folder, commit in miniature.expected['commits'].items():
        assert provenance['experiments'][folder]['commit'] == commit
        assert provenance['experiments'][folder]['arguments'][0] == '--output-dir'
    assert list(provenance['experiments']) == list(miniature.expected['commits'])
    assert provenance['experiments']['split-study-v11']['package_tree'] is None
    assert provenance['experiments']['training-count-control-v11']['package_tree'] == 'a' * 40


def test_output_files_have_the_same_bytes_on_every_platform(miniature):
    assert sorted(path.name for path in miniature.output.iterdir()) == ['Revision_Tables.md', 'revision_tables.json']
    report = (miniature.output / 'revision_tables.json').read_bytes()
    assert report.decode('utf-8') == json.dumps(miniature.report, indent=2) + '\n'
    markdown = (miniature.output / 'Revision_Tables.md').read_bytes()
    assert b'\r' not in report + markdown and markdown.endswith(b'that were read.\n')
    assert markdown.decode('utf-8').splitlines() == miniature.lines


def test_counter_of_comparisons_is_the_number_of_checks_that_passed(miniature):
    run, total = miniature.run, sum(COMPARISONS.values())
    assert {name: record.checks for name, record in run.functions.items()} == COMPARISONS
    assert miniature.report['provenance']['comparisons_of_recomputed_numbers_with_the_records'] == len(run.passed)
    assert len(run.passed) == total == 7031
    assert len(set(run.passed)) == total                # no comparison is made twice
    assert run.printed == (f'Reproduced the tables from 154 evidence files; {total} comparisons of a recomputed '
                           f'number with a recorded one were made, and all agree.\n')


def test_every_method_is_compared_by_stratum_and_every_single_network_on_its_own(miniature):
    """The comparisons by stratum and network by network that the driver made, named from the builder's values."""
    passed, expected = miniature.run.passed, miniature.expected
    # Split study: every method of a panel and each fitted network, in every stratum of the panel. The messages
    # for the panel of all windows do not name the panel.
    sizes = expected['split']['strata_sizes']
    wanted = [f'split study, dataset {seed}, {policy}{where}, stratum {label}, {method}, {key}'
              for seed, policy in DATASETS
              for panel, methods, where in (('full', FULL_METHODS, ''), ('ecd_subset', SUBSET_METHODS, ', ecd_subset'))
              for label in sizes[seed, policy][panel] for method in methods for key in STATISTICS]
    assert sorted(name for name in passed if name.startswith('split study, dataset') and ', stratum ' in name) == \
        sorted(wanted)
    # A stratum is recorded for a panel if it holds a window of the panel. The subset lacks strata of its
    # dataset, and in one dataset it holds two windows of one position.
    assert [len(sizes[dataset]['full']) for dataset in DATASETS] == [15, 15, 14, 15, 14, 15]
    assert [len(sizes[dataset]['ecd_subset']) for dataset in DATASETS] == [13, 15, 13, 14, 13, 15]
    assert len(wanted) == 3 * (7 * 88 + 8 * 83)
    for dataset in DATASETS:
        full, subset = sizes[dataset]['full'], sizes[dataset]['ecd_subset']
        assert [label for label in full if label in subset] == list(subset), dataset
        assert all(0 < positions <= windows for windows, positions in (*full.values(), *subset.values())), dataset
        assert all(subset[label][0] <= full[label][0] for label in subset), dataset
    assert sizes['862', 'position']['full']['distance_10_20'] == (1, 1)
    assert 'distance_10_20' not in sizes['862', 'position']['ecd_subset']
    assert sizes['864', 'region']['ecd_subset']['population_0'] == (6, 5)
    assert [dataset for dataset in DATASETS
            if any(windows != positions for windows, positions in sizes[dataset]['ecd_subset'].values())] == [
                ('864', 'region')]

    # Patch study: every method and each fitted network, in every stratum.
    wanted = [f'patch study, dataset {seed}, {policy}, stratum {label}, {method}, {key}' for seed, policy in DATASETS
              for label in expected['patch']['strata'] for method in PATCH_METHODS for key in STATISTICS]
    assert sorted(name for name in passed if name.startswith('patch study, dataset') and ', stratum ' in name) == \
        sorted(wanted)
    assert len(wanted) == 6 * 6 * 6 * 3

    # Adaptation: each fine-tuned network and each network trained on the boundary-element windows alone.
    wanted = [f'adaptation, dataset {seed}, {kind}_{size}, network {fit}, {key}' for seed in SEEDS
              for kind, size, fit in SINGLE_NETWORKS for key in STATISTICS]
    assert sorted(name for name in passed if name.startswith('adaptation, dataset') and ', network ' in name) == \
        sorted(wanted)
    assert len(wanted) == 3 * 12 * 3
    # The single networks are those of the arms that were fine-tuned or trained alone, one per model seed.
    assert {f'{kind}_{size}' for kind, size, _ in SINGLE_NETWORKS} == {
        arm for arm in mini_evidence.ARMS if arm not in mini_evidence.VARIANTS}
    assert len(SINGLE_NETWORKS) == len(set(SINGLE_NETWORKS)) == 6 * len(FITS)


# --------------------------------------------------------------------------- the tables

def test_split_tables_hold_the_headline_the_strata_and_the_values_without_a_record(miniature):
    split, expected = miniature.report['split'], miniature.expected['split']
    assert split['table11'] == expected['headline']
    assert set(expected['headline']['region']['ecd_subset']) - set(expected['headline']['region']['full']) == {'dipole'}

    # Network and dictionary by stratum: the mean over the datasets with at least 30 positions in the stratum.
    assert [row['stratum'] for row in split['strata']] == [row['stratum'] for row in expected['strata']]
    assert len(split['strata']) == 17
    for row, wanted in zip(split['strata'], expected['strata']):
        for policy in POLICIES:
            assert row[policy]['qualifying_dataset_seeds'] == wanted[policy]['seeds'], row['stratum']
            assert same(row[policy]['mean_of_seed_medians_mm'], wanted[policy]['network']), row['stratum']
            assert same(row[policy]['dictionary_mean_of_seed_medians_mm'], wanted[policy]['dictionary']), row['stratum']
    by_label = {row['stratum']: row for row in expected['strata']}
    # The folder holds both sides of the rule: 30 positions count, and 29 positions do not although they
    # have more than 30 windows. A stratum can also be missing in a dataset or under a policy.
    below = by_label['below_montage']
    assert below['position']['positions'] == {'862': 30, '863': 34, '864': 29}
    assert below['position']['windows']['864'] > 30 and below['position']['seeds'] == ['862', '863']
    assert below['region']['positions'] == {'862': 29, '863': 31, '864': 30}
    assert below['region']['windows']['862'] > 30 and below['region']['seeds'] == ['863', '864']
    assert by_label['population_0']['region']['positions']['864'] == 30
    assert by_label['population_0']['region']['seeds'] == list(SEEDS)
    assert by_label['distance_10_20']['position']['positions'] == {'862': 1, '863': None, '864': None}
    assert by_label['distance_0_5']['region']['seeds'] == [] and by_label['distance_0_5']['region']['network'] is None
    assert by_label['radius_1_30_45']['position']['network'] is None

    assert [(row['dataset_seed'], row['policy']) for row in split['observed_composition']] == DATASETS
    for row in split['observed_composition']:
        assert row['distinct_training_positions'] == expected['training_positions'][row['dataset_seed'], row['policy']]
        assert set(row['cells']) == {'train', 'validation', 'test'}

    # Share of the windows with an error below the distance to the nearest training location.
    floor = {(row['dataset_seed'], row['policy'], row['method']): row for row in split['below_training_location_floor']}
    assert len(floor) == len(split['below_training_location_floor']) == 6 * 5
    for dataset, by_method in expected['below_floor'].items():
        for method, share in by_method.items():
            row = floor[(*dataset, method)]
            assert row['below_nearest_training_fraction'] == close(share), (dataset, method)
            assert (row['windows'], row['fits']) == (72, 2 if method == 'mlp_pooled' else 1)
    assert 0.2 < expected['below_floor']['862', 'position']['knn1'] < 0.8
    # One error of the first dataset equals the distance, and one equals the distance less the margin of 1e-8 mm.
    assert expected['floor_cases'] == [1, 1]

    assert len(split['per_fit']) == 6 * len(FITS)
    assert [(row['dataset_seed'], row['policy'], row['fit_seed']) for row in split['per_fit']] == [
        (*dataset, fit) for dataset in DATASETS for fit in FITS]
    statistics = 0
    for row in split['per_fit']:
        dataset = row['dataset_seed'], row['policy']
        epoch, median = expected['per_fit'][dataset][row['fit_seed']]
        assert row['fit']['best_epoch'] == epoch and row['metrics']['median_mm'] == close(median)
        # The record of the network holds its strata on all windows, which the driver compared one by one with
        # the windows of the stratum: the strata of the dataset in the order of the record, each with its
        # windows, its positions and the three statistics of this network.
        strata, sizes = expected['per_fit_strata'][dataset][row['fit_seed']], expected['strata_sizes'][dataset]['full']
        assert list(row['metrics']['strata']) == list(strata) == list(sizes), dataset
        for label, wanted in strata.items():
            cell = row['metrics']['strata'][label]
            assert (cell['windows'], cell['positions']) == (wanted['windows'], wanted['positions']) == sizes[label]
            assert [cell[key] for key in STATISTICS] == close([wanted[key] for key in STATISTICS]), (dataset, label)
            statistics += len(STATISTICS)
    assert statistics == len(FITS) * 3 * 88
    for dataset in DATASETS:
        # The two networks of a dataset differ in every statistic of every stratum, and in a stratum with more
        # than one window the three statistics of a network differ from each other.
        first, second = (expected['per_fit_strata'][dataset][fit] for fit in FITS)
        assert all(first[label][key] != second[label][key] for label in first for key in STATISTICS), dataset
        assert all(len({entry[key] for key in STATISTICS}) == (3 if entry['windows'] > 1 else 1)
                   for entry in (*first.values(), *second.values())), dataset
    assert [(row['dataset_seed'], row['policy']) for row in split['distance']] == DATASETS
    for row in split['distance']:
        eligible, used = expected['distance'][row['dataset_seed'], row['policy']]
        assert (row['eligible']['all']['min'], row['used']['all']['min']) == close((eligible, used))
        assert set(row['eligible']) == {'all', 'cortical', 'subcortical'}

    # The paired contrasts with the comparators are differences of the medians of the headline.
    assert len(split['paired_comparators']) == 6 * (3 + 4)
    for row in split['paired_comparators']:
        medians = expected['headline'][row['policy']][row['panel']]
        difference = (medians['mlp_pooled']['dataset_seed_medians_mm'][row['dataset_seed']]
                      - medians[row['comparator']]['dataset_seed_medians_mm'][row['dataset_seed']])
        assert row['median_difference_mm'] == close(difference)
        assert len(row['ci95_mm']) == 2 and row['interval_scope'].startswith('conditional')

    for policy in POLICIES:
        offsets = seed_values(expected['mean_offset'][policy])
        assert seed_values(split['training_mean_offset_mm'][policy]) == close(offsets)
    assert split['continuity_geometry'] == dict(
        training_mean_offset_mm=close(expected['mean_offset']['position']['862']),
        median_test_radius_mm=close(expected['median_test_radius']))
    assert 60 < expected['median_test_radius'] < 80

    # The two training regimes on the same windows: per dataset, and over the windows of all datasets.
    for seed in SEEDS:
        assert split['paired_split_by_dataset_seed'][seed] == expected['paired'][seed]
    assert split['paired_split']['strata']['all'] == expected['paired']['pooled']
    assert expected['paired']['pooled']['windows'] == 16 + 14 + 15
    assert expected['paired']['862']['distinct_positions'] < expected['paired']['862']['windows']
    assert set(split['adequacy']) == set(SEEDS)


def test_count_control_tables_hold_the_headline_and_the_dataset_records(miniature):
    count, expected = miniature.report['count_control'], miniature.expected['count_control']
    assert count['headline'] == expected
    assert set(expected['all_test_windows']) == {'exact', 'count_matched', 'count_matched_minus_exact'}
    assert set(expected['same_test_windows']) == {'exact', 'count_matched', 'regional', 'count_matched_minus_exact',
                                                  'regional_minus_count_matched'}
    for number, seed in enumerate(SEEDS):
        record = count['by_dataset_seed'][seed]
        assert record['model_seed'] == 862
        for panel, entries in expected.items():
            for name, entry in entries.items():
                key = 'median_difference_mm' if '_minus_' in name else 'median_mm'
                assert record[panel][name][key] == entry['dataset_seed_values'][number]
        assert record['all_test_windows']['exact']['windows'] == 72
        assert record['same_test_windows']['regional']['windows'] == (16, 14, 15)[number]


def test_patch_tables_hold_the_headline_the_audits_and_the_strata(miniature):
    patch, expected = miniature.report['patch'], miniature.expected['patch']
    assert patch['headline'] == expected['headline']
    assert [(row['dataset_seed'], row['policy']) for row in patch['audits']] == DATASETS
    for row in patch['audits']:
        wanted = expected['audits'][row['dataset_seed'], row['policy']]
        assert row['scored_test_patches'] == wanted['scored'] == (72 if row['policy'] == 'position' else 70)
        assert row['distinct_training_centres'] == wanted['centres']
        assert row['best_overlap']['median'] == close(wanted['overlap'])
        assert same(row['minimum_footprint_gap_mm'], wanted['gap'])
        assert same(row['minimum_support_clearance_mm'], wanted['clearance'])
        assert row['minimum_used_position_distance_mm'] == close(wanted['used_distance'])
        assert row['nearest_centre_euclidean_mm']['n'] == 72
        assert (row['containment'] is None) == (row['minimum_footprint_gap_mm'] is None)
        assert (row['containment'] is None) == (row['policy'] == 'position')
    # The study record of one region dataset differs from the dataset report by rounding, which is accepted.
    first = expected['audits']['862', 'region']
    assert 0 < first['clearance'] - first['used_distance'] < 1e-6

    assert [row['stratum'] for row in patch['strata']] == list(expected['strata'])
    assert len(patch['strata']) == 6
    for row in patch['strata']:
        for policy in POLICIES:
            by_seed, mean, limits = expected['strata'][row['stratum']][policy]
            assert row[policy]['dataset_seed_values'] == close(seed_values(by_seed))
            assert row[policy]['mean'] == close(mean) and row[policy]['range'] == close(limits)
    for policy in POLICIES:
        by_seed, mean, limits = expected['below_centre'][policy]
        cell = patch['below_nearest_training_centre'][policy]
        assert cell['dataset_seed_values'] == close(seed_values(by_seed))
        assert cell['mean'] == close(mean) and cell['range'] == close(limits)
        assert 0.2 < limits[0] < limits[1] < 0.9
    for seed in SEEDS:
        for policy in POLICIES:
            lookup = patch['paired_mlp_minus_lookup'][seed][policy]
            medians = expected['headline'][policy]
            for name in ('nearest_training_patch', 'ten_nearest_training_patches'):
                assert lookup[name]['median_difference_mm'] == close(
                    medians['mlp_pooled']['dataset_seed_medians_mm'][seed]
                    - medians[name]['dataset_seed_medians_mm'][seed])
            assert set(patch['fits'][seed][policy]) == {'mlp_862', 'mlp_863'}
    assert set(patch['adequacy']) == set(SEEDS)


def test_transfer_tables_hold_the_strata_the_census_and_the_paired_statistics(miniature):
    transfer, expected = miniature.report['transfer'], miniature.expected['transfer']
    assert transfer['summary']['paired_headline'] == expected['headline']

    # Strata of the summary: the mean over the datasets with at least 30 positions in the stratum.
    assert [row['stratum'] for row in transfer['strata']] == [row['stratum'] for row in expected['strata']]
    for row, wanted in zip(transfer['strata'], expected['strata']):
        assert row['qualifying_dataset_seeds'] == wanted['seeds'] and row['windows'] == wanted['windows']
        assert same(row['package_sphere_median_mm'], wanted['sphere']), row['stratum']
        assert same(row['bem_median_mm'], wanted['bem']) and same(row['median_difference_mm'], wanted['difference'])
    by_label = {row['stratum']: row for row in expected['strata']}
    assert by_label['population_0']['positions'] == {'862': 32, '863': 30, '864': 31}
    assert by_label['population_0']['seeds'] == list(SEEDS)
    assert by_label['population_1']['positions'] == {'862': 30, '863': 33, '864': 29}
    assert by_label['population_1']['window_counts']['864'] > 30 and by_label['population_1']['seeds'] == ['862', '863']
    assert by_label['radius_1_30_45']['seeds'] == [] and by_label['radius_1_30_45']['windows'] == 0

    # Strata of the windows: averaged only if every dataset has at least 30 positions in the stratum.
    names = [row['stratum'] for row in expected['window_strata']]
    assert [row['stratum'] for row in transfer['window_strata']] == names
    assert len(transfer['window_strata']) == 13
    for row, wanted in zip(transfer['window_strata'], expected['window_strata']):
        assert row['windows'] == wanted['windows'], row['stratum']
        assert row['smallest_number_of_positions_in_a_dataset'] == wanted['positions'], row['stratum']
        assert same(row['package_sphere_mean_of_seed_medians_mm'], wanted['sphere']), row['stratum']
        assert same(row['bem_mean_of_seed_medians_mm'], wanted['bem']), row['stratum']
        assert same(row['mean_difference_mm'], wanted['difference']), row['stratum']
    by_name = {row['stratum']: row for row in expected['window_strata']}
    assert by_name['population_0']['positions'] == 30 and by_name['population_0']['sphere'] is not None
    assert by_name['population_1']['positions'] == 29 and by_name['population_1']['sphere'] is None
    assert by_name['population_1']['windows'] == 31 + 35 + 31
    averaged = [name for name, row in by_name.items() if row['sphere'] is not None]
    assert averaged == ['population_0', 'snr_15_20', 'clearance_10_20']
    for family in ('population', 'radius', 'snr', 'clearance'):
        assert sum(row['windows'] for name, row in by_name.items() if name.startswith(family)) == 65 + 66 + 64

    # The census of the eligible windows, per dataset and over the three datasets.
    assert [row['stratum'] for row in transfer['eligibility_census']] == list(expected['census_counts']['862'])
    assert len(transfer['eligibility_census']) == 12
    for row in transfer['eligibility_census']:
        counts = {seed: expected['census_counts'][seed][row['stratum']] for seed in SEEDS}
        assert {seed: (cell['eligible_windows'], cell['total_windows'])
                for seed, cell in row['by_dataset_seed'].items()} == counts
        eligible, total = (sum(pair[index] for pair in counts.values()) for index in (0, 1))
        assert (row['eligible_windows'], row['total_windows']) == (eligible, total)
        assert row['fraction'] == close(eligible / total)
    assert expected['census_counts']['864']['all'] == (64, 72)
    assert expected['census_counts']['862']['subcortical'] == (31, 32)
    outside = transfer['cortical_outside_inner_skull']
    assert {seed: (cell['outside_windows'], cell['total_windows'])
            for seed, cell in outside['by_dataset_seed'].items()} == expected['outside']
    counted, total = (sum(pair[index] for pair in expected['outside'].values()) for index in (0, 1))
    assert (outside['outside_windows'], outside['total_windows']) == (counted, total)
    assert outside['fraction'] == close(counted / total) and counted > 0
    for seed in SEEDS:
        recorded = transfer['eligibility'][seed]['strata']
        assert (recorded['population_0']['eligible_windows'], recorded['population_0']['windows']) == \
            expected['census_counts'][seed]['cortical']

    # Values without a record: the independent sphere, the directions, and the statistics of the windows.
    independent = seed_values(expected['independent'])
    assert seed_values(transfer['independent_sphere_seed_medians_mm']) == close(independent)
    assert transfer['independent_sphere_mean_seed_median_mm'] == close(np.mean(independent))
    assert len(transfer['directional_eligibility']) == 12
    pooled = {direction: dict(everything=[0, 0], cortical=[0, 0])
              for direction in ('anterior', 'posterior', 'superior', 'inferior')}
    for row in transfer['directional_eligibility']:
        wanted = expected['directional'][row['dataset_seed'], row['direction']]
        for key, name in (('all', 'everything'), ('cortical', 'cortical')):
            eligible, total = wanted[name]
            assert row[key] == dict(eligible_windows=eligible, total_windows=total, fraction=close(eligible / total))
            pooled[row['direction']][name][0] += eligible
            pooled[row['direction']][name][1] += total
        assert row['eligible_cortical_difference_of_medians_mm'] == close(wanted['difference'])
    assert list(transfer['pooled_directional_eligibility']) == list(pooled)
    for direction, wanted in pooled.items():
        for key, name in (('all', 'everything'), ('cortical', 'cortical')):
            eligible, total = wanted[name]
            assert transfer['pooled_directional_eligibility'][direction][key] == dict(
                eligible_windows=eligible, total_windows=total, fraction=close(eligible / total))
    for row in transfer['paired_window_statistics']:
        wanted = expected['window_statistics'][row['dataset_seed']]
        assert (row['windows'], row['fits']) == (wanted['windows'], 2)
        assert row['fraction_bem_error_larger'] == close(wanted['larger'])
        assert row['median_prediction_displacement_mm'] == close(wanted['displacement'])
        assert row['median_of_window_error_differences_mm'] == close(wanted['difference'])
    assert [row['dataset_seed'] for row in transfer['paired_window_statistics']] == list(SEEDS)
    assert set(transfer['gates']) == set(SEEDS) and transfer['gates']['863']['check5_model']['mlp_863']['passes']


def test_adaptation_tables_hold_the_headline_the_fine_tuning_sizes_and_the_dipole_fits(miniature):
    adaptation, expected = miniature.report['adaptation'], miniature.expected['adaptation']
    assert list(adaptation['bem_windows']) == list(mini_evidence.ARMS) and len(mini_evidence.ARMS) == 11
    for arm, entry in adaptation['bem_windows'].items():
        medians = seed_values(expected['medians'][arm])
        assert entry['dataset_seed_values'] == close(medians)
        assert entry['mean'] == close(np.mean(medians)) and entry['range'] == close([min(medians), max(medians)])
        if arm == 'default':
            assert set(entry) == {'dataset_seed_values', 'mean', 'range'}
            continue
        differences = seed_values(expected['differences'][arm])
        assert entry['difference_from_default_mm']['dataset_seed_values'] == close(differences)
        assert entry['difference_from_default_mm']['mean'] == close(np.mean(differences))
        assert entry['difference_from_default_mm']['range'] == close([min(differences), max(differences)])
        assert differences == close(np.subtract(medians, seed_values(expected['medians']['default'])))
        # The intervals are the recorded ones, one per dataset.
        assert entry['difference_ci95_mm_by_seed'] == [close(mini_evidence.around(value)) for value in differences]
    assert adaptation['own_sphere_eligible_windows'] == expected['own_sphere']
    assert list(expected['own_sphere']) == list(mini_evidence.VARIANTS)

    # One row per size of the fine-tuning set, with the numbers of training windows that the datasets used.
    assert [(row['size'], row['bem_training_windows']) for row in adaptation['finetuning']] == expected['finetuning']
    assert expected['finetuning'] == [('100', [100]), ('1000', [1000]), ('all', [298, 305, 310])]
    for row in adaptation['finetuning']:
        assert row['finetuned'] == adaptation['bem_windows'][f"finetuned_{row['size']}"]
        assert row['scratch'] == adaptation['bem_windows'][f"scratch_{row['size']}"]
    # The records hold every fine-tuned and every separately trained network on its own. The median of the two
    # networks of an arm together, which the tables give, lies between the medians of the two.
    for seed in SEEDS:
        assert list(expected['single_networks'][seed]) == SINGLE_NETWORKS
        for kind, size, _ in SINGLE_NETWORKS:
            first, second = sorted(expected['single_networks'][seed][kind, size, fit] for fit in FITS)
            assert first < expected['medians'][f'{kind}_{size}'][seed] < second, (seed, kind, size)

    assert adaptation['dipole_windows'] == expected['dipole_windows'] == 12
    assert adaptation['dipole_subset'] == expected['dipole_subset']
    assert len(expected['dipole_subset']) == 13
    assert adaptation['geometry']['variants']['size_matched']['radii_m'] == [0.0873, 0.0884, 0.0939, 0.1004]
    for seed in SEEDS:
        assert set(adaptation['strata'][seed]) == {'all', 'population_1', 'radius_0_75_79.501'}
        assert adaptation['bem_training_eligibility'][seed]['train']['windows'] == 480
        assert len(adaptation['fits'][seed]) == 8
    assert expected['bem_windows'] == {'862': 65, '863': 66, '864': 64}


def test_benchmark_table_holds_the_runs_of_every_case(miniature):
    benchmark, expected = miniature.report['benchmark'], miniature.expected['benchmark']
    assert len(benchmark['rows']) == len(expected['rows']) == 5
    for row, wanted in zip(benchmark['rows'], expected['rows']):
        assert (row['case'], row['format'], row['compressed'], row['workers'], row['windows'], row['repetitions']) == (
            wanted['case'], wanted['form'], wanted['compressed'], wanted['workers'], wanted['windows'],
            wanted['repetitions'])
        assert row['seconds'] == wanted['seconds'] and row['windows_per_second'] == wanted['rate']
        for key in ('median', 'minimum', 'maximum'):
            assert row['peak_rss_gib'][key] == close(wanted['memory_gib'][key])
        assert row['output_gib'] == close(wanted['output_gib'])
        # The range of the setup time over all runs of the case, and the stage times of the median run.
        assert (row['setup_seconds_of_all_runs']['minimum'], row['setup_seconds_of_all_runs']['maximum']) == close(
            wanted['setup_range'])
        assert row['median_run']['setup_seconds'] == close(wanted['setup'])
        assert row['median_run']['split_generation_and_writing_seconds'] == close(wanted['generation'])
        assert same(row['median_run']['hdf5_writing_seconds'], wanted['writing'])
        assert wanted['setup_range'][0] <= wanted['setup'] <= wanted['setup_range'][1]
    by_case = {row['case']: row for row in expected['rows']}
    assert by_case['npz-1']['writing'] is None and by_case['point-1']['writing'] > 1.0
    assert by_case['point-1']['setup_range'][0] < by_case['point-1']['setup_range'][1]
    assert by_case['default-scale-6']['repetitions'] == 1
    assert benchmark['host'] == expected['host']
    assert benchmark['patch_overlap_of_patch_case'] == dict(algorithm='exact small-cap lens and size pruning')
    assert benchmark['report']['sampling_freq_hz'] == 500.0


def test_settings_are_read_from_the_records_of_every_dataset(miniature):
    assert miniature.report['settings'] == miniature.expected['settings']
    # Passed through: the records of the earlier release and of the shared-position demonstration.
    continuity = miniature.report['continuity']
    assert list(continuity) == ['v100-continuity', 'v11-continuity']
    assert continuity['v100-continuity']['software_environment']['software_versions']['spherical_eeg_sim'] == '1.0.0'
    assert continuity['v11-continuity']['figure'] == 'v11-continuity.png'
    assert miniature.report['shared'] == miniature.expected['shared']


def test_rerun_tables_compare_the_releases_and_the_platforms(miniature):
    rerun, expected = miniature.report['validation_rerun'], miniature.expected['rerun']
    assert rerun['array_digests_of_the_two_releases'] == dict(
        datasets=expected['datasets'], arrays_in_both=expected['arrays'], arrays_with_equal_digest=expected['arrays'],
        arrays_only_in_first=[], arrays_only_in_second=[])
    assert (expected['datasets'], expected['arrays']) == (6, 30)
    assert rerun['two_releases_on_one_machine'] == dict(
        per_dipole_file_identical=True, dipole_recovery_and_sweep_equal=True, patch_comparison_equal=True,
        transfer_coefficients_equal=True, legendre_convergence_equal=True)

    platforms = rerun['two_platforms']
    assert platforms['dipoles'] == expected['dipoles'] == 12
    assert platforms['same_sources_and_realized_snr'] is True and platforms['transfer_coefficients_equal'] is True
    assert platforms['environments'] == dict(linux=expected['environments'][0], windows=expected['environments'][1])
    moved = platforms['fitted_position_difference_mm']
    assert set(moved) == {'n', 'median', 'p90', 'maximum', 'share_above_0p1_mm'} and moved['n'] == 12
    for key in ('median', 'p90', 'maximum', 'share_above_0p1_mm'):
        assert moved[key] == close(expected['moved'][key]), key
    assert 0 < expected['moved']['share_above_0p1_mm'] < 1
    for key, (linux, windows) in expected['position'].items():
        assert platforms['position_error_mm'][key] == dict(linux=linux, windows=windows)
    for name, key in (('orientation', 'orientation_error_median_deg'), ('moment', 'moment_relative_error_median'),
                      ('goodness', 'goodness_of_fit_median_percent'),
                      ('patch', 'patch_median_relative_field_difference'),
                      ('point', 'point_median_relative_field_difference'), ('ratio', 'patch_to_point_median_ratio')):
        linux, windows = expected[name]
        assert platforms[key] == dict(linux=linux, windows=windows) and linux != windows, name
    assert platforms['sweep_median_position_error_mm'] == [
        dict(snr_db=level, linux=close(linux), windows=close(windows)) for level, linux, windows in expected['sweep']]
    assert [level for level, _, _ in expected['sweep']] == [5.0, 20.0, 35.0]
    changes = [abs(windows - linux) for _, linux, windows in expected['sweep']]
    assert platforms['largest_change_of_a_sweep_median_mm'] == close(max(changes)) and min(changes) < max(changes)

    legendre, wanted = platforms['legendre'], expected['legendre']
    assert legendre['cases'] == wanted['cases'] and legendre['superficial_cases'] == wanted['superficial']
    assert len(wanted['cases']) == 4 and len(wanted['superficial']) == 2
    for key, name in (('largest_relative_change_superficial_orders_40_and_80', 'superficial_40_80'),
                      ('largest_relative_change_superficial_orders_100_and_120', 'superficial_100_120'),
                      ('largest_relative_change_any_case_orders_100_and_120', 'any_100_120'),
                      ('largest_relative_change_other_cases', 'others')):
        assert legendre[key] == pytest.approx(wanted[name], rel=1e-6), key
    # Each of the four numbers comes from another source case and order of the series.
    assert [wanted[name] for name in ('superficial_40_80', 'superficial_100_120', 'any_100_120', 'others')] == \
        pytest.approx([2e-8, 0.03, 0.05, 0.08], rel=1e-6)
    assert legendre['superficial_radial_order_100'] == dict(linux=wanted['radial_100'][0],
                                                            windows=wanted['radial_100'][1])


def test_sphere_fit_tables_hold_the_arms_the_trials_the_kernels_and_the_spheres(miniature):
    fit, expected = miniature.report['sphere_fit'], miniature.expected['sphere_fit']
    assert list(fit['arms']) == ['mne_limit', 'raised_limit']
    for name, arm in expected['arms'].items():
        cell = fit['arms'][name]
        assert cell['fit']['evaluations'] == arm['fit']['evaluations']
        assert cell['fit']['stopped_by_itself'] is (name == 'raised_limit')
        assert cell['fit']['objective'] == arm['fit']['objective']
        assert cell['dipole_recovery'] == arm['dipole_recovery'] and cell['patch_comparison'] == arm['patch_comparison']
        assert cell['sweep'] == arm['sweep'] and len(arm['sweep']) == 3
        assert cell['dipole_recovery']['position_error_mm']['n'] == 12 and cell['patch_comparison']['point']['n'] == 5
    assert fit['evaluation_limits'] == dict(mne_limit=1000, raised_limit=100000)
    assert fit['sizes']['perturbations'] == 3 and fit['sizes']['sweep_levels'] == [5, 20, 35]
    assert fit['head_model'] == miniature.report['adaptation']['geometry']['variants']['default']
    assert fit['optimizer_call'] == expected['optimizer_call'] and fit['optimizer_call']['arguments']['rhobeg'] == 0.5
    assert fit['default_sphere_model_of_mne_python']['mne_limit']['evaluations'] == 87
    assert fit['dipole_fits']['raised_limit']['fits'] == 12
    assert fit['software_environment']['software_versions']['mne'] == '1.12.1'

    # The trials: smallest and largest value of every quantity, by the fit that was changed and the sphere fit.
    assert fit['perturbation']['steps'] == expected['steps'] == [1, 2, 3]
    assert fit['perturbation']['ranges'] == expected['ranges']
    limit = expected['ranges']['sphere_fit']['mne_limit']
    assert (limit['trials'], limit['trials_in_which_the_parameter_fit_stopped_by_itself']) == (3, 0)
    assert limit['parameter_fit_evaluations'] == [1000.0, 1000.0]
    assert expected['ranges']['sphere_fit']['raised_limit']['trials_in_which_the_parameter_fit_stopped_by_itself'] == 3
    assert 'patch_median_relative_field_difference' in limit and 'dipole_fits' not in limit
    refits = expected['ranges']['dipole_fit']['raised_limit']
    assert refits['dipole_fits'] == [12.0, 12.0] and 'patch_median_relative_field_difference' not in refits
    assert all(low < high for low, high in (limit[key] for key in (
        'position_error_median_mm', 'fitted_position_change_p90_mm', 'point_median_relative_field_difference')))
    # The list of the trials is the file itself: every cell as it was written.
    assert len(fit['perturbation']['trials']) == len(expected['trials']) == 2 * 3 * 2
    for row, wanted in zip(fit['perturbation']['trials'], expected['trials']):
        assert list(row) == list(wanted) and len(row) == 23
        for column, value in wanted.items():
            assert row[column] == value if isinstance(value, str) else float(row[column]) == value, column

    # The kernels and the spheres come from the record, after every statistic was recomputed.
    assert list(fit['library_kernels']['arms']) == list(expected['kernels']) == ['Sandybridge', 'Nehalem']
    for kernel, entry in expected['kernels'].items():
        cell = fit['library_kernels']['arms'][kernel]
        for name in fit['arms']:
            assert cell['arms'][name]['dipole_recovery'] == entry['arms'][name]['dipole_recovery']
            assert set(cell['arms'][name]['dipole_recovery']) == {
                'position_error_mm', 'orientation_error_deg', 'moment_relative_error', 'goodness_of_fit_percent'}
            assert cell['arms'][name]['patch_comparison'] == entry['arms'][name]['patch_comparison']
            assert cell['fitted_position_change_from_the_parent_mm'][name] == \
                entry['fitted_position_change_from_the_parent_mm'][name]
        assert cell['default_sphere_model_of_mne_python']['raised_limit']['evaluation_limit'] == 100000
    assert list(fit['spheres']['variants']) == list(mini_evidence.VARIANTS)
    shares = set()
    for sphere, entry in expected['spheres'].items():
        cell = fit['spheres']['variants'][sphere]
        for name in fit['arms']:
            assert cell[name]['position_error_mm'] == entry[name]['position_error_mm']
            assert cell[name]['point_field_difference'] == entry[name]['point_field_difference']
            assert cell[name]['fit']['evaluations'] == entry[name]['fit']['evaluations']
        change = cell['position_error_change_mm']
        assert change['median_of_raised_limit_minus_median_of_mne_limit'] == close(
            entry['raised_limit']['position_error_mm']['median'] - entry['mne_limit']['position_error_mm']['median'])
        # How far the converged parameter fit moved the fitted positions.
        assert change['fitted_position'] == entry['position_error_change_mm']['fitted_position']
        assert change['fitted_position']['n'] == 12
        assert 0 < change['fitted_position']['median'] < change['fitted_position']['p90'] < \
            change['fitted_position']['maximum'] < 0.5
        shares.add(change['fitted_position']['share_above_0p1_mm'])
    # In every sphere some positions moved by more than 0.1 mm and some by less, and not the same share in all.
    assert len(shares) > 1 and all(0 < share < 1 for share in shares)

    # Ranges over every run of one sphere fit: the unchanged run, the trials and the kernels.
    assert list(fit['all_runs']) == list(fit['arms'])
    for name, wanted in expected['all_runs'].items():
        cell = fit['all_runs'][name]
        assert cell['runs_with_a_dipole_recovery'] == wanted['recoveries'] == 1 + 3 + 3 + 2
        assert cell['runs_with_a_field_comparison'] == wanted['comparisons'] == 1 + 3 + 2
        assert cell['parameter_fits'] == wanted['fits'] == 6
        assert cell['parameter_fits_that_stopped_by_themselves'] == wanted['stopped']
        assert wanted['stopped'] == (6 if name == 'raised_limit' else 1)
        assert cell['parameter_fit_evaluations'] == wanted['evaluations']
        for key in ('median', 'p90', 'maximum'):
            assert cell['position_error_mm'][key] == close(wanted['position'][key]), (name, key)
        for source in ('patch', 'point'):
            assert cell['median_relative_field_difference'][source] == close(wanted['field'][source]), (name, source)
    assert fit['links'] == dict(same_datasets_as_the_validation_rerun=True,
                                mne_limit_arm_equals_the_validation_rerun=True)
    # Every kind of run sets an end of some range, so none of them can be left out of the ranges unnoticed.
    for name, wanted in expected['all_runs'].items():
        arm = expected['arms'][name]
        trials = {kind: [row for row in expected['trials'] if (row['perturbed_fit'], row['sphere_fit']) == (kind, name)]
                  for kind in ('sphere_fit', 'dipole_fit')}
        kernels = [entry['arms'][name] for entry in expected['kernels'].values()]
        assert wanted['position']['maximum'][1] == arm['dipole_recovery']['position_error_mm']['maximum']
        assert wanted['field']['patch'][0] == arm['patch_comparison']['patch']['median']
        assert wanted['field']['point'][0] == min(row['point_median_relative_field_difference']
                                                  for row in trials['sphere_fit'])
        assert wanted['field']['patch'][1] == max(item['patch_comparison']['patch']['median'] for item in kernels)
        assert wanted['field']['point'][1] == arm['patch_comparison']['point']['median']
        ends = [end for key in ('median', 'p90', 'maximum') for end in wanted['position'][key]]
        for kind in ('sphere_fit', 'dipole_fit'):
            assert any(row[f'position_error_{key}_mm'] in ends for row in trials[kind]
                       for key in ('median', 'p90', 'maximum')), (name, kind)
        assert any(item['dipole_recovery']['position_error_mm'][key] in ends for item in kernels
                   for key in ('median', 'p90', 'maximum')), name
    raised = expected['all_runs']['raised_limit']
    assert raised['evaluations'] == [expected['arms']['raised_limit']['fit']['evaluations'], max(
        row['parameter_fit_evaluations'] for row in expected['trials'] if row['sphere_fit'] == 'raised_limit')]
    assert raised['evaluations'][0] < 1700 < 2500 < raised['evaluations'][1]
    # At the limit of SciPy only the fit under the last kernel stops by itself, with the fewest evaluations.
    assert expected['all_runs']['mne_limit']['evaluations'] == [950, 1000]
    assert expected['kernels']['Nehalem']['arms']['mne_limit']['fit']['evaluations'] == 950


def test_sphere_fit_tables_hold_the_runs_with_the_small_final_radius(miniature):
    fit, expected = miniature.report['sphere_fit'], miniature.expected['sphere_fit']
    small, wanted = fit['small_tolerance'], expected['small']
    assert set(small) == {'final_radius_m', 'final_radius_of_mne_python_m', 'steps', 'kernel_of_the_main_process',
                          'ranges', 'runs'}
    # The sentence in which the record explains its measure of the changes is not passed on.
    assert set(wanted) - set(small) == {'fitted_position_change'}
    assert (small['final_radius_m'], small['final_radius_of_mne_python_m']) == (1e-7, 5e-5)
    # Fewer objectives were changed than in the trials.
    assert small['steps'] == wanted['steps'] == [1, 2] and fit['perturbation']['steps'] == [1, 2, 3]

    # The list of the runs is the file itself: every cell as it was written.
    assert len(small['runs']) == len(wanted['runs']) == 2 * (1 + 2 * 2 + 2)
    for row, run in zip(small['runs'], wanted['runs']):
        assert list(row) == list(run) and len(row) == 16
        for column, value in run.items():
            assert row[column] == value if isinstance(value, str) else float(row[column]) == value, column
    # The unchanged runs of the main process, its runs with a changed objective, and the other kernels.
    keys = [(run['kernel'], run['changed_objective'], run['sphere_fit'], run['objective_scale_minus_one_in_eps'])
            for run in wanted['runs']]
    assert keys[:2] == [('Haswell', 'none', name, 0) for name in fit['arms']]
    assert keys[2:10] == [('Haswell', changed, name, step) for step in (1, 2) for name in fit['arms']
                          for changed in ('parameter_fit', 'dipole_fits')]
    assert keys[10:] == [(kernel, 'none', name, 0) for kernel in ('Sandybridge', 'Nehalem') for name in fit['arms']]
    # Only the unchanged runs of the main process record no change of the fitted positions.
    assert [run['fitted_position_change_median_mm'] == '' for run in wanted['runs']] == [True, True] + [False] * 12
    # A parameter fit with a changed objective ended as in the trial with the same objective, and the parameter
    # fit of a run with changed dipole fits as in the unchanged run.
    trials = {(row['sphere_fit'], row['objective_scale_minus_one_in_eps']): row for row in expected['trials']
              if row['perturbed_fit'] == 'sphere_fit'}
    for (_, changed, name, step), run in zip(keys[2:10], wanted['runs'][2:10]):
        assert run['parameter_fit_evaluations'] == (
            trials[name, step]['parameter_fit_evaluations'] if changed == 'parameter_fit'
            else expected['arms'][name]['fit']['evaluations']), (changed, name, step)

    # Counts and ranges over the runs of each sphere fit.
    assert small['ranges'] == wanted['ranges'] and list(small['ranges']) == list(fit['arms'])
    limit, raised = (wanted['ranges'][name] for name in fit['arms'])
    for entry in (limit, raised):
        assert len(entry) == 5 + 6 and (entry['runs'], entry['dipole_fits']) == (7, 7 * 12)
        # The kernels in the order of the alphabet, which is not the order of the runs.
        assert entry['kernels'] == ['Haswell', 'Nehalem', 'Sandybridge']
        assert all(low < high for low, high in (entry[column] for column in entry if column.endswith('_mm')))
    # At the limit of SciPy only the parameter fit under the last kernel stopped by itself, and in one run one
    # dipole fit did not.
    assert (limit['runs_in_which_the_parameter_fit_stopped_by_itself'],
            raised['runs_in_which_the_parameter_fit_stopped_by_itself']) == (1, 7)
    assert (limit['dipole_fits_stopped_by_themselves'], raised['dipole_fits_stopped_by_themselves']) == (83, 84)
    # In one run a single dipole moved: no change for half of the dipoles and for nine in ten of them.
    assert keys[9] == ('Haswell', 'dipole_fits', 'raised_limit', 2)
    single = wanted['runs'][9]
    assert (single['fitted_position_change_median_mm'], single['fitted_position_change_p90_mm']) == (0.0, 0.0)
    assert single['fitted_position_change_maximum_mm'] == pytest.approx(3e-4, rel=1e-6)
    assert raised['fitted_position_change_median_mm'][0] == 0.0 and raised['fitted_position_change_p90_mm'][0] == 0.0
    assert single['position_error_maximum_mm'] == wanted['runs'][1]['position_error_maximum_mm']
    # With the converged parameter fit no position moved by a micrometre. At the limit of SciPy a changed
    # dipole fit moved them as little, and a changed parameter fit or another kernel by far more.
    assert raised['fitted_position_change_maximum_mm'][1] < 1e-3
    assert limit['fitted_position_change_median_mm'][0] < 1e-3 < 0.01 < limit['fitted_position_change_median_mm'][1]

    # The unchanged run under every other kernel, from the record after its statistics were recomputed.
    for kernel, entry in expected['kernels'].items():
        cell = fit['library_kernels']['arms'][kernel]
        for name in fit['arms']:
            recorded, built = cell['small_tolerance'][name], entry['small_tolerance'][name]
            assert recorded['position_error_mm'] == built['position_error_mm']
            assert recorded['fitted_position_change_from_the_parent_mm'] == \
                built['fitted_position_change_from_the_parent_mm']
            assert recorded['dipole_fits']['fits'] == 12
            run = wanted['runs'][keys.index((kernel, 'none', name, 0))]
            assert run['position_error_p90_mm'] == built['position_error_mm']['p90']
            assert run['fitted_position_change_maximum_mm'] == \
                built['fitted_position_change_from_the_parent_mm']['maximum']
            assert run['parameter_fit_evaluations'] == cell['arms'][name]['fit']['evaluations']
    # Under the first kernel one of the twelve dipoles moved by more than 0.1 mm at the limit of SciPy.
    shares = {(kernel, name): entry['small_tolerance'][name]['fitted_position_change_from_the_parent_mm'][
        'share_above_0p1_mm'] for kernel, entry in expected['kernels'].items() for name in fit['arms']}
    assert shares == {('Sandybridge', 'mne_limit'): pytest.approx(1 / 12), ('Sandybridge', 'raised_limit'): 0.0,
                      ('Nehalem', 'mne_limit'): 0.0, ('Nehalem', 'raised_limit'): 0.0}


def test_sphere_fit_tables_name_the_kernel_of_the_main_process(miniature):
    fit = miniature.report['sphere_fit']
    small = fit['small_tolerance']
    # The name that the record gives: a text, and no statistic that a later step of the driver computed.
    record = json.loads((miniature.evidence / SPHERE_SUMMARY).read_text(encoding='utf-8'))['small_tolerance']
    assert small['kernel_of_the_main_process'] == 'Haswell' and type(small['kernel_of_the_main_process']) is str
    assert small['kernel_of_the_main_process'] == record['kernel_of_the_main_process']
    assert miniature.expected['sphere_fit']['small']['kernel_of_the_main_process'] == 'Haswell'
    # It is the kernel of the two runs that record no change of their positions, and none of the other kernels.
    assert [run['kernel'] for run in small['runs'] if run['fitted_position_change_median_mm'] == ''] == ['Haswell'] * 2
    assert 'Haswell' not in fit['library_kernels']['arms']
    assert {entry['kernel'] for entry in fit['library_kernels']['of_the_parent_process']} == {'Haswell'}
    # The ranges over every run of a sphere fit, which are computed after the kernel was read, hold numbers.
    for runs in fit['all_runs'].values():
        assert all(type(end) is float for key in ('median', 'p90', 'maximum') for end in runs['position_error_mm'][key])
        assert all(type(end) is float for ends in runs['median_relative_field_difference'].values() for end in ends)


def test_comparator_tables_hold_every_group_and_the_means_of_the_split_study(miniature):
    comparators, expected = miniature.report['comparators'], miniature.expected['comparators']
    assert list(comparators) == ['windows_per_subset', 'evaluation_limits',
                                 'default_limit_fits_are_those_of_the_recorded_experiments', 'groups',
                                 'split_study_means', 'largest_change_of_a_median_mm']
    assert comparators['windows_per_subset'] == expected['windows_per_subset'] == 12
    assert comparators['evaluation_limits'] == dict(mne_limit=1000, raised_limit=100000)
    # The driver stops unless the fits with the limit of SciPy are those of the three experiments, so the
    # tables can only say that they are.
    assert comparators['default_limit_fits_are_those_of_the_recorded_experiments'] is True

    # One group per dataset and policy of the split study, one of the transfer experiment, and one per sphere.
    labels = [(group['experiment'], group['dataset_seed'], group['case']) for group in comparators['groups']]
    assert labels == [*[('split_study', seed, policy) for seed, policy in DATASETS],
                      ('transfer', '862', 'package_sphere_matched'),
                      *[('adaptation', '862', f'{sphere}_sphere_on_bem') for sphere in mini_evidence.VARIANTS]]
    spheres = miniature.report['sphere_fit']['spheres']['variants']
    for group, wanted, label in zip(comparators['groups'], expected['groups'], labels):
        assert list(group) == ['experiment', 'dataset_seed', 'case', 'sphere', 'mne_limit', 'raised_limit',
                               'parameter_fit', 'change_of_the_median_mm', 'fitted_position_change_mm'], label
        assert label == (wanted['experiment'], wanted['dataset_seed'], wanted['case'])
        # The statistics of both fits, from the record after they were recomputed.
        for name in ('mne_limit', 'raised_limit'):
            assert group[name] == wanted[name] and set(group[name]) == {'windows', 'median_mm', 'mean_mm', 'p90_mm'}
            # Every fit has the number of windows that the record names once.
            assert group[name]['windows'] == comparators['windows_per_subset'], label
            # The parameter fit ended as the sphere-fit tables say for the sphere of the group.
            assert group['parameter_fit'][name] == {key: spheres[group['sphere']][name]['fit'][key]
                                                    for key in ('evaluations', 'stopped_by_itself', 'objective')}, label
        # As MNE-Python runs it, the parameter fit ends at the limit of SciPy; with the raised limit it stops by
        # itself, before that limit.
        limit, raised = (wanted['parameter_fit'][name] for name in ('mne_limit', 'raised_limit'))
        assert (limit['evaluations'], limit['stopped_by_itself']) == (1000, False), label
        assert raised['stopped_by_itself'] is True and 1000 < raised['evaluations'] < 100000, label
        assert group['sphere'] == wanted['sphere'] == (
            label[2][:-len('_sphere_on_bem')] if label[0] == 'adaptation' else 'default'), label
        assert group['change_of_the_median_mm'] == close(
            wanted['raised_limit']['median_mm'] - wanted['mne_limit']['median_mm']), label
        assert group['fitted_position_change_mm'] == wanted['fitted_position_change_mm'], label
        moved = group['fitted_position_change_mm']
        assert list(moved) == ['median', 'p90', 'maximum'] and 0 < moved['median'] < moved['p90'] < moved['maximum']
    assert len({group['parameter_fit']['raised_limit']['evaluations'] for group in comparators['groups']}) == 5

    # The converged fit lowers the median error of most groups and raises it in one group of the split study
    # and for one sphere. The largest change is a reduction; the tables give its size.
    changes = {label: group['change_of_the_median_mm'] for label, group in zip(labels, comparators['groups'])}
    assert [label for label, change in changes.items() if change > 0] == [
        ('split_study', '863', 'region'), ('adaptation', '862', 'size_matched_sphere_on_bem')]
    assert comparators['largest_change_of_a_median_mm'] == close(expected['largest_change'])
    assert expected['largest_change'] == close(-min(changes.values())) and max(changes.values()) < 0.5 * expected[
        'largest_change']

    # The matched-sphere fit of the split study: mean and range of the dataset medians under each policy.
    assert list(comparators['split_study_means']) == list(POLICIES)
    for policy in POLICIES:
        for name in ('mne_limit', 'raised_limit'):
            mean, limits = expected['split_means'][policy][name]
            assert comparators['split_study_means'][policy][name] == dict(mean_of_dataset_medians_mm=close(mean),
                                                                          range_mm=close(limits)), (policy, name)
        # With the limit of SciPy the fits are those of the split study, whose headline has the same numbers.
        headline = miniature.report['split']['table11'][policy]['ecd_subset']['dipole']
        mean, limits = expected['split_means'][policy]['mne_limit']
        assert (mean, limits) == (close(headline['mean_median_mm']), close(headline['range_mm']))
        assert abs(expected['split_means'][policy]['raised_limit'][0] - mean) > 0.01
    # The other groups have the medians of the dipole fits that the two experiments recorded.
    by_label = dict(zip(labels, expected['groups']))
    transfer = miniature.expected['transfer']['dipole_subset']['errors']['ecd_package_sphere_matched']
    assert by_label['transfer', '862', 'package_sphere_matched']['mne_limit']['median_mm'] == close(np.median(transfer))
    for sphere in mini_evidence.VARIANTS:
        recorded = miniature.expected['adaptation']['dipole_subset'][f'dipole_{sphere}_sphere_on_bem']
        assert by_label['adaptation', '862', f'{sphere}_sphere_on_bem']['mne_limit']['median_mm'] == close(
            recorded['median_mm'])


def test_miniature_record_of_the_dipole_comparators_belongs_to_the_files_of_the_experiments(miniature):
    """What the driver requires of the record of the dipole comparators, read from the files without the driver."""
    root = miniature.evidence

    def rows_of(file):
        return list(csv.DictReader(io.StringIO((root / file).read_text(encoding='utf-8'))))

    def numbers(rows, *names):
        return np.array([[float(row[name]) for name in names] for row in rows])

    def position(rows, prefix, suffix=''):
        return numbers(rows, *[f'{prefix}{axis}_mm{suffix}' for axis in 'xyz'])

    # Every fit that an experiment recorded: its windows, their true source positions, the fitted positions and
    # the errors. The split study records no errors; the other two take the source positions from a file of
    # their own, by the number of the window.
    recorded = {}
    for seed, policy in DATASETS:
        subset = rows_of(f'{SPLIT}/{seed}/{policy}/ecd_subset/predictions.csv')
        windows = json.loads((root / SPLIT / seed / policy / 'ecd_subset/test_rows.json').read_text(encoding='utf-8'))
        recorded['split_study', seed, policy] = (windows, position(subset, ''), position(subset, 'dipole_'), None)
    fits = {(TRANSFER, 'ecd_subset.csv', 'transfer'): {'package_sphere_matched': 'ecd_package_sphere_matched'},
            (ADAPTATION, 'dipole_subset.csv', 'adaptation'): {
                f'{sphere}_sphere_on_bem': f'dipole_{sphere}' for sphere in mini_evidence.VARIANTS}}
    for (folder, file, experiment), names in fits.items():
        subset = rows_of(f'{folder}/862/{file}')
        sources = {row['window_id']: row for row in rows_of(f'{folder}/862/eligibility.csv')}
        truth = position([sources[row['window_id']] for row in subset], '')
        for name, prefix in names.items():
            column = f'{prefix}_sphere_on_bem_error_mm' if experiment == 'adaptation' else f'{prefix}_error_mm'
            recorded[experiment, '862', name] = ([int(row['window_id']) for row in subset], truth,
                                                 position(subset, f'{prefix}_'), numbers(subset, column).ravel())
    assert len(recorded) == 6 + 1 + 5

    # The record holds these groups and no others, in its per-window file and in its summary.
    record = json.loads((root / COMPARATORS_SUMMARY).read_text(encoding='utf-8'))
    groups = {}
    for row in rows_of(COMPARATORS_CSV):
        groups.setdefault((row['experiment'], row['dataset_seed'], row['case']), []).append(row)
    assert list(groups) == list(recorded)
    assert [(experiment, seed, name) for experiment in ('split_study', 'transfer', 'adaptation')
            for seed, cases in record[experiment].items() for name in cases] == list(recorded)
    spheres = json.loads((root / SPHERE_SUMMARY).read_text(encoding='utf-8'))['spheres']['variants']
    for label, members in groups.items():
        windows, truth, fitted, errors = recorded[label]
        # Every fit has the number of windows that the record names once, and the windows of its experiment.
        assert len(members) == record['windows_per_subset'] == 12, label
        assert [int(row['window_id']) for row in members] == windows and len(set(windows)) == 12, label
        # With the limit of SciPy the fit is the recorded one: the same positions and the same errors.
        assert np.array_equal(position(members, 'fitted_', '_mne_limit'), fitted), label
        if errors is not None:
            assert np.array_equal(numbers(members, 'position_error_mm_mne_limit').ravel(), errors), label
        # Both errors are the distances of the fitted positions from the true source positions, and the
        # converged fit puts no dipole where the other fit put it.
        for name in ('mne_limit', 'raised_limit'):
            distance = np.linalg.norm(position(members, 'fitted_', f'_{name}') - truth, axis=1)
            assert numbers(members, f'position_error_mm_{name}').ravel() == pytest.approx(distance, rel=0, abs=1e-12)
        assert np.all(numbers(members, 'position_error_mm_raised_limit')
                      != numbers(members, 'position_error_mm_mne_limit')), label
        # The parameter fits are those of the sphere in the sphere-fit record: one ends at the limit of SciPy,
        # the other one by itself.
        entry = record[label[0]][label[1]][label[2]]
        sphere = label[2][:-len('_sphere_on_bem')] if label[0] == 'adaptation' else 'default'
        for name, stopped in (('mne_limit', False), ('raised_limit', True)):
            assert entry['parameter_fit'][name] == {key: spheres[sphere][name]['fit'][key] for key in (
                'evaluations', 'stopped_by_itself', 'objective')}, label
            assert entry['parameter_fit'][name]['stopped_by_itself'] is stopped, label
        assert entry['parameter_fit']['mne_limit']['evaluations'] == record['evaluation_limits']['mne_limit'] == 1000
    # The adaptation experiment wrote the eligibility file of the transfer experiment once more. The miniature
    # holds the file for the one dataset whose dipole fits the driver reads; in the archive every dataset of the
    # adaptation experiment has it.
    assert (root / ADAPTATION / '862/eligibility.csv').read_bytes() == \
        (root / TRANSFER / '862/eligibility.csv').read_bytes()
    assert [seed for seed in SEEDS if (root / ADAPTATION / seed / 'eligibility.csv').exists()] == ['862']


# --------------------------------------------------------------------------- the Markdown file

def table_rows(lines, heading):
    """The rows of the tables below a heading, each as the list of its cells."""
    rows = []
    for line in lines[lines.index(heading) + 1:]:
        if line.startswith('## '):
            break
        if line.startswith('|') and not line.startswith('| ---'):
            rows.append([cell.strip() for cell in line.strip().strip('|').split('|')])
    return rows


def with_range(mean, limits, digits=2):
    return f'{mean:.{digits}f} ({limits[0]:.{digits}f}–{limits[1]:.{digits}f})'


def mm(value):
    return '—' if value is None else f'{value:.2f}'


def test_markdown_prints_the_cells_of_every_table(miniature):
    lines, expected, report = miniature.lines, miniature.expected, miniature.report
    assert lines[0] == '# Reproduced experiment statistics'
    # Every table has a header row and, below it, a separator row with the same number of columns.
    separators = [number for number, line in enumerate(lines) if line.startswith('| ---')]
    assert len(separators) == 23
    assert all(lines[number].count('|') == lines[number - 1].count('|') for number in separators)
    # A table and a heading stand below an empty line, and a table ends at an empty line.
    assert all(lines[number - 2] == '' for number in separators)
    assert all(lines[number - 1] == '' for number, line in enumerate(lines) if line.startswith('## '))
    assert all(following == '' or following.startswith('|')
               for line, following in zip(lines, lines[1:]) if line.startswith('|'))
    assert [line for line in lines if line.startswith('## ')] == [
        '## Localization comparison', '## Observed training locations',
        '## Network and lead-field dictionary by stratum', '## Regional contrast on the same test windows',
        '## Mean training position: distance from the sphere centre (mm)', '## Transfer eligibility census',
        '## Transfer errors by stratum', '## Independent sphere implementation, median error by dataset seed (mm)',
        '## Transfer eligibility by direction', '## Training-location count control', '## Patch split comparison',
        '## Coordinate reuse under shared positions', '## Head-model adaptation on the boundary-element windows',
        '## Generation benchmark', '## Numerical validation on two platforms and with two releases',
        '## Parameter fit of the sphere model in MNE-Python',
        '## Dipole-fit comparators with a converged parameter fit']

    def headline_cell(entry):
        return with_range(entry['mean_median_mm'], entry['range_mm'])

    split = expected['split']
    rows = table_rows(lines, '## Localization comparison')
    assert len(rows) == 1 + 5 + 6
    for panel, method in (('full', 'mlp_pooled'), ('full', 'training_mean'), ('ecd_subset', 'dipole'),
                          ('ecd_subset', 'knn10')):
        assert [method, headline_cell(split['headline']['position'][panel][method]),
                headline_cell(split['headline']['region'][panel][method]), panel] in rows
    assert [row[0] for row in rows[1:6]] == ['mlp_pooled', 'dictionary', 'knn1', 'knn10', 'training_mean']
    assert rows[10][0] == 'dipole' and rows[11] == [
        'training_mean', headline_cell(split['headline']['position']['ecd_subset']['training_mean']),
        headline_cell(split['headline']['region']['ecd_subset']['training_mean']), 'ecd_subset']

    rows = table_rows(lines, '## Observed training locations')
    assert rows[1:] == [[seed, policy, str(split['training_positions'][seed, policy])] for seed, policy in DATASETS]

    rows = table_rows(lines, '## Network and lead-field dictionary by stratum')
    assert rows[1:] == [[row['stratum'], mm(row['position']['network']), mm(row['position']['dictionary']),
                         str(len(row['position']['seeds'])), mm(row['region']['network']),
                         mm(row['region']['dictionary']), str(len(row['region']['seeds']))] for row in split['strata']]
    assert ['radius_1_30_45', '—', '—', '0', '—', '—', '0'] in rows
    assert [row[3] for row in rows if row[0] == 'below_montage'] == ['2']

    rows = table_rows(lines, '## Regional contrast on the same test windows')
    assert rows[1:4] == [[seed, str(record['windows']), str(record['distinct_positions']),
                          f"{record['region_minus_position_mm']:.2f}",
                          f"{record['ci95_mm'][0]:.2f} to {record['ci95_mm'][1]:.2f}"]
                         for seed, record in ((seed, split['paired'][seed]) for seed in SEEDS)]
    pooled = split['paired']['pooled']
    assert rows[4] == ['pooled', '45', str(pooled['seed_position_clusters']),
                       f"{pooled['region_minus_position_mm']:.2f}",
                       f"{pooled['conditional_ci95_mm'][0]:.2f} to {pooled['conditional_ci95_mm'][1]:.2f}"]

    rows = table_rows(lines, '## Mean training position: distance from the sphere centre (mm)')
    assert rows == [['Policy', *SEEDS], *[[policy, *[f'{split["mean_offset"][policy][seed]:.2f}' for seed in SEEDS]]
                                           for policy in POLICIES]]

    transfer = expected['transfer']
    rows = table_rows(lines, '## Transfer eligibility census')
    assert rows[0] == ['Stratum', *SEEDS, 'Pooled'] and len(rows) == 1 + 12 + 1
    for name in ('all', 'subcortical', 'radius_55_65', 'cortical_inferior_anterior'):
        counts = [transfer['census_counts'][seed][name] for seed in SEEDS]
        eligible, total = sum(pair[0] for pair in counts), sum(pair[1] for pair in counts)
        assert [name, *[f'{pair[0]}/{pair[1]}' for pair in counts],
                f'{eligible}/{total} ({eligible / total:.1%})'] in rows
    counts = [transfer['outside'][seed] for seed in SEEDS]
    outside, total = sum(pair[0] for pair in counts), sum(pair[1] for pair in counts)
    assert rows[-1] == ['cortical_outside_inner_skull', *[f'{pair[0]}/{pair[1]}' for pair in counts],
                        f'{outside}/{total} ({outside / total:.1%})']

    rows = table_rows(lines, '## Transfer errors by stratum')
    assert rows[1:] == [[row['stratum'], str(row['windows']), mm(row['sphere']), mm(row['bem']), mm(row['difference'])]
                        for row in transfer['window_strata']]
    assert ['population_1', '97', '—', '—', '—'] in rows

    rows = table_rows(lines, '## Independent sphere implementation, median error by dataset seed (mm)')
    assert rows == [list(SEEDS), [f'{transfer["independent"][seed]:.2f}' for seed in SEEDS]]

    rows = table_rows(lines, '## Transfer eligibility by direction')
    for direction in ('anterior', 'posterior', 'superior', 'inferior'):
        cells = []
        for name in ('everything', 'cortical'):
            eligible = sum(transfer['directional'][seed, direction][name][0] for seed in SEEDS)
            total = sum(transfer['directional'][seed, direction][name][1] for seed in SEEDS)
            cells.append(f'{eligible}/{total} ({eligible / total:.1%})')
        assert [direction, *cells] in rows
    assert [row[0] for row in rows[1:]] == ['anterior', 'posterior', 'superior', 'inferior']

    rows = table_rows(lines, '## Training-location count control')
    for panel, entries in expected['count_control'].items():
        assert [panel, *[with_range(entries[name]['mean'], entries[name]['range'])
                         for name in ('exact', 'count_matched', 'count_matched_minus_exact')]] in rows
    assert len(rows) == 3

    patch = expected['patch']
    rows = table_rows(lines, '## Patch split comparison')
    assert rows[1:5] == [[method, headline_cell(patch['headline']['position'][method]),
                          headline_cell(patch['headline']['region'][method])]
                         for method in ('mlp_pooled', 'nearest_training_patch', 'ten_nearest_training_patches',
                                        'training_mean')]
    assert rows[6:] == [[seed, policy, f"{audit['overlap']:.3f}", mm(audit['gap']), f"{audit['used_distance']:.2f}"]
                        for (seed, policy), audit in ((dataset, patch['audits'][dataset]) for dataset in DATASETS)]
    assert rows[6][2:4] != ['0.000', '—'] and rows[6][3] == '—' and rows[7][2] == '0.000'

    rows = table_rows(lines, '## Coordinate reuse under shared positions')
    assert rows[1:] == [['0.01', '6 of 72 (8.3%)'], ['1', '63 of 72 (87.5%)']]

    adaptation = expected['adaptation']
    rows = table_rows(lines, '## Head-model adaptation on the boundary-element windows')
    for arm in mini_evidence.ARMS:
        medians = seed_values(adaptation['medians'][arm])
        cells = [arm, with_range(np.mean(medians), [min(medians), max(medians)]), '—']
        if arm != 'default':
            differences = seed_values(adaptation['differences'][arm])
            cells[2] = with_range(np.mean(differences), [min(differences), max(differences)])
        assert cells in rows
    assert rows[1][0] == 'default' and rows[11][0] == 'scratch_all'
    assert rows[13:] == [[name, f"{entry['median_mm']:.2f}",
                          f"{entry['median_ci95_mm'][0]:.2f} to {entry['median_ci95_mm'][1]:.2f}"]
                         for name, entry in adaptation['dipole_subset'].items()]

    def run_cell(entry, digits=1):
        return f"{entry['median']:.{digits}f} ({entry['minimum']:.{digits}f}–{entry['maximum']:.{digits}f})"

    rows = table_rows(lines, '## Generation benchmark')
    assert rows[1:] == [[row['case'], row['form'] + ('' if row['compressed'] else ' (uncompressed)'),
                         str(row['workers']), f"{row['windows']:,}", run_cell(row['seconds']), run_cell(row['rate']),
                         run_cell(row['memory_gib'], 2), f"{row['output_gib']:.2f}"]
                        for row in expected['benchmark']['rows']]
    assert rows[3][1] == 'hdf5 (uncompressed)' and rows[5][2:4] == ['6', '360,000'] and rows[4][1] == 'npz'

    rerun = expected['rerun']
    heading = '## Numerical validation on two platforms and with two releases'
    assert lines[lines.index(heading) + 2] == ('Arrays of the 6 validation datasets with equal digests under '
                                               'releases 1.0.0 and 1.1.0: 30 of 30.')
    rows = table_rows(lines, heading)
    assert rows[1:] == [
        ['Position error, median (mm)', *[f'{value:.3f}' for value in rerun['position']['median']]],
        ['Position error, 90th percentile (mm)', *[f'{value:.3f}' for value in rerun['position']['p90']]],
        ['Position error, maximum (mm)', *[f'{value:.3f}' for value in rerun['position']['maximum']]],
        ['Orientation error, median (degrees)', *[f'{value:.3f}' for value in rerun['orientation']]],
        ['Moment-amplitude error, median', *[f'{value:.5f}' for value in rerun['moment']]],
        *[[f'Sweep at {level:g} dB, median position error (mm)', f'{linux:.3f}', f'{windows:.3f}']
          for level, linux, windows in rerun['sweep']],
        ['Patch comparison, median relative field difference', *[f'{value:.2e}' for value in rerun['patch']]],
        ['Point-source control, median relative field difference', *[f'{value:.2e}' for value in rerun['point']]]]
    moved = rerun['moved']
    assert (f"The fitted positions of the same 12 dipoles differ between the platforms by a median of "
            f"{moved['median']:.3f} mm, by more than 0.1 mm for {moved['share_above_0p1_mm']:.1%} of them, and by "
            f"at most {moved['maximum']:.2f} mm.") in lines

    fit = expected['sphere_fit']
    limit, raised = fit['arms']['mne_limit'], fit['arms']['raised_limit']
    rows = table_rows(lines, '## Parameter fit of the sphere model in MNE-Python')
    assert rows[:9] == [
        ['Quantity', 'mne_limit', 'raised_limit'],
        ['Function evaluations', '1000', str(raised['fit']['evaluations'])],
        ['Optimizer stopped by itself', 'no', 'yes'],
        ['Objective at the end', f"{limit['fit']['objective']:.3e}", f"{raised['fit']['objective']:.3e}"],
        ['Patch, median relative field difference',
         *[f"{arm['patch_comparison']['patch']['median']:.2e}" for arm in (limit, raised)]],
        ['Point sources, median relative field difference',
         *[f"{arm['patch_comparison']['point']['median']:.2e}" for arm in (limit, raised)]],
        *[[f'Position error at 35 dB, {label} (mm)',
           *[f"{arm['dipole_recovery']['position_error_mm'][key]:.3f}" for arm in (limit, raised)]]
          for label, key in (('median', 'median'), ('90th percentile', 'p90'), ('maximum', 'maximum'))]]
    runs = [fit['all_runs'][name] for name in ('mne_limit', 'raised_limit')]
    assert rows[9:17] == [
        ['Quantity', 'mne_limit', 'raised_limit'],
        ['Runs with a dipole recovery', '9', '9'],
        ['Parameter fits that stopped by themselves', '1 of 6', '6 of 6'],
        *[[f'{label}, median relative field difference (10⁻³)',
           *[f"{1000 * run['field'][source][0]:.2f}–{1000 * run['field'][source][1]:.2f}" for run in runs]]
          for label, source in (('Patch', 'patch'), ('Point sources', 'point'))],
        *[[f'Position error at 35 dB, {label} (mm)',
           *[f"{run['position'][key][0]:.3f}–{run['position'][key][1]:.3f}" for run in runs]]
          for label, key in (('median', 'median'), ('90th percentile', 'p90'), ('maximum', 'maximum'))]]
    assert 'Other kernels of the linear-algebra library (kernel of the main run: Haswell):' in lines

    def kernel_row(label, arms):
        return [label, f"{arms['mne_limit']['fit']['first_evaluations'][1]['objective']:.5f}",
                f"{arms['mne_limit']['fit']['objective']:.3e}",
                f"{arms['mne_limit']['patch_comparison']['patch']['median']:.2e}",
                f"{arms['mne_limit']['dipole_recovery']['position_error_mm']['median']:.3f}",
                str(arms['raised_limit']['fit']['evaluations']),
                f"{arms['raised_limit']['patch_comparison']['patch']['median']:.2e}",
                f"{arms['raised_limit']['dipole_recovery']['position_error_mm']['median']:.3f}"]

    assert rows[18:21] == [kernel_row('Haswell (main run)', fit['arms']),
                           *[kernel_row(kernel, entry['arms']) for kernel, entry in fit['kernels'].items()]]
    assert 'Spheres of the adaptation experiment (point sources; dipole recovery at 10 to 25 dB):' in lines
    assert rows[22:27] == [[sphere, '1000', str(entry['raised_limit']['fit']['evaluations']),
                            f"{entry['mne_limit']['point_field_difference']['median']:.2e}",
                            f"{entry['raised_limit']['point_field_difference']['median']:.2e}",
                            f"{entry['mne_limit']['position_error_mm']['median']:.3f}",
                            f"{entry['raised_limit']['position_error_mm']['median']:.3f}"]
                           for sphere, entry in fit['spheres'].items()]
    assert report['sphere_fit']['spheres']['snr_db_range'] == [10.0, 25.0]
    assert ('Dipole fits ended at a trust-region radius of 1e-07 m (MNE-Python: 5e-05 m), ranges over the runs of '
            'each sphere fit:') in lines
    ranges = [fit['small']['ranges'][name] for name in ('mne_limit', 'raised_limit')]
    assert rows[27:] == [
        ['Quantity', 'mne_limit', 'raised_limit'],
        ['Runs', '7', '7'],
        ['Kernels', 'Haswell, Nehalem, Sandybridge', 'Haswell, Nehalem, Sandybridge'],
        ['Parameter fits that stopped by themselves', '1 of 7', '7 of 7'],
        ['Dipole fits that stopped by themselves', '83 of 84', '84 of 84'],
        *[[f'Position error at 35 dB, {label} (mm)',
           *[f"{entry[f'position_error_{key}_mm'][0]:.4f}–{entry[f'position_error_{key}_mm'][1]:.4f}"
             for entry in ranges]]
          for label, key in (('median', 'median'), ('90th percentile', 'p90'), ('maximum', 'maximum'))]]
    # With the converged fit the two ends of a range differ in the fourth decimal.
    low, high = rows[32][2].split('–')
    assert low[:-1] == high[:-1] and low != high

    rows = table_rows(lines, '## Dipole-fit comparators with a converged parameter fit')
    assert rows[0] == ['Experiment', 'Dataset seed', 'Case', 'Windows', 'Median, limit of SciPy (mm)',
                       'Median, converged fit (mm)', 'Change (mm)']
    assert rows[1:] == [[group['experiment'], group['dataset_seed'], group['case'], str(group['mne_limit']['windows']),
                         f"{group['mne_limit']['median_mm']:.3f}", f"{group['raised_limit']['median_mm']:.3f}",
                         f"{group['raised_limit']['median_mm'] - group['mne_limit']['median_mm']:+.3f}"]
                        for group in expected['comparators']['groups']]
    # A change is printed with its sign.
    assert ''.join(row[6][0] for row in rows[1:]) == '---+-----+--'
    assert [row[3] for row in rows[1:]] == ['12'] * 12
    assert rows[7][:3] == ['transfer', '862', 'package_sphere_matched']


# --------------------------------------------------------------------------- one changed value stops the driver

SPLIT, COUNT, PATCH = 'split-study-v11', 'training-count-control-v11', 'patch-split-study-v11'
TRANSFER, ADAPTATION, RERUN = 'head-transfer-v11', 'head-adaptation-v11', 'validation-rerun-v11'
SPHERE, BENCHMARK = 'sphere-fit-convergence-v11', 'generation-benchmark-v11/generation_benchmark.json'
COMPARATORS = 'dipole-comparators-converged-v11'
LINUX, WINDOWS, WINDOWS_OLD = (f'{RERUN}/release-1.0.0-linux-archived', f'{RERUN}/release-1.1.0-windows',
                               f'{RERUN}/release-1.0.0-windows')
DROP = object()         # stands for the removal of a key or of a list entry


@pytest.fixture(scope='module')
def copied(miniature, tmp_path_factory):
    """A copy of the miniature folder for the corruption cases.

    Copying 154 files for every case would take most of the run time, so the cases share this copy: a case
    changes its files, runs the driver and puts the files back. At the end the copy must equal the unchanged
    folder again.
    """
    folder = tmp_path_factory.mktemp('changed') / 'evidence'
    shutil.copytree(miniature.evidence, folder)
    yield folder
    assert file_hashes(folder) == miniature.hashes, 'a corruption case did not put its file back'


def in_json(file, route, change):
    """An edit of one value of a JSON record; ``route`` lists the keys and indices that lead to the value."""
    def edit(root):
        data = json.loads((root / file).read_text(encoding='utf-8'))
        if route:
            item = data
            for key in route[:-1]:
                item = item[key]
            if change is DROP:
                del item[route[-1]]
            else:
                item[route[-1]] = change(item[route[-1]])
        else:
            data = change(data)
        (root / file).write_bytes((json.dumps(data, indent=1) + '\n').encode())

    edit.file = file
    return edit


def in_rows(file, change):
    """An edit of the rows of a CSV file; ``change`` takes the list of all rows and returns the rows to write."""
    def edit(root):
        content = (root / file).read_bytes().decode()
        rows = list(csv.DictReader(io.StringIO(content, newline='')))
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator='\r\n' if '\r\n' in content else '\n')
        writer.writeheader()
        writer.writerows(change(rows))
        (root / file).write_bytes(buffer.getvalue().encode())

    edit.file = file
    return edit


def in_csv(file, row, column, change):
    """An edit of one cell of a CSV file; ``row`` is a number or a function that picks a row from all rows."""
    def one_cell(rows):
        chosen = rows[row(rows) if callable(row) else row]
        chosen[column] = change(chosen[column], chosen)
        return rows

    return in_rows(file, one_cell)


def without_rows(file, **cells):
    """An edit that removes the rows of a CSV file which hold the given cells."""
    return in_rows(file, lambda rows: [row for row in rows if any(row[name] != value for name, value in cells.items())])


def with_row_twice(file, row):
    """An edit that appends a copy of one row of a CSV file; ``row`` picks it from all rows."""
    return in_rows(file, lambda rows: [*rows, rows[row(rows)]])


def with_cell_of(file, source, row, column):
    """An edit that copies a cell of a CSV file from the row that ``source`` picks into the row that ``row`` picks."""
    def copy(rows):
        rows[row(rows)][column] = rows[source(rows)][column]
        return rows

    return in_rows(file, copy)


def in_npz(file, name, index, amount):
    """An edit of one number of one array of an archive of arrays."""
    def edit(root):
        with np.load(root / file) as archive:
            arrays = {key: archive[key] for key in archive.files}
        arrays[name][index] += amount
        np.savez_compressed(root / file, **arrays)

    edit.file = file
    return edit


def plus(amount):
    """Add to a number of a record or of a cell."""
    return lambda old, row=None: repr(float(old) + amount) if isinstance(old, str) else old + amount


def times(factor):
    return lambda old, row=None: repr(float(old) * factor) if isinstance(old, str) else old * factor


def to(value):
    return lambda old, row=None: value


def count_plus(amount):
    """Add to a count in a cell."""
    return lambda old, row=None: str(int(old) + amount)


def rounded_plus(amount, digits):
    """Add to a cell that holds a number with a fixed number of decimals."""
    return lambda old, row=None: f'{float(old) + amount:.{digits}f}'


def where(**cells):
    """Picks the first row that holds the given cells."""
    return lambda rows: next(index for index, row in enumerate(rows)
                             if all(row[name] == value for name, value in cells.items()))


def middle(column, **cells):
    """Picks, among the rows that hold the given cells, the row with the median of a column."""
    def pick(rows):
        members = [index for index, row in enumerate(rows) if all(row[name] == value for name, value in cells.items())]
        return sorted(members, key=lambda index: float(rows[index][column]))[len(members) // 2]
    return pick


def smallest(column):
    return lambda rows: min(range(len(rows)), key=lambda index: float(rows[index][column]))


def smallest_where(column, **cells):
    """Picks, among the rows that hold the given cells, the row with the smallest value of a column."""
    def pick(rows):
        members = [index for index, row in enumerate(rows) if all(row[name] == value for name, value in cells.items())]
        return min(members, key=lambda index: float(rows[index][column]))
    return pick


def largest(column):
    return lambda rows: max(range(len(rows)), key=lambda index: float(rows[index][column]))


def in_both_releases(file, route, change):
    """The same edit in the Windows results of both releases, which must stay equal."""
    return [in_json(f'{folder}/{file}', route, change) for folder in (WINDOWS, WINDOWS_OLD)]


def without_the_last_dipole(folder, inner=''):
    """Two edits: a per-dipole file loses its last dipole, and its summary is made to agree with the file."""
    per_sample = f'{folder}/{inner}ecd_69ch_35db_per_sample.csv'
    record = f'{folder}/{inner}ecd_69ch_35db_summary.json'

    def shorten(root):
        lines = (root / per_sample).read_text(encoding='utf-8').splitlines()
        (root / per_sample).write_bytes(('\n'.join(lines[:-1]) + '\n').encode())

    def follow(root):
        rows = list(csv.DictReader(io.StringIO((root / per_sample).read_text(encoding='utf-8'))))
        position, orientation, moment, goodness = (
            [float(row[name]) for row in rows]
            for name in ('err_mm', 'orientation_error_deg', 'moment_relative_error', 'gof_pct'))
        data = json.loads((root / record).read_text(encoding='utf-8'))
        data['position_error_mm'] = dict(median=float(np.median(position)), p90=float(np.percentile(position, 90)),
                                         maximum=max(position))
        data['orientation_error_deg']['median'] = float(np.median(orientation))
        data['moment_amplitude']['relative_error']['median'] = float(np.median(moment))
        data['goodness_of_fit_percent']['median'] = float(np.median(goodness))
        (root / record).write_bytes((json.dumps(data, indent=1) + '\n').encode())

    shorten.file, follow.file = per_sample, record
    return shorten, follow


def renamed(old, new):
    """Give one entry of a record another name."""
    return lambda record: {new if name == old else name: value for name, value in record.items()}


def outside_every_band(old, row):
    """An x coordinate that puts the source 85 mm from the origin, beyond every radius band."""
    return repr((85.0 ** 2 - float(row['y_mm']) ** 2 - float(row['z_mm']) ** 2) ** 0.5)


class Whole(str):
    """A message that the driver must give as it stands, from its first character to its last."""


def whole(message):
    return Whole(message)


def case(name, message, *edits, defect=None, error=None):
    """One corruption: its name, the text that the driver's message must hold, and the edits.

    The text is a part of the message, or with ``whole`` the message from its first character to its last. A
    case with ``defect`` describes a defect of the driver and must fail until the driver is repaired; with
    ``error`` it must fail with this kind of error, which the driver raises in place of a message of its own.
    """
    marks = pytest.mark.xfail(strict=True, reason=defect, raises=error) if defect else ()
    return pytest.param(list(edits), message, id=name, marks=marks)


def summary(folder, seed, policy=None):
    return f'{folder}/{seed}/{policy}/summary.json' if policy else f'{folder}/{seed}/summary.json'


def trial(kind, name, step):
    return where(perturbed_fit=kind, sphere_fit=name, objective_scale_minus_one_in_eps=str(step))


def small_run(kernel, changed, name, step, **cells):
    """Picks a run with the small final radius or, with the cell ``sample_index``, one of its dipoles."""
    return where(kernel=kernel, changed_objective=changed, sphere_fit=name, objective_scale_minus_one_in_eps=str(step),
                 **cells)


def group(experiment, seed, name, **cells):
    """Picks the first window of a group of the dipole comparators, or the window that holds the given cells."""
    return where(experiment=experiment, dataset_seed=seed, case=name, **cells)


def last_window(experiment, seed, name):
    """Picks the last window of a group of the dipole comparators."""
    return lambda rows: max(index for index, row in enumerate(rows)
                            if (row['experiment'], row['dataset_seed'], row['case']) == (experiment, seed, name))


TRANSFER_SUMMARY, ADAPTATION_SUMMARY = f'{TRANSFER}/summary.json', f'{ADAPTATION}/summary.json'
SPHERE_SUMMARY = f'{SPHERE}/summary.json'
INVERSE, PER_SAMPLE = 'inverse/ecd_69ch_35db_summary.json', 'inverse/ecd_69ch_35db_per_sample.csv'
LINKED = 'sphere fit: the record is not linked to the validation rerun: '
UNCHANGED = ': the two per-dipole files disagree on the unchanged run'
SMALL_RUNS, SMALL_DIPOLES = f'{SPHERE}/small_tolerance.csv', f'{SPHERE}/small_tolerance_dipoles.csv'
SMALL = 'sphere fit, small final radius, kernel '
OTHER_RUNS = 'sphere fit: the two files of the runs with the small final radius hold different runs'
NO_CHANGE = (': every run except the unchanged run of the main process records how far the fitted positions moved, '
             'and that run does not')
NO_MAIN_RUN = 'sphere fit: the runs with the small final radius lack the unchanged run of the main process with '
OTHER_TRIALS = 'sphere fit: the trials of the runs with the small final radius are not those that the record names'
SMALL_RANGES = 'sphere fit: the ranges over the runs with the small final radius differ from the record'
COMPARATORS_SUMMARY, COMPARATORS_CSV = f'{COMPARATORS}/summary.json', f'{COMPARATORS}/comparators.csv'
OTHER_GROUPS = 'dipole comparators: the per-window file and the summary hold different groups'
REPORTED_FITS = 'dipole comparators: the record does not hold the sphere-model fits that the paper reports'
OTHER_WINDOWS = ': the recorded experiment fitted other windows'
OTHER_FIT = ': the fit with the limit of SciPy differs from the recorded fit'
OTHER_ERRORS = ': the position errors with the limit of SciPy differ from the recorded ones'
NOT_THE_DISTANCES = ' are not those of the fitted positions'
PAPER_FITS = ': the parameter fits did not end as the paper says'
NO_SOURCE = ': the record lacks the source position of a fitted window'
OTHER_MAIN_KERNEL = ('sphere fit: the runs with the small final radius and the record of the kernels name different '
                     'kernels of the main process')
NOT_THE_RUNS = ('sphere fit: the runs with the small final radius are not the unchanged run and the trials of the '
                'main process and the unchanged run under every other kernel. ')
TWICE = ': the record names a window twice'
TRANSFER_GROUP = ('transfer', '862', 'package_sphere_matched')


def with_the_summary_of_the_comparators(*edits):
    """Edits of the per-window file of the dipole comparators, and its summary made to agree with the file."""
    def follow(root):
        rows = list(csv.DictReader(io.StringIO((root / COMPARATORS_CSV).read_text(encoding='utf-8'))))
        data = json.loads((root / COMPARATORS_SUMMARY).read_text(encoding='utf-8'))
        for experiment in ('split_study', 'transfer', 'adaptation'):
            for seed, groups in data[experiment].items():
                for name, entry in groups.items():
                    members = [row for row in rows
                               if (row['experiment'], row['dataset_seed'], row['case']) == (experiment, seed, name)]
                    errors = {fit: [float(row[f'position_error_mm_{fit}']) for row in members]
                              for fit in ('mne_limit', 'raised_limit')}
                    for fit, values in errors.items():
                        entry[fit] = dict(windows=len(values), median_mm=float(np.median(values)),
                                          mean_mm=float(np.mean(values)), p90_mm=float(np.percentile(values, 90)))
                    entry['median_of_raised_limit_minus_median_of_mne_limit_mm'] = float(
                        np.median(errors['raised_limit']) - np.median(errors['mne_limit']))
        (root / COMPARATORS_SUMMARY).write_bytes((json.dumps(data, indent=1) + '\n').encode())

    follow.file = COMPARATORS_SUMMARY
    return *edits, follow


def members_of(rows, experiment, seed, name):
    """The windows of a group of the dipole comparators among the rows of the per-window file."""
    return [row for row in rows if (row['experiment'], row['dataset_seed'], row['case']) == (experiment, seed, name)]


def without_the_last_window(experiment, seed, name):
    """An edit of the per-window file of the dipole comparators: a group loses its last window."""
    def change(rows):
        last = members_of(rows, experiment, seed, name)[-1]
        return [row for row in rows if row is not last]

    return in_rows(COMPARATORS_CSV, change)


def both_middle_errors(experiment, seed, name, amount):
    """An edit of the two middle errors of a group with the limit of SciPy: its median moves by the amount."""
    def change(rows):
        members = sorted(members_of(rows, experiment, seed, name),
                         key=lambda row: float(row['position_error_mm_mne_limit']))
        for row in members[len(members) // 2 - 1:len(members) // 2 + 1]:
            row['position_error_mm_mne_limit'] = repr(float(row['position_error_mm_mne_limit']) + amount)
        return rows

    return in_rows(COMPARATORS_CSV, change)


def exchanged(experiment, seed, name, column):
    """An edit of the per-window file of the dipole comparators: the first two windows of a group exchange a cell."""
    def change(rows):
        first, second = members_of(rows, experiment, seed, name)[:2]
        first[column], second[column] = second[column], first[column]
        return rows

    return in_rows(COMPARATORS_CSV, change)


def without_group(experiment, seed, name):
    """Two edits that take a group of the dipole comparators out of the per-window file and out of the summary."""
    return (without_rows(COMPARATORS_CSV, experiment=experiment, dataset_seed=seed, case=name),
            in_json(COMPARATORS_SUMMARY, [experiment, seed, name], DROP))


def group_as(source, target, keep=False):
    """Two edits that give a group of the dipole comparators another label, in the per-window file and in the summary.

    With ``keep`` the group stays under its own label as well, so that the record holds one group more.
    """
    def relabel(rows):
        own = members_of(rows, *source)
        if keep:
            return [*rows, *[dict(row, experiment=target[0], dataset_seed=target[1], case=target[2]) for row in own]]
        for row in own:
            row.update(experiment=target[0], dataset_seed=target[1], case=target[2])
        return rows

    def rename(data):
        entry = data[source[0]][source[1]][source[2]] if keep else data[source[0]][source[1]].pop(source[2])
        data[target[0]].setdefault(target[1], {})[target[2]] = entry
        return data

    return in_rows(COMPARATORS_CSV, relabel), in_json(COMPARATORS_SUMMARY, [], rename)


def parameter_fit_in_both_records(sphere, name, key, value):
    """Two edits that give a parameter fit of a sphere another end in both records that hold it.

    The sphere-fit record changes, and alike every group of the dipole comparators that was fitted with the
    sphere: the two records still agree.
    """
    def groups(data):
        for experiment in ('split_study', 'transfer', 'adaptation'):
            for cases in data[experiment].values():
                for label, entry in cases.items():
                    if (label[:-len('_sphere_on_bem')] if experiment == 'adaptation' else 'default') == sphere:
                        entry['parameter_fit'][name][key] = value
        return data

    return (in_json(COMPARATORS_SUMMARY, [], groups),
            in_json(SPHERE_SUMMARY, ['spheres', 'variants', sphere, name, 'fit', key], to(value)))


def source_of_a_fitted_window(folder, subset, change=None):
    """An edit of the eligibility file of an experiment at the first window of its dipole-fit subset.

    The x coordinate of the true source position is changed or, without a change, the window is taken out.
    """
    file = f'{folder}/862/eligibility.csv'

    def edit(root):
        fitted = list(csv.DictReader(io.StringIO((root / folder / '862' / subset).read_text(encoding='utf-8'))))
        window = fitted[0]['window_id']
        if change is None:
            without_rows(file, window_id=window)(root)
        else:
            in_csv(file, where(window_id=window), 'x_mm', change)(root)

    edit.file = file
    return edit


def in_both_small_files(under, changed, name, step, copy=False, **labels):
    """Two edits of a run with the small final radius, in the list of the runs and in the per-dipole file alike.

    The run is named by the kernel under which it was made, its changed objective, its sphere fit and the
    factor of the objective. It gets other labels, which are cells of the files; with ``copy`` it stays as it
    is and a copy of it gets them. Without labels the run is taken out of both files.
    """
    own = dict(kernel=under, changed_objective=changed, sphere_fit=name, objective_scale_minus_one_in_eps=str(step))

    def change(rows):
        chosen = [row for row in rows if all(row[key] == value for key, value in own.items())]
        if not labels:
            return [row for row in rows if not any(row is item for item in chosen)]
        if copy:
            return [*rows, *[dict(row, **labels) for row in chosen]]
        for row in chosen:
            row.update(labels)
        return rows

    return in_rows(SMALL_RUNS, change), in_rows(SMALL_DIPOLES, change)


def trial_as(old, new):
    """Three edits that give every run with one changed objective another factor, in both files and in the record."""
    def change(rows):
        for row in rows:
            if row['changed_objective'] != 'none' and row['objective_scale_minus_one_in_eps'] == str(old):
                row['objective_scale_minus_one_in_eps'] = str(new)
        return rows

    return (in_rows(SMALL_RUNS, change), in_rows(SMALL_DIPOLES, change),
            in_json(SPHERE_SUMMARY, ['small_tolerance', 'steps'],
                    lambda steps: [new if step == old else step for step in steps]))


def with_the_ranges_of_the_small_runs(*edits):
    """Edits of the list of the runs with the small final radius, and the ranges of the record made to agree."""
    def follow(root):
        runs = list(csv.DictReader(io.StringIO((root / SMALL_RUNS).read_text(encoding='utf-8'))))
        data = json.loads((root / SPHERE_SUMMARY).read_text(encoding='utf-8'))
        for name, entry in data['small_tolerance']['ranges'].items():
            own = [run for run in runs if run['sphere_fit'] == name]
            entry.update(runs=len(own), kernels=sorted({run['kernel'] for run in own}))
            for key in ('dipole_fits', 'dipole_fits_stopped_by_themselves'):
                entry[key] = sum(int(run[key]) for run in own)
            entry['runs_in_which_the_parameter_fit_stopped_by_itself'] = sum(
                int(run['parameter_fit_stopped_by_itself']) for run in own)
            for column in [key for key in entry if key.endswith('_mm')]:
                values = [float(run[column]) for run in own if run[column] != '']
                entry[column] = [min(values), max(values)]
        (root / SPHERE_SUMMARY).write_bytes((json.dumps(data, indent=1) + '\n').encode())

    follow.file = SPHERE_SUMMARY
    return *edits, follow


def main_kernel_as(name):
    """An edit of the sphere-fit record: both of its statements name another kernel of the main process."""
    def change(data):
        data['small_tolerance']['kernel_of_the_main_process'] = name
        for entry in data['library_kernels']['of_the_parent_process']:
            entry['kernel'] = name
        return data

    return in_json(SPHERE_SUMMARY, [], change)


def fitted_window_twice(folder, subset):
    """An edit of the eligibility file of an experiment: the first window of its dipole-fit subset stands twice.

    The first of its two rows gives another source position and the second the true one. A reader that keeps the
    last row of a window finds the errors of the record to be the distances, so only the count of the windows
    can notice.
    """
    file = f'{folder}/862/eligibility.csv'

    def edit(root):
        fitted = list(csv.DictReader(io.StringIO((root / folder / '862' / subset).read_text(encoding='utf-8'))))
        window = fitted[0]['window_id']

        def change(rows):
            own = next(row for row in rows if row['window_id'] == window)
            return [dict(own, x_mm=repr(float(own['x_mm']) + 5.0)), *rows]

        in_rows(file, change)(root)

    edit.file = file
    return edit


def exchanged_rows(file, column, first=0, second=1):
    """An edit that exchanges one cell between two rows of a per-window file of an experiment."""
    def change(rows):
        rows[first][column], rows[second][column] = rows[second][column], rows[first][column]
        return rows

    return in_rows(file, change)


def boundary_element_fit_as_a_group(replace=False):
    """Two edits that add the boundary-element fit of the transfer experiment to the dipole comparators.

    The group agrees with itself and with the transfer experiment: with both parameter fits it has the recorded
    positions and errors of that fit, so that nothing but its presence is wrong. With ``replace`` it stands in
    place of the matched-sphere fit, and the record holds as many groups as the paper reports.
    """
    def recorded(root):
        return list(csv.DictReader(io.StringIO((root / TRANSFER / '862/ecd_subset.csv').read_text(encoding='utf-8'))))

    def add_rows(root):
        fit = recorded(root)
        in_rows(COMPARATORS_CSV, lambda rows: [
            *[row for row in rows if not replace or row['experiment'] != 'transfer'],
            *[{'experiment': 'transfer', 'dataset_seed': '862', 'case': 'bem_matched', 'window_id': row['window_id'],
               **{f'position_error_mm_{name}': row['ecd_bem_matched_error_mm']
                  for name in ('mne_limit', 'raised_limit')},
               **{f'fitted_{axis}_mm_{name}': row[f'ecd_bem_matched_{axis}_mm']
                  for name in ('mne_limit', 'raised_limit') for axis in 'xyz'}} for row in fit]])(root)

    def add_entry(root):
        errors = [float(row['ecd_bem_matched_error_mm']) for row in recorded(root)]
        statistics = dict(windows=len(errors), median_mm=float(np.median(errors)), mean_mm=float(np.mean(errors)),
                          p90_mm=float(np.percentile(errors, 90)))

        def change(data):
            groups = data['transfer']['862']
            matched = groups.pop('package_sphere_matched') if replace else groups['package_sphere_matched']
            groups['bem_matched'] = dict(
                matched, mne_limit=statistics, raised_limit=statistics,
                median_of_raised_limit_minus_median_of_mne_limit_mm=0.0,
                fitted_position_change_mm=dict(median=0.0, p90=0.0, maximum=0.0))
            return data

        in_json(COMPARATORS_SUMMARY, [], change)(root)

    add_rows.file, add_entry.file = COMPARATORS_CSV, COMPARATORS_SUMMARY
    return add_rows, add_entry


def without_entry(name):
    """A record without one of its entries."""
    return lambda record: {key: value for key, value in record.items() if key != name}


def in_the_opposite_order(record):
    """The entries of a record, or the rows of a list, in the opposite order."""
    return dict(reversed(list(record.items()))) if isinstance(record, dict) else record[::-1]


def with_entry_as(old, new):
    """A record that holds one of its entries once more, at its end and under another name."""
    return lambda record: {**record, new: record[old]}


def without(*labels):
    """A record without some of its entries."""
    return lambda record: {name: value for name, value in record.items() if name not in labels}


def in_the_strata_of_every_method(file, panel, change):
    """An edit of a dataset summary of the split study: the strata of every method of a panel change alike."""
    def every(data):
        for entry in data[panel].values():
            if 'strata' in entry:
                entry['strata'] = change(entry['strata'])
        return data

    return in_json(file, [], every)


POSITIONS = ': the number of distinct positions differs from the record'
OTHER_STRATA = ': the strata are not those of the pooled network, in the same order'
NO_STRATA = (': the record has no stratum under a label that another record of the study uses and for which the '
             'dataset has windows: ')
OTHER_PATCH_STRATA = ': the strata are not those of the first dataset, in the same order'
OTHER_DATASET = ': the dataset seed of a row of a single network is not written as '
STUDY_SUMMARY = f'{SPLIT}/summary.json'
OTHER_STUDY_SUMMARY = 'split study: the study-level summary differs from the dataset summaries'
OTHER_ARMS = 'adaptation: the arms differ between the datasets'
# One stratum statistic of every method of both panels of the split study: the panel, the method, the dataset,
# the stratum and the statistic. Every dataset and each of the three statistics occur in both panels. Among the
# strata are the first and the last of a record, two with a single window, and a stratum of the subset that
# holds two windows of one position.
STRATUM_STATISTICS = (
    ('full', 'mlp_pooled', '862', 'position', 'population_0', 'mean_mm'),
    ('full', 'dictionary', '862', 'region', 'snr_15_20', 'p90_mm'),
    ('full', 'knn1', '863', 'position', 'radius_1_30_45', 'median_mm'),
    ('full', 'knn10', '863', 'region', 'distance_30_inf', 'mean_mm'),
    ('full', 'training_mean', '864', 'position', 'within_montage', 'p90_mm'),
    ('full', 'mlp_862', '864', 'region', 'radius_0_75_79.501', 'median_mm'),
    ('full', 'mlp_863', '862', 'position', 'distance_10_20', 'mean_mm'),
    ('ecd_subset', 'mlp_pooled', '862', 'region', 'population_1', 'p90_mm'),
    ('ecd_subset', 'dictionary', '863', 'position', 'snr_20_25.001', 'median_mm'),
    ('ecd_subset', 'knn1', '863', 'region', 'below_montage', 'mean_mm'),
    ('ecd_subset', 'knn10', '864', 'position', 'distance_5_10', 'p90_mm'),
    ('ecd_subset', 'dipole', '864', 'region', 'radius_0_70.5_75', 'median_mm'),
    ('ecd_subset', 'training_mean', '862', 'position', 'radius_1_55_65.001', 'mean_mm'),
    ('ecd_subset', 'mlp_862', '862', 'region', 'within_montage', 'p90_mm'),
    ('ecd_subset', 'mlp_863', '863', 'position', 'radius_1_30_45', 'median_mm'))
# Strata of one method that are not those of the pooled network, for every other method of both panels: the
# panel, the method, the dataset, what is different, and the change of the strata. A stratum more is one that
# holds no window of the panel, with the record of another stratum.
STRATA_OF_ONE_METHOD = (
    ('full', 'dictionary', '862', 'position', 'without the first of them', without_entry('radius_1_30_45')),
    ('full', 'knn1', '862', 'region', 'in the opposite order', in_the_opposite_order),
    ('full', 'knn10', '863', 'position', 'with one stratum more', with_entry_as('distance_5_10', 'distance_10_20')),
    ('full', 'training_mean', '863', 'region', 'without the last of them', without_entry('within_montage')),
    ('full', 'mlp_862', '864', 'position', 'with one stratum more', with_entry_as('snr_15_20', 'snr_25_30')),
    ('full', 'mlp_863', '864', 'region', 'in the opposite order', in_the_opposite_order),
    ('ecd_subset', 'dictionary', '862', 'position', 'with one stratum more',
     with_entry_as('radius_1_30_45', 'radius_1_45_55')),
    ('ecd_subset', 'knn1', '862', 'region', 'without the first of them', without_entry('radius_1_30_45')),
    ('ecd_subset', 'knn10', '863', 'position', 'in the opposite order', in_the_opposite_order),
    ('ecd_subset', 'dipole', '863', 'region', 'without one of them', without_entry('population_1')),
    ('ecd_subset', 'training_mean', '864', 'position', 'with one stratum more',
     with_entry_as('snr_15_20', 'snr_25_30')),
    ('ecd_subset', 'mlp_862', '864', 'region', 'in the opposite order', in_the_opposite_order),
    ('ecd_subset', 'mlp_863', '862', 'position', 'without the last of them', without_entry('within_montage')))
# One stratum statistic of every method of the patch study besides the pooled network: the method, the dataset,
# the stratum and the statistic.
PATCH_STRATUM_STATISTICS = (
    ('nearest_training_patch', '862', 'region', 'diameter_10_25', 'median_mm'),
    ('ten_nearest_training_patches', '863', 'position', 'snr_20_25.001', 'mean_mm'),
    ('training_mean', '863', 'region', 'diameter_40_60.001', 'p90_mm'),
    ('mlp_862', '864', 'position', 'snr_10_15', 'mean_mm'),
    ('mlp_863', '864', 'region', 'diameter_25_40', 'median_mm'))


def panel_of(panel):
    """How a message names a panel of the split study, and how a case names it."""
    return ('', 'all windows') if panel == 'full' else (', ecd_subset', 'the dipole-fit subset')


def stratum_statistic_cases():
    """A case for every row of STRATUM_STATISTICS. The message for the panel of all windows does not name it."""
    cases = []
    for panel, method, seed, policy, label, key in STRATUM_STATISTICS:
        where, name = panel_of(panel)
        cases.append(case(
            f'a stratum statistic of {method} on {name}',
            f'split study, dataset {seed}, {policy}{where}, stratum {label}, {method}, {key}: recomputed',
            in_json(summary(SPLIT, seed, policy), [panel, method, 'strata', label, key], plus(0.01))))
    return cases


def strata_of_one_method_cases():
    """A case for every row of STRATA_OF_ONE_METHOD: the message names the method whose strata differ."""
    cases = []
    for panel, method, seed, policy, what, change in STRATA_OF_ONE_METHOD:
        where, name = panel_of(panel)
        cases.append(case(f'the strata of {method} on {name} {what}',
                          whole(f'split study, dataset {seed}, {policy}{where}, {method}' + OTHER_STRATA),
                          in_json(summary(SPLIT, seed, policy), [panel, method, 'strata'], change)))
    return cases


def stratum_position_cases():
    """The number of positions of one stratum of every method of both panels, one too few or one too many in turn.

    The strata are those of STRATUM_STATISTICS. The record of every method holds the number and each one is
    compared; the message names the stratum and not the method.
    """
    cases = []
    for number, (panel, method, seed, policy, label, _) in enumerate(STRATUM_STATISTICS):
        where, name = panel_of(panel)
        cases.append(case(
            f'the positions of a stratum of {method} on {name}',
            whole(f'split study, dataset {seed}, {policy}{where}, stratum {label}' + POSITIONS),
            in_json(summary(SPLIT, seed, policy), [panel, method, 'strata', label, 'positions'],
                    plus(1 if number % 2 else -1))))
    return cases


def missing_stratum_cases():
    """A stratum that every method lacks, for every dataset and both panels in turn.

    Every method of the panel lacks the stratum, so that the strata of the methods agree and every recorded
    number is right; another dataset, or the other panel, still records it. The stratum changes from case to
    case, and each of the six holds windows of every panel of the miniature.
    """
    labels = ('population_0', 'snr_15_20', 'below_montage', 'radius_0_70.5_75', 'within_montage', 'population_1')
    cases = []
    for number, (seed, policy) in enumerate(DATASETS):
        for panel in tables.PANELS:
            where, name = panel_of(panel)
            label = labels[(number + (panel != 'full')) % len(labels)]
            cases.append(case(f'a stratum that every method lacks on {name} of dataset {seed}, {policy}',
                              whole(f'split study, dataset {seed}, {policy}{where}' + NO_STRATA + label),
                              in_the_strata_of_every_method(summary(SPLIT, seed, policy), panel, without(label))))
    return cases


def network(kind, size, fit):
    """The place of a single network among the rows of a dataset in the summary of the adaptation experiment."""
    return SINGLE_NETWORKS.index((kind, size, fit))


def single_network_cases():
    """A statistic of every single network of the adaptation experiment.

    The dataset and the statistic change from network to network: each of the three statistics is changed in
    every dataset.
    """
    cases = []
    for number, (kind, size, fit) in enumerate(SINGLE_NETWORKS):
        seed, key = SEEDS[number % 3], STATISTICS[(number + number // 3) % 3]
        cases.append(case(f'a statistic of the network {fit} of {kind}_{size}',
                          f'adaptation, dataset {seed}, {kind}_{size}, network {fit}, {key}: recomputed',
                          in_json(ADAPTATION_SUMMARY, ['by_dataset_seed', seed, 'finetune', number, key], plus(0.01))))
    return cases


def exchanged_cells(file, row, first, second):
    """An edit that exchanges two cells of one row of a CSV file."""
    def change(rows):
        rows[row][first], rows[row][second] = rows[row][second], rows[row][first]
        return rows

    return in_rows(file, change)


def network_of_another_size(seed, size, like):
    """Two edits that add a single network of a size that no arm has to a dataset of the adaptation experiment.

    The prediction file gets a column with its errors, which are those of the network ``like``, and the summary
    a row with the statistics of these errors: the record agrees with itself, and only the arms lack the network.
    """
    kind, _, fit = like
    file = f'{ADAPTATION}/{seed}/adaptation_predictions.csv'

    def add_column(root):
        content = (root / file).read_bytes().decode()
        rows = list(csv.DictReader(io.StringIO(content, newline='')))
        source, target = (f'{kind}_{item}_mlp_{fit}_error_mm' for item in (like[1], size))
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=[*rows[0], target],
                                lineterminator='\r\n' if '\r\n' in content else '\n')
        writer.writeheader()
        writer.writerows({**row, target: row[source]} for row in rows)
        (root / file).write_bytes(buffer.getvalue().encode())

    add_column.file = file
    return add_column, in_json(ADAPTATION_SUMMARY, ['by_dataset_seed', seed, 'finetune'],
                               lambda rows: [*rows, dict(rows[network(*like)], size=size)])


def runs_under(kernel, *kinds):
    """Runs with the small final radius under one kernel, as a message of the driver lists them.

    ``kinds`` are the changed objectives of the runs, each with the factors of its runs; every run was made
    with both sphere fits. The driver lists runs in the order of their kernel, their changed objective, their
    sphere fit and their factor.
    """
    return '; '.join(f'kernel {kernel}, {changed} {step} with {name}' for changed, steps in kinds
                     for name in ('mne_limit', 'raised_limit') for step in steps)


def with_the_main_process_as(name, kernels, old='Haswell'):
    """Three edits of the sphere-fit record: other libraries of linear algebra in the main process.

    ``kernels`` are the kernels that the libraries of the main process report, in the order of the record, and
    ``name`` is the name that the record gives to the kernel of the main process. The runs of the main process
    carry this name in both files of the runs with the small final radius, and the ranges list it among their
    kernels: the record agrees with itself in everything but, possibly, the rule by which the name is made.
    """
    def rename(rows):
        for row in rows:
            if row['kernel'] == old:
                row['kernel'] = name
        return rows

    def change(data):
        data['small_tolerance']['kernel_of_the_main_process'] = name
        for entry in data['small_tolerance']['ranges'].values():
            entry['kernels'] = sorted(name if kernel == old else kernel for kernel in entry['kernels'])
        # A library that is not OpenBLAS reports no kernel, which the experiment script writes as 'None'.
        data['library_kernels']['of_the_parent_process'] = [
            dict(library='mkl' if kernel == 'None' else 'openblas', version=f'0.3.{31 - number}', kernel=kernel)
            for number, kernel in enumerate(kernels)]
        return data

    return in_rows(SMALL_RUNS, rename), in_rows(SMALL_DIPOLES, rename), in_json(SPHERE_SUMMARY, [], change)


def without_the_other_kernel(kernel):
    """Edits that take a kernel which was not available out of the sphere-fit record.

    The experiment script then writes no run, no row and no entry for the kernel, and lists it as not available.
    """
    def change(data):
        del data['library_kernels']['arms'][kernel]
        data['library_kernels']['not_available'][kernel] = 'the library kept the kernel of the parent process'
        return data

    return (without_rows(f'{SPHERE}/kernels_dipole_recovery_35db.csv', kernel=kernel),
            without_rows(f'{SPHERE}/kernels_field_differences.csv', kernel=kernel),
            *with_the_ranges_of_the_small_runs(without_rows(SMALL_RUNS, kernel=kernel),
                                               without_rows(SMALL_DIPOLES, kernel=kernel)),
            in_json(SPHERE_SUMMARY, [], change))


CORRUPTIONS = [
    # ---- split study
    case('a dataset median, by 5e-10 mm', 'split study, dataset 863, region, full, mlp_pooled, median_mm: recomputed',
         in_json(summary(SPLIT, '863', 'region'), ['full', 'mlp_pooled', 'median_mm'], plus(5e-10))),
    case('the mean of one fitted network', 'split study, dataset 862, position, full, mlp_863, mean_mm: recomputed',
         in_json(summary(SPLIT, '862', 'position'), ['full', 'mlp_863', 'mean_mm'], plus(0.01))),
    case('a 90th percentile on the dipole-fit subset',
         'split study, dataset 864, position, ecd_subset, dipole, p90_mm: recomputed',
         in_json(summary(SPLIT, '864', 'position'), ['ecd_subset', 'dipole', 'p90_mm'], plus(0.01))),
    case('a recorded number of windows', 'split study, dataset 862, position, full, knn10: 72 windows, recorded 73',
         in_json(summary(SPLIT, '862', 'position'), ['full', 'knn10', 'windows'], plus(1))),
    case('a record without a statistic', 'split study, dataset 864, region, full, knn1: the record lacks p90_mm',
         in_json(summary(SPLIT, '864', 'region'), ['full', 'knn1', 'p90_mm'], DROP)),
    case('a predicted position', 'split study, dataset 862, position, full, mlp_pooled, me',
         in_csv(f'{SPLIT}/862/position/predictions.csv', 5, 'mlp_862_x_mm', plus(30.0))),
    case('a predicted position on the dipole-fit subset', 'split study, dataset 863, region, ecd_subset, dipole, me',
         in_csv(f'{SPLIT}/863/region/ecd_subset/predictions.csv', 2, 'dipole_z_mm', plus(30.0))),
    case('a paired contrast with a comparator', 'split study, dataset 862, region, ecd_subset, network minus dipole',
         in_json(summary(SPLIT, '862', 'region'),
                 ['ecd_subset', 'paired_mlp_minus_baseline', 'dipole', 'median_difference_mm'], plus(0.01))),
    case('a paired contrast on all windows', 'split study, dataset 863, position, full, network minus knn1',
         in_json(summary(SPLIT, '863', 'position'),
                 ['full', 'paired_mlp_minus_baseline', 'knn1', 'median_difference_mm'], plus(-0.01))),
    case('a stratum position count',
         'split study, dataset 863, position, stratum population_1: the number of distinct positions differs',
         in_json(summary(SPLIT, '863', 'position'), ['full', 'mlp_pooled', 'strata', 'population_1', 'positions'],
                 plus(1))),
    case('a stratum position count at the rule of 30',
         'split study, dataset 864, position, stratum below_montage: the number of distinct positions differs',
         in_json(summary(SPLIT, '864', 'position'), ['full', 'mlp_pooled', 'strata', 'below_montage', 'positions'],
                 plus(1))),
    case('a stratum median of the dictionary',
         'split study, dataset 864, region, stratum snr_15_20, dictionary, median_mm: recomputed',
         in_json(summary(SPLIT, '864', 'region'), ['full', 'dictionary', 'strata', 'snr_15_20', 'median_mm'],
                 plus(0.01))),
    case('a stratum 90th percentile of the network',
         'split study, dataset 862, position, stratum below_montage, mlp_pooled, p90_mm: recomputed',
         in_json(summary(SPLIT, '862', 'position'), ['full', 'mlp_pooled', 'strata', 'below_montage', 'p90_mm'],
                 plus(0.01))),
    # Every method of the panel has the stratum under the unknown label, so that the strata of the methods agree.
    case('a stratum with an unknown label', whole('Unknown stratum label: depth_1_2'),
         in_the_strata_of_every_method(summary(SPLIT, '863', 'position'), 'full',
                                       renamed('population_0', 'depth_1_2'))),
    case('a stratum of the dipole-fit subset with an unknown label', whole('Unknown stratum label: depth_1_2'),
         in_the_strata_of_every_method(summary(SPLIT, '864', 'region'), 'ecd_subset',
                                       renamed('population_1', 'depth_1_2'))),
    case('a missing fitted network',
         'split study, dataset 862, position: the fitted networks are not those of the model seeds',
         in_json(summary(SPLIT, '862', 'position'), ['fits', 'mlp_863'], DROP)),
    # So small a change leaves every statistic of the training mean within the tolerance of its record.
    case('a training mean that differs between windows, by 1e-11 mm', 'Training-mean comparator is not constant',
         in_csv(f'{SPLIT}/863/region/predictions.csv', 9, 'training_mean_y_mm', plus(1e-11))),
    case('a headline with the median of a fourth dataset',
         "split study headline, position, full, knn1: values recorded for the datasets ['862', '863', '864', '865']",
         in_json(f'{SPLIT}/headline.json', ['position', 'full', 'knn1', 'dataset_seed_medians_mm'],
                 lambda medians: dict(medians, **{'865': 1.0}))),
    case('a headline whose mean is not a number',
         'split study headline, region, full, dictionary, mean over the datasets: the record holds no number (None)',
         in_json(f'{SPLIT}/headline.json', ['region', 'full', 'dictionary', 'mean_median_mm'], to(None))),
    case('a headline without one method',
         'split study headline, region, ecd_subset: the methods differ from the dataset records',
         in_json(f'{SPLIT}/headline.json', ['region', 'ecd_subset', 'dipole'], DROP)),
    case('a dataset median of the headline', 'split study headline, position, full, knn1, dataset 863: recomputed',
         in_json(f'{SPLIT}/headline.json', ['position', 'full', 'knn1', 'dataset_seed_medians_mm', '863'], plus(0.01))),
    case('a headline mean', 'split study headline, position, full, mlp_pooled, mean over the datasets: recomputed',
         in_json(f'{SPLIT}/headline.json', ['position', 'full', 'mlp_pooled', 'mean_median_mm'], plus(0.5))),
    case('the upper end of a headline range',
         'split study headline, region, full, dictionary, largest dataset value: recomputed',
         in_json(f'{SPLIT}/headline.json', ['region', 'full', 'dictionary', 'range_mm', 1], plus(1.0))),
    case('the lower end of a headline range',
         'split study headline, region, ecd_subset, dipole, smallest dataset value: recomputed',
         in_json(f'{SPLIT}/headline.json', ['region', 'ecd_subset', 'dipole', 'range_mm', 0], plus(-0.01))),
    case('the window lists of the two regimes', 'The two training regimes were scored on different windows',
         in_csv(f'{SPLIT}/863/same_test_windows/region/predictions.csv', 3, 'window', to('99'))),
    case('the regional contrast of a dataset', 'paired split contrast, dataset 864: recomputed',
         in_json(f'{SPLIT}/864/same_test_windows/paired_mlp.json', ['region_minus_position_mm'], plus(0.01))),
    case('the windows of a regional contrast',
         'paired split contrast, dataset 862: windows or positions differ from the record',
         in_json(f'{SPLIT}/862/same_test_windows/paired_mlp.json', ['windows'], plus(-1))),
    case('the positions of a regional contrast',
         'paired split contrast, dataset 863: windows or positions differ from the record',
         in_json(f'{SPLIT}/863/same_test_windows/paired_mlp.json', ['distinct_positions'], plus(1))),
    case('the regional contrast over all datasets', 'paired split contrast, three datasets: recomputed',
         in_json(f'{SPLIT}/paired_split_summary.json', ['strata', 'all', 'region_minus_position_mm'], plus(0.01))),
    case('the windows of the regional contrast over all datasets',
         'paired split contrast: the number of windows differs from the record',
         in_json(f'{SPLIT}/paired_split_summary.json', ['strata', 'all', 'windows'], plus(1))),
    case('the regional contrast of a stratum',
         'paired split contrast, three datasets, stratum population_0: recomputed',
         in_json(f'{SPLIT}/paired_split_summary.json', ['strata', 'population_0', 'region_minus_position_mm'],
                 plus(1e-9))),
    case('the windows of a stratum of the regional contrast',
         'paired split contrast, stratum population_1: the number of windows differs from the record',
         in_json(f'{SPLIT}/paired_split_summary.json', ['strata', 'population_1', 'windows'], plus(1))),
    case('the positions of the regional contrast over all datasets',
         'paired split contrast: the number of positions differs from the record',
         in_json(f'{SPLIT}/paired_split_summary.json', ['strata', 'all', 'seed_position_clusters'], plus(1))),
    case('the positions of a stratum of the regional contrast',
         'paired split contrast, stratum population_0: the number of positions differs from the record',
         in_json(f'{SPLIT}/paired_split_summary.json', ['strata', 'population_0', 'seed_position_clusters'],
                 plus(-1))),
    case('a regional contrast without the stratum of all windows',
         'paired split contrast: the record lacks the contrast over all windows',
         in_json(f'{SPLIT}/paired_split_summary.json', ['strata', 'all'], DROP)),

    # ---- split study: every method and each fitted network by stratum, in both panels
    *stratum_statistic_cases(),
    case('a stratum 90th percentile of the dipole fit, by 5e-10 mm',
         'split study, dataset 862, position, ecd_subset, stratum population_0, dipole, p90_mm: recomputed',
         in_json(summary(SPLIT, '862', 'position'), ['ecd_subset', 'dipole', 'strata', 'population_0', 'p90_mm'],
                 plus(5e-10))),
    case('a stratum mean of one fitted network, by 5e-10 mm',
         'split study, dataset 863, region, stratum within_montage, mlp_863, mean_mm: recomputed',
         in_json(summary(SPLIT, '863', 'region'), ['full', 'mlp_863', 'strata', 'within_montage', 'mean_mm'],
                 plus(-5e-10))),
    case('the windows of a stratum of the dipole fit',
         whole('split study, dataset 862, position, ecd_subset, stratum population_0, dipole: 9 windows, recorded 10'),
         in_json(summary(SPLIT, '862', 'position'), ['ecd_subset', 'dipole', 'strata', 'population_0', 'windows'],
                 plus(1))),
    case('the windows of a stratum of a lookup method',
         whole('split study, dataset 863, region, stratum snr_10_15, knn10: 14 windows, recorded 13'),
         in_json(summary(SPLIT, '863', 'region'), ['full', 'knn10', 'strata', 'snr_10_15', 'windows'], plus(-1))),
    case('a stratum of one fitted network without a statistic',
         whole('split study, dataset 864, region, ecd_subset, stratum below_montage, mlp_863: the record lacks '
               'mean_mm'),
         in_json(summary(SPLIT, '864', 'region'), ['ecd_subset', 'mlp_863', 'strata', 'below_montage', 'mean_mm'],
                 DROP)),
    # Two methods differ in one stratum: the driver stops at the first of them in the order of the panel.
    case('a stratum statistic of two methods on the dipole-fit subset',
         'split study, dataset 862, position, ecd_subset, stratum population_0, knn1, median_mm: recomputed',
         *[in_json(summary(SPLIT, '862', 'position'), ['ecd_subset', method, 'strata', 'population_0', 'median_mm'],
                   plus(0.01)) for method in ('mlp_863', 'knn1')]),
    # The strata of the subset are those of its own windows: a window with another label moves between two of
    # them, and no statistic of the panel notices.
    case('a window of the dipole-fit subset on the other side of the montage',
         whole('split study, dataset 863, position, ecd_subset, stratum below_montage' + POSITIONS),
         in_csv(f'{SPLIT}/863/position/ecd_subset/predictions.csv', 0, 'below_montage',
                lambda old, row: str(1 - int(old)))),
    # The record of every method holds the number of positions of a stratum, and each one is compared: in both
    # panels, for a lookup method, for the dipole fit and for the last fitted network, in both directions.
    case('a stratum position count of a lookup method',
         whole('split study, dataset 862, region, stratum snr_15_20' + POSITIONS),
         in_json(summary(SPLIT, '862', 'region'), ['full', 'knn10', 'strata', 'snr_15_20', 'positions'], plus(1))),
    case('a stratum position count of the last fitted network',
         whole('split study, dataset 864, position, stratum population_0' + POSITIONS),
         in_json(summary(SPLIT, '864', 'position'), ['full', 'mlp_863', 'strata', 'population_0', 'positions'],
                 plus(-1))),
    case('a stratum position count on the dipole-fit subset',
         whole('split study, dataset 863, region, ecd_subset, stratum population_1' + POSITIONS),
         in_json(summary(SPLIT, '863', 'region'), ['ecd_subset', 'mlp_pooled', 'strata', 'population_1', 'positions'],
                 plus(1))),
    case('a stratum position count of the dipole fit',
         whole('split study, dataset 862, position, ecd_subset, stratum below_montage' + POSITIONS),
         in_json(summary(SPLIT, '862', 'position'), ['ecd_subset', 'dipole', 'strata', 'below_montage', 'positions'],
                 plus(-1))),
    case('a stratum position count of the last fitted network on the dipole-fit subset',
         whole('split study, dataset 864, position, ecd_subset, stratum within_montage' + POSITIONS),
         in_json(summary(SPLIT, '864', 'position'), ['ecd_subset', 'mlp_863', 'strata', 'within_montage', 'positions'],
                 plus(1))),
    # In one dataset two windows of the subset lie at one position. A record that counts the windows of every
    # stratum in place of its positions, for every method alike, differs in the first stratum that holds both.
    case('strata of the dipole-fit subset with their windows counted as positions',
         whole('split study, dataset 864, region, ecd_subset, stratum radius_0_70.5_75' + POSITIONS),
         in_the_strata_of_every_method(
             summary(SPLIT, '864', 'region'), 'ecd_subset',
             lambda strata: {label: dict(entry, positions=entry['windows']) for label, entry in strata.items()})),
    # The same for one stratum of every method of both panels: no method is left out of the comparison.
    *stratum_position_cases(),
    # Every method of a panel has the strata of the pooled network, under the same labels in the same order:
    # no stratum of a method is left without a comparison, and none is looked for in vain.
    *strata_of_one_method_cases(),
    # One stratum of one method stands under another label, the first or the last of its strata: as many strata
    # as the pooled network has, and every other label in its place.
    case('the first stratum of one method under another label',
         whole('split study, dataset 862, region, knn10' + OTHER_STRATA),
         in_json(summary(SPLIT, '862', 'region'), ['full', 'knn10', 'strata'],
                 renamed('radius_1_30_45', 'radius_1_30_44'))),
    case('the last stratum of one method on the dipole-fit subset under another label',
         whole('split study, dataset 863, position, ecd_subset, dipole' + OTHER_STRATA),
         in_json(summary(SPLIT, '863', 'position'), ['ecd_subset', 'dipole', 'strata'],
                 renamed('within_montage', 'beside_montage'))),
    # The pooled network itself differs: every other method then has other strata, and the first is named.
    case('a pooled network without a stratum that the other methods have',
         whole('split study, dataset 863, position, dictionary' + OTHER_STRATA),
         in_json(summary(SPLIT, '863', 'position'), ['full', 'mlp_pooled', 'strata', 'snr_10_15'], DROP)),
    case('a pooled network with its strata in another order than the other methods on the dipole-fit subset',
         whole('split study, dataset 864, region, ecd_subset, dictionary' + OTHER_STRATA),
         in_json(summary(SPLIT, '864', 'region'), ['ecd_subset', 'mlp_pooled', 'strata'], in_the_opposite_order)),

    # ---- count control
    case('an error of the count control', 'count control, dataset 863, all_test_windows, count_matched, me',
         in_csv(f'{COUNT}/863/errors.csv', 17, 'count_matched_error_mm', plus(30.0))),
    case('the count-control contrast on all windows',
         'count control, dataset 862, all_test_windows, count_matched_minus_exact: recomputed',
         in_json(summary(COUNT, '862'), ['all_test_windows', 'count_matched_minus_exact', 'median_difference_mm'],
                 plus(0.3))),
    case('the count-control contrast on the common windows',
         'count control, dataset 862, same_test_windows, count_matched_minus_exact: recomputed',
         in_json(summary(COUNT, '862'), ['same_test_windows', 'count_matched_minus_exact', 'median_difference_mm'],
                 plus(0.3))),
    case('the regional contrast of the count control',
         'count control, dataset 864, same_test_windows, regional_minus_count_matched: recomputed',
         in_json(summary(COUNT, '864'), ['same_test_windows', 'regional_minus_count_matched', 'median_difference_mm'],
                 plus(0.01))),
    case('a median on the common windows', 'count control, dataset 863, same_test_windows, regional, median_mm',
         in_json(summary(COUNT, '863'), ['same_test_windows', 'regional', 'median_mm'], plus(0.01))),
    case('the network of the count control', 'count control, dataset 862, same_test_windows, regional, me',
         in_json(summary(COUNT, '862'), ['model_seed'], to(863))),
    case('a common window listed twice', 'The count control does not hold every common test window once',
         in_json(f'{SPLIT}/863/same_test_windows/exact_position_test_rows.json', [],
                 lambda rows: [rows[1], *rows[1:]])),
    case('a common window that the count control lacks',
         'The count control does not hold every common test window once',
         in_json(f'{SPLIT}/863/same_test_windows/exact_position_test_rows.json', [0], to(500))),
    case('another common window', 'The count control and the split study disagree on the common test windows',
         in_json(f'{SPLIT}/864/same_test_windows/exact_position_test_rows.json', [],
                 lambda rows: [next(window for window in range(72) if window not in rows), *rows[1:]])),
    case('a count-control headline without one entry',
         'count control headline, same_test_windows: the entries differ from the dataset records',
         in_json(f'{COUNT}/headline.json', ['same_test_windows', 'regional'], DROP)),
    case('a count-control headline without one panel',
         'count control headline: the panels differ from the dataset records',
         in_json(f'{COUNT}/headline.json', ['same_test_windows'], DROP)),
    case('a mean of the count-control headline',
         'count control headline, same_test_windows, regional, mean over the datasets: recomputed',
         in_json(f'{COUNT}/headline.json', ['same_test_windows', 'regional', 'mean'], plus(1.0))),
    case('a dataset value of the count-control headline',
         'count control headline, all_test_windows, count_matched_minus_exact, dataset 864: recomputed',
         in_json(f'{COUNT}/headline.json', ['all_test_windows', 'count_matched_minus_exact', 'dataset_seed_values', 2],
                 plus(0.01))),
    case('a range end of the count-control headline',
         'count control headline, all_test_windows, exact, largest dataset value: recomputed',
         in_json(f'{COUNT}/headline.json', ['all_test_windows', 'exact', 'range', 1], plus(0.01))),
    case('a count-control headline with two dataset values',
         'count control headline, all_test_windows, exact: 2 dataset values recorded, 3 expected',
         in_json(f'{COUNT}/headline.json', ['all_test_windows', 'exact', 'dataset_seed_values'],
                 lambda values: values[:2])),

    # ---- patch study
    case('a predicted patch centre', 'patch study, dataset 863, region, mlp_pooled, me',
         in_csv(f'{PATCH}/863/region/predictions.csv', 10, 'mlp_863_z_mm', plus(300.0))),
    case('a patch statistic of a lookup method',
         'patch study, dataset 862, position, ten_nearest_training_patches, p90_mm: recomputed',
         in_json(summary(PATCH, '862', 'position'), ['methods', 'ten_nearest_training_patches', 'p90_mm'], plus(0.01))),
    case('a patch stratum median',
         'patch study, dataset 862, position, stratum diameter_25_40, mlp_pooled, median_mm: recomputed',
         in_json(summary(PATCH, '862', 'position'), ['methods', 'mlp_pooled', 'strata', 'diameter_25_40', 'median_mm'],
                 plus(0.01))),
    # Every method of the patch study by stratum, each fitted network on its own included.
    *[case(f'a patch stratum statistic of {method}',
           f'patch study, dataset {seed}, {policy}, stratum {label}, {method}, {key}: recomputed',
           in_json(summary(PATCH, seed, policy), ['methods', method, 'strata', label, key], plus(0.01)))
      for method, seed, policy, label, key in PATCH_STRATUM_STATISTICS],
    case('a patch stratum statistic of a lookup method, by 5e-10 mm',
         'patch study, dataset 862, position, stratum snr_15_20, ten_nearest_training_patches, p90_mm: recomputed',
         in_json(summary(PATCH, '862', 'position'),
                 ['methods', 'ten_nearest_training_patches', 'strata', 'snr_15_20', 'p90_mm'], plus(5e-10))),
    case('the windows of a patch stratum of a lookup method',
         whole('patch study, dataset 862, position, stratum snr_10_15, nearest_training_patch: 22 windows, recorded '
               '23'),
         in_json(summary(PATCH, '862', 'position'), ['methods', 'nearest_training_patch', 'strata', 'snr_10_15',
                                                     'windows'], plus(1))),
    case('a patch stratum of one fitted network without a statistic',
         whole('patch study, dataset 863, region, stratum diameter_10_25, mlp_862: the record lacks p90_mm'),
         in_json(summary(PATCH, '863', 'region'), ['methods', 'mlp_862', 'strata', 'diameter_10_25', 'p90_mm'], DROP)),
    # Two methods differ in one stratum: the driver stops at the first of them in the order of the methods.
    case('a patch stratum statistic of two methods',
         'patch study, dataset 864, region, stratum snr_15_20, nearest_training_patch, mean_mm: recomputed',
         *[in_json(summary(PATCH, '864', 'region'), ['methods', method, 'strata', 'snr_15_20', 'mean_mm'], plus(0.01))
           for method in ('mlp_863', 'nearest_training_patch')]),
    case('a patch stratum with an unknown label', whole('Unknown stratum label: radius_10_15'),
         in_json(summary(PATCH, '864', 'position'), ['methods', 'mlp_pooled', 'strata'],
                 renamed('snr_10_15', 'radius_10_15'))),
    # A label that only one method has: the strata of that method are no longer those of the pooled network.
    case('a patch stratum of a lookup method with an unknown label',
         whole('patch study, dataset 863, region, training_mean' + OTHER_STRATA),
         in_json(summary(PATCH, '863', 'region'), ['methods', 'training_mean', 'strata'],
                 renamed('snr_10_15', 'depth_10_15'))),
    case('a missing fitted network of the patch study',
         'patch study, dataset 864, region: the fitted networks are not those of the model seeds',
         in_json(summary(PATCH, '864', 'region'), ['fits', 'mlp_862'], DROP)),
    case('the patch share below the nearest centre, by 5e-12',
         'Patch share below the nearest training centre differs from the record',
         in_json(summary(PATCH, '863', 'position'), ['mlp_error_below_nearest_training_centre'], plus(5e-12))),
    case('the support clearance of the study record',
         'The study record and the dataset report disagree on the support clearance',
         in_json(summary(PATCH, '863', 'region'), ['audit', 'minimum_support_clearance_mm'], plus(1e-5))),
    case('the support clearance of the dataset report',
         'The study record and the dataset report disagree on the support clearance',
         in_json(f'{PATCH}/864/region/dataset/validation_report.json',
                 ['source_split', 'serialized_used_position_nearest_neighbour_mm', 'val_to_train', 'all', 'min'],
                 plus(-0.5))),
    case('a patch headline without one method',
         'patch study headline, position: the methods differ from the dataset records',
         in_json(f'{PATCH}/headline.json', ['position', 'training_mean'], DROP)),
    case('a mean of the patch headline', 'patch study headline, region, mlp_pooled, mean over the datasets: recomputed',
         in_json(f'{PATCH}/headline.json', ['region', 'mlp_pooled', 'mean_median_mm'], plus(-10.0))),
    case('the mean overlap of the patch headline',
         'patch study headline, position: the overlap differs from the dataset records',
         in_json(f'{PATCH}/headline.json', ['position', 'median_best_test_to_train_overlap', 'mean'], plus(5e-12))),
    case('a dataset overlap of the patch headline',
         'patch study headline, position: the overlap differs from the dataset records',
         in_json(f'{PATCH}/headline.json', ['position', 'median_best_test_to_train_overlap', 'dataset_seed_values', 1],
                 plus(0.01))),
    case('a range end of the overlap of the patch headline',
         'patch study headline, position: the overlap differs from the dataset records',
         in_json(f'{PATCH}/headline.json', ['position', 'median_best_test_to_train_overlap', 'range', 0], plus(0.01))),
    case('a paired contrast of the patch study',
         'patch study, dataset 862, position, network minus nearest_training_patch: recomputed',
         in_json(summary(PATCH, '862', 'position'),
                 ['methods', 'paired_mlp_minus_lookup', 'nearest_training_patch', 'median_difference_mm'], plus(1e-9))),

    # ---- transfer to the template head
    case('an ineligible transfer window', 'An ineligible transfer window was evaluated',
         in_csv(f'{TRANSFER}/862/eligibility.csv', where(eligible='True'), 'eligible', to('False'))),
    case('the eligibility threshold', 'Eligibility does not follow the recorded clearance threshold',
         in_json(f'{TRANSFER}/862/eligibility.json', ['minimum_inner_skull_clearance_mm'], to(6.0))),
    case('a window below the threshold that is recorded as eligible',
         'Eligibility does not follow the recorded clearance threshold',
         in_csv(f'{TRANSFER}/863/eligibility.csv', where(eligible='False'), 'eligible', to('True'))),
    case('a transfer median of a stratum', 'transfer, dataset 863, stratum all, bem: recomputed',
         in_json(TRANSFER_SUMMARY, ['by_dataset_seed', '863', 'paired_pooled_model_seeds', 'all', 'bem_median_mm'],
                 plus(0.01))),
    case('a sphere median of a stratum', 'transfer, dataset 862, stratum population_0, package_sphere: recomputed',
         in_json(TRANSFER_SUMMARY,
                 ['by_dataset_seed', '862', 'paired_pooled_model_seeds', 'population_0', 'package_sphere_median_mm'],
                 plus(0.01))),
    case('a transfer difference of medians',
         'transfer, dataset 862, stratum all, difference of the medians: recomputed',
         in_json(TRANSFER_SUMMARY,
                 ['by_dataset_seed', '862', 'paired_pooled_model_seeds', 'all', 'median_difference_mm'], plus(1.0))),
    case('the windows of a transfer stratum',
         'transfer, dataset 863, stratum all: windows or positions differ from the record',
         in_json(TRANSFER_SUMMARY, ['by_dataset_seed', '863', 'paired_pooled_model_seeds', 'all', 'windows'], plus(1))),
    case('the positions of a transfer stratum at the rule of 30',
         'transfer, dataset 864, stratum population_1: windows or positions differ from the record',
         in_json(TRANSFER_SUMMARY,
                 ['by_dataset_seed', '864', 'paired_pooled_model_seeds', 'population_1', 'position_clusters'],
                 plus(1))),
    case('a transfer stratum with an unknown label', 'Unknown stratum label: snr_10_15',
         in_json(TRANSFER_SUMMARY, ['by_dataset_seed', '863', 'paired_pooled_model_seeds'],
                 renamed('population_1', 'snr_10_15'))),
    case('a transfer statistic of one method',
         'transfer, dataset 864, stratum radius_1_45_55, mlp_863_independent_sphere, mean_mm: recomputed',
         in_json(TRANSFER_SUMMARY,
                 ['by_dataset_seed', '864', 'strata', 'radius_1_45_55', 'mlp_863_independent_sphere', 'mean_mm'],
                 plus(0.01))),
    case('a transfer error of one window', 'transfer, dataset 862, stratum all, bem',
         in_csv(f'{TRANSFER}/862/paired_predictions.csv', smallest('mlp_863_bem_error_mm'), 'mlp_863_bem_error_mm',
                plus(300.0))),
    case('the dipole-fit subset of the transfer experiment',
         'transfer, dataset 862, dipole-fit subset, ecd_package_sphere_matched, median_mm: recomputed',
         in_json(TRANSFER_SUMMARY, ['by_dataset_seed', '862', 'ecd_subset', 'ecd_package_sphere_matched', 'median_mm'],
                 plus(1.0))),
    case('the matched difference of the dipole-fit subset',
         'transfer, dataset 862, dipole-fit subset, ecd_matched_bem_minus_sphere: recomputed',
         in_json(TRANSFER_SUMMARY,
                 ['by_dataset_seed', '862', 'ecd_subset', 'ecd_matched_bem_minus_sphere', 'median_difference_mm'],
                 plus(0.01))),
    case('a dipole fit of the transfer experiment',
         'transfer, dataset 862, dipole-fit subset, ecd_bem_fitted_sphere, me',
         in_csv(f'{TRANSFER}/862/ecd_subset.csv', 3, 'ecd_bem_fitted_sphere_error_mm', plus(300.0))),
    case('the pairing check: difference of the medians',
         'transfer, dataset 862, pairing check of network 862, difference of the medians: recomputed',
         in_json(f'{TRANSFER}/862/adapter_gates.json', ['check5_model', 'mlp_862', 'median_difference_mm'], plus(0.3))),
    case('the pairing check: all windows',
         'transfer, dataset 863, pairing check of network 863, all windows, median_mm: recomputed',
         in_json(f'{TRANSFER}/863/adapter_gates.json', ['check5_model', 'mlp_863', 'package_sphere_all', 'median_mm'],
                 plus(0.01))),
    case('the pairing check: the package sphere',
         'transfer, dataset 864, pairing check of network 862, package_sphere, mean_mm: recomputed',
         in_json(f'{TRANSFER}/864/adapter_gates.json',
                 ['check5_model', 'mlp_862', 'package_sphere_eligible', 'mean_mm'], plus(0.01))),
    case('the pairing check: the independent sphere',
         'transfer, dataset 862, pairing check of network 863, independent_sphere, p90_mm: recomputed',
         in_json(f'{TRANSFER}/862/adapter_gates.json',
                 ['check5_model', 'mlp_863', 'independent_sphere_eligible', 'p90_mm'], plus(0.01))),
    case('the pairing check: the shift of the predictions',
         'transfer, dataset 863, pairing check of network 862, prediction shift, p99: recomputed',
         in_json(f'{TRANSFER}/863/adapter_gates.json',
                 ['check5_model', 'mlp_862', 'per_window_prediction_shift_mm', 'p99'], plus(0.01))),
    case('the pairing check: the largest shift',
         'transfer, dataset 864, pairing check of network 863, prediction shift, maximum: recomputed',
         in_json(f'{TRANSFER}/864/adapter_gates.json',
                 ['check5_model', 'mlp_863', 'per_window_prediction_shift_mm', 'maximum'], plus(0.01))),
    case('the mean of the transfer headline',
         'transfer headline, package_sphere_median_mm, mean over the datasets: recomputed',
         in_json(TRANSFER_SUMMARY, ['paired_headline', 'package_sphere_median_mm', 'mean_of_dataset_seed_medians'],
                 plus(0.5))),
    case('a range end of the transfer headline',
         'transfer headline, median_difference_mm, smallest dataset value: recomputed',
         in_json(TRANSFER_SUMMARY, ['paired_headline', 'median_difference_mm', 'range', 0], plus(-0.01))),
    case('the upper range end of the transfer headline', 'transfer headline, bem_median_mm, largest dataset value',
         in_json(TRANSFER_SUMMARY, ['paired_headline', 'bem_median_mm', 'range', 1], plus(0.01))),
    case('the eligibility record of a population',
         'transfer, dataset 863: the eligibility record of population_1 differs from its windows',
         in_json(f'{TRANSFER}/863/eligibility.json', ['strata', 'population_1', 'eligible_windows'], plus(-1))),
    case('the eligibility record of all windows',
         'transfer, dataset 862: the eligibility record of all differs from its windows',
         in_json(f'{TRANSFER}/862/eligibility.json', ['strata', 'all', 'windows'], plus(1))),
    case('the eligibility record of a radius band',
         'transfer, dataset 864: the eligibility record of radius_0_70.5_75 differs from its windows',
         in_json(f'{TRANSFER}/864/eligibility.json', ['strata', 'radius_0_70.5_75', 'eligible_positions'], plus(1))),
    case('the eligibility record of a half of the head',
         'transfer, dataset 863: the eligibility record of negative_anterior differs from its windows',
         in_json(f'{TRANSFER}/863/eligibility.json', ['strata', 'negative_anterior', 'positions'], plus(-1))),
    case('an eligibility record with an unknown stratum', 'Unknown stratum label: left',
         in_json(f'{TRANSFER}/862/eligibility.json', ['strata'], renamed('right', 'left'))),
    case('a transfer stratum that drops a window', 'Transfer strata, dataset 863: the snr strata hold 65 of 66 windows',
         in_csv(f'{TRANSFER}/863/paired_predictions.csv', 10, 'snr_db_referenced', to('9.0'))),
    case('a census window outside every radius band',
         'Eligibility census, dataset 864: the radius strata do not hold every window once',
         in_csv(f'{TRANSFER}/864/eligibility.csv', where(eligible='False'), 'x_mm', outside_every_band)),
    case('a census window of an unknown population',
         'Eligibility census, dataset 862: the population strata do not hold every window once',
         in_csv(f'{TRANSFER}/862/eligibility.csv', where(eligible='False'), 'population', to('2'))),

    # ---- adaptation of the head model
    case('the number of adaptation windows', 'Adaptation prediction file has an unexpected number of windows',
         in_json(ADAPTATION_SUMMARY, ['by_dataset_seed', '863', 'bem_windows'], plus(1))),
    case('an adaptation median', 'adaptation, dataset 864, finetuned_1000, median_mm: recomputed',
         in_json(ADAPTATION_SUMMARY, ['by_dataset_seed', '864', 'arms', 'finetuned_1000', 'median_mm'], plus(0.01))),
    case('an adaptation difference from the default sphere',
         'adaptation, dataset 862, three_shell, difference from the default sphere: recomputed',
         in_json(ADAPTATION_SUMMARY,
                 ['by_dataset_seed', '862', 'arms', 'three_shell', 'median_difference_from_default_mm'], plus(1.0))),
    case('an adaptation median of a stratum', 'adaptation, dataset 862, stratum population_1, default, median_mm',
         in_json(ADAPTATION_SUMMARY, ['by_dataset_seed', '862', 'strata', 'population_1', 'default', 'median_mm'],
                 plus(1.0))),
    case('an adaptation difference of a stratum',
         'adaptation, dataset 863, stratum radius_0_75_79.501, scratch_all, difference from the default sphere',
         in_json(ADAPTATION_SUMMARY, ['by_dataset_seed', '863', 'strata', 'radius_0_75_79.501', 'scratch_all',
                                      'median_difference_from_default_mm'], plus(0.01))),
    case('an adaptation stratum with an unknown label', 'Unknown stratum label: depth_1',
         in_json(ADAPTATION_SUMMARY, ['by_dataset_seed', '864', 'strata'], renamed('population_1', 'depth_1'))),
    case('a geometry of the summary that differs from the geometry record',
         'adaptation: the geometry of the summary differs from the geometry record',
         in_json(ADAPTATION_SUMMARY, ['geometry', 'variants', 'default', 'radii_m', 0], plus(0.001))),
    case('an adaptation error of one window', 'adaptation, dataset 863, size_matched, me',
         in_csv(f'{ADAPTATION}/863/adaptation_predictions.csv', 4, 'size_matched_mlp_863_error_mm', plus(300.0))),
    case('a statistic of a sphere variant on its own sphere',
         'adaptation, dataset 863, size_matched on its own sphere, p90_mm: recomputed',
         in_json(ADAPTATION_SUMMARY,
                 ['by_dataset_seed', '863', 'own_sphere', 'size_matched', 'eligible_windows', 'p90_mm'], plus(0.01))),
    case('a predicted position on its own sphere', 'adaptation, dataset 862, three_shell on its own sphere, me',
         in_npz(f'{ADAPTATION}/862/predictions.npz', 'three_shell_own_sphere_mlp_863', (4, 0), 300.0)),
    case('the own-sphere headline', 'adaptation headline, size_matched on its own sphere, mean over the datasets',
         in_json(ADAPTATION_SUMMARY, ['own_sphere_eligible_headline', 'size_matched', 'mean_of_dataset_seed_medians'],
                 plus(0.01))),
    case('a range end of the own-sphere headline',
         'adaptation headline, default on its own sphere, smallest dataset value',
         in_json(ADAPTATION_SUMMARY, ['own_sphere_eligible_headline', 'default', 'range', 0], plus(-1.0))),
    case('an own-sphere headline without one sphere variant',
         'adaptation headline: the sphere variants differ from the dataset records',
         in_json(ADAPTATION_SUMMARY, ['own_sphere_eligible_headline', 'size_matched'], DROP)),
    case('an adaptation headline without one arm', 'adaptation headline: the arms differ from the dataset records',
         in_json(ADAPTATION_SUMMARY, ['bem_windows_headline', 'scratch_all'], DROP)),
    case('the adaptation headline', 'adaptation headline, finetuned_all, mean over the datasets: recomputed',
         in_json(ADAPTATION_SUMMARY, ['bem_windows_headline', 'finetuned_all', 'mean_of_dataset_seed_medians'],
                 plus(0.01))),
    case('a dataset median of the adaptation headline', 'adaptation headline, default, dataset 863: recomputed',
         in_json(ADAPTATION_SUMMARY, ['bem_windows_headline', 'default', 'dataset_seed_medians_mm', 1], plus(0.01))),
    case('a range end of the adaptation headline', 'adaptation headline, scratch_100, largest dataset value',
         in_json(ADAPTATION_SUMMARY, ['bem_windows_headline', 'scratch_100', 'range', 1], plus(0.01))),
    case('an adaptation headline without a difference',
         'adaptation headline, three_shell: a difference from the default sphere is missing',
         in_json(ADAPTATION_SUMMARY, ['bem_windows_headline', 'three_shell', 'mean_difference_from_default_mm'], DROP)),
    case('the mean difference of the adaptation headline',
         'adaptation headline, scratch_100, mean difference from the default sphere: recomputed',
         in_json(ADAPTATION_SUMMARY, ['bem_windows_headline', 'scratch_100', 'mean_difference_from_default_mm'],
                 plus(0.01))),
    case('a dipole fit of the adaptation experiment', 'adaptation, dipole fits, dipole_size_matched_sphere_on_bem, me',
         in_csv(f'{ADAPTATION}/862/dipole_subset.csv', 3, 'dipole_size_matched_sphere_on_bem_error_mm', plus(300.0))),
    case('a network statistic on the dipole-fit subset',
         'adaptation, dipole fits, default_mlp_863, mean_mm: recomputed',
         in_json(ADAPTATION_SUMMARY, ['dipole_subset', 'default_mlp_863', 'mean_mm'], plus(0.01))),
    case('a dipole-fit difference from the matched model',
         'adaptation, dipole fits, dipole_default_sphere_on_bem, difference from the matched model: recomputed',
         in_json(ADAPTATION_SUMMARY,
                 ['dipole_subset', 'dipole_default_sphere_on_bem', 'median_difference_from_bem_matched_mm'],
                 plus(1.0))),

    # ---- adaptation: each fine-tuned network and each network trained on the boundary-element windows alone
    *single_network_cases(),
    case('a statistic of a single network, by 5e-10 mm',
         'adaptation, dataset 863, scratch_1000, network 863, p90_mm: recomputed',
         in_json(ADAPTATION_SUMMARY,
                 ['by_dataset_seed', '863', 'finetune', network('scratch', '1000', '863'), 'p90_mm'], plus(5e-10))),
    # The two networks of an arm exchange their errors in one window. The statistics of the arm, which pool both
    # networks, stay what they are, in every stratum: only the statistics of each network can notice.
    case('the errors of one window exchanged between the two networks of an arm',
         'adaptation, dataset 864, finetuned_all, network 862, me',
         exchanged_cells(f'{ADAPTATION}/864/adaptation_predictions.csv', 0, 'finetuned_all_mlp_862_error_mm',
                         'finetuned_all_mlp_863_error_mm')),
    # The rows are the networks of the arms that were fine-tuned or trained alone: each one once, and no other.
    # The rows that remain agree with the prediction file, so that only the list of the networks is wrong.
    case('a single network that the summary lacks',
         whole('adaptation, dataset 862: the single networks are not those of the arms'),
         in_json(ADAPTATION_SUMMARY, ['by_dataset_seed', '862', 'finetune', network('scratch', '1000', '863')], DROP)),
    case('the last dataset without its last single network',
         whole('adaptation, dataset 864: the single networks are not those of the arms'),
         in_json(ADAPTATION_SUMMARY, ['by_dataset_seed', '864', 'finetune', network('scratch', 'all', '863')], DROP)),
    case('a single network that the summary holds twice',
         whole('adaptation, dataset 863: the single networks are not those of the arms'),
         in_json(ADAPTATION_SUMMARY, ['by_dataset_seed', '863', 'finetune'],
                 lambda rows: [*rows, rows[network('finetuned', '100', '862')]])),
    case('a dataset without its single networks',
         whole('adaptation, dataset 863: the single networks are not those of the arms'),
         in_json(ADAPTATION_SUMMARY, ['by_dataset_seed', '863', 'finetune'], to([]))),
    # One network stands twice and the other network of its arm is missing: as many rows as networks.
    case('a single network in place of the other network of its arm',
         whole('adaptation, dataset 862: the single networks are not those of the arms'),
         in_json(ADAPTATION_SUMMARY, ['by_dataset_seed', '862', 'finetune'],
                 lambda rows: [rows[network('finetuned', '1000', '862')]
                               if number == network('finetuned', '1000', '863') else row
                               for number, row in enumerate(rows)])),
    # One network stands twice and the network of another size is missing: as many rows as networks, with every
    # kind and every model seed as often as before, so that only the sizes can notice.
    case('a single network in place of the network of another size',
         whole('adaptation, dataset 864: the single networks are not those of the arms'),
         in_json(ADAPTATION_SUMMARY, ['by_dataset_seed', '864', 'finetune'],
                 lambda rows: [rows[network('scratch', '1000', '863')]
                               if number == network('scratch', '100', '863') else row
                               for number, row in enumerate(rows)])),
    # The same for the two networks that stand first and last when the networks are put in order.
    case('the first single network in place of the other network of its arm',
         whole('adaptation, dataset 863: the single networks are not those of the arms'),
         in_json(ADAPTATION_SUMMARY, ['by_dataset_seed', '863', 'finetune'],
                 lambda rows: [rows[network('finetuned', '100', '863')]
                               if number == network('finetuned', '100', '862') else row
                               for number, row in enumerate(rows)])),
    case('the last single network in place of the other network of its arm',
         whole('adaptation, dataset 862: the single networks are not those of the arms'),
         in_json(ADAPTATION_SUMMARY, ['by_dataset_seed', '862', 'finetune'],
                 lambda rows: [rows[network('scratch', 'all', '862')]
                               if number == network('scratch', 'all', '863') else row
                               for number, row in enumerate(rows)])),
    case('a single network of a model seed that no arm has',
         whole('adaptation, dataset 864: the single networks are not those of the arms'),
         in_json(ADAPTATION_SUMMARY,
                 ['by_dataset_seed', '864', 'finetune', network('finetuned', 'all', '863'), 'model_seed'], to(864))),
    case('a single network of an arm that is no arm of the experiment',
         whole('adaptation, dataset 862: the single networks are not those of the arms'),
         in_json(ADAPTATION_SUMMARY, ['by_dataset_seed', '862', 'finetune', network('scratch', '100', '862'), 'arm'],
                 to('frozen'))),
    # The model seed of a network is the one of its column in the prediction file, digit by digit.
    case('a single network whose model seed is written with a zero before it',
         whole('adaptation, dataset 863: the single networks are not those of the arms'),
         in_json(ADAPTATION_SUMMARY,
                 ['by_dataset_seed', '863', 'finetune', network('scratch', '100', '863'), 'model_seed'], to('0863'))),
    # A thirteenth network with errors of its own in the prediction file and with their statistics in its row.
    case('a single network of a size that no arm has',
         whole('adaptation, dataset 863: the single networks are not those of the arms'),
         *network_of_another_size('863', '300', like=('finetuned', '100', '862'))),

    # ---- benchmark and settings
    case('a benchmark summary that disagrees with its runs',
         'Benchmark case point-1: median of seconds differs from its runs',
         in_json(BENCHMARK, ['cases', 0, 'summary', 'seconds', 'median'], times(1 + 5e-12))),
    case('stage times of a benchmark case that are not those of its median run',
         'Benchmark case point-1: the stage times are not those of the median run',
         in_json(BENCHMARK, ['cases', 0, 'summary', 'median_run_setup_seconds'], plus(0.5))),
    case('a stage time of the median run of a benchmark case',
         'Benchmark case patch-1: the stage times are not those of the median run',
         in_json(BENCHMARK, ['cases', 1, 'summary', 'median_run_phase_seconds', 'basis_forward'], plus(0.5))),
    case('an output size of a benchmark case that differs between the runs',
         'Benchmark case point-1: the output size differs between the runs',
         in_json(BENCHMARK, ['cases', 0, 'repetitions', 1, 'output_bytes'], plus(1))),
    case('the largest memory of a benchmark case',
         'Benchmark case patch-1: maximum of peak_process_tree_rss_bytes differs from its runs',
         in_json(BENCHMARK, ['cases', 1, 'summary', 'peak_process_tree_rss_bytes', 'maximum'], plus(1024.0))),
    case('the smallest rate of a benchmark case',
         'Benchmark case npz-1: minimum of windows_per_second differs from its runs',
         in_json(BENCHMARK, ['cases', 3, 'summary', 'windows_per_second', 'minimum'], plus(-0.001))),
    case('datasets that disagree on a setting', 'The datasets disagree on head.n_legendre',
         in_json(f'{SPLIT}/863/region/dataset/metadata.json', ['config', 'head', 'n_legendre'], plus(20))),
    case('a patch dataset that disagrees on a setting', 'The datasets disagree on signal.window_samples',
         in_json(f'{PATCH}/864/position/dataset/metadata.json', ['config', 'signal', 'window_samples'], plus(50))),
    case('region datasets that disagree on the number of sectors', 'The datasets disagree on split.region_count',
         in_json(f'{PATCH}/863/region/dataset/metadata.json', ['config', 'split', 'region_count'], plus(4))),
    case('point datasets that disagree on the number of positions',
         'The datasets disagree on sources.n_cortical_positions',
         in_json(f'{SPLIT}/862/position/dataset/metadata.json', ['config', 'sources', 'n_cortical_positions'],
                 plus(1))),
    case('patch datasets that disagree on the diameters', 'The datasets disagree on patch.diameter_range_mm',
         in_json(f'{PATCH}/862/region/dataset/metadata.json', ['config', 'patch', 'diameter_range_mm', 1], plus(5.0))),
    case('a dataset whose recorded seed differs from its folder',
         'patch-split-study-v11/863/region: the dataset records seed 864',
         in_json(f'{PATCH}/863/region/dataset/metadata.json', ['config', 'master_seed'], to(864))),
    case('datasets that disagree on the clearance threshold',
         'The datasets disagree on the clearance threshold of the transfer experiment',
         in_json(f'{TRANSFER}/863/eligibility.json', ['minimum_inner_skull_clearance_mm'], to(4.95))),

    # ---- validation rerun
    case('an array digest that differs between the releases',
         'validation rerun: the two releases generated different arrays (29 of 30 equal)',
         in_json(f'{RERUN}/generated_arrays_release-1.1.0.json',
                 ['datasets', 'patch_comparison_patch', 'train.h5', 'eeg', 'sha256'], to('f' * 64))),
    case('an array type that differs between the releases',
         'validation rerun: the two releases generated different arrays (29 of 30 equal)',
         in_json(f'{RERUN}/generated_arrays_release-1.0.0.json',
                 ['datasets', 'inverse_headline_35dB', 'test.h5', 'eeg', 'dtype'], to('float32'))),
    case('an array that only the later release has',
         'validation rerun: the two releases generated different arrays (29 of 29 equal)',
         in_json(f'{RERUN}/generated_arrays_release-1.0.0.json',
                 ['datasets', 'inverse_sweep_5dB', 'source_pool.h5', 'positions_xyz'], DROP)),
    case('an array that only the earlier release has',
         'validation rerun: the two releases generated different arrays (29 of 29 equal)',
         in_json(f'{RERUN}/generated_arrays_release-1.1.0.json',
                 ['datasets', 'patch_comparison_point', 'test.h5', 'ground_truth/position_pool_idx'], DROP)),
    case('a dataset that one release lacks', 'The two releases generated different sets of datasets',
         in_json(f'{RERUN}/generated_arrays_release-1.0.0.json', ['datasets', 'inverse_sweep_20dB'], DROP)),
    case('a position error of the rerun, by 0.5 micrometre',
         'validation rerun, windows_release_1_1_0, position error p90: recomputed',
         in_json(f'{WINDOWS}/{INVERSE}', ['position_error_mm', 'p90'], plus(5e-4))),
    case('the median position error of the archived platform',
         'validation rerun, linux_release_1_0_0, position error median: recomputed',
         in_json(f'{LINUX}/ecd_69ch_35db_summary.json', ['position_error_mm', 'median'], plus(5e-4))),
    case('the largest position error of the earlier release',
         'validation rerun, windows_release_1_0_0, position error maximum: recomputed',
         in_json(f'{WINDOWS_OLD}/{INVERSE}', ['position_error_mm', 'maximum'], plus(5e-4))),
    case('the median orientation error of the rerun',
         'validation rerun, linux_release_1_0_0, median orientation error: recomputed',
         in_json(f'{LINUX}/ecd_69ch_35db_summary.json', ['orientation_error_deg', 'median'], plus(5e-4))),
    case('the median moment error of the rerun',
         'validation rerun, linux_release_1_0_0, median moment error: recomputed',
         in_json(f'{LINUX}/ecd_69ch_35db_summary.json', ['moment_amplitude', 'relative_error', 'median'], plus(5e-4))),
    case('the median goodness of fit of the rerun',
         'validation rerun, linux_release_1_0_0, median goodness of fit: recomputed',
         in_json(f'{LINUX}/ecd_69ch_35db_summary.json', ['goodness_of_fit_percent', 'median'], plus(-5e-4))),
    case('a per-dipole error of the archived platform',
         'validation rerun, linux_release_1_0_0: the fitted positions do not give the recorded position errors',
         in_csv(f'{LINUX}/ecd_69ch_35db_per_sample.csv', 0, 'err_mm', rounded_plus(0.0002, 4))),
    case('a fitted position of the archived platform',
         'validation rerun, linux_release_1_0_0: the fitted positions do not give the recorded position errors',
         in_csv(f'{LINUX}/ecd_69ch_35db_per_sample.csv', 9, 'estimated_x_m', rounded_plus(0.005, 9))),
    case('a release summary that differs on one machine',
         'validation rerun: the two releases differ on one machine: patch_comparison_equal',
         in_json(f'{WINDOWS_OLD}/patch/patch_forward_mne_summary.json', ['distributed_patch', 'median'], plus(1e-3))),
    case('a sweep median that differs between the releases',
         'validation rerun: the two releases differ on one machine: dipole_recovery_and_sweep_equal',
         in_json(f'{WINDOWS_OLD}/{INVERSE}', ['snr_degradation_sweep', 0, 'position_error_median_mm'], plus(1.0))),
    case('a position error that differs between the releases within the rounding of the file',
         'validation rerun: the two releases differ on one machine: dipole_recovery_and_sweep_equal',
         in_json(f'{WINDOWS_OLD}/{INVERSE}', ['position_error_mm', 'median'], plus(2e-6))),
    case('a per-dipole file that differs between the releases',
         'validation rerun: the two releases differ on one machine: per_dipole_file_identical',
         in_csv(f'{WINDOWS_OLD}/{PER_SAMPLE}', 0, 'fit_sample', to('101'))),
    case('transfer coefficients that differ between the releases',
         'validation rerun: the two releases differ on one machine: transfer_coefficients_equal',
         in_json(f'{WINDOWS_OLD}/coefficients/transfer_coefficients.json',
                 ['cases', 'homogeneous', 'max_relative_difference'], times(2.0))),
    case('a series error that differs between the releases',
         'validation rerun: the two releases differ on one machine: legendre_convergence_equal',
         in_json(f'{WINDOWS_OLD}/legendre/legendre_convergence.json',
                 ['relative_error', 'superficial_radial_79_5mm', '100'], times(3.0))),
    case('platforms that fitted different sources', 'validation rerun: the two platforms fitted different sources',
         in_csv(f'{LINUX}/ecd_69ch_35db_per_sample.csv', 4, 'source_radius_mm', rounded_plus(0.1, 3))),
    case('a true position of the archived platform',
         'validation rerun, linux_release_1_0_0: the fitted positions do not give the recorded position errors',
         in_csv(f'{LINUX}/ecd_69ch_35db_per_sample.csv', 4, 'true_x_m', rounded_plus(0.001, 9))),
    case('platforms that fitted sources at another SNR', 'validation rerun: the two platforms fitted different sources',
         in_csv(f'{LINUX}/ecd_69ch_35db_per_sample.csv', 11, 'snr_db', rounded_plus(0.1, 3))),
    case('platforms that fitted different numbers of sources',
         'validation rerun: the two platforms fitted different sources', *without_the_last_dipole(LINUX)),
    case('transfer coefficients that differ between the platforms',
         'validation rerun: the transfer coefficients differ between the platforms',
         in_json(f'{LINUX}/transfer_coefficients.json', ['cases', 'homogeneous', 'max_relative_difference'],
                 times(2.0))),
    case('sweep levels that differ between the platforms',
         'validation rerun: the sweep levels of the two platforms differ',
         in_json(f'{LINUX}/ecd_69ch_35db_summary.json', ['snr_degradation_sweep', 1, 'snr_db'], plus(1.0))),
    case('a platform without one sweep level', 'validation rerun: the sweep levels of the two platforms differ',
         in_json(f'{LINUX}/ecd_69ch_35db_summary.json', ['snr_degradation_sweep', 1], DROP)),

    # ---- sphere-model fit of MNE-Python
    case('a smoke sphere-fit record', 'The sphere-fit record is a smoke run or did not reproduce the fits',
         in_json(SPHERE_SUMMARY, ['smoke'], to(True))),
    case('a parameter fit that MNE-Python did not reproduce',
         'The sphere-fit record is a smoke run or did not reproduce the fits',
         in_json(SPHERE_SUMMARY, ['instrumented_call_reproduces_installed_mne_python'], to(False))),
    case('dipole fits that MNE-Python did not reproduce',
         'The sphere-fit record is a smoke run or did not reproduce the fits',
         in_json(SPHERE_SUMMARY, ['instrumented_dipole_fit_reproduces_installed_mne_python'], to(False))),
    case('a sphere-fit record with one arm', "sphere fit: unexpected arms ['mne_limit']",
         in_json(SPHERE_SUMMARY, ['arms', 'raised_limit'], DROP)),
    case('a sphere-fit statistic, by 5e-12 mm', 'sphere fit, raised_limit, position_error_mm, median: recomputed',
         in_json(SPHERE_SUMMARY, ['arms', 'raised_limit', 'dipole_recovery', 'position_error_mm', 'median'],
                 plus(5e-12))),
    case('the goodness of fit of an arm', 'sphere fit, mne_limit, goodness_of_fit_percent, p90: recomputed',
         in_json(SPHERE_SUMMARY, ['arms', 'mne_limit', 'dipole_recovery', 'goodness_of_fit_percent', 'p90'],
                 plus(1e-6))),
    case('the number of dipoles of an arm', 'sphere fit, mne_limit, orientation_error_deg: 12 values, recorded 13',
         in_json(SPHERE_SUMMARY, ['arms', 'mne_limit', 'dipole_recovery', 'orientation_error_deg', 'n'], plus(1))),
    case('the largest field difference of the point sources',
         'sphere fit, raised_limit, point field difference, maximum: recomputed',
         in_json(SPHERE_SUMMARY, ['arms', 'raised_limit', 'patch_comparison', 'point', 'maximum'], plus(5e-12))),
    case('a field difference of a patch', 'sphere fit, mne_limit, patch field difference, median: recomputed',
         in_csv(f'{SPHERE}/field_differences.csv', middle('relative_field_difference_mne_limit', source='patch'),
                'relative_field_difference_mne_limit', plus(5e-12))),
    case('a sweep statistic of an arm', 'sphere fit, raised_limit, sweep at 20 dB, p90: recomputed',
         in_json(SPHERE_SUMMARY, ['arms', 'raised_limit', 'sweep', 1, 'position_error_mm', 'p90'], plus(5e-12))),
    case('a sweep error of one dipole', 'sphere fit, raised_limit, sweep at 5 dB',
         in_csv(f'{SPHERE}/snr_sweep.csv', 2, 'position_error_mm_raised_limit', plus(500.0))),
    case('an inner value of perturbation.csv', 'with mne_limit, position_error_median_mm: recomputed',
         in_csv(f'{SPHERE}/perturbation.csv',
                middle('position_error_median_mm', perturbed_fit='sphere_fit', sphere_fit='mne_limit'),
                'position_error_median_mm', plus(-5e-12))),
    case('the 90th percentile of a trial',
         'sphere fit, dipole_fit trial 2 with raised_limit, position_error_p90_mm: recomputed',
         in_csv(f'{SPHERE}/perturbation.csv', trial('dipole_fit', 'raised_limit', 2), 'position_error_p90_mm',
                plus(0.004))),
    case('the largest error of a trial',
         'sphere fit, sphere_fit trial 3 with raised_limit, position_error_maximum_mm: recomputed',
         in_csv(f'{SPHERE}/perturbation.csv', trial('sphere_fit', 'raised_limit', 3), 'position_error_maximum_mm',
                plus(0.004))),
    case('the median change of the fitted positions of a trial',
         'sphere fit, dipole_fit trial 1 with mne_limit, change of the fitted positions, median: recomputed',
         in_csv(f'{SPHERE}/perturbation.csv', trial('dipole_fit', 'mne_limit', 1), 'fitted_position_change_median_mm',
                plus(5e-12))),
    case('the largest change of the fitted positions of a trial',
         'sphere fit, sphere_fit trial 3 with raised_limit, change of the fitted positions, maximum: recomputed',
         in_csv(f'{SPHERE}/perturbation.csv', trial('sphere_fit', 'raised_limit', 3),
                'fitted_position_change_maximum_mm', plus(0.004))),
    case('one more position that moved in a trial',
         'sphere fit, sphere_fit trial 2 with mne_limit, change of the fitted positions: the share of positions',
         in_csv(f'{SPHERE}/perturbation.csv', trial('sphere_fit', 'mne_limit', 2),
                'fitted_positions_changed_by_more_than_0p1_mm', count_plus(1))),
    case('one position fewer that moved in a trial', 'change of the fitted positions: the share of positions',
         in_csv(f'{SPHERE}/perturbation.csv', largest('fitted_positions_changed_by_more_than_0p1_mm'),
                'fitted_positions_changed_by_more_than_0p1_mm', count_plus(-1))),
    case('the field difference of a trial',
         'sphere fit, sphere_fit trial 1 with mne_limit, patch_median_relative_field_difference: recomputed',
         in_csv(f'{SPHERE}/perturbation.csv', trial('sphere_fit', 'mne_limit', 1),
                'patch_median_relative_field_difference', plus(5e-12))),
    case('a trial of the parameter fit without its field comparison',
         'sphere fit, sphere_fit trial 2 with raised_limit: only the trials of the parameter fit repeat the field',
         in_csv(f'{SPHERE}/perturbation.csv', trial('sphere_fit', 'raised_limit', 2),
                'point_median_relative_field_difference', to(''))),
    case('a trial of the dipole fits with a field comparison',
         'sphere fit, dipole_fit trial 1 with mne_limit: only the trials of the parameter fit repeat the field',
         in_csv(f'{SPHERE}/perturbation.csv', trial('dipole_fit', 'mne_limit', 1),
                'patch_median_relative_field_difference', to('0.001'))),
    case('a value of the per-dipole file of the trials',
         'sphere fit, sphere_fit trial 1 with mne_limit, position_error_',
         in_csv(f'{SPHERE}/perturbation_dipoles.csv',
                where(perturbed_fit='sphere_fit', sphere_fit='mne_limit', objective_scale_minus_one_in_eps='1',
                      sample_index='4'), 'position_error_mm', plus(50.0))),
    case('a fitted position of the per-dipole file of the trials',
         'sphere fit, dipole_fit trial 3 with raised_limit, change of the fitted positions',
         in_csv(f'{SPHERE}/perturbation_dipoles.csv',
                where(perturbed_fit='dipole_fit', sphere_fit='raised_limit', objective_scale_minus_one_in_eps='3',
                      sample_index='7'), 'fitted_y_mm', plus(50.0))),
    case('the unchanged run of the per-dipole file',
         'sphere fit, raised_limit: the two per-dipole files disagree on the unchanged run',
         in_csv(f'{SPHERE}/perturbation_dipoles.csv', where(perturbed_fit='none', sphere_fit='raised_limit',
                                                            sample_index='4'), 'position_error_mm', plus(1e-9))),
    case('a run that holds a dipole twice', 'sphere fit: a run does not hold every dipole once, in order',
         in_csv(f'{SPHERE}/perturbation_dipoles.csv',
                where(perturbed_fit='dipole_fit', sphere_fit='mne_limit', objective_scale_minus_one_in_eps='2',
                      sample_index='3'), 'sample_index', to('5'))),
    case('a run that is not in the list of trials',
         'sphere fit: the per-dipole file and the list of trials hold different runs',
         in_csv(f'{SPHERE}/perturbation_dipoles.csv',
                where(perturbed_fit='dipole_fit', sphere_fit='mne_limit', objective_scale_minus_one_in_eps='2',
                      sample_index='11'), 'objective_scale_minus_one_in_eps', to('9'))),
    case('a value of the per-source file of the trials',
         'sphere fit, sphere_fit trial 2 with raised_limit, point_median_relative_field_difference: recomputed',
         in_csv(f'{SPHERE}/perturbation_fields.csv',
                middle('relative_field_difference', sphere_fit='raised_limit', objective_scale_minus_one_in_eps='2',
                       source='point'), 'relative_field_difference', plus(5e-12))),
    case('a range over the trials', 'sphere fit: the ranges over the trials differ from the record',
         in_json(SPHERE_SUMMARY, ['perturbation', 'ranges', 'sphere_fit', 'raised_limit', 'position_error_p90_mm', 1],
                 plus(0.01))),
    case('the evaluations of a trial', 'sphere fit: the ranges over the trials differ from the record',
         in_csv(f'{SPHERE}/perturbation.csv', trial('sphere_fit', 'mne_limit', 1), 'parameter_fit_evaluations',
                to('5000'))),
    case('the end of the parameter fit of a trial', 'sphere fit: the ranges over the trials differ from the record',
         in_csv(f'{SPHERE}/perturbation.csv', trial('sphere_fit', 'mne_limit', 3), 'parameter_fit_stopped_by_itself',
                to('1'))),
    case('a kernel value, by 5e-12 mm', 'sphere fit, kernel Sandybridge, mne_limit, position error, median: recomputed',
         in_json(SPHERE_SUMMARY, ['library_kernels', 'arms', 'Sandybridge', 'arms', 'mne_limit', 'dipole_recovery',
                                  'position_error_mm', 'median'], plus(5e-12))),
    case('a fitted position of a kernel', 'sphere fit, kernel Nehalem, mne_limit, change of the fitted positions',
         in_csv(f'{SPHERE}/kernels_dipole_recovery_35db.csv', where(kernel='Nehalem', sphere_fit='mne_limit',
                                                                    sample_index='0'), 'fitted_z_mm', plus(50.0))),
    case('the change of the fitted positions of a kernel',
         'sphere fit, kernel Nehalem, raised_limit, change of the fitted positions, p90: recomputed',
         in_json(SPHERE_SUMMARY, ['library_kernels', 'arms', 'Nehalem', 'fitted_position_change_from_the_parent_mm',
                                  'raised_limit', 'p90'], plus(5e-12))),
    case('the share of the positions that moved under a kernel',
         'sphere fit, kernel Sandybridge, raised_limit, change of the fitted positions: the share of positions',
         in_json(SPHERE_SUMMARY, ['library_kernels', 'arms', 'Sandybridge', 'fitted_position_change_from_the_parent_mm',
                                  'raised_limit', 'share_above_0p1_mm'], plus(1 / 12))),
    case('a field difference of a kernel',
         'sphere fit, kernel Sandybridge, raised_limit, patch field difference, median: recomputed',
         in_csv(f'{SPHERE}/kernels_field_differences.csv',
                middle('relative_field_difference', kernel='Sandybridge', sphere_fit='raised_limit', source='patch'),
                'relative_field_difference', plus(5e-12))),
    case('a kernel that the record does not hold',
         'sphere fit: the per-dipole file of the kernels and the record hold different kernels',
         in_csv(f'{SPHERE}/kernels_dipole_recovery_35db.csv', where(kernel='Nehalem', sphere_fit='raised_limit',
                                                                    sample_index='11'), 'kernel', to('Zen'))),
    case('a missing spheres part', 'sphere fit: the record lacks the spheres of the adaptation experiment',
         in_json(SPHERE_SUMMARY, ['spheres'], to(None))),
    case('spheres of another geometry record',
         'sphere fit: the spheres are not those of the archived adaptation experiment',
         in_json(SPHERE_SUMMARY, ['spheres', 'geometry_sha256'], to('0' * 64))),
    case('a geometry record that changed after the sphere fits',
         'sphere fit: the spheres are not those of the archived adaptation experiment',
         in_json(f'{ADAPTATION}/geometry.json', ['scale'], plus(0.1)),
         in_json(ADAPTATION_SUMMARY, ['geometry', 'scale'], plus(0.1))),
    case('a geometry record that differs from the summary of its experiment',
         'adaptation: the geometry of the summary differs from the geometry record',
         in_json(f'{ADAPTATION}/geometry.json', ['scale'], plus(0.1))),
    case('a sphere that the record lacks',
         'sphere fit: the spheres are not those of the archived adaptation experiment',
         in_json(SPHERE_SUMMARY, ['spheres', 'variants', 'three_shell'], DROP)),
    case('spheres in another order', 'sphere fit: the spheres are not those of the archived adaptation experiment',
         in_json(SPHERE_SUMMARY, ['spheres', 'variants'], lambda variants: dict(reversed(list(variants.items()))))),
    case('a sphere with another radius',
         'sphere fit, sphere size_matched: the head model differs from the adaptation experiment',
         in_json(SPHERE_SUMMARY, ['spheres', 'variants', 'size_matched', 'radii_m', 0], plus(0.001))),
    case('a sphere with another conductivity',
         'sphere fit, sphere three_shell: the head model differs from the adaptation experiment',
         in_json(SPHERE_SUMMARY, ['spheres', 'variants', 'three_shell', 'conductivities_S_per_m', 1], plus(0.1))),
    case('a sphere value', 'sphere fit, sphere default, raised_limit, position error, median: recomputed',
         in_json(SPHERE_SUMMARY, ['spheres', 'variants', 'default', 'raised_limit', 'position_error_mm', 'median'],
                 plus(5e-12))),
    case('a field difference of a sphere', 'sphere fit, sphere three_shell, mne_limit, field difference, maximum',
         in_json(SPHERE_SUMMARY, ['spheres', 'variants', 'three_shell', 'mne_limit', 'point_field_difference',
                                  'maximum'], plus(5e-12))),
    case('a position error of one dipole of a sphere', 'sphere fit, sphere size_matched, mne_limit, position error',
         in_csv(f'{SPHERE}/spheres_dipole_recovery.csv', where(sphere='size_matched', sample_index='5'),
                'position_error_mm_mne_limit', plus(500.0))),
    case('the change of the median error of a sphere',
         'sphere fit, sphere size_matched, change of the median position error: recomputed',
         in_json(SPHERE_SUMMARY, ['spheres', 'variants', 'size_matched', 'position_error_change_mm',
                                  'median_of_raised_limit_minus_median_of_mne_limit'], plus(5e-12))),
    case('a sphere-fit record made on other datasets', LINKED + 'same_datasets_as_the_validation_rerun',
         in_json(SPHERE_SUMMARY, ['dataset_array_sha256', 'patch'], to('e' * 64))),
    case('a sphere-fit record made on another sweep dataset', LINKED + 'same_datasets_as_the_validation_rerun',
         in_json(SPHERE_SUMMARY, ['dataset_array_sha256', 'snr_sweep', '20'], to('e' * 64))),
    case('a limit arm without one sweep level',
         'sphere fit, mne_limit: the sweep levels differ from those of the per-dipole file',
         in_json(SPHERE_SUMMARY, ['arms', 'mne_limit', 'sweep', 2], DROP)),
    case('a converged arm without one sweep level',
         'sphere fit, raised_limit: the sweep levels differ from those of the per-dipole file',
         in_json(SPHERE_SUMMARY, ['arms', 'raised_limit', 'sweep', 0], DROP)),
    case('the ratio of the two field differences of an arm',
         'sphere fit, raised_limit, ratio of the two field differences: recomputed',
         in_json(SPHERE_SUMMARY, ['arms', 'raised_limit', 'patch_comparison', 'patch_to_point_median_ratio'],
                 plus(5e-12))),
    case('the ratio of the two field differences under a kernel',
         'sphere fit, kernel Nehalem, mne_limit, ratio of the two field differences: recomputed',
         in_json(SPHERE_SUMMARY, ['library_kernels', 'arms', 'Nehalem', 'arms', 'mne_limit', 'patch_comparison',
                                  'patch_to_point_median_ratio'], plus(5e-12))),
    case('a head model of the two arms that is not the default sphere',
         'sphere fit: the head model of the two arms is not the default sphere of the adaptation experiment',
         in_json(SPHERE_SUMMARY, ['head_model', 'radii_m', 0], plus(0.001))),
    # One dipole of the limit arm differs from the rerun by 0.2 micrometres. Both per-dipole files of the
    # sphere-fit record hold the changed error, and no statistic of the arm moves.
    case('a limit arm that differs from the rerun in one dipole', LINKED + 'mne_limit_arm_equals_the_validation_rerun',
         in_csv(f'{SPHERE}/dipole_recovery_35db.csv', smallest('position_error_mm_mne_limit'),
                'position_error_mm_mne_limit', plus(0.0002)),
         in_csv(f'{SPHERE}/perturbation_dipoles.csv',
                smallest_where('position_error_mm', perturbed_fit='none', sphere_fit='mne_limit'),
                'position_error_mm', plus(0.0002))),
    # The Windows results of the two releases must stay equal, so the next cases change the same value in both:
    # one statistic of the summaries by five billionths of its value.
    *[case(f'a limit arm that differs from the rerun in {name}', LINKED + 'mne_limit_arm_equals_the_validation_rerun',
           *in_both_releases(file, route, times(1 + 5e-9)))
      for name, file, route in (
          ('the median position error', INVERSE, ['position_error_mm', 'median']),
          ('the 90th percentile of the position error', INVERSE, ['position_error_mm', 'p90']),
          ('the largest position error', INVERSE, ['position_error_mm', 'maximum']),
          ('the median orientation error', INVERSE, ['orientation_error_deg', 'median']),
          ('the median moment error', INVERSE, ['moment_amplitude', 'relative_error', 'median']),
          ('the median goodness of fit', INVERSE, ['goodness_of_fit_percent', 'median']),
          ('a sweep median', INVERSE, ['snr_degradation_sweep', 1, 'position_error_median_mm']),
          ('the field difference of the patches', 'patch/patch_forward_mne_summary.json',
           ['distributed_patch', 'median']),
          ('the field difference of the point sources', 'patch/patch_forward_mne_summary.json',
           ['point_source_control', 'median']))],
    # One dipole fewer on both platforms and in both releases: the rerun agrees with itself, not with the record.
    case('a rerun with one dipole fewer than the sphere-fit record',
         LINKED + 'mne_limit_arm_equals_the_validation_rerun',
         *without_the_last_dipole(LINUX), *without_the_last_dipole(WINDOWS, 'inverse/'),
         *without_the_last_dipole(WINDOWS_OLD, 'inverse/')),

    # ---- sphere-model fit: the other errors of a dipole and the fitted positions
    case('an orientation error of the unchanged run in the file of the trials', 'sphere fit, mne_limit' + UNCHANGED,
         in_csv(f'{SPHERE}/perturbation_dipoles.csv', where(perturbed_fit='none', sphere_fit='mne_limit',
                                                            sample_index='2'), 'orientation_error_deg', plus(1e-9))),
    case('a moment error of the unchanged run in the file of the trials', 'sphere fit, raised_limit' + UNCHANGED,
         in_csv(f'{SPHERE}/perturbation_dipoles.csv', where(perturbed_fit='none', sphere_fit='raised_limit',
                                                            sample_index='7'), 'moment_relative_error', plus(1e-12))),
    # The smallest value of a sample moves no statistic of the arm.
    case('a goodness of fit of the unchanged run in the per-window file', 'sphere fit, mne_limit' + UNCHANGED,
         in_csv(f'{SPHERE}/dipole_recovery_35db.csv', smallest('goodness_of_fit_percent_mne_limit'),
                'goodness_of_fit_percent_mne_limit', plus(1e-9))),
    case('the median moment error of a trial',
         'sphere fit, dipole_fit trial 2 with mne_limit, moment_relative_error_median: recomputed',
         in_csv(f'{SPHERE}/perturbation.csv', trial('dipole_fit', 'mne_limit', 2), 'moment_relative_error_median',
                plus(5e-12))),
    case('a moment error of one dipole of a trial',
         'sphere fit, sphere_fit trial 3 with raised_limit, moment_relative_error_median: recomputed',
         in_csv(f'{SPHERE}/perturbation_dipoles.csv',
                middle('moment_relative_error', perturbed_fit='sphere_fit', sphere_fit='raised_limit',
                       objective_scale_minus_one_in_eps='3'), 'moment_relative_error', plus(5e-12))),
    case('the median orientation error under a kernel',
         'sphere fit, kernel Sandybridge, raised_limit, orientation_error_deg, median: recomputed',
         in_json(SPHERE_SUMMARY, ['library_kernels', 'arms', 'Sandybridge', 'arms', 'raised_limit', 'dipole_recovery',
                                  'orientation_error_deg', 'median'], plus(5e-12))),
    case('the 90th percentile of the moment error under a kernel',
         'sphere fit, kernel Nehalem, mne_limit, moment_relative_error, p90: recomputed',
         in_json(SPHERE_SUMMARY, ['library_kernels', 'arms', 'Nehalem', 'arms', 'mne_limit', 'dipole_recovery',
                                  'moment_relative_error', 'p90'], plus(5e-12))),
    case('the largest goodness of fit under a kernel',
         'sphere fit, kernel Sandybridge, mne_limit, goodness_of_fit_percent, maximum: recomputed',
         in_json(SPHERE_SUMMARY, ['library_kernels', 'arms', 'Sandybridge', 'arms', 'mne_limit', 'dipole_recovery',
                                  'goodness_of_fit_percent', 'maximum'], plus(5e-12))),
    case('the number of moment errors under a kernel',
         'sphere fit, kernel Nehalem, raised_limit, moment_relative_error: 12 values, recorded 11',
         in_json(SPHERE_SUMMARY, ['library_kernels', 'arms', 'Nehalem', 'arms', 'raised_limit', 'dipole_recovery',
                                  'moment_relative_error', 'n'], plus(-1))),
    case('an orientation error of one dipole under a kernel',
         'sphere fit, kernel Nehalem, raised_limit, orientation_error_deg, ',
         in_csv(f'{SPHERE}/kernels_dipole_recovery_35db.csv',
                where(kernel='Nehalem', sphere_fit='raised_limit', sample_index='3'), 'orientation_error_deg',
                plus(50.0))),
    case('a moment error of one dipole under a kernel',
         'sphere fit, kernel Sandybridge, raised_limit, moment_relative_error, median: recomputed',
         in_csv(f'{SPHERE}/kernels_dipole_recovery_35db.csv',
                middle('moment_relative_error', kernel='Sandybridge', sphere_fit='raised_limit'),
                'moment_relative_error', plus(5e-12))),
    case('a goodness of fit of one dipole under a kernel',
         'sphere fit, kernel Nehalem, mne_limit, goodness_of_fit_percent, median: recomputed',
         in_csv(f'{SPHERE}/kernels_dipole_recovery_35db.csv',
                middle('goodness_of_fit_percent', kernel='Nehalem', sphere_fit='mne_limit'),
                'goodness_of_fit_percent', plus(5e-12))),
    case('a fitted position of a sphere with the converged fit',
         'sphere fit, sphere size_matched, change of the fitted positions, ',
         in_csv(f'{SPHERE}/spheres_dipole_recovery.csv', where(sphere='size_matched', sample_index='4'),
                'fitted_y_mm_raised_limit', plus(50.0))),
    case('a fitted position of a sphere with the fit at the limit of SciPy',
         'sphere fit, sphere surface_fitted_three_shell, change of the fitted positions, ',
         in_csv(f'{SPHERE}/spheres_dipole_recovery.csv', where(sphere='surface_fitted_three_shell', sample_index='9'),
                'fitted_x_mm_mne_limit', plus(50.0))),
    case('the change of the fitted positions of a sphere',
         'sphere fit, sphere three_shell, change of the fitted positions, p90: recomputed',
         in_json(SPHERE_SUMMARY, ['spheres', 'variants', 'three_shell', 'position_error_change_mm', 'fitted_position',
                                  'p90'], plus(5e-12))),
    case('the share of the positions of a sphere that moved',
         'sphere fit, sphere default, change of the fitted positions: the share of positions',
         in_json(SPHERE_SUMMARY, ['spheres', 'variants', 'default', 'position_error_change_mm', 'fitted_position',
                                  'share_above_0p1_mm'], plus(1 / 12))),
    case('the number of fitted positions of a sphere',
         'sphere fit, sphere size_matched_three_shell, change of the fitted positions: 12 values, recorded 13',
         in_json(SPHERE_SUMMARY, ['spheres', 'variants', 'size_matched_three_shell', 'position_error_change_mm',
                                  'fitted_position', 'n'], plus(1))),

    # ---- sphere-model fit: dipole fits that ended at a small trust-region radius
    case('the median error of a run with the small final radius',
         SMALL + 'Haswell, none 0 with mne_limit, position_error_median_mm: recomputed',
         in_csv(SMALL_RUNS, small_run('Haswell', 'none', 'mne_limit', 0), 'position_error_median_mm', plus(5e-12))),
    case('the 90th percentile of the errors of a run with the small final radius',
         SMALL + 'Haswell, parameter_fit 2 with mne_limit, position_error_p90_mm: recomputed',
         in_csv(SMALL_RUNS, small_run('Haswell', 'parameter_fit', 'mne_limit', 2), 'position_error_p90_mm',
                plus(5e-12))),
    case('the largest error of a run with the small final radius',
         SMALL + 'Nehalem, none 0 with raised_limit, position_error_maximum_mm: recomputed',
         in_csv(SMALL_RUNS, small_run('Nehalem', 'none', 'raised_limit', 0), 'position_error_maximum_mm',
                plus(5e-12))),
    case('a position error of one dipole of a run with the small final radius',
         SMALL + 'Haswell, dipole_fits 1 with raised_limit, position_error_',
         in_csv(SMALL_DIPOLES, small_run('Haswell', 'dipole_fits', 'raised_limit', 1, sample_index='4'),
                'position_error_mm', plus(50.0))),
    case('a fitted position of one dipole of a run with the small final radius',
         SMALL + 'Sandybridge, none 0 with mne_limit, fitted_position_change_',
         in_csv(SMALL_DIPOLES, small_run('Sandybridge', 'none', 'mne_limit', 0, sample_index='8'), 'fitted_z_mm',
                plus(50.0))),
    # The other runs of the sphere fit are compared with this run: the first of them no longer agrees.
    case('a fitted position of the unchanged run with the small final radius',
         SMALL + 'Haswell, parameter_fit 1 with raised_limit, fitted_position_change_',
         in_csv(SMALL_DIPOLES, small_run('Haswell', 'none', 'raised_limit', 0, sample_index='8'), 'fitted_x_mm',
                plus(50.0))),
    case('a run with the small final radius that the list of the runs lacks', OTHER_RUNS,
         without_rows(SMALL_RUNS, kernel='Nehalem', sphere_fit='raised_limit')),
    case('a run with the small final radius that the list holds twice', OTHER_RUNS,
         with_row_twice(SMALL_RUNS, small_run('Haswell', 'dipole_fits', 'mne_limit', 2))),
    case('a run with the small final radius that the list holds under another objective', OTHER_RUNS,
         in_csv(SMALL_RUNS, small_run('Haswell', 'parameter_fit', 'raised_limit', 2),
                'objective_scale_minus_one_in_eps', to('3'))),
    case('a dipole of a run with the small final radius that is not in the list of the runs', OTHER_RUNS,
         in_csv(SMALL_DIPOLES, small_run('Sandybridge', 'none', 'raised_limit', 0, sample_index='11'), 'kernel',
                to('Zen'))),
    case('a run with the small final radius that holds a dipole twice',
         'sphere fit: a run does not hold every dipole once, in order',
         in_csv(SMALL_DIPOLES, small_run('Haswell', 'dipole_fits', 'mne_limit', 1, sample_index='3'), 'sample_index',
                to('5'))),
    case('an unchanged run with the small final radius without its last dipole',
         'sphere fit: a run does not hold every dipole once, in order',
         without_rows(SMALL_DIPOLES, kernel='Haswell', changed_objective='none', sphere_fit='mne_limit',
                      sample_index='11')),
    case('the median change of the positions of a run with the small final radius',
         SMALL + 'Haswell, dipole_fits 1 with raised_limit, fitted_position_change_median_mm: recomputed',
         in_csv(SMALL_RUNS, small_run('Haswell', 'dipole_fits', 'raised_limit', 1), 'fitted_position_change_median_mm',
                plus(5e-9))),
    case('the 90th percentile of the changes of the positions of a run with the small final radius',
         SMALL + 'Sandybridge, none 0 with mne_limit, fitted_position_change_p90_mm: recomputed',
         in_csv(SMALL_RUNS, small_run('Sandybridge', 'none', 'mne_limit', 0), 'fitted_position_change_p90_mm',
                plus(5e-9))),
    case('the largest change of the positions of a run with the small final radius',
         SMALL + 'Haswell, parameter_fit 2 with raised_limit, fitted_position_change_maximum_mm: recomputed',
         in_csv(SMALL_RUNS, small_run('Haswell', 'parameter_fit', 'raised_limit', 2),
                'fitted_position_change_maximum_mm', plus(-5e-9))),
    # A change of zero is a change that the row records.
    case('an unchanged run of the main process that records a change of its positions',
         SMALL + 'Haswell, none 0 with mne_limit' + NO_CHANGE,
         in_csv(SMALL_RUNS, small_run('Haswell', 'none', 'mne_limit', 0), 'fitted_position_change_median_mm',
                to('0.0'))),
    case('a run under another kernel without the change of its positions',
         SMALL + 'Sandybridge, none 0 with raised_limit' + NO_CHANGE,
         in_csv(SMALL_RUNS, small_run('Sandybridge', 'none', 'raised_limit', 0), 'fitted_position_change_median_mm',
                to(''))),
    case('a run with a changed objective without the change of its positions',
         SMALL + 'Haswell, dipole_fits 2 with mne_limit' + NO_CHANGE,
         in_csv(SMALL_RUNS, small_run('Haswell', 'dipole_fits', 'mne_limit', 2), 'fitted_position_change_median_mm',
                to(''))),
    # The record names the kernel of the main process twice: with the runs with the small final radius and with
    # the kernels of the library. The two statements must agree, whichever of them is changed.
    case('another kernel named as that of the main process', whole(OTHER_MAIN_KERNEL),
         in_json(SPHERE_SUMMARY, ['small_tolerance', 'kernel_of_the_main_process'], to('Sandybridge'))),
    case('a record of the kernels that names another kernel of the main process', whole(OTHER_MAIN_KERNEL),
         in_json(SPHERE_SUMMARY, ['library_kernels', 'of_the_parent_process'],
                 lambda entries: [dict(entry, kernel='Sandybridge') for entry in entries])),
    # Every library of the record counts, the first as well as the last: with two kernels the experiment script
    # would have named both.
    case('a record of the kernels that names two kernels of the main process', whole(OTHER_MAIN_KERNEL),
         in_json(SPHERE_SUMMARY, ['library_kernels', 'of_the_parent_process', 1, 'kernel'], to('Zen'))),
    case('a record of the kernels whose first library names another kernel of the main process',
         whole(OTHER_MAIN_KERNEL),
         in_json(SPHERE_SUMMARY, ['library_kernels', 'of_the_parent_process', 0, 'kernel'], to('Zen'))),
    # With no library the script would have named the kernel 'unknown'.
    case('a record of the kernels that names no library of the main process', whole(OTHER_MAIN_KERNEL),
         in_json(SPHERE_SUMMARY, ['library_kernels', 'of_the_parent_process'], to([]))),
    # The experiment script names the kernel of the main process by a rule: the kernels that the loaded
    # libraries report, each once and in the order of the alphabet, joined by '+', and 'unknown' if no library
    # is reported. In the records below the runs of the main process carry the name that the record gives, so
    # that the record agrees with itself and only the rule can notice.
    case('a main process without a library whose kernel has no name', whole(OTHER_MAIN_KERNEL),
         *with_the_main_process_as('', [])),
    case('a main process with libraries whose kernel is named unknown', whole(OTHER_MAIN_KERNEL),
         *with_the_main_process_as('unknown', ['Haswell', 'Haswell'])),
    case('a main process with a library that reports no kernel whose kernel is named unknown',
         whole(OTHER_MAIN_KERNEL), *with_the_main_process_as('unknown', ['None'])),
    case('two kernels of the main process named in the order of the libraries', whole(OTHER_MAIN_KERNEL),
         *with_the_main_process_as('Zen+Haswell', ['Zen', 'Haswell'])),
    case('the kernel of two libraries of the main process named twice', whole(OTHER_MAIN_KERNEL),
         *with_the_main_process_as('Haswell+Haswell', ['Haswell', 'Haswell'])),
    case('two kernels of the main process named by the last of them', whole(OTHER_MAIN_KERNEL),
         *with_the_main_process_as('Zen', ['Haswell', 'Zen'])),
    case('two kernels of the main process joined by a comma', whole(OTHER_MAIN_KERNEL),
         *with_the_main_process_as('Haswell, Zen', ['Haswell', 'Zen'])),
    case('three kernels of the main process of which the record names two', whole(OTHER_MAIN_KERNEL),
         *with_the_main_process_as('Haswell+Zen', ['Haswell', 'Prescott', 'Zen'])),
    case('the kernel of the main process named with a small first letter', whole(OTHER_MAIN_KERNEL),
         *with_the_main_process_as('haswell', ['Haswell', 'Haswell'])),
    case('the kernel of the main process named with a blank after it', whole(OTHER_MAIN_KERNEL),
         *with_the_main_process_as('Haswell ', ['Haswell', 'Haswell'])),
    # Both statements name a kernel under which the unchanged runs were made but no trial. The message lists
    # the eight missing runs and the ten runs of the other kernel in their order: no order of a set gives this
    # text by chance.
    case('another kernel named as that of the main process by both statements of the record',
         whole(NOT_THE_RUNS
               + 'Missing: ' + runs_under('Sandybridge', ('dipole_fits', (1, 2)), ('parameter_fit', (1, 2)))
               + '. Not expected: '
               + runs_under('Haswell', ('dipole_fits', (1, 2)), ('none', (0,)), ('parameter_fit', (1, 2)))),
         main_kernel_as('Sandybridge')),
    # Each of the three cells counts by itself, in both directions. The first value comes from another run of
    # the same sphere fit, so it lies within the recorded range and no later check would notice it.
    case('an unchanged run of the main process that records the largest change of its positions',
         SMALL + 'Haswell, none 0 with raised_limit' + NO_CHANGE,
         with_cell_of(SMALL_RUNS, small_run('Haswell', 'dipole_fits', 'raised_limit', 1),
                      small_run('Haswell', 'none', 'raised_limit', 0), 'fitted_position_change_maximum_mm')),
    case('an unchanged run of the main process that records the 90th percentile of the changes of its positions',
         SMALL + 'Haswell, none 0 with mne_limit' + NO_CHANGE,
         with_cell_of(SMALL_RUNS, small_run('Sandybridge', 'none', 'mne_limit', 0),
                      small_run('Haswell', 'none', 'mne_limit', 0), 'fitted_position_change_p90_mm')),
    case('a run without the 90th percentile of the changes of its positions',
         SMALL + 'Sandybridge, none 0 with mne_limit' + NO_CHANGE,
         in_csv(SMALL_RUNS, small_run('Sandybridge', 'none', 'mne_limit', 0), 'fitted_position_change_p90_mm',
                to(''))),
    case('a run without the largest change of its positions',
         SMALL + 'Haswell, parameter_fit 1 with raised_limit' + NO_CHANGE,
         in_csv(SMALL_RUNS, small_run('Haswell', 'parameter_fit', 'raised_limit', 1),
                'fitted_position_change_maximum_mm', to(''))),
    # The last run of the list: every run is looked at.
    case('the last run without the largest change of its positions',
         SMALL + 'Nehalem, none 0 with raised_limit' + NO_CHANGE,
         in_csv(SMALL_RUNS, small_run('Nehalem', 'none', 'raised_limit', 0), 'fitted_position_change_maximum_mm',
                to(''))),

    # ---- sphere-model fit: the runs with the small final radius that the record must hold
    case('runs with the small final radius without the unchanged run of the main process at the limit of SciPy',
         NO_MAIN_RUN + 'mne_limit', *in_both_small_files('Haswell', 'none', 'mne_limit', 0)),
    case('runs with the small final radius without the unchanged run of the main process with the converged fit',
         NO_MAIN_RUN + 'raised_limit', *in_both_small_files('Haswell', 'none', 'raised_limit', 0)),
    case('runs with the small final radius without both unchanged runs of the main process',
         NO_MAIN_RUN + 'mne_limit, raised_limit', *in_both_small_files('Haswell', 'none', 'mne_limit', 0),
         *in_both_small_files('Haswell', 'none', 'raised_limit', 0)),
    case('a kernel of the main process under which no run was made', NO_MAIN_RUN + 'mne_limit, raised_limit',
         main_kernel_as('Zen')),
    # The unchanged run of the main process has the factor 0: under another factor it is another run.
    case('an unchanged run of the main process under the number of a trial', NO_MAIN_RUN + 'raised_limit',
         *in_both_small_files('Haswell', 'none', 'raised_limit', 0, objective_scale_minus_one_in_eps='1')),
    case('a record that names one more trial than the runs with the small final radius', OTHER_TRIALS,
         in_json(SPHERE_SUMMARY, ['small_tolerance', 'steps'], to([1, 2, 3]))),
    case('a record that names one trial fewer than the runs with the small final radius', OTHER_TRIALS,
         in_json(SPHERE_SUMMARY, ['small_tolerance', 'steps'], to([1]))),
    case('a record that names another trial than the runs with the small final radius', OTHER_TRIALS,
         in_json(SPHERE_SUMMARY, ['small_tolerance', 'steps'], to([1, 3]))),
    case('a record that names no trial of the runs with the small final radius', OTHER_TRIALS,
         in_json(SPHERE_SUMMARY, ['small_tolerance', 'steps'], to([]))),
    case('a record that names the trials of the runs with the small final radius in another order', OTHER_TRIALS,
         in_json(SPHERE_SUMMARY, ['small_tolerance', 'steps'], to([2, 1]))),
    case('a record that names the trials of the runs with the small final radius as texts', OTHER_TRIALS,
         in_json(SPHERE_SUMMARY, ['small_tolerance', 'steps'], to(['1', '2']))),
    # A run with changed dipole fits, and a run with a changed parameter fit that another kernel made: the
    # trials of the record are those of every run with a changed objective.
    case('a run with the small final radius under a trial that the record does not name', OTHER_TRIALS,
         *in_both_small_files('Haswell', 'dipole_fits', 'mne_limit', 2, objective_scale_minus_one_in_eps='3')),
    case('a run under another kernel with a trial that the record does not name', OTHER_TRIALS,
         *in_both_small_files('Haswell', 'parameter_fit', 'raised_limit', 2, kernel='Nehalem',
                              objective_scale_minus_one_in_eps='3')),
    # An unchanged run has the factor 0, under every kernel and also next to the unchanged run that has it.
    # Under another factor it is a run that the record should not hold: the trials of the record still are
    # those of the runs with a changed objective, and the message names the run.
    case('an unchanged run under another kernel under the number of a trial',
         whole(NOT_THE_RUNS + 'Missing: kernel Nehalem, none 0 with raised_limit. Not expected: kernel Nehalem, '
               'none 1 with raised_limit'),
         *in_both_small_files('Nehalem', 'none', 'raised_limit', 0, objective_scale_minus_one_in_eps='1')),
    case('an unchanged run under another kernel under a number that is no trial',
         whole(NOT_THE_RUNS + 'Missing: kernel Sandybridge, none 0 with mne_limit. Not expected: kernel '
               'Sandybridge, none 5 with mne_limit'),
         *in_both_small_files('Sandybridge', 'none', 'mne_limit', 0, objective_scale_minus_one_in_eps='5')),
    case('an unchanged run under another kernel under a number below zero',
         whole(NOT_THE_RUNS + 'Missing: kernel Sandybridge, none 0 with raised_limit. Not expected: kernel '
               'Sandybridge, none -1 with raised_limit'),
         *in_both_small_files('Sandybridge', 'none', 'raised_limit', 0, objective_scale_minus_one_in_eps='-1')),
    case('a second unchanged run of the main process under the number of a trial',
         whole(NOT_THE_RUNS + 'Missing: none. Not expected: kernel Haswell, none 2 with mne_limit'),
         *in_both_small_files('Haswell', 'none', 'mne_limit', 0, copy=True, objective_scale_minus_one_in_eps='2')),
    # Both files and the ranges of the record lack the run, so that the record agrees with itself.
    case('runs with the small final radius without the unchanged run under another kernel',
         NOT_THE_RUNS + 'Missing: kernel Nehalem, none 0 with raised_limit. Not expected: none',
         *with_the_ranges_of_the_small_runs(*in_both_small_files('Nehalem', 'none', 'raised_limit', 0))),
    case('runs with the small final radius without both runs under a kernel of the record',
         NOT_THE_RUNS + 'Missing: kernel Nehalem, none 0 with mne_limit; kernel Nehalem, none 0 with raised_limit. '
         'Not expected: none',
         *with_the_ranges_of_the_small_runs(*in_both_small_files('Nehalem', 'none', 'mne_limit', 0),
                                            *in_both_small_files('Nehalem', 'none', 'raised_limit', 0))),
    # The other runs of the record still name every trial, so that only the list of the expected runs can notice.
    case('runs with the small final radius without a trial of the dipole fits with one sphere fit',
         NOT_THE_RUNS + 'Missing: kernel Haswell, dipole_fits 2 with raised_limit. Not expected: none',
         *with_the_ranges_of_the_small_runs(*in_both_small_files('Haswell', 'dipole_fits', 'raised_limit', 2))),
    case('runs with the small final radius without a trial of the parameter fit with one sphere fit',
         NOT_THE_RUNS + 'Missing: kernel Haswell, parameter_fit 1 with mne_limit. Not expected: none',
         *with_the_ranges_of_the_small_runs(*in_both_small_files('Haswell', 'parameter_fit', 'mne_limit', 1))),
    # Copies of the two unchanged runs under a kernel that the record of the kernels does not hold. Each copy
    # agrees with its rows and the ranges follow, so that nothing but the name of the kernel is wrong.
    case('runs with the small final radius under a kernel that the record of the kernels does not name',
         NOT_THE_RUNS + 'Missing: none. Not expected: kernel Zen, none 0 with mne_limit; kernel Zen, none 0 with '
         'raised_limit',
         *with_the_ranges_of_the_small_runs(
             *in_both_small_files('Nehalem', 'none', 'mne_limit', 0, copy=True, kernel='Zen'),
             *in_both_small_files('Nehalem', 'none', 'raised_limit', 0, copy=True, kernel='Zen'))),
    case('a run with a changed parameter fit under another kernel, the ranges following',
         NOT_THE_RUNS + 'Missing: kernel Haswell, parameter_fit 2 with mne_limit. Not expected: kernel Nehalem, '
         'parameter_fit 2 with mne_limit',
         *with_the_ranges_of_the_small_runs(
             *in_both_small_files('Haswell', 'parameter_fit', 'mne_limit', 2, kernel='Nehalem'))),
    case('a trial of the dipole fits that another kernel repeated, the ranges following',
         NOT_THE_RUNS + 'Missing: none. Not expected: kernel Sandybridge, dipole_fits 1 with mne_limit',
         *with_the_ranges_of_the_small_runs(
             *in_both_small_files('Haswell', 'dipole_fits', 'mne_limit', 1, copy=True, kernel='Sandybridge'))),
    case('a run with the small final radius with an objective that the experiment did not change',
         NOT_THE_RUNS + 'Missing: kernel Haswell, parameter_fit 1 with raised_limit. Not expected: kernel Haswell, '
         'both 1 with raised_limit',
         *in_both_small_files('Haswell', 'parameter_fit', 'raised_limit', 1, changed_objective='both')),
    case('the upper end of a range over the runs with the small final radius', SMALL_RANGES,
         in_json(SPHERE_SUMMARY, ['small_tolerance', 'ranges', 'mne_limit', 'position_error_p90_mm', 1], plus(1e-12))),
    case('the lower end of a range of the changes of the positions over the runs with the small final radius',
         SMALL_RANGES,
         in_json(SPHERE_SUMMARY, ['small_tolerance', 'ranges', 'raised_limit', 'fitted_position_change_median_mm', 0],
                 plus(1e-12))),
    case('the number of the runs with the small final radius', SMALL_RANGES,
         in_json(SPHERE_SUMMARY, ['small_tolerance', 'ranges', 'raised_limit', 'runs'], plus(1))),
    case('the kernels of the runs with the small final radius in the order of the runs', SMALL_RANGES,
         in_json(SPHERE_SUMMARY, ['small_tolerance', 'ranges', 'mne_limit', 'kernels'],
                 to(['Haswell', 'Sandybridge', 'Nehalem']))),
    case('the end of the parameter fit of a run with the small final radius', SMALL_RANGES,
         in_csv(SMALL_RUNS, small_run('Haswell', 'parameter_fit', 'mne_limit', 1), 'parameter_fit_stopped_by_itself',
                to('1'))),
    case('the number of dipole fits of a run with the small final radius', SMALL_RANGES,
         in_csv(SMALL_RUNS, small_run('Nehalem', 'none', 'mne_limit', 0), 'dipole_fits', count_plus(1))),
    case('the dipole fits of a run with the small final radius that stopped by themselves', SMALL_RANGES,
         in_csv(SMALL_RUNS, small_run('Haswell', 'dipole_fits', 'raised_limit', 2),
                'dipole_fits_stopped_by_themselves', count_plus(-1))),
    case('the median error under a kernel with the small final radius',
         SMALL + 'Sandybridge, mne_limit, position error, median: recomputed',
         in_json(SPHERE_SUMMARY, ['library_kernels', 'arms', 'Sandybridge', 'small_tolerance', 'mne_limit',
                                  'position_error_mm', 'median'], plus(5e-12))),
    case('the number of dipoles under a kernel with the small final radius',
         SMALL + 'Nehalem, mne_limit, position error: 12 values, recorded 13',
         in_json(SPHERE_SUMMARY, ['library_kernels', 'arms', 'Nehalem', 'small_tolerance', 'mne_limit',
                                  'position_error_mm', 'n'], plus(1))),
    case('the largest change of the positions under a kernel with the small final radius',
         SMALL + 'Nehalem, raised_limit, change of the fitted positions, maximum: recomputed',
         in_json(SPHERE_SUMMARY, ['library_kernels', 'arms', 'Nehalem', 'small_tolerance', 'raised_limit',
                                  'fitted_position_change_from_the_parent_mm', 'maximum'], plus(5e-12))),
    case('the share of the positions that moved under a kernel with the small final radius',
         SMALL + 'Sandybridge, mne_limit, change of the fitted positions: the share of positions',
         in_json(SPHERE_SUMMARY, ['library_kernels', 'arms', 'Sandybridge', 'small_tolerance', 'mne_limit',
                                  'fitted_position_change_from_the_parent_mm', 'share_above_0p1_mm'], plus(-1 / 12))),

    # ---- dipole comparators with a converged parameter fit
    case('a smoke record of the dipole comparators',
         'The record of the dipole comparators is a smoke run or was not compared with the recorded experiments',
         in_json(COMPARATORS_SUMMARY, ['smoke'], to(True))),
    case('dipole comparators that were not compared with the recorded experiments',
         'The record of the dipole comparators is a smoke run or was not compared with the recorded experiments',
         in_json(COMPARATORS_SUMMARY, ['compared_with_the_recorded_experiments'], to(False))),
    case('dipole comparators with another evaluation limit',
         'dipole comparators: the evaluation limits differ from those of the sphere-fit record',
         in_json(COMPARATORS_SUMMARY, ['evaluation_limits', 'raised_limit'], plus(1))),
    case('a sphere-fit record with another evaluation limit',
         'dipole comparators: the evaluation limits differ from those of the sphere-fit record',
         in_json(SPHERE_SUMMARY, ['evaluation_limits', 'mne_limit'], plus(-1))),
    # Both records name the same limits, so that they agree, but not the two fits of the paper: one of them
    # under another name, a third one, or one of the two only. Every group still holds both fits, so that only
    # the names can notice. The message lists the arms of the record, which may be too many or too few.
    case('dipole comparators and a sphere-fit record with another arm',
         whole("dipole comparators: the arms are not mne_limit and raised_limit: ['long_limit', 'mne_limit']"),
         in_json(SPHERE_SUMMARY, ['evaluation_limits'], renamed('raised_limit', 'long_limit')),
         in_json(COMPARATORS_SUMMARY, ['evaluation_limits'], renamed('raised_limit', 'long_limit'))),
    case('dipole comparators and a sphere-fit record with a third arm',
         whole("dipole comparators: the arms are not mne_limit and raised_limit: ['long_limit', 'mne_limit', "
               "'raised_limit']"),
         in_json(SPHERE_SUMMARY, ['evaluation_limits'], lambda limits: dict(limits, long_limit=10 ** 7)),
         in_json(COMPARATORS_SUMMARY, ['evaluation_limits'], lambda limits: dict(limits, long_limit=10 ** 7))),
    case('dipole comparators and a sphere-fit record without the arm with the raised limit',
         whole("dipole comparators: the arms are not mne_limit and raised_limit: ['mne_limit']"),
         in_json(SPHERE_SUMMARY, ['evaluation_limits', 'raised_limit'], DROP),
         in_json(COMPARATORS_SUMMARY, ['evaluation_limits', 'raised_limit'], DROP)),
    case('dipole comparators and a sphere-fit record without the arm at the limit of SciPy',
         whole("dipole comparators: the arms are not mne_limit and raised_limit: ['raised_limit']"),
         in_json(SPHERE_SUMMARY, ['evaluation_limits', 'mne_limit'], DROP),
         in_json(COMPARATORS_SUMMARY, ['evaluation_limits', 'mne_limit'], DROP)),
    case('a fitted window of the adaptation experiment twice in its eligibility file',
         'adaptation, dataset 862' + TWICE, fitted_window_twice(ADAPTATION, 'dipole_subset.csv')),
    case('a window of the adaptation experiment that was not fitted twice in its eligibility file',
         'adaptation, dataset 862' + TWICE,
         with_row_twice(f'{ADAPTATION}/862/eligibility.csv', lambda rows: len(rows) - 1)),
    case('a group of the dipole comparators without its windows', OTHER_GROUPS,
         without_rows(COMPARATORS_CSV, experiment='transfer')),
    case('a window of a group that the summary of the dipole comparators lacks', OTHER_GROUPS,
         in_csv(COMPARATORS_CSV, group('adaptation', '862', 'three_shell_sphere_on_bem'), 'case',
                to('four_shell_sphere_on_bem'))),
    case('a summary of the dipole comparators without one group', OTHER_GROUPS,
         in_json(COMPARATORS_SUMMARY, ['adaptation', '862', 'size_matched_sphere_on_bem'], DROP)),
    case('a group of the dipole comparators under another name in the summary', OTHER_GROUPS,
         in_json(COMPARATORS_SUMMARY, ['split_study', '863'], renamed('region', 'sector'))),
    case('the median of a group with the limit of SciPy',
         'dipole comparators, split_study, dataset 863, region, mne_limit, median_mm: recomputed',
         in_json(COMPARATORS_SUMMARY, ['split_study', '863', 'region', 'mne_limit', 'median_mm'], plus(5e-10))),
    case('the mean of a group with the converged fit',
         'dipole comparators, transfer, dataset 862, package_sphere_matched, raised_limit, mean_mm: recomputed',
         in_json(COMPARATORS_SUMMARY, ['transfer', '862', 'package_sphere_matched', 'raised_limit', 'mean_mm'],
                 plus(5e-10))),
    case('the 90th percentile of a group with the converged fit',
         'dipole comparators, adaptation, dataset 862, default_sphere_on_bem, raised_limit, p90_mm: recomputed',
         in_json(COMPARATORS_SUMMARY, ['adaptation', '862', 'default_sphere_on_bem', 'raised_limit', 'p90_mm'],
                 plus(5e-10))),
    case('the number of windows of a group',
         'dipole comparators, transfer, dataset 862, package_sphere_matched, mne_limit: 12 windows, recorded 13',
         in_json(COMPARATORS_SUMMARY, ['transfer', '862', 'package_sphere_matched', 'mne_limit', 'windows'], plus(1))),
    case('the number of windows of a group with the converged fit',
         'dipole comparators, split_study, dataset 863, position, raised_limit: 12 windows, recorded 11',
         in_json(COMPARATORS_SUMMARY, ['split_study', '863', 'position', 'raised_limit', 'windows'], plus(-1))),
    case('a group of the dipole comparators without a statistic',
         'dipole comparators, split_study, dataset 862, position, raised_limit: the record lacks mean_mm',
         in_json(COMPARATORS_SUMMARY, ['split_study', '862', 'position', 'raised_limit', 'mean_mm'], DROP)),
    case('an error of one window with the converged fit',
         'dipole comparators, split_study, dataset 862, position, raised_limit, me',
         in_csv(COMPARATORS_CSV, group('split_study', '862', 'position'), 'position_error_mm_raised_limit',
                plus(300.0))),
    case('an error of one window with the limit of SciPy',
         'dipole comparators, adaptation, dataset 862, size_matched_three_shell_sphere_on_bem, mne_limit, me',
         in_csv(COMPARATORS_CSV, group('adaptation', '862', 'size_matched_three_shell_sphere_on_bem'),
                'position_error_mm_mne_limit', plus(300.0))),
    case('the change of the median of a group',
         'dipole comparators, split_study, dataset 864, position, change of the median: recomputed',
         in_json(COMPARATORS_SUMMARY,
                 ['split_study', '864', 'position', 'median_of_raised_limit_minus_median_of_mne_limit_mm'],
                 plus(5e-10))),
    case('the change of the median of a group with the other sign',
         'dipole comparators, adaptation, dataset 862, size_matched_sphere_on_bem, change of the median: recomputed',
         in_json(COMPARATORS_SUMMARY, ['adaptation', '862', 'size_matched_sphere_on_bem',
                                       'median_of_raised_limit_minus_median_of_mne_limit_mm'], times(-1.0))),
    case('the median change of the fitted positions of a group',
         'dipole comparators, split_study, dataset 862, region, fitted position change, median: recomputed',
         in_json(COMPARATORS_SUMMARY, ['split_study', '862', 'region', 'fitted_position_change_mm', 'median'],
                 plus(5e-9))),
    case('the 90th percentile of the changes of the fitted positions of a group',
         'dipole comparators, adaptation, dataset 862, three_shell_sphere_on_bem, fitted position change, p90: recomp',
         in_json(COMPARATORS_SUMMARY, ['adaptation', '862', 'three_shell_sphere_on_bem', 'fitted_position_change_mm',
                                       'p90'], plus(5e-9))),
    case('the largest change of the fitted positions of a group',
         'dipole comparators, transfer, dataset 862, package_sphere_matched, fitted position change, maximum: recomp',
         in_json(COMPARATORS_SUMMARY, ['transfer', '862', 'package_sphere_matched', 'fitted_position_change_mm',
                                       'maximum'], plus(-5e-9))),
    case('a fitted position of one window with the converged fit',
         'dipole comparators, split_study, dataset 863, position, fitted position change, ',
         in_csv(COMPARATORS_CSV, group('split_study', '863', 'position'), 'fitted_z_mm_raised_limit', plus(50.0))),
    case('a window of the dipole comparators that the transfer experiment did not fit',
         'dipole comparators, transfer, dataset 862, package_sphere_matched' + OTHER_WINDOWS,
         in_csv(COMPARATORS_CSV, group('transfer', '862', 'package_sphere_matched'), 'window_id', count_plus(1))),
    case('a window of the transfer experiment that the dipole comparators did not fit',
         'dipole comparators, transfer, dataset 862, package_sphere_matched' + OTHER_WINDOWS,
         in_csv(f'{TRANSFER}/862/ecd_subset.csv', 11, 'window_id', count_plus(1))),
    case('a window of the adaptation experiment that the dipole comparators did not fit',
         'dipole comparators, adaptation, dataset 862, default_sphere_on_bem' + OTHER_WINDOWS,
         in_csv(f'{ADAPTATION}/862/dipole_subset.csv', 6, 'window_id', count_plus(1))),
    case('a test window of the split study that the dipole comparators did not fit',
         'dipole comparators, split_study, dataset 863, position' + OTHER_WINDOWS,
         in_json(f'{SPLIT}/863/position/ecd_subset/test_rows.json', [4], plus(1))),
    case('the test windows of the split study in another order',
         'dipole comparators, split_study, dataset 862, region' + OTHER_WINDOWS,
         in_json(f'{SPLIT}/862/region/ecd_subset/test_rows.json', [], lambda rows: [rows[1], rows[0], *rows[2:]])),
    # The split study names the windows of its dipole fits in a file of their own: one number for every fit.
    case('one more test window of the split study than dipole fits',
         'split study, dataset 864, region: the dipole-fit subset holds 12 rows and 13 window numbers',
         in_json(f'{SPLIT}/864/region/ecd_subset/test_rows.json', [], lambda rows: [*rows, 71])),
    case('one test window of the split study fewer than dipole fits',
         'split study, dataset 862, position: the dipole-fit subset holds 12 rows and 11 window numbers',
         in_json(f'{SPLIT}/862/position/ecd_subset/test_rows.json', [], lambda rows: rows[:-1])),
    case('a dipole fit of the transfer experiment on a window without a source position',
         'transfer, dataset 862' + NO_SOURCE,
         in_csv(f'{TRANSFER}/862/ecd_subset.csv', 4, 'window_id', to('500')),
         in_rows(COMPARATORS_CSV, lambda rows: [
             dict(row, window_id='500') if row is members_of(rows, *TRANSFER_GROUP)[4] else row for row in rows])),
    case('an adaptation experiment without the source position of a window of its dipole fits',
         'adaptation, dataset 862' + NO_SOURCE, source_of_a_fitted_window(ADAPTATION, 'dipole_subset.csv')),
    case('a fit with the limit of SciPy that is not the fit of the transfer experiment',
         'dipole comparators, transfer, dataset 862, package_sphere_matched' + OTHER_FIT,
         in_csv(f'{TRANSFER}/862/ecd_subset.csv', 5, 'ecd_package_sphere_matched_y_mm', plus(5e-9))),
    case('a fit with the limit of SciPy that is not the fit of the adaptation experiment',
         'dipole comparators, adaptation, dataset 862, surface_fitted_three_shell_sphere_on_bem' + OTHER_FIT,
         in_csv(f'{ADAPTATION}/862/dipole_subset.csv', 9, 'dipole_surface_fitted_three_shell_z_mm', plus(5e-9))),
    # Both fits of one window move by the same step, so that no statistic of the group changes.
    case('a fit with the limit of SciPy that is not the fit of the split study',
         'dipole comparators, split_study, dataset 864, region' + OTHER_FIT,
         *[in_csv(COMPARATORS_CSV, group('split_study', '864', 'region'), f'fitted_x_mm_{name}', plus(5e-9))
           for name in ('mne_limit', 'raised_limit')]),
    case('a parameter fit of the dipole comparators with one more evaluation',
         'dipole comparators, split_study, dataset 862, position: the parameter fit with raised_limit did not end as',
         in_json(COMPARATORS_SUMMARY, ['split_study', '862', 'position', 'parameter_fit', 'raised_limit',
                                       'evaluations'], plus(1))),
    case('a parameter fit of the dipole comparators that stopped by itself',
         'dipole comparators, adaptation, dataset 862, three_shell_sphere_on_bem: the parameter fit with mne_limit did',
         in_json(COMPARATORS_SUMMARY, ['adaptation', '862', 'three_shell_sphere_on_bem', 'parameter_fit', 'mne_limit',
                                       'stopped_by_itself'], to(True))),
    case('a parameter fit of the dipole comparators with another objective',
         'dipole comparators, transfer, dataset 862, package_sphere_matched: the parameter fit with raised_limit did',
         in_json(COMPARATORS_SUMMARY, ['transfer', '862', 'package_sphere_matched', 'parameter_fit', 'raised_limit',
                                       'objective'], times(1 + 5e-9))),
    case('a sphere whose parameter fit ended in another way than for the dipole comparators',
         'dipole comparators, adaptation, dataset 862, size_matched_sphere_on_bem: the parameter fit with raised_limit',
         in_json(SPHERE_SUMMARY, ['spheres', 'variants', 'size_matched', 'raised_limit', 'fit', 'evaluations'],
                 plus(1))),
    case('a default sphere whose parameter fit has another objective than for the dipole comparators',
         'dipole comparators, split_study, dataset 862, position: the parameter fit with mne_limit did not end as in',
         in_json(SPHERE_SUMMARY, ['spheres', 'variants', 'default', 'mne_limit', 'fit', 'objective'],
                 times(1 - 5e-9))),

    # ---- dipole comparators: the fits that the paper reports, and no others
    # The per-window file and the summary change alike, so that the record agrees with itself.
    case('dipole comparators without one dataset of the split study', REPORTED_FITS,
         *without_group('split_study', '864', 'region')),
    case('dipole comparators without the fit of the transfer experiment', REPORTED_FITS,
         *without_group(*TRANSFER_GROUP)),
    # The record also lists its spheres without this one: the spheres that count are those of the sphere-fit
    # record, which are those of the adaptation experiment.
    case('dipole comparators without one sphere of the adaptation experiment', REPORTED_FITS,
         *without_group('adaptation', '862', 'size_matched_sphere_on_bem'),
         in_json(COMPARATORS_SUMMARY, ['spheres', 'size_matched'], DROP)),
    case('dipole comparators without the last sphere of the adaptation experiment', REPORTED_FITS,
         *without_group('adaptation', '862', 'surface_fitted_three_shell_sphere_on_bem')),
    case('dipole comparators with the fit of the transfer experiment for a second dataset', REPORTED_FITS,
         *group_as(TRANSFER_GROUP, ('transfer', '863', 'package_sphere_matched'), keep=True)),
    case('dipole comparators with a sphere that the sphere-fit record does not hold', REPORTED_FITS,
         *group_as(('adaptation', '862', 'three_shell_sphere_on_bem'),
                   ('adaptation', '862', 'four_shell_sphere_on_bem'), keep=True),
         in_json(COMPARATORS_SUMMARY, ['spheres'], lambda spheres: dict(spheres, four_shell=spheres['three_shell']))),
    case('dipole comparators with a third policy of the split study', REPORTED_FITS,
         *group_as(('split_study', '862', 'region'), ('split_study', '862', 'patch'), keep=True)),
    # As many groups as the paper reports, one of which is another fit.
    case('dipole comparators with another fit of the transfer experiment in place of the matched sphere',
         REPORTED_FITS, *group_as(TRANSFER_GROUP, ('transfer', '862', 'bem_matched'))),
    case('dipole comparators with the fit of the transfer experiment for another dataset', REPORTED_FITS,
         *group_as(TRANSFER_GROUP, ('transfer', '863', 'package_sphere_matched'))),
    case('dipole comparators with a sphere of the adaptation experiment for another dataset', REPORTED_FITS,
         *group_as(('adaptation', '862', 'default_sphere_on_bem'), ('adaptation', '864', 'default_sphere_on_bem'))),

    # ---- dipole comparators: one number of windows for every fit
    case('a record of the dipole comparators that names another number of windows of a fit',
         'dipole comparators, split_study, dataset 862, position: 12 windows, the record names 13 per fit',
         in_json(COMPARATORS_SUMMARY, ['windows_per_subset'], plus(1))),
    case('a group of the dipole comparators with one window fewer than the record names',
         'dipole comparators, transfer, dataset 862, package_sphere_matched: 11 windows, the record names 12 per fit',
         *with_the_summary_of_the_comparators(without_the_last_window(*TRANSFER_GROUP))),
    case('the last group of the dipole comparators with one window more than the record names',
         'dipole comparators, adaptation, dataset 862, surface_fitted_three_shell_sphere_on_bem: 13 windows, the '
         'record names 12 per fit',
         *with_the_summary_of_the_comparators(with_row_twice(
             COMPARATORS_CSV, group('adaptation', '862', 'surface_fitted_three_shell_sphere_on_bem')))),

    # ---- dipole comparators: the errors with the limit of SciPy are those of the recorded experiments
    # One error of a group changes by 5e-9 mm and the summary follows it: the group agrees with itself, and its
    # fitted positions are those of the experiment. The changed error of the split study is the smallest of its
    # group, so that no median moves and the mean of the dataset medians still is the one of the split study.
    case('an error with the limit of SciPy that is not the error of the transfer experiment',
         'dipole comparators, transfer, dataset 862, package_sphere_matched' + OTHER_ERRORS,
         *with_the_summary_of_the_comparators(
             in_csv(COMPARATORS_CSV, group(*TRANSFER_GROUP), 'position_error_mm_mne_limit', plus(5e-9)))),
    case('an error with the limit of SciPy that is not the error of the adaptation experiment',
         'dipole comparators, adaptation, dataset 862, three_shell_sphere_on_bem' + OTHER_ERRORS,
         *with_the_summary_of_the_comparators(
             in_csv(COMPARATORS_CSV, last_window('adaptation', '862', 'three_shell_sphere_on_bem'),
                    'position_error_mm_mne_limit', plus(-5e-9)))),
    case('an error with the limit of SciPy that is not the error of the split study',
         'dipole comparators, split_study, dataset 862, region' + OTHER_ERRORS,
         *with_the_summary_of_the_comparators(
             in_csv(COMPARATORS_CSV, smallest_where('position_error_mm_mne_limit', experiment='split_study',
                                                    dataset_seed='862', case='region'),
                    'position_error_mm_mne_limit', plus(5e-9)))),
    # Two windows of a group exchange their errors: every statistic of the group stays what the summary says.
    case('the errors of two windows with the limit of SciPy exchanged',
         'dipole comparators, adaptation, dataset 862, size_matched_sphere_on_bem' + OTHER_ERRORS,
         exchanged('adaptation', '862', 'size_matched_sphere_on_bem', 'position_error_mm_mne_limit')),
    # Both middle errors of a group change by 9e-10 mm, which is within the tolerance of a window, and the
    # summary follows: the group agrees with itself and, window by window, with the split study, but its median
    # no longer with the headline of the split study.
    case('a median with the limit of SciPy that is not the median of the split study',
         'dipole comparators, split study, position, mean of the dataset medians: recomputed',
         *with_the_summary_of_the_comparators(both_middle_errors('split_study', '863', 'position', 9e-10))),

    # ---- dipole comparators: every error is the distance of the fitted position from the true source position
    case('an error of the split study with the converged fit that is not the distance of the fitted position',
         'dipole comparators, split_study, dataset 864, position: the position errors with raised_limit'
         + NOT_THE_DISTANCES,
         *with_the_summary_of_the_comparators(
             in_csv(COMPARATORS_CSV, group('split_study', '864', 'position'), 'position_error_mm_raised_limit',
                    plus(5e-9)))),
    case('an error of the transfer experiment with the converged fit that is not the distance of the fitted position',
         'dipole comparators, transfer, dataset 862, package_sphere_matched: the position errors with raised_limit'
         + NOT_THE_DISTANCES,
         *with_the_summary_of_the_comparators(
             in_csv(COMPARATORS_CSV, last_window(*TRANSFER_GROUP), 'position_error_mm_raised_limit', plus(-5e-9)))),
    case('an error of the adaptation experiment with the converged fit that is not the distance of the fitted '
         'position',
         'dipole comparators, adaptation, dataset 862, size_matched_three_shell_sphere_on_bem: the position errors '
         'with raised_limit' + NOT_THE_DISTANCES,
         *with_the_summary_of_the_comparators(
             in_csv(COMPARATORS_CSV, smallest_where('position_error_mm_raised_limit', experiment='adaptation',
                                                    dataset_seed='862', case='size_matched_three_shell_sphere_on_bem'),
                    'position_error_mm_raised_limit', plus(5e-9)))),
    case('the errors of two windows with the converged fit exchanged',
         'dipole comparators, split_study, dataset 863, region: the position errors with raised_limit'
         + NOT_THE_DISTANCES, exchanged('split_study', '863', 'region', 'position_error_mm_raised_limit')),
    # The true source position of a fitted window moves by 0.01 mm in the eligibility file of the experiment.
    # The fits and their errors still are those of the experiment, but the errors are no longer the distances.
    case('a source position of the adaptation experiment that does not give the recorded errors',
         'dipole comparators, adaptation, dataset 862, default_sphere_on_bem: the position errors with mne_limit'
         + NOT_THE_DISTANCES, source_of_a_fitted_window(ADAPTATION, 'dipole_subset.csv', plus(0.01))),
    case('a source position of the transfer experiment that does not give the recorded errors',
         'dipole comparators, transfer, dataset 862, package_sphere_matched: the position errors with mne_limit'
         + NOT_THE_DISTANCES, source_of_a_fitted_window(TRANSFER, 'ecd_subset.csv', plus(0.01))),

    # ---- dipole comparators: the parameter fits end as the paper says
    # The sphere-fit record and the record of the dipole comparators change alike, so that they agree.
    case('a parameter fit at the limit of SciPy that stopped by itself, in both records',
         'dipole comparators, adaptation, dataset 862, three_shell_sphere_on_bem' + PAPER_FITS,
         *parameter_fit_in_both_records('three_shell', 'mne_limit', 'stopped_by_itself', True)),
    case('a parameter fit that ended one evaluation before the limit of SciPy, in both records',
         'dipole comparators, adaptation, dataset 862, size_matched_sphere_on_bem' + PAPER_FITS,
         *parameter_fit_in_both_records('size_matched', 'mne_limit', 'evaluations', 999)),
    case('a parameter fit that ended one evaluation after the limit of SciPy, in both records',
         'dipole comparators, adaptation, dataset 862, surface_fitted_three_shell_sphere_on_bem' + PAPER_FITS,
         *parameter_fit_in_both_records('surface_fitted_three_shell', 'mne_limit', 'evaluations', 1001)),
    case('a parameter fit with the raised limit that did not stop by itself, in both records',
         'dipole comparators, adaptation, dataset 862, size_matched_three_shell_sphere_on_bem' + PAPER_FITS,
         *parameter_fit_in_both_records('size_matched_three_shell', 'raised_limit', 'stopped_by_itself', False)),
    # The default sphere has the fits of the split study and of the transfer experiment: the first of them stops.
    case('a parameter fit of the default sphere with the raised limit that did not stop by itself, in both records',
         'dipole comparators, split_study, dataset 862, position' + PAPER_FITS,
         *parameter_fit_in_both_records('default', 'raised_limit', 'stopped_by_itself', False)),

    # ---- proposed: records that only the named check refuses
    # The file of the experiment changes and the record of the dipole comparators does not: its errors still are
    # the distances of its positions, so only the comparison with the recorded errors can notice.
    case('the recorded errors of two windows of the transfer experiment exchanged',
         'dipole comparators, transfer, dataset 862, package_sphere_matched' + OTHER_ERRORS,
         exchanged_rows(f'{TRANSFER}/862/ecd_subset.csv', 'ecd_package_sphere_matched_error_mm')),
    case('the recorded errors of two windows of the adaptation experiment exchanged',
         'dipole comparators, adaptation, dataset 862, size_matched_sphere_on_bem' + OTHER_ERRORS,
         exchanged_rows(f'{ADAPTATION}/862/dipole_subset.csv', 'dipole_size_matched_sphere_on_bem_error_mm')),
    # The run with the factor 0 stays and the ranges of the record follow: nothing else is wrong with the record,
    # and only the list of the expected runs can notice.
    case('a second unchanged run under another kernel under the number of a trial, the ranges following',
         whole(NOT_THE_RUNS + 'Missing: none. Not expected: kernel Nehalem, none 1 with raised_limit'),
         *with_the_ranges_of_the_small_runs(*in_both_small_files(
             'Nehalem', 'none', 'raised_limit', 0, copy=True, objective_scale_minus_one_in_eps='1'))),
    case('a second unchanged run of the main process under the number of a trial, the ranges following',
         whole(NOT_THE_RUNS + 'Missing: none. Not expected: kernel Haswell, none 2 with mne_limit'),
         *with_the_ranges_of_the_small_runs(*in_both_small_files(
             'Haswell', 'none', 'mne_limit', 0, copy=True, objective_scale_minus_one_in_eps='2'))),
    case('dipole comparators with the boundary-element fit of the transfer experiment as one more group',
         REPORTED_FITS, *boundary_element_fit_as_a_group()),
    case('dipole comparators with the boundary-element fit of the transfer experiment in place of the sphere fit',
         REPORTED_FITS, *boundary_element_fit_as_a_group(replace=True)),
    # The recorded errors and the errors of the dipole comparators change alike, so that the two records agree:
    # only the fit with the limit of SciPy no longer has the distances of its positions.
    case('the errors of two windows of the transfer experiment exchanged in both records',
         'dipole comparators, transfer, dataset 862, package_sphere_matched: the position errors with mne_limit'
         + NOT_THE_DISTANCES,
         exchanged_rows(f'{TRANSFER}/862/ecd_subset.csv', 'ecd_package_sphere_matched_error_mm'),
         exchanged(*TRANSFER_GROUP, 'position_error_mm_mne_limit')),
    # The value comes from another run of the same sphere fit, as in the two cases for the other cells.
    case('an unchanged run of the main process that records the median change of another run',
         SMALL + 'Haswell, none 0 with mne_limit' + NO_CHANGE,
         with_cell_of(SMALL_RUNS, small_run('Haswell', 'dipole_fits', 'mne_limit', 1),
                      small_run('Haswell', 'none', 'mne_limit', 0), 'fitted_position_change_median_mm')),
    # The ranges of the record count the run twice as well, so that only the number of the runs can notice.
    case('a run with the small final radius that the list holds twice, the ranges following', whole(OTHER_RUNS),
         *with_the_ranges_of_the_small_runs(
             with_row_twice(SMALL_RUNS, small_run('Haswell', 'dipole_fits', 'mne_limit', 2)))),

    # ---- split study: a record does not leave out a stratum for which the dataset has windows
    # Every method of the panel lacks the stratum, so that the strata of the methods agree and every recorded
    # number is right. Without the check the table of the strata counted one dataset fewer without a word.
    case('a stratum that every method of a dataset lacks although the dataset has its windows',
         whole('split study, dataset 864, region' + NO_STRATA + 'below_montage'),
         in_the_strata_of_every_method(summary(SPLIT, '864', 'region'), 'full', without('below_montage'))),
    case('a stratum that every method lacks on the dipole-fit subset although the subset has its windows',
         whole('split study, dataset 863, position, ecd_subset' + NO_STRATA + 'population_1'),
         in_the_strata_of_every_method(summary(SPLIT, '863', 'position'), 'ecd_subset', without('population_1'))),
    case('two strata that every method of a dataset lacks',
         whole('split study, dataset 862, position' + NO_STRATA + 'snr_10_15, within_montage'),
         in_the_strata_of_every_method(summary(SPLIT, '862', 'position'), 'full',
                                       without('snr_10_15', 'within_montage'))),
    case('every stratum of the dipole-fit subset of a dataset left out',
         'split study, dataset 862, region, ecd_subset' + NO_STRATA,
         in_the_strata_of_every_method(summary(SPLIT, '862', 'region'), 'ecd_subset', lambda strata: {})),
    # One dataset writes a label in another way. Its own numbers are right under that label; the first dataset
    # of the study then lacks a stratum that the study has.
    case('a stratum that one dataset records under another spelling of its label',
         whole('split study, dataset 862, position' + NO_STRATA + 'snr_10.0_15.0'),
         in_the_strata_of_every_method(summary(SPLIT, '863', 'position'), 'full',
                                       renamed('snr_10_15', 'snr_10.0_15.0'))),

    # The same for every dataset and both panels: none is left out of the examination.
    *missing_stratum_cases(),
    # A stratum that holds a single window of the dataset counts like any other.
    case('a stratum with a single window that every method of a dataset lacks',
         whole('split study, dataset 862, position' + NO_STRATA + 'distance_10_20'),
         in_the_strata_of_every_method(summary(SPLIT, '862', 'position'), 'full', without('distance_10_20'))),
    # The missing strata are listed in the order in which the datasets record them, not in that of their names.
    case('two strata that every method of a dataset lacks, listed in the order of the records',
         whole('split study, dataset 863, region' + NO_STRATA + 'radius_1_30_45, population_0'),
         in_the_strata_of_every_method(summary(SPLIT, '863', 'region'), 'full',
                                       without('radius_1_30_45', 'population_0'))),
    # The strata of the study are those that any dataset records in either panel. One panel of one dataset holds
    # a stratum once more under another spelling, with the same numbers: every other panel with such windows then
    # lacks a stratum of the study, and the first of them is named.
    case('a second spelling of a stratum on all windows of the first dataset only',
         whole('split study, dataset 862, position, ecd_subset' + NO_STRATA + 'snr_10.0_15.0'),
         in_the_strata_of_every_method(summary(SPLIT, '862', 'position'), 'full',
                                       with_entry_as('snr_10_15', 'snr_10.0_15.0'))),
    case('a second spelling of a stratum on the dipole-fit subset of the last dataset only',
         whole('split study, dataset 862, position' + NO_STRATA + 'snr_10.0_15.0'),
         in_the_strata_of_every_method(summary(SPLIT, '864', 'region'), 'ecd_subset',
                                       with_entry_as('snr_10_15', 'snr_10.0_15.0'))),

    # ---- split study: the study-level summary, from which Figure 5 is drawn, holds the dataset summaries
    case('a number of the study-level summary that the dataset summary does not have', whole(OTHER_STUDY_SUMMARY),
         in_json(STUDY_SUMMARY, ['863', 'region', 'full', 'knn1', 'median_mm'], plus(1e-9))),
    case('a number of a dataset summary within the tolerance that the study-level summary does not have',
         whole(OTHER_STUDY_SUMMARY),
         in_json(summary(SPLIT, '864', 'position'), ['ecd_subset', 'mlp_862', 'strata', 'snr_15_20', 'mean_mm'],
                 plus(5e-11))),
    case('a study-level summary without one dataset', whole(OTHER_STUDY_SUMMARY),
         in_json(STUDY_SUMMARY, ['864', 'region'], DROP)),
    # The comparison leaves out no dataset and no part of a summary: a panel of the first dataset, a fit record
    # and a count of training positions, which no other check examines.
    case('a number of the study-level summary of the first dataset that its summary does not have',
         whole(OTHER_STUDY_SUMMARY),
         in_json(STUDY_SUMMARY, ['862', 'position', 'ecd_subset', 'dipole', 'p90_mm'], plus(1e-9))),
    case('a fit record of the study-level summary that the dataset summary does not have',
         whole(OTHER_STUDY_SUMMARY),
         in_json(STUDY_SUMMARY, ['862', 'region', 'fits', 'mlp_863', 'best_epoch'], plus(1))),
    case('a count of training positions of the study-level summary that the dataset summary does not have',
         whole(OTHER_STUDY_SUMMARY),
         in_json(STUDY_SUMMARY, ['863', 'position', 'observed_split_composition', 'train', 0, 'distinct_positions'],
                 plus(1))),
    # The study-level summary holds nothing that the dataset summaries do not hold: no dataset, no policy and
    # no entry of a summary.
    case('a study-level summary with one dataset more', whole(OTHER_STUDY_SUMMARY),
         in_json(STUDY_SUMMARY, [], with_entry_as('864', '865'))),
    case('a study-level summary with one policy more for a dataset', whole(OTHER_STUDY_SUMMARY),
         in_json(STUDY_SUMMARY, ['863'], with_entry_as('region', 'region_again'))),
    case('a study-level summary with one stratum more for one method', whole(OTHER_STUDY_SUMMARY),
         in_json(STUDY_SUMMARY, ['864', 'position', 'full', 'knn10', 'strata'],
                 with_entry_as('snr_15_20', 'snr_25_30'))),
    # The experiment script writes the study-level summary from the dataset summaries as it read them, so the
    # numbers are equal to the last digit: the next number that the machine can write is another one.
    case('a number of the study-level summary that differs in its last digit', whole(OTHER_STUDY_SUMMARY),
         in_json(STUDY_SUMMARY, ['863', 'position', 'full', 'mlp_pooled', 'strata', 'distance_5_10', 'median_mm'],
                 lambda value: float(np.nextafter(value, np.inf)))),
    # The study-level summary is examined before the headline of the study.
    case('a study-level summary and a headline that both differ from the datasets', whole(OTHER_STUDY_SUMMARY),
         in_json(STUDY_SUMMARY, ['863', 'region', 'full', 'knn1', 'median_mm'], plus(1e-9)),
         in_json(f'{SPLIT}/headline.json', ['region', 'full', 'knn1', 'mean_median_mm'], plus(0.01))),

    # ---- patch study: every method and every dataset has the same strata
    case('a patch stratum that one lookup method lacks',
         whole('patch study, dataset 863, position, nearest_training_patch' + OTHER_STRATA),
         in_json(summary(PATCH, '863', 'position'), ['methods', 'nearest_training_patch', 'strata', 'snr_10_15'],
                 DROP)),
    case('the patch strata of one fitted network in another order',
         whole('patch study, dataset 864, region, mlp_863' + OTHER_STRATA),
         in_json(summary(PATCH, '864', 'region'), ['methods', 'mlp_863', 'strata'], in_the_opposite_order)),
    case('a patch stratum that every method of a dataset lacks',
         whole('patch study, dataset 863, position' + OTHER_PATCH_STRATA),
         in_the_strata_of_every_method(summary(PATCH, '863', 'position'), 'methods', without('snr_10_15'))),
    case('a patch stratum that every method of the last dataset lacks',
         whole('patch study, dataset 864, region' + OTHER_PATCH_STRATA),
         in_the_strata_of_every_method(summary(PATCH, '864', 'region'), 'methods', without('diameter_25_40'))),
    # The first dataset gives the strata of the table: without one of them the next dataset has one too many.
    case('a patch stratum that every method of the first dataset lacks',
         whole('patch study, dataset 862, region' + OTHER_PATCH_STRATA),
         in_the_strata_of_every_method(summary(PATCH, '862', 'position'), 'methods', without('snr_10_15'))),
    case('the patch strata of every method of a dataset in another order',
         whole('patch study, dataset 863, region' + OTHER_PATCH_STRATA),
         in_the_strata_of_every_method(summary(PATCH, '863', 'region'), 'methods', in_the_opposite_order)),

    # One stratum of one method stands under another label, the first or the last of its strata: as many strata
    # as the pooled network has, every other label in its place, and the same windows under the new label.
    case('the first patch stratum of one lookup method under another label',
         whole('patch study, dataset 862, region, ten_nearest_training_patches' + OTHER_STRATA),
         in_json(summary(PATCH, '862', 'region'), ['methods', 'ten_nearest_training_patches', 'strata'],
                 renamed('diameter_10_25', 'diameter_10_25.0'))),
    case('the last patch stratum of one fitted network under another label',
         whole('patch study, dataset 864, position, mlp_862' + OTHER_STRATA),
         in_json(summary(PATCH, '864', 'position'), ['methods', 'mlp_862', 'strata'],
                 renamed('snr_20_25.001', 'snr_20_25.0010'))),
    # The same in every method of a dataset: its methods agree, and only the first dataset can tell.
    case('the first patch stratum of every method of a dataset under another label',
         whole('patch study, dataset 864, position' + OTHER_PATCH_STRATA),
         in_the_strata_of_every_method(summary(PATCH, '864', 'position'), 'methods',
                                       renamed('diameter_10_25', 'diameter_10_25.0'))),
    case('the last patch stratum of every method of a dataset under another label',
         whole('patch study, dataset 863, region' + OTHER_PATCH_STRATA),
         in_the_strata_of_every_method(summary(PATCH, '863', 'region'), 'methods',
                                       renamed('snr_20_25.001', 'snr_20_25.0010'))),

    # ---- adaptation experiment: the arms of the datasets, and the dataset of a single network
    case('an arm that one dataset has and the others lack', whole(OTHER_ARMS),
         in_json(ADAPTATION_SUMMARY, ['by_dataset_seed', '863', 'arms'], with_entry_as('default', 'default_again'))),
    case('an arm that the last dataset lacks', whole(OTHER_ARMS),
         in_json(ADAPTATION_SUMMARY, ['by_dataset_seed', '864', 'arms', 'scratch_all'], DROP)),
    case('an arm that the first dataset lacks', whole(OTHER_ARMS),
         in_json(ADAPTATION_SUMMARY, ['by_dataset_seed', '862', 'arms', 'three_shell'], DROP)),
    case('a row of a single network that names another dataset',
         whole('adaptation, dataset 862' + OTHER_DATASET + '862'),
         in_json(ADAPTATION_SUMMARY, ['by_dataset_seed', '862', 'finetune', network('scratch', '1000', '862'),
                                      'dataset_seed'], to(864))),
    case('the last row of the single networks of a dataset that names another dataset',
         whole('adaptation, dataset 864' + OTHER_DATASET + '864'),
         in_json(ADAPTATION_SUMMARY, ['by_dataset_seed', '864', 'finetune', len(SINGLE_NETWORKS) - 1,
                                      'dataset_seed'], to(863))),

    # One arm of one dataset stands under another name: as many arms as the other datasets have. The first case
    # takes the arm that stands last in the order of the names, the second the one that stands first.
    case('the last arm of one dataset in the order of the names under another name', whole(OTHER_ARMS),
         in_json(ADAPTATION_SUMMARY, ['by_dataset_seed', '863', 'arms'], renamed('three_shell', 'three_shells'))),
    case('the first arm of one dataset in the order of the names under another name', whole(OTHER_ARMS),
         in_json(ADAPTATION_SUMMARY, ['by_dataset_seed', '864', 'arms'], renamed('default', 'Default'))),
    # The first row of the second dataset, a fine-tuned network: no row, no kind of network and no dataset is
    # left out of the comparison.
    case('the first row of the single networks of a dataset that names another dataset',
         whole('adaptation, dataset 863' + OTHER_DATASET + '863'),
         in_json(ADAPTATION_SUMMARY, ['by_dataset_seed', '863', 'finetune', 0, 'dataset_seed'], to(862))),
    # The dataset of a row is the one of its record digit by digit, as its model seed is.
    case('a row of a single network that names its dataset with a zero before it',
         whole('adaptation, dataset 864' + OTHER_DATASET + '864'),
         in_json(ADAPTATION_SUMMARY, ['by_dataset_seed', '864', 'finetune', network('finetuned', '1000', '863'),
                                      'dataset_seed'], to('0864'))),
    # A row whose arm is no text names no network of an arm: the driver says so in its own words.
    case('a single network whose arm is not a text',
         whole('adaptation, dataset 862: the single networks are not those of the arms'),
         in_json(ADAPTATION_SUMMARY, ['by_dataset_seed', '862', 'finetune', network('scratch', '100', '862'), 'arm'],
                 to(None))),
    # A row without the field is told in the same words.
    case('a row of a single network without its dataset',
         whole('adaptation, dataset 864' + OTHER_DATASET + '864'),
         in_json(ADAPTATION_SUMMARY, ['by_dataset_seed', '864', 'finetune', network('scratch', 'all', '862'),
                                      'dataset_seed'], DROP)),
    # A row can fail both checks: the networks of the arms are examined first.
    case('a row that is no network of an arm and names another dataset',
         whole('adaptation, dataset 863: the single networks are not those of the arms'),
         in_json(ADAPTATION_SUMMARY, ['by_dataset_seed', '863', 'finetune', 0],
                 lambda row: dict(row, arm=None, dataset_seed=862))),
    # Other spellings of the dataset of the record are not its seed digit by digit.
    case('a row of a single network that names its dataset with a blank before it',
         whole('adaptation, dataset 862' + OTHER_DATASET + '862'),
         in_json(ADAPTATION_SUMMARY, ['by_dataset_seed', '862', 'finetune', network('finetuned', 'all', '862'),
                                      'dataset_seed'], to(' 862'))),
    case('a row of a single network that names its dataset with a decimal point',
         whole('adaptation, dataset 863' + OTHER_DATASET + '863'),
         in_json(ADAPTATION_SUMMARY, ['by_dataset_seed', '863', 'finetune', network('scratch', '100', '863'),
                                      'dataset_seed'], to(863.0))),

    # ---- sphere-fit record: a library whose kernel has no name
    case('a library of the main process whose kernel has no name', whole(OTHER_MAIN_KERNEL),
         in_json(SPHERE_SUMMARY, ['library_kernels', 'of_the_parent_process', 0, 'kernel'], to(None))),
]

# Differences that the driver accepts: each lies within the tolerance of a comparison, and ten times the
# difference is one of the cases above. The entries after the small differences are other records that are as
# valid as the unchanged one. The second trial of the runs with the small final radius has the factor 8, so
# that the trials of the runs must be put in the order of their numbers before they equal those of the record.
# The summary of the dipole comparators lists the spheres of the adaptation experiment in the opposite order:
# the fits that the paper reports may stand in any order. So may the two evaluation limits of that summary and
# the two parameter fits of a group, because the driver takes the two fits by their names. Every method of a
# panel of the split study has its strata in the opposite order: the strata of a method must be those of the
# pooled network, in whatever order that one has them. And the single networks of a dataset of the adaptation
# experiment stand in the opposite order: each of them must be there once, wherever it stands.
TOLERATED = [
    pytest.param([in_csv(f'{SPHERE}/perturbation.csv',
                         middle('moment_relative_error_median', perturbed_fit='sphere_fit', sphere_fit='mne_limit'),
                         'moment_relative_error_median', plus(5e-13))],
                 id='the median moment error of a trial, by 5e-13'),
    pytest.param([in_csv(SMALL_RUNS, middle('position_error_p90_mm', sphere_fit='mne_limit'),
                         'position_error_p90_mm', plus(5e-13))],
                 id='an error statistic of a run with the small final radius, by 5e-13 mm'),
    pytest.param([in_csv(SMALL_RUNS, middle('fitted_position_change_median_mm', changed_objective='dipole_fits',
                                            sphere_fit='mne_limit'),
                         'fitted_position_change_median_mm', plus(5e-10))],
                 id='the change of the positions of a run with the small final radius, by 5e-10 mm'),
    pytest.param([in_json(COMPARATORS_SUMMARY, ['adaptation', '862', 'three_shell_sphere_on_bem',
                                                'fitted_position_change_mm', 'p90'], plus(5e-10))],
                 id='the change of the fitted positions of a group, by 5e-10 mm'),
    pytest.param([in_csv(f'{TRANSFER}/862/ecd_subset.csv', 5, 'ecd_package_sphere_matched_y_mm', plus(5e-10))],
                 id='a recorded fit of the transfer experiment, by 5e-10 mm'),
    pytest.param([in_json(COMPARATORS_SUMMARY, ['transfer', '862', 'package_sphere_matched', 'parameter_fit',
                                                'raised_limit', 'objective'], times(1 + 5e-10))],
                 id='the objective of a parameter fit, by five parts in ten billion'),
    pytest.param(list(with_the_summary_of_the_comparators(
        in_csv(COMPARATORS_CSV, group(*TRANSFER_GROUP), 'position_error_mm_mne_limit', plus(5e-10)))),
        id='an error with the limit of SciPy, by 5e-10 mm'),
    pytest.param(list(with_the_summary_of_the_comparators(
        in_csv(COMPARATORS_CSV, group('split_study', '864', 'position'), 'position_error_mm_raised_limit',
               plus(5e-10)))),
        id='an error with the converged fit, by 5e-10 mm'),
    # The study-level summary holds the dataset summaries, so the same number changes in both files.
    pytest.param([in_json(summary(SPLIT, '862', 'position'),
                          ['ecd_subset', 'dipole', 'strata', 'population_0', 'p90_mm'], plus(5e-11)),
                  in_json(STUDY_SUMMARY,
                          ['862', 'position', 'ecd_subset', 'dipole', 'strata', 'population_0', 'p90_mm'],
                          plus(5e-11))],
                 id='a stratum 90th percentile of the dipole fit, by 5e-11 mm'),
    pytest.param([in_json(summary(SPLIT, '863', 'region'), ['full', 'mlp_863', 'strata', 'within_montage', 'mean_mm'],
                          plus(-5e-11)),
                  in_json(STUDY_SUMMARY, ['863', 'region', 'full', 'mlp_863', 'strata', 'within_montage', 'mean_mm'],
                          plus(-5e-11))],
                 id='a stratum mean of one fitted network, by 5e-11 mm'),
    pytest.param([in_json(summary(PATCH, '862', 'position'),
                          ['methods', 'ten_nearest_training_patches', 'strata', 'snr_15_20', 'p90_mm'], plus(5e-11))],
                 id='a patch stratum statistic of a lookup method, by 5e-11 mm'),
    pytest.param([in_json(ADAPTATION_SUMMARY,
                          ['by_dataset_seed', '863', 'finetune', network('scratch', '1000', '863'), 'p90_mm'],
                          plus(5e-11))],
                 id='a statistic of a single network, by 5e-11 mm'),
    pytest.param(list(trial_as(2, 8)), id='the second trial of the runs with the small final radius as the eighth'),
    pytest.param([in_json(COMPARATORS_SUMMARY, ['adaptation', '862'],
                          lambda groups: dict(reversed(list(groups.items()))))],
                 id='the spheres of the dipole comparators in the opposite order'),
    pytest.param([in_json(COMPARATORS_SUMMARY, ['evaluation_limits'],
                          lambda limits: dict(reversed(list(limits.items()))))],
                 id='the two evaluation limits of the dipole comparators in the opposite order'),
    pytest.param([in_json(SPHERE_SUMMARY, ['evaluation_limits', 'mne_limit'], to(999)),
                  in_json(COMPARATORS_SUMMARY, ['evaluation_limits', 'mne_limit'], to(999)),
                  *[edit for sphere in mini_evidence.VARIANTS
                    for edit in parameter_fit_in_both_records(sphere, 'mne_limit', 'evaluations', 999)]],
                 id='another limit of SciPy in both records, with every fit at the limit ending there'),
    pytest.param([in_json(COMPARATORS_SUMMARY, ['adaptation', '862', 'three_shell_sphere_on_bem', 'parameter_fit'],
                          in_the_opposite_order)],
                 id='the two parameter fits of a group of the dipole comparators in the opposite order'),
    pytest.param([in_the_strata_of_every_method(summary(SPLIT, '863', 'region'), 'ecd_subset',
                                                in_the_opposite_order)],
                 id='the strata of every method on the dipole-fit subset in the opposite order'),
    pytest.param([in_json(ADAPTATION_SUMMARY, ['by_dataset_seed', '863', 'finetune'], in_the_opposite_order)],
                 id='the single networks of a dataset in the opposite order'),
    # The experiment script writes the size of a network as text and its seeds as numbers. The other way round,
    # the rows name the same networks of the same dataset.
    pytest.param([in_json(ADAPTATION_SUMMARY, ['by_dataset_seed', '862', 'finetune',
                                               network('finetuned', '100', '863'), 'size'], to(100)),
                  in_json(ADAPTATION_SUMMARY, ['by_dataset_seed', '862', 'finetune',
                                               network('scratch', '1000', '862'), 'dataset_seed'], to('862'))],
                 id='the size of a single network as a number and its dataset as text'),
    pytest.param([in_json(ADAPTATION_SUMMARY, ['by_dataset_seed', '863', 'arms'], in_the_opposite_order)],
                 id='the arms of one dataset of the adaptation experiment in the opposite order'),
    # The study-level summary must hold the numbers of the dataset summaries, in whatever way they are written:
    # a count with a decimal point is the same number.
    pytest.param([in_json(STUDY_SUMMARY, ['862', 'position', 'full', 'knn1', 'windows'], float)],
                 id='a count of the study-level summary written with a decimal point'),
]


@pytest.mark.parametrize('edits, message', CORRUPTIONS)
def test_changed_value_stops_the_driver_with_a_message_that_names_it(miniature, copied, tmp_path, edits, message):
    changed = [edit.file for edit in edits]
    # A message is a part of the driver's message unless the case gives it whole.
    pattern = f'^{re.escape(message)}$' if isinstance(message, Whole) else re.escape(message)
    try:
        for edit in edits:
            edit(copied)
        with pytest.raises(ValueError, match=pattern):
            run_driver(copied, tmp_path / 'tables', miniature.run.functions, changed)
    finally:
        for file in changed:
            shutil.copyfile(miniature.evidence / file, copied / file)
    assert not (tmp_path / 'tables').exists()           # nothing is written for evidence that does not agree


@pytest.mark.parametrize('edits', TOLERATED)
def test_difference_within_the_tolerance_does_not_stop_the_driver(miniature, copied, tmp_path, edits):
    changed = [edit.file for edit in edits]
    try:
        for edit in edits:
            edit(copied)
        assert file_hashes(copied) != miniature.hashes
        run_driver(copied, tmp_path / 'tables', miniature.run.functions, changed)
    finally:
        for file in changed:
            shutil.copyfile(miniature.evidence / file, copied / file)
    assert sorted(path.name for path in (tmp_path / 'tables').iterdir()) == ['Revision_Tables.md',
                                                                             'revision_tables.json']


def test_kernel_of_the_main_process_is_the_one_of_the_record_whatever_the_order_of_the_runs(miniature, copied,
                                                                                            tmp_path):
    """The list of the runs with the small final radius in the opposite order is as valid a record."""
    try:
        in_rows(SMALL_RUNS, lambda rows: rows[::-1])(copied)
        run_driver(copied, tmp_path / 'tables', miniature.run.functions, [SMALL_RUNS])
    finally:
        shutil.copyfile(miniature.evidence / SMALL_RUNS, copied / SMALL_RUNS)
    report = json.loads((tmp_path / 'tables/revision_tables.json').read_text(encoding='utf-8'))
    small, unchanged = report['sphere_fit']['small_tolerance'], miniature.report['sphere_fit']['small_tolerance']
    # The tables list the runs as the file does: a run under another kernel comes first and last.
    assert small['runs'] == unchanged['runs'][::-1]
    assert (small['runs'][0]['kernel'], small['runs'][-1]['kernel']) == ('Nehalem', 'Haswell')
    assert small['kernel_of_the_main_process'] == 'Haswell'
    # The ranges of a sphere fit do not depend on the order; the sphere fits stand in the order of their first run.
    assert small['ranges'] == unchanged['ranges'] and list(small['ranges']) == ['raised_limit', 'mne_limit']
    assert small['steps'] == [1, 2]
    assert {key: value for key, value in report['sphere_fit'].items() if key != 'small_tolerance'} == {
        key: value for key, value in miniature.report['sphere_fit'].items() if key != 'small_tolerance'}
    # The Markdown file prints no run and takes the sphere fits in the order of the arms: it stays as it is, with
    # the kernel of the main process that the record names, which is not the kernel of the first run of the list.
    assert (tmp_path / 'tables/Revision_Tables.md').read_text(encoding='utf-8').splitlines() == miniature.lines
    assert 'Other kernels of the linear-algebra library (kernel of the main run: Haswell):' in miniature.lines


def test_kernel_of_the_main_process_is_named_as_the_record_names_it(miniature, copied, tmp_path):
    """Another name of the kernel of the main process, everywhere in the sphere-fit record, is as valid a record."""
    def renamed(file):
        def edit(root):
            content = (root / file).read_bytes().decode().replace('Haswell', 'Zen')
            if file.endswith('.json'):
                data = json.loads(content)
                for entry in data['small_tolerance']['ranges'].values():
                    entry['kernels'] = sorted(entry['kernels'])
                content = json.dumps(data, indent=1) + '\n'
            (root / file).write_bytes(content.encode())

        edit.file = file
        return edit

    edits = [renamed(SMALL_RUNS), renamed(SMALL_DIPOLES), renamed(SPHERE_SUMMARY)]
    try:
        for edit in edits:
            edit(copied)
        run_driver(copied, tmp_path / 'tables', miniature.run.functions, [edit.file for edit in edits])
    finally:
        for edit in edits:
            shutil.copyfile(miniature.evidence / edit.file, copied / edit.file)
    report = json.loads((tmp_path / 'tables/revision_tables.json').read_text(encoding='utf-8'))
    small = report['sphere_fit']['small_tolerance']
    # The name stands last in the order of the alphabet: it is the name of the record, and no kernel that
    # was found among the runs by their order.
    assert small['kernel_of_the_main_process'] == 'Zen'
    assert [entry['kernels'] for entry in small['ranges'].values()] == [['Nehalem', 'Sandybridge', 'Zen']] * 2
    assert {run['kernel'] for run in small['runs'] if run['fitted_position_change_median_mm'] == ''} == {'Zen'}


# Records that run_sphere_fit_convergence.py writes by its own rule for other libraries of linear algebra in the
# main process: the name that it gives to the kernel of the main process, and the kernels that the libraries
# report, in the order of the record. A library that is not OpenBLAS reports no kernel, written as 'None'. The
# script joins the names in the order in which Python sorts them: a capital letter comes before a small one.
# With eight kernels no order of a set gives the name by chance.
KERNELS_OF_THE_MAIN_PROCESS = [
    pytest.param('unknown', [], id='no library'),
    pytest.param('unknown', [''], id='one library whose kernel has an empty name'),
    pytest.param('Haswell+Zen', ['Haswell', 'Zen'], id='two libraries with two kernels'),
    pytest.param('Haswell+Zen', ['Zen', 'Haswell', 'Zen'], id='three libraries with two kernels in another order'),
    pytest.param('Haswell+None', ['Haswell', 'None'], id='a library without a kernel next to another one'),
    pytest.param('None', ['None'], id='one library without a kernel'),
    pytest.param('Haswell+armv8', ['armv8', 'Haswell'], id='a kernel with a small first letter'),
    pytest.param('Atom+Core2+Haswell+Penryn+Prescott+Skylakex+Steamroller+Zen',
                 ['Zen', 'Steamroller', 'Skylakex', 'Prescott', 'Penryn', 'Haswell', 'Core2', 'Atom'],
                 id='eight libraries with eight kernels'),
    pytest.param('+Haswell', ['Haswell', ''], id='a library whose kernel has an empty name next to another one'),
    pytest.param(' Zen+Haswell', ['Haswell', ' Zen'], id='a kernel whose name begins with a blank'),
    # A record that holds no text for the kernel of a library is read as the experiment script writes such a
    # library: None.
    pytest.param('Haswell+None', ['Haswell', None], id='a library whose kernel the record leaves empty'),
]


@pytest.mark.parametrize('name, kernels', KERNELS_OF_THE_MAIN_PROCESS)
def test_kernel_of_the_main_process_is_named_by_the_rule_of_the_experiment_script(miniature, copied, tmp_path, name,
                                                                                  kernels):
    """A record that names the kernel of the main process as the experiment script does is as valid a record.

    The script joins the kernels that the loaded libraries report, each once and in the order of the alphabet,
    with '+', and writes 'unknown' if no library is reported. The tables name the kernel as the record does.
    """
    edits = with_the_main_process_as(name, kernels)
    try:
        for edit in edits:
            edit(copied)
        run_driver(copied, tmp_path / 'tables', miniature.run.functions, [edit.file for edit in edits])
    finally:
        for edit in edits:
            shutil.copyfile(miniature.evidence / edit.file, copied / edit.file)
    report = json.loads((tmp_path / 'tables/revision_tables.json').read_text(encoding='utf-8'))
    fit, unchanged = report['sphere_fit'], miniature.report['sphere_fit']
    small, listed = fit['small_tolerance'], sorted([name, 'Nehalem', 'Sandybridge'])
    assert small['kernel_of_the_main_process'] == name
    assert [entry['kernel'] for entry in fit['library_kernels']['of_the_parent_process']] == kernels
    assert [entry['kernels'] for entry in small['ranges'].values()] == [listed] * 2
    assert [run['kernel'] for run in small['runs']] == [name] * 10 + ['Sandybridge'] * 2 + ['Nehalem'] * 2
    # Everything else is as in the unchanged tables.
    assert small['runs'] == [dict(run, kernel=name if run['kernel'] == 'Haswell' else run['kernel'])
                             for run in unchanged['small_tolerance']['runs']]
    for key in ('arms', 'perturbation', 'spheres', 'all_runs', 'links'):
        assert fit[key] == unchanged[key], key
    assert fit['library_kernels']['arms'] == unchanged['library_kernels']['arms']
    # The Markdown file names the kernel in three lines, as the record names it: in a sentence, in the row of
    # the main run, and among the kernels of the runs with the small final radius.
    renamed_lines = {
        'Other kernels of the linear-algebra library (kernel of the main run: Haswell):':
            f'Other kernels of the linear-algebra library (kernel of the main run: {name}):',
        '| Kernels | Haswell, Nehalem, Sandybridge | Haswell, Nehalem, Sandybridge |':
            f"| Kernels | {', '.join(listed)} | {', '.join(listed)} |"}
    main_run = next(line for line in miniature.lines if line.startswith('| Haswell (main run) | '))
    renamed_lines[main_run] = f'| {name} (main run) | ' + main_run[len('| Haswell (main run) | '):]
    assert all(miniature.lines.count(line) == 1 for line in renamed_lines)
    lines = (tmp_path / 'tables/Revision_Tables.md').read_text(encoding='utf-8').splitlines()
    assert lines == [renamed_lines.get(line, line) for line in miniature.lines]


def test_record_without_a_library_and_without_another_kernel_is_accepted(miniature, copied, tmp_path):
    """Where no library of linear algebra is reported, the experiment script can run no other kernel either.

    It names the kernel of the main process 'unknown', lists every requested kernel as not available, and
    writes no run, no row and no entry for one: the record of such a machine.
    """
    edits = [*with_the_main_process_as('unknown', []), *without_the_other_kernel('Sandybridge'),
             *without_the_other_kernel('Nehalem')]
    try:
        for edit in edits:
            edit(copied)
        run_driver(copied, tmp_path / 'tables', miniature.run.functions, [edit.file for edit in edits])
    finally:
        for edit in edits:
            shutil.copyfile(miniature.evidence / edit.file, copied / edit.file)
    report = json.loads((tmp_path / 'tables/revision_tables.json').read_text(encoding='utf-8'))
    fit = report['sphere_fit']
    small = fit['small_tolerance']
    assert small['kernel_of_the_main_process'] == 'unknown'
    assert fit['library_kernels']['of_the_parent_process'] == [] and fit['library_kernels']['arms'] == {}
    assert set(fit['library_kernels']['not_available']) == {'Haswell', 'Sandybridge', 'Nehalem'}
    # The unchanged run and the runs with a changed objective, all of the main process.
    assert [(entry['runs'], entry['kernels']) for entry in small['ranges'].values()] == [(5, ['unknown'])] * 2
    assert {run['kernel'] for run in small['runs']} == {'unknown'} and len(small['runs']) == 10
    assert [runs['runs_with_a_dipole_recovery'] for runs in fit['all_runs'].values()] == [1 + 3 + 3] * 2
    lines = (tmp_path / 'tables/Revision_Tables.md').read_text(encoding='utf-8').splitlines()
    assert 'Other kernels of the linear-algebra library (kernel of the main run: unknown):' in lines
    assert '| Kernels | unknown | unknown |' in lines
    # The table of the kernels holds the row of the main run and no other.
    rows = table_rows(lines, '## Parameter fit of the sphere model in MNE-Python')
    assert [row[0] for row in rows[17:20]] == ['Kernel', 'unknown (main run)', 'Sphere']


def test_size_of_a_single_network_is_all_that_follows_the_kind_of_its_arm(miniature, copied, tmp_path):
    """An arm of single networks is named by their kind and their size, and the size may hold an underscore.

    The largest size has another name everywhere in the record of the adaptation experiment: in the arms, in the
    rows of the single networks and in the columns of the prediction files. It is as valid a record.
    """
    def renamed(file):
        def edit(root):
            content = (root / file).read_bytes().decode()
            for kind in ('finetuned', 'scratch'):
                content = content.replace(f'{kind}_all', f'{kind}_all_windows')
            (root / file).write_bytes(content.replace('"size": "all"', '"size": "all_windows"').encode())

        edit.file = file
        return edit

    edits = [renamed(ADAPTATION_SUMMARY),
             *[renamed(f'{ADAPTATION}/{seed}/adaptation_predictions.csv') for seed in SEEDS]]
    try:
        for edit in edits:
            edit(copied)
        run = run_driver(copied, tmp_path / 'tables', miniature.run.functions, [edit.file for edit in edits])
    finally:
        for edit in edits:
            shutil.copyfile(miniature.evidence / edit.file, copied / edit.file)
    report = json.loads((tmp_path / 'tables/revision_tables.json').read_text(encoding='utf-8'))
    adaptation, unchanged = report['adaptation'], miniature.report['adaptation']
    # The same numbers under the other names of the two arms.
    assert list(adaptation['bem_windows']) == [arm.replace('_all', '_all_windows') for arm in unchanged['bem_windows']]
    assert list(adaptation['bem_windows'])[-2:] == ['finetuned_all_windows', 'scratch_all_windows']
    assert list(adaptation['bem_windows'].values()) == list(unchanged['bem_windows'].values())
    assert [(row['size'], row['bem_training_windows']) for row in adaptation['finetuning']] == [
        ('100', [100]), ('1000', [1000]), ('all_windows', [298, 305, 310])]
    # Every single network was compared, the four of the renamed arms of each dataset under the new name.
    assert run.functions['adaptation_tables'].checks == COMPARISONS['adaptation_tables']
    assert sum('_all_windows, network ' in name for name in run.passed) == 3 * 4 * 3


def test_sizes_written_as_numbers_give_the_same_tables_of_the_adaptation_experiment(miniature, copied, tmp_path):
    """Every row of one size names it as a number: the driver reads a size as text, and the tables do not change."""
    def as_numbers(data):
        for record in data['by_dataset_seed'].values():
            for row in record['finetune']:
                if row['size'] == '100':
                    row['size'] = 100
        return data

    edit = in_json(ADAPTATION_SUMMARY, [], as_numbers)
    try:
        edit(copied)
        run_driver(copied, tmp_path / 'tables', miniature.run.functions, [edit.file])
    finally:
        shutil.copyfile(miniature.evidence / edit.file, copied / edit.file)
    report = json.loads((tmp_path / 'tables/revision_tables.json').read_text(encoding='utf-8'))
    # The numbers of training windows of every size are those of the unchanged record.
    assert [(row['size'], row['bem_training_windows']) for row in report['adaptation']['finetuning']] == [
        ('100', [100]), ('1000', [1000]), ('all', [298, 305, 310])]
    assert report['adaptation'] == miniature.report['adaptation']


def test_training_windows_of_a_size_are_those_of_every_row_of_the_size(miniature, copied, tmp_path):
    """The table of the sizes lists the training windows of every row of a size: both kinds and both model seeds.

    The experiment script fits the two networks of a size and model seed on the same windows. Here the network
    trained alone of the first model seed and the fine-tuned network of the second record other numbers.
    """
    def other_numbers(rows):
        rows[network('scratch', '100', '862')]['bem_training_windows'] = 99
        rows[network('finetuned', '100', '863')]['bem_training_windows'] = 101
        return rows

    edit = in_json(ADAPTATION_SUMMARY, ['by_dataset_seed', '863', 'finetune'], other_numbers)
    try:
        edit(copied)
        run_driver(copied, tmp_path / 'tables', miniature.run.functions, [edit.file])
    finally:
        shutil.copyfile(miniature.evidence / edit.file, copied / edit.file)
    report = json.loads((tmp_path / 'tables/revision_tables.json').read_text(encoding='utf-8'))
    assert [(row['size'], row['bem_training_windows']) for row in report['adaptation']['finetuning']] == [
        ('100', [99, 100, 101]), ('1000', [1000]), ('all', [298, 305, 310])]


def test_order_of_the_two_evaluation_limits_does_not_change_the_tables_of_the_dipole_comparators(miniature, copied,
                                                                                                tmp_path):
    """The summary of the dipole comparators with its two limits in the opposite order gives the same tables."""
    edit = in_json(COMPARATORS_SUMMARY, ['evaluation_limits'], in_the_opposite_order)
    try:
        edit(copied)
        run_driver(copied, tmp_path / 'tables', miniature.run.functions, [edit.file])
    finally:
        shutil.copyfile(miniature.evidence / edit.file, copied / edit.file)
    report = json.loads((tmp_path / 'tables/revision_tables.json').read_text(encoding='utf-8'))
    comparators, unchanged = report['comparators'], miniature.report['comparators']
    assert comparators == unchanged and list(comparators['evaluation_limits']) == ['raised_limit', 'mne_limit']
    # The two fits of a group and of a policy stand in the order of the paper, not in the order of the record.
    assert [list(group) for group in comparators['groups']] == [list(group) for group in unchanged['groups']]
    assert [list(entry) for entry in comparators['split_study_means'].values()] == [['mne_limit', 'raised_limit']] * 2
    assert (tmp_path / 'tables/Revision_Tables.md').read_text(encoding='utf-8').splitlines() == miniature.lines


@pytest.mark.parametrize('folder', [SPLIT, COUNT, PATCH, 'shared-position-v11', TRANSFER, ADAPTATION,
                                    'generation-benchmark-v11', SPHERE, COMPARATORS])
def test_experiment_without_its_provenance_record_stops_the_driver(miniature, copied, tmp_path, folder):
    record = f'{folder}/provenance.json'
    (copied / record).unlink()
    try:
        with pytest.raises(FileNotFoundError, match=f'{folder}.+provenance'):
            run_driver(copied, tmp_path / 'tables', miniature.run.functions, [record])
    finally:
        shutil.copyfile(miniature.evidence / record, copied / record)
    assert not (tmp_path / 'tables').exists()


@pytest.mark.parametrize('missing', ['--evidence', '--output-dir'])
def test_both_folders_must_be_named(miniature, tmp_path, monkeypatch, capsys, missing):
    folders = {'--evidence': str(miniature.evidence), '--output-dir': str(tmp_path / 'tables')}
    del folders[missing]
    (option, folder), = folders.items()
    monkeypatch.setattr(sys, 'argv', ['make_revision_tables.py', option, folder])
    with pytest.raises(SystemExit) as stop:
        tables.main()
    assert stop.value.code == 2 and f'the following arguments are required: {missing}' in capsys.readouterr().err
    assert not (tmp_path / 'tables').exists()


def test_existing_output_directory_is_refused(miniature, tmp_path, capsys):
    (tmp_path / 'tables').mkdir()
    with pytest.raises(SystemExit) as stop:
        run_driver(miniature.evidence, tmp_path / 'tables')
    assert stop.value.code == 2
    assert 'choose a new output directory' in capsys.readouterr().err
    assert list((tmp_path / 'tables').iterdir()) == []
