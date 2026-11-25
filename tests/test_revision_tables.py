"""The table driver: strata, pooling rules and record checks on hand-made data, and the archived tables."""

import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import make_revision_tables as tables

SEEDS = tables.SEEDS


def just_below(value):
    """The largest number that is smaller than ``value``."""
    return float(np.nextafter(value, -np.inf))


def test_driver_names_the_datasets_networks_and_methods_of_the_paper():
    assert tables.SEEDS == ('862', '863', '864') and tables.FITS == ('862', '863')
    assert tables.POLICIES == ('position', 'region') and tables.PANELS == ('full', 'ecd_subset')
    # The dipole fit exists on the subset only; each fitted network is also scored on its own.
    assert tables.panel_methods('full') == ['mlp_pooled', 'dictionary', 'knn1', 'knn10', 'training_mean',
                                           'mlp_862', 'mlp_863']
    assert tables.panel_methods('ecd_subset') == ['mlp_pooled', 'dictionary', 'knn1', 'knn10', 'dipole',
                                                 'training_mean', 'mlp_862', 'mlp_863']


# --------------------------------------------------------------------------- strata of the split and patch studies

def split_row(population, radius, snr=17.0, distance=3.0, below=0, position_id=0, direction=(1.0, 0.0, 0.0)):
    # On the x axis the radius is exact, which the rows on a band edge need.
    x, y, z = (radius * component for component in direction)
    return {'population': str(population), 'x_mm': repr(x), 'y_mm': repr(y), 'z_mm': repr(z),
            'snr_db': repr(snr), 'nearest_train_mm': repr(distance), 'below_montage': str(below),
            'position_id': str(position_id)}


def members(label, rows):
    return [index for index, keep in enumerate(tables.split_stratum_mask(label, rows)) if keep]


def test_band_is_closed_below_and_open_above():
    assert tables.band([4.999, 5.0, 9.999, 10.0], 5, 10).tolist() == [False, True, True, False]
    assert tables.band([1e9], 30, 'inf').tolist() == [True]
    # The edges may come as text, as the labels of the strata give them.
    assert tables.band([70.499, 70.5, 74.999, 75.0], '70.5', '75').tolist() == [False, True, True, False]


def test_select_keeps_the_rows_of_a_mask():
    assert tables.select(['a', 'b', 'c', 'd'], np.array([True, False, False, True])) == ['a', 'd']
    assert tables.select(['a', 'b'], [False, False]) == []


def test_split_strata_follow_population_radius_snr_distance_and_montage_cover():
    rows = [split_row(0, 70.499), split_row(0, 70.5), split_row(0, 74.999), split_row(0, 75.0),
            split_row(0, 79.5), split_row(0, 79.501),
            split_row(1, 30.0), split_row(1, 44.999), split_row(1, 45.0), split_row(1, 54.999),
            split_row(1, 55.0), split_row(1, 65.0005), split_row(1, 65.001),
            split_row(1, 72.0),                     # a subcortical row at a cortical radius
            # Every coordinate counts: these two lie at 73.5 and 31.5 mm, and no single coordinate does.
            split_row(0, 73.5, direction=(2 / 7, 3 / 7, 6 / 7)), split_row(1, 31.5, direction=(6 / 7, 3 / 7, 2 / 7))]
    assert members('all', rows) == list(range(16))
    assert members('population_0', rows) == [0, 1, 2, 3, 4, 5, 14]
    assert members('population_1', rows) == [6, 7, 8, 9, 10, 11, 12, 13, 15]
    assert members('radius_0_70.5_75', rows) == [1, 2, 14]
    assert members('radius_0_75_79.501', rows) == [3, 4]
    assert members('radius_1_30_45', rows) == [6, 7, 15]
    assert members('radius_1_45_55', rows) == [8, 9]
    assert members('radius_1_55_65.001', rows) == [10, 11]

    rows = [split_row(0, 72, snr=snr) for snr in (10.0, 14.999, 15.0, 19.999, 20.0, 25.0005)]
    assert members('snr_10_15', rows) == [0, 1]
    assert members('snr_15_20', rows) == [2, 3]
    assert members('snr_20_25.001', rows) == [4, 5]

    rows = [split_row(0, 72, distance=distance) for distance in (0.0, 4.999, 5.0, 9.999, 10.0, 19.999, 20.0,
                                                                  29.999, 30.0, 500.0)]
    assert members('distance_0_5', rows) == [0, 1]
    assert members('distance_5_10', rows) == [2, 3]
    assert members('distance_10_20', rows) == [4, 5]
    assert members('distance_20_30', rows) == [6, 7]
    assert members('distance_30_inf', rows) == [8, 9]

    rows = [split_row(0, 72, below=1), split_row(0, 72, below=0), split_row(1, 40, below=1)]
    assert members('below_montage', rows) == [0, 2]
    assert members('within_montage', rows) == [1]
    for label in ('depth_1_2', 'radius_0_70', 'snr_10', 'population',
                  # A label with one part too many, or with none, is not a shorter label that happens to fit.
                  'snr_10_15_20', 'distance_0_5_10', 'radius_0_70.5_75_80', 'population_0_1', 'within_montage_0',
                  'radius', 'snr', 'distance', ''):
        with pytest.raises(ValueError, match='Unknown stratum'):
            tables.split_stratum_mask(label, rows)


def test_patch_strata_follow_diameter_and_snr():
    rows = [{'diameter_mm': repr(diameter), 'snr_db': repr(snr)}
            for diameter, snr in ((10.0, 10.0), (24.999, 14.999), (25.0, 15.0), (39.999, 19.999), (40.0, 20.0),
                                  (60.0005, 25.0005), (60.001, 25.001))]

    def patch_members(label):
        return [index for index, keep in enumerate(tables.patch_stratum_mask(label, rows)) if keep]

    assert patch_members('diameter_10_25') == [0, 1]
    assert patch_members('diameter_25_40') == [2, 3]
    assert patch_members('diameter_40_60.001') == [4, 5]
    assert patch_members('snr_10_15') == [0, 1]
    assert patch_members('snr_15_20') == [2, 3]
    assert patch_members('snr_20_25.001') == [4, 5]
    for label in ('radius_10_25', 'diameter_10', 'diameter_10_25_40', 'snr_10_15_20', 'diameter', 'snr', 'all', ''):
        with pytest.raises(ValueError, match='Unknown stratum'):
            tables.patch_stratum_mask(label, rows)


def test_transfer_and_adaptation_strata_follow_population_and_source_radius():
    rows = [{'population': population, 'source_radius_mm': repr(radius)}
            for population, radius in (('1', 30.0), ('1', 44.999), ('1', 45.0), ('1', 54.999), ('1', 55.0),
                                       ('1', 65.0005), ('1', 65.001), ('0', 70.499), ('0', 70.5), ('0', 74.999),
                                       ('0', 75.0), ('0', 79.5), ('0', 79.501),
                                       ('1', 72.0),         # a subcortical row at a cortical radius
                                       ('0', 40.0))]        # a cortical row at a subcortical radius

    def chosen(label):
        return [index for index, keep in enumerate(tables.population_radius_mask(label, rows)) if keep]

    assert chosen('all') == list(range(15))
    assert chosen('population_0') == [7, 8, 9, 10, 11, 12, 14]
    assert chosen('population_1') == [0, 1, 2, 3, 4, 5, 6, 13]
    assert chosen('radius_1_30_45') == [0, 1]
    assert chosen('radius_1_45_55') == [2, 3]
    assert chosen('radius_1_55_65.001') == [4, 5]
    assert chosen('radius_0_70.5_75') == [8, 9]
    assert chosen('radius_0_75_79.501') == [10, 11]
    # These files have no strata of the signal-to-noise ratio, of the distance or of the montage cover.
    for label in ('snr_10_15', 'distance_0_5', 'below_montage', 'radius_0_70', 'radius_0_70.5_75_80', 'radius',
                  'population', 'population_0_1', 'snr_0_10_15', 'clearance_1_5_10', ''):
        with pytest.raises(ValueError, match='Unknown stratum'):
            tables.population_radius_mask(label, rows)


def test_errors_pool_both_model_seeds_and_the_difference_uses_medians():
    # The true positions lie away from the origin. The first network misses along x, the second along y,
    # and the dictionary along all three axes: 2, 3 and 6 mm apart are 7 mm.
    truths = ((40.0, -30.0, 20.0), (-12.0, 5.0, 60.0), (7.0, 70.0, -9.0))
    misses = ((1.0, 3.0, 1.0), (2.0, 4.0, 1.0), (10.0, 30.0, 5.0))
    rows = [{'x_mm': repr(x), 'y_mm': repr(y), 'z_mm': repr(z),
             'mlp_862_x_mm': repr(x + a), 'mlp_862_y_mm': repr(y), 'mlp_862_z_mm': repr(z),
             'mlp_863_x_mm': repr(x), 'mlp_863_y_mm': repr(y - b), 'mlp_863_z_mm': repr(z),
             'dictionary_x_mm': repr(x + 2.0 * c), 'dictionary_y_mm': repr(y - 3.0 * c),
             'dictionary_z_mm': repr(z + 6.0 * c)}
            for (x, y, z), (a, b, c) in zip(truths, misses)]
    pooled = tables.errors(rows, 'mlp_pooled')
    assert pooled.tolist() == [[1.0, 2.0, 10.0], [3.0, 4.0, 30.0]]
    # The pooled median is 3.5; the mean of the two per-fit medians would be 3.0 and the mean 8.33.
    assert float(np.median(pooled)) == 3.5
    assert tables.errors(rows, 'dictionary').tolist() == [[7.0, 7.0, 35.0]]
    # Each fitted network on its own is one row.
    assert tables.errors(rows, 'mlp_862').tolist() == [[1.0, 2.0, 10.0]]
    assert tables.errors(rows, 'mlp_863').tolist() == [[3.0, 4.0, 30.0]]
    # Medians 3.5 and 7. The means differ by -8, and the median of the differences would be -5.
    assert tables.difference_of_medians(pooled, tables.errors(rows, 'dictionary')) == -3.5
    assert tables.difference_of_medians(tables.errors(rows, 'dictionary'), pooled) == 3.5


# --------------------------------------------------------------------------- transfer strata

RADII = (30.0, 44.999, 45.0, 54.999, 55.0, 65.0005, 70.5, 74.999, 75.0, 79.5)
SNRS = (10.0, 14.999, 15.0, 19.999, 20.0, 25.0005)
CLEARANCES = (5.0, 9.999, 10.0, 19.999, 20.0, 80.0)
EXPECTED_BANDS = {                       # the bands of the paper, written out once more for the test
    'radius_30_45': ('radius', 30, 45), 'radius_45_55': ('radius', 45, 55), 'radius_55_65': ('radius', 55, 65.001),
    'radius_70.5_75': ('radius', 70.5, 75), 'radius_75_79.5': ('radius', 75, 79.501),
    'snr_10_15': ('snr', 10, 15), 'snr_15_20': ('snr', 15, 20), 'snr_20_25': ('snr', 20, 25.001),
    'clearance_5_10': ('clearance', 5, 10), 'clearance_10_20': ('clearance', 10, 20),
    'clearance_20_inf': ('clearance', 20, float('inf'))}


def transfer_windows(number, count):
    """Windows of one dataset whose attributes sit on the edges of the bands, with skewed errors."""
    windows = []
    for index in range(count):
        radius = RADII[index % len(RADII)]
        # Skewed, without ties, and different in every dataset.
        sphere = 1.0 + (index % 7) ** 2 + 0.01 * (index % 11) + (0.0, 1.0, 10.0)[number]
        windows.append(dict(window=index, position=index // 2 if number == 2 else index,
                            population='1' if radius < 70 else '0', radius=radius,
                            snr=SNRS[(index // 3) % len(SNRS)], clearance=CLEARANCES[(index // 5) % len(CLEARANCES)],
                            sphere=(sphere, sphere + 3.0 + index % 4),            # the two model seeds differ
                            bem=(2.0 * sphere + index % 3, 2.0 * sphere + 4.0)))
    return windows


def transfer_inputs(datasets):
    paired, clearance = {}, {}
    for seed, windows in zip(SEEDS, datasets):
        paired[seed] = [{'window_id': str(w['window']), 'position_pool_id': str(w['position']),
                         'population': w['population'], 'source_radius_mm': repr(w['radius']),
                         'snr_db_referenced': repr(w['snr']),
                         **{f'mlp_{fit}_package_sphere_error_mm': repr(w['sphere'][k])
                            for k, fit in enumerate(tables.FITS)},
                         **{f'mlp_{fit}_bem_error_mm': repr(w['bem'][k]) for k, fit in enumerate(tables.FITS)}}
                        for w in windows]
        clearance[seed] = {str(w['window']): w['clearance'] for w in windows}
    return paired, clearance


def expected_stratum(name, datasets):
    """Windows, smallest number of positions, and means of the dataset medians, derived without the driver."""
    def member(window):
        if name.startswith('population_'):
            return window['population'] == name[-1]
        kind, low, high = EXPECTED_BANDS[name]
        return low <= window[kind] < high

    selected = [[window for window in windows if member(window)] for windows in datasets]
    sphere = [statistics.median([value for window in chosen for value in window['sphere']]) for chosen in selected]
    bem = [statistics.median([value for window in chosen for value in window['bem']]) for chosen in selected]
    return dict(windows=sum(len(chosen) for chosen in selected),
                positions=min(len({window['position'] for window in chosen}) for chosen in selected),
                sphere=statistics.fmean(sphere), bem=statistics.fmean(bem),
                difference=statistics.fmean(b - s for b, s in zip(bem, sphere)))


def test_transfer_strata_count_edge_windows_and_average_the_pooled_dataset_medians():
    datasets = [transfer_windows(number, count) for number, count in enumerate((600, 660, 720))]
    strata = {record['stratum']: record for record in tables.transfer_window_strata(*transfer_inputs(datasets))}
    assert list(strata) == [name for name, _, _ in tables.TRANSFER_STRATA]
    assert set(strata) == {'population_0', 'population_1', *EXPECTED_BANDS}
    for name, record in strata.items():
        expected = expected_stratum(name, datasets)
        assert record['windows'] == expected['windows'], name
        assert record['smallest_number_of_positions_in_a_dataset'] == expected['positions'] >= 30, name
        assert record['package_sphere_mean_of_seed_medians_mm'] == pytest.approx(expected['sphere'], abs=1e-12), name
        assert record['bem_mean_of_seed_medians_mm'] == pytest.approx(expected['bem'], abs=1e-12), name
        assert record['mean_difference_mm'] == pytest.approx(expected['difference'], abs=1e-12), name
    # Every window is in exactly one stratum of each family.
    total = sum(len(windows) for windows in datasets)
    for family in ('population', 'radius', 'snr', 'clearance'):
        assert sum(record['windows'] for name, record in strata.items() if name.startswith(family)) == total
    # The data separate the pooling rules: in the first stratum the pooled median of a dataset differs from
    # the median of either fit and from their mean, and the mean of the dataset medians from the median of
    # all values.
    chosen = [[w for w in windows if w['population'] == '0'] for windows in datasets]
    pooled = [statistics.median([v for w in c for v in w['sphere']]) for c in chosen]
    fits = [[statistics.median([w['sphere'][k] for w in c]) for k in range(2)] for c in chosen]
    assert all(abs(pooled[0] - other) > 0.05 for other in (*fits[0], statistics.fmean(fits[0])))
    assert abs(statistics.fmean(pooled) - statistics.median(
        [v for c in chosen for w in c for v in w['sphere']])) > 0.05


def keep_positions(windows, radii, count):
    """The windows without those at the given radii, except the windows of the first ``count`` positions there."""
    kept = list(dict.fromkeys(w['position'] for w in windows if w['radius'] in radii))[:count]
    return [w for w in windows if w['radius'] not in radii or w['position'] in kept]


def test_transfer_stratum_is_averaged_only_with_thirty_positions_in_every_dataset():
    datasets = [transfer_windows(number, 600) for number in range(3)]
    # The third dataset has two windows at every position: 29 positions of the deepest band are 58 windows.
    datasets[2] = keep_positions(datasets[2], (30.0, 44.999), 29)
    # Exactly 30 positions of another band in the second dataset are enough.
    datasets[1] = keep_positions(datasets[1], (55.0, 65.0005), 30)
    strata = {record['stratum']: record for record in tables.transfer_window_strata(*transfer_inputs(datasets))}
    record = strata['radius_30_45']
    assert record['smallest_number_of_positions_in_a_dataset'] == 29
    assert record['windows'] == 120 + 120 + 58
    assert record['package_sphere_mean_of_seed_medians_mm'] is None
    assert record['bem_mean_of_seed_medians_mm'] is None and record['mean_difference_mm'] is None
    record, expected = strata['radius_55_65'], expected_stratum('radius_55_65', datasets)
    assert record['smallest_number_of_positions_in_a_dataset'] == expected['positions'] == 30
    assert record['windows'] == expected['windows'] == 120 + 30 + 120
    assert record['package_sphere_mean_of_seed_medians_mm'] == pytest.approx(expected['sphere'], abs=1e-12)
    assert record['bem_mean_of_seed_medians_mm'] == pytest.approx(expected['bem'], abs=1e-12)
    assert record['mean_difference_mm'] == pytest.approx(expected['difference'], abs=1e-12)
    assert strata['radius_45_55']['mean_difference_mm'] is not None
    # A band that is empty in one dataset is listed without values as well.
    datasets[0] = [w for w in datasets[0] if w['radius'] not in (45.0, 54.999)]
    strata = {record['stratum']: record for record in tables.transfer_window_strata(*transfer_inputs(datasets))}
    assert strata['radius_45_55']['smallest_number_of_positions_in_a_dataset'] == 0
    assert strata['radius_45_55']['windows'] == 120 + 120
    assert strata['radius_45_55']['package_sphere_mean_of_seed_medians_mm'] is None
    assert strata['radius_45_55']['bem_mean_of_seed_medians_mm'] is None
    assert strata['radius_45_55']['mean_difference_mm'] is None


@pytest.mark.parametrize('change, family', [
    (dict(radius=68.0), 'radius'), (dict(radius=29.9), 'radius'), (dict(radius=79.6), 'radius'),
    (dict(snr=9.0), 'snr'), (dict(snr=26.0), 'snr'), (dict(clearance=3.0), 'clearance'),
    (dict(population='2'), 'population'),
    # The outer edges of the bands, and the two edges of the gap between the subcortical and the cortical radii.
    (dict(radius=just_below(30.0)), 'radius'), (dict(radius=65.001), 'radius'),
    (dict(radius=just_below(70.5)), 'radius'), (dict(radius=79.501), 'radius'),
    (dict(snr=just_below(10.0)), 'snr'), (dict(snr=25.001), 'snr'), (dict(clearance=just_below(5.0)), 'clearance')])
def test_transfer_window_outside_every_band_of_a_family_stops_the_driver(change, family):
    datasets = [transfer_windows(number, 600) for number in range(3)]
    datasets[1][7].update(change)
    with pytest.raises(ValueError, match=f'dataset 863: the {family} strata hold 599 of 600 windows'):
        tables.transfer_window_strata(*transfer_inputs(datasets))


def test_transfer_window_just_inside_an_outer_edge_of_a_band_is_counted():
    datasets = [transfer_windows(number, 600) for number in range(3)]
    before = {record['stratum']: record['windows']
              for record in tables.transfer_window_strata(*transfer_inputs(datasets))}
    assert (datasets[1][5]['radius'], datasets[1][9]['radius'], datasets[1][15]['snr']) == (65.0005, 79.5, 25.0005)
    datasets[1][5]['radius'] = just_below(65.001)
    datasets[1][9]['radius'] = just_below(79.501)
    datasets[1][15]['snr'] = just_below(25.001)
    after = {record['stratum']: record['windows']
             for record in tables.transfer_window_strata(*transfer_inputs(datasets))}
    assert after == before


def test_transfer_window_in_two_bands_of_a_family_stops_the_driver(monkeypatch):
    datasets = [transfer_windows(number, 600) for number in range(3)]
    twice = sum(window['snr'] == 15.0 for window in datasets[0])
    assert twice > 0
    # A table of strata in which the windows at 15 dB belong to two bands.
    wider = ('snr_10_15', 'snr', lambda row, clearance: 10 <= float(row['snr_db_referenced']) <= 15)
    monkeypatch.setattr(tables, 'TRANSFER_STRATA', tuple(
        wider if name == 'snr_10_15' else (name, family, test) for name, family, test in tables.TRANSFER_STRATA))
    with pytest.raises(ValueError, match=f'dataset 862: the snr strata hold {600 + twice} of 600 windows'):
        tables.transfer_window_strata(*transfer_inputs(datasets))


# --------------------------------------------------------------------------- eligibility census

def census_row(window, population, xyz, clearance, eligible):
    x, y, z = xyz
    return {'window_id': str(window), 'population': str(population), 'x_mm': repr(x), 'y_mm': repr(y),
            'z_mm': repr(z), 'inner_skull_clearance_mm': repr(clearance), 'eligible': 'True' if eligible else 'False'}


def at_radius(radius, direction):
    norm = sum(component ** 2 for component in direction) ** 0.5
    return tuple(radius * component / norm for component in direction)


def test_eligibility_census_counts_strata_quadrants_and_sources_outside_the_skull():
    first = [
        census_row(0, 0, at_radius(72.0, (3, -4, 12)), 12.0, True),     # superior, posterior
        census_row(1, 0, at_radius(74.9, (3, 4, 12)), 2.0, False),      # superior, anterior, too close
        census_row(2, 0, at_radius(75.1, (3, -4, -12)), -4.0, False),   # inferior, posterior, outside the skull
        census_row(3, 0, at_radius(79.4, (3, 4, -12)), -1.0, False),    # inferior, anterior, outside the skull
        census_row(4, 0, (70.5, 0.0, 0.0), 0.0, False),                 # y = 0 and z = 0: anterior and superior
        census_row(5, 1, (30.0, 0.0, 0.0), 30.0, True),                 # radii on a band edge, on the x axis
        census_row(6, 1, (44.999, 0.0, 0.0), 28.0, True),
        census_row(7, 1, (45.0, 0.0, 0.0), 20.0, True),
        census_row(8, 1, (54.999, 0.0, 0.0), 15.0, True),
        census_row(9, 1, (55.0, 0.0, 0.0), 8.0, True),
        census_row(10, 1, (65.0005, 0.0, 0.0), 5.0, True),
        census_row(11, 0, (0.0, -79.5, 0.0), 6.0, True),                # the largest radius; posterior, z = 0
    ]
    second = first[:3] + first[5:7]          # a different dataset: five of the windows
    third = [first[4], first[9], first[11]]
    census, outside = tables.eligibility_census(dict(zip(SEEDS, (first, second, third))))
    by_name = {record['stratum']: record for record in census}
    assert list(by_name) == [name for name, _, _ in tables.CENSUS_STRATA]

    def counts(name):
        return [(by_name[name]['by_dataset_seed'][seed]['eligible_windows'],
                 by_name[name]['by_dataset_seed'][seed]['total_windows']) for seed in SEEDS]

    assert counts('all') == [(8, 12), (3, 5), (2, 3)]
    assert counts('cortical') == [(2, 6), (1, 3), (1, 2)]
    assert counts('subcortical') == [(6, 6), (2, 2), (1, 1)]
    assert counts('radius_30_45') == [(2, 2), (2, 2), (0, 0)]
    assert counts('radius_45_55') == [(2, 2), (0, 0), (0, 0)]
    assert counts('radius_55_65') == [(2, 2), (0, 0), (1, 1)]
    assert counts('radius_70.5_75') == [(1, 3), (1, 2), (0, 1)]
    assert counts('radius_75_79.5') == [(1, 3), (0, 1), (1, 1)]
    assert counts('cortical_superior_posterior') == [(2, 2), (1, 1), (1, 1)]
    assert counts('cortical_superior_anterior') == [(0, 2), (0, 1), (0, 1)]
    assert counts('cortical_inferior_posterior') == [(0, 1), (0, 1), (0, 0)]
    assert counts('cortical_inferior_anterior') == [(0, 1), (0, 0), (0, 0)]
    assert (by_name['all']['eligible_windows'], by_name['all']['total_windows']) == (13, 20)
    assert by_name['all']['fraction'] == pytest.approx(13 / 20)     # pooled, not the mean of 8/12, 3/5 and 2/3
    assert by_name['radius_45_55']['fraction'] == 1.0
    assert (by_name['radius_75_79.5']['eligible_windows'], by_name['radius_75_79.5']['total_windows']) == (2, 5)
    assert by_name['radius_75_79.5']['fraction'] == pytest.approx(2 / 5)
    assert by_name['cortical_inferior_anterior']['fraction'] == 0.0
    # Outside the inner skull means a negative clearance; a clearance of zero is inside.
    assert [(outside['by_dataset_seed'][seed]['outside_windows'], outside['by_dataset_seed'][seed]['total_windows'])
            for seed in SEEDS] == [(2, 6), (1, 3), (0, 2)]
    assert (outside['outside_windows'], outside['total_windows']) == (3, 11)
    assert outside['fraction'] == pytest.approx(3 / 11)


@pytest.mark.parametrize('row, message', [
    (census_row(99, 0, at_radius(68.0, (1, 2, 2)), 9.0, True), 'the radius strata do not hold every window once'),
    (census_row(99, 2, at_radius(40.0, (1, 2, 2)), 9.0, True), 'the population strata do not hold every window once'),
    # The outer edges of the bands, and the two edges of the gap between the subcortical and the cortical radii.
    (census_row(99, 1, (just_below(30.0), 0.0, 0.0), 9.0, True), 'the radius strata do not hold every window once'),
    (census_row(99, 1, (0.0, 65.001, 0.0), 9.0, True), 'the radius strata do not hold every window once'),
    (census_row(99, 0, (0.0, 0.0, just_below(70.5)), 9.0, True), 'the radius strata do not hold every window once'),
    (census_row(99, 0, (79.501, 0.0, 0.0), 9.0, True), 'the radius strata do not hold every window once'),
])
def test_census_window_outside_every_stratum_of_a_family_stops_the_driver(row, message):
    ordinary = census_row(0, 1, at_radius(40.0, (1, 2, 2)), 30.0, True)
    with pytest.raises(ValueError, match=f'dataset 863: {message}'):
        tables.eligibility_census({SEEDS[0]: [ordinary], SEEDS[1]: [ordinary, row], SEEDS[2]: [ordinary]})


def test_census_window_in_two_strata_of_a_family_stops_the_driver(monkeypatch):
    on_the_edge = census_row(0, 1, (45.0, 0.0, 0.0), 30.0, True)
    # A table of strata in which a source at 45 mm belongs to two bands.
    wider = ('radius_30_45', 'radius', lambda row: 30 <= float(row['x_mm']) <= 45)
    monkeypatch.setattr(tables, 'CENSUS_STRATA', tuple(
        wider if name == 'radius_30_45' else (name, family, test) for name, family, test in tables.CENSUS_STRATA))
    with pytest.raises(ValueError, match='dataset 862: the radius strata do not hold every window once'):
        tables.eligibility_census({seed: [on_the_edge] for seed in SEEDS})


def test_census_bands_and_quadrants_at_their_edges():
    def totals(rows):
        census, _ = tables.eligibility_census({seed: rows for seed in SEEDS})
        return {record['stratum']: record['by_dataset_seed'][SEEDS[0]]['total_windows'] for record in census}

    # Just inside the upper edge of the widest subcortical and of the widest cortical band.
    inside = totals([census_row(0, 1, (just_below(65.001), 0.0, 0.0), 9.0, True),
                     census_row(1, 0, (0.0, just_below(79.501), 0.0), 9.0, True)])
    assert (inside['radius_55_65'], inside['radius_75_79.5']) == (1, 1)
    # Half a millimetre on either side of the planes y = 0 and z = 0, and on the planes.
    sides = totals([census_row(index, 0, (73.0, y, z), 9.0, True) for index, (y, z) in enumerate((
        (0.5, 0.5), (0.0, 0.0), (0.5, -0.5), (0.0, -0.5), (-0.5, 0.5), (-0.5, 0.0), (-0.5, -0.5)))])
    assert sides['cortical_superior_anterior'] == 2 and sides['cortical_inferior_anterior'] == 2
    assert sides['cortical_superior_posterior'] == 2 and sides['cortical_inferior_posterior'] == 1


def test_census_empty_stratum_has_no_fraction():
    ordinary = census_row(0, 1, at_radius(40.0, (1, 2, 2)), 30.0, True)
    census, outside = tables.eligibility_census({seed: [ordinary] for seed in SEEDS})
    by_name = {record['stratum']: record for record in census}
    assert by_name['cortical']['total_windows'] == 0 and by_name['cortical']['fraction'] is None
    assert outside['total_windows'] == 0 and outside['fraction'] is None


def test_directional_count_splits_the_windows_at_zero_and_can_keep_the_cortical_ones():
    # Population, y, z and eligibility. A coordinate of zero belongs to the anterior or superior half.
    rows = [census_row(index, population, (10.0, y, z), 9.0, eligible)
            for index, (population, y, z, eligible) in enumerate((
                (0, 5.0, -3.0, True), (0, 0.0, 0.0, False), (0, -5.0, 3.0, True), (1, 5.0, 3.0, True),
                (1, -5.0, -3.0, False), (0, 4.0, 2.0, True), (1, 1.0, -1.0, True)))]

    def count(axis, positive, cortical_only):
        record = tables.directional_count(rows, axis, positive, cortical_only)
        assert record['fraction'] == pytest.approx(record['eligible_windows'] / record['total_windows'])
        return record['eligible_windows'], record['total_windows']

    assert count('y', True, False) == (4, 5) and count('y', False, False) == (1, 2)
    assert count('z', True, False) == (3, 4) and count('z', False, False) == (2, 3)
    assert count('y', True, True) == (2, 3) and count('y', False, True) == (1, 1)
    assert count('z', True, True) == (2, 3) and count('z', False, True) == (1, 1)


# --------------------------------------------------------------------------- count control and record checks

def test_common_window_errors_follow_the_recorded_rows_and_the_network_of_the_count_control():
    rows = [{'window': str(window), 'position_id': str(100 + window), 'exact_error_mm': repr(float(window)),
             'count_matched_error_mm': repr(10.0 + window)} for window in range(6)]
    # The true positions lie away from the origin. The first network misses by 2, 3 and 6 mm along the axes,
    # which is 7 mm, times the number of the row; the second network misses by 9 mm.
    region = [{'position_id': str(100 + window), 'x_mm': repr(20.0 + window), 'y_mm': '-30.0', 'z_mm': '40.0',
               'mlp_862_x_mm': repr(20.0 + window + 2.0 * k), 'mlp_862_y_mm': repr(-30.0 + 3.0 * k),
               'mlp_862_z_mm': repr(40.0 - 6.0 * k),
               'mlp_863_x_mm': repr(20.0 + window), 'mlp_863_y_mm': '-30.0', 'mlp_863_z_mm': '49.0'}
              for k, window in enumerate((4, 1, 5), start=1)]
    common = tables.common_window_errors(rows, [4, 1, 5], region, 862)
    assert common['exact'].tolist() == [4.0, 1.0, 5.0]
    assert common['count_matched'].tolist() == [14.0, 11.0, 15.0]
    assert common['regional'].tolist() == [7.0, 14.0, 21.0]             # the network of the given model seed only
    assert tables.common_window_errors(rows, [4, 1, 5], region, 863)['regional'].tolist() == [9.0, 9.0, 9.0]
    with pytest.raises(ValueError, match='disagree on the common test windows'):
        tables.common_window_errors(rows, [1, 4, 5], region, 862)       # another order: other positions
    with pytest.raises(ValueError, match='disagree on the common test windows'):
        tables.common_window_errors(rows, [4, 1], region, 862)          # fewer windows than regional rows
    with pytest.raises(ValueError, match='every common test window once'):
        tables.common_window_errors(rows, [4, 4, 5], region, 862)
    with pytest.raises(ValueError, match='every common test window once'):
        tables.common_window_errors(rows, [4, 1, 7], region, 862)


def test_recomputed_number_must_equal_the_recorded_one_within_the_tolerance():
    evidence = tables.Evidence(Path('.'))
    kept = evidence.check(np.float64(73.5), 73.5, 'example')
    assert kept == 73.5 and type(kept) is float and evidence.checked == 1
    # A difference of 1e-11 mm is accepted and one of 1e-9 mm is not, in either direction.
    assert evidence.check(73.5 + 1e-11, 73.5, 'example') == 73.5 + 1e-11
    evidence.check(73.5, 73.5 + 1e-11, 'example')
    assert evidence.checked == 3
    for value, recorded in ((73.5 + 1e-9, 73.5), (73.5, 73.5 + 1e-9)):
        with pytest.raises(ValueError, match=f'example: recomputed {value}, recorded {recorded}'):
            evidence.check(value, recorded, 'example')
    assert evidence.checked == 3                # a comparison that fails is not counted
    # Files with four decimals are compared with a wider tolerance.
    evidence.check(0.2716, 0.27163, 'four decimals', tolerance=1e-4)
    assert evidence.checked == 4
    with pytest.raises(ValueError, match='four decimals: recomputed 0.2716, recorded 0.2718'):
        evidence.check(0.2716, 0.2718, 'four decimals', tolerance=1e-4)


def test_recomputed_statistics_must_equal_the_record():
    evidence = tables.Evidence(Path('.'))
    values = [1.0, 2.0, 3.0, 4.0, 100.0]
    # The 90th percentile lies 0.6 of the way from 4 to 100. Other entries of a record are not looked at.
    record = dict(median_mm=3.0, mean_mm=22.0, p90_mm=61.6, windows=5, median_ci95_mm=[2.0, 4.0], positions=3)
    evidence.check_metrics(values, record, 'example')
    assert evidence.checked == 3
    for key, wrong in (('median_mm', 3.0001), ('mean_mm', 21.9), ('p90_mm', 61.0), ('median_mm', 3.0 + 1e-9)):
        with pytest.raises(ValueError, match=f'example, {key}: recomputed'):
            evidence.check_metrics(values, dict(record, **{key: wrong}), 'example')
    with pytest.raises(ValueError, match='example, mean_mm: recomputed 22.0, recorded 21.9'):
        evidence.check_metrics(values, dict(record, mean_mm=21.9), 'example')
    for windows in (4, 6):
        with pytest.raises(ValueError, match=f'example: 5 windows, recorded {windows}'):
            evidence.check_metrics(values, dict(record, windows=windows), 'example')
    # Two fits of the same windows: the statistics pool both rows, the window count is that of one. The pooled
    # median is 15 and the mean of the two fit medians 16.5; the pooled 90th percentile is 55, that of the
    # first fit 61.6.
    both = [values, [10.0, 20.0, 30.0, 40.0, 50.0]]
    checked = evidence.checked
    evidence.check_metrics(np.array(both), dict(median_mm=15.0, mean_mm=26.0, p90_mm=55.0, windows=5), 'pooled')
    assert evidence.checked == checked + 3
    with pytest.raises(ValueError, match='pooled: 5 windows, recorded 10'):
        evidence.check_metrics(both, dict(median_mm=15.0, mean_mm=26.0, p90_mm=55.0, windows=10), 'pooled')


@pytest.mark.parametrize('missing', [('p90_mm',), ('windows',), ('mean_mm',), ('median_mm', 'p90_mm'),
                                     ('median_mm', 'mean_mm', 'p90_mm', 'windows')])
def test_record_without_a_statistic_or_the_window_count_stops_the_driver(missing):
    evidence = tables.Evidence(Path('.'))
    record = dict(median_mm=3.0, mean_mm=22.0, p90_mm=61.6, windows=5)
    incomplete = {key: value for key, value in record.items() if key not in missing}
    with pytest.raises(ValueError, match=f"example: the record lacks {', '.join(missing)}$"):
        evidence.check_metrics([1.0, 2.0, 3.0, 4.0, 100.0], incomplete, 'example')
    assert evidence.checked == 0                # nothing is compared with an incomplete record


# One value per dataset: the mean is 6 and the median 5; the smallest value belongs to the second dataset
# and the largest to the third.
SEED_VALUES = {'863': 4.0, '864': 9.0, '862': 5.0}
SEED_RECORD = dict(dataset_seed_values=[5.0, 4.0, 9.0], mean=6.0, range=[4.0, 9.0])
SEED_KEYS = dict(values_key='dataset_seed_values', mean_key='mean', range_key='range')


def test_dataset_values_their_mean_and_their_range_are_compared_with_the_headline():
    evidence = tables.Evidence(Path('.'))
    evidence.check_seed_values(SEED_VALUES, SEED_RECORD, 'example', **SEED_KEYS)
    assert evidence.checked == 6
    # The record may hold the dataset values as a mapping, in any order of the seeds.
    as_mapping = dict(SEED_RECORD, dataset_seed_values={'864': 9.0, '862': 5.0, '863': 4.0})
    evidence.check_seed_values(SEED_VALUES, as_mapping, 'example', **SEED_KEYS)
    assert evidence.checked == 12
    # Or not at all: then the mean and the two ends of the range are compared.
    evidence.check_seed_values(SEED_VALUES, dict(average=6.0, ends=[4.0, 9.0]), 'example',
                               values_key=None, mean_key='average', range_key='ends')
    assert evidence.checked == 15
    # The largest value in the first dataset and the smallest in the last one.
    evidence.check_seed_values({'862': 9.0, '863': 5.0, '864': 4.0},
                               dict(SEED_RECORD, dataset_seed_values=[9.0, 5.0, 4.0]), 'example', **SEED_KEYS)
    assert evidence.checked == 21


@pytest.mark.parametrize('change, message', [
    (dict(dataset_seed_values=[5.0, 4.5, 9.0]), 'example, dataset 863: recomputed 4.0, recorded 4.5'),
    (dict(dataset_seed_values=[5.0, 4.0, 9.0 + 1e-9]), 'example, dataset 864: recomputed 9.0, recorded 9.0'),
    (dict(dataset_seed_values=[4.0, 5.0, 9.0]), 'example, dataset 862: recomputed 5.0, recorded 4.0'),
    (dict(dataset_seed_values={'862': 5.0, '863': 4.0, '864': 9.5}),
     'example, dataset 864: recomputed 9.0, recorded 9.5'),
    (dict(dataset_seed_values={'863': 5.0, '862': 4.0, '864': 9.0}),
     'example, dataset 862: recomputed 5.0, recorded 4.0'),
    (dict(dataset_seed_values=[5.0, 4.0]), 'example: 2 dataset values recorded, 3 expected'),
    (dict(dataset_seed_values=[5.0, 4.0, 9.0, 9.0]), 'example: 4 dataset values recorded, 3 expected'),
    (dict(mean=5.0), 'example, mean over the datasets: recomputed 6.0, recorded 5.0'),
    (dict(mean=6.0 + 1e-9), 'example, mean over the datasets: recomputed 6.0, recorded 6.0'),
    (dict(range=[4.5, 9.0]), 'example, smallest dataset value: recomputed 4.0, recorded 4.5'),
    (dict(range=[4.0, 8.5]), 'example, largest dataset value: recomputed 9.0, recorded 8.5'),
    (dict(range=[4.0 - 1e-9, 9.0]), 'example, smallest dataset value: recomputed 4.0, recorded 3.9'),
    (dict(range=[4.0, 9.0 + 1e-9]), 'example, largest dataset value: recomputed 9.0, recorded 9.0'),
    (dict(range=[9.0, 4.0]), 'example, smallest dataset value: recomputed 4.0, recorded 9.0'),
    # The first dataset, the mean or the median as an end of the range.
    (dict(range=[5.0, 9.0]), 'example, smallest dataset value: recomputed 4.0, recorded 5.0'),
    (dict(range=[6.0, 9.0]), 'example, smallest dataset value: recomputed 4.0, recorded 6.0'),
    (dict(range=[4.0, 5.0]), 'example, largest dataset value: recomputed 9.0, recorded 5.0'),
    (dict(range=[4.0, 6.0]), 'example, largest dataset value: recomputed 9.0, recorded 6.0'),
])
def test_headline_that_differs_from_the_dataset_values_stops_the_driver(change, message):
    with pytest.raises(ValueError, match=message):
        tables.Evidence(Path('.')).check_seed_values(SEED_VALUES, dict(SEED_RECORD, **change), 'example', **SEED_KEYS)


def test_headline_without_dataset_values_is_still_compared_in_mean_and_range():
    keys = dict(SEED_KEYS, values_key=None)
    for change, message in ((dict(mean=5.0), 'mean over the datasets'), (dict(range=[4.5, 9.0]), 'smallest dataset'),
                            (dict(range=[4.0, 8.5]), 'largest dataset')):
        with pytest.raises(ValueError, match=f'example, {message}'):
            tables.Evidence(Path('.')).check_seed_values(SEED_VALUES, dict(SEED_RECORD, **change), 'example', **keys)


def test_dataset_spread_and_the_cells_of_the_printed_tables():
    spread = tables.seed_spread([4.0, 9.0, 5.126])
    assert spread == dict(dataset_seed_values=[4.0, 9.0, 5.126], mean=pytest.approx(6.042, abs=1e-12),
                          range=[4.0, 9.0])
    assert tables.seed_cell(spread) == '6.04 (4.00–9.00)'
    seconds = dict(median=12.34, minimum=11.96, maximum=13.06)
    assert tables.interval(seconds) == '12.3 (12.0–13.1)'
    assert tables.interval(seconds, 2) == '12.34 (11.96–13.06)'


def test_settings_must_agree_between_the_datasets():
    assert tables.common_setting([60, 60, 60], 'the order') == 60
    assert tables.common_setting(['one dataset'], 'the name') == 'one dataset'
    assert tables.common_setting(([10.0, 25.0] for _ in SEEDS), 'the range') == [10.0, 25.0]
    for values in ([60, 60, 80], [60, 80, 60], [80, 60, 60]):
        with pytest.raises(ValueError, match='disagree on the order'):
            tables.common_setting(values, 'the order')


def test_evidence_reader_returns_the_content_and_records_the_hash_of_every_file(tmp_path):
    (tmp_path / 'study/862').mkdir(parents=True)
    record = b'\xef\xbb\xbf{"median_mm": 3.5}\n'           # a file that starts with a byte-order mark
    table = b'window,error_mm\r\n0,1.5\r\n1,2.5\r\n'        # a table with Windows line ends
    (tmp_path / 'study/record.json').write_bytes(record)
    (tmp_path / 'study/862/errors.csv').write_bytes(table)
    np.savez(tmp_path / 'study/862/predictions.npz', truth=np.arange(6.0).reshape(2, 3))
    evidence = tables.Evidence(tmp_path)
    assert evidence.json(Path('study') / 'record.json') == dict(median_mm=3.5)
    assert evidence.rows('study/862/errors.csv') == [dict(window='0', error_mm='1.5'), dict(window='1', error_mm='2.5')]
    stored = evidence.arrays(Path('study/862') / 'predictions.npz')
    assert stored['truth'].tolist() == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
    # The hashes are those of the files as they are, under their paths with forward slashes.
    assert evidence.hashes == {
        'study/record.json': hashlib.sha256(record).hexdigest(),
        'study/862/errors.csv': hashlib.sha256(table).hexdigest(),
        'study/862/predictions.npz': hashlib.sha256((tmp_path / 'study/862/predictions.npz').read_bytes()).hexdigest()}
    assert all(type(key) is str for key in evidence.hashes) and evidence.checked == 0


# --------------------------------------------------------------------------- paired split contrast from files

def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def prediction_rows(errors_by_fit):
    """Prediction rows with the truth away from the origin and the two networks at the given distances from it."""
    rows = []
    for window, (first, second) in enumerate(errors_by_fit):
        x, y, z = 50.0 - window, 20.0, -10.0 * window
        rows.append({'window': str(window), 'position_id': str(window // 2),
                     'x_mm': repr(x), 'y_mm': repr(y), 'z_mm': repr(z),
                     'mlp_862_x_mm': repr(x + first), 'mlp_862_y_mm': repr(y), 'mlp_862_z_mm': repr(z),
                     'mlp_863_x_mm': repr(x), 'mlp_863_y_mm': repr(y - second), 'mlp_863_z_mm': repr(z)})
    return rows


# Errors of the two networks on every window, by dataset and training regime.
PAIRED_ERRORS = {
    '862': dict(position=[(1.0, 4.0), (2.0, 6.0), (9.0, 8.0)], region=[(5.0, 12.0), (6.0, 15.0), (30.0, 16.0)]),
    '863': dict(position=[(1.0, 3.0), (3.0, 7.0)], region=[(8.0, 11.0), (10.0, 20.0)]),
    '864': dict(position=[(2.0, 2.0), (2.0, 3.0), (2.0, 4.0), (50.0, 5.0)],
                region=[(21.0, 25.0), (21.0, 26.0), (21.0, 27.0), (1.0, 28.0)])}


def paired_tree(root, contrasts=(8.5, 7.5, 20.5), pooled=15.0, records=None, pooled_windows=9, pooled_positions=5):
    """Files of the paired contrast; ``records`` changes entries of the record of a dataset.

    The datasets have 3, 2 and 4 windows at 2, 1 and 2 positions.
    """
    for (seed, regimes), contrast in zip(PAIRED_ERRORS.items(), contrasts):
        folder = root / 'split' / seed / 'same_test_windows'
        for policy, values in regimes.items():
            write_csv(folder / policy / 'predictions.csv', prediction_rows(values))
        windows = len(regimes['position'])
        record = dict(region_minus_position_mm=contrast, ci95_mm=[0.0, 1.0], windows=windows,
                      distinct_positions=(windows + 1) // 2)
        record.update((records or {}).get(seed, {}))
        (folder / 'paired_mlp.json').write_text(json.dumps(record), encoding='utf-8')
    (root / 'split' / 'paired_split_summary.json').write_text(json.dumps(dict(strata=dict(all=dict(
        windows=pooled_windows, seed_position_clusters=pooled_positions, region_minus_position_mm=pooled,
        conditional_ci95_mm=[0.0, 1.0])))), encoding='utf-8')
    return tables.Evidence(root)


def test_paired_split_contrast_is_recomputed_per_dataset_and_over_all_windows(tmp_path):
    # First dataset, both fits pooled: exact {1,2,4,6,8,9} against region {5,6,12,15,16,30}, medians 5 and 13.5.
    # The first network alone would give 6 - 2 = 4, the second 15 - 6 = 9, and the means of the two fit
    # medians 10.5 - 4 = 6.5. The other datasets give 10.5 - 3 = 7.5 and 23 - 2.5 = 20.5.
    # Over the 18 values of the three datasets the medians are 3 and 18: the contrast is 15, which is
    # neither the mean (12.17) nor the median (8.5) of the three dataset contrasts.
    evidence = paired_tree(tmp_path)
    by_seed, summary = tables.paired_split_tables(evidence, Path('split'))
    assert [by_seed[seed]['region_minus_position_mm'] for seed in SEEDS] == [8.5, 7.5, 20.5]
    assert [by_seed[seed]['windows'] for seed in SEEDS] == [3, 2, 4]
    assert summary['strata']['all']['region_minus_position_mm'] == 15.0
    assert evidence.checked == 4
    assert len(evidence.hashes) == 3 * 3 + 1


@pytest.mark.parametrize('contrasts, pooled, message', [
    # 7.0 is the contrast of the first network alone, and 8.5 the median of the three dataset contrasts.
    ((8.5, 7.0, 20.5), 15.0, 'paired split contrast, dataset 863: recomputed 7.5, recorded 7.0'),
    ((8.5, 7.5, 20.5), 8.5, 'paired split contrast, three datasets: recomputed 15.0, recorded 8.5')])
def test_paired_split_contrast_that_differs_from_the_record_stops_the_driver(tmp_path, contrasts, pooled, message):
    with pytest.raises(ValueError, match=message):
        tables.paired_split_tables(paired_tree(tmp_path, contrasts, pooled), Path('split'))


@pytest.mark.parametrize('changes, message', [
    (dict(records={'863': dict(windows=3)}), 'dataset 863: windows or positions differ from the record'),
    (dict(records={'863': dict(windows=1)}), 'dataset 863: windows or positions differ from the record'),
    (dict(records={'864': dict(distinct_positions=3)}), 'dataset 864: windows or positions differ from the record'),
    (dict(records={'864': dict(distinct_positions=1)}), 'dataset 864: windows or positions differ from the record'),
    (dict(pooled_windows=8), 'paired split contrast: the number of windows differs from the record'),
    (dict(pooled_windows=10), 'paired split contrast: the number of windows differs from the record')])
def test_paired_split_counts_that_differ_from_the_record_stop_the_driver(tmp_path, changes, message):
    with pytest.raises(ValueError, match=message):
        tables.paired_split_tables(paired_tree(tmp_path, **changes), Path('split'))


@pytest.mark.parametrize('positions', [4, 6])
def test_paired_split_pooled_positions_that_differ_from_the_datasets_stop_the_driver(tmp_path, positions):
    # The pooled record counts the positions within each dataset: 2 + 1 + 2.
    with pytest.raises(ValueError, match='paired split contrast'):
        tables.paired_split_tables(paired_tree(tmp_path, pooled_positions=positions), Path('split'))


@pytest.mark.parametrize('windows', [(5, 6), (1, 0)])
def test_paired_split_regimes_must_cover_the_same_windows(tmp_path, windows):
    evidence = paired_tree(tmp_path)
    # Other windows, or the same windows in another order.
    rows = [dict(row, window=str(window)) for row, window in zip(prediction_rows(PAIRED_ERRORS['863']['region']),
                                                                windows)]
    write_csv(tmp_path / 'split/863/same_test_windows/region/predictions.csv', rows)
    with pytest.raises(ValueError, match='different windows'):
        tables.paired_split_tables(evidence, Path('split'))


# --------------------------------------------------------------------------- rerun and sphere-fit records

def digest_record(**datasets):
    return dict(datasets={name: {file: {array: dict(dtype='float64', shape=[2], sha256=value)
                                        for array, value in arrays.items()} for file, arrays in files.items()}
                          for name, files in datasets.items()})


def test_array_digests_of_two_releases_are_compared_array_by_array():
    first = digest_record(a={'test.h5': dict(eeg='11', labels='22')}, b={'train.h5': dict(eeg='33')})
    same = tables.compare_array_digests(first, json.loads(json.dumps(first)))
    assert same == dict(datasets=2, arrays_in_both=3, arrays_with_equal_digest=3, arrays_only_in_first=[],
                        arrays_only_in_second=[])
    second = digest_record(a={'test.h5': dict(eeg='11', labels='99', extra='44')}, b={'pool.h5': dict(xyz='55')})
    different = tables.compare_array_digests(first, second)
    assert (different['arrays_in_both'], different['arrays_with_equal_digest']) == (2, 1)
    assert different['arrays_only_in_first'] == ['b/train.h5/eeg']
    assert different['arrays_only_in_second'] == ['a/test.h5/extra', 'b/pool.h5/xyz']
    with pytest.raises(ValueError, match='different sets of datasets'):
        tables.compare_array_digests(first, digest_record(a={'test.h5': dict(eeg='11')}))
    with pytest.raises(ValueError, match='different sets of datasets'):
        tables.compare_array_digests(digest_record(a={'test.h5': dict(eeg='11')}), first)


def test_arrays_that_one_release_lacks_are_listed_in_sorted_order():
    names = ('n', 'c', 'x', 'a', 'q', 'f', 'k')
    first = digest_record(b={f'{name}.h5': dict(eeg='1') for name in names},
                          a={'test.h5': {name: '1' for name in names}})
    second = digest_record(b={'common.h5': dict(eeg='1')}, a={'test.h5': dict(eeg='1')})
    compared = tables.compare_array_digests(first, second)
    assert compared['arrays_in_both'] == 0 and compared['arrays_with_equal_digest'] == 0
    assert compared['arrays_only_in_first'] == [*[f'a/test.h5/{name}' for name in sorted(names)],
                                                *[f'b/{name}.h5/eeg' for name in sorted(names)]]
    assert compared['arrays_only_in_second'] == ['a/test.h5/eeg', 'b/common.h5/eeg']
    exchanged = tables.compare_array_digests(second, first)
    assert exchanged['arrays_only_in_first'] == compared['arrays_only_in_second']
    assert exchanged['arrays_only_in_second'] == compared['arrays_only_in_first']


@pytest.mark.parametrize('key, value', [('shape', [1, 2]), ('dtype', 'float32'), ('sha256', '12')])
def test_array_with_another_shape_type_or_content_is_not_equal(key, value):
    first = digest_record(a={'test.h5': dict(eeg='11', labels='22')}, b={'train.h5': dict(eeg='33')})
    second = json.loads(json.dumps(first))
    second['datasets']['a']['test.h5']['eeg'][key] = value      # the same bytes can be another array
    compared = tables.compare_array_digests(first, second)
    assert (compared['arrays_in_both'], compared['arrays_with_equal_digest']) == (3, 2)
    assert compared['arrays_only_in_first'] == [] and compared['arrays_only_in_second'] == []


def test_largest_relative_change_and_record_without_keys():
    first = dict(near=dict(a=2.0, b=4.0), far=dict(a=1.0, b=1.0))
    second = dict(near=dict(a=2.2, b=3.0), far=dict(a=5.0, b=1.0))
    assert tables.largest_relative_change(first, second, ['near'], ['a', 'b']) == pytest.approx(0.25)
    assert tables.largest_relative_change(first, second, ['near'], ['a']) == pytest.approx(0.1)
    assert tables.largest_relative_change(first, second, ['near', 'far'], ['a']) == pytest.approx(4.0)
    assert tables.without(dict(a=1, b=2, c=3), 'a', 'c') == dict(b=2)
    assert tables.without(dict(a=1, b=2), 'z') == dict(a=1, b=2)


def test_perturbation_ranges_by_perturbed_fit_and_sphere_fit():
    def row(perturbed, name, step, evaluations, stopped, objective, dipole_fits, patch, error):
        return dict(perturbed_fit=perturbed, sphere_fit=name, objective_scale_minus_one_in_eps=str(step),
                    parameter_fit_evaluations=str(evaluations), parameter_fit_stopped_by_itself=str(int(stopped)),
                    parameter_fit_objective=repr(objective),
                    dipole_fits='' if dipole_fits is None else str(dipole_fits),
                    patch_median_relative_field_difference='' if patch is None else repr(patch),
                    position_error_median_mm=repr(error))

    # The trials of the two sphere fits alternate, and no smallest or largest value stands first or last.
    rows = [row('sphere_fit', 'mne_limit', 1, 1000, False, 2e-4, None, 3e-3, 0.30),
            row('sphere_fit', 'raised_limit', 1, 1710, True, 4e-6, None, 5e-4, 0.22),
            row('sphere_fit', 'mne_limit', 2, 1000, False, 1e-4, None, 2e-3, 0.25),
            row('sphere_fit', 'raised_limit', 2, 100000, False, 5e-6, None, 6e-4, 0.24),
            row('sphere_fit', 'raised_limit', 3, 1650, True, 3e-6, None, 4e-4, 0.21),
            row('sphere_fit', 'raised_limit', 4, 1700, False, 4.5e-6, None, 4.5e-4, 0.23),
            row('dipole_fit', 'raised_limit', 1, 1749, True, 4e-6, 300, None, 0.20),
            row('dipole_fit', 'raised_limit', 2, 1749, True, 4e-6, 300, None, 0.23),
            row('dipole_fit', 'mne_limit', 1, 1000, False, 1.5e-4, 299, None, 0.27)]     # a single trial
    ranges = tables.perturbation_ranges(rows)
    assert ranges == {
        'sphere_fit': {
            'mne_limit': dict(
                parameter_fit_evaluations=[1000.0, 1000.0], parameter_fit_objective=[1e-4, 2e-4],
                patch_median_relative_field_difference=[2e-3, 3e-3], position_error_median_mm=[0.25, 0.30],
                trials=2, trials_in_which_the_parameter_fit_stopped_by_itself=0),
            'raised_limit': dict(
                parameter_fit_evaluations=[1650.0, 100000.0], parameter_fit_objective=[3e-6, 5e-6],
                patch_median_relative_field_difference=[4e-4, 6e-4], position_error_median_mm=[0.21, 0.24],
                trials=4, trials_in_which_the_parameter_fit_stopped_by_itself=2)},
        # The trials of the dipole fits repeat no field comparison: the empty column is left out.
        'dipole_fit': {
            'raised_limit': dict(
                parameter_fit_evaluations=[1749.0, 1749.0], parameter_fit_objective=[4e-6, 4e-6],
                dipole_fits=[300.0, 300.0], position_error_median_mm=[0.20, 0.23],
                trials=2, trials_in_which_the_parameter_fit_stopped_by_itself=2),
            'mne_limit': dict(
                parameter_fit_evaluations=[1000.0, 1000.0], parameter_fit_objective=[1.5e-4, 1.5e-4],
                dipole_fits=[299.0, 299.0], position_error_median_mm=[0.27, 0.27],
                trials=1, trials_in_which_the_parameter_fit_stopped_by_itself=0)}}
    # The fits appear in the order of their first trial.
    assert list(ranges) == ['sphere_fit', 'dipole_fit']
    assert list(ranges['sphere_fit']) == ['mne_limit', 'raised_limit']
    assert list(ranges['dipole_fit']) == ['raised_limit', 'mne_limit']


def small_row(kernel, changed, name, step, stopped, dipole_fits, errors, changes=('', '', '')):
    """A row of small_tolerance.csv: a dipole recovery whose dipole fits ended at the small final radius.

    ``dipole_fits`` gives the number of dipole fits and how many of them stopped by themselves. ``errors`` and
    ``changes`` give the median, the 90th percentile and the maximum of the position errors and of the changes
    of the fitted positions; the unchanged run of the main process records no changes.
    """
    row = dict(kernel=kernel, changed_objective=changed, sphere_fit=name, objective_scale_minus_one_in_eps=str(step),
               parameter_fit_evaluations='1000', parameter_fit_stopped_by_itself=str(int(stopped)),
               dipole_fits=str(dipole_fits[0]), dipole_fits_stopped_by_themselves=str(dipole_fits[1]),
               dipole_fit_evaluations_median='76.5', dipole_fit_evaluations_maximum='92.0')
    for quantity, values in (('position_error', errors), ('fitted_position_change', changes)):
        row.update({f'{quantity}_{key}_mm': value if value == '' else repr(value)
                    for key, value in zip(('median', 'p90', 'maximum'), values)})
    return row


def test_run_with_the_small_final_radius_is_named_by_kernel_changed_objective_sphere_fit_and_factor():
    row = small_row('Haswell', 'dipole_fits', 'raised_limit', 3, True, (8, 8), (0.2, 0.3, 0.4), (1e-4, 2e-4, 3e-4))
    assert tables.small_run_key(row) == ('Haswell', 'dipole_fits', 'raised_limit', '3')
    # A row of the per-dipole file carries the same four labels; its other cells do not count.
    dipole = dict(kernel='Haswell', changed_objective='dipole_fits', sphere_fit='raised_limit',
                  objective_scale_minus_one_in_eps='3', sample_index='5', fitted_x_mm='1.0', position_error_mm='0.2')
    assert tables.small_run_key(dipole) == tables.small_run_key(row)
    # A run differs from another in any one of the four.
    others = [dict(dipole, kernel='Nehalem'), dict(dipole, changed_objective='parameter_fit'),
              dict(dipole, sphere_fit='mne_limit'), dict(dipole, objective_scale_minus_one_in_eps='2')]
    assert len({tables.small_run_key(item) for item in [dipole, *others]}) == 5


def test_ranges_over_the_runs_with_the_small_final_radius_by_sphere_fit():
    # The runs of the two sphere fits alternate, and the kernels do not appear in the order of the alphabet.
    rows = [small_row('Haswell', 'none', 'raised_limit', 0, True, (8, 8), (0.14, 0.27, 0.29)),
            small_row('Haswell', 'none', 'mne_limit', 0, False, (8, 8), (0.18, 0.29, 0.31)),
            small_row('Haswell', 'parameter_fit', 'raised_limit', 1, True, (8, 8), (0.13, 0.28, 0.30),
                      (2e-4, 3e-4, 5e-4)),
            small_row('Haswell', 'parameter_fit', 'mne_limit', 1, False, (8, 7), (0.19, 0.30, 0.32),
                      (0.02, 0.04, 0.07)),
            # A run in which one dipole moved: the median and the 90th percentile of the changes are zero.
            small_row('Haswell', 'dipole_fits', 'raised_limit', 1, True, (8, 8), (0.15, 0.26, 0.28), (0.0, 0.0, 4e-4)),
            small_row('Sandybridge', 'none', 'mne_limit', 0, False, (8, 8), (0.17, 0.31, 0.37), (0.03, 0.07, 0.08)),
            small_row('Nehalem', 'none', 'raised_limit', 0, True, (10, 9), (0.145, 0.275, 0.295), (2e-5, 5e-5, 9e-5)),
            small_row('Nehalem', 'none', 'mne_limit', 0, True, (8, 8), (0.185, 0.295, 0.30), (0.025, 0.05, 0.075)),
            # A sphere fit with its unchanged run only: no run records a change of the fitted positions.
            small_row('Haswell', 'none', 'third_fit', 0, True, (8, 8), (0.5, 0.6, 0.7))]
    ranges = tables.small_tolerance_ranges(rows)
    assert ranges == {
        'raised_limit': dict(
            runs=4, kernels=['Haswell', 'Nehalem'], runs_in_which_the_parameter_fit_stopped_by_itself=4,
            dipole_fits=34, dipole_fits_stopped_by_themselves=33,
            position_error_median_mm=[0.13, 0.15], position_error_p90_mm=[0.26, 0.28],
            position_error_maximum_mm=[0.28, 0.30],
            # A change of zero is a recorded value. The cells are compared as numbers: as text, 9e-05 would be
            # the largest of the three largest changes.
            fitted_position_change_median_mm=[0.0, 2e-4], fitted_position_change_p90_mm=[0.0, 3e-4],
            fitted_position_change_maximum_mm=[9e-5, 5e-4]),
        'mne_limit': dict(
            runs=4, kernels=['Haswell', 'Nehalem', 'Sandybridge'], runs_in_which_the_parameter_fit_stopped_by_itself=1,
            dipole_fits=32, dipole_fits_stopped_by_themselves=31,
            position_error_median_mm=[0.17, 0.19], position_error_p90_mm=[0.29, 0.31],
            position_error_maximum_mm=[0.30, 0.37],
            fitted_position_change_median_mm=[0.02, 0.03], fitted_position_change_p90_mm=[0.04, 0.07],
            fitted_position_change_maximum_mm=[0.07, 0.08]),
        'third_fit': dict(
            runs=1, kernels=['Haswell'], runs_in_which_the_parameter_fit_stopped_by_itself=1,
            dipole_fits=8, dipole_fits_stopped_by_themselves=8,
            position_error_median_mm=[0.5, 0.5], position_error_p90_mm=[0.6, 0.6], position_error_maximum_mm=[0.7, 0.7],
            fitted_position_change_median_mm=None, fitted_position_change_p90_mm=None,
            fitted_position_change_maximum_mm=None)}
    # The sphere fits appear in the order of their first run. The counts are whole numbers.
    assert list(ranges) == ['raised_limit', 'mne_limit', 'third_fit']
    assert all(type(ranges['mne_limit'][key]) is int
               for key in ('runs', 'runs_in_which_the_parameter_fit_stopped_by_itself', 'dipole_fits',
                           'dipole_fits_stopped_by_themselves'))
    # The evaluations of the fits are not summarized.
    assert not any('evaluations' in key for key in ranges['mne_limit'])
    assert tables.small_tolerance_ranges([]) == {}
    # Every kernel of a sphere fit is listed once, whatever the number and the order of its runs.
    kernels = ['Zen', 'Haswell', 'Sandybridge', 'Nehalem', 'Prescott', 'Atom', 'Skylakex', 'Core2', 'Haswell']
    runs = [small_row(kernel, 'none', 'mne_limit', 0, False, (8, 8), (0.2, 0.3, 0.4)) for kernel in kernels]
    assert tables.small_tolerance_ranges(runs)['mne_limit']['kernels'] == [
        'Atom', 'Core2', 'Haswell', 'Nehalem', 'Prescott', 'Sandybridge', 'Skylakex', 'Zen']


def test_sample_spread_must_equal_the_record():
    evidence = tables.Evidence(Path('.'))
    values = [1.0, 2.0, 3.0, 4.0, 100.0]
    record = dict(n=5, median=3.0, p90=61.6, maximum=100.0)     # 0.6 of the way from 4 to 100
    spread = tables.sample_spread(values)
    assert spread == pytest.approx(record, abs=1e-12) and type(spread['n']) is int
    tables.check_spread(evidence, values, record, 'example')
    assert evidence.checked == 3
    # A difference of 1e-13 is accepted and one of 1e-11 is not.
    tables.check_spread(evidence, values, dict(record, median=3.0 + 1e-13), 'example')
    for key in ('median', 'p90', 'maximum'):
        for difference in (1.0, -1e-9, 1e-9, 1e-11):
            with pytest.raises(ValueError, match=f'example, {key}: recomputed'):
                tables.check_spread(evidence, values, dict(record, **{key: record[key] + difference}), 'example')
    for size in (4, 6):
        with pytest.raises(ValueError, match=f'example: 5 values, recorded {size}'):
            tables.check_spread(evidence, values, dict(record, n=size), 'example')


def test_position_change_is_the_distance_between_the_fitted_positions_of_two_runs():
    reference = [[10.0, -20.0, 30.0], [1.0, 2.0, 3.0], [-5.0, 0.0, 7.0]]
    fitted = [[12.0, -17.0, 36.0], [1.0, 2.0, 3.0], [-5.0, 0.5, 7.0]]       # 2, 3 and 6 mm apart are 7 mm
    assert tables.position_change(fitted, reference).tolist() == [7.0, 0.0, 0.5]
    assert tables.position_change(np.array(reference), np.array(fitted)).tolist() == [7.0, 0.0, 0.5]


def test_change_of_the_fitted_positions_must_equal_the_record():
    evidence = tables.Evidence(Path('.'))
    moved = np.array([0.0, 0.05, 0.2, 0.4])                 # two of four positions moved by more than 0.1 mm
    record = dict(n=4, median=0.125, p90=0.34, maximum=0.4, share_above_0p1_mm=0.5)
    tables.check_position_change(evidence, moved, record, 'example')
    assert evidence.checked == 3
    # The share may differ in its last digits only.
    for share in (0.5 - 1e-13, 0.5 + 1e-13):
        tables.check_position_change(evidence, moved, dict(record, share_above_0p1_mm=share), 'example')
    for share in (0.25, 0.75, 0.5 - 1e-11, 0.5 + 1e-11):
        with pytest.raises(ValueError, match='example: the share of positions that moved by more than 0.1 mm'):
            tables.check_position_change(evidence, moved, dict(record, share_above_0p1_mm=share), 'example')
    with pytest.raises(ValueError, match='example, maximum: recomputed 0.4, recorded 0.5'):
        tables.check_position_change(evidence, moved, dict(record, maximum=0.5), 'example')
    with pytest.raises(ValueError, match='example: 4 values, recorded 5'):
        tables.check_position_change(evidence, moved, dict(record, n=5), 'example')


@pytest.mark.parametrize('middle, accepted, refused', [
    # A distance within 1e-9 mm of 0.1 mm may have been counted on either side.
    (0.1 + 5e-10, (1 / 3, 2 / 3), (0.0, 1.0)), (0.1 - 5e-10, (1 / 3, 2 / 3), (0.0, 1.0)),
    # A distance that is 2e-9 mm away from 0.1 mm may not.
    (0.1 + 2e-9, (2 / 3,), (1 / 3, 1.0)), (0.1 - 2e-9, (1 / 3,), (0.0, 2 / 3))])
def test_share_of_moved_positions_allows_for_a_distance_at_the_threshold(middle, accepted, refused):
    # Distances of 0.0, ``middle`` and 0.3 mm: the 90th percentile lies 0.8 of the way from the second to the third.
    evidence = tables.Evidence(Path('.'))
    moved = np.array([0.3, 0.0, middle])
    record = dict(n=3, median=middle, p90=middle + 0.8 * (0.3 - middle), maximum=0.3)
    for share in accepted:
        tables.check_position_change(evidence, moved, dict(record, share_above_0p1_mm=share), 'example')
    for share in refused:
        with pytest.raises(ValueError, match='example: the share of positions that moved by more than 0.1 mm'):
            tables.check_position_change(evidence, moved, dict(record, share_above_0p1_mm=share), 'example')


def test_merged_range_spans_the_smallest_lower_end_and_the_largest_upper_end():
    assert tables.merged_range([[2.0, 3.0], [1.0, 2.5], [1.5, 3.5]]) == [1.0, 3.5]
    # A single value is a range without width, as the unchanged run and every kernel contribute one.
    assert tables.merged_range([[0.27, 0.27], [0.26, 0.29], [0.28, 0.28]]) == [0.26, 0.29]
    assert tables.merged_range([[0.31, 0.31], [0.26, 0.29], [0.25, 0.25]]) == [0.25, 0.31]
    merged = tables.merged_range([[1000, 1000], [1672, 2080]])
    assert merged == [1000.0, 2080.0] and all(type(value) is float for value in merged)


# --------------------------------------------------------------------------- recorded fits of the dipole comparators

def fit_columns(prefix, position):
    return {f'{prefix}_{axis}_mm': repr(value) for axis, value in zip('xyz', position)}


def source_columns(position):
    """The true source position of a window, as the per-window files give it."""
    return {f'{axis}_mm': repr(value) for axis, value in zip('xyz', position)}


SUBSET_WINDOWS = (5, 2, 64)             # the windows of the dipole fits of the transfer and adaptation experiments


def comparator_tree(root, transfer_sources=(64, 0, 2, 9, 5), adaptation_sources=(2, 5, 64)):
    """Per-window files of the three experiments that recorded dipole fits: two or three windows each.

    The split study gives the true source position next to the fitted one. The other two experiments give it
    in an eligibility file of their own, which lists the windows in another order than the file of the fits;
    here the two files disagree on every source position, and the file of the transfer experiment holds more
    windows than were fitted. The errors that these two experiments recorded are no distances of the positions
    in the files, so that a recorded error cannot be taken for a recomputed one.
    """
    for (seed, policy), fits, sources, windows in (
            (('863', 'region'), ((11.5, -20.25, 30.125), (-4.0, 5.5, 60.75)),
             ((11.5, -20.25, 31.125), (-1.0, 9.5, 60.75)), [41, 7]),
            (('862', 'position'), ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)), ((1.0, 2.0, 3.0), (4.0, 5.0, 18.0)), [3, 9])):
        folder = root / 'split-study-v11' / seed / policy / 'ecd_subset'
        # The file numbers its rows from zero; the test windows of the subset stand in a file of their own.
        write_csv(folder / 'predictions.csv', [
            {'window': str(row), **source_columns(source), **fit_columns('dictionary', (90.0 + row, 91.0, 92.0)),
             **fit_columns('dipole', fit)} for row, (fit, source) in enumerate(zip(fits, sources))])
        (folder / 'test_rows.json').write_text(json.dumps(windows), encoding='utf-8')
    write_csv(root / 'head-transfer-v11/862/ecd_subset.csv', [
        {'window_id': str(window), 'ecd_package_sphere_matched_error_mm': repr(1.5 + window),
         'ecd_bem_matched_error_mm': repr(0.25 + window),
         **fit_columns('ecd_package_sphere_matched', (10.0 + window, 20.0, -30.0)),
         **fit_columns('ecd_bem_matched', (40.0 + window, 50.0, -60.0))} for window in SUBSET_WINDOWS])
    write_csv(root / 'head-transfer-v11/862/eligibility.csv', [
        {'window_id': str(window), **source_columns((window + 0.5, 21.0, -31.0)), 'eligible': 'True'}
        for window in transfer_sources])
    write_csv(root / 'head-adaptation-v11/862/dipole_subset.csv', [
        {'window_id': str(window), 'dipole_default_sphere_on_bem_error_mm': repr(4.5 + window),
         'dipole_size_matched_sphere_on_bem_error_mm': repr(3.5 + window),
         'dipole_size_matched_three_shell_sphere_on_bem_error_mm': repr(2.5 + window),
         **fit_columns('dipole_default', (1.0 + window, 0.5, -0.25)),
         **fit_columns('dipole_size_matched', (2.0 + window, 1.5, -1.25)),
         **fit_columns('dipole_size_matched_three_shell', (3.0 + window, 2.5, -2.25))} for window in SUBSET_WINDOWS])
    write_csv(root / 'head-adaptation-v11/862/eligibility.csv', [
        {'window_id': str(window), **source_columns((100.0 + window, 0.75, -0.5)), 'eligible': 'True'}
        for window in adaptation_sources])
    return tables.Evidence(root)


def test_recorded_comparator_gives_the_fit_the_sources_the_errors_and_the_windows_of_each_experiment(tmp_path):
    evidence = comparator_tree(tmp_path)
    # The split study: the dipole fit of one dataset under one policy, the true positions of the same file, the
    # distances between the two, and the test windows of the subset.
    positions, sources, errors, windows = tables.recorded_comparator(evidence, 'split_study', '863', 'region')
    assert positions.tolist() == [[11.5, -20.25, 30.125], [-4.0, 5.5, 60.75]]
    assert sources.tolist() == [[11.5, -20.25, 31.125], [-1.0, 9.5, 60.75]]
    assert errors.tolist() == [1.0, 5.0]                    # 1 mm along z; 3 and 4 mm along x and y
    assert windows == [41, 7] and all(type(window) is int for window in windows)
    positions, sources, errors, windows = tables.recorded_comparator(evidence, 'split_study', '862', 'position')
    assert positions.tolist() == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]] and windows == [3, 9]
    assert sources.tolist() == [[1.0, 2.0, 3.0], [4.0, 5.0, 18.0]] and errors.tolist() == [0.0, 12.0]
    # The transfer experiment: the fit that the case names and the error that the experiment recorded for it,
    # with the windows in the order of the file. The source of a window comes from the eligibility file of
    # the experiment, by the number of the window.
    positions, sources, errors, windows = tables.recorded_comparator(evidence, 'transfer', '862',
                                                                     'package_sphere_matched')
    assert positions.tolist() == [[15.0, 20.0, -30.0], [12.0, 20.0, -30.0], [74.0, 20.0, -30.0]]
    assert sources.tolist() == [[5.5, 21.0, -31.0], [2.5, 21.0, -31.0], [64.5, 21.0, -31.0]]
    assert errors.tolist() == [6.5, 3.5, 65.5] and errors.shape == (3,)
    assert windows == [5, 2, 64] and all(type(window) is int for window in windows)
    positions, sources, errors, _ = tables.recorded_comparator(evidence, 'transfer', '862', 'bem_matched')
    assert positions.tolist() == [[45.0, 50.0, -60.0], [42.0, 50.0, -60.0], [104.0, 50.0, -60.0]]
    assert sources.tolist() == [[5.5, 21.0, -31.0], [2.5, 21.0, -31.0], [64.5, 21.0, -31.0]]
    assert errors.tolist() == [5.25, 2.25, 64.25]
    # The adaptation experiment: the case names the sphere, and one sphere is not taken for another whose name
    # begins in the same way, neither in the positions nor in the errors. The sources come from the
    # eligibility file of this experiment, not from that of the transfer experiment.
    for case, first, error in (('default_sphere_on_bem', [6.0, 0.5, -0.25], 4.5),
                               ('size_matched_sphere_on_bem', [7.0, 1.5, -1.25], 3.5),
                               ('size_matched_three_shell_sphere_on_bem', [8.0, 2.5, -2.25], 2.5)):
        positions, sources, errors, windows = tables.recorded_comparator(evidence, 'adaptation', '862', case)
        assert positions[0].tolist() == first and positions.shape == (3, 3) and windows == [5, 2, 64], case
        assert sources.tolist() == [[105.0, 0.75, -0.5], [102.0, 0.75, -0.5], [164.0, 0.75, -0.5]], case
        assert errors.tolist() == [error + 5, error + 2, error + 64], case
    # Every file that was read has its hash recorded, and nothing was compared.
    assert sorted(evidence.hashes) == [
        'head-adaptation-v11/862/dipole_subset.csv', 'head-adaptation-v11/862/eligibility.csv',
        'head-transfer-v11/862/ecd_subset.csv', 'head-transfer-v11/862/eligibility.csv',
        *[f'split-study-v11/{dataset}/ecd_subset/{file}' for dataset in ('862/position', '863/region')
          for file in ('predictions.csv', 'test_rows.json')]]
    assert evidence.checked == 0


def test_recorded_comparator_knows_three_experiments_and_reads_the_dataset_that_is_named(tmp_path):
    evidence = comparator_tree(tmp_path)
    for experiment in ('patch_study', 'split', ''):
        with pytest.raises(ValueError, match=f'Unknown experiment of a dipole comparator: {experiment}$'):
            tables.recorded_comparator(evidence, experiment, '862', 'position')
    # Only the first dataset has dipole fits of the transfer and of the adaptation experiment.
    for experiment, case in (('transfer', 'package_sphere_matched'), ('adaptation', 'default_sphere_on_bem'),
                             ('split_study', 'region')):
        with pytest.raises(FileNotFoundError):
            tables.recorded_comparator(evidence, experiment, '864', case)
    # A fit that the file does not hold is not replaced by another one.
    for experiment, case in (('transfer', 'independent_sphere'), ('adaptation', 'three_shell_sphere_on_bem')):
        with pytest.raises(KeyError):
            tables.recorded_comparator(evidence, experiment, '862', case)


@pytest.mark.parametrize('windows', [[41], [41, 7, 12], []])
def test_recorded_comparator_needs_one_window_number_for_every_dipole_fit_of_the_split_study(tmp_path, windows):
    evidence = comparator_tree(tmp_path)
    (tmp_path / 'split-study-v11/863/region/ecd_subset/test_rows.json').write_text(json.dumps(windows),
                                                                                    encoding='utf-8')
    with pytest.raises(ValueError, match='^split study, dataset 863, region: the dipole-fit subset holds 2 rows and '
                                         f'{len(windows)} window numbers$'):
        tables.recorded_comparator(evidence, 'split_study', '863', 'region')
    # The other dataset has a number for each of its fits.
    assert tables.recorded_comparator(evidence, 'split_study', '862', 'position')[3] == [3, 9]


@pytest.mark.parametrize('sources', [
    (5, 2), (64, 5),                            # a fitted window is missing
    (5, 2, 63), (63, 5, 2, 0, 9, 71),           # and other windows do not make up for it
    ()])
def test_recorded_comparator_needs_the_source_position_of_every_fitted_window(tmp_path, sources):
    for experiment, case, changed in (('transfer', 'package_sphere_matched', dict(transfer_sources=sources)),
                                      ('adaptation', 'size_matched_sphere_on_bem', dict(adaptation_sources=sources))):
        folder = tmp_path / experiment
        if sources:
            evidence = comparator_tree(folder, **changed)
        else:
            # A file with its header and without any window.
            evidence = comparator_tree(folder)
            file = folder / f'head-{experiment}-v11/862/eligibility.csv'
            file.write_text(file.read_text(encoding='utf-8').splitlines()[0] + '\n', encoding='utf-8')
        with pytest.raises(ValueError, match=f'^{experiment}, dataset 862: the record lacks the source position of '
                                             'a fitted window$'):
            tables.recorded_comparator(evidence, experiment, '862', case)


def test_recorded_comparator_takes_the_sources_by_window_number_whatever_else_the_file_holds(tmp_path):
    # The eligibility file may hold exactly the fitted windows, in any order.
    evidence = comparator_tree(tmp_path / 'exact', transfer_sources=(64, 2, 5), adaptation_sources=(5, 64, 2))
    for experiment, case, first in (('transfer', 'bem_matched', 0.5), ('adaptation', 'default_sphere_on_bem', 100.0)):
        _, sources, _, windows = tables.recorded_comparator(evidence, experiment, '862', case)
        assert windows == [5, 2, 64] and sources[:, 0].tolist() == [first + 5, first + 2, first + 64], experiment
    # A window that the file of the fits holds twice has its source twice, also when the eligibility file then
    # holds fewer rows than the file of the fits: whether the windows are the right ones is decided elsewhere.
    evidence = comparator_tree(tmp_path / 'twice', transfer_sources=(64, 2, 5))
    file = tmp_path / 'twice/head-transfer-v11/862/ecd_subset.csv'
    lines = file.read_text(encoding='utf-8').splitlines()
    file.write_text('\n'.join([*lines, lines[1]]) + '\n', encoding='utf-8')
    positions, sources, errors, windows = tables.recorded_comparator(evidence, 'transfer', '862',
                                                                     'package_sphere_matched')
    assert windows == [5, 2, 64, 5] and sources[:, 0].tolist() == [5.5, 2.5, 64.5, 5.5]
    assert positions.shape == (4, 3) and errors.tolist() == [6.5, 3.5, 65.5, 6.5]


@pytest.mark.parametrize('sources', [
    (64, 2, 5, 5), (5, 64, 2, 5),               # a fitted window twice
    (9, 64, 0, 2, 9, 5),                        # a window that was not fitted twice
    (5, 5, 5)])                                 # one window three times and no other: the first complaint is this one
def test_recorded_comparator_refuses_an_eligibility_file_that_names_a_window_twice(tmp_path, sources):
    for experiment, case, changed in (('transfer', 'package_sphere_matched', dict(transfer_sources=sources)),
                                      ('adaptation', 'size_matched_sphere_on_bem', dict(adaptation_sources=sources))):
        evidence = comparator_tree(tmp_path / experiment, **changed)
        with pytest.raises(ValueError, match=f'^{experiment}, dataset 862: the record names a window twice$'):
            tables.recorded_comparator(evidence, experiment, '862', case)


def test_recorded_comparator_counts_a_window_by_its_number_whatever_its_spelling(tmp_path):
    """The window 5 written once as 5 and once as 05 stands twice in the eligibility file."""
    for experiment, case in (('transfer', 'package_sphere_matched'), ('adaptation', 'size_matched_sphere_on_bem')):
        evidence = comparator_tree(tmp_path / experiment)
        file = tmp_path / experiment / f'head-{experiment}-v11/862/eligibility.csv'
        lines = file.read_text(encoding='utf-8').splitlines()
        again = next(line for line in lines[1:] if line.split(',')[0] == '5')
        file.write_text('\n'.join([*lines, '0' + again]) + '\n', encoding='utf-8')
        with pytest.raises(ValueError, match=f'^{experiment}, dataset 862: the record names a window twice$'):
            tables.recorded_comparator(evidence, experiment, '862', case)


# --------------------------------------------------------------------------- the archived tables

# The number of comparisons of a recomputed number with a recorded one that the driver made for the archived
# tables. The driver prints it in its last line. Change it whenever the archive is regenerated.
ARCHIVED_COMPARISONS = 8978


def archived():
    return json.loads((ROOT / 'validation_results/revision_tables.json').read_text(encoding='utf-8'))


def test_archived_tables_come_from_the_shipped_driver():
    report = archived()
    script = hashlib.sha256((ROOT / 'scripts/make_revision_tables.py').read_bytes()).hexdigest()
    assert report['provenance']['script_sha256'] == script
    assert type(report['provenance']['comparisons_of_recomputed_numbers_with_the_records']) is int


def test_archived_tables_record_the_number_of_comparisons():
    assert archived()['provenance']['comparisons_of_recomputed_numbers_with_the_records'] == ARCHIVED_COMPARISONS


def test_archived_tables_hold_every_part_of_the_reported_tables():
    report = archived()
    for row in report['split']['strata']:
        for policy in tables.POLICIES:
            entry = row[policy]
            assert (entry['mean_of_seed_medians_mm'] is None) == (entry['dictionary_mean_of_seed_medians_mm'] is None)
    assert set(report['split']['paired_split_by_dataset_seed']) == set(SEEDS)
    offsets = report['split']['training_mean_offset_mm']
    assert set(offsets) == set(tables.POLICIES) and all(set(offsets[policy]) == set(SEEDS) for policy in offsets)
    transfer = report['transfer']
    assert [row['stratum'] for row in transfer['window_strata']] == [name for name, _, _ in tables.TRANSFER_STRATA]
    assert all(row['smallest_number_of_positions_in_a_dataset'] >= tables.MINIMUM_POSITIONS
               and row['mean_difference_mm'] is not None for row in transfer['window_strata'])
    assert [row['stratum'] for row in transfer['eligibility_census']] == [name for name, _, _ in tables.CENSUS_STRATA]
    assert set(transfer['independent_sphere_seed_medians_mm']) == set(SEEDS)
    assert np.isclose(np.mean(list(transfer['independent_sphere_seed_medians_mm'].values())),
                      transfer['independent_sphere_mean_seed_median_mm'])
    census = {row['stratum']: row['total_windows'] for row in transfer['eligibility_census']}
    assert census['all'] == census['cortical'] + census['subcortical']
    assert transfer['cortical_outside_inner_skull']['total_windows'] == census['cortical']
    for row in report['patch']['audits']:
        assert row['minimum_used_position_distance_mm'] > 0
        if row['policy'] == 'region':
            assert row['minimum_used_position_distance_mm'] == pytest.approx(row['minimum_support_clearance_mm'])
            assert row['minimum_used_position_distance_mm'] >= report['settings']['patch']['buffer_mm']
    settings = report['settings']
    assert settings['windows'] == dict(train=20000, validation=2000, test=3000)
    assert settings['transfer']['minimum_inner_skull_clearance_mm'] == 5.0
    assert settings['dipole_fit_subset_windows'] == dict(split_study_per_dataset=500, transfer_and_adaptation=500)
    for row in report['benchmark']['rows']:
        setup = row['setup_seconds_of_all_runs']
        assert setup['minimum'] <= row['median_run']['setup_seconds'] <= setup['maximum']


def test_archived_tables_hold_the_strata_of_each_fitted_network():
    """The record of a fitted network in the tables holds its strata, which the driver compared one by one."""
    report = archived()
    rows = report['split']['per_fit']
    assert [(row['dataset_seed'], row['policy'], row['fit_seed']) for row in rows] == [
        (seed, policy, fit) for seed in SEEDS for policy in tables.POLICIES for fit in tables.FITS]
    labels = {row['stratum'] for row in report['split']['strata']}
    statistics = 0
    for first, second in zip(rows[::2], rows[1::2]):
        # The two networks of a dataset have the same strata, each with the same windows and positions and with
        # other errors. The strata are among those of the table of the strata.
        strata = first['metrics']['strata']
        assert list(strata) == list(second['metrics']['strata']) and set(strata) <= labels
        for label, entry in strata.items():
            other = second['metrics']['strata'][label]
            assert (entry['windows'], entry['positions']) == (other['windows'], other['positions'])
            assert 0 < entry['positions'] <= entry['windows'] <= first['metrics']['windows']
            for item in (entry, other):
                assert 0.0 < item['median_mm'] <= item['p90_mm'] and item['mean_mm'] > 0.0
            assert entry['median_mm'] != other['median_mm']
            statistics += 2 * 3
    # Three statistics of two networks in each of the fifteen strata of six datasets. Every stratum of the table
    # is a stratum of some dataset.
    assert statistics == 3 * 2 * 15 * 6
    assert {label for row in rows for label in row['metrics']['strata']} == labels


def test_archived_tables_record_the_rerun_and_the_sphere_fit():
    report = archived()
    rerun = report['validation_rerun']
    digests = rerun['array_digests_of_the_two_releases']
    assert digests['arrays_in_both'] == digests['arrays_with_equal_digest'] > 0
    assert not digests['arrays_only_in_first'] and not digests['arrays_only_in_second']
    assert all(rerun['two_releases_on_one_machine'].values())
    platforms = rerun['two_platforms']
    assert platforms['same_sources_and_realized_snr'] and platforms['transfer_coefficients_equal']
    assert len(platforms['sweep_median_position_error_mm']) == 7
    moved = platforms['fitted_position_difference_mm']
    assert moved['n'] == platforms['dipoles'] and 0.0 <= moved['share_above_0p1_mm'] <= 1.0
    assert 0.0 <= moved['median'] <= moved['p90'] <= moved['maximum']
    fit = report['sphere_fit']
    assert set(fit) >= {'head_model', 'optimizer_call', 'evaluation_limits', 'sizes', 'arms',
                        'default_sphere_model_of_mne_python', 'dipole_fits', 'small_tolerance', 'perturbation',
                        'library_kernels', 'spheres', 'all_runs', 'links', 'software_environment'}
    assert set(fit['optimizer_call']) == {'function', 'arguments', 'evaluation_limit'}
    assert list(fit['arms']) == ['mne_limit', 'raised_limit']
    limit, raised = fit['arms']['mne_limit']['fit'], fit['arms']['raised_limit']['fit']
    assert limit['evaluations'] == fit['evaluation_limits']['mne_limit'] and not limit['stopped_by_itself']
    assert raised['stopped_by_itself'] and raised['objective'] < limit['objective']
    assert all(fit['links'].values()) and set(fit['links']) == {'same_datasets_as_the_validation_rerun',
                                                               'mne_limit_arm_equals_the_validation_rerun'}
    ranges = fit['perturbation']['ranges']
    assert set(ranges) == {'sphere_fit', 'dipole_fit'}
    for by_arm in ranges.values():
        assert set(by_arm) == set(fit['arms'])
        for entry in by_arm.values():
            assert entry['trials'] == len(fit['perturbation']['steps'])
            assert 0 <= entry['trials_in_which_the_parameter_fit_stopped_by_itself'] <= entry['trials']
    assert set(fit['dipole_fits']) == set(fit['arms'])
    for entry in fit['library_kernels']['arms'].values():
        assert set(entry['arms']) == set(fit['arms'])
        assert set(entry['fitted_position_change_from_the_parent_mm']) == set(fit['arms'])
        assert set(entry['small_tolerance']) == set(fit['arms'])
        for arm in entry['arms'].values():
            assert set(arm['dipole_recovery']) == {'position_error_mm', 'orientation_error_deg',
                                                   'moment_relative_error', 'goodness_of_fit_percent'}
    assert fit['spheres']['variants'] and all(
        {'mne_limit', 'raised_limit', 'position_error_change_mm'} <= set(entry)
        for entry in fit['spheres']['variants'].values())
    for entry in fit['spheres']['variants'].values():
        moved = entry['position_error_change_mm']['fitted_position']
        assert moved['n'] == entry['mne_limit']['position_error_mm']['n']
        assert 0.0 <= moved['median'] <= moved['p90'] <= moved['maximum'] and 0.0 <= moved['share_above_0p1_mm'] <= 1.0


def test_archived_runs_with_the_small_final_radius_follow_from_the_archived_list_of_runs():
    fit = archived()['sphere_fit']
    small, kernels = fit['small_tolerance'], fit['library_kernels']['arms']
    assert set(small) == {'final_radius_m', 'final_radius_of_mne_python_m', 'steps', 'kernel_of_the_main_process',
                          'ranges', 'runs'}
    assert 0.0 < small['final_radius_m'] < small['final_radius_of_mne_python_m']
    assert small['steps'] and set(small['steps']) <= set(fit['perturbation']['steps'])
    # The trials that the tables name are those of the runs with a changed objective, in the order of their
    # numbers; an unchanged run has the number 0. The kernel of the main process is a name.
    assert small['steps'] == sorted({int(run['objective_scale_minus_one_in_eps']) for run in small['runs']
                                     if run['changed_objective'] != 'none'})
    assert all(run['objective_scale_minus_one_in_eps'] == '0' for run in small['runs']
               if run['changed_objective'] == 'none')
    assert type(small['kernel_of_the_main_process']) is str and small['kernel_of_the_main_process'] not in kernels
    assert set(small['ranges']) == set(fit['arms'])
    statistics = [f'{quantity}_{key}_mm' for quantity in ('position_error', 'fitted_position_change')
                  for key in ('median', 'p90', 'maximum')]
    for name, entry in small['ranges'].items():
        runs = [run for run in small['runs'] if run['sphere_fit'] == name]
        # The unchanged run, a run for each changed objective of either kind, and a run under every other kernel.
        assert entry['runs'] == len(runs) == 1 + 2 * len(small['steps']) + len(kernels)
        assert entry['kernels'] == sorted({run['kernel'] for run in runs}) and set(entry['kernels']) > set(kernels)
        assert entry['dipole_fits'] == sum(int(run['dipole_fits']) for run in runs)
        assert entry['dipole_fits'] == len(runs) * fit['sizes']['dipoles']
        assert 0 <= entry['dipole_fits_stopped_by_themselves'] <= entry['dipole_fits']
        assert entry['runs_in_which_the_parameter_fit_stopped_by_itself'] == sum(
            int(run['parameter_fit_stopped_by_itself']) for run in runs)
        for column in statistics:
            values = [float(run[column]) for run in runs if run[column] != '']
            assert entry[column] == [min(values), max(values)], (name, column)
        # Only the unchanged run of the main process records no change of the fitted positions: it leaves the
        # three statistics of the change empty, and no other run leaves one of them empty.
        first = [run for run in runs if run['fitted_position_change_median_mm'] == '']
        assert [(run['changed_objective'], run['objective_scale_minus_one_in_eps']) for run in first] == [('none', '0')]
        assert {first[0]['kernel']} == set(entry['kernels']) - set(kernels) == {small['kernel_of_the_main_process']}
        assert all((run[column] == '') == (run is first[0]) for run in runs
                   for column in statistics if column.startswith('fitted_position_change'))
    # The unchanged run under every other kernel is in the list with the statistics of the record.
    for kernel, entry in kernels.items():
        for name in fit['arms']:
            run, = [run for run in small['runs'] if (run['kernel'], run['changed_objective'], run['sphere_fit']) == (
                kernel, 'none', name)]
            recorded = entry['small_tolerance'][name]
            for key in ('median', 'p90', 'maximum'):
                assert float(run[f'position_error_{key}_mm']) == pytest.approx(recorded['position_error_mm'][key],
                                                                             abs=1e-12)
                assert float(run[f'fitted_position_change_{key}_mm']) == pytest.approx(
                    recorded['fitted_position_change_from_the_parent_mm'][key], abs=1e-9)


def test_archived_tables_record_the_dipole_comparators_with_a_converged_parameter_fit():
    report = archived()
    comparators, fit = report['comparators'], report['sphere_fit']
    assert set(comparators) == {'windows_per_subset', 'evaluation_limits',
                                'default_limit_fits_are_those_of_the_recorded_experiments', 'groups',
                                'split_study_means', 'largest_change_of_a_median_mm'}
    assert comparators['evaluation_limits'] == fit['evaluation_limits']
    # The driver writes the tables only if the fits with the limit of SciPy are the recorded ones.
    assert comparators['default_limit_fits_are_those_of_the_recorded_experiments'] is True
    groups = comparators['groups']
    # Every dataset and policy of the split study, the subset of the transfer experiment, and every sphere of
    # the adaptation experiment.
    assert [(group['experiment'], group['dataset_seed'], group['case']) for group in groups] == [
        *[('split_study', seed, policy) for seed in SEEDS for policy in tables.POLICIES],
        ('transfer', SEEDS[0], 'package_sphere_matched'),
        *[('adaptation', SEEDS[0], f'{sphere}_sphere_on_bem') for sphere in fit['spheres']['variants']]]
    subsets = report['settings']['dipole_fit_subset_windows']
    for group in groups:
        split = group['experiment'] == 'split_study'
        assert group['sphere'] == (group['case'][:-len('_sphere_on_bem')] if group['experiment'] == 'adaptation'
                                   else 'default')
        for name in fit['arms']:
            assert set(group[name]) == {'windows', 'median_mm', 'mean_mm', 'p90_mm'}
            # Every fit has the one number of windows that the record names, which is the size of the
            # dipole-fit subset of its experiment.
            assert group[name]['windows'] == comparators['windows_per_subset']
            assert group[name]['windows'] == subsets['split_study_per_dataset' if split else 'transfer_and_adaptation']
            # The parameter fit ended as for the same sphere in the sphere-fit tables.
            ended, recorded = group['parameter_fit'][name], fit['spheres']['variants'][group['sphere']][name]['fit']
            assert (ended['evaluations'], ended['stopped_by_itself']) == (recorded['evaluations'],
                                                                          recorded['stopped_by_itself'])
            assert ended['objective'] == pytest.approx(recorded['objective'], rel=1e-9)
        # As MNE-Python runs it, the parameter fit ended at the limit of SciPy; with the raised limit it stopped
        # by itself.
        limit, raised = group['parameter_fit']['mne_limit'], group['parameter_fit']['raised_limit']
        assert limit['stopped_by_itself'] is False and raised['stopped_by_itself'] is True
        assert limit['evaluations'] == comparators['evaluation_limits']['mne_limit']
        assert group['change_of_the_median_mm'] == pytest.approx(
            group['raised_limit']['median_mm'] - group['mne_limit']['median_mm'], abs=1e-10)
        moved = group['fitted_position_change_mm']
        assert 0.0 <= moved['median'] <= moved['p90'] <= moved['maximum']
    assert comparators['largest_change_of_a_median_mm'] == max(abs(group['change_of_the_median_mm'])
                                                               for group in groups)
    # The matched-sphere fit of the split study: mean and range of the dataset medians. With the limit of SciPy
    # the mean is the one of the localization comparison.
    for policy in tables.POLICIES:
        for name in fit['arms']:
            medians = [group[name]['median_mm'] for group in groups
                       if (group['experiment'], group['case']) == ('split_study', policy)]
            entry = comparators['split_study_means'][policy][name]
            assert entry['mean_of_dataset_medians_mm'] == pytest.approx(np.mean(medians), abs=1e-12)
            assert entry['range_mm'] == [min(medians), max(medians)]
        headline = report['split']['table11'][policy]['ecd_subset']['dipole']
        assert comparators['split_study_means'][policy]['mne_limit']['mean_of_dataset_medians_mm'] == pytest.approx(
            headline['mean_median_mm'], abs=1e-10)


def test_archived_ranges_over_every_run_of_a_sphere_fit_follow_from_the_archived_runs():
    fit = archived()['sphere_fit']
    ranges, kernels = fit['perturbation']['ranges'], fit['library_kernels']['arms']
    assert set(fit['all_runs']) == set(fit['arms'])
    for name, runs in fit['all_runs'].items():
        # The unchanged run and one run with every other kernel, the trials of the parameter fit, and the
        # trials of the dipole fits, which repeat no field comparison and no parameter fit.
        own = [fit['arms'][name], *[entry['arms'][name] for entry in kernels.values()]]
        parameter, dipole = ranges['sphere_fit'][name], ranges['dipole_fit'][name]
        assert runs['parameter_fits'] == runs['runs_with_a_field_comparison'] == len(own) + parameter['trials']
        assert runs['runs_with_a_dipole_recovery'] == len(own) + parameter['trials'] + dipole['trials']
        for key in ('median', 'p90', 'maximum'):
            values = [item['dipole_recovery']['position_error_mm'][key] for item in own]
            values += parameter[f'position_error_{key}_mm'] + dipole[f'position_error_{key}_mm']
            assert runs['position_error_mm'][key] == [min(values), max(values)]
        for source in ('patch', 'point'):
            values = [item['patch_comparison'][source]['median'] for item in own]
            values += parameter[f'{source}_median_relative_field_difference']
            assert runs['median_relative_field_difference'][source] == [min(values), max(values)]
        evaluations = [item['fit']['evaluations'] for item in own] + parameter['parameter_fit_evaluations']
        assert runs['parameter_fit_evaluations'] == [min(evaluations), max(evaluations)]
        assert runs['parameter_fits_that_stopped_by_themselves'] == sum(
            item['fit']['stopped_by_itself'] for item in own
        ) + parameter['trials_in_which_the_parameter_fit_stopped_by_itself']


def test_markdown_prints_the_main_tables(tmp_path):
    report = archived()
    tables.write_markdown(report, tmp_path / 'tables.md')
    text = (tmp_path / 'tables.md').read_text(encoding='utf-8')
    assert text.startswith('# Reproduced experiment statistics')
    for heading in ('## Localization comparison', '## Network and lead-field dictionary by stratum',
                    '## Transfer eligibility census', '## Transfer errors by stratum', '## Patch split comparison',
                    '## Generation benchmark', '## Numerical validation on two platforms and with two releases',
                    '## Parameter fit of the sphere model in MNE-Python',
                    '## Dipole-fit comparators with a converged parameter fit'):
        assert heading in text
    lines = text.splitlines()
    start = lines.index('## Transfer eligibility census')
    census = [line for line in lines[start + 4:start + 4 + len(tables.CENSUS_STRATA) + 1]]
    assert len(census) == len(tables.CENSUS_STRATA) + 1 and census[-1].startswith('| cortical_outside_inner_skull |')
    headline = report['split']['table11']['position']['full']['mlp_pooled']
    assert f"| mlp_pooled | {headline['mean_median_mm']:.2f} (" in text
    # One row of three more tables, cell by cell.
    row = report['transfer']['eligibility_census'][0]
    cells = [f"{row['by_dataset_seed'][seed]['eligible_windows']}/{row['by_dataset_seed'][seed]['total_windows']}"
             for seed in SEEDS]
    assert census[0] == (f"| all | {' | '.join(cells)} | {row['eligible_windows']}/{row['total_windows']} "
                         f"({row['fraction']:.1%}) |")
    row = report['transfer']['window_strata'][0]
    assert (f"| {row['stratum']} | {row['windows']} | {row['package_sphere_mean_of_seed_medians_mm']:.2f} | "
            f"{row['bem_mean_of_seed_medians_mm']:.2f} | {row['mean_difference_mm']:.2f} |") in lines
    entry = report['validation_rerun']['two_platforms']['position_error_mm']['maximum']
    assert f"| Position error, maximum (mm) | {entry['linux']:.3f} | {entry['windows']:.3f} |" in lines
    # The runs with the small final radius, and the first group of the dipole comparators.
    small = report['sphere_fit']['small_tolerance']
    ranges = [small['ranges'][name] for name in report['sphere_fit']['arms']]
    assert f"| Runs | {ranges[0]['runs']} | {ranges[1]['runs']} |" in lines
    cells = [f"{entry['position_error_median_mm'][0]:.4f}–{entry['position_error_median_mm'][1]:.4f}"
             for entry in ranges]
    assert f'| Position error at 35 dB, median (mm) | {cells[0]} | {cells[1]} |' in lines
    group = report['comparators']['groups'][0]
    assert (f"| {group['experiment']} | {group['dataset_seed']} | {group['case']} | {group['mne_limit']['windows']} | "
            f"{group['mne_limit']['median_mm']:.3f} | {group['raised_limit']['median_mm']:.3f} | "
            f"{group['change_of_the_median_mm']:+.3f} |") in lines
    assert len([line for line in lines if line.startswith(('| split_study |', '| transfer |', '| adaptation |'))]) == \
        len(report['comparators']['groups'])
