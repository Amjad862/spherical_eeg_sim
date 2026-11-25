"""The instrumented fits of scripts/run_sphere_fit_convergence.py, its records, and a small complete run."""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

mne = pytest.importorskip('mne')
bem = pytest.importorskip('mne.bem')
dipole = pytest.importorskip('mne.dipole')
scipy = pytest.importorskip('scipy')
# The script replaces the optimizer call inside MNE-Python 1.12 and belongs to the COBYLA of SciPy 1.16 and later,
# with which every record was made. MNE-Python 1.13 imports the optimizer inside its functions.
if not (hasattr(bem, 'fmin_cobyla') and hasattr(dipole, 'fmin_cobyla')):
    pytest.skip('the script replaces the optimizer call inside MNE-Python 1.12', allow_module_level=True)
if tuple(int(part) for part in scipy.__version__.split('.')[:2]) < (1, 16):
    pytest.skip('the records of the script belong to the COBYLA of SciPy 1.16 and later', allow_module_level=True)

from scipy.optimize import fmin_cobyla  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_sphere_fit_convergence as study  # noqa: E402

START = np.array([-1.2, 1.0])


def valley(x):
    """A curved valley that COBYLA needs a few hundred evaluations to follow."""
    return float((1.0 - x[0]) ** 2 + 30.0 * (x[1] - x[0] ** 2) ** 2)


def inside(x):
    return 2.0 - np.abs(x)


def inside_scaled(x, bound):
    return bound - np.abs(x)


@pytest.mark.parametrize('module, block', [(bem, study.instrumented_fit), (dipole, study.instrumented_dipole_fit)])
def test_block_replaces_the_optimizer_call_and_restores_it(module, block):
    arguments = (50,) if block is study.instrumented_fit else ()
    original = module.fmin_cobyla
    with block(*arguments):
        assert module.fmin_cobyla is not original
    assert module.fmin_cobyla is original
    with pytest.raises(RuntimeError, match='stop'):
        with block(*arguments):
            raise RuntimeError('stop')
    assert module.fmin_cobyla is original


@pytest.mark.parametrize('limit, finished', [(40, False), (5000, True)])
def test_replacement_makes_the_same_call_as_fmin_cobyla_and_reports_how_it_ended(limit, finished):
    expected = fmin_cobyla(valley, START, [inside], rhobeg=0.5, rhoend=1e-5, catol=1e-6, maxfun=limit)
    log = []
    with study.instrumented_fit(limit, log=log):
        # The call of mne.bem._fwd_eeg_fit_berg_scherg.
        found = bem.fmin_cobyla(valley, START, [inside], rhobeg=0.5, rhoend=1e-5, catol=1e-6)
    assert np.array_equal(found, expected)
    assert len(log) == 1
    record = log[0]
    assert record['stopped_by_itself'] is finished
    assert record['objective'] == valley(found)
    if finished:
        assert record['evaluations'] < limit
        assert np.allclose(found, [1.0, 1.0], atol=1e-2)
    else:
        assert record['evaluations'] == limit


def test_dipole_replacement_makes_the_same_call_as_fmin_cobyla():
    # The call of mne.dipole._fit_dipole, with a constraint that takes an argument.
    expected = fmin_cobyla(valley, START, (inside_scaled,), consargs=(2.0,), rhobeg=5e-2, rhoend=5e-5, disp=False)
    log = []
    with study.instrumented_dipole_fit(log=log):
        found = dipole.fmin_cobyla(valley, START, (inside_scaled,), consargs=(2.0,), rhobeg=5e-2, rhoend=5e-5,
                                   disp=False)
    assert np.array_equal(found, expected)
    assert len(log) == 1 and log[0]['evaluations'] <= study.MNE_LIMIT
    assert isinstance(log[0]['stopped_by_itself'], bool)
    assert log[0]['rhoend'] == 5e-5            # the final radius that the caller asked for


def test_replacement_refuses_an_evaluation_limit_from_its_caller():
    # MNE-Python passes none; a call with one must not go unnoticed.
    with study.instrumented_fit(50):
        with pytest.raises(TypeError, match='maxfun'):
            bem.fmin_cobyla(valley, START, [inside], rhobeg=0.5, rhoend=1e-5, catol=1e-6, maxfun=200)


@pytest.mark.parametrize('block, call', [
    (lambda log: study.instrumented_fit(5000, scale=-1.0, log=log),
     lambda: bem.fmin_cobyla(valley, START, [inside], rhobeg=0.5, rhoend=1e-5, catol=1e-6)),
    (lambda log: study.instrumented_dipole_fit(scale=-1.0, log=log),
     lambda: dipole.fmin_cobyla(valley, START, (inside,), consargs=(), rhobeg=0.5, rhoend=1e-5, disp=False)),
])
def test_scale_multiplies_the_objective_that_the_optimizer_sees(block, call):
    # With a factor of -1 the optimizer climbs instead: it ends on the boundary, far from the minimum at (1, 1).
    log = []
    with block(log):
        summit = call()
    assert np.abs(summit).max() == pytest.approx(2.0, abs=1e-3)
    assert valley(summit) > 100.0
    if 'objective' in log[0]:
        # The record holds the value of the function itself at the end point, not the scaled one.
        assert log[0]['objective'] == valley(summit)


def test_every_parameter_fit_records_its_first_evaluations():
    calls, log = [], []

    def counted(x):
        calls.append([float(item) for item in x])
        return valley(x)

    with study.instrumented_fit(5000, log=log):
        bem.fmin_cobyla(counted, START, [inside], rhobeg=0.5, rhoend=1e-5, catol=1e-6)
    record = log[0]
    # One call for every counted evaluation, and one more for the value at the end point.
    assert len(calls) == record['evaluations'] + 1
    assert record['call'] == dict(start=[-1.2, 1.0], constraints=1, rhobeg=0.5, rhoend=1e-5, catol=1e-6)
    first = record['first_evaluations']
    assert len(first) == study.FIRST_EVALUATIONS
    assert [entry['mu'] for entry in first] == calls[:study.FIRST_EVALUATIONS]
    assert first[0] == dict(mu=[-1.2, 1.0], objective=valley(START))
    assert first[1]['mu'] == [-0.7, 1.0] and first[1]['objective'] == valley([-0.7, 1.0])


def test_replace_hands_the_optimizer_another_value_at_one_evaluation():
    expected = fmin_cobyla(valley, START, [inside], rhobeg=0.5, rhoend=1e-5, catol=1e-6, maxfun=60)
    unchanged, replaced, same_value = [], [], []
    with study.instrumented_fit(60, log=unchanged):
        plain = bem.fmin_cobyla(valley, START, [inside], rhobeg=0.5, rhoend=1e-5, catol=1e-6)
    with study.instrumented_fit(60, log=replaced, replace={2: 1e6}):
        other = bem.fmin_cobyla(valley, START, [inside], rhobeg=0.5, rhoend=1e-5, catol=1e-6)
    with study.instrumented_fit(60, log=same_value, replace={2: valley([-0.7, 1.0])}):
        again = bem.fmin_cobyla(valley, START, [inside], rhobeg=0.5, rhoend=1e-5, catol=1e-6)
    assert np.array_equal(plain, expected)
    # The optimizer saw the replacement, at the second evaluation only, and took another path.
    assert replaced[0]['first_evaluations'][1] == dict(mu=[-0.7, 1.0], objective=1e6)
    assert replaced[0]['first_evaluations'][0] == unchanged[0]['first_evaluations'][0]
    assert not np.array_equal(other, plain)
    # The record still holds the value of the function itself at the end point.
    assert replaced[0]['objective'] == valley(other)
    # Replacing a value by itself changes nothing.
    assert np.array_equal(again, plain) and same_value == unchanged


def test_spread_reports_median_ninetieth_percentile_and_maximum():
    values = [1.0, 2.0, 3.0, 4.0, 100.0]
    assert study.spread(values) == dict(n=5, median=3.0, p90=float(np.percentile(values, 90)), maximum=100.0)


def test_dipole_fit_record_counts_the_fits_that_stopped_by_themselves():
    log = [dict(evaluations=40, stopped_by_itself=True), dict(evaluations=1000, stopped_by_itself=False),
           dict(evaluations=44, stopped_by_itself=True)]
    assert study.dipole_fit_record(log) == dict(
        fits=3, stopped_by_themselves=2,
        evaluations=dict(n=3, median=44.0, p90=float(np.percentile([40, 1000, 44], 90)), maximum=1000.0))


def test_position_change_is_the_distance_between_fitted_positions_in_mm():
    reference = np.array([[0.01, 0.02, 0.03], [0.0, 0.0, 0.05], [0.04, 0.0, 0.0], [0.0, 0.06, 0.0]])
    moved = reference + np.array([[0.0, 0.0, 0.0], [3e-5, 0.0, 4e-5], [0.0, 1e-4, 0.0], [2e-4, 0.0, 0.0]])
    change = study.position_change(moved, reference)
    # Distances of 0, 0.05, about 0.1 and 0.2 mm.
    assert change['n'] == 4
    assert change['median'] == pytest.approx(0.075) and change['maximum'] == pytest.approx(0.2)
    assert study.position_change(reference + [[0.0, 0.0, 1.5e-4]], reference)['share_above_0p1_mm'] == 1.0
    assert study.position_change(reference + [[0.0, 0.0, 0.5e-4]], reference)['share_above_0p1_mm'] == 0.0
    assert study.position_change(moved[[0, 1, 3]], reference[[0, 1, 3]])['share_above_0p1_mm'] == pytest.approx(1 / 3)


def summary(evaluations, objective, error_median, patch_median=None):
    record = dict(fit=dict(evaluations=evaluations, stopped_by_itself=evaluations < 1000, objective=objective,
                           mu=[0.9, 0.6, 0.3]),
                  dipole_recovery=dict(position_error_mm=dict(median=error_median, p90=2.0, maximum=3.0),
                                       moment_relative_error=dict(median=0.01)))
    if patch_median is not None:
        record['patch_comparison'] = dict(patch=dict(median=patch_median), point=dict(median=2 * patch_median))
    return record


def moved_by(median, above=0):
    return dict(n=300, median=median, p90=2 * median, maximum=10 * median, share_above_0p1_mm=above / 300)


def test_perturbation_rows_and_their_ranges():
    fits = dict(fits=300, stopped_by_themselves=299, evaluations=dict(n=300, median=43.0, p90=60.0, maximum=115.0))
    rows = [study.perturbation_row('sphere_fit', 'mne_limit', 1, summary(1000, 2e-4, 0.30, 3e-3), moved_by(0.05, 60)),
            study.perturbation_row('sphere_fit', 'mne_limit', 2, summary(1000, 1e-4, 0.25, 2e-3), moved_by(0.07, 75)),
            study.perturbation_row('sphere_fit', 'raised_limit', 1, summary(900, 4e-6, 0.21, 5e-4), moved_by(0.04)),
            study.perturbation_row('dipole_fit', 'raised_limit', 1, summary(900, 4e-6, 0.22), moved_by(0.04, 51),
                                   fits),
            study.perturbation_row('dipole_fit', 'raised_limit', 2, summary(900, 4e-6, 0.20), moved_by(0.03, 45),
                                   dict(fits, stopped_by_themselves=300))]
    header = study.PERTURBATION_HEADER
    assert [len(row) for row in rows] == [len(header)] * 5
    assert rows[0][:5] == ['sphere_fit', 'mne_limit', 1, 1000, 0]
    assert header[3:6] == ['parameter_fit_evaluations', 'parameter_fit_stopped_by_itself', 'parameter_fit_objective']
    # A trial of the parameter fit leaves the dipole fits of MNE-Python untouched: no record of them.
    dipoles = header.index('dipole_fits')
    assert rows[0][dipoles:dipoles + 4] == ['', '', '', '']
    assert rows[3][dipoles:dipoles + 4] == [300, 299, '43.0', '115.0']
    # A dipole-fit trial does not repeat the field comparison: both fields stay empty.
    position = header.index('patch_median_relative_field_difference')
    assert rows[3][position:position + 2] == ['', '']
    assert rows[0][header.index('fitted_position_change_median_mm'):] == ['0.05', '0.1', '0.5', 60]
    assert study.STATISTICS[0] == 'patch_median_relative_field_difference'
    ranges = study.perturbation_ranges(rows)
    assert set(ranges) == {'sphere_fit', 'dipole_fit'}
    assert set(ranges['sphere_fit']) == {'mne_limit', 'raised_limit'} and set(ranges['dipole_fit']) == {'raised_limit'}
    limit = ranges['sphere_fit']['mne_limit']
    assert limit['trials'] == 2 and limit['trials_in_which_the_parameter_fit_stopped_by_itself'] == 0
    assert limit['parameter_fit_objective'] == [1e-4, 2e-4]
    assert limit['patch_median_relative_field_difference'] == [2e-3, 3e-3]
    assert limit['point_median_relative_field_difference'] == [4e-3, 6e-3]
    assert limit['position_error_median_mm'] == [0.25, 0.30]
    assert limit['fitted_position_change_median_mm'] == [0.05, 0.07]
    assert limit['fitted_positions_changed_by_more_than_0p1_mm'] == [60.0, 75.0]
    assert 'dipole_fits' not in limit
    assert ranges['sphere_fit']['raised_limit']['trials_in_which_the_parameter_fit_stopped_by_itself'] == 1
    fits_range = ranges['dipole_fit']['raised_limit']
    assert fits_range['position_error_median_mm'] == [0.20, 0.22] and fits_range['trials'] == 2
    assert fits_range['dipole_fits_stopped_by_themselves'] == [299.0, 300.0]
    assert fits_range['fitted_position_change_maximum_mm'] == pytest.approx([0.3, 0.4])
    assert 'patch_median_relative_field_difference' not in fits_range


RECOVERY = dict(estimated_xyz=np.array([[0.25, -0.5, 0.125], [0.0, 0.0625, 1.0]]), err_mm=np.array([0.25, 1.5]),
                orientation_error_deg=np.array([0.5, 2.0]), moment_relative_error=np.array([0.125, 0.0625]),
                gof=np.array([99.5, 98.0]))


def test_per_dipole_rows_hold_positions_in_mm_and_the_four_errors():
    rows = study.dipole_rows('dipole_fit', 'raised_limit', 3, RECOVERY)
    assert rows == [['dipole_fit', 'raised_limit', 3, 0, '250.0', '-500.0', '125.0', '0.25', '0.5', '0.125', '99.5'],
                    ['dipole_fit', 'raised_limit', 3, 1, '0.0', '62.5', '1000.0', '1.5', '2.0', '0.0625', '98.0']]
    assert len(rows[0]) == len(study.DIPOLE_HEADER)
    assert study.DIPOLE_HEADER[4:] == study.DIPOLE_VALUES == [
        'fitted_x_mm', 'fitted_y_mm', 'fitted_z_mm', 'position_error_mm', 'orientation_error_deg',
        'moment_relative_error', 'goodness_of_fit_percent']


def test_rows_of_a_run_with_the_small_final_radius():
    fit = dict(evaluations=1749, stopped_by_itself=True)
    fits = dict(fits=2, stopped_by_themselves=2, evaluations=dict(n=2, median=80.5, p90=90.0, maximum=92.0))
    moved = dict(n=2, median=0.001, p90=0.002, maximum=0.003, share_above_0p1_mm=0.0)
    row = study.small_tolerance_row('Haswell', 'dipole_fits', 'raised_limit', 2, fit, RECOVERY, fits, moved)
    header = study.SMALL_TOLERANCE_HEADER
    assert len(row) == len(header)
    assert dict(zip(header, row)) == dict(
        kernel='Haswell', changed_objective='dipole_fits', sphere_fit='raised_limit',
        objective_scale_minus_one_in_eps=2, parameter_fit_evaluations=1749, parameter_fit_stopped_by_itself=1,
        dipole_fits=2, dipole_fits_stopped_by_themselves=2, dipole_fit_evaluations_median='80.5',
        dipole_fit_evaluations_maximum='92.0', position_error_median_mm='0.875',
        position_error_p90_mm=repr(float(np.percentile([0.25, 1.5], 90))), position_error_maximum_mm='1.5',
        fitted_position_change_median_mm='0.001', fitted_position_change_p90_mm='0.002',
        fitted_position_change_maximum_mm='0.003')
    # An unchanged run of the main process has nothing to be compared with.
    unchanged = study.small_tolerance_row('Haswell', 'none', 'raised_limit', 0, fit, RECOVERY, fits, None)
    assert unchanged[-3:] == ['', '', '']
    dipoles = study.small_tolerance_dipoles('Nehalem', 'none', 'mne_limit', 0, RECOVERY['estimated_xyz'],
                                            RECOVERY['err_mm'])
    assert dipoles == [['Nehalem', 'none', 'mne_limit', 0, 0, '250.0', '-500.0', '125.0', '0.25'],
                       ['Nehalem', 'none', 'mne_limit', 0, 1, '0.0', '62.5', '1000.0', '1.5']]
    assert len(dipoles[0]) == len(study.SMALL_TOLERANCE_DIPOLE_HEADER)


def test_ranges_over_the_runs_with_the_small_final_radius():
    def row(kernel, changed, name, median, stopped=1, fits_stopped=300, moved=None):
        fit = dict(evaluations=1000 if name == 'mne_limit' else 1749, stopped_by_itself=bool(stopped))
        fits = dict(fits=300, stopped_by_themselves=fits_stopped, evaluations=dict(n=300, median=80.0, maximum=300.0))
        recovery = dict(err_mm=np.array([median - 0.25, median, median + 0.5]))
        change = None if moved is None else dict(n=300, median=moved, p90=2 * moved, maximum=4 * moved)
        return study.small_tolerance_row(kernel, changed, name, 0, fit, recovery, fits, change)

    rows = [row('Haswell', 'none', 'mne_limit', 1.0, stopped=0),
            row('Haswell', 'none', 'raised_limit', 2.0),
            row('Haswell', 'dipole_fits', 'raised_limit', 2.25, moved=0.5, fits_stopped=299),
            row('Nehalem', 'none', 'raised_limit', 1.75, moved=0.25)]
    ranges = study.small_tolerance_ranges(rows)
    assert list(ranges) == ['mne_limit', 'raised_limit']
    limit, raised = ranges['mne_limit'], ranges['raised_limit']
    assert limit['runs'] == 1 and limit['kernels'] == ['Haswell']
    assert limit['runs_in_which_the_parameter_fit_stopped_by_itself'] == 0
    assert limit['position_error_median_mm'] == [1.0, 1.0] and limit['fitted_position_change_median_mm'] is None
    assert raised['runs'] == 3 and raised['kernels'] == ['Haswell', 'Nehalem']
    assert raised['runs_in_which_the_parameter_fit_stopped_by_itself'] == 3
    assert (raised['dipole_fits'], raised['dipole_fits_stopped_by_themselves']) == (900, 899)
    assert raised['position_error_median_mm'] == [1.75, 2.25]
    assert raised['position_error_maximum_mm'] == [2.25, 2.75]
    assert raised['fitted_position_change_median_mm'] == [0.25, 0.5]
    assert raised['fitted_position_change_maximum_mm'] == [1.0, 2.0]


def test_kernel_rows_keep_the_arms_and_the_sources_apart():
    arrays = {}
    for offset, (name, _) in enumerate(study.FITS):
        arrays[f'{name}_fitted_m'] = np.array([[0.25, 0.0, 0.0], [0.0, 0.5, 0.0]]) * (1 + offset)
        arrays[f'{name}_position_error_mm'] = np.array([0.5, 0.75]) + offset
        arrays[f'{name}_orientation_error_deg'] = np.array([1.0, 2.0]) + offset
        arrays[f'{name}_moment_relative_error'] = np.array([0.125, 0.25]) * (1 + offset)
        arrays[f'{name}_goodness_of_fit_percent'] = np.array([99.0, 98.5]) - offset
        arrays[f'{name}_patch'] = np.array([0.25, 0.5, 0.75]) * (1 + offset)
        arrays[f'{name}_point'] = np.array([0.125]) * (1 + offset)
    dipoles, fields = study.kernel_rows('Nehalem', arrays)
    assert dipoles == [['Nehalem', 'mne_limit', 0, '250.0', '0.0', '0.0', '0.5', '1.0', '0.125', '99.0'],
                       ['Nehalem', 'mne_limit', 1, '0.0', '500.0', '0.0', '0.75', '2.0', '0.25', '98.5'],
                       ['Nehalem', 'raised_limit', 0, '500.0', '0.0', '0.0', '1.5', '2.0', '0.25', '98.0'],
                       ['Nehalem', 'raised_limit', 1, '0.0', '1000.0', '0.0', '1.75', '3.0', '0.5', '97.5']]
    assert [row[:4] for row in fields] == [
        ['Nehalem', 'mne_limit', 'patch', 0], ['Nehalem', 'mne_limit', 'patch', 1],
        ['Nehalem', 'mne_limit', 'patch', 2], ['Nehalem', 'mne_limit', 'point', 0],
        ['Nehalem', 'raised_limit', 'patch', 0], ['Nehalem', 'raised_limit', 'patch', 1],
        ['Nehalem', 'raised_limit', 'patch', 2], ['Nehalem', 'raised_limit', 'point', 0]]
    assert [row[4] for row in fields] == ['0.25', '0.5', '0.75', '0.125', '0.5', '1.0', '1.5', '0.25']


def test_comparison_config_holds_what_the_field_comparison_reads():
    config = study.comparison_config((0.08, 0.081, 0.086, 0.092), (0.33, 1.79, 0.0042, 0.33), 256)
    assert config.signal.sampling_freq_hz == 256.0
    assert config.head.radii_m == [0.08, 0.081, 0.086, 0.092]
    assert config.head.conductivities_S_per_m == [0.33, 1.79, 0.0042, 0.33]


def read(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


@pytest.fixture(scope='module')
def smoke_run(tmp_path_factory):
    """One complete run with the smallest sizes: every part of the script, one kernel and two spheres."""
    pytest.importorskip('h5py')
    base = tmp_path_factory.mktemp('sphere-fit')
    geometry = base / 'geometry.json'
    geometry.write_text(json.dumps(dict(variants=dict(
        default=dict(radii_m=[0.080, 0.081, 0.086, 0.092], conductivities_S_per_m=[0.33, 1.79, 0.0042, 0.33]),
        three_shell=dict(radii_m=[0.080, 0.081, 0.086, 0.092], conductivities_S_per_m=[0.33, 0.33, 0.0042, 0.33]),
        left_out_of_a_smoke_run=dict(radii_m=[0.07, 0.08, 0.09, 0.1],
                                     conductivities_S_per_m=[0.33, 1.79, 0.0042, 0.33])))), encoding='utf-8')
    output = base / 'run'
    study.main(['--output-dir', str(output), '--smoke', '--geometry', str(geometry)])
    return output, json.loads((output / 'summary.json').read_text(encoding='utf-8'))


def test_smoke_run_writes_every_file_and_no_dataset(smoke_run):
    output, record = smoke_run
    assert sorted(path.name for path in output.iterdir()) == [
        'dipole_recovery_35db.csv', 'environment.txt', 'field_differences.csv',
        'kernels_dipole_recovery_35db.csv', 'kernels_field_differences.csv', 'output_sha256.json',
        'perturbation.csv', 'perturbation_dipoles.csv', 'perturbation_fields.csv', 'provenance.json',
        'small_tolerance.csv', 'small_tolerance_dipoles.csv', 'snr_sweep.csv', 'spheres_dipole_recovery.csv',
        'spheres_field_differences.csv', 'summary.json']
    assert record['smoke'] is True
    assert record['instrumented_call_reproduces_installed_mne_python'] is True
    assert record['instrumented_dipole_fit_reproduces_installed_mne_python'] is True
    assert set(record['default_sphere_model_of_mne_python']) == {'mne_limit', 'raised_limit', 'layers'}
    # The arguments of the optimizer call are those that MNE-Python passed, not constants of the script.
    assert record['optimizer_call']['arguments'] == record['arms']['mne_limit']['fit']['call']
    assert record['optimizer_call']['arguments']['start'] == [0.0, 0.0, 0.0]
    provenance = json.loads((output / 'provenance.json').read_text(encoding='utf-8'))
    assert set(provenance['inputs']) == {'geometry.json'} and provenance['library_kernels']
    inventory = json.loads((output / 'output_sha256.json').read_text(encoding='utf-8'))
    assert set(inventory) == {path.name for path in output.iterdir()} - {'output_sha256.json'}


def test_smoke_run_records_both_fits_and_their_comparisons(smoke_run):
    output, record = smoke_run
    sizes = record['sizes']
    assert set(record['arms']) == {'mne_limit', 'raised_limit'}
    for name, limit in study.FITS:
        arm = record['arms'][name]
        assert arm['fit']['evaluation_limit'] == limit and len(arm['fit']['first_evaluations']) == 4
        assert arm['dipole_recovery']['position_error_mm']['n'] == sizes['dipoles']
        assert arm['patch_comparison']['patch']['n'] == arm['patch_comparison']['point']['n'] == sizes['samples']
        assert [level['snr_db'] for level in arm['sweep']] == [float(level) for level in sizes['sweep_levels']]
        assert record['dipole_fits'][name]['fits'] == sizes['dipoles']
        assert record['dipole_fits'][name]['final_radius_m'] == [5e-5]
    dipoles = read(output / 'dipole_recovery_35db.csv')
    assert len(dipoles) == sizes['dipoles']
    for name, _ in study.FITS:
        errors = [float(row[f'position_error_mm_{name}']) for row in dipoles]
        assert record['arms'][name]['dipole_recovery']['position_error_mm'] == study.spread(errors)
    fields = read(output / 'field_differences.csv')
    assert [row['source'] for row in fields] == ['patch'] * sizes['samples'] + ['point'] * sizes['samples']
    assert len(read(output / 'snr_sweep.csv')) == sizes['sweep_dipoles'] * len(sizes['sweep_levels'])


def test_smoke_run_trials_agree_with_their_per_dipole_rows(smoke_run):
    output, record = smoke_run
    count = record['sizes']['dipoles']
    trials = read(output / 'perturbation.csv')
    assert [(row['perturbed_fit'], row['sphere_fit'], row['objective_scale_minus_one_in_eps']) for row in trials] == [
        ('sphere_fit', 'mne_limit', '1'), ('sphere_fit', 'raised_limit', '1'),
        ('dipole_fit', 'mne_limit', '1'), ('dipole_fit', 'raised_limit', '1')]
    assert list(trials[0]) == study.PERTURBATION_HEADER
    dipoles = read(output / 'perturbation_dipoles.csv')
    assert list(dipoles[0]) == study.DIPOLE_HEADER and len(dipoles) == 6 * count

    def fitted(perturbed, name):
        rows = [row for row in dipoles if (row['perturbed_fit'], row['sphere_fit']) == (perturbed, name)]
        assert [int(row['sample_index']) for row in rows] == list(range(count))
        return (np.array([[float(row[f'fitted_{axis}_mm']) for axis in 'xyz'] for row in rows]),
                np.array([float(row['position_error_mm']) for row in rows]))

    for name, _ in study.FITS:
        # The unchanged run of the trial file is the run of the per-window file.
        _, errors = fitted('none', name)
        assert record['arms'][name]['dipole_recovery']['position_error_mm'] == study.spread(errors)
    for trial in trials:
        positions, errors = fitted(trial['perturbed_fit'], trial['sphere_fit'])
        reference, _ = fitted('none', trial['sphere_fit'])
        assert float(trial['position_error_median_mm']) == float(np.median(errors))
        assert float(trial['position_error_maximum_mm']) == errors.max()
        moments = [float(row['moment_relative_error']) for row in dipoles
                   if (row['perturbed_fit'], row['sphere_fit']) == (trial['perturbed_fit'], trial['sphere_fit'])]
        assert float(trial['moment_relative_error_median']) == float(np.median(moments))
        moved = np.linalg.norm(positions - reference, axis=1)
        assert float(trial['fitted_position_change_median_mm']) == pytest.approx(float(np.median(moved)), abs=1e-9)
        assert int(trial['fitted_positions_changed_by_more_than_0p1_mm']) == int((moved > 0.1).sum())
        if trial['perturbed_fit'] == 'dipole_fit':
            assert int(trial['dipole_fits']) == count and trial['patch_median_relative_field_difference'] == ''
            # The sphere model of a dipole-fit trial is the unchanged one of its arm.
            assert float(trial['parameter_fit_objective']) == record['arms'][trial['sphere_fit']]['fit']['objective']
        else:
            assert trial['dipole_fits'] == '' and float(trial['patch_median_relative_field_difference']) > 0
    fields = read(output / 'perturbation_fields.csv')
    assert list(fields[0]) == study.FIELD_HEADER and len(fields) == 2 * 2 * record['sizes']['samples']
    for trial in trials[:2]:
        for source in ('patch', 'point'):
            values = [float(row['relative_field_difference']) for row in fields
                      if (row['sphere_fit'], row['source']) == (trial['sphere_fit'], source)]
            assert float(trial[f'{source}_median_relative_field_difference']) == float(np.median(values))
    ranges = record['perturbation']['ranges']
    assert set(ranges) == {'sphere_fit', 'dipole_fit'}
    assert all(set(ranges[kind]) == {'mne_limit', 'raised_limit'} for kind in ranges)
    assert ranges == study.perturbation_ranges([[row[column] for column in study.PERTURBATION_HEADER]
                                                for row in trials])


def test_smoke_run_accounts_for_every_requested_kernel(smoke_run):
    output, record = smoke_run
    kernels = record['library_kernels']
    assert kernels['variable'] == 'OPENBLAS_CORETYPE' and kernels['requested'] == list(study.KERNELS[:2])
    assert set(kernels['arms']) | set(kernels['not_available']) == set(kernels['requested'])
    own = {entry['kernel'].lower() for entry in kernels['of_the_parent_process']}
    rows = read(output / 'kernels_dipole_recovery_35db.csv') if kernels['arms'] else []
    assert {row['kernel'] for row in rows} == set(kernels['arms'])
    for kernel, entry in kernels['arms'].items():
        # A child counts only if its libraries report another kernel than those of the parent.
        assert entry['libraries'] and not {library['kernel'].lower() for library in entry['libraries']} & own
        assert set(entry['arms']) == {'mne_limit', 'raised_limit'}
        for name, _ in study.FITS:
            errors = [float(row['position_error_mm']) for row in rows
                      if (row['kernel'], row['sphere_fit']) == (kernel, name)]
            assert entry['arms'][name]['dipole_recovery']['position_error_mm'] == study.spread(errors)
            assert entry['fitted_position_change_from_the_parent_mm'][name]['n'] == len(errors)
            for quantity in ('orientation_error_deg', 'moment_relative_error', 'goodness_of_fit_percent'):
                values = [float(row[quantity]) for row in rows if (row['kernel'], row['sphere_fit']) == (kernel, name)]
                assert entry['arms'][name]['dipole_recovery'][quantity] == study.spread(values)
        exchanged = entry['parent_fit_with_the_second_value_of_this_kernel']
        assert exchanged['replaced_evaluation'] == 2
        assert exchanged['value'] == entry['arms']['mne_limit']['fit']['first_evaluations'][1]['objective']
        # The child also records the default sphere of MNE-Python and the runs with the small final radius.
        assert set(entry['default_sphere_model_of_mne_python']) == {'mne_limit', 'raised_limit', 'layers'}
        assert set(entry['small_tolerance']) == {'mne_limit', 'raised_limit'}


def test_smoke_run_with_the_small_final_radius(smoke_run):
    output, record = smoke_run
    count = record['sizes']['dipoles']
    small = record['small_tolerance']
    assert small['final_radius_m'] == 1e-7 and small['final_radius_of_mne_python_m'] == 5e-5
    assert small['steps'] == [1]
    runs = read(output / 'small_tolerance.csv')
    dipoles = read(output / 'small_tolerance_dipoles.csv')
    assert list(runs[0]) == study.SMALL_TOLERANCE_HEADER and list(dipoles[0]) == study.SMALL_TOLERANCE_DIPOLE_HEADER
    own = small['kernel_of_the_main_process']
    kernels = list(record['library_kernels']['arms'])
    # Two unchanged runs, one changed objective of each kind for each sphere fit, and two runs per other kernel.
    keys = [(row['kernel'], row['changed_objective'], row['sphere_fit']) for row in runs]
    assert keys == [(own, 'none', 'mne_limit'), (own, 'none', 'raised_limit'),
                    (own, 'parameter_fit', 'mne_limit'), (own, 'dipole_fits', 'mne_limit'),
                    (own, 'parameter_fit', 'raised_limit'), (own, 'dipole_fits', 'raised_limit'),
                    *[(kernel, 'none', name) for kernel in kernels for name, _ in study.FITS]]
    assert len(dipoles) == len(runs) * count

    def fitted(run):
        rows = [row for row in dipoles if all(row[key] == run[key] for key in (
            'kernel', 'changed_objective', 'sphere_fit', 'objective_scale_minus_one_in_eps'))]
        assert [int(row['sample_index']) for row in rows] == list(range(count))
        return (np.array([[float(row[f'fitted_{axis}_mm']) for axis in 'xyz'] for row in rows]),
                np.array([float(row['position_error_mm']) for row in rows]))

    reference = {run['sphere_fit']: fitted(run)[0] for run in runs[:2]}
    for run in runs:
        positions, errors = fitted(run)
        assert int(run['dipole_fits']) == count
        assert float(run['position_error_median_mm']) == float(np.median(errors))
        assert float(run['position_error_maximum_mm']) == errors.max()
        if (run['kernel'], run['changed_objective']) == (own, 'none'):
            assert run['fitted_position_change_median_mm'] == ''
        else:
            moved = np.linalg.norm(positions - reference[run['sphere_fit']], axis=1)
            assert float(run['fitted_position_change_maximum_mm']) == pytest.approx(moved.max(), abs=1e-9)
    assert small['ranges'] == study.small_tolerance_ranges([[row[column] for column in study.SMALL_TOLERANCE_HEADER]
                                                            for row in runs])
    assert small['ranges']['raised_limit']['runs'] == 3 + len(kernels)


def test_smoke_run_covers_the_first_two_spheres_of_the_geometry(smoke_run):
    output, record = smoke_run
    spheres = record['spheres']
    assert spheres['snr_db_range'] == [10.0, 25.0]
    assert spheres['spheres_of_the_geometry_record'] == ['default', 'three_shell', 'left_out_of_a_smoke_run']
    assert list(spheres['variants']) == ['default', 'three_shell']
    assert spheres['variants']['three_shell']['conductivities_S_per_m'] == [0.33, 0.33, 0.0042, 0.33]
    dipoles, fields = read(output / 'spheres_dipole_recovery.csv'), read(output / 'spheres_field_differences.csv')
    for sphere, entry in spheres['variants'].items():
        fitted = {name: np.array([[float(row[f'fitted_{axis}_mm_{name}']) for axis in 'xyz']
                                  for row in dipoles if row['sphere'] == sphere]) for name, _ in study.FITS}
        moved = np.linalg.norm(fitted['raised_limit'] - fitted['mne_limit'], axis=1)
        assert entry['position_error_change_mm']['fitted_position']['maximum'] == pytest.approx(moved.max(), abs=1e-9)
        for name, limit in study.FITS:
            errors = [float(row[f'position_error_mm_{name}']) for row in dipoles if row['sphere'] == sphere]
            assert entry[name]['position_error_mm'] == study.spread(errors)
            residuals = [float(row[f'relative_field_difference_{name}']) for row in fields if row['sphere'] == sphere]
            assert entry[name]['point_field_difference'] == study.spread(residuals)
            assert entry[name]['fit']['evaluation_limit'] == limit
        assert all(10.0 <= float(row['snr_db']) <= 25.0 for row in dipoles if row['sphere'] == sphere)
        change = entry['position_error_change_mm']['median_of_raised_limit_minus_median_of_mne_limit']
        assert change == pytest.approx(entry['raised_limit']['position_error_mm']['median']
                                       - entry['mne_limit']['position_error_mm']['median'], abs=1e-12)
    # The default sphere of this part is fitted like the head model of parts 1 and 2.
    assert spheres['variants']['default']['mne_limit']['fit']['mu'] == record['arms']['mne_limit']['fit']['mu']


@pytest.mark.skipif((mne.__version__, scipy.__version__) != ('1.12.1', '1.17.1'),
                    reason='the recorded fit belongs to MNE-Python 1.12.1 with SciPy 1.17.1')
def test_recorded_fit_of_the_default_head_model_stops_at_the_limit_of_scipy():
    radii, sigmas = (0.080, 0.081, 0.086, 0.092), (0.33, 1.79, 0.0042, 0.33)
    unmodified = study.sphere_model(radii, sigmas)
    at_limit, raised = [], []
    with study.instrumented_fit(study.MNE_LIMIT, log=at_limit):
        model = study.sphere_model(radii, sigmas)
    assert np.array_equal(model['mu'], unmodified['mu'])
    assert np.array_equal(model['lambda'], unmodified['lambda'])
    assert at_limit[0]['evaluations'] == study.MNE_LIMIT and not at_limit[0]['stopped_by_itself']
    # The second trial point is the one at which two of the three terms coincide.
    assert at_limit[0]['first_evaluations'][1]['mu'] == [0.5, 0.0, 0.0]
    with study.instrumented_fit(study.RAISED_LIMIT, log=raised):
        converged = study.sphere_model(radii, sigmas)
    assert raised[0]['stopped_by_itself'] and study.MNE_LIMIT < raised[0]['evaluations'] < study.RAISED_LIMIT
    assert raised[0]['objective'] < at_limit[0]['objective'] / 10
    assert not np.allclose(converged['mu'], unmodified['mu'], atol=1e-3)


def test_default_sphere_of_mne_python_is_recorded_for_both_limits():
    # How this fit ends depends on the kernel of the linear-algebra library, so only its record is tested.
    record = study.default_sphere_of_mne_python()
    assert [layer['conductivity_S_per_m'] for layer in record['layers']] == [0.33, 1.0, 0.004, 0.33]
    assert record['layers'][-1]['radius_m'] == pytest.approx(0.09)
    for name, limit in study.FITS:
        assert record[name]['evaluation_limit'] == limit and record[name]['evaluations'] <= limit
    assert record['raised_limit']['stopped_by_itself']
    if record['mne_limit']['stopped_by_itself']:
        # A fit that stops by itself within the limit is the same fit under either limit.
        assert record['mne_limit']['mu'] == record['raised_limit']['mu']
    else:
        assert record['mne_limit']['evaluations'] == study.MNE_LIMIT < record['raised_limit']['evaluations']
