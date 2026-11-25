"""The records of scripts/run_dipole_comparators_converged.py: subsets, summaries and the check against a record."""

import sys
from pathlib import Path

import numpy as np
import pytest

# The script records its provenance with threadpoolctl, which an installation with the core dependencies lacks.
pytest.importorskip('threadpoolctl')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_dipole_comparators_converged as study  # noqa: E402


def test_subsets_are_drawn_as_the_experiments_draw_them():
    # The split study draws its dipole-fit subset from stream 601 of the dataset seed, the transfer and the
    # adaptation experiment from stream 602.
    expected = np.sort(np.random.default_rng([862, 601]).choice(3000, size=500, replace=False))
    chosen = study.subset_rows(862, 601, 3000, 500)
    assert np.array_equal(chosen, expected) and len(set(chosen.tolist())) == 500
    assert not np.array_equal(study.subset_rows(862, 602, 3000, 500), chosen)
    assert not np.array_equal(study.subset_rows(863, 601, 3000, 500), chosen)
    # A subset cannot be larger than the windows there are.
    assert np.array_equal(study.subset_rows(862, 602, 4, 500), np.arange(4))


def test_metrics_of_a_group_of_errors():
    values = [1.0, 2.0, 3.0, 4.0, 100.0]
    assert study.metrics(values) == dict(windows=5, median_mm=3.0, mean_mm=22.0,
                                         p90_mm=float(np.percentile(values, 90)))


def group():
    fitted = dict(mne_limit=np.array([[10.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 30.0]]),
                  raised_limit=np.array([[10.0, 0.0, 3.0], [0.0, 20.0, 0.0], [0.0, 4.0, 30.0]]))
    truth = np.array([[11.0, 0.0, 0.0], [0.0, 22.0, 0.0], [0.0, 0.0, 35.0]])
    errors = {name: np.linalg.norm(positions - truth, axis=1) for name, positions in fitted.items()}
    fits = dict(mne_limit=dict(evaluations=1000, stopped_by_itself=False, objective=2e-4),
                raised_limit=dict(evaluations=1749, stopped_by_itself=True, objective=4e-6))
    return fits, errors, fitted


def test_summary_of_a_group_compares_the_two_parameter_fits():
    fits, errors, fitted = group()
    entry = study.group_summary(fits, errors, fitted)
    assert entry['parameter_fit'] == fits
    # Errors of 1, 2 and 5 mm with the limit of SciPy; the converged fit moves the first and the third dipole.
    assert entry['mne_limit'] == dict(windows=3, median_mm=2.0, mean_mm=pytest.approx(8 / 3), p90_mm=pytest.approx(4.4))
    assert entry['raised_limit']['median_mm'] == pytest.approx(np.sqrt(10.0))
    assert entry['median_of_raised_limit_minus_median_of_mne_limit_mm'] == pytest.approx(np.sqrt(10.0) - 2.0)
    assert entry['fitted_position_change_mm'] == dict(median=3.0, p90=pytest.approx(3.8), maximum=4.0)


def test_rows_of_a_group_hold_both_fits_of_every_window():
    _, errors, fitted = group()
    rows = study.group_rows('adaptation', 862, 'default_sphere_on_bem', [7, 11, 40], errors, fitted)
    assert [len(row) for row in rows] == [len(study.HEADER)] * 3
    assert study.HEADER[:4] == ['experiment', 'dataset_seed', 'case', 'window_id']
    assert study.HEADER[4:8] == ['position_error_mm_mne_limit', 'fitted_x_mm_mne_limit', 'fitted_y_mm_mne_limit',
                                 'fitted_z_mm_mne_limit']
    assert rows[0] == ['adaptation', 862, 'default_sphere_on_bem', 7, '1.0', '10.0', '0.0', '0.0',
                       repr(float(np.sqrt(10.0))), '10.0', '0.0', '3.0']
    assert rows[2][3] == 40 and rows[2][4] == '5.0' and rows[2][9:] == ['0.0', '4.0', '30.0']


def test_fits_at_the_limit_must_be_the_recorded_ones(tmp_path):
    _, _, fitted = group()
    record = tmp_path / 'dipole_subset.csv'
    record.write_text('window_id,dipole_default_x_mm,dipole_default_y_mm,dipole_default_z_mm,other\n'
                      '7,10.0,0.0,0.0,a\n11,0.0,20.0,0.0,b\n40,0.0,0.0,30.0,c\n', encoding='utf-8')
    positions, windows = study.recorded_positions(record, 'dipole_default', 'window_id')
    assert windows == [7, 11, 40] and np.array_equal(positions, fitted['mne_limit'])
    assert study.recorded_positions(record, 'dipole_default')[1] is None
    assert study.check_recorded('adaptation', fitted['mne_limit'], [7, 11, 40], positions, windows) == 0.0
    # Without window numbers in the record only the positions are compared.
    assert study.check_recorded('split study', fitted['mne_limit'], [1, 2, 3], positions, None) == 0.0
    with pytest.raises(ValueError, match='adaptation: the recorded experiment fitted other windows'):
        study.check_recorded('adaptation', fitted['mne_limit'], [7, 11, 41], positions, windows)
    with pytest.raises(ValueError, match='fitted other windows'):
        study.check_recorded('adaptation', fitted['mne_limit'][:2], [7, 11], positions, windows)
    moved = fitted['mne_limit'] + [[0.0, 0.0, 0.0], [0.0, 2e-9, 0.0], [0.0, 0.0, 0.0]]
    with pytest.raises(ValueError, match='differs from the recorded fit by up to 2e-09 mm'):
        study.check_recorded('adaptation', moved, [7, 11, 40], positions, windows)
    # A difference within the rounding of the record passes and is reported.
    close = fitted['mne_limit'] + [[0.0, 0.0, 0.0], [0.0, 5e-10, 0.0], [0.0, 0.0, 0.0]]
    assert study.check_recorded('adaptation', close, [7, 11, 40], positions, windows) == pytest.approx(5e-10)
    # A position that is not finite is a difference, in the new fit and in the record alike.
    broken = fitted['mne_limit'] + [[0.0, 0.0, 0.0], [0.0, np.nan, 0.0], [0.0, 0.0, 0.0]]
    with pytest.raises(ValueError, match='differs from the recorded fit'):
        study.check_recorded('adaptation', broken, [7, 11, 40], positions, windows)
    with pytest.raises(ValueError, match='differs from the recorded fit'):
        study.check_recorded('adaptation', fitted['mne_limit'], [7, 11, 40], broken, windows)


def test_subsets_come_from_the_streams_that_the_experiments_use():
    # The three experiments draw their dipole-fit subsets inline; the repetition must use the same streams.
    scripts = ROOT / 'scripts'
    for name, stream in (('run_split_study.py', 601), ('run_head_transfer.py', 602), ('run_head_adaptation.py', 602)):
        source = (scripts / name).read_text(encoding='utf-8')
        assert f'np.sort(np.random.default_rng([seed, {stream}]).choice(' in source, name
    repetition = (scripts / 'run_dipole_comparators_converged.py').read_text(encoding='utf-8')
    assert repetition.count('subset_rows(seed, 601, ') == 1 and repetition.count('subset_rows(seed, 602, ') == 1


def test_both_arms_fit_the_same_windows_and_leave_mne_python_as_it_was():
    mne = pytest.importorskip('mne')
    bem = pytest.importorskip('mne.bem')
    scipy = pytest.importorskip('scipy')
    # The fits replace the optimizer call inside MNE-Python 1.12 and belong to the COBYLA of SciPy 1.16 and later.
    if not hasattr(bem, 'fmin_cobyla'):
        pytest.skip('the fits replace the optimizer call inside MNE-Python 1.12')
    if tuple(int(part) for part in scipy.__version__.split('.')[:2]) < (1, 16):
        pytest.skip('the records of the script belong to the COBYLA of SciPy 1.16 and later')
    from run_split_study import dipole_fit
    from validate_patch_forward_mne import build_sphere_and_info, mne_superposed_field

    # The default head model of the package, 32 electrodes on the upper part of its scalp, and three dipoles.
    radii, sigmas = (0.08, 0.081, 0.086, 0.092), (0.33, 1.79, 0.0042, 0.33)
    rng = np.random.default_rng(20261115)
    directions = rng.normal(size=(32, 3))
    directions[:, 2] = np.abs(directions[:, 2]) + 0.2
    electrodes = radii[-1] * directions / np.linalg.norm(directions, axis=1, keepdims=True)
    names = [f'E{index + 1}' for index in range(len(electrodes))]
    original = bem.fmin_cobyla
    info, sphere = build_sphere_and_info(electrodes, names, radii, sigmas, 250.0, mne)
    positions = np.array([[0.02, 0.01, 0.05], [-0.03, 0.02, 0.04], [0.0, -0.04, 0.03]])
    normals = positions / np.linalg.norm(positions, axis=1, keepdims=True)
    windows = np.stack([mne_superposed_field(position[None], normal[None], [1e-8], info, sphere, mne)
                        for position, normal in zip(positions, normals)])
    expected = dipole_fit(windows, info, sphere)            # MNE-Python as installed

    fits, fitted = study.both_fits(windows, electrodes, names, radii, sigmas, 250.0)
    assert set(fits) == set(fitted) == {'mne_limit', 'raised_limit'}
    # The arm with the limit of SciPy is the fit of the unmodified MNE-Python, bit for bit.
    assert np.array_equal(fitted['mne_limit'], expected) and fitted['raised_limit'].shape == expected.shape
    assert fits['mne_limit']['evaluations'] <= study.MNE_LIMIT
    assert fits['raised_limit']['stopped_by_itself'] and fits['raised_limit']['evaluations'] <= study.RAISED_LIMIT
    if not fits['mne_limit']['stopped_by_itself']:
        # The unfinished fit ended at the limit, the other one later and at a smaller objective, and the dipole fits
        # of the two sphere models differ.
        assert fits['mne_limit']['evaluations'] == study.MNE_LIMIT < fits['raised_limit']['evaluations']
        assert fits['raised_limit']['objective'] < fits['mne_limit']['objective']
        assert not np.array_equal(fitted['raised_limit'], fitted['mne_limit'])
    # The fitted positions are those of the three dipoles, and nothing of the replacement is left behind.
    assert np.abs(fitted['mne_limit'] - 1000 * positions).max() < 0.5
    assert np.abs(fitted['raised_limit'] - 1000 * positions).max() < 3.0
    assert bem.fmin_cobyla is original
    again = dipole_fit(windows, *build_sphere_and_info(electrodes, names, radii, sigmas, 250.0, mne))
    assert np.array_equal(again, expected)


@pytest.mark.parametrize('arguments, message', [
    (['--ecd-count', '0'], '--ecd-count must be positive'),
    (['--smoke', '--recorded', 'evidence'], '--recorded belongs to a full run'),
])
def test_arguments_are_checked_before_anything_is_read(tmp_path, capsys, arguments, message):
    with pytest.raises(SystemExit) as stop:
        study.main(['--split-run', str(tmp_path / 'split'), '--subjects-dir', str(tmp_path / 'templates'),
                    '--output-dir', str(tmp_path / 'run'), *arguments])
    assert stop.value.code == 2 and message in capsys.readouterr().err
    assert not (tmp_path / 'run').exists()
