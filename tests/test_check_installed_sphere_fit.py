"""scripts/check_installed_sphere_fit.py: the counted optimizer call and the record of a small run.

The script changes nothing in the fit of MNE-Python, so these tests are written to hold whatever the version of
MNE-Python and of SciPy: they compare the counted call with the same call of SciPy and never expect a particular
number of evaluations. They were run with MNE-Python 1.12.1 and 1.13.2, both with SciPy 1.17.1.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

mne = pytest.importorskip('mne')
bem = pytest.importorskip('mne.bem')
optimize = pytest.importorskip('scipy.optimize')
# The script records its provenance with threadpoolctl, which an installation with the core dependencies lacks.
pytest.importorskip('threadpoolctl')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import check_installed_sphere_fit as check  # noqa: E402

START = np.array([-1.2, 1.0])
HEAD = dict(radii=(0.08, 0.081, 0.086, 0.092), sigmas=(0.33, 1.79, 0.0042, 0.33))


def valley(x):
    """A curved valley that COBYLA follows for many evaluations."""
    return float((1.0 - x[0]) ** 2 + 30.0 * (x[1] - x[0] ** 2) ** 2)


def inside(x):
    return 2.0 - np.abs(x)


def homes():
    """The modules in which MNE-Python may look the optimizer up."""
    return [optimize] + ([bem] if hasattr(bem, 'fmin_cobyla') else [])


def test_block_replaces_the_optimizer_wherever_mne_python_looks_it_up_and_restores_it():
    originals = [home.fmin_cobyla for home in homes()]
    with check.counted_fits(50, []):
        assert all(home.fmin_cobyla is not original for home, original in zip(homes(), originals))
    assert [home.fmin_cobyla for home in homes()] == originals
    with pytest.raises(RuntimeError, match='stop'):
        with check.counted_fits(50, []):
            raise RuntimeError('stop')
    assert [home.fmin_cobyla for home in homes()] == originals


@pytest.mark.parametrize('limit', [7, 40, 100_000])
def test_counted_call_is_the_call_of_scipy_with_the_limit(limit):
    expected = optimize.fmin_cobyla(valley, START, [inside], rhobeg=0.5, rhoend=1e-5, catol=1e-6, maxfun=limit)
    evaluations = 0

    def counted(x):
        nonlocal evaluations
        evaluations += 1
        return valley(x)

    optimize.fmin_cobyla(counted, START, [inside], rhobeg=0.5, rhoend=1e-5, catol=1e-6, maxfun=limit)
    log = []
    with check.counted_fits(limit, log):
        # The call as MNE-Python makes it: without an evaluation limit.
        found = homes()[-1].fmin_cobyla(valley, START, [inside], rhobeg=0.5, rhoend=1e-5, catol=1e-6)
    assert np.array_equal(found, expected)
    assert len(log) == 1
    record = log[0]
    assert record['evaluations'] == evaluations <= limit
    assert record['stopped_before_the_limit'] == (evaluations < limit)
    assert record['objective'] == valley(expected)
    assert record['call'] == dict(start=[-1.2, 1.0], constraints=1, catol=1e-6, rhobeg=0.5, rhoend=1e-5)


def test_large_limit_lets_the_optimizer_stop_by_itself_and_a_small_one_does_not():
    log = []
    with check.counted_fits(100_000, log):
        optimize.fmin_cobyla(valley, START, [inside], rhobeg=0.5, rhoend=1e-5, catol=1e-6)
    with check.counted_fits(5, log):
        optimize.fmin_cobyla(valley, START, [inside], rhobeg=0.5, rhoend=1e-5, catol=1e-6)
    assert log[0]['stopped_before_the_limit'] and 5 < log[0]['evaluations'] < 100_000
    assert not log[1]['stopped_before_the_limit'] and log[1]['evaluations'] == 5
    assert log[0]['objective'] < log[1]['objective']


def test_call_that_brings_its_own_limit_is_refused():
    with check.counted_fits(50, []):
        with pytest.raises(TypeError, match='evaluation limit'):
            optimize.fmin_cobyla(valley, START, [inside], rhobeg=0.5, rhoend=1e-5, catol=1e-6, maxfun=200)
        with pytest.raises(TypeError, match='positional options'):
            optimize.fmin_cobyla(valley, START, [inside], ())


def test_record_of_the_default_head_model_holds_both_limits_and_the_installed_fit():
    installed = check.package_sphere(HEAD['radii'], HEAD['sigmas'])
    record = check.fit_record(lambda: check.package_sphere(HEAD['radii'], HEAD['sigmas']))
    assert [layer['conductivity_S_per_m'] for layer in record['layers']] == list(HEAD['sigmas'])
    assert [layer['radius_m'] for layer in record['layers']] == pytest.approx(HEAD['radii'])
    first, second = record['mne_limit'], record['raised_limit']
    assert (first['evaluation_limit'], second['evaluation_limit']) == (check.MNE_LIMIT, check.RAISED_LIMIT)
    # The fit at the default limit is the one that MNE-Python makes by itself.
    assert first['mu'] == [float(value) for value in installed['mu']]
    assert first['lambda_'] == [float(value) for value in installed['lambda']]
    assert first['evaluations'] <= check.MNE_LIMIT
    assert first['stopped_before_the_limit'] == (first['evaluations'] < check.MNE_LIMIT)
    assert second['stopped_before_the_limit'] and second['evaluations'] < check.RAISED_LIMIT
    assert first['call'] == second['call'] and first['call']['start'] == [0.0, 0.0, 0.0]
    if first['stopped_before_the_limit']:
        # A fit that stops by itself within the limit is the same fit under either limit.
        assert (first['evaluations'], first['mu']) == (second['evaluations'], second['mu'])
    else:
        assert second['evaluations'] > check.MNE_LIMIT and second['objective'] <= first['objective']


def test_a_fit_with_the_raised_limit_is_not_kept_for_later_sphere_models():
    # MNE-Python 1.13 keeps the result of a fit. A sphere model built after the block must be the installed one.
    before = check.package_sphere(HEAD['radii'], HEAD['sigmas'])
    with check.counted_fits(check.RAISED_LIMIT, []):
        check.package_sphere(HEAD['radii'], HEAD['sigmas'])
    after = check.package_sphere(HEAD['radii'], HEAD['sigmas'])
    assert np.array_equal(after['mu'], before['mu']) and np.array_equal(after['lambda'], before['lambda'])
    log = []
    with check.counted_fits(check.MNE_LIMIT, log):
        check.package_sphere(HEAD['radii'], HEAD['sigmas'])
        check.package_sphere(HEAD['radii'], HEAD['sigmas'])
    # Inside a block the first sphere model is always fitted, whatever was kept before.
    assert 1 <= len(log) <= 2


def test_record_refuses_a_counted_fit_that_is_not_the_installed_one(monkeypatch):
    models = iter([dict(mu=np.array([0.1, 0.2, 0.3]), **{'lambda': np.array([1.0, 2.0, 3.0])}, layers=[]),
                   dict(mu=np.array([0.1, 0.2, 0.4]), **{'lambda': np.array([1.0, 2.0, 3.0])}, layers=[])])

    @check.contextmanager
    def one_call(limit, log):
        log.append(dict(evaluations=limit, stopped_before_the_limit=False, objective=1.0, call={}))
        yield

    monkeypatch.setattr(check, 'counted_fits', one_call)
    with pytest.raises(RuntimeError, match='is not the fit of the installed MNE-Python'):
        check.fit_record(lambda: next(models))


def test_smoke_run_writes_the_record_of_the_default_head_model(tmp_path):
    output = tmp_path / 'run'
    check.main(['--output-dir', str(output), '--smoke'])
    assert sorted(path.name for path in output.iterdir()) == ['environment.txt', 'output_sha256.json',
                                                              'provenance.json', 'summary.json']
    record = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
    assert record['smoke'] and record['counted_fit_at_the_default_limit_is_the_installed_fit']
    assert record['versions']['mne'] == mne.__version__
    assert record['evaluation_limits'] == dict(mne_limit=1000, raised_limit=100000)
    assert record['optimizer']['imported_into_the_module_of_mne_python'] == hasattr(bem, 'fmin_cobyla')
    assert list(record['spheres']) == ['default']
    assert [layer['radius_m'] for layer in record['spheres']['default']['layers']] == pytest.approx(HEAD['radii'])
    assert [layer['conductivity_S_per_m'] for layer in record['spheres']['default']['layers']] == list(HEAD['sigmas'])
    for fits in (record['spheres']['default'], record['default_sphere_model_of_mne_python']):
        assert fits['raised_limit']['stopped_before_the_limit'] and fits['mne_limit']['evaluations'] <= 1000
    # At most one other kernel in a smoke run; each one is either recorded or named as not available.
    assert len(record['other_kernels']) <= 1
    for kernel, entry in record['other_kernels'].items():
        assert {library['kernel'].lower() for library in entry['libraries']} == {kernel.lower()}
        assert list(entry['spheres']) == ['default']
    assert set(record['other_kernels']).isdisjoint(record['not_available'])
    assert record['kernel_of_the_main_process'] == ('+'.join(sorted({library['kernel'] for library
                                                                    in record['libraries']})) or 'unknown')
    inventory = json.loads((output / 'output_sha256.json').read_text(encoding='utf-8'))
    assert sorted(inventory) == ['environment.txt', 'provenance.json', 'summary.json']
    for name, digest in inventory.items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    provenance = json.loads((output / 'provenance.json').read_text(encoding='utf-8'))
    assert provenance['smoke'] is True and provenance['inputs'] == {}
    assert 'scripts/check_installed_sphere_fit.py' in provenance['source_sha256']


def test_run_needs_a_new_output_folder_and_an_existing_geometry_file(tmp_path, capsys):
    (tmp_path / 'there').mkdir()
    for arguments, message in ((['--smoke'], '--output-dir is required'),
                               (['--smoke', '--output-dir', str(tmp_path / 'there')], 'Output already exists'),
                               (['--smoke', '--output-dir', str(tmp_path / 'new'), '--geometry',
                                 str(tmp_path / 'none.json')], 'no such file')):
        with pytest.raises(SystemExit):
            check.main(arguments)
        assert message in capsys.readouterr().err
    assert not (tmp_path / 'new').exists()
