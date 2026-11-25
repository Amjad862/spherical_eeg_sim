"""The command-line tool starts a run only when its size has been chosen."""

import json

import pytest

from spherical_eeg_sim import Config, cli


@pytest.fixture
def started(monkeypatch):
    """Replace the pipeline with a recorder of the configurations it receives."""
    runs = []
    monkeypatch.setattr(cli, 'run_pipeline', runs.append)
    return runs


def test_bare_invocation_explains_itself_and_starts_nothing(started, capsys):
    assert cli.main([]) == 2
    assert started == []
    printed = capsys.readouterr().out
    assert 'No run was started' in printed
    assert '360,000 windows' in printed and '69 channels' in printed
    assert '--default-run' in printed and '--quickstart' in printed


@pytest.mark.parametrize('arguments', [
    ['--output-dir', 'somewhere'],
    ['--master-seed', '7'],
    ['--source-mode', 'patch'],
])
def test_options_that_do_not_choose_a_size_start_nothing(started, arguments):
    assert cli.main(arguments) == 2
    assert started == []


def test_default_run_generates_the_unmodified_defaults(started, capsys):
    assert cli.main(['--default-run']) == 0
    (cfg,) = started
    default = Config()
    assert (cfg.dataset.n_train, cfg.dataset.n_val, cfg.dataset.n_test) == (
        default.dataset.n_train, default.dataset.n_val, default.dataset.n_test)
    assert 'Planned dataset: 360,000 windows' in capsys.readouterr().out


def test_quickstart_and_size_overrides_start_a_run(started):
    assert cli.main(['--quickstart']) == 0
    assert cli.main(['--n-train', '50', '--n-val', '10', '--n-test', '10', '--output-dir', 'x']) == 0
    quick, sized = started
    assert quick.dataset.n_train == 800
    assert (sized.dataset.n_train, sized.dataset.n_val, sized.dataset.n_test) == (50, 10, 10)
    assert sized.output_dir == 'x'


def test_config_file_starts_a_run(started, tmp_path):
    path = tmp_path / 'config.json'
    cfg = Config()
    cfg.dataset.n_train = 30
    cfg.to_json(path)
    assert cli.main(['--config', str(path)]) == 0
    assert started[0].dataset.n_train == 30


def test_save_config_alone_writes_the_file_and_starts_nothing(started, tmp_path):
    path = tmp_path / 'defaults.json'
    assert cli.main(['--save-config', str(path)]) == 0
    assert started == []
    assert json.loads(path.read_text(encoding='utf-8'))['dataset']['n_train'] == Config().dataset.n_train


@pytest.mark.parametrize('arguments', [
    ['--quickstart', '--config', 'file.json'],
    ['--default-run', '--quickstart'],
    ['--default-run', '--config', 'file.json'],
])
def test_conflicting_base_selections_are_usage_errors(started, arguments):
    with pytest.raises(SystemExit) as raised:
        cli.main(arguments)
    assert raised.value.code == 2
    assert started == []
