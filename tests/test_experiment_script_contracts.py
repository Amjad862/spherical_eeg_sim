"""Contracts of the experiment scripts that need no template data or long runs."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

for module in ('sklearn', 'mne', 'psutil', 'threadpoolctl', 'mpmath', 'matplotlib'):
    pytest.importorskip(module)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_head_adaptation as adaptation
import run_patch_split_study as patch_study
import run_training_count_control as count_control
from run_split_study import DEPTH_BANDS, Windows, predict, train_mlp

SCRIPTS = sorted(path.name for path in (ROOT / 'scripts').glob('*.py') if not path.name.startswith('_'))


@pytest.mark.parametrize('script', SCRIPTS)
def test_every_script_imports_and_prints_its_help(script):
    # Reaches import errors and undefined module-level names in scripts that
    # the rest of the suite never executes.
    completed = subprocess.run([sys.executable, '-B', str(ROOT / 'scripts' / script), '--help'],
                               capture_output=True, text=True, cwd=ROOT)
    assert completed.returncode == 0, completed.stderr
    assert 'usage:' in completed.stdout


def _geometry():
    return dict(scale=1.09, shells=dict(inner_skull=dict(median_upper_m=0.085),
                                        outer_skull=dict(median_upper_m=0.097),
                                        scalp=dict(median_upper_m=0.101)))


def test_sphere_variants_keep_sources_and_change_only_the_shells():
    variants = adaptation.sphere_variants(_geometry())
    assert list(variants) == ['default', 'three_shell', 'size_matched', 'size_matched_three_shell',
                              'surface_fitted_three_shell']
    default = variants['default']
    assert tuple(default['radii_m']) == adaptation.DEFAULT_RADII_M
    assert variants['three_shell']['radii_m'] == default['radii_m']
    assert np.allclose(variants['size_matched']['radii_m'], 1.09 * np.array(default['radii_m']))
    assert variants['size_matched_three_shell']['radii_m'] == variants['size_matched']['radii_m']
    assert variants['surface_fitted_three_shell']['radii_m'] == pytest.approx([0.085, 0.086, 0.097, 0.101])
    for name, variant in variants.items():
        radii, sigma = variant['radii_m'], variant['conductivities_S_per_m']
        assert all(a < b for a, b in zip(radii[:-1], radii[1:]))
        # Only the cerebrospinal-fluid conductivity differs between presets.
        assert [sigma[0], sigma[2], sigma[3]] == [0.33, 0.0042, 0.33]
        assert sigma[1] == (0.33 if name.endswith('three_shell') else 1.79)


def test_variant_config_changes_the_head_and_nothing_else(tmp_path):
    variant = adaptation.sphere_variants(_geometry())['size_matched_three_shell']
    cfg = adaptation.variant_config(862, variant, tmp_path / 'dataset', smoke=False)
    reference = adaptation.config_for(862, 'position', tmp_path / 'dataset')
    assert cfg.head.radii_m == variant['radii_m']
    assert cfg.head.conductivities_S_per_m == variant['conductivities_S_per_m']
    assert not cfg.output.save_components and not cfg.output.save_leadfield
    cfg.head, cfg.output = reference.head, reference.output
    assert cfg.to_dict() == reference.to_dict()
    cfg.validate()


def _windows(rng, count, shift=0.0):
    topography = rng.normal(size=(count, 6))
    mixing = np.linspace(-1.0, 1.0, 18).reshape(6, 3)
    position = topography @ mixing * 20.0 + shift
    return Windows(topography, position, np.arange(count), np.zeros(count, dtype=np.int8), np.zeros(count))


def test_fine_tuning_improves_on_new_data_and_otherwise_keeps_the_start():
    rng = np.random.default_rng(3)
    start = train_mlp(_windows(rng, 400), _windows(rng, 100), 862, 25)
    shifted_train, shifted_val = _windows(rng, 400, shift=15.0), _windows(rng, 100, shift=15.0)

    tuned = adaptation.finetune_mlp(start, shifted_train, shifted_val, 40)
    assert tuned['best_epoch'] > 0
    before = np.linalg.norm(predict(start, shifted_val.topography) - shifted_val.position, axis=1)
    after = np.linalg.norm(predict(tuned, shifted_val.topography) - shifted_val.position, axis=1)
    assert np.median(after) < np.median(before)
    # The starting network itself is left untouched.
    assert tuned['model'] is not start['model']
    assert tuned['x_scaler'] is start['x_scaler'] and tuned['y_scaler'] is start['y_scaler']

    noise = _windows(rng, 60)
    noise = Windows(noise.topography, rng.normal(scale=200.0, size=noise.position.shape),
                    noise.pool_id, noise.population, noise.snr)
    kept = adaptation.finetune_mlp(start, noise, _windows(rng, 100), 20)
    assert kept['best_epoch'] == 0
    for a, b in zip(kept['model'].coefs_, start['model'].coefs_):
        assert np.array_equal(a, b)


@pytest.mark.parametrize('seed', [862, 863, 864])
def test_declared_patch_regions_satisfy_the_adequacy_rules(tmp_path, seed):
    result = patch_study.adequacy(patch_study.config_for(seed, 'region', tmp_path))
    assert all(result['checks'].values())
    assert min(result['centres_admitting_largest_diameter'].values()) >= patch_study.MIN_CENTRES_FOR_LARGEST_PATCH
    assert result['positions'] == patch_study.MESH_POSITIONS


def test_patch_study_policies_have_equal_support_density(tmp_path):
    exact = patch_study.config_for(862, 'position', tmp_path)
    region = patch_study.config_for(862, 'region', tmp_path)
    assert exact.sources.n_cortical_positions == 3 * region.sources.n_cortical_positions
    assert exact.source_mode == region.source_mode == 'patch'


def test_count_matched_rows_follow_the_regional_cell_counts():
    rng = np.random.default_rng(5)
    populations, radii, ids = [], [], []
    for cell, (population, low, high) in enumerate(DEPTH_BANDS):
        for position in range(40):
            windows = int(rng.integers(1, 4))
            populations += [population] * windows
            radii += [rng.uniform(low, min(high, low + 5))] * windows
            ids += [1000 * cell + position] * windows
    direction = rng.normal(size=(len(ids), 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    train = Windows(np.zeros((len(ids), 2)), direction * np.array(radii)[:, None], np.array(ids),
                    np.array(populations, dtype=np.int8), np.zeros(len(ids)))
    regional = [dict(population=population, radius_mm=[low, high], windows=0,
                     distinct_positions=wanted)
                for (population, low, high), wanted in zip(DEPTH_BANDS, (10, 40, 55, 25, 1))]
    rows, cells = count_control.matched_training_rows(train, regional, np.random.default_rng([862, 101]))
    for cell in cells:
        assert cell['matched_positions'] == min(cell['exact_positions'], cell['regional_positions'])
    kept = train.subset(rows)
    assert len(np.unique(kept.pool_id)) == sum(cell['matched_positions'] for cell in cells)
    # Every window at a chosen location is kept.
    assert np.array_equal(np.flatnonzero(np.isin(train.pool_id, np.unique(kept.pool_id))), rows)


def test_count_control_errors_are_written_as_plain_numbers(tmp_path):
    # NumPy 2 prints a scalar as "np.float64(...)"; the table driver reads the
    # file with float(), and so will anyone who opens it in another program.
    exact = np.array([4.6868122416625315, 0.1, 1e-12])
    matched = np.array([0.9101819373361163, 2.5, 3.0], dtype=np.float32)
    path = tmp_path / 'errors.csv'
    count_control.write_errors(path, np.array([7889, 9116, 12], dtype=np.int64), exact, matched)
    lines = path.read_text(encoding='utf-8').splitlines()
    assert lines[0] == 'window,position_id,exact_error_mm,count_matched_error_mm'
    rows = [line.split(',') for line in lines[1:]]
    assert [int(row[0]) for row in rows] == [0, 1, 2]
    assert [int(row[1]) for row in rows] == [7889, 9116, 12]
    assert np.array_equal([float(row[2]) for row in rows], exact)
    assert np.array_equal([float(row[3]) for row in rows], matched.astype(float))
    assert 'np.' not in path.read_text(encoding='utf-8')
