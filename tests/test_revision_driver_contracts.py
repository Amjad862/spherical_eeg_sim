"""Portable source-export preflight and paired statistical contracts."""

import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('sklearn')
pytest.importorskip('mne')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from _run_provenance import check_new_output, collect_provenance, hash_file
from run_split_study import adequacy, cluster_interval, config_for
from spherical_eeg_sim import plan_position_split


def test_output_preflight_is_non_destructive(tmp_path):
    existing = tmp_path / 'already-there'
    existing.mkdir()
    with pytest.raises(ValueError, match='already exists'):
        check_new_output(existing, ROOT)
    with pytest.raises(ValueError, match='outside validation_results'):
        check_new_output(ROOT / 'validation_results/new-run', ROOT)
    proposed = tmp_path / 'new'
    assert check_new_output(proposed, ROOT) == proposed
    assert not proposed.exists()


def test_smoke_provenance_supports_a_source_export(monkeypatch):
    original = subprocess.check_output

    def without_git(command, *args, **kwargs):
        if command[0] == 'git':
            raise subprocess.CalledProcessError(128, command)
        return original(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, 'check_output', without_git)
    record = collect_provenance(ROOT, [], ['--smoke'], require_clean=False)
    assert record['commit'] is None
    assert record['source_sha256']['spherical_eeg_sim/provenance.py']
    with pytest.raises(ValueError, match='clean Git checkout'):
        collect_provenance(ROOT, [], [], require_clean=True)


def test_streamed_hash_matches_standard_digest(tmp_path):
    source = tmp_path / 'content.bin'
    source.write_bytes(b'validation' * 120000)
    assert hash_file(source) == hashlib.sha256(source.read_bytes()).hexdigest()


def test_paired_bootstrap_keeps_model_seeds_and_positions_together():
    ids = np.repeat(np.arange(35), 2)
    error = np.stack([np.arange(len(ids), dtype=float), np.arange(len(ids), dtype=float) + 2])
    assert cluster_interval(error, ids, np.random.default_rng(4), error + 5, draws=20) == [-5., -5.]
    assert cluster_interval(error[:, :4], ids[:4], np.random.default_rng(4), draws=20) is None


def test_declared_geometry_satisfies_the_adequacy_rules(tmp_path):
    cfg = config_for(862, 'region', tmp_path)
    result = adequacy(plan_position_split(cfg), cfg)
    assert all(result['checks'].values())
