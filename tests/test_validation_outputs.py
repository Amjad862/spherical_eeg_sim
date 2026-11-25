"""Validation runs preserve the shipped archive unless explicitly regenerated."""

import csv
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import _validation_outputs, validate_inverse_mne


REPO_ROOT = Path(__file__).resolve().parent.parent


def _snapshot(directory):
    return {
        path.relative_to(directory): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in directory.rglob("*") if path.is_file()
    }


@pytest.fixture
def checkout(tmp_path):
    return shutil.copytree(
        REPO_ROOT,
        tmp_path / "checkout",
        ignore=shutil.ignore_patterns(
            ".git", ".pytest_cache", "__pycache__", "validation_runs",
            ".venv", "venv", "output_*", "build", "dist",
        ),
    )


def _run(checkout, *args):
    return subprocess.run(
        [sys.executable, *args], cwd=checkout, capture_output=True, text=True,
        timeout=600,
        env={
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "_MNE_FAKE_HOME_DIR": str(checkout.parent),
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
        },
    )


@pytest.mark.skipif(importlib.util.find_spec("mne") is None, reason="MNE is optional")
def test_quick_inverse_command_preserves_checkout(checkout):
    before = _snapshot(checkout)
    result = _run(checkout, "scripts/validate_inverse_mne.py", "--quick", "--no-sweep")
    assert result.returncode == 0, result.stdout + result.stderr
    after = _snapshot(checkout)
    assert all(after.get(path) == digest for path, digest in before.items())

    new_paths = after.keys() - before.keys()
    assert len(new_paths) == 2
    assert all(len(path.parts) == 3 and path.parts[0] == "validation_runs" for path in new_paths)
    parents = {path.parent for path in new_paths}
    assert len(parents) == 1
    run = parents.pop()
    assert new_paths == {
        run / "inverse_recovery_metrics.csv",
        run / "inverse_recovery_summary.json",
    }
    summary = json.loads((checkout / run / "inverse_recovery_summary.json").read_text())
    assert summary["n_test"] == 60
    assert summary["snr_degradation_sweep"] == []
    assert summary["metrics_csv_sha256"] == after[run / summary["metrics_csv"]]
    with (checkout / run / summary["metrics_csv"]).open(newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 60


@pytest.fixture
def destinations(tmp_path, monkeypatch):
    archive = tmp_path / "validation_results"
    runs = tmp_path / "validation_runs"
    monkeypatch.setattr(_validation_outputs, "ARCHIVE", archive)
    monkeypatch.setattr(_validation_outputs, "RUNS", runs)
    return archive, runs


@pytest.mark.parametrize("mode", ("quick_canonical", "archive_alias", "existing"))
def test_inverse_refuses_unsafe_paths_before_generation(
    mode, destinations, tmp_path, monkeypatch, capsys,
):
    archive, _ = destinations
    if mode == "quick_canonical":
        options = ["--write-canonical", "--quick", "--no-sweep"]
        message = "publication settings"
    elif mode == "archive_alias":
        # A path that becomes canonical after dot-segment normalization.
        destination = tmp_path / "experiment" / ".." / archive.name / "ecd.csv"
        options = ["--csv", str(destination), "--overwrite"]
        message = "validation_results/ is protected"
    else:
        destination = tmp_path / "existing.csv"
        destination.write_text("keep this")
        options = ["--csv", str(destination)]
        message = "Output already exists"

    monkeypatch.setattr(sys, "argv", ["validate_inverse_mne.py", *options])
    monkeypatch.setattr(
        validate_inverse_mne, "_run",
        lambda *args: pytest.fail("generation started before output preflight"),
    )
    with pytest.raises(SystemExit) as raised:
        validate_inverse_mne.main()
    assert raised.value.code == 2
    assert message in capsys.readouterr().err
    assert not archive.exists()
    if mode == "existing":
        assert destination.read_text() == "keep this"


def test_inverse_canonical_bundle_publishes_after_success(destinations, monkeypatch):
    archive, runs = destinations
    archive.mkdir()
    canonical_files = {
        "ecd_69ch_35db_per_sample.csv",
        "ecd_69ch_35db_summary.json",
        "ecd_69ch_35db.png",
    }
    for filename in canonical_files:
        (archive / filename).write_text("previous")
    original = _snapshot(archive)
    monkeypatch.setattr(sys, "argv", ["validate_inverse_mne.py", "--write-canonical"])

    def generate(args, paths, final_paths):
        assert args.n_test == 300
        assert set(final_paths.values()) == {archive / name for name in canonical_files}
        assert all(paths[key] != final_paths[key] for key in paths)
        for path in paths.values():
            path.write_text("next")

    def fail_after_generation(*args):
        generate(*args)
        raise RuntimeError("late validation failure")

    monkeypatch.setattr(validate_inverse_mne, "_run", fail_after_generation)
    with pytest.raises(RuntimeError, match="late validation failure"):
        validate_inverse_mne.main()
    assert _snapshot(archive) == original

    monkeypatch.setattr(validate_inverse_mne, "_run", generate)
    validate_inverse_mne.main()
    assert {path.name for path in archive.iterdir()} == canonical_files
    assert all((archive / name).read_text() == "next" for name in canonical_files)
    assert not runs.exists()


def test_explicit_bundle_keeps_a_destination_created_during_generation(
    destinations, tmp_path, monkeypatch,
):
    output = tmp_path / "metrics.csv"
    summary = tmp_path / "inverse_recovery_summary.json"
    monkeypatch.setattr(
        sys, "argv", ["validate_inverse_mne.py", "--csv", str(output)],
    )

    def generate(args, paths, final_paths):
        for path in paths.values():
            path.write_text("complete")
        summary.write_text("another run")

    monkeypatch.setattr(validate_inverse_mne, "_run", generate)
    with pytest.raises(FileExistsError):
        validate_inverse_mne.main()
    assert not output.exists()
    assert summary.read_text() == "another run"


def test_manifest_checks_by_default_and_refuses_inconsistent_archive(checkout):
    script = "scripts/make_validation_manifest.py"
    archive = checkout / "validation_results"
    manifest = archive / "SHA256SUMS"
    result = _run(checkout, script, "--write-canonical")
    assert result.returncode == 0, result.stdout + result.stderr
    original = manifest.read_bytes()
    verify = _run(checkout, script)
    assert verify.returncode == 0, verify.stdout + verify.stderr
    assert manifest.read_bytes() == original

    # A correctly hashed CSV with the wrong number of rows is still invalid.
    summary_path = archive / "ecd_69ch_35db_summary.json"
    summary = json.loads(summary_path.read_text())
    metrics = archive / summary["metrics_csv"]
    metrics.write_bytes(b"\n".join(metrics.read_bytes().splitlines()[:61]) + b"\n")
    summary["metrics_csv_sha256"] = hashlib.sha256(metrics.read_bytes()).hexdigest()
    summary_path.write_text(json.dumps(summary))
    assert _run(checkout, script).returncode == 1
    result = _run(checkout, script, "--write-canonical")
    assert result.returncode == 1
    assert "does not contain 300 indexed samples" in result.stderr
    assert manifest.read_bytes() == original


def test_manifest_refuses_results_that_come_from_different_evidence(checkout):
    """The tables, the benchmark record and the figure records must share their sources."""
    script = "scripts/make_validation_manifest.py"
    archive = checkout / "validation_results"
    assert _run(checkout, script, "--write-canonical").returncode == 0

    # A benchmark record other than the one the tables were built from.
    benchmark = archive / "generation_benchmark.json"
    original = benchmark.read_bytes()
    benchmark.write_bytes(original + b" ")
    result = _run(checkout, script, "--write-canonical")
    assert result.returncode == 1
    assert "generation_benchmark.json disagrees" in result.stderr
    benchmark.write_bytes(original)
    assert _run(checkout, script, "--write-canonical").returncode == 0

    # A figure record that names another summary than the one the tables used.
    record_path = archive / "head_adaptation_finetuning_source.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["summary_sha256"] = "0" * 64
    record_path.write_text(json.dumps(record), encoding="utf-8")
    result = _run(checkout, script, "--write-canonical")
    assert result.returncode == 1
    assert "different summary" in result.stderr

    # A figure file that its record does not describe.
    record_path.write_bytes((REPO_ROOT / "validation_results" / record_path.name).read_bytes())
    figure = archive / "region_split.png"
    figure.write_bytes(figure.read_bytes() + b" ")
    result = _run(checkout, script, "--write-canonical")
    assert result.returncode == 1
    assert "region_split.png disagrees" in result.stderr
