"""Behavioural contracts that the rest of the suite does not pin directly.

Each test fixes one guarantee the package makes to its users: what a split file
declares about its own physics, how referencing and band edges behave at their
limits, what provenance a run records, and whether the archived validation
artifacts stay bound to the scripts and documentation that produced them. A
failure here means a documented promise no longer holds.
"""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from spherical_eeg_sim import Config, run_pipeline
from spherical_eeg_sim.core.electrodes import (
    apply_reference,
    load_electrodes,
    resolve_electrode_path,
)
from spherical_eeg_sim.sources.modes import generate_multi_point_samples


def _small_config(output_dir: Path, **overrides) -> Config:
    cfg = Config()
    cfg.output_dir = str(output_dir)
    cfg.dataset.n_train = 10
    cfg.dataset.n_val = 5
    cfg.dataset.n_test = 5
    cfg.sources.n_cortical_positions = 120
    cfg.sources.n_subcortical_positions = 40
    cfg.head.n_legendre = 40
    cfg.output.chunk_samples = 5
    for dotted, value in overrides.items():
        target = cfg
        parts = dotted.split(".")
        for part in parts[:-1]:
            target = getattr(target, part)
        setattr(target, parts[-1], value)
    return cfg


# ---------------------------------------------------------------------------
# A split file must be self-describing about the forward operator
# ---------------------------------------------------------------------------

def test_head_model_parameters_are_embedded_in_split_metadata(tmp_path):
    """A split separated from metadata.json still declares its forward model."""
    h5py = pytest.importorskip("h5py")
    cfg = _small_config(tmp_path / "ds")
    run_pipeline(cfg, log_level=logging.ERROR)

    with h5py.File(tmp_path / "ds" / "train.h5", "r") as f:
        metadata = f["metadata"]
        assert int(metadata.attrs["n_legendre"]) == 40
        np.testing.assert_allclose(
            np.asarray(metadata["radii_m"][:]), cfg.head.radii_m
        )
        np.testing.assert_allclose(
            np.asarray(metadata["conductivities_S_per_m"][:]),
            cfg.head.conductivities_S_per_m,
        )


# ---------------------------------------------------------------------------
# apply_reference must never alias its input
# ---------------------------------------------------------------------------

def test_apply_reference_none_returns_an_independent_array():
    """Mode 'none' returned the caller's array, aliasing signal into result.

    Every other mode allocates. Returning the input here meant a later in-place
    operation on the referenced array would silently rewrite the unreferenced
    one.
    """
    eeg = np.arange(12, dtype=float).reshape(3, 4)
    out = apply_reference(eeg, {"mode": "none"})
    assert out is not eeg
    np.testing.assert_array_equal(out, eeg)
    out[0, 0] = 999.0
    assert eeg[0, 0] == 0.0


# ---------------------------------------------------------------------------
# A band edge exactly at Nyquist is degenerate
# ---------------------------------------------------------------------------

def test_band_edge_at_nyquist_is_rejected():
    """A sinusoid sampled exactly at Nyquist has phase-dependent amplitude."""
    cfg = Config()
    cfg.signal.sampling_freq_hz = 500.0
    cfg.signal.bands_hz = {"wide": [30.0, 250.0]}
    cfg.signal.band_probabilities = {"wide": 1.0}
    cfg.signal.band_order = ["wide"]
    with pytest.raises(ValueError, match="Nyquist"):
        cfg.validate()

    cfg.signal.bands_hz = {"wide": [30.0, 249.0]}
    cfg.validate()


# ---------------------------------------------------------------------------
# graph_k must be checked against the montage, not only in the exporter
# ---------------------------------------------------------------------------

def test_graph_k_is_validated_against_montage_channel_count():
    """Configuration validation accounts for the montage channel count."""
    cfg = Config()
    cfg.electrode_file = "standard_1020_19.csv"
    cfg.output.save_graph = True
    cfg.output.graph_method = "knn"
    cfg.output.graph_k = 50
    with pytest.raises(ValueError, match="graph_k"):
        cfg.validate()

    cfg.output.graph_k = 6
    cfg.validate()


def test_graph_k_check_is_skipped_when_no_graph_is_requested():
    cfg = Config()
    cfg.electrode_file = "standard_1020_19.csv"
    cfg.output.save_graph = False
    cfg.output.graph_k = 50
    cfg.validate()


# ---------------------------------------------------------------------------
# The generator's own version is the key provenance field
# ---------------------------------------------------------------------------

def test_runtime_provenance_always_records_the_generator_version(tmp_path):
    """Running from a source checkout must not drop the package version."""
    import json

    cfg = _small_config(tmp_path / "ds")
    run_pipeline(cfg, log_level=logging.ERROR)
    runtime = json.loads((tmp_path / "ds.runtime.json").read_text())
    versions = runtime["software_versions"]
    assert "spherical_eeg_sim" in versions
    assert versions["spherical_eeg_sim"]
    environment = json.loads((tmp_path / "ds" / "environment.json").read_text())
    assert environment["software_versions"]["spherical_eeg_sim"]
    assert environment["python_version"]
    assert environment["platform"]


# ---------------------------------------------------------------------------
# Two dipoles at one pool position are one dipole
# ---------------------------------------------------------------------------

def test_multi_point_never_reuses_a_pool_position_within_a_sample(tmp_path):
    """Two dipoles at one pool position are one dipole, at any separation.

    A separation test alone cannot enforce this: with min_separation_mm=0 the
    condition 0.0 >= 0.0 accepts a repeated position, so the pool index is
    excluded explicitly instead.
    """
    h5py = pytest.importorskip("h5py")
    cfg = _small_config(
        tmp_path / "ds",
        source_mode="multi_point",
        **{
            "multi_point.min_separation_mm": 0.0,
            "multi_point.n_sources_min": 3,
            "multi_point.n_sources_max": 5,
        },
    )
    cfg.dataset.n_train = 40
    run_pipeline(cfg, log_level=logging.ERROR)

    with h5py.File(tmp_path / "ds" / "train.h5", "r") as f:
        positions = f["ground_truth/positions_xyz"][:]
        counts = f["ground_truth/n_dipoles_per_sample"][:]

    for sample, count in zip(positions, counts):
        active = sample[: int(count)]
        distinct = {tuple(np.round(p, 12)) for p in active}
        assert len(distinct) == int(count), "a pool position was reused"


def test_multi_point_flagged_fallback_still_uses_distinct_positions():
    """Relaxing separation never relaxes the distinct-position invariant."""
    cfg = Config()
    cfg.source_mode = "multi_point"
    cfg.sources.subcortical_fraction = 0.0
    cfg.multi_point.n_sources_min = 2
    cfg.multi_point.n_sources_max = 2
    cfg.multi_point.min_separation_mm = 1000.0
    cfg.multi_point.max_separation_attempts = 1
    cfg.multi_point.separation_failure_policy = "accept_and_flag"

    positions = np.array([
        [0.050, 0.000, 0.000],
        [0.051, 0.000, 0.000],
    ])
    leadfields = np.zeros((2, 3, 3), dtype=np.float64)
    populations = np.zeros(2, dtype=np.int8)
    time_courses = np.ones((1, cfg.signal.window_samples), dtype=np.float64)

    batch = generate_multi_point_samples(
        cfg,
        n_samples=1,
        position_pool_xyz=positions,
        leadfield_pool=leadfields,
        population_pool=populations,
        time_courses=time_courses,
        rng=np.random.default_rng(4),
    )

    active = batch.extras["position_idx_in_pool"][0, :2]
    assert len(set(active.tolist())) == 2
    assert bool(batch.extras["separation_violated"][0])


def test_multi_point_rejects_a_pool_smaller_than_the_configured_maximum():
    """Every configured source count must be representable by distinct positions."""
    cfg = Config()
    cfg.source_mode = "multi_point"
    cfg.sources.subcortical_fraction = 0.0
    cfg.multi_point.n_sources_min = 1
    cfg.multi_point.n_sources_max = 2
    cfg.multi_point.min_separation_mm = 0.0
    cfg.multi_point.separation_failure_policy = "accept_and_flag"

    with pytest.raises(ValueError, match="distinct eligible positions"):
        generate_multi_point_samples(
            cfg,
            n_samples=1,
            position_pool_xyz=np.array([[0.050, 0.000, 0.000]]),
            leadfield_pool=np.zeros((1, 3, 3), dtype=np.float64),
            population_pool=np.zeros(1, dtype=np.int8),
            time_courses=np.ones((1, cfg.signal.window_samples), dtype=np.float64),
            rng=np.random.default_rng(4),
        )


# ---------------------------------------------------------------------------
# The swapped-convention branch must not leave the two representations
# divergent (Cartesian stays authoritative)
# ---------------------------------------------------------------------------

def test_spherical_columns_are_recomputed_from_cartesian():
    array = load_electrodes(
        resolve_electrode_path("standard_1010_69.csv"), target_radius_m=0.092
    )
    radii = np.linalg.norm(array.cartesian, axis=1)
    theta = np.arccos(np.clip(array.cartesian[:, 2] / radii, -1.0, 1.0))
    phi = np.mod(np.arctan2(array.cartesian[:, 1], array.cartesian[:, 0]), 2 * np.pi)
    np.testing.assert_allclose(array.spherical[:, 0], radii, atol=1e-15)
    np.testing.assert_allclose(array.spherical[:, 1], theta, atol=1e-15)
    np.testing.assert_allclose(array.spherical[:, 2], phi, atol=1e-15)


# ---------------------------------------------------------------------------
# The validation manifest must bind everything the paper says it binds
# ---------------------------------------------------------------------------

def test_validation_manifest_is_in_sync():
    """The manifest binds results, scripts, and documentation together.

    This guard fails on any drift, so a regenerated result or an edited script
    cannot silently desynchronize the archive from the prose.
    """
    import subprocess

    repo_root = Path(__file__).resolve().parent.parent
    script = repo_root / "scripts" / "make_validation_manifest.py"
    if not script.is_file():  # pragma: no cover - script ships with the package
        pytest.skip("manifest script is not present in this tree")

    completed = subprocess.run(
        [sys.executable, str(script), "--check"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, (
        "Review validation archive drift before updating SHA256SUMS with "
        "`python scripts/make_validation_manifest.py --write-canonical`.\n"
        + completed.stdout + completed.stderr
    )


def test_headline_ecd_validation_artifacts_are_archived():
    """The abstract's position, orientation, and moment figures come from this
    run, so its summary and per-sample CSV must ship with the release."""
    import json

    repo_root = Path(__file__).resolve().parent.parent
    results = repo_root / "validation_results"
    summary_path = results / "ecd_69ch_35db_summary.json"
    assert summary_path.is_file(), "the headline ECD summary is not archived"

    summary = json.loads(summary_path.read_text())
    csv_path = results / summary["metrics_csv"]
    assert csv_path.is_file(), "the per-sample ECD CSV is not archived"

    import hashlib

    assert (
        hashlib.sha256(csv_path.read_bytes()).hexdigest()
        == summary["metrics_csv_sha256"]
    ), "the archived CSV does not match the hash recorded in the summary"

    # Equivalent-current-dipole fitting is iterative, so the archived numbers
    # are only interpretable against the stack that produced them.
    assert "software_environment" in summary
    environment = summary["software_environment"]
    assert "python_version" in environment
    for key in ("numpy", "scipy", "mne"):
        assert key in environment["software_versions"]


# ---------------------------------------------------------------------------
# The declared version must be single-sourced
# ---------------------------------------------------------------------------

def test_citation_metadata_declares_the_package_version():
    """CITATION.cff cannot drift from spherical_eeg_sim.__version__."""
    from spherical_eeg_sim import __version__

    citation = Path(__file__).resolve().parent.parent / "CITATION.cff"
    if not citation.is_file():
        pytest.skip("CITATION.cff is not present in this tree")
    declared = None
    for line in citation.read_text(encoding="utf-8").splitlines():
        if line.startswith("version:"):
            declared = line.split(":", 1)[1].strip().strip('"').strip("'")
            break
    assert declared is not None, "CITATION.cff does not declare a version"
    assert declared == __version__


def test_packaging_metadata_declares_the_package_version():
    """pyproject.toml must take its version from the package attribute."""
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    if not pyproject.is_file():
        pytest.skip("pyproject.toml is not present in this tree")
    text = pyproject.read_text(encoding="utf-8")
    assert 'dynamic = ["version"]' in text
    assert 'version = {attr = "spherical_eeg_sim.__version__"}' in text


# ---------------------------------------------------------------------------
# Round-off in the transfer recurrence must be reported, not silent
# ---------------------------------------------------------------------------

def test_transfer_coefficient_accuracy_bound_is_reported_per_order():
    """The estimated bound must grow with order and flag the unusable tail."""
    from spherical_eeg_sim.core.forward import (
        COEFFICIENT_RELIABILITY_LIMIT,
        estimate_transfer_coefficient_accuracy,
    )

    radii = [0.080, 0.081, 0.086, 0.092]
    sigma = [0.330, 1.790, 0.0042, 0.330]
    bound = estimate_transfer_coefficient_accuracy(radii, sigma, 120)

    assert bound.shape == (120,)
    # The default order stays far inside the reliable range.
    assert bound[79] < COEFFICIENT_RELIABILITY_LIMIT
    # The top of the accepted range does not.
    assert bound[119] > COEFFICIENT_RELIABILITY_LIMIT
    # Cancellation worsens with order rather than fluctuating arbitrarily.
    assert bound[99] > bound[59] > bound[19]


def test_unreliable_transfer_coefficients_warn_but_still_return():
    """High orders warn, name the first bad order, and remain computable."""
    from spherical_eeg_sim.core.forward import compute_transfer_coefficients

    radii = [0.080, 0.081, 0.086, 0.092]
    sigma = [0.330, 1.790, 0.0042, 0.330]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        high = compute_transfer_coefficients(radii, sigma, 120)
    assert high.shape == (120,)
    messages = [str(w.message) for w in caught
                if issubclass(w.category, RuntimeWarning)]
    assert any("Legendre order >=" in m for m in messages)

    # The audited default order must stay silent, or the warning is noise.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compute_transfer_coefficients(radii, sigma, 80)
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]


# ---------------------------------------------------------------------------
# Patch support density is set by the pool, not by the configured cap
# ---------------------------------------------------------------------------

def test_sparse_patch_support_is_reported_before_generation():
    """A pool too coarse to resolve the smallest patch must be flagged."""
    cfg = Config()
    cfg.source_mode = "patch"
    cfg.sources.n_cortical_positions = 2000

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg.validate()
    messages = [str(w.message) for w in caught
                if issubclass(w.category, RuntimeWarning)]
    assert any("Patch support is sparse" in m for m in messages)
    assert any("dipoles_per_mm2" in m for m in messages)


def test_default_patch_configuration_does_not_warn():
    """The shipped defaults must resolve the smallest configured patch."""
    cfg = Config()
    cfg.source_mode = "patch"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg.validate()
    assert not [w for w in caught
                if "Patch support is sparse" in str(w.message)]


def test_point_mode_does_not_report_patch_support():
    """The patch-support check must not fire for non-patch source modes."""
    cfg = Config()
    cfg.source_mode = "point"
    cfg.sources.n_cortical_positions = 300

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg.validate()
    assert not [w for w in caught
                if "Patch support is sparse" in str(w.message)]
