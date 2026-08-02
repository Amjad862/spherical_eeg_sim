"""
Main pipeline orchestrator.

Wires together every other module into a single end-to-end run:

    config -> validate forward model
           -> load electrodes
           -> compute transfer coefficients (M_tilde)
           -> sample cortical and subcortical positions
           -> compute basis lead-field tensor
           -> compute background-noise lead-field pool
           -> assign train/val/test position-level splits
           -> for each split, in chunks of cfg.output.chunk_samples:
                generate signals, noise, and EEG via the mode dispatcher
                append the chunk to an extendable HDF5 dataset (or
                accumulate-then-write for NPZ/MAT)
           -> validate distributions on the full training split and audit
              serialized split indices
           -> write top-level metadata.json + README + diagnostic plots

Usage:

    from spherical_eeg_sim import Config, run_pipeline
    cfg = Config()
    run_pipeline(cfg)
"""

from __future__ import annotations
import json
import hashlib
import logging
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .config import Config
from ._compat import import_hdf5storage
from .provenance import environment_snapshot
from .ragged import RaggedArray, concatenate_ragged
from .core.forward import (
    compute_transfer_coefficients,
    forward_potential_batch,
)
from .core.electrodes import load_electrodes, resolve_reference, apply_reference
from .sources.sampling import (
    sample_cortical_positions,
    sample_subcortical_positions,
    cartesian_to_spherical,
    make_position_split,
)
from .sources.modes import (
    generate_point_samples,
    generate_multi_point_samples,
    generate_patch_samples,
    compute_background_leadfield_pool,
)
from .signals.source_signals import (
    generate_signal_batch,
    resolve_effective_band_probabilities,
)
from .signals.noise import (
    generate_pink_noise,
    generate_white_noise,
    generate_physiological_background,
    mix_noise_components,
    sample_snr_db,
)
from .io.writer import (
    canonical_config,
    open_split_writer,
    write_leadfield_operator,
    write_source_pool,
    write_checksum_manifest,
    write_run_manifest,
    write_run_readme,
)
from .io.graph_export import write_electrode_graph
from .validation.forward_check import validate_forward_model
from .validation.distribution_check import validate_distributions


logger = logging.getLogger('spherical_eeg_sim')


# =============================================================================
# Run pipeline
# =============================================================================

def run_pipeline(cfg: Config, log_level: int = logging.INFO) -> Dict[str, Any]:
    """Execute the full data-generation pipeline.

    Parameters
    ----------
    cfg : Config
        The fully-populated configuration object.
    log_level : int
        Python logging level (default INFO).

    Returns
    -------
    summary : dict
        Run summary with paths to outputs and key statistics.
    """
    cfg.validate()
    _setup_logging(log_level)
    run_started = time.time()
    target = Path(cfg.output_dir).expanduser().resolve()
    if target in (Path('/'), Path.cwd().resolve()):
        raise ValueError("output_dir must be a dedicated dataset directory")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not target.is_dir():
        raise NotADirectoryError(
            f"Output path {target} exists and is not a directory"
        )
    if _recover_interrupted_publish(target, cfg):
        wall_time_seconds = time.time() - run_started
        runtime_path = target.parent / f'{target.name}.runtime.json'
        runtime_temp = target.parent / f'.{target.name}.runtime.writing.json'
        runtime = {
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'output_dir': str(target),
            'recovered_completed_staging': True,
            'wall_time_seconds': float(wall_time_seconds),
            **environment_snapshot(),
        }
        with runtime_temp.open('w', newline='\n') as f:
            json.dump(runtime, f, indent=2, sort_keys=True)
            f.write('\n')
        _replace_with_retry(runtime_temp, runtime_path)
        suffix = {'hdf5': 'h5', 'npz': 'npz', 'mat': 'mat'}[cfg.output.format]
        logger.warning("Recovered and reused a checksum-complete staged dataset at %s", target)
        return {
            'output_dir': str(target),
            'splits': {
                split: str(target / f'{split}.{suffix}')
                for split in ('train', 'val', 'test')
            },
            'manifest': str(target / 'metadata.json'),
            'validation_report': str(target / 'validation_report.json'),
            'checksums': str(target / 'checksums.sha256'),
            'runtime_provenance': str(runtime_path),
            'wall_time_seconds': float(wall_time_seconds),
        }
    target_nonempty = target.exists() and any(target.iterdir())
    if target_nonempty and not cfg.output.overwrite:
        raise FileExistsError(
            f"Output directory {target} is not empty. Choose a new directory "
            "or set output.overwrite=True for staged replacement."
        )
    if target_nonempty:
        _assert_replaceable_dataset_directory(target)

    staging = Path(tempfile.mkdtemp(
        prefix=f'.{target.name}.staging-', dir=target.parent,
    ))
    # This internal marker makes an incomplete staging tree demonstrably
    # generator-owned. It is removed before a completed tree is published.
    marker = {
        'package': 'spherical_eeg_sim',
        'transaction_kind': 'dataset_staging',
        'target_name': target.name,
        'canonical_config': canonical_config(cfg.to_dict()),
    }
    (staging / '.spherical_eeg_sim.transaction.json').write_text(
        json.dumps(marker, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )
    try:
        summary = _run_pipeline_impl(
            cfg, staging, log_level, published_output_dir=target,
        )
        _remove_staging_marker(staging)
        _commit_staged_directory(staging, target, cfg.output.overwrite)
    except Exception:
        # Preserve a checksum-complete staging tree.  In particular, an
        # exhausted Windows sharing-violation retry during publication must not
        # discard hours of completed generation; the next invocation can
        # verify and publish this tree without regenerating it.
        if staging.exists() and not _directory_checksum_complete(staging):
            _rmtree_with_retry(staging)
        elif staging.exists():
            logger.error(
                "Preserving checksum-complete staging tree for startup recovery: %s",
                staging,
            )
        raise

    runtime_path = target.parent / f'{target.name}.runtime.json'
    runtime_temp = target.parent / f'.{target.name}.runtime.writing.json'
    wall_time_seconds = time.time() - run_started
    runtime = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'output_dir': str(target),
        'chunk_samples': int(cfg.output.chunk_samples),
        'n_workers_requested': int(cfg.output.n_workers),
        'n_workers_used': int(_resolve_n_workers(cfg.output.n_workers)),
        'wall_time_seconds': float(wall_time_seconds),
        **environment_snapshot(),
    }
    with runtime_temp.open('w', newline='\n') as f:
        json.dump(runtime, f, indent=2, sort_keys=True)
        f.write('\n')
    _replace_with_retry(runtime_temp, runtime_path)

    logger.info("")
    logger.info("=" * 70)
    logger.info(" Generation complete in %.1fs", wall_time_seconds)
    logger.info(" Output: %s", target)
    logger.info("=" * 70)

    split_paths = {
        name: str(target / Path(path).name)
        for name, path in summary['splits'].items()
    }
    return {
        'output_dir': str(target),
        'splits': split_paths,
        'manifest': str(target / 'metadata.json'),
        'validation_report': str(target / 'validation_report.json'),
        'checksums': str(target / 'checksums.sha256'),
        'runtime_provenance': str(runtime_path),
        'wall_time_seconds': float(wall_time_seconds),
    }


def _run_pipeline_impl(
    cfg: Config,
    output_dir: Path,
    log_level: int = logging.INFO,
    published_output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Generate one complete canonical dataset inside a staging directory."""

    logger.info("=" * 70)
    logger.info(" EEG SOURCE LOCALIZATION DATASET GENERATOR")
    logger.info("=" * 70)
    logger.info(
        "Output directory: %s",
        (published_output_dir or output_dir).resolve(),
    )
    logger.info("Source mode     : %s", cfg.source_mode)
    logger.info(
        "Target sizes    : train=%d  val=%d  test=%d",
        cfg.dataset.n_train, cfg.dataset.n_val, cfg.dataset.n_test,
    )

    t0 = time.time()

    # -------------------------------------------------------------------------
    # 1. Forward-model validation (Layer 1)
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("[1/8] Validating forward model...")
    fv = validate_forward_model(
        radii_m=tuple(cfg.head.radii_m),
        conductivities_S_per_m=tuple(cfg.head.conductivities_S_per_m),
        n_legendre=cfg.head.n_legendre,
        n_test_dipoles=50,
    )
    if not fv.all_passed:
        logger.error("Forward-model validation FAILED")
        for n in fv.numerical_notes:
            logger.error(" - %s", n)
        raise RuntimeError(
            "Forward-model self-validation failed. See log for details."
        )
    logger.info(
        "      transfer-coeff match  : err = %.2e (passed)",
        fv.transfer_coefficient_max_error,
    )
    logger.info(
        "      basis consistency     : max diff = %.2e (passed)",
        fv.basis_max_error,
    )
    logger.info(
        "      rotational covariance : relative err = %.2e (passed)",
        fv.rotation_covariance_max_relative_error,
    )
    for n in fv.numerical_notes:
        logger.info("      %s", n)

    # -------------------------------------------------------------------------
    # 2. Load electrodes
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("[2/8] Loading electrode array...")
    electrodes = load_electrodes(
        cfg.electrode_file,
        target_radius_m=cfg.head.radii_m[-1],
    )
    logger.info(
        "      Loaded %d electrodes from %s (scalp radius = %.4f m)",
        electrodes.n_electrodes, cfg.electrode_file, electrodes.radius_m,
    )
    # Montage-dependent configuration errors should fail before lead-field
    # computation begins.
    ref_desc = resolve_reference(
        electrodes.names,
        cfg.output.reference_mode,
        cfg.output.reference_channel,
        cfg.output.reference_weights,
    )
    if (cfg.output.save_graph and cfg.output.graph_method == 'knn'
            and cfg.output.graph_k >= electrodes.n_electrodes):
        raise ValueError(
            "output.graph_k must be smaller than the montage electrode count"
        )
    if (cfg.output.save_graph and cfg.output.graph_method == 'delaunay'
            and electrodes.n_electrodes < 4):
        raise ValueError("Delaunay graph export requires at least 4 electrodes")

    # -------------------------------------------------------------------------
    # 3. Compute transfer coefficients
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("[3/8] Computing transfer coefficients (M_tilde, n_legendre=%d)...",
                cfg.head.n_legendre)
    t = time.time()
    M_tilde = compute_transfer_coefficients(
        radii_m=cfg.head.radii_m,
        conductivities_S_per_m=cfg.head.conductivities_S_per_m,
        n_legendre=cfg.head.n_legendre,
    )
    logger.info("      Done in %.2fs", time.time() - t)

    # -------------------------------------------------------------------------
    # 4. Sample positions and assign splits
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("[4/8] Sampling positions and computing position-level splits...")

    position_seed = cfg.master_seed if cfg.position_seed is None else cfg.position_seed
    sample_seed = cfg.master_seed if cfg.sample_seed is None else cfg.sample_seed
    master_rng = np.random.default_rng(position_seed)
    # Independent sub-streams for the one-off setup stages only: the position
    # pool, the train/val/test split assignment, and the background-noise
    # lead-field pool. Per-sample data generation does NOT draw from these.
    # Instead every sample derives its own seeds from
    # (master_seed, split_id, global_sample_index), so the generated data are
    # a pure function of the master seed and the sample's global index and are
    # therefore identical regardless of the streaming chunk size. See
    # ``_generate_one_sample``.
    rng_pos = np.random.default_rng(master_rng.integers(0, 2**31))
    rng_split = np.random.default_rng(master_rng.integers(0, 2**31))
    rng_bg_pool = np.random.default_rng(master_rng.integers(0, 2**31))

    cort_xyz = sample_cortical_positions(
        cfg.sources.n_cortical_positions,
        cfg.sources.cortical_radius_range_mm,
        cfg.sources.cortical_sampling,
        rng_pos,
    )
    if cfg.source_mode == 'patch':
        sub_xyz = np.empty((0, 3), dtype=np.float64)
    else:
        sub_xyz = sample_subcortical_positions(
            cfg.sources.n_subcortical_positions,
            cfg.sources.subcortical_radius_range_mm,
            cfg.sources.subcortical_sampling,
            rng_pos,
        )
    positions_xyz = np.concatenate([cort_xyz, sub_xyz], axis=0)
    population = np.concatenate([
        np.zeros(cort_xyz.shape[0], dtype=np.int8),
        np.ones(sub_xyz.shape[0], dtype=np.int8),
    ])
    logger.info(
        "      %d cortical + %d subcortical = %d positions",
        cort_xyz.shape[0], sub_xyz.shape[0], positions_xyz.shape[0],
    )

    # Patch support is drawn from the split-local cortical mesh.  Equal-sized
    # position meshes keep its spatial sampling density comparable across
    # train/validation/test even when the numbers of generated samples differ.
    # Point and multi-point modes retain sample-proportional position pools.
    if cfg.source_mode == 'patch':
        position_split_targets = (1, 1, 1)
        position_split_basis = 'equal-density patch support meshes'
    else:
        position_split_targets = (
            cfg.dataset.n_train, cfg.dataset.n_val, cfg.dataset.n_test,
        )
        position_split_basis = 'sample-count proportions'
    split = make_position_split(
        positions_xyz,
        *position_split_targets,
        cfg.split.stratify_by_depth, cfg.split.depth_bins,
        rng_split,
        population=population,
    )
    train_global = np.where(split.split == 0)[0]
    val_global = np.where(split.split == 1)[0]
    test_global = np.where(split.split == 2)[0]
    n_train_pos = int(train_global.size)
    n_val_pos = int(val_global.size)
    n_test_pos = int(test_global.size)
    logger.info(
        "      Position split: train=%d, val=%d, test=%d (positions, not samples)",
        n_train_pos, n_val_pos, n_test_pos,
    )
    logger.info("      Position allocation basis: %s", position_split_basis)

    # -------------------------------------------------------------------------
    # 5. Compute basis lead-field tensor at all positions
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("[5/8] Computing basis lead-field tensor (%d positions x %d channels x 3)...",
                positions_xyz.shape[0], electrodes.n_electrodes)
    positions_spherical = cartesian_to_spherical(positions_xyz)
    t = time.time()
    leadfield = forward_potential_batch(
        positions_spherical,
        electrodes.theta_rad,
        electrodes.phi_rad,
        M_tilde,
        brain_radius_m=cfg.head.radii_m[0],
        progress_callback=_make_progress_logger("      lead-field", interval=0.1),
    )
    logger.info(
        "      Done in %.2fs (size %.1f MB)",
        time.time() - t,
        leadfield.nbytes / 1e6,
    )

    # -------------------------------------------------------------------------
    # 6. Background lead-field pool for the source-projected noise proxy
    # -------------------------------------------------------------------------
    logger.info("")
    bg_pool_size = cfg.noise.background_pool_size
    logger.info("[6/8] Computing background lead-field pool (%d sources)...", bg_pool_size)
    bg_pool = compute_background_leadfield_pool(
        n_pool=bg_pool_size,
        background_radius_range_mm=cfg.noise.background_radius_range_mm,
        elec_theta=electrodes.theta_rad,
        elec_phi=electrodes.phi_rad,
        M_tilde=M_tilde,
        brain_radius_m=cfg.head.radii_m[0],
        rng=rng_bg_pool,
    )
    logger.info("      Pool shape: %s", bg_pool.shape)

    # -------------------------------------------------------------------------
    # 7. Generate splits (chunked)
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("[7/8] Generating samples per split (chunk_samples=%d)...",
                cfg.output.chunk_samples)

    split_specs = [
        ('train', cfg.dataset.n_train, train_global),
        ('val',   cfg.dataset.n_val,   val_global),
        ('test',  cfg.dataset.n_test,  test_global),
    ]

    split_paths: Dict[str, Path] = {}
    sample_summary: Dict[str, Any] = {}

    # Full-split ground-truth statistics from the training split, retained in
    # memory for Layer-2 distribution validation. Only lightweight ground
    # truth is kept (never the full EEG tensor), so the validator inspects the
    # ENTIRE training split rather than a single representative chunk.
    train_val_stats: Optional[Dict[str, Any]] = None
    # Fixed leading-sample snapshot of the training split for diagnostic plots.
    # Its size is independent of generation chunks.
    train_diagnostic_sample: Optional[Dict[str, Any]] = None

    # The EEG reference was resolved immediately after montage loading so
    # montage-dependent errors fail before lead-field computation.
    logger.info("  Reference: %s", ref_desc['description'])

    # Resolve window-vs-band resolvability once and log the decision.
    eff_band_probs, band_info = resolve_effective_band_probabilities(
        cfg.signal.bands_hz,
        cfg.signal.band_probabilities,
        cfg.signal.window_samples,
        cfg.signal.sampling_freq_hz,
        cfg.signal.min_band_cycles,
        cfg.signal.low_freq_band_policy,
        band_order=cfg.signal.band_order,
    )
    logger.info("  Band policy: %s", band_info['note'])
    if cfg.signal.zero_mean_signal:
        logger.info("  Source time courses are zero-mean (DC removed per window).")

    # Self-describing electrode block embedded in every output file.
    electrodes_block = {
        'names': list(electrodes.names),
        'cartesian_xyz': np.asarray(electrodes.cartesian, dtype=np.float64),
        'spherical_r_theta_phi': np.asarray(electrodes.spherical, dtype=np.float64),
        'reference_mode': cfg.output.reference_mode,
        'reference_info': {
            k: v for k, v in ref_desc.items() if k not in ('w',)
        },
    }

    # Optional: write the position-pool lead-field operator for downstream
    # reuse (classical inverse operators, physics-informed models needing the
    # forward map). Shared across splits, so written once.
    if cfg.output.save_leadfield:
        leadfield_to_write = apply_reference(leadfield, ref_desc)
        lf_path = write_leadfield_operator(
            output_dir=output_dir,
            leadfield=leadfield_to_write,
            positions_xyz=positions_xyz,
            population=population,
            split_label=split.split,
            electrodes=electrodes_block,
            head={
                'radii_m': list(cfg.head.radii_m),
                'conductivities_S_per_m': list(cfg.head.conductivities_S_per_m),
                'n_legendre': int(cfg.head.n_legendre),
            },
            fmt=cfg.output.format,
            compress=cfg.output.compress,
        )
        logger.info("  Wrote lead-field operator %s (%.1f MB)",
                    lf_path.name, lf_path.stat().st_size / 1e6)

    # Every sample records an index into this global source-position pool.
    # Patch CSR support indices reference the same table.
    pool_path = write_source_pool(
        output_dir=output_dir,
        positions_xyz=positions_xyz,
        population=population,
        split_label=split.split,
        fmt=cfg.output.format,
        compress=cfg.output.compress,
    )
    logger.info("  Wrote source-position pool %s", pool_path.name)

    # Optional: write the electrode adjacency graph for GNN workflows. It
    # depends only on the montage, so it is written once.
    if cfg.output.save_graph:
        graph_path = write_electrode_graph(
            output_dir=output_dir,
            electrodes=electrodes_block,
            method=cfg.output.graph_method,
            k=cfg.output.graph_k,
            distance_threshold_m=cfg.output.graph_distance_threshold_m,
            fmt=cfg.output.format,
            compress=cfg.output.compress,
        )
        logger.info("  Wrote electrode graph %s (method=%s)",
                    graph_path.name, cfg.output.graph_method)

    for split_name, n_samples, pool_indices in split_specs:
        logger.info("")
        logger.info("  --- Split: %s (%d samples) ---", split_name, n_samples)
        if pool_indices.size == 0:
            raise RuntimeError(f"Empty position pool for split {split_name!r}")

        pool_pos = positions_xyz[pool_indices]
        pool_lf = leadfield[pool_indices]
        pool_pop = population[pool_indices]

        path, snr_db_split, diagnostic_sample, val_stats = _run_one_split(
            cfg=cfg,
            split_name=split_name,
            split_id=SPLIT_IDS[split_name],
            n_samples=n_samples,
            output_dir=output_dir,
            master_seed=sample_seed,
            pool_pos_xyz=pool_pos,
            pool_leadfield=pool_lf,
            pool_population=pool_pop,
            pool_global_indices=pool_indices,
            bg_pool=bg_pool,
            n_channels=electrodes.n_electrodes,
            ref_desc=ref_desc,
            eff_band_probs=eff_band_probs,
            band_info=band_info,
            electrodes_block=electrodes_block,
        )
        split_paths[split_name] = path
        sample_summary[split_name] = {
            'n_samples': n_samples,
            'snr_db_min': float(snr_db_split.min()),
            'snr_db_max': float(snr_db_split.max()),
            'snr_db_mean': float(snr_db_split.mean()),
        }
        if 'separation_violated_rate' in val_stats:
            sample_summary[split_name]['separation_violated_rate'] = \
                val_stats['separation_violated_rate']
            sample_summary[split_name]['separation_violated_count'] = \
                val_stats['separation_violated_count']

        if split_name == 'train':
            train_val_stats = val_stats
            train_diagnostic_sample = diagnostic_sample

    if train_val_stats is None or train_diagnostic_sample is None:
        raise RuntimeError("training split did not produce validation statistics")

    # -------------------------------------------------------------------------
    # 8. Distribution validation (Layer 2)
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("[8/8] Running distribution validation on the full train split...")

    serialized_position_indices = {
        name: _read_serialized_position_indices(path, cfg.source_mode)
        for name, path in split_paths.items()
    }
    assigned_position_indices = {
        'train': train_global,
        'val': val_global,
        'test': test_global,
    }

    dist_result = validate_distributions(
        positions_xyz=train_val_stats['positions_xyz'],
        populations=train_val_stats['population'],
        magnitudes_nAm=train_val_stats['magnitudes_nAm'],
        snr_db=train_val_stats['snr_db'],
        peak_uV=train_val_stats['peak_uV'],
        cortical_radius_range_mm=cfg.sources.cortical_radius_range_mm,
        subcortical_radius_range_mm=cfg.sources.subcortical_radius_range_mm,
        cortical_magnitude_range_nAm=cfg.dipole.cortical_magnitude_range_nAm,
        subcortical_magnitude_range_nAm=cfg.dipole.subcortical_magnitude_range_nAm,
        snr_db_range=cfg.noise.snr_db_range,
        expected_subcortical_fraction=cfg.sources.subcortical_fraction,
        magnitude_distribution=cfg.dipole.magnitude_distribution,
        orientation_mode=cfg.dipole.orientation_mode,
        orientations_world=train_val_stats['orientations_world'],
        train_position_indices=train_global,
        val_position_indices=val_global,
        test_position_indices=test_global,
        assigned_position_indices=assigned_position_indices,
        serialized_position_indices=serialized_position_indices,
        source_mode=cfg.source_mode,
        noise_model=cfg.noise.noise_model,
    )
    for c in dist_result.checks:
        status = 'PASS' if c.passed else 'FAIL'
        logger.info("      [%s] %-32s %s", status, c.name, c.detail)
    for diagnostic in dist_result.diagnostics:
        logger.info(
            "      [INFO] %-32s %s", diagnostic.name, diagnostic.detail,
        )
    val_report_path = output_dir / 'validation_report.json'
    val_report_temp = output_dir / '.validation_report.writing.json'
    from .io.writer import _jsonify
    with val_report_temp.open('w', newline='\n') as f:
        json.dump(_jsonify({
            'forward_model': {
                'transfer_match': fv.transfer_coefficient_match,
                'transfer_max_error': fv.transfer_coefficient_max_error,
                'basis_consistency': fv.basis_consistency,
                'basis_max_error': fv.basis_max_error,
                'rotation_covariance': fv.rotation_covariance,
                'rotation_covariance_max_relative_error': (
                    fv.rotation_covariance_max_relative_error
                ),
                'numerical_sanity': fv.numerical_sanity,
                'numerical_notes': fv.numerical_notes,
                'n_test_dipoles': fv.n_test_dipoles,
            },
            'distributions': dist_result.to_dict(),
        }), f, indent=2, sort_keys=True)
        f.write('\n')
    _replace_with_retry(val_report_temp, val_report_path)
    if not dist_result.all_passed:
        failed = ', '.join(c.name for c in dist_result.checks if not c.passed)
        raise RuntimeError(
            "Distribution validation failed; staged dataset was not published. "
            f"Failed checks: {failed}"
        )

    environment_path = output_dir / 'environment.json'
    environment_temp = output_dir / '.environment.writing.json'
    with environment_temp.open('w', encoding='utf-8', newline='\n') as handle:
        json.dump(environment_snapshot(), handle, indent=2, sort_keys=True)
        handle.write('\n')
    _replace_with_retry(environment_temp, environment_path)

    # -------------------------------------------------------------------------
    # Manifest + README
    # -------------------------------------------------------------------------
    extras = {
        'electrode_array': {
            'n_electrodes': electrodes.n_electrodes,
            'names': electrodes.names,
        },
        'source_pool': {
            'n_total': int(positions_xyz.shape[0]),
            'n_cortical': int((population == 0).sum()),
            'n_subcortical': int((population == 1).sum()),
            'split_train_positions': n_train_pos,
            'split_val_positions': n_val_pos,
            'split_test_positions': n_test_pos,
            'allocation_basis': position_split_basis,
        },
        'samples_per_split': sample_summary,
        'reference': {
            'mode': cfg.output.reference_mode,
            'description': ref_desc.get('description', ''),
        },
        'output_dtype': cfg.output.dtype,
        'eeg_normalization': cfg.output.eeg_normalization,
        'eeg_units': (
            'dimensionless_rms_normalized'
            if cfg.output.eeg_normalization == 'per_sample_rms' else 'volts'
        ),
        'physical_voltage_reconstruction': (
            'eeg_V = eeg * ground_truth/eeg_normalization_scale_V'
        ),
        'signal_zero_mean': bool(cfg.signal.zero_mean_signal),
        'band_resolvability': band_info,
        'forward_model_validation': {
            'all_passed': fv.all_passed,
            'transfer_max_error': fv.transfer_coefficient_max_error,
            'basis_max_error': fv.basis_max_error,
            'rotation_covariance_max_relative_error': (
                fv.rotation_covariance_max_relative_error
            ),
        },
        'distribution_validation_passed': dist_result.all_passed,
    }
    write_run_readme(output_dir, cfg.to_dict(), {
        'n_channels': electrodes.n_electrodes,
        'n_train': cfg.dataset.n_train,
        'n_val': cfg.dataset.n_val,
        'n_test': cfg.dataset.n_test,
    })

    # -------------------------------------------------------------------------
    # Diagnostic plots
    # -------------------------------------------------------------------------
    if cfg.output.diagnostic_plots:
        try:
            _make_diagnostic_plots(
                output_dir / 'diagnostic_plots',
                positions_xyz, population,
                train_diagnostic_sample['snr_db'],
                train_diagnostic_sample['magnitudes_nAm'],
                train_diagnostic_sample['eeg_signal'],
            )
        except ImportError as e:
            raise RuntimeError(
                "output.diagnostic_plots=True requires matplotlib; install "
                "spherical_eeg_sim[plot]"
            ) from e
        except Exception as e:
            raise RuntimeError("Diagnostic plot generation failed") from e

    # The manifest is written last so its inventory covers every canonical
    # artifact, including auxiliaries, diagnostics, the README, the validation
    # report, and the manifest itself. Runtime provenance remains outside the
    # canonical directory by design.
    output_files = sorted(
        [
            str(path.relative_to(output_dir).as_posix())
            for path in output_dir.rglob('*')
            if path.is_file() and not path.name.startswith('.')
        ]
        + ['metadata.json', 'checksums.sha256']
    )
    extras['output_files'] = output_files
    write_run_manifest(output_dir, cfg.to_dict(), extras)
    checksum_path = write_checksum_manifest(output_dir)

    return {
        'output_dir': str(output_dir.resolve()),
        'splits': {k: str(v) for k, v in split_paths.items()},
        'manifest': str((output_dir / 'metadata.json').resolve()),
        'validation_report': str(val_report_path.resolve()),
        'checksums': str(checksum_path.resolve()),
        'wall_time_seconds': time.time() - t0,
    }


def _replace_with_retry(source: Path, destination: Path, attempts: int = 6) -> None:
    """Retry transient Windows sharing violations without masking other errors."""
    for attempt in range(attempts):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt + 1 == attempts:
                raise
            time.sleep(0.1 * (2 ** attempt))


def _rmtree_with_retry(path: Path, attempts: int = 6) -> None:
    """Remove an owned transaction tree despite transient Windows locks."""
    for attempt in range(attempts):
        try:
            shutil.rmtree(path)
            return
        except PermissionError:
            if attempt + 1 == attempts:
                raise
            time.sleep(0.1 * (2 ** attempt))


def _remove_staging_marker(staging: Path, attempts: int = 6) -> None:
    """Remove the private staging marker despite transient sharing locks."""
    marker = staging / '.spherical_eeg_sim.transaction.json'
    for attempt in range(attempts):
        try:
            marker.unlink(missing_ok=True)
            return
        except PermissionError:
            if attempt + 1 == attempts:
                raise
            time.sleep(0.1 * (2 ** attempt))


_ROLLBACK_MARKER = '.spherical_eeg_sim.rollback.json'


def _write_rollback_marker(target: Path) -> None:
    """Authenticate a dataset tree before it can become a rollback backup."""
    metadata_path = target / 'metadata.json'
    marker = {
        'package': 'spherical_eeg_sim',
        'transaction_kind': 'dataset_rollback',
        'target_name': target.name,
        'metadata_sha256': hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
    }
    marker_path = target / _ROLLBACK_MARKER
    with marker_path.open('w', encoding='utf-8', newline='\n') as handle:
        json.dump(marker, handle, indent=2, sort_keys=True)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())


def _remove_rollback_marker(directory: Path, attempts: int = 6) -> None:
    """Remove the private rollback marker despite transient sharing locks."""
    marker = directory / _ROLLBACK_MARKER
    for attempt in range(attempts):
        try:
            marker.unlink(missing_ok=True)
            return
        except PermissionError:
            if attempt + 1 == attempts:
                raise
            time.sleep(0.1 * (2 ** attempt))


def _is_owned_rollback(directory: Path, target: Path) -> bool:
    """Return whether a complete tree has an authentic rollback marker."""
    marker_path = directory / _ROLLBACK_MARKER
    metadata_path = directory / 'metadata.json'
    try:
        marker = json.loads(marker_path.read_text(encoding='utf-8'))
        metadata_digest = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    except (OSError, json.JSONDecodeError):
        return False
    return (
        marker.get('package') == 'spherical_eeg_sim'
        and marker.get('transaction_kind') == 'dataset_rollback'
        and marker.get('target_name') == target.name
        and marker.get('metadata_sha256') == metadata_digest
        and _directory_checksum_complete(directory)
    )


def _directory_checksum_complete(directory: Path) -> bool:
    """Return whether a staged/published directory has a valid full inventory."""
    metadata_path = directory / 'metadata.json'
    checksum_path = directory / 'checksums.sha256'
    try:
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        if metadata.get('package') != 'spherical_eeg_sim':
            return False
        declared = metadata.get('extras', {}).get('output_files')
        if not isinstance(declared, list) or not declared:
            return False
        if not all(isinstance(relative, str) and relative for relative in declared):
            return False
        declared_set = set(declared)
        if len(declared_set) != len(declared):
            return False
        for relative in declared:
            candidate = Path(relative)
            if (
                candidate.is_absolute()
                or '..' in candidate.parts
                or '\\' in relative
            ):
                return False
            resolved = directory / candidate
            if not resolved.is_file() or resolved.is_symlink():
                return False
        actual_files = {
            path.relative_to(directory).as_posix()
            for path in directory.rglob('*')
            if path.is_file() or path.is_symlink()
        }
        actual_canonical = actual_files - {
            '.spherical_eeg_sim.transaction.json',
            _ROLLBACK_MARKER,
        }
        if actual_canonical != declared_set:
            return False
        lines = checksum_path.read_text(encoding='ascii').splitlines()
        if not lines:
            return False
        checksum_members = set()
        for line in lines:
            expected, relative = line.split(maxsplit=1)
            relative = relative.strip()
            candidate = Path(relative)
            if (
                len(expected) != 64
                or any(char not in '0123456789abcdefABCDEF' for char in expected)
                or candidate.is_absolute()
                or '..' in candidate.parts
                or '\\' in relative
                or relative in checksum_members
            ):
                return False
            path = directory / candidate
            if not path.is_file() or path.is_symlink():
                return False
            checksum_members.add(relative)
            digest = hashlib.sha256()
            with path.open('rb') as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b''):
                    digest.update(block)
            if digest.hexdigest().lower() != expected.lower():
                return False
        if checksum_members != declared_set - {'checksums.sha256'}:
            return False
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return True


def _staging_matches_config(staging: Path, cfg: Config) -> bool:
    try:
        metadata = json.loads((staging / 'metadata.json').read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return False
    return metadata.get('config') == canonical_config(cfg.to_dict())


def _is_owned_incomplete_staging(staging: Path, target: Path) -> bool:
    """Validate the package/config marker before deleting an incomplete tree."""
    marker_path = staging / '.spherical_eeg_sim.transaction.json'
    try:
        marker = json.loads(marker_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        marker.get('package') == 'spherical_eeg_sim'
        and marker.get('transaction_kind') == 'dataset_staging'
        and marker.get('target_name') == target.name
        and isinstance(marker.get('canonical_config'), dict)
    )


def _recover_interrupted_publish(target: Path, cfg: Config) -> bool:
    """Recover the only crash points around staged directory publication.

    Incomplete generator-owned staging trees cannot be resumed at HDF5 chunk
    granularity and are removed before a clean restart.  A checksum-complete
    matching staging tree is published, while a previous-directory backup is
    restored if the target name is absent.  Ambiguous states fail closed.
    """
    parent = target.parent
    stagings = sorted(parent.glob(f'.{target.name}.staging-*'))
    backups = sorted(parent.glob(f'.{target.name}.previous-*'))
    complete_stagings = [p for p in stagings if _directory_checksum_complete(p)]

    if target.is_dir() and (target / _ROLLBACK_MARKER).exists():
        if backups or not _is_owned_rollback(target, target):
            raise RuntimeError(
                f"Ambiguous or unauthenticated rollback marker in {target}; "
                "no data were deleted."
            )
        _remove_rollback_marker(target)

    unowned_backups = [p for p in backups if not _is_owned_rollback(p, target)]
    if unowned_backups:
        raise RuntimeError(
            f"Refusing to use or delete unverified rollback-like directory/directories "
            f"{unowned_backups}; move them aside or inspect them manually."
        )

    if not target.exists() and backups:
        if len(backups) != 1 or len(complete_stagings) > 1:
            raise RuntimeError(
                f"Ambiguous interrupted publication for {target}: found "
                f"{len(backups)} backup(s) and {len(stagings)} staging tree(s). "
                "Inspect them manually; no data were deleted."
            )
        if complete_stagings:
            staging = complete_stagings[0]
            if cfg.output.overwrite and _staging_matches_config(staging, cfg):
                _remove_staging_marker(staging)
                _replace_with_retry(staging, target)
                _rmtree_with_retry(backups[0])
                for incomplete in stagings:
                    if (
                        incomplete != staging
                        and incomplete.is_dir()
                        and _is_owned_incomplete_staging(incomplete, target)
                    ):
                        _rmtree_with_retry(incomplete)
                logger.warning(
                    "Completed an interrupted replacement and removed its valid rollback"
                )
                return True
            _replace_with_retry(backups[0], target)
            _remove_rollback_marker(target)
            raise RuntimeError(
                f"Restored {target} from rollback, but complete staging tree {staging} "
                "was not published because overwrite/configuration did not match. "
                "No staging data were deleted."
            )
        _replace_with_retry(backups[0], target)
        _remove_rollback_marker(target)
        for incomplete in stagings:
            if incomplete.is_dir() and _is_owned_incomplete_staging(incomplete, target):
                _rmtree_with_retry(incomplete)
        logger.warning("Restored interrupted-publication backup to %s", target)
        return False

    if len(complete_stagings) > 1:
        raise RuntimeError(
            f"Multiple complete staging trees exist for {target}; refusing to guess: "
            f"{complete_stagings}"
        )
    if complete_stagings:
        staging = complete_stagings[0]
        if not _staging_matches_config(staging, cfg):
            raise RuntimeError(
                f"Complete staging tree {staging} belongs to a different scientific "
                "configuration; inspect it manually."
            )
        if target.exists() and not cfg.output.overwrite:
            raise FileExistsError(
                f"A complete interrupted staging tree exists at {staging}, but target "
                f"{target} also exists and overwrite is disabled."
            )
        if target.exists() and backups:
            raise RuntimeError(
                f"Ambiguous interrupted publication for {target}: a target, complete "
                "staging tree, and rollback backup are all present. No data were deleted."
            )
        _remove_staging_marker(staging)
        _commit_staged_directory(staging, target, cfg.output.overwrite)
        logger.warning("Published checksum-complete staging tree recovered at startup")
        stagings.remove(staging)
        for incomplete in stagings:
            if incomplete.is_dir() and _is_owned_incomplete_staging(incomplete, target):
                _rmtree_with_retry(incomplete)
        if backups:
            if not _directory_checksum_complete(target):
                raise RuntimeError("Recovered target failed checksum verification")
            for backup in backups:
                if backup.is_dir():
                    _rmtree_with_retry(backup)
        return True

    for staging in stagings:
        if staging.is_dir() and _is_owned_incomplete_staging(staging, target):
            _rmtree_with_retry(staging)
            logger.warning("Removed incomplete staging tree %s", staging)
        elif staging.is_dir():
            raise RuntimeError(
                f"Refusing to delete unverified staging-like directory {staging}; "
                "move it aside or inspect it manually."
            )

    if backups:
        if not target.exists() or not _directory_checksum_complete(target):
            raise RuntimeError(
                f"Found publication backup(s) {backups}, but the target is not a "
                "checksum-complete dataset. No backup was removed."
            )
        for backup in backups:
            if backup.is_dir():
                _rmtree_with_retry(backup)
                logger.warning("Removed obsolete publication backup %s", backup)
    return False


def _commit_staged_directory(staging: Path, target: Path, overwrite: bool) -> None:
    """Publish a complete staging directory, with rollback on replacement failure."""
    if target.exists() and not any(target.iterdir()):
        target.rmdir()
    if not target.exists():
        _replace_with_retry(staging, target)
        return
    if not overwrite:
        raise FileExistsError(f"Output directory {target} is not empty")

    _assert_replaceable_dataset_directory(target)
    _write_rollback_marker(target)

    backup = Path(tempfile.mkdtemp(
        prefix=f'.{target.name}.previous-', dir=target.parent,
    ))
    backup.rmdir()
    try:
        _replace_with_retry(target, backup)
    except Exception:
        _remove_rollback_marker(target)
        raise
    try:
        _replace_with_retry(staging, target)
    except Exception:
        _replace_with_retry(backup, target)
        _remove_rollback_marker(target)
        raise
    _rmtree_with_retry(backup)


def _assert_replaceable_dataset_directory(target: Path) -> None:
    """Refuse replacement unless every file is owned by a prior dataset run."""
    metadata_path = target / 'metadata.json'
    try:
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise FileExistsError(
            f"Refusing to replace {target}: it is not identified as a "
            "spherical_eeg_sim dataset directory"
        ) from exc
    if metadata.get('package') != 'spherical_eeg_sim':
        raise FileExistsError(
            f"Refusing to replace {target}: metadata.json is not owned by "
            "spherical_eeg_sim"
        )
    declared = metadata.get('extras', {}).get('output_files')
    if not isinstance(declared, list) or not all(
        isinstance(path, str) and path for path in declared
    ):
        raise FileExistsError(
            f"Refusing to replace {target}: metadata.json has no complete "
            "canonical artifact inventory"
        )
    actual = {
        path.relative_to(target).as_posix()
        for path in target.rglob('*') if path.is_file() or path.is_symlink()
    }
    unowned = actual - set(declared)
    if unowned:
        raise FileExistsError(
            f"Refusing to replace {target}: unowned file(s) are present: "
            f"{sorted(unowned)}"
        )
    if not _directory_checksum_complete(target):
        raise FileExistsError(
            f"Refusing to replace {target}: its canonical inventory or checksums "
            "are incomplete, mismatched, or unsafe"
        )


# =============================================================================
# Helpers
# =============================================================================

# Fixed integer id per split. Used (with the master seed and the per-sample
# global index) to derive each sample's seeds, so the three splits never share
# a random stream.
SPLIT_IDS = {'train': 0, 'val': 1, 'test': 2}


# Per-worker context for parallel generation. Populated once per worker process
# by ``_worker_init`` so the large read-only pools are not re-sent per task.
_WORKER_CTX: Dict[str, Any] = {}


def _worker_init(cfg, master_seed, split_id, pool_pos_xyz, pool_leadfield,
                 pool_population, pool_global_indices, bg_pool, n_channels, T,
                 ref_desc, eff_band_probs):
    """Initializer run once in each worker process: stash the shared state."""
    _WORKER_CTX.update(
        cfg=cfg, master_seed=master_seed, split_id=split_id,
        pool_pos_xyz=pool_pos_xyz, pool_leadfield=pool_leadfield,
        pool_population=pool_population, pool_global_indices=pool_global_indices,
        bg_pool=bg_pool, n_channels=n_channels, T=T,
        ref_desc=ref_desc, eff_band_probs=eff_band_probs,
    )


def _worker_generate(global_index: int):
    """Generate one sample inside a worker, using the stashed context."""
    c = _WORKER_CTX
    return _generate_one_sample(
        c['cfg'], c['master_seed'], c['split_id'], global_index,
        c['pool_pos_xyz'], c['pool_leadfield'], c['pool_population'],
        c['pool_global_indices'], c['bg_pool'], c['n_channels'], c['T'],
        c['ref_desc'], c['eff_band_probs'],
    )


def _resolve_n_workers(n_workers: int) -> int:
    """Map the configured n_workers (-1 = all cores) to a concrete count >= 1."""
    if n_workers == -1:
        import os
        return max(1, os.cpu_count() or 1)
    return max(1, int(n_workers))


def _generate_one_sample(
    cfg: Config,
    master_seed: int,
    split_id: int,
    global_index: int,
    pool_pos_xyz: np.ndarray,
    pool_leadfield: np.ndarray,
    pool_population: np.ndarray,
    pool_global_indices: np.ndarray,
    bg_pool: np.ndarray,
    n_channels: int,
    T: int,
    ref_desc: Dict[str, Any],
    eff_band_probs: Dict[str, float],
) -> Tuple[np.ndarray, Dict[str, Any], np.ndarray, Any]:
    """Generate a single sample's EEG and ground truth.

    Every random quantity for the sample is derived from
    ``SeedSequence([master_seed, split_id, global_index])``, spawned into six
    independent role generators:

        role 0 -> source time course (signal)
        role 1 -> source mode (position, orientation, magnitude, ...)
        role 2 -> physiological background noise
        role 3 -> pink instrumentation noise
        role 4 -> white instrumentation noise
        role 5 -> target SNR

    Because the seed depends only on the sample's global index within its
    split, the produced sample is a pure function of the master seed and that
    index -- and therefore independent of how the split is divided into
    streaming chunks. This fixed role ordering is the canonical reproducibility
    contract. Isolating target SNR and noise components prevents sensor count
    or one component's array shape from perturbing unrelated draws.

    Returns
    -------
    eeg : ndarray (1, n_channels, T)
        Signal + noise scalp potentials (volts).
    gt : dict
        Per-sample ground truth (each value has a leading dim of 1).
    eeg_signal_only : ndarray (1, n_channels, T)
        Signal-only scalp potentials (before noise).
    band_names : list of str
        Band-name ordering for the band index (constant across samples).
    """
    ss_sig, ss_mode, ss_phys, ss_pink, ss_white, ss_snr = np.random.SeedSequence(
        [int(master_seed), int(split_id), int(global_index)]
    ).spawn(6)
    rng_sig = np.random.default_rng(ss_sig)
    rng_mode = np.random.default_rng(ss_mode)
    rng_phys = np.random.default_rng(ss_phys)
    rng_pink = np.random.default_rng(ss_pink)
    rng_white = np.random.default_rng(ss_white)
    rng_snr = np.random.default_rng(ss_snr)

    sig = generate_signal_batch(
        n_samples=1,
        window_samples=T,
        sampling_freq_hz=cfg.signal.sampling_freq_hz,
        bands_hz=cfg.signal.bands_hz,
        band_probabilities=eff_band_probs,
        n_components_min=cfg.signal.n_components_min,
        n_components_max=cfg.signal.n_components_max,
        component_amplitude_jitter=cfg.signal.component_amplitude_jitter,
        component_dirichlet_concentration=(
            cfg.signal.component_dirichlet_concentration
        ),
        rng=rng_sig,
        zero_mean=cfg.signal.zero_mean_signal,
        band_order=cfg.signal.band_order,
    )
    time_courses = sig['signal']  # (1, T)

    if cfg.source_mode == 'point':
        batch = generate_point_samples(
            cfg, 1, pool_pos_xyz, pool_leadfield, pool_population,
            time_courses, rng_mode,
        )
    elif cfg.source_mode == 'multi_point':
        batch = generate_multi_point_samples(
            cfg, 1, pool_pos_xyz, pool_leadfield, pool_population,
            time_courses, rng_mode,
            band_probabilities=eff_band_probs,
            zero_mean_signal=cfg.signal.zero_mean_signal,
            primary_signal_metadata=sig,
        )
    elif cfg.source_mode == 'patch':
        batch = generate_patch_samples(
            cfg, 1, pool_pos_xyz, pool_leadfield, pool_population,
            time_courses, rng_mode,
        )
    else:
        raise ValueError(f"Unsupported source_mode: {cfg.source_mode!r}")

    pink = generate_pink_noise(
        1, n_channels, T, alpha=cfg.noise.background_freq_alpha, rng=rng_pink,
    )
    white = generate_white_noise(1, n_channels, T, rng=rng_white)
    phys = generate_physiological_background(
        1, n_channels, T,
        background_leadfields=bg_pool,
        n_background_sources=cfg.noise.n_background_sources,
        amplitude_concentration=cfg.noise.background_amplitude_concentration,
        alpha=cfg.noise.background_freq_alpha,
        rng=rng_phys,
    )
    noise_mixed = mix_noise_components(
        phys, pink, white,
        cfg.noise.mix_physiological, cfg.noise.mix_pink, cfg.noise.mix_white,
    )
    # The target stream is independent of sensor count and all noise arrays, so
    # paired datasets on different montages retain the same requested SNR.
    snr_db = sample_snr_db(1, cfg.noise.snr_db_range, rng=rng_snr)
    # Scale the noise, combine with signal, then reference. Referencing is a
    # linear channel-axis operation, so it commutes with signal+noise
    # superposition: the written /eeg equals reference(signal) + k * reference(
    # noise). For 'snr_target' the scale k makes the target SNR hold on the
    # montage selected by cfg.noise.snr_definition; for 'absolute_floor' it fixes
    # the raw-montage noise RMS and the SNR emerges. Both montages' SNRs are
    # recorded (snr_db_raw, snr_db_referenced).
    (eeg, snr_db_raw, snr_db_ref, sig_ref, noise_scaled_ref,
     eeg_normalization_scale_V) = _combine_snr_reference(
        batch.eeg_signal, noise_mixed, snr_db, ref_desc, cfg.noise.snr_definition,
        cfg.noise.noise_model, cfg.noise.absolute_noise_rms_uV,
        cfg.output.eeg_normalization,
    )
    # snr_db is the target SNR under 'snr_target'; under 'absolute_floor' it holds
    # the EMERGENT SNR on the snr_definition montage.
    if cfg.noise.noise_model == 'absolute_floor':
        snr_db_out = snr_db_ref if cfg.noise.snr_definition == 'referenced' else snr_db_raw
    else:
        snr_db_out = snr_db

    # Descriptive concentration of the average-referenced signal topography at
    # peak global field power.  This is not used as, or claimed to be, a source
    # depth label; physical source radius and depth are stored separately.
    concentration = _signal_topographic_concentration(batch.eeg_signal)

    # Per-pool -> global position index for provenance.
    pos_idx_in_pool = batch.extras.get('position_idx_in_pool')
    if pos_idx_in_pool is None:
        pos_idx_global = None
    elif pos_idx_in_pool.ndim == 1:
        pos_idx_global = pool_global_indices[pos_idx_in_pool]
    else:
        pos_idx_global = _gather_pool_indices(pool_global_indices, pos_idx_in_pool)

    source_radius_mm = np.linalg.norm(batch.positions_xyz, axis=-1) * 1000.0
    source_depth_mm = cfg.head.radii_m[0] * 1000.0 - source_radius_mm

    gt = {
        'positions_xyz': batch.positions_xyz.astype(np.float64),
        'orientations_world': batch.orientations_world.astype(np.float64),
        'magnitudes_nAm': batch.magnitudes_nAm.astype(np.float64),
        'population': batch.population.astype(np.int8),
        'source_radius_mm': source_radius_mm.astype(np.float64),
        'source_depth_mm': source_depth_mm.astype(np.float64),
        'snr_db': np.asarray(snr_db_out, dtype=np.float64),
        'snr_db_raw': np.asarray(snr_db_raw, dtype=np.float64),
        'snr_db_referenced': np.asarray(snr_db_ref, dtype=np.float64),
        'signal_topographic_concentration': np.asarray(
            concentration, dtype=np.float64
        ),
        'eeg_normalization_scale_V': np.asarray(
            eeg_normalization_scale_V, dtype=np.float64
        ),
        'band_idx': sig['band'].astype(np.int32),
        'frequencies_hz': sig['frequencies_hz'].astype(np.float64),
        'phases_rad': sig['phases_rad'].astype(np.float64),
        'component_amplitudes': sig['component_amplitudes'].astype(np.float64),
        'n_components': sig['n_components'].astype(np.int32),
        'source_time_course': time_courses.astype(np.float64),
    }
    if pos_idx_global is not None:
        gt['position_pool_idx'] = pos_idx_global.astype(np.int64)
    # Mode-specific extras include the complete padded per-source temporal
    # construction for multi-point samples and compact exact patch support.
    # The common signal metadata above describes the primary source.
    for k, v in batch.extras.items():
        if k == 'position_idx_in_pool':
            continue
        if k == 'patch_active_position_idx_in_pool':
            if not isinstance(v, RaggedArray):
                raise TypeError("Patch support indices must be a RaggedArray")
            gt['patch_active_position_idx'] = RaggedArray(
                pool_global_indices[v.values].astype(np.int64), v.row_splits,
            )
            continue
        gt[k] = v if isinstance(v, RaggedArray) else np.asarray(v)

    if cfg.output.save_components:
        # sig_ref and noise_scaled_ref are the referenced signal and the
        # referenced, SNR-scaled noise. Cast both components first and perform
        # their stored-dtype sum here so signal_only + noise_only reproduces the
        # written /eeg with the same floating-point operation.
        eeg_dtype = np.dtype(cfg.output.dtype)
        signal_stored = sig_ref.astype(eeg_dtype)
        noise_stored = noise_scaled_ref.astype(eeg_dtype)
        eeg = signal_stored + noise_stored
        gt['eeg_signal_only'] = signal_stored
        gt['eeg_noise_only'] = noise_stored

    # The third return value is the RAW signal-only montage (pre-reference,
    # pre-noise); the split runner uses it for a finite/nonzero amplitude
    # diagnostic and for diagnostic plots.
    return eeg, gt, batch.eeg_signal, sig['band_names']


def _run_one_split(
    cfg: Config,
    split_name: str,
    split_id: int,
    n_samples: int,
    output_dir: Path,
    master_seed: int,
    pool_pos_xyz: np.ndarray,
    pool_leadfield: np.ndarray,
    pool_population: np.ndarray,
    pool_global_indices: np.ndarray,
    bg_pool: np.ndarray,
    n_channels: int,
    ref_desc: Dict[str, Any],
    eff_band_probs: Dict[str, float],
    band_info: Dict[str, Any],
    electrodes_block: Dict[str, Any],
) -> Tuple[Path, np.ndarray, Dict[str, Any], Dict[str, Any]]:
    """Generate one split end-to-end and stream-write it to disk.

    Samples are generated one at a time with per-sample seeding (see
    ``_generate_one_sample``) and buffered into write chunks of
    ``cfg.output.chunk_samples`` samples. ``chunk_samples`` therefore controls
    generation buffering and progress granularity; ``hdf5_chunk_rows`` controls
    the canonical HDF5 layout. Neither changes the generated array values.

    Returns
    -------
    path : Path
        Path of the written split file.
    snr_db : ndarray (n_samples,)
        Per-sample SNR applied across the whole split.
    diagnostic_sample : dict
        In-memory snapshot of the fixed leading sample set configured by
        ``output.diagnostic_sample_count`` (positions, orientations,
        magnitudes, population, SNR, and signal-only EEG). Its membership is
        independent of write chunks and it is used only for diagnostic plots.
    val_stats : dict
        Full-split lightweight ground truth (positions_xyz, orientations_world,
        magnitudes_nAm, population, snr_db) plus a per-sample peak amplitude in
        microvolts (peak_uV). Lets Layer-2 validation inspect the ENTIRE split
        instead of a single chunk, without retaining the EEG tensor.
    """
    T = cfg.signal.window_samples
    chunk_size = max(1, int(cfg.output.chunk_samples))
    n_workers = _resolve_n_workers(cfg.output.n_workers)

    # Optional process pool for per-sample generation. Created once per split
    # so workers initialise the shared pools only once. The output is identical
    # to the serial path because each sample is a pure function of its global
    # index (see _generate_one_sample).
    executor = None
    if n_workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        executor = ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_worker_init,
            initargs=(cfg, master_seed, SPLIT_IDS[split_name], pool_pos_xyz,
                      pool_leadfield, pool_population, pool_global_indices,
                      bg_pool, n_channels, T, ref_desc, eff_band_probs),
        )
        logger.info("    Using %d worker processes for generation", n_workers)

    try:
        return _run_one_split_inner(
            cfg, split_name, n_samples, output_dir, master_seed,
            pool_pos_xyz, pool_leadfield, pool_population, pool_global_indices,
            bg_pool, n_channels, T, chunk_size, executor,
            ref_desc, eff_band_probs, band_info, electrodes_block,
        )
    finally:
        if executor is not None:
            executor.shutdown(wait=True)


def _run_one_split_inner(
    cfg: Config,
    split_name: str,
    n_samples: int,
    output_dir: Path,
    master_seed: int,
    pool_pos_xyz: np.ndarray,
    pool_leadfield: np.ndarray,
    pool_population: np.ndarray,
    pool_global_indices: np.ndarray,
    bg_pool: np.ndarray,
    n_channels: int,
    T: int,
    chunk_size: int,
    executor,
    ref_desc: Dict[str, Any],
    eff_band_probs: Dict[str, float],
    band_info: Dict[str, Any],
    electrodes_block: Dict[str, Any],
) -> Tuple[Path, np.ndarray, Dict[str, Any], Dict[str, Any]]:
    """Body of _run_one_split, optionally using a process pool ``executor``."""
    writer = open_split_writer(
        output_dir=output_dir,
        split_name=split_name,
        fmt=cfg.output.format,
        compress=cfg.output.compress,
        source_mode=cfg.source_mode,
        chunk_rows=cfg.output.hdf5_chunk_rows,
        electrodes=electrodes_block,
        eeg_dtype=cfg.output.dtype,
    )
    all_snr_db = np.empty(n_samples, dtype=np.float64)
    peak_uV = np.empty(n_samples, dtype=np.float64)
    diagnostic_n = min(cfg.output.diagnostic_sample_count, n_samples)
    diagnostic_gt: Dict[str, list] = {
        'positions_xyz': [], 'orientations_world': [],
        'magnitudes_nAm': [], 'population': [], 'snr_db': [],
    }
    diagnostic_signal: list = []
    band_names: Optional[Any] = None
    # Multi-point min-separation auditing: count samples in which the
    # min-separation constraint could not be satisfied, aggregated over the
    # whole split (the per-sample flag is also written to ground truth).
    sep_violated_count = 0

    # Full-split lightweight ground-truth accumulators (never the EEG tensor).
    gt_accum: Dict[str, list] = {
        'positions_xyz': [], 'orientations_world': [],
        'magnitudes_nAm': [], 'population': [],
    }

    n_chunks = (n_samples + chunk_size - 1) // chunk_size
    for chunk_idx in range(n_chunks):
        cs = chunk_idx * chunk_size
        ce = min(cs + chunk_size, n_samples)
        n_chunk = ce - cs
        logger.info(
            "    Chunk %d/%d  (%d samples)  [%s]",
            chunk_idx + 1, n_chunks, n_chunk, split_name,
        )

        # Generate each sample independently (per-sample seeding) and buffer
        # the chunk before writing. With a process pool, samples in the chunk
        # are generated in parallel; results are gathered in global-index
        # order so the written data is identical to the serial path.
        # The finite/nonzero amplitude diagnostic is computed from the
        # signal-only montage, not the signal+noise EEG, so it checks the
        # forward projection rather than the selected noise convention.
        # Signal-only tensors are retained only for the fixed leading
        # diagnostic set, whose membership is independent of chunks.
        eeg_list: list = []
        sig_only_list: list = []
        peak_list: list = []
        gt_lists: Dict[str, list] = {}

        indices = range(cs, ce)
        if executor is None:
            results_iter = (
                _generate_one_sample(
                    cfg, master_seed, SPLIT_IDS[split_name], gi,
                    pool_pos_xyz, pool_leadfield, pool_population,
                    pool_global_indices, bg_pool, n_channels, T,
                    ref_desc, eff_band_probs,
                )
                for gi in indices
            )
        else:
            map_chunk = max(1, n_chunk // (executor._max_workers * 4))
            results_iter = executor.map(_worker_generate, indices, chunksize=map_chunk)

        for local_idx, (eeg_s, gt_s, sig_only_s, bnames) in enumerate(results_iter):
            if band_names is None:
                band_names = bnames
            eeg_list.append(eeg_s)
            peak_list.append(float(np.max(np.abs(sig_only_s))))
            if cs + local_idx < diagnostic_n:
                sig_only_list.append(sig_only_s)
            for k, v in gt_s.items():
                gt_lists.setdefault(k, []).append(v)

        eeg = np.concatenate(eeg_list, axis=0)
        gt: Dict[str, Any] = {}
        for key, parts in gt_lists.items():
            if parts and isinstance(parts[0], RaggedArray):
                gt[key] = concatenate_ragged(parts)
            else:
                gt[key] = np.concatenate(parts, axis=0)
        snr_db = gt['snr_db']
        all_snr_db[cs:ce] = snr_db
        peak_uV[cs:ce] = np.asarray(peak_list, dtype=np.float64) * 1e6

        writer.append(eeg.astype(np.dtype(cfg.output.dtype), copy=False), gt)

        if 'separation_violated' in gt:
            sep_violated_count += int(np.asarray(gt['separation_violated']).sum())

        for k in gt_accum:
            gt_accum[k].append(gt[k])

        if cs < diagnostic_n:
            take = min(ce, diagnostic_n) - cs
            for key in diagnostic_gt:
                diagnostic_gt[key].append(gt[key][:take].copy())
            diagnostic_signal.append(
                np.concatenate(sig_only_list, axis=0)[:take].copy()
            )

    metadata = {
        'split_name': split_name,
        'n_samples': n_samples,
        'n_channels': n_channels,
        'window_samples': T,
        'sampling_freq_hz': cfg.signal.sampling_freq_hz,
        'source_mode': cfg.source_mode,
        # The forward operator that produced this split, embedded so a split
        # file remains self-describing when it is separated from the canonical
        # directory and its metadata.json sidecar. Downstream physics-informed
        # consumers must reproduce the generating forward model exactly; these
        # are the only three parameters that determine it.
        'radii_m': [float(r) for r in cfg.head.radii_m],
        'conductivities_S_per_m': [
            float(s) for s in cfg.head.conductivities_S_per_m
        ],
        'n_legendre': int(cfg.head.n_legendre),
        # Unit-standard-deviation courses are unit RMS only when centered.
        # Otherwise magnitudes_nAm is the multiplier of the stored waveform.
        'magnitude_convention': (
            'rms' if cfg.signal.zero_mean_signal else 'waveform_scale'
        ),
        'position_units': 'm',
        'electrode_position_units': 'm',
        'magnitude_units': 'nAm',
        'band_names': band_names if band_names is not None else [],
        'reference_mode': cfg.output.reference_mode,
        'reference_description': ref_desc.get('description', ''),
        'output_dtype': cfg.output.dtype,
        'eeg_normalization': cfg.output.eeg_normalization,
        'eeg_units': (
            'dimensionless_rms_normalized'
            if cfg.output.eeg_normalization == 'per_sample_rms' else 'volts'
        ),
        'physical_voltage_reconstruction': (
            'eeg_V = eeg * ground_truth/eeg_normalization_scale_V'
        ),
        'zero_mean_signal': bool(cfg.signal.zero_mean_signal),
        'noise_model': cfg.noise.noise_model,
        'absolute_noise_rms_uV': float(cfg.noise.absolute_noise_rms_uV),
        'band_resolvability': band_info,
    }
    path = writer.finalize(metadata)
    logger.info("    Wrote %s (%.1f MB)", path.name, path.stat().st_size / 1e6)

    val_stats: Dict[str, Any] = {
        'positions_xyz': np.concatenate(gt_accum['positions_xyz'], axis=0),
        'orientations_world': np.concatenate(gt_accum['orientations_world'], axis=0),
        'magnitudes_nAm': np.concatenate(gt_accum['magnitudes_nAm'], axis=0),
        'population': np.concatenate(gt_accum['population'], axis=0),
        'snr_db': all_snr_db,
        'peak_uV': peak_uV,
    }
    if cfg.source_mode == 'multi_point':
        rate = sep_violated_count / max(n_samples, 1)
        val_stats['separation_violated_count'] = int(sep_violated_count)
        val_stats['separation_violated_rate'] = float(rate)
        if sep_violated_count > 0:
            logger.warning(
                "    Min-separation constraint violated in %d/%d samples "
                "(%.1f%%) for split %r. Consider lowering "
                "multi_point.min_separation_mm, reducing n_sources_max, or "
                "enlarging the position pool.",
                sep_violated_count, n_samples, 100.0 * rate, split_name,
            )
    diagnostic_sample = {
        key: np.concatenate(parts, axis=0)
        for key, parts in diagnostic_gt.items()
    }
    diagnostic_sample['eeg_signal'] = np.concatenate(diagnostic_signal, axis=0)
    return path, all_snr_db, diagnostic_sample, val_stats


def _gather_pool_indices(pool_global: np.ndarray, pool_local_idx: np.ndarray) -> np.ndarray:
    """Map per-pool indices to global indices, preserving -1 padding."""
    out = np.full_like(pool_local_idx, fill_value=-1, dtype=np.int64)
    valid = pool_local_idx >= 0
    out[valid] = pool_global[pool_local_idx[valid]]
    return out


def _read_serialized_position_indices(path: Path, source_mode: str) -> np.ndarray:
    """Read the global source-pool indices actually serialized in one split."""
    path = Path(path)
    keys = ['position_pool_idx']
    if source_mode == 'patch':
        keys.append('patch_active_position_idx')

    arrays = []
    if path.suffix == '.h5':
        import h5py
        with h5py.File(path, 'r') as f:
            for key in keys:
                node = f[f'ground_truth/{key}']
                if isinstance(node, h5py.Group):
                    arrays.append(np.asarray(node['values'][:]))
                else:
                    arrays.append(np.asarray(node[:]))
    elif path.suffix == '.npz':
        with np.load(path, allow_pickle=False) as payload:
            for key in keys:
                dense_key = f'gt_{key}'
                ragged_key = f'gt_{key}_values'
                arrays.append(np.asarray(
                    payload[ragged_key] if ragged_key in payload else payload[dense_key]
                ))
    elif path.suffix == '.mat':
        try:
            hdf5storage = import_hdf5storage()
        except ImportError as exc:  # pragma: no cover - writer already needs it
            raise ImportError(
                "Reading MATLAB validation data requires hdf5storage"
            ) from exc
        ground_truth = hdf5storage.loadmat(path)['ground_truth']
        if isinstance(ground_truth, dict):
            for key in keys:
                value = ground_truth[key]
                if isinstance(value, dict):
                    value = value['values']
                elif isinstance(value, np.ndarray) and value.dtype.names:
                    value = value.reshape(-1)[0]['values']
                arrays.append(np.asarray(value))
        elif isinstance(ground_truth, np.ndarray) and ground_truth.dtype.names:
            record = ground_truth.reshape(-1)[0]
            for key in keys:
                value = record[key]
                if isinstance(value, np.ndarray) and value.dtype.names:
                    value = value.reshape(-1)[0]['values']
                arrays.append(np.asarray(value))
        else:  # pragma: no cover - incompatible third-party reader behavior
            raise ValueError(
                f"Could not decode ground_truth structure from {path}"
            )
    else:  # pragma: no cover - pipeline controls output suffixes
        raise ValueError(f"Unsupported split-file suffix: {path.suffix!r}")

    flattened = [array.astype(np.int64, copy=False).ravel() for array in arrays]
    return np.concatenate(flattened)


def _signal_topographic_concentration(sig_only: np.ndarray) -> np.ndarray:
    """Peak-to-RMS concentration at peak average-referenced GFP.

    Parameters
    ----------
    sig_only : ndarray (B, N, T)
        Signal-only scalp potentials (no noise, pre-reference).

    Returns
    -------
    concentration : ndarray (B,)
        For each sample, ``max_ch |v| / sqrt(mean_ch v**2)`` where ``v`` is the
        average-referenced spatial map at the time of peak global field power.
        Dimensionless and scale-invariant.  It describes the sensor map; it is
        not an anatomical depth estimator.  Zero for an all-zero signal.
    """
    sig_only = np.asarray(sig_only, dtype=np.float64)
    B, N, T = sig_only.shape
    sig_ar = sig_only - sig_only.mean(axis=1, keepdims=True)
    gfp = np.sqrt(np.mean(sig_ar ** 2, axis=1))           # (B, T)
    t_star = np.argmax(gfp, axis=1)                       # (B,)
    concentration = np.zeros(B, dtype=np.float64)
    for b in range(B):
        v = sig_ar[b, :, t_star[b]]                       # (N,)
        rms = np.sqrt(np.mean(v ** 2))
        if rms > 0:
            concentration[b] = float(np.max(np.abs(v)) / rms)
    return concentration


def _combine_snr_reference(
    sig_raw, noise_raw, snr_db, ref_desc, snr_definition,
    noise_model, absolute_noise_rms_uV, eeg_normalization='none',
):
    """Scale the mixed noise, combine it with the signal, then apply the reference.

    Referencing is linear on the channel axis, so reference(signal + k*noise)
    == reference(signal) + k*reference(noise); a single per-sample scale ``k``
    is computed and applied to the referenced noise. How ``k`` is chosen depends
    on ``noise_model``:

      'absolute_floor' : ``k`` fixes the mixed-noise RMS at
                         ``absolute_noise_rms_uV`` (uV) on the RAW sensor
                         montage, independent of the signal. The SNR is not
                         targeted; it emerges from the (fixed) signal power and
                         this fixed noise power. ``snr_db`` is ignored.
      'snr_target'     : ``k`` makes each sample hit the target ``snr_db`` on the
                         montage named by ``snr_definition`` ('referenced' -> the
                         final /eeg; 'raw' -> the pre-reference potentials).

    Either way BOTH montages' achieved/emergent SNRs are computed and returned.
    When the reference is 'none' the two montages coincide.

    Returns
    -------
    eeg : ndarray (B, N, T)
        Referenced signal + referenced, scaled noise.
    snr_db_raw, snr_db_referenced : ndarray (B,)
        Achieved (snr_target) or emergent (absolute_floor) SNR (dB) on each
        montage.
    sig_ref : ndarray (B, N, T)
        Referenced signal-only montage.
    noise_scaled_ref : ndarray (B, N, T)
        Referenced, scaled noise montage (so sig_ref + noise_scaled_ref == eeg
        exactly).
    eeg_normalization_scale_V : ndarray (B,)
        Physical volts represented by one stored EEG unit.  It is one for
        physical-voltage output and the pre-normalization total EEG RMS when
        ``eeg_normalization='per_sample_rms'``.
    """
    sig_raw = np.asarray(sig_raw, dtype=np.float64)
    noise_raw = np.asarray(noise_raw, dtype=np.float64)
    if (
        sig_raw.shape != noise_raw.shape
        or sig_raw.ndim != 3
        or not np.isfinite(sig_raw).all()
        or not np.isfinite(noise_raw).all()
    ):
        raise ValueError(
            "signal and noise must be finite arrays with identical "
            "[samples, channels, time] shapes"
        )
    sig_ref = apply_reference(sig_raw, ref_desc)
    noise_ref = apply_reference(noise_raw, ref_desc)

    # Per-sample signal/noise power over (channel, time) on each montage.
    sp_raw = (sig_raw ** 2).mean(axis=(-1, -2))
    np_raw = (noise_raw ** 2).mean(axis=(-1, -2))
    sp_ref = (sig_ref ** 2).mean(axis=(-1, -2))
    np_ref = (noise_ref ** 2).mean(axis=(-1, -2))

    if noise_model == 'absolute_floor':
        # Fix the mixed-noise RMS at a physical level on the RAW sensor montage,
        # independent of the signal. sqrt(np_raw) is the raw-montage noise RMS,
        # so this scale yields exactly absolute_noise_rms_uV; the SNR emerges.
        if (np_raw <= 0).any():
            raise ValueError("cannot impose an absolute noise RMS on zero-RMS noise")
        target_rms = float(absolute_noise_rms_uV) * 1e-6              # uV -> V
        scale = target_rms / np.sqrt(np_raw)                           # (B,)
    else:  # 'snr_target'
        snr_values = np.asarray(snr_db, dtype=np.float64)
        if snr_values.shape != (sig_raw.shape[0],) or not np.isfinite(snr_values).all():
            raise ValueError("snr_db must contain one finite target per sample")
        snr_lin = 10.0 ** (snr_values / 10.0)                            # (B,)
        if snr_definition == 'raw':
            selected_signal_power, selected_noise_power = sp_raw, np_raw
        else:  # 'referenced' (default)
            selected_signal_power, selected_noise_power = sp_ref, np_ref
        if (selected_signal_power <= 0).any():
            raise ValueError(
                "cannot target SNR for a sample whose clean signal has zero RMS "
                f"under the {snr_definition!r} SNR definition"
            )
        if (selected_noise_power <= 0).any():
            raise ValueError(
                "cannot target SNR for a sample whose noise has zero RMS "
                f"under the {snr_definition!r} SNR definition"
            )
        scale = np.sqrt(selected_signal_power / (snr_lin * selected_noise_power))
    if not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("noise scaling overflowed or underflowed for the requested model")

    noise_scaled_ref = noise_ref * scale[:, None, None]
    eeg = sig_ref + noise_scaled_ref

    s2 = scale ** 2
    ach_raw = sp_raw / np.maximum(s2 * np_raw, 1e-300)
    ach_ref = sp_ref / np.maximum(s2 * np_ref, 1e-300)
    snr_db_raw = 10.0 * np.log10(np.maximum(ach_raw, 1e-300))
    snr_db_referenced = 10.0 * np.log10(np.maximum(ach_ref, 1e-300))
    if eeg_normalization == 'per_sample_rms':
        # Normalization is an explicit output transform rather than a side
        # effect of target-SNR generation. It preserves the signal/noise power
        # ratio, and the stored divisor restores physical volts exactly.
        eeg_normalization_scale_V = np.sqrt(np.mean(eeg ** 2, axis=(-1, -2)))
        eeg_normalization_scale_V = np.maximum(eeg_normalization_scale_V, 1e-300)
        denom = eeg_normalization_scale_V[:, None, None]
        eeg = eeg / denom
        sig_ref = sig_ref / denom
        noise_scaled_ref = noise_scaled_ref / denom
    elif eeg_normalization == 'none':
        eeg_normalization_scale_V = np.ones(eeg.shape[0], dtype=np.float64)
    else:
        raise ValueError(
            "eeg_normalization must be 'none' or 'per_sample_rms'"
        )
    return (
        eeg, snr_db_raw, snr_db_referenced, sig_ref, noise_scaled_ref,
        eeg_normalization_scale_V,
    )


def _make_progress_logger(label: str, interval: float = 1.0):
    state = {'last_t': 0.0}

    def cb(current: int, total: int):
        now = time.time()
        if now - state['last_t'] >= interval or current == total:
            pct = 100.0 * current / total
            logger.info("    %s: %d / %d (%.1f%%)", label, current, total, pct)
            state['last_t'] = now

    return cb


def _setup_logging(level: int):
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(h)
    logger.setLevel(level)


def _make_diagnostic_plots(out_dir, positions_xyz, population, snr_db, magnitudes_nAm, eeg_signal):
    """Generate sanity-check figures."""
    out_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Figure 1: position scatter
    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    is_cort = (population == 0)
    ax1.scatter(*positions_xyz[is_cort].T * 1000, s=1, alpha=0.3, label='cortical', c='C0')
    ax1.scatter(*positions_xyz[~is_cort].T * 1000, s=1, alpha=0.5, label='subcortical', c='C3')
    ax1.set_xlabel('x (mm)'); ax1.set_ylabel('y (mm)'); ax1.set_zlabel('z (mm)')
    ax1.set_title('Position pool'); ax1.legend()

    ax2 = fig.add_subplot(1, 2, 2)
    radii_mm = np.linalg.norm(positions_xyz, axis=1) * 1000
    ax2.hist(radii_mm[is_cort], bins=40, alpha=0.6, label='cortical', color='C0')
    ax2.hist(radii_mm[~is_cort], bins=40, alpha=0.6, label='subcortical', color='C3')
    ax2.set_xlabel('Radius (mm)'); ax2.set_ylabel('count'); ax2.legend()
    ax2.set_title('Radial depth distribution')
    plt.tight_layout()
    plt.savefig(out_dir / 'positions.png', dpi=120)
    plt.close()

    # Figure 2: SNR + magnitude histograms
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(snr_db, bins=30)
    axes[0].set_xlabel('SNR (dB)'); axes[0].set_ylabel('count')
    axes[0].set_title('SNR distribution (sample)')
    flat_mags = magnitudes_nAm[~np.isnan(magnitudes_nAm)] if magnitudes_nAm.ndim > 1 else magnitudes_nAm
    axes[1].hist(np.log10(flat_mags + 1e-12), bins=30)
    axes[1].set_xlabel('log10(magnitude / nA*m)'); axes[1].set_ylabel('count')
    axes[1].set_title('Dipole magnitude distribution (sample)')
    plt.tight_layout()
    plt.savefig(out_dir / 'snr_magnitude.png', dpi=120)
    plt.close()

    # Figure 3: EEG sample traces
    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    n_show = min(5, eeg_signal.shape[0])
    for i in range(n_show):
        ax = axes[i % 3]
        for ch in range(min(4, eeg_signal.shape[1])):
            ax.plot(eeg_signal[i, ch] * 1e6, alpha=0.6)
    for ax in axes:
        ax.set_ylabel('uV')
    axes[-1].set_xlabel('time sample')
    axes[0].set_title('Example EEG signals (signal-only, before noise)')
    plt.tight_layout()
    plt.savefig(out_dir / 'sample_eeg.png', dpi=120)
    plt.close()

    logger.info("      Diagnostic plots written to %s", out_dir)
