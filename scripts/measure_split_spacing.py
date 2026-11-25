#!/usr/bin/env python
"""Measure the spatial spacing implied by a position-level split.

Position-level splitting guarantees that no exact source location is shared
between splits. It does not, and cannot, guarantee spatial separation: with a
dense pool a held-out position still has a training position a short distance
away. That distance is what determines whether held-out evaluation measures
generalization to unseen coordinates inside a densely sampled shell or
extrapolation to an unsampled region, so it is measured rather than asserted.

Two configurations are reported:

``default``
    The shipped defaults, which set the figure quoted for the package as a
    whole.

``baseline``
    The pool used by ``scripts/train_baseline.py``, which is coarser than the
    default. Its spacing is the one that applies to the supervised-usability
    numbers in ``validation_results/baseline_benchmark_summary.json``.

Both use the pipeline's own seed derivation, so the pools are the ones a run at
that configuration actually produces.

Usage::

    python scripts/measure_split_spacing.py
    python scripts/measure_split_spacing.py --write-canonical

Ordinary runs save the summary in a new validation_runs/ directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._validation_outputs import ValidationOutputs, add_output_options
from spherical_eeg_sim.config import Config           # noqa: E402
from spherical_eeg_sim.provenance import environment_snapshot  # noqa: E402
from spherical_eeg_sim.sources.sampling import (      # noqa: E402
    make_position_split,
    sample_cortical_positions,
    sample_subcortical_positions,
)

def build_pool(cfg: Config):
    """Rebuild a run's position pool and split using the pipeline's seeding."""
    position_seed = (
        cfg.master_seed if cfg.position_seed is None else cfg.position_seed
    )
    master_rng = np.random.default_rng(position_seed)
    rng_pos = np.random.default_rng(master_rng.integers(0, 2 ** 31))
    rng_split = np.random.default_rng(master_rng.integers(0, 2 ** 31))

    cortical = sample_cortical_positions(
        cfg.sources.n_cortical_positions,
        cfg.sources.cortical_radius_range_mm,
        cfg.sources.cortical_sampling,
        rng_pos,
    )
    subcortical = sample_subcortical_positions(
        cfg.sources.n_subcortical_positions,
        cfg.sources.subcortical_radius_range_mm,
        cfg.sources.subcortical_sampling,
        rng_pos,
    )
    positions = np.concatenate([cortical, subcortical], axis=0)
    population = np.concatenate([
        np.zeros(cortical.shape[0], dtype=np.int8),
        np.ones(subcortical.shape[0], dtype=np.int8),
    ])
    targets = (
        (1, 1, 1) if cfg.source_mode == 'patch'
        else (cfg.dataset.n_train, cfg.dataset.n_val, cfg.dataset.n_test)
    )
    split = make_position_split(
        positions, *targets,
        cfg.split.stratify_by_depth, cfg.split.depth_bins,
        rng_split, population=population,
    ).split
    return positions, population, split, cortical


def _summary(distances_mm: np.ndarray) -> dict:
    return {
        'median_mm': float(np.median(distances_mm)),
        'mean_mm': float(np.mean(distances_mm)),
        'p90_mm': float(np.percentile(distances_mm, 90)),
        'min_mm': float(np.min(distances_mm)),
        'n': int(distances_mm.size),
    }


def measure(cfg: Config) -> dict:
    """Nearest-neighbour spacing within the pool and across the split."""
    positions, population, split, cortical = build_pool(cfg)

    within, _ = cKDTree(cortical).query(cortical, k=2)
    within_mm = within[:, 1] * 1e3

    result = {
        'n_cortical_positions': int(cfg.sources.n_cortical_positions),
        'n_subcortical_positions': int(cfg.sources.n_subcortical_positions),
        'cortical_radius_range_mm': list(cfg.sources.cortical_radius_range_mm),
        'master_seed': int(cfg.master_seed),
        'source_mode': cfg.source_mode,
        'within_pool_cortical_nearest_neighbour': _summary(within_mm),
    }

    for label, mask in (
        ('cortical', population == 0),
        ('all_positions', np.ones_like(population, dtype=bool)),
    ):
        train = positions[(split == 0) & mask]
        test = positions[(split == 2) & mask]
        if train.size == 0 or test.size == 0:
            continue
        distances, _ = cKDTree(train).query(test, k=1)
        result[f'test_to_nearest_train_{label}'] = _summary(distances * 1e3)
    return result


def default_config() -> Config:
    """The shipped defaults."""
    return Config()


def baseline_config() -> Config:
    """The pool used by ``scripts/train_baseline.py`` at its own defaults."""
    cfg = Config()
    cfg.source_mode = 'point'
    cfg.dataset.n_train = 20_000
    cfg.dataset.n_val = 2_000
    cfg.dataset.n_test = 3_000
    cfg.sources.n_cortical_positions = 8_000
    cfg.sources.n_subcortical_positions = 1_500
    cfg.master_seed = 862
    return cfg


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('--json', default=None,
                    help='path for the deterministic summary')
    add_output_options(ap)
    args = ap.parse_args(argv)
    outputs = ValidationOutputs(
        ap, args, name='split-spacing', canonical={'json': 'split_spacing.json'},
        explicit={'json': args.json} if args.json is not None else None,
    )

    payload = {
        'software_environment': environment_snapshot(),
        'default': measure(default_config()),
        'baseline': measure(baseline_config()),
    }

    for name in ('default', 'baseline'):
        block = payload[name]
        within = block['within_pool_cortical_nearest_neighbour']
        across = block['test_to_nearest_train_cortical']
        print(f"{name} (n_cortical={block['n_cortical_positions']}, "
              f"seed={block['master_seed']})")
        print(f"  within-pool cortical nearest neighbour : "
              f"median {within['median_mm']:.2f} mm")
        print(f"  test -> nearest training position      : "
              f"median {across['median_mm']:.2f} mm, "
              f"mean {across['mean_mm']:.2f} mm, "
              f"90th percentile {across['p90_mm']:.2f} mm "
              f"(n = {across['n']})")
        print()

    with outputs.stage() as paths:
        with paths['json'].open('w', newline='\n') as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write('\n')
    print(f"Wrote {outputs.paths['json']}")
    return payload


if __name__ == '__main__':
    main()
