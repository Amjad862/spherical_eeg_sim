"""
Command-line interface for the EEG source dataset generator.

Installing the package provides the ``spherical-eeg-sim`` command; the module
form ``python -m spherical_eeg_sim.cli`` accepts the same arguments.

Usage:

    # Small smoke-test dataset (800/100/100 windows)
    spherical-eeg-sim --quickstart

    # Load a config from JSON
    spherical-eeg-sim --config my_config.json

    # Default configuration with chosen sizes
    spherical-eeg-sim --output-dir my_run --n-train 50000 --n-val 5000 --n-test 5000

    # The full default dataset (360,000 windows, about 18.5 GiB of EEG)
    spherical-eeg-sim --default-run

A run starts only when its size has been chosen: by ``--quickstart``, by
``--config``, by one of the size overrides, or by ``--default-run`` for the
full default dataset. Called without any of these, the command prints the
size of the default dataset and stops.

The command exits with status zero on success. Use the Python API when the
run summary returned by :func:`spherical_eeg_sim.run_pipeline` is needed.
"""

from __future__ import annotations
import argparse

import numpy as np

from .config import Config, quickstart_config
from .core.electrodes import load_electrodes
from .pipeline import run_pipeline


def planned_size(cfg: Config) -> str:
    """One-line description of the dataset a configuration will generate."""
    windows = cfg.dataset.n_train + cfg.dataset.n_val + cfg.dataset.n_test
    channels = load_electrodes(
        cfg.electrode_file, target_radius_m=cfg.head.radii_m[-1],
    ).n_electrodes
    eeg_bytes = (
        windows * channels * cfg.signal.window_samples * np.dtype(cfg.output.dtype).itemsize
    )
    return (
        f"{windows:,} windows ({cfg.dataset.n_train:,} train, {cfg.dataset.n_val:,} validation, "
        f"{cfg.dataset.n_test:,} test) x {channels} channels x {cfg.signal.window_samples} samples, "
        f"about {eeg_bytes / 2**30:.1f} GiB of {cfg.output.dtype} EEG before compression"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog='spherical-eeg-sim',
        description='Configurable EEG source localization dataset generator.',
    )
    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to a JSON configuration file.',
    )
    parser.add_argument(
        '--quickstart', action='store_true',
        help='Use the reduced 800/100/100-sample smoke-test preset.',
    )
    parser.add_argument(
        '--default-run', action='store_true',
        help='Generate the full default dataset (300,000/30,000/30,000 windows). '
             'Required to start a run whose size was not chosen by --config, '
             '--quickstart, --n-train, --n-val or --n-test.',
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Override config.output_dir.',
    )
    parser.add_argument(
        '--electrode-file', type=str, default=None,
        help='Override config.electrode_file.',
    )
    parser.add_argument(
        '--n-train', type=int, default=None,
        help='Override config.dataset.n_train.',
    )
    parser.add_argument(
        '--n-val', type=int, default=None,
        help='Override config.dataset.n_val.',
    )
    parser.add_argument(
        '--n-test', type=int, default=None,
        help='Override config.dataset.n_test.',
    )
    parser.add_argument(
        '--source-mode', type=str, default=None,
        choices=['point', 'multi_point', 'patch'],
        help='Override config.source_mode.',
    )
    parser.add_argument(
        '--master-seed', type=int, default=None,
        help='Override config.master_seed.',
    )
    parser.add_argument(
        '--save-config', type=str, default=None,
        help='After applying overrides, save the config to this JSON path '
             '(useful for reproducing later).',
    )
    args = parser.parse_args(argv)

    if args.quickstart and args.config:
        parser.error("Use --quickstart OR --config, not both.")
    if args.default_run and (args.quickstart or args.config):
        parser.error("--default-run selects the built-in defaults; do not combine it "
                     "with --quickstart or --config.")

    if args.quickstart:
        cfg = quickstart_config()
    elif args.config:
        cfg = Config.from_json(args.config)
    else:
        cfg = Config()

    # Apply overrides
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.electrode_file is not None:
        cfg.electrode_file = args.electrode_file
    if args.n_train is not None:
        cfg.dataset.n_train = args.n_train
    if args.n_val is not None:
        cfg.dataset.n_val = args.n_val
    if args.n_test is not None:
        cfg.dataset.n_test = args.n_test
    if args.source_mode is not None:
        cfg.source_mode = args.source_mode
    if args.master_seed is not None:
        cfg.master_seed = args.master_seed

    if args.save_config:
        cfg.to_json(args.save_config)
        print(f"Saved configuration to {args.save_config}")

    cfg.validate()
    size_chosen = (
        args.quickstart or args.config is not None or args.default_run
        or any(value is not None for value in (args.n_train, args.n_val, args.n_test))
    )
    if not size_chosen:
        # The built-in defaults describe a large dataset. Starting it must be a
        # decision, not the result of calling the command to see what it does.
        print(f"No run was started. The default configuration generates {planned_size(cfg)}.")
        print("Choose one of: --quickstart (small smoke test), --config FILE, "
              "--n-train/--n-val/--n-test, or --default-run for the full default dataset.")
        print("Run with --help for all options.")
        return 0 if args.save_config else 2

    print(f"Planned dataset: {planned_size(cfg)}")
    run_pipeline(cfg)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
