"""
Command-line interface for the EEG source dataset generator.

Installing the package provides the ``spherical-eeg-sim`` command; the module
form ``python -m spherical_eeg_sim.cli`` accepts the same arguments.

Usage:

    # Use defaults
    spherical-eeg-sim

    # Load a config from JSON
    spherical-eeg-sim --config my_config.json

    # Override a few fields
    spherical-eeg-sim --output-dir my_run --n-train 50000

    # Quickstart preset
    spherical-eeg-sim --quickstart

The command exits with status zero on success. Use the Python API when the
run summary returned by :func:`spherical_eeg_sim.run_pipeline` is needed.
"""

from __future__ import annotations
import argparse

from .config import Config, quickstart_config
from .pipeline import run_pipeline


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog='spherical-eeg-sim',
        description='Configurable EEG source localization dataset generator.',
    )
    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to a JSON configuration file. If omitted, uses defaults.',
    )
    parser.add_argument(
        '--quickstart', action='store_true',
        help='Use the reduced 800/100/100-sample smoke-test preset.',
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
    run_pipeline(cfg)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
