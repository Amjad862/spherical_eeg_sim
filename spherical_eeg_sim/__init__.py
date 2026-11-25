"""
SphericalEEGSim
================

A configurable, well-documented EEG source localization dataset generator.

Generates simulated scalp-EEG datasets with full ground truth (position,
orientation, magnitude, SNR, frequency band) for training and evaluating
EEG source localization algorithms. Supports three source modes:

  - ``point``        single dipole per sample
  - ``multi_point``  1..N dipoles per sample with optional time coupling
  - ``patch``        spatially-extended cortical patch with finite support

The generator accepts arbitrary electrode montages via CSV, samples
continuous orientations on the unit sphere, draws magnitudes log-uniformly
within configurable literature-guided ranges, and separates exact source
locations between splits by default. A region policy holds out buffered
spatial sectors in every source mode and, for patches, keeps each patch
footprint inside one split; point and multi-point modes also support
deliberately shared source positions.

Quick start
-----------

    from spherical_eeg_sim import Config, run_pipeline
    cfg = Config()                      # use all defaults
    cfg.dataset.n_train = 50_000        # override individual fields
    cfg.output_dir = 'my_dataset'
    run_pipeline(cfg)

For configuration documentation see :mod:`spherical_eeg_sim.config` and
``docs/parameters.md`` in the source repository.
"""

from .config import Config, quickstart_config
from .pipeline import run_pipeline
from .sources.partition import plan_position_split

__version__ = '1.1.0'

__all__ = [
    'Config',
    'quickstart_config',
    'run_pipeline',
    'plan_position_split',
    '__version__',
]
