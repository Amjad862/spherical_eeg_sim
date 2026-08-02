"""
The two noise models side by side.

spherical_eeg_sim supports two noise conventions:

  'absolute_floor' (default)
      Noise is fixed at a physical RMS level (absolute_noise_rms_uV, default
      1.00 uV). The documented 3.16 uV reference calibration is available as
      an explicit setting. SNR is therefore EMERGENT: it
      depends on the source amplitude and depth rather than being imposed.
      Absolute voltage scale is preserved, so magnitude-dependent gain remains
      in the input. Source depth is stored explicitly and should be evaluated
      while controlling for magnitude, orientation, and position.

  'snr_target'
      Noise is scaled per sample to hit a drawn target SNR (snr_db_range).
      Every sample lands in a controlled SNR band while physical volts are
      preserved. Output normalization is an independent choice: set
      ``cfg.output.eeg_normalization = 'per_sample_rms'`` when absolute input
      scale should be removed, with an exact stored scale for reconstruction.

This script writes one dataset in each mode so you can compare their ground
truth (see the emergent vs targeted SNR distributions in the manifests).

Run from the repository root:
    python examples/snr_target_vs_absolute_floor.py

Generates 5,000 train per convention; output is well under a GB.
"""

from spherical_eeg_sim import Config, run_pipeline


def base_cfg():
    cfg = Config()
    cfg.dataset.n_train = 5_000
    cfg.dataset.n_val = 1_000
    cfg.dataset.n_test = 1_000
    cfg.electrode_file = 'standard_1010_69.csv'
    return cfg


def main():
    # Fixed physical noise floor (emergent SNR).
    cfg = base_cfg()
    cfg.noise.noise_model = 'absolute_floor'
    cfg.noise.absolute_noise_rms_uV = 1.00
    cfg.output_dir = 'output_absolute_floor'
    run_pipeline(cfg)

    # Controlled target-SNR band.
    cfg = base_cfg()
    cfg.noise.noise_model = 'snr_target'
    cfg.noise.snr_db_range = [0.0, 20.0]
    cfg.output_dir = 'output_snr_target'
    run_pipeline(cfg)

    print('Compare snr_db distributions in the two validation_report.json files.')


if __name__ == '__main__':
    main()
