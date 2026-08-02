"""
High-SNR dataset for debugging and controlled ceiling studies.

Useful for:
  - Testing whether model error decreases when additive noise is small
  - Separating noise sensitivity from other modeling and optimization errors

This example uses the 'snr_target' noise model so that a very high target SNR
can be requested directly. (Under the default 'absolute_floor' model the noise
level is fixed and SNR is emergent, so snr_db_range would be ignored -- hence
noise_model is set explicitly here.) Noise is kept on rather than removed, so
the full pipeline is still exercised end-to-end.

Run from the repository root:
    python examples/high_snr_clean.py

Generates 50,000 train; output is a few GB.
"""

from spherical_eeg_sim import Config, run_pipeline


def main():
    cfg = Config()
    cfg.noise.noise_model = 'snr_target'    # request an explicit SNR per sample
    cfg.noise.snr_db_range = [40.0, 50.0]

    cfg.dataset.n_train = 50_000
    cfg.dataset.n_val = 5_000
    cfg.dataset.n_test = 5_000

    cfg.electrode_file = 'standard_1010_69.csv'
    cfg.output_dir = 'output_high_snr'

    summary = run_pipeline(cfg)
    print()
    print('Output:', summary['output_dir'])


if __name__ == '__main__':
    main()
