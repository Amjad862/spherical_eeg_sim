"""
Cortex-only dataset: subcortical_fraction = 0.

Useful for a declared cortex-constrained benchmark. Direct comparison with a
published generator still requires matching that study's complete source,
head, signal, noise, montage, and preprocessing model.

Generates 100,000 train; output is several GB.
"""

from spherical_eeg_sim import Config, run_pipeline


def main():
    cfg = Config()
    cfg.sources.subcortical_fraction = 0.0
    cfg.sources.n_subcortical_positions = 0

    cfg.dataset.n_train = 100_000
    cfg.dataset.n_val = 10_000
    cfg.dataset.n_test = 10_000

    cfg.electrode_file = 'standard_1010_69.csv'
    cfg.output_dir = 'output_cortex_only'

    summary = run_pipeline(cfg)
    print()
    print('Output:', summary['output_dir'])


if __name__ == '__main__':
    main()
