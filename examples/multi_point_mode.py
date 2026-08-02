"""
Multi-point dataset: 1-5 dipoles per sample with min-separation enforcement.

Each sample contains a variable number of dipoles with configurable temporal
coupling. Useful
for evaluating:
  - Multi-source detectors
  - How localization accuracy degrades with source count
  - Source-separation methods

Ground truth is padded to ``max_K`` with documented sentinels. It includes the
complete per-source base spectral construction, blend parameters, and final
time courses, so the multi-source signal can be reconstructed from the file.

Generates 50,000 train; output is a few GB.
"""

from spherical_eeg_sim import Config, run_pipeline


def main():
    cfg = Config()
    cfg.source_mode = 'multi_point'
    cfg.multi_point.n_sources_min = 1
    cfg.multi_point.n_sources_max = 5
    cfg.multi_point.min_separation_mm = 20.0
    cfg.multi_point.max_separation_attempts = 100
    cfg.multi_point.time_coupling = 0.0   # 0=independent courses, 1=fully coherent

    cfg.dataset.n_train = 50_000
    cfg.dataset.n_val = 5_000
    cfg.dataset.n_test = 5_000

    cfg.electrode_file = 'standard_1010_69.csv'
    cfg.output_dir = 'output_multi_point'

    summary = run_pipeline(cfg)
    print()
    print('Output:', summary['output_dir'])


if __name__ == '__main__':
    main()
