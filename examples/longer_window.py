"""
Longer analysis window for better low-frequency resolution.

The default 100-sample window at 500 Hz (200 ms) under-resolves the lowest EEG
bands (a single delta cycle is longer than the window). If your method needs
well-resolved low-frequency content, lengthen the window. Here we use 1,000
samples at 500 Hz (2 s), so the 0.5 Hz delta lower edge completes one cycle and
passes the default ``min_band_cycles=1`` policy.

Run from the repository root:
    python examples/longer_window.py

Generates 20,000 train; output is a few GB at the longer window.
"""

from spherical_eeg_sim import Config, run_pipeline


def main():
    cfg = Config()
    cfg.signal.window_samples = 1_000        # 2 seconds at 500 Hz

    cfg.dataset.n_train = 20_000
    cfg.dataset.n_val = 2_000
    cfg.dataset.n_test = 2_000

    cfg.electrode_file = 'standard_1010_69.csv'
    cfg.output_dir = 'output_long_window'

    summary = run_pipeline(cfg)
    print()
    print('Output:', summary['output_dir'])


if __name__ == '__main__':
    main()
