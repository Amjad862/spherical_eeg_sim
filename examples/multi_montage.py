"""
Multiple montages, identical sources: electrode-density comparison.

Generates one dataset per electrode montage while holding the source
configuration and the master seed fixed. Because every sample is a pure
function of (master_seed, split_id, global_index) -- independent of the
montage -- the same dipoles, orientations, magnitudes, time courses, and
train/val/test position split are used across all montages. Scalp projections
and sensor-space noise depend on each channel array. The comparison therefore
measures the complete observation-system change, not electrode count alone;
layouts should be chosen carefully when attributing an effect to density.

Run from the repository root:
    python examples/multi_montage.py

Generates 5,000 train per montage; output is well under a GB.
"""

from spherical_eeg_sim import Config, run_pipeline

MONTAGES = [
    'standard_1020_19.csv',           # 19 channels
    'standard_1010_69.csv',           # 69 channels
    'biosemi_128_spherical.csv',      # 128 channels
    'gsn_hydrocel_256_spherical.csv', # 256 channels
    'standard_1005_336.csv',          # 336 channels
]


def main():
    for montage in MONTAGES:
        cfg = Config()                 # identical source config every time
        cfg.master_seed = 862          # SAME seed -> same sources & same split
        cfg.electrode_file = montage
        # Reduced tutorial counts; measure the required scale for the task.
        cfg.dataset.n_train = 5_000
        cfg.dataset.n_val = 1_000
        cfg.dataset.n_test = 1_000
        cfg.output_dir = f"output_montages/{montage.replace('.csv', '')}"
        summary = run_pipeline(cfg)
        print('Done:', summary['output_dir'])


if __name__ == '__main__':
    main()
