"""Cross-montage generalization protocol: train on montage A, test on montage B.

A model that localizes well on the electrode layout it was trained on may fail
when the layout changes. Because spherical_eeg_sim separates the source pool
from the montage, it can pose electrode-transfer generalization cleanly:

  * Generate a dataset under montage A and under montage B with the SAME
    master_seed. The source positions, orientations, magnitudes, time courses,
    and train/val/test position split are then identical across the two
    datasets. Sensor-space noise and EEG necessarily depend on the channel
    array. This script verifies the source-side alignment.
  * Train your model on ``A/train.h5`` and evaluate it on ``B/test.h5``. Any drop
    relative to evaluating on ``A/test.h5`` isolates the effect of changing the
    sensor array and its sensor-space noise, with source-side factors held fixed.

Run from the repository root:
    python examples/cross_montage_generalization.py

Generates 5,000 train per montage; output is well under a GB.
"""

import numpy as np

from spherical_eeg_sim import Config, run_pipeline


def generate_for_montages(montages, master_seed=862, out_prefix='xmont',
                          n_train=5000, n_val=1000, n_test=1000):
    """Generate one dataset per montage sharing sources and splits.

    Returns a dict mapping montage name -> output directory.
    """
    out_dirs = {}
    for montage in montages:
        cfg = Config()
        cfg.source_mode = 'point'
        cfg.dataset.n_train = n_train
        cfg.dataset.n_val = n_val
        cfg.dataset.n_test = n_test
        cfg.electrode_file = montage
        cfg.master_seed = master_seed          # shared seed => aligned sources
        out = f"{out_prefix}_{montage.replace('.csv', '')}"
        cfg.output_dir = out
        run_pipeline(cfg)
        out_dirs[montage] = out
    return out_dirs


def verify_sources_aligned(dir_a, dir_b, split='test'):
    """Confirm two datasets share the serialized source construction."""
    import h5py
    with h5py.File(f"{dir_a}/{split}.h5", 'r') as a, \
         h5py.File(f"{dir_b}/{split}.h5", 'r') as b:
        source_keys = [
            'positions_xyz', 'orientations_world', 'magnitudes_nAm',
            'population', 'source_radius_mm', 'source_depth_mm', 'band_idx',
            'frequencies_hz', 'phases_rad', 'component_amplitudes',
            'n_components', 'source_time_course', 'position_pool_idx',
        ]
        aligned = all(
            np.array_equal(
                a[f'ground_truth/{key}'][:], b[f'ground_truth/{key}'][:],
                equal_nan=True,
            )
            for key in source_keys
        )
        na = a['eeg'].shape[1]
        nb = b['eeg'].shape[1]
    print(f"[{split}] source construction aligned across montages: {aligned} "
          f"(channels: {dir_a}={na}, {dir_b}={nb})")
    return aligned


def main():
    montage_a = 'standard_1010_32.csv'   # sparser array (train here)
    montage_b = 'standard_1010_69.csv'   # denser array (test transfer here)

    dirs = generate_for_montages([montage_a, montage_b])
    verify_sources_aligned(dirs[montage_a], dirs[montage_b], split='test')

    print()
    print("Protocol:")
    print(f"  train on {dirs[montage_a]}/train.h5")
    print(f"  test  on {dirs[montage_b]}/test.h5  (same sources, different sensors)")
    print("The gap vs. the matched montage measures transfer to the other sensor array.")


if __name__ == '__main__':
    main()
