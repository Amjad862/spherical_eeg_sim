"""Inspect explicit source depth and a descriptive scalp-map statistic.

The generator stores source radius and depth directly in millimetres. It also
stores ``signal_topographic_concentration``: peak absolute channel amplitude
divided by channel RMS in the average-referenced signal map at peak global
field power. The statistic is scale-invariant, but montage, position, and
orientation all affect it; it is not an anatomical depth label.

This example reports their empirical association for one generated dataset
without assuming that the relationship is monotonic or universal.

Run from the repository root:
    python examples/depth_and_topography.py

Generates 8,000 train; output is well under a GB.
"""

import h5py
import numpy as np

from spherical_eeg_sim import Config, run_pipeline


def main():
    cfg = Config()
    cfg.dataset.n_train = 8_000
    cfg.dataset.n_val = 1_000
    cfg.dataset.n_test = 1_000
    cfg.electrode_file = 'standard_1010_69.csv'
    cfg.output_dir = 'output_depth_topography_demo'
    run_pipeline(cfg)

    with h5py.File('output_depth_topography_demo/train.h5', 'r') as f:
        depth_mm = f['ground_truth/source_depth_mm'][:]
        population = f['ground_truth/population'][:]
        concentration = f[
            'ground_truth/signal_topographic_concentration'
        ][:]

    cortical = population == 0
    d = depth_mm[cortical]
    c = concentration[cortical]
    print(
        'corr(cortical depth, topographic concentration) = '
        f'{np.corrcoef(d, c)[0, 1]:+.3f}'
    )
    print(f"{'depth band (mm)':>18}  {'median concentration':>22}")
    edges = np.linspace(d.min(), d.max(), 6)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (d >= lo) & ((d <= hi) if i == len(edges) - 2 else (d < hi))
        if mask.any():
            print(f'{lo:6.1f}-{hi:5.1f}      {np.median(c[mask]):22.2f}')


if __name__ == '__main__':
    main()
