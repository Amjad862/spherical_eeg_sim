"""
Export the position-pool lead-field operator and use it.

Setting output.save_leadfield writes leadfield.h5, leadfield.npz, or a MATLAB
v7.3 leadfield.mat next to the dataset. It contains the referenced forward operator evaluated
once at every source position, plus the source positions, population labels,
and the per-position train/val/test split. Any dipole moment m = [p_r, p_theta,
p_phi] (local spherical frame) projects to the configured sensor reference as
L @ m.

This is what a physics-informed model needs (the forward map itself), and it
lets you build classical inverse operators without recomputing the forward
solution. Below we load it and build a trivial minimum-norm-style pseudo-inverse
over the training positions as a sanity demonstration.

Run from the repository root:
    python examples/export_leadfield.py

Generates 2,000 train; output is well under a GB.
"""

import numpy as np
import h5py

from spherical_eeg_sim import Config, run_pipeline


def main():
    cfg = Config()
    cfg.dataset.n_train = 2_000
    cfg.dataset.n_val = 500
    cfg.dataset.n_test = 500
    cfg.sources.n_cortical_positions = 3_000
    cfg.sources.n_subcortical_positions = 500
    cfg.electrode_file = 'standard_1010_69.csv'
    cfg.output.save_leadfield = True
    cfg.output_dir = 'output_leadfield'
    run_pipeline(cfg)

    with h5py.File('output_leadfield/leadfield.h5', 'r') as f:
        L = f['leadfield'][:]                 # (P, N, 3), V per A*m
        split = f['split'][:]                 # 0=train 1=val 2=test

    P, N, _ = L.shape
    print(f'Lead-field operator: {P} positions x {N} channels x 3 components')
    print(f'  train/val/test positions: '
          f'{(split == 0).sum()}/{(split == 1).sum()}/{(split == 2).sum()}')

    # Flatten the fixed dipole-orientation basis into a dense gain matrix
    # G (N x 3P) over the TRAIN positions, then form a Tikhonov-regularized
    # pseudo-inverse. (Illustrative only -- not a serious inverse method.)
    train = np.where(split == 0)[0]
    G = L[train].transpose(1, 0, 2).reshape(N, 3 * train.size)   # (N, 3*Ptrain)
    lam = 1e-2 * np.trace(G @ G.T) / N
    G_pinv = G.T @ np.linalg.inv(G @ G.T + lam * np.eye(N))       # (3*Ptrain, N)
    print(f'  built regularized inverse operator of shape {G_pinv.shape}')


if __name__ == '__main__':
    main()
