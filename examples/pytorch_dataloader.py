"""
Minimal PyTorch DataLoader over a generated dataset.

Requires PyTorch (not a dependency of this package):
    pip install torch

Shows how to wrap a generated HDF5 split file as a torch Dataset that yields
(eeg, target) pairs, where the target here is the single-source 3-D position
(swap in whatever ground-truth field your task needs -- orientation, magnitude,
SNR, source depth, topographic concentration, or band). The file is opened
lazily per worker so it works with ``num_workers > 0``.

Run from the repository root:
    python examples/pytorch_dataloader.py

Generates 2,000 train; output is well under a GB.
"""

import numpy as np
import h5py

from spherical_eeg_sim import Config, run_pipeline


class EEGSourceDataset:
    """Lazy HDF5-backed dataset yielding (eeg, position) tensors.

    Subclasses torch.utils.data.Dataset at runtime so importing this module
    does not require torch to be installed.
    """

    def __init__(self, h5_path, target_key='positions_xyz'):
        self.h5_path = str(h5_path)
        self.target_key = target_key
        self._f = None
        with h5py.File(self.h5_path, 'r') as f:
            self._len = f['eeg'].shape[0]

    def _file(self):
        if self._f is None:
            self._f = h5py.File(self.h5_path, 'r')   # opened per worker
        return self._f

    def __len__(self):
        return self._len

    def __getitem__(self, i):
        import torch
        f = self._file()
        eeg = f['eeg'][i].astype(np.float32)                 # (N, T)
        target = f[f'ground_truth/{self.target_key}'][i].astype(np.float32)
        return torch.from_numpy(eeg), torch.from_numpy(np.atleast_1d(target))


def main():
    cfg = Config()
    cfg.dataset.n_train = 2_000
    cfg.dataset.n_val = 500
    cfg.dataset.n_test = 500
    cfg.electrode_file = 'standard_1010_69.csv'
    cfg.output.dtype = 'float32'
    cfg.output_dir = 'output_torch_demo'
    run_pipeline(cfg)

    try:
        from torch.utils.data import DataLoader
    except ImportError:
        print('PyTorch not installed; the EEGSourceDataset class above is ready '
              'to use once you `pip install torch`.')
        return

    ds = EEGSourceDataset('output_torch_demo/train.h5', target_key='positions_xyz')
    # num_workers=0 keeps this example simple and portable; the lazy per-worker
    # file handle above also supports num_workers > 0.
    loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=0)
    eeg, pos = next(iter(loader))
    print('Batch EEG shape:', tuple(eeg.shape), '| target shape:', tuple(pos.shape))


if __name__ == '__main__':
    main()
