"""
Half-size storage with float32 EEG for large-scale ML training.

The generator computes the forward model in float64. The EEG tensor is then
cast to ``output.dtype`` on write. Float32 reduces the uncompressed EEG-array
byte width by half and is often convenient for GPU training; whether its
quantization is acceptable is a task-specific decision that should be
measured. Scientific ground-truth arrays retain their defined dtypes.

Run from the repository root:
    python examples/float32_storage.py

Generates 100,000 train; output is several GB, halved by the float32 storage dtype.
"""

from spherical_eeg_sim import Config, run_pipeline


def main():
    cfg = Config()
    cfg.output.dtype = 'float32'

    cfg.dataset.n_train = 100_000
    cfg.dataset.n_val = 10_000
    cfg.dataset.n_test = 10_000

    cfg.electrode_file = 'standard_1010_69.csv'
    cfg.output_dir = 'output_float32'

    summary = run_pipeline(cfg)
    print()
    print('Output:', summary['output_dir'])


if __name__ == '__main__':
    main()
