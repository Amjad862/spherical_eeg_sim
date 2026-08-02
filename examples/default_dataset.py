"""
Default dataset: 300K / 30K / 30K samples with all defaults.

This is the default standalone benchmark configuration. Measure runtime and
disk use on the target system: they depend on hardware, compression, montage,
dtype, and optional artifacts. HDF5 split generation is streamed in chunks.

Run from the repository root:
    python examples/default_dataset.py

Generates 300,000 train / 30,000 val / 30,000 test; output is the full shipped default; tens of GB and a long run.
"""

from spherical_eeg_sim import Config, run_pipeline


def main():
    cfg = Config()
    # All defaults: 300K/30K/30K, 35K cortical + 6.5K subcortical positions,
    # n_legendre=80, source_mode='point', subcortical_fraction=0.15,
    # IFCN bands, 100-sample window at 500 Hz, three-component noise,
    # absolute_floor noise (fixed 1.00 uV RMS, so SNR is emergent), and
    # log-uniform magnitudes.

    cfg.electrode_file = 'standard_1010_69.csv'
    cfg.output_dir = 'output_default_dataset'

    summary = run_pipeline(cfg)
    print()
    print('Output:', summary['output_dir'])


if __name__ == '__main__':
    main()
