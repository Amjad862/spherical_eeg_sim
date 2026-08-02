"""
Load a generated dataset into MNE-Python.

Requires MNE (an optional dependency):
    pip install mne
    # or: pip install spherical_eeg_sim[mne]

dataset_to_epochs reads a generated HDF5 split file and returns an
mne.EpochsArray with the electrode montage attached and the per-sample ground
truth (population, SNR, source depth, source position and orientation, ...)
attached as epochs.metadata. Physical-voltage datasets pass through unchanged;
target-SNR datasets are restored from their stored per-sample voltage scale.

Run from the repository root:
    python examples/load_with_mne.py

Generates 1,000 train; output is well under a GB.
"""

from spherical_eeg_sim import Config, run_pipeline


def main():
    cfg = Config()
    cfg.dataset.n_train = 1_000
    cfg.dataset.n_val = 200
    cfg.dataset.n_test = 200
    cfg.electrode_file = 'standard_1010_69.csv'
    cfg.output_dir = 'output_mne_demo'
    run_pipeline(cfg)

    from spherical_eeg_sim.io.mne_export import dataset_to_epochs

    epochs = dataset_to_epochs('output_mne_demo/train.h5')
    print(epochs)
    print('Channels:', len(epochs.ch_names), '| sfreq:', epochs.info['sfreq'])
    print('Metadata columns:', list(epochs.metadata.columns))

    # The returned object supports standard MNE operations, e.g.:
    #   epochs.plot_sensors(show_names=True)
    #   evoked = epochs['snr_db > 10'].average()
    #   epochs.get_data()  # (n_epochs, n_channels, n_times) in volts
    depth = epochs.metadata['source_depth_mm']
    print(f'source_depth_mm: min={depth.min():.2f}, '
          f'median={depth.median():.2f}, max={depth.max():.2f}')


if __name__ == '__main__':
    main()
