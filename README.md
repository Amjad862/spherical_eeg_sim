# SphericalEEGSim

SphericalEEGSim generates controlled synthetic scalp-EEG datasets for training
and benchmarking electrical source-imaging methods. It uses the analytic
concentric-shell series of de Munck and Peters, supports point, multi-point,
and cortical-patch sources, and splits the source-position pool before samples
are drawn so positions cannot leak between training, validation, and test data.

The package is a controlled spherical benchmark generator, not a substitute
for subject-specific BEM/FEM modeling or real-data validation.

## Highlights

- Four concentric layers (brain, CSF, skull, scalp), configurable geometry and
  conductivities, and a numerically audited Legendre expansion.
- Point, multi-point, and distributed spherical-cortical patch modes.
- Explicit position, orientation, magnitude, source depth, source time course,
  spectral construction, SNR, and position-pool provenance.
- A three-component noise model: forward-projected background dipoles,
  independent pink noise, and independent white noise.
- Two noise conventions: a physical absolute noise floor or per-sample target
  SNR, plus an independent optional unit-RMS output transform with a stored
  voltage scale.
- Sixteen bundled montage files plus custom electrode CSV input.
- HDF5, deterministic NPZ, and genuine MATLAB v7.3 output.
- Optional referenced lead-field, electrode-graph, and MNE-Python export.
- Byte-identical canonical output across generation chunk sizes and worker
  counts for a fixed scientific configuration and seed in the tested software
  environment; cross-platform identity must be verified rather than assumed.

Scientific defaults and their evidence are catalogued in
[`docs/parameters.md`](docs/parameters.md); complete BibTeX metadata is in
[`docs/references.bib`](docs/references.bib).

## Install

```bash
git clone https://github.com/Amjad862/spherical_eeg_sim.git
cd spherical_eeg_sim
pip install -e .
pip install -e ".[plot]"       # diagnostic plots
pip install -e ".[mne]"        # MNE export
pip install -e ".[mat]"        # MATLAB v7.3 output
pip install -e ".[all]"        # all optional and test dependencies
```

Python 3.9 or later is required. The core dependencies are NumPy, SciPy, and
h5py. MATLAB output additionally requires `hdf5storage`.

## Quick start

```python
from spherical_eeg_sim import quickstart_config, run_pipeline

cfg = quickstart_config()
cfg.output_dir = "my_dataset"

summary = run_pipeline(cfg)
print(summary["splits"])
```

`run_pipeline` refuses to write into a non-empty dataset directory. Set
`cfg.output.overwrite = True` only when the complete previous directory should
be replaced. Generation occurs in a sibling staging directory and the finished
directory is published as one operation, so a failed run does not expose a
partially updated dataset. An internal ownership marker permits safe cleanup of
an incomplete staging tree; the marker is removed before publication. A
checksum-complete tree is preserved for recovery after a transient rename
failure.

For a small smoke test:

```bash
python examples/quickstart_small.py
```

Installing the package also provides a command-line entry point that wraps the
same pipeline:

```bash
spherical-eeg-sim --quickstart --output-dir my_dataset
spherical-eeg-sim --config my_config.json
spherical-eeg-sim --source-mode patch --n-train 50000 --save-config used.json
```

Run `spherical-eeg-sim --help` for the full option list, or invoke the module
directly as `python -m spherical_eeg_sim.cli`. The Python API is the better
choice when the run summary is needed programmatically.

The full default configuration requests 300,000/30,000/30,000 samples. These
are declared benchmark-capacity settings, not a literature-prescribed sample
size or a statistical power calculation; `docs/parameters.md` gives the cited
rationale and scope. Runtime and storage depend strongly on montage size,
compression, hardware, and enabled auxiliary outputs; the package does not
promise an estimated time or size in place of a measured run.

## Source modes

`cfg.source_mode = "point"` generates one dipole per sample.

`cfg.source_mode = "multi_point"` generates a configurable number of dipoles.
Unused source slots are padded. Each active source has a stored base sinusoid
construction and stored blend weights, normalization standard deviation, and
final `source_time_courses`, so the serialized EEG construction is auditable.
The default separation policy raises if rejection sampling cannot meet the
requested minimum Euclidean separation. Exploratory runs can explicitly choose
`accept_and_flag`; only then does `separation_violated` report accepted failures.
`multi_point.time_coupling_values` can sample coupling per window and records the
realized value in `ground_truth/time_coupling`.

`cfg.source_mode = "patch"` generates a radially oriented source patch on the
spherical cortical shell. Membership is based on great-circle surface distance.
The default profile is a Gaussian with `sigma = diameter/4` truncated at the
patch radius; it has a nonzero limiting boundary weight before the hard cutoff.
The exact active source-pool indices and signed radial dipole moments are stored
without padding in CSR row encoding. `source_pool.*` maps those indices to
physical coordinates and split assignments. Patch mode allocates equal-size
cortical support meshes to train, validation, and test so support density does
not vary with the requested sample counts.

## Noise and physical scale

With the default `cfg.noise.noise_model = "absolute_floor"`, mixed noise is
scaled to `absolute_noise_rms_uV` on the raw sensor montage. The referenced EEG
is stored in volts. SNR is emergent rather than prescribed, and
`eeg_normalization_scale_V` is one.

The default floor is 1.00 µV RMS, selected to retain a useful emergent-SNR
range across the package's broad default source and montage settings. The
3.16 µV RMS calibration used by Goldenholz et al. (2009) and Piastra et al.
(2021) remains available by setting `absolute_noise_rms_uV = 3.16`; for a fixed
signal, that setting lowers SNR by approximately 10 dB relative to 1.00 µV.

With `noise_model = "snr_target"`, mixed noise is scaled to the drawn target on
the montage selected by `snr_definition`; the referenced EEG remains in volts.
Output normalization is controlled independently by
`cfg.output.eeg_normalization`. Setting it to `"per_sample_rms"` produces
dimensionless unit-RMS arrays, and `eeg_normalization_scale_V` records the exact
physical voltage represented by one stored unit:

```text
eeg_in_volts = eeg * eeg_normalization_scale_V
```

This optional transform can prevent absolute input scale from trivially
carrying the magnitude label while preserving reversible physical provenance.
Without it, `eeg_normalization_scale_V` is one. Both noise conventions store
`snr_db_raw` and `snr_db_referenced`; `snr_db` selects one of them according to
`cfg.noise.snr_definition`.

Each split declares `magnitude_convention`. With the default centered source
courses it is `rms`, so `magnitudes_nAm` is the RMS dipole moment. If
`cfg.signal.zero_mean_signal=False`, it is `waveform_scale`, and the
instantaneous moment is `magnitudes_nAm * source_time_course` (or the matching
row of `source_time_courses` for a multi-point sample).

Source radius and depth are stored explicitly in millimetres. The additional
`signal_topographic_concentration` field is the peak absolute channel value
divided by channel RMS in the average-referenced signal map at peak global
field power. It is a descriptive, scale-invariant sensor-map statistic, not an
anatomical depth estimator.

## Output schema

Each split is `train.h5`, `val.h5`, and `test.h5` for HDF5; `.npz` and `.mat`
are used for the other backends. HDF5 has the following common layout:

```text
/eeg                                           (B, channels, time)
/ground_truth/positions_xyz                    (B, 3) or (B, max_K, 3)
/ground_truth/orientations_world               (B, 3) or (B, max_K, 3)
/ground_truth/magnitudes_nAm                   (B,) or (B, max_K)
/ground_truth/population                       cortical=0, subcortical=1
/ground_truth/source_radius_mm
/ground_truth/source_depth_mm
/ground_truth/snr_db
/ground_truth/snr_db_raw
/ground_truth/snr_db_referenced
/ground_truth/eeg_normalization_scale_V
/ground_truth/signal_topographic_concentration
/ground_truth/band_idx                         index into metadata band_names
/ground_truth/frequencies_hz
/ground_truth/phases_rad
/ground_truth/component_amplitudes
/ground_truth/n_components
/ground_truth/source_time_course
/ground_truth/position_pool_idx
/electrodes/names
/electrodes/cartesian_xyz
/electrodes/spherical_r_theta_phi
/metadata/...
```

Multi-point additions:

```text
/ground_truth/n_dipoles_per_sample
/ground_truth/separation_violated
/ground_truth/time_coupling
/ground_truth/source_time_courses
/ground_truth/source_base_band_idx
/ground_truth/source_base_frequencies_hz
/ground_truth/source_base_phases_rad
/ground_truth/source_base_component_amplitudes
/ground_truth/source_base_n_components
/ground_truth/source_primary_blend_weight
/ground_truth/source_independent_blend_weight
/ground_truth/source_blend_std
```

Patch additions:

```text
/ground_truth/patch_diameter_mm
/ground_truth/n_active_per_patch
/ground_truth/patch_active_position_idx/{values,row_splits}
/ground_truth/patch_active_moments_nAm/{values,row_splits}
```

When `save_components=True`, `eeg_signal_only` and `eeg_noise_only` are also
stored in ground truth and use the configured EEG dtype and units.

The canonical dataset directory also contains:

- `metadata.json`: deterministic scientific configuration and a complete,
  sorted inventory of canonical files;
- `environment.json`: Python, platform, package, and numerical-library
  versions that produced the canonical artifacts;
- `validation_report.json`: automatic internal checks and their measurements;
- `README.txt`: generated format summary;
- `diagnostic_plots/`: generated plots when enabled;
- `source_pool.*`: coordinates and split/population labels for every source
  position referenced by the split files;
- optional `leadfield.*` and `graph.*`;
- `checksums.sha256`: SHA-256 verification for every other canonical artifact;
  verification also requires exact agreement with the metadata inventory.

Verify a completed directory with `sha256sum -c checksums.sha256` from inside
that directory.

Run-specific operational provenance is written beside the canonical artifact
directory as `<output_dir>.runtime.json`. It records the timestamp, wall time,
output path, generation chunk size, requested and resolved worker counts, and
the same software-environment fields carried by `environment.json`.

HDF5 streams to disk. NPZ and MATLAB v7.3 collect a split in memory before
writing, so HDF5 is the appropriate backend for default-scale generation.

## Reproducibility contract

Setup stages use deterministic substreams. `position_seed` controls the source
pool, position split, and background-projection pool; `sample_seed` controls
per-window source, signal, and noise streams. Either can be left null to use
`master_seed`. Every sample derives its own random streams from
`(sample_seed, split_id, global_sample_index)`. Canonical metadata
excludes output paths, timestamps, `chunk_samples`, worker count, and overwrite
policy. Deterministic container settings and a generation-independent HDF5
chunk layout make complete canonical directory hashes invariant to
`cfg.output.chunk_samples` and `cfg.output.n_workers`. MATLAB v7.3 output also
replaces `hdf5storage`'s wall-clock `Created on` user-block field with a fixed
value before publication.

The test suite compares SHA-256 hashes of complete point, multi-point, and patch
directories for HDF5, NPZ, and MATLAB output, including diagnostics and
requested auxiliary artifacts, across different generation chunk sizes and
worker counts. Changing
a scientific parameter, output dtype, compression, or `hdf5_chunk_rows` is a
different canonical configuration and may change the hash.

## Referencing and lead fields

`cfg.output.reference_mode` supports `none`, `average`, `cz`, and `custom`.
Electrode coordinates and the resolved reference description travel with every
split. Signal and noise are referenced separately before their achieved SNRs
are measured.
`none` means that no discrete electrode is subtracted; the analytic series,
which begins at Legendre order one, uses the zero-spherical-mean potential
gauge.

Set `cfg.output.save_leadfield = True` to write the basis operator at every
position in the pool. Its shape is `(P, channels, 3)`, its columns correspond
to local spherical radial/theta/phi moment components, its units are V/(A·m),
and the configured reference has already been applied. The file also stores
position, population, and split labels.

The low-level `forward_potential`, `forward_potential_basis`, and
`forward_potential_batch` functions require `brain_radius_m` and reject source
radii on or outside that boundary. Transfer coefficients and the supplied
brain radius must describe the same head model. At configuration level,
`cfg.head.truncation_tail_tolerance` also rejects a source domain whose
geometric Legendre weighting decays too slowly at the selected expansion order.

Set `cfg.output.save_graph = True` to write a bidirectional electrode graph.
Available constructions are k-nearest neighbors, spherical Delaunay, and a
distance threshold. Node order matches the embedded electrode order.

## MNE-Python

```python
from spherical_eeg_sim.io.mne_export import dataset_to_epochs

epochs = dataset_to_epochs("my_dataset/train.h5", selection=slice(0, 256))
```

The helper restores any normalized samples to volts before constructing
`mne.EpochsArray`, attaches the embedded montage, and exposes scalar and
three-vector ground truth (including position and orientation) in
`epochs.metadata` when pandas is installed. It reads an optional epoch slice
without loading the full split and records an average-reference projector or
custom-reference state in MNE metadata without re-referencing the stored data.
Direct NPZ/MAT loading remains the caller’s responsibility.

## Montage library

The package contains six standard-system montages and ten spherical vendor-net
approximations:

- standard: 19, 20, 32, 61, 69, and 336 channels;
- BioSemi: 16, 32, 64, 128, and 256 channels;
- EGI/Philips HydroCel: nominal 32, 64, 128, 256, and 257 nets.

The nominal HydroCel-32 template contains its 32 numbered sensors plus Cz, so
the exported file has 33 recording channels. BioSemi labels retain Iz as a
recording channel; only fiducials are excluded. Vendor files are spherical
projections of attributed MNE-Python templates and are explicitly named with a
`_spherical` suffix. All bundled coordinates are rescaled to the configured
scalp radius when loaded. See `spherical_eeg_sim/montages/_templates/NOTICE`.

## Validation scope

Every generation run executes internal checks for the transfer-coefficient
cross-check, direct-versus-basis consistency, rotational covariance,
finite/nonzero numerical sanity, configured distributions, and position-split
leakage. The split audit reads the source-pool indices back from each serialized
backend and checks both cross-split overlap and membership in the assigned pool.
Any failed distribution check aborts staged publication.
The algebraic checks share implementation assumptions and are correctly treated
as self-consistency tests.

Independent evidence is separate:

- `scripts/validate_inverse_mne.py` compares serialized point-source datasets
  with MNE-Python’s independent Berg–Scherg sphere approximation, including
  position, signed orientation, and instantaneous moment-amplitude recovery;
- `scripts/validate_patch_forward_mne.py` rebuilds each serialized patch from
  its stored CSR support and superposes the same independent forward solution,
  against a point-source control that establishes the approximation floor;
- `scripts/validate_transfer_coefficients.py` reproduces the surface coefficient
  from a direct 7x7 boundary-value solve that shares no algebra with the
  recurrence, at 60 decimal digits;
- `scripts/legendre_convergence.py` accepts the generation JSON configuration,
  derives valid cases from its source domain, and measures topography error
  against an arbitrary-precision reference (120 decimal digits, summed to order
  400) after verifying agreement with the order-200 reference;
- `scripts/measure_split_spacing.py` measures how far a held-out position
  actually lies from the nearest training position, which bounds what a
  position-level split can claim;
- `scripts/train_baseline.py` is a demonstration benchmark, not a validation
  of anatomical realism.

Install the exact direct numerical stack used for the archived reference runs
with `pip install -r validation-requirements.txt`.

These checks do not establish clinical validity or equivalence to an
individualized head model. The source release ships compact validation
summaries, per-sample metrics, and publication figures, but not full generated
training datasets.

## Examples

Each example states its own sample count and expected output size in its
docstring. They are working configurations rather than demonstrations, so most
generate real datasets; `quickstart_small.py` is the one intended as a smoke
test.

| Example | Train samples |
| --- | --- |
| `quickstart_small.py` | 800 |
| `load_with_mne.py` | 1,000 |
| `export_leadfield.py`, `export_graph.py`, `pytorch_dataloader.py` | 2,000 |
| `head_model_presets.py` | 3,000 |
| `multi_montage.py`, `cross_montage_generalization.py`, `snr_target_vs_absolute_floor.py` | 5,000 |
| `depth_and_topography.py` | 8,000 |
| `longer_window.py` | 20,000 |
| `high_snr_clean.py`, `multi_point_mode.py`, `patch_mode.py` | 50,000 |
| `cortex_only.py`, `float32_storage.py` | 100,000 |
| `default_dataset.py` | 300,000 (the full default; tens of GB) |

The examples use explicit output directories. Re-running one requires deleting
its old output yourself or setting `cfg.output.overwrite = True` deliberately.

## Citation and license

Machine-readable citation metadata is in `CITATION.cff`. SphericalEEGSim is
released under the [BSD 3-Clause License](LICENSE). The bundled third-party
montage templates retain their attributed notices in
`spherical_eeg_sim/montages/_templates/NOTICE`.
