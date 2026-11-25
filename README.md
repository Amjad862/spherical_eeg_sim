# SphericalEEGSim

SphericalEEGSim generates controlled synthetic scalp-EEG datasets for training
and benchmarking electrical source-imaging methods. It uses the analytic
concentric-shell series of de Munck and Peters, supports point, multi-point,
and cortical-patch sources. By default, it assigns disjoint sets of exact
source-pool positions to training, validation, and test before drawing samples.
A region policy instead holds out buffered spatial sectors, and in patch mode
keeps every patch footprint inside its own split.

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
- Three split policies: disjoint exact positions (default), buffered region
  holdout for every source mode, and explicitly shared positions.
- Sixteen bundled montage files plus custom electrode CSV input.
- HDF5, deterministic NPZ, and genuine MATLAB v7.3 output.
- Optional referenced lead-field, electrode-graph, and MNE-Python export.
- Output that does not depend on the generation chunk size or the number of
  worker processes: for a fixed scientific configuration and seed the files are
  byte-identical within one software environment (tested with HDF5 1.14.5 and
  2.0.0). Identity across platforms or library versions must be verified
  rather than assumed.

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

Python 3.10 or later is required. The core dependencies are NumPy, SciPy, and
h5py. MATLAB output additionally requires `hdf5storage`.
After pulling a new version into an editable checkout, reinstall with
`pip install -e ".[all]"` to refresh its distribution metadata and dependencies.

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

Two runs can write to different directories at the same time. A second run
aimed at an active output directory stops before recovery. The empty sibling
file `.<dataset-name>.generation.lock` can remain after the run finishes. The
operating system releases the lock when the generating process exits.

For a small smoke test:

```bash
python examples/quickstart_small.py
```

Installing the package also provides a command-line entry point that wraps the
same pipeline:

```bash
spherical-eeg-sim --quickstart --output-dir my_cli_dataset --save-config my_cli_config.json
spherical-eeg-sim --config my_cli_config.json --output-dir my_configured_dataset
spherical-eeg-sim --source-mode patch --n-train 50000 --save-config used.json
```

Run `spherical-eeg-sim --help` or `python -m spherical_eeg_sim.cli --help` for
the full option list. The command starts a run only when its size has been
chosen: by `--quickstart`, by `--config`, by one of `--n-train`, `--n-val` and
`--n-test`, or by `--default-run` for the full default dataset. Called without
any of these it prints the size of the default dataset and stops. Every run
prints its planned size first. The Python API returns the run summary for
programmatic use.

The full default configuration requests 300,000/30,000/30,000 samples, about
18.5 GiB of float64 EEG on the default 69-channel montage. These are declared
benchmark-capacity settings, not a literature-prescribed sample size or a
statistical power calculation; `docs/parameters.md` gives the cited rationale
and scope. Runtime and storage depend on montage size, compression, hardware,
and enabled auxiliary outputs. Measured times, memory and sizes for
representative runs, including the full default run, are in
`validation_results/generation_benchmark.json` and are described under
[Generation benchmark](#generation-benchmark).

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

`cfg.source_mode = "patch"` generates one radially oriented source patch per window on the
spherical cortical shell. Membership is based on great-circle surface distance.
The default profile is a Gaussian with `sigma = diameter/4` truncated at the
patch radius; it has a nonzero limiting boundary weight before the hard cutoff.
The exact active source-pool indices and signed radial dipole moments are stored
without padding in CSR row encoding. `source_pool.*` maps those indices to
physical coordinates and split assignments. Under the default split policy,
patch mode allocates equal-size cortical support meshes to train, validation,
and test so support density does not vary with the requested sample counts.

## Source split policies

| `cfg.split.policy` | Source locations eligible for each split |
|---|---|
| `position` (default) | Disjoint exact position indices. Near neighbours and overlapping patch footprints can still occur across splits. |
| `region` | Separate rotated spherical sectors with a clearance buffer, for every source mode. Set `region_count`, `buffer_mm`, and `region_rotation_seed`. Validation stays clear of test, and training stays clear of both. In patch mode a patch is centred only where its whole footprint stays inside its own split region. |
| `shared` | Point and multi-point modes draw independently seeded windows. Set `shared_position_fraction` in `(0, 1]` to share part or all of the common finite position pool. |

Use `plan_position_split(cfg)` from `spherical_eeg_sim` to inspect positions,
populations, and eligible indices without computing a lead field or EEG.
The generated `validation_report.json` gives nearest-neighbour distances for
eligible and used positions and, for `shared`, the observed fraction of query
windows with a position also used in the reference split. Sharing a location
does not reuse the EEG window. The source-position and background-projection
pools use the same seeds across policies.

Use `position` to measure interpolation to unseen locations in the same sampled
domain, `region` to test spatial separation, and `shared` for new windows at
known locations. The region layout is rotated by its declared seed. Sector
directions are ordered by height, and one test sector is drawn per height group.
The remaining sectors are regrouped by height for validation, with one sector
drawn per group. All other sectors form training before the clearance buffer
is applied. The package checks clearance and population
availability. Additional study-specific coverage thresholds are enforced by
`scripts/run_split_study.py`.

`shared_position_fraction` is a fraction of the entire source pool, rather than
the test pool or test windows. A value of 0.01 produced 95 shared eligible
locations, 81 used by both training and test, and 231 of 3,000 test windows
(7.7%) at observed training locations in the recorded seed-862 demonstration.
With a fraction of 1.0, a full redraw of test locations from the training
pool, 2,618 of 3,000 test windows (87.3%) fell on a training location in the
same configuration, which is why disjoint positions are the default.
The realized window reuse depends on split sizes and sampling. Read
`observed_reuse` in the validation report for each dataset. Shared source-pool
rows carry split code `-2` and are eligible in all splits. Keep these rows when
selecting a split's eligible pool.

Patch mode supports `position` and `region`. Each window contains one patch.

Under `position`, each split receives one third of the cortical pool as its
support mesh. The three meshes interleave over the whole sphere, so patches
from different splits can cover the same cortex: this is a within-distribution
benchmark for patches.

Under `region`, each split keeps every pool position inside its own sectors,
and a patch is admitted only where its whole footprint lies inside those
sectors and contains no position of another split or of the buffer. The
footprint is the cap of directions within `diameter / 2` of the centre,
measured on the centre's own radius. Near sector corners the planner uses
a conservative bound, so it can exclude some centres whose footprints would
fit. The diameter is drawn first and the centre is then drawn uniformly from
the admitted positions, so every split keeps the configured diameter distribution. Footprints of
different splits are disjoint by construction, and their active dipoles are at
least `buffer_mm` apart. `plan_position_split(cfg)` raises, before any data are
generated, if the planner admits no centre for the largest patch in a split. Use
fewer, larger sectors (six to eight suit the default 10 to 60 mm diameters), a
smaller buffer, or a smaller maximum diameter. Region meshes keep the full
pool density, so a region dataset needs one third of the pool size of a
default-policy dataset for the same support density.

The validation report of every patch dataset gives, for each split pair, the
nearest centre, the highest spherical-cap intersection over union of the
footprints and, when no footprints overlap, the smallest gap between them.
For region datasets it also verifies from the written files that no patch was
clipped, and reports how many positions the planner admits for the smallest
and largest diameters. Set `cfg.patch.report_overlap = False` to omit the overlap
diagnostic. The report labels any deterministic query subset used to limit
work for large custom datasets.

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
pool, background-projection pool, and the `position` and `shared` assignments.
The `region` assignment uses `region_rotation_seed`. `sample_seed` controls
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
membership. The split audit reads the source-pool indices back from each
serialized backend and checks membership and cross-split reuse against the
configured split policy.
Any failed distribution check aborts staged publication.
The algebraic checks share implementation assumptions and are correctly treated
as self-consistency tests.

The validation scripts use several complementary checks.

- `scripts/validate_inverse_mne.py` checks point-source position, signed
  orientation, and instantaneous moment-amplitude recovery against MNE-Python's
  Berg–Scherg approximation under matched spherical head-model assumptions.
- `scripts/validate_patch_forward_mne.py` rebuilds each serialized patch from
  its stored CSR support and superposes the MNE-Python forward solution under
  the same matched-sphere assumptions, with a point-source comparison to
  characterize the difference between implementations.
- `scripts/run_sphere_fit_convergence.py` records how the Berg–Scherg parameter
  fit of MNE-Python ends for the head models of this package, and how both
  comparisons react to a converged fit, to last-bit changes of the fit
  objectives, to another kernel of the linear-algebra library and to dipole
  fits that are run to a small trust-region radius.
- `scripts/run_dipole_comparators_converged.py` repeats the sphere-model
  dipole fits of the experiments below on their own windows with a converged
  parameter fit.
- `scripts/check_installed_sphere_fit.py` counts how the Berg–Scherg parameter
  fit of the installed MNE-Python ends for the same head models, without
  changing the fit, so it runs with any version of MNE-Python; the two
  scripts above replace the optimizer call inside MNE-Python 1.12.1.
- `scripts/validate_transfer_coefficients.py` reproduces the surface coefficient
  from a direct 7x7 boundary-value solve that shares no algebra with the
  recurrence, at 60 decimal digits.
- `scripts/legendre_convergence.py` accepts the generation JSON configuration,
  derives valid cases from its source domain, and measures topography error
  against an arbitrary-precision reference (120 decimal digits, summed to order
  400) after verifying agreement with the order-200 reference.
- `scripts/measure_split_spacing.py` measures how far a held-out position
  actually lies from the nearest training position, which bounds what a
  position-level split can claim.
- `scripts/train_baseline.py` demonstrates supervised localization within the
  spherical model using the default position split.
- `scripts/run_split_study.py` compares the exact-position and region policies
  with a multilayer perceptron, a lead-field dictionary, nearest-neighbour
  lookups and a matched-sphere dipole fit.
- `scripts/run_patch_split_study.py` makes the same comparison for patches,
  with the overlap and containment audits of both policies.
- `scripts/run_head_transfer.py` scores the sphere-trained networks on paired
  windows generated with the fsaverage boundary-element model, and
  `scripts/run_head_adaptation.py` repeats the test with spheres matched to
  that head, sphere-model dipole fits, and networks trained or fine-tuned on
  boundary-element windows.
- `scripts/run_shared_position_demo.py` measures coordinate reuse when test
  locations are redrawn from a shared pool.
- `scripts/benchmark_generation.py` measures generation time, memory and size.

`validation-requirements.txt` records the direct numerical stack for the
historical v1.0.0 Linux validation. For the recorded Windows environment of the experiments,
use CPython 3.12.16 and `pip install -r validation-windows-requirements.txt`, then
`pip install -e ".[all]"`. The companion contains ordinary version pins.
`validation-windows-lock.txt` preserves the original environment record, including
two conda build paths, because the executed studies hash that original file.

The archived summaries of the numerical validation were recorded with v1.0.0 on
Linux. Versions 1.0.0 and 1.1.0 generate identical arrays for the two
comparisons with MNE-Python and give identical results on one machine. The
statistics of these comparisons are not identical between platforms, however:
a rerun on Windows with the same library versions gave, for example, a
90th-percentile position error of 0.84 mm where the archived run has
0.88 mm. Differences of this size arise in two fits of MNE-Python whose
results depend on rounding and on the kernel of the linear-algebra library.
For the head
model of this package, the fit of the Berg–Scherg parameters in MNE-Python
1.12.1 ended at the default limit of 1,000 function evaluations of SciPy
1.17.1 before it had converged, in every recorded run on the Windows machine;
where it ends sets the field difference of the patch
comparison. The dipole fits end when the trust-region radius of their
optimizer reaches 0.05 mm, and a last-bit change of their objective then
moves a fitted position by up to 1.12 mm, which moves the 90th percentile
and the maximum of the position error even with a converged parameter fit.
Over 19 runs on one machine that differed in the last bits of a fit objective
or in the kernel of OpenBLAS, the median position error was
0.26 to 0.29 mm, the 90th percentile 0.80 to 0.88 mm and the maximum
2.39 to 3.03 mm; in the eleven of these runs that include the patch
comparison, its median field difference was 1.8e-3 to 4.4e-3. With a
converged parameter fit the field difference was 5.0e-4 in every run. With
the dipole fits also ended at a radius of 1e-7 m (argument `tol` of
`mne.fit_dipole`), the three statistics agreed within 0.01 mm over eleven
runs (0.25, 0.69 and 2.47 mm without changes), although single fitted
positions still differed from those of the run without changes by up to
0.15 mm. These runs are examples; their ranges are not bounds for other
platforms.
`scripts/run_sphere_fit_convergence.py` records how the fits end and how the
comparisons react, `scripts/run_dipole_comparators_converged.py` repeats the
sphere-model dipole fits of the paper's experiments with a converged parameter
fit, and `validation_results/README.txt` has the numbers.

These numbers belong to MNE-Python 1.12.1. MNE-Python 1.13 ends the parameter
fit at a final trust-region radius of 1e-4 in place of 1e-5. With MNE-Python
1.13.2 on the same Windows machine, the fit for the head model of this package
again reached the limit of 1,000 evaluations, and 13 of the 15 fits for the
five spheres of the paper under three kernels of OpenBLAS stopped by
themselves after 52 to 326 evaluations, at objectives about 100 to 210 times
those at which the same fits stopped with MNE-Python 1.12.1 and the limit
raised. The dipole recovery then gave a median position
error of 0.27 mm (90th percentile 0.81 mm, maximum 2.30 mm) and the patch
comparison a median field difference of 1.8e-3.
`scripts/check_installed_sphere_fit.py` counts how the fit of the installed
MNE-Python ends, without changing it.

### Reproduce or verify the validation results

The version 1.1.0 test suite has **356 test functions covering 1,185 cases**
in the reference Windows Python 3.12 environment. Pytest runs each set of
inputs to a parameterized function as a separate case. Run `python -m pytest`
from the repository root to see the counts for your installation and selection.

Run these commands from the repository root. Fresh validation runs use a new
directory under `validation_runs/` and print their output paths. Explicit
validation output paths require `--overwrite` if the requested files already
exist. `validation_results/` contains the historical numerical validation,
the publication figures, and the tables of the experiments. Its README
identifies the roles and source records of these artifacts.

| Task | Command | Output and repeat behavior |
| --- | --- | --- |
| Verify the shipped archive | `python scripts/make_validation_manifest.py` | Read-only. Reports checksum drift and checks that the archived result bundles agree. `--check` is an alias. |
| Check the inverse pipeline | `python scripts/validate_inverse_mne.py --quick --no-sweep` | Writes `inverse_recovery_metrics.csv` and `inverse_recovery_summary.json` for 60 dipoles in a new `validation_runs/` directory each time. |
| Reproduce the archived spacing measurement | `python scripts/measure_split_spacing.py` | Writes a new `split_spacing.json` each time. Compare the cortical test-to-nearest-training medians with the archive at two decimal places (1.71 mm for the default pool, 3.23 mm for the baseline pool). |
| Generate a small dataset | `spherical-eeg-sim --quickstart --output-dir my_cli_dataset` | Writes a dataset to the chosen directory. Use a fresh directory for another run. |
| Measure dataset generation | `python scripts/benchmark_generation.py --output-dir validation_runs/generation` | Writes per-case configurations and logs, plus `generation_benchmark.json` with elapsed time, sampled summed process-tree RSS, and storage. Install `.[benchmark]`. Choose a fresh directory for each run. Like the experiment scripts, it records the commit and therefore runs only in a Git checkout without modified or untracked files. |
| Check patch fields against MNE-Python | `python scripts/validate_patch_forward_mne.py` | Writes `patch_forward_mne_summary.json` in a new `validation_runs/` directory each time. Needs the `mne` extra. |
| Check the transfer coefficients independently | `python scripts/validate_transfer_coefficients.py` | Writes `transfer_coefficients.json` in a new `validation_runs/` directory each time. Needs `mpmath`. |
| Audit Legendre truncation | `python scripts/legendre_convergence.py` | Writes `legendre_convergence.json` in a new `validation_runs/` directory each time. Needs `mpmath`; takes several minutes. |
| Train the baseline localizer | `python scripts/train_baseline.py` | Generates a 25,000-window dataset and writes `baseline_benchmark_summary.json` in a new `validation_runs/` directory each time; `--figure <path>` also writes the figure. Needs scikit-learn. |
| Regenerate the descriptive figures | `python scripts/make_validation_figures.py` | Writes `pipeline_schematic`, `topographies` and `distributions` as `.png` and `.pdf`, with `publication_figures.json`, in a new `validation_runs/` directory each time. Needs matplotlib. |
| Regenerate canonical results for maintenance | `python scripts/validate_inverse_mne.py --write-canonical` | Replaces `ecd_69ch_35db_per_sample.csv`, `ecd_69ch_35db_summary.json`, and `ecd_69ch_35db.png` in `validation_results/` using the publication settings. Review the results and rerun `python scripts/make_validation_manifest.py --write-canonical` to record the new checksums. |

Standard checksum verification also works with
`sha256sum -c validation_results/SHA256SUMS` on Linux or
`shasum -a 256 -c validation_results/SHA256SUMS` on macOS.

These checks do not establish clinical validity or equivalence to an
individualized head model. The source release ships compact validation
summaries, per-sample metrics, and publication figures, but not full generated
training datasets.

Every script in the table writes to a new location by default and leaves
`validation_results/` untouched. The `--write-canonical` option replaces the
archived result of a script and is meant for maintainers. The seven scripts
that have archived results accept it (`validate_inverse_mne.py`,
`validate_patch_forward_mne.py`, `validate_transfer_coefficients.py`,
`legendre_convergence.py`, `measure_split_spacing.py`, `train_baseline.py`,
`make_validation_figures.py`), as does `make_validation_manifest.py` for the
checksum list; the last row of the table shows one such command.

### Experiments reported in the paper

The protocols are in `docs/revision_experiments.md`. The per-window predictions
and summaries of the recorded runs are archived with the paper's evidence
deposit (see [Citation and license](#citation-and-license));
`scripts/make_revision_tables.py` rebuilds the experiment tables of the paper
from that archive. It recomputes the error statistics from the per-window
files, compares each with the recorded summary and stops at the first
difference; its description lists the numbers that it reads without
recomputation. It loads no model files and generates no EEG.

```bash
python scripts/run_split_study.py --output-dir validation_runs/split-smoke --smoke
python scripts/run_split_study.py --output-dir validation_runs/split-recorded
python scripts/plot_split_study.py validation_runs/split-recorded --output-dir validation_runs/split-figures
python scripts/run_training_count_control.py --split-run validation_runs/split-recorded --output-dir validation_runs/count-control
python scripts/plot_region_split.py --output-dir validation_runs/region-figure
python scripts/run_patch_split_study.py --output-dir validation_runs/patch-split-recorded
python scripts/run_shared_position_demo.py --output-dir validation_runs/shared-positions
python scripts/run_sphere_fit_convergence.py --output-dir validation_runs/sphere-fit --geometry path/to/evidence/head-adaptation-v11/geometry.json
python scripts/check_installed_sphere_fit.py --output-dir validation_runs/sphere-fit-installed --geometry path/to/evidence/head-adaptation-v11/geometry.json
python scripts/make_revision_tables.py --evidence path/to/evidence --output-dir validation_runs/revision-tables
```

Use a clean committed checkout for recorded runs; the scripts record the
commit and refuse a modified tree. The experiment drivers (`run_*.py`) also
have a `--smoke` option that exercises smaller datasets and fits and does not
reproduce the reported estimates; the plotting scripts and the table driver
read existing results and have none. The full split study uses three dataset seeds, two model seeds, two
policies, and fixed 500-window dipole subsets. The recorded Windows run took
about 41 minutes. Its raw HDF5 files, saved signal and noise components, lead
fields, and models occupy 15.1 GiB. Allow additional space for temporary
outputs and template inputs. Each study command requires a new output folder.

The transfer driver uses MNE-Python 1.12.1 and the fsaverage input files fetched
separately from MNE. Fetch them once before the study.

```bash
python -c "import mne; mne.datasets.fetch_fsaverage(subjects_dir='validation_runs/templates')"
python scripts/run_head_transfer.py --split-run validation_runs/split-smoke --subjects-dir validation_runs/templates --output validation_runs/transfer-smoke --smoke
python scripts/run_head_transfer.py --split-run validation_runs/split-recorded --subjects-dir validation_runs/templates --output validation_runs/transfer-gates --gates-only
python scripts/run_head_transfer.py --split-run validation_runs/split-recorded --subjects-dir validation_runs/templates --output validation_runs/transfer-recorded
python scripts/run_head_adaptation.py --split-run validation_runs/split-recorded --subjects-dir validation_runs/templates --transfer-run validation_runs/transfer-recorded --output validation_runs/adaptation-recorded
python scripts/plot_head_adaptation.py validation_runs/adaptation-recorded --output-dir validation_runs/adaptation-figure
python scripts/run_dipole_comparators_converged.py --split-run validation_runs/split-recorded --subjects-dir validation_runs/templates --recorded path/to/evidence --output-dir validation_runs/comparators
```

The first template download is substantial. The transfer run includes dipole
fits with the boundary-element model on 500 seed-862 windows, which took
roughly two hours on the recorded host. The adaptation run reuses those
recorded fits through `--transfer-run`, generates the sphere-variant datasets
itself, and deletes them once their features are extracted unless
`--keep-datasets` is given. The saved prediction CSVs reproduce the reported
medians without opening `.joblib` files. Those model files use Python pickle
serialization and are optional for numerical review. Load them only when their
source is trusted.

The transfer scripts pin MNE-Python 1.12.1 and use three of its internal
functions for the inside-surface test and the scalp projection, so they are a
record of the reported experiment, not a general interface.

### Generation benchmark

```bash
python scripts/benchmark_generation.py --output-dir validation_runs/generation --suite all --scratch-dir /path/with/40GiB
```

The suites are `modes` (point, multi-point and patch sources with one worker),
`workers` (point sources with the worker counts given by `--workers`),
`formats` (uncompressed HDF5, NPZ and MATLAB v7.3) and `default-scale` (the
unmodified default configuration, 360,000 windows, with one worker and with
each `--workers` value). Fixed-size cases generate `--n-train` training windows
plus one tenth each for validation and test, and are repeated `--repeats`
times; the report gives the median and the range. Run the benchmark on an
otherwise idle machine; each case records the free memory and CPU load it
started with. The memory statistic is the largest sampled sum of resident
memory over the generator and its worker processes, including shared pages in
each process, and can miss peaks between samples.

The recorded run is in `validation_results/generation_benchmark.json`. On that
host (AMD Ryzen 5 4600H with six cores, 15.4 GiB of memory, Windows 11, Python
3.12.16, NumPy 2.4.4, h5py 3.16.0 with HDF5 2.0.0):

| Case | Windows | Time | Peak memory | Output |
| --- | --- | --- | --- | --- |
| Point sources, HDF5, 1 worker | 12,000 | 76 s | 0.51 GiB | 0.60 GiB |
| Point sources, HDF5, 6 workers | 12,000 | 92 s | 1.38 GiB | 0.60 GiB |
| Multi-point sources, HDF5, 1 worker | 12,000 | 87 s | 0.53 GiB | 0.63 GiB |
| Patches, HDF5, 1 worker | 12,000 | 79 s | 0.53 GiB | 0.62 GiB |
| Point sources, uncompressed HDF5, 1 worker | 12,000 | 48 s | 0.50 GiB | 0.63 GiB |
| Point sources, NPZ, 1 worker | 12,000 | 73 s | 2.71 GiB | 0.60 GiB |
| Point sources, MATLAB v7.3, 1 worker | 12,000 | 85 s | 1.93 GiB | 0.56 GiB |
| Full default dataset, 1 worker | 360,000 | 26.0 min | 0.56 GiB | 18.0 GiB |
| Full default dataset, 6 workers | 360,000 | 19.6 min | 1.50 GiB | 18.0 GiB |

Between 17 and 21 s of every run is setup that does not depend on the number
of windows (the lead-field basis of 41,500 pool positions, or of 35,000 in
patch mode, and the background pool). Workers parallelize the synthesis of windows only; the parent process
writes the files, and a pool is started for each split, so workers pay off for
large datasets and cost time for small ones. Compression saves about 5% of
the size of float64 EEG and takes about a third of the single-worker time.

What the archived measurements show, and why:

- Worker processes parallelise only the synthesis of windows. Results return to
  the parent process, which does all writing, and one pool of workers is
  started for each split. For runs of about ten thousand windows a single
  worker is therefore fastest; workers pay off for large splits.
- gzip compression costs about as much time as synthesis and saves only a few
  percent on float64 EEG. Set `cfg.output.compress = False` when write speed
  matters more than those few percent.
- HDF5 streams, so its memory use does not grow with the dataset. NPZ and
  MATLAB v7.3 hold a complete split in memory before writing.

An interrupted generation run cannot be resumed. It publishes nothing: the
incomplete staging directory is removed at the next start and the run begins
again. In a fixed numerical environment, the same scientific configuration and
seeds reproduce the same canonical data, so a long dataset can also be
generated as several smaller datasets with different seeds.

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

Machine-readable citation metadata is in `CITATION.cff`. Every release is
archived on Zenodo under the concept DOI
[10.5281/zenodo.21763143](https://doi.org/10.5281/zenodo.21763143), which
always resolves to the latest version; cite the DOI of the version you used.
The per-window predictions and summaries behind the paper's experiments are
archived separately as its evidence deposit, cited in the paper's data
availability statement. SphericalEEGSim is
released under the [BSD 3-Clause License](LICENSE). The bundled third-party
montage templates retain their attributed notices in
`spherical_eeg_sim/montages/_templates/NOTICE`.
