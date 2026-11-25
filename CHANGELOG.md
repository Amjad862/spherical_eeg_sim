# Changelog

## 1.1.0

### Source splitting

- Keep exact-position holdout as the default and add two selectable policies.
  `region` holds out rotated spherical sectors with a Euclidean clearance
  buffer between splits. `shared` makes a chosen fraction of the position pool
  eligible in every split.
- In patch mode the region policy admits a patch only where its whole footprint
  lies inside the sectors of its own split and contains no position of another
  split or of the buffer. Footprints of different splits are therefore disjoint
  and their active dipoles are at least the buffer apart. The diameter is drawn
  first and the centre from the positions admitted by the containment rule,
  so every split keeps the configured diameter distribution. The rule is
  conservative near sector corners. The planner rejects a layout when a split
  has no admitted centre for the largest configured patch.
- Report, in every dataset, the distances between split positions for eligible
  and for used positions, the availability of each source population, the
  observed coordinate reuse under the shared policy, and, in patch mode, the
  nearest-centre distances, the best footprint overlap, the smallest footprint
  gap and the containment verification.
- Provide `plan_position_split(cfg)`, which returns the position pool and the
  split assignment without computing lead fields or EEG.

### Command line

- The `spherical-eeg-sim` command no longer starts the full default run
  (360,000 windows; about 18.5 GiB of EEG before compression, 18.0 GiB on
  disk) when it is called without a size. A run
  starts when `--quickstart`, `--config`, a size override or the new
  `--default-run` option is given; otherwise the command prints the size of the
  default dataset and stops. Every run prints its planned size first.

### Validation and reproducibility

- Serialize generation and recovery for each output directory, so a second
  run cannot remove the active writer's staging files. Runs with different
  output directories can proceed independently.
- Buffer small HDF5 appends as fragments and join them once per storage block.
  This avoids repeatedly copying partially filled blocks and preserves the
  order in which the writer stores the data.

- Validation scripts write to new folders under `validation_runs/` by default.
  Replacing an archived result requires `--write-canonical`. The quick inverse
  check no longer overwrites the archived per-sample file.
- The manifest script verifies the archive by default, including the
  relationships between archived results.
- Reject a frequency band whose upper edge equals or exceeds the Nyquist
  frequency in the configuration, the signal generator and the band resolver.
- Add the experiment scripts of the accompanying paper: the split comparison,
  the training-location control, the patch split comparison, the shared-position
  demonstration, the boundary-element transfer test, the head-model adaptation
  experiment, the study of the Berg–Scherg parameter fit in MNE-Python, the
  repetition of the sphere-model dipole fits with a converged fit, a check of
  that fit for the installed version of MNE-Python, and the generation
  benchmark, with a driver that rebuilds the reported experiment tables from
  the evidence archive, recomputes the error statistics from the per-window
  files and compares them with the recorded summaries.
- Record the version of the code that actually ran in dataset provenance, also
  when another version of the package is installed.
- The HDF5 writer stores blocks of complete storage chunks in a fixed order
  and keeps the rows of an unfinished block in memory; the block size depends
  on `output.hdf5_chunk_rows` alone. Version 1.0.0 wrote each generation chunk
  as it arrived, so the layout of a file, and under HDF5 1.14 with compression
  also its size, could depend on `output.chunk_samples` although the arrays
  did not. This occurred with the default settings under HDF5 1.14.5 and under
  HDF5 2.0.0 (for example between `chunk_samples` 1000 and 500), with and
  without compression; the tests of 1.0.0 compared small chunk sizes only.
  Files are now byte-identical across `chunk_samples` and `n_workers` for any
  dataset shape, tested with HDF5 1.14.5 and 2.0.0. The arrays are unchanged.
- A dense ground-truth field must keep its dtype between appends to the HDF5
  writer; a different dtype is an error and is no longer cast silently. A
  ragged field without any value can be finalized.

### Recovery and compatibility

- Remove incomplete staging directories of interrupted runs at the next start,
  and keep recovery information when files remain locked on Windows.
- Clean up overwrite backups after publication without discarding the newly
  completed dataset.
- Generated numerical arrays are unchanged from 1.0.0 under the default split
  policy. A test compares small point, multi-point and patch datasets with
  arrays recorded from 1.0.0, and the ten datasets of the two comparisons with
  MNE-Python have identical arrays under both versions. New metadata and
  validation report fields change whole-directory checksums.
- Record a non-default split policy in standalone split files.
- Python 3.10 is now the minimum supported version. Continuous integration
  covers Python 3.10 to 3.13 on Linux and Python 3.12 on Windows, with a lint
  job over the package, the scripts, the tests and the examples.

### Documentation and figures

- Document the split policies, the command-line behaviour, every validation and
  experiment script, the generation benchmark and the limits of synthetic
  validation.
- Redraw the pipeline, topography and distribution figures at print width,
  with no text below 8 pt, and add vector versions of the publication figures.
- State that the statistics of the two comparisons with MNE-Python are not
  identical between platforms, with the values of a rerun on Windows, and
  that for the head models of this package the Berg–Scherg parameter fit of
  MNE-Python 1.12.1 with SciPy 1.17.1 ended at SciPy's evaluation limit before
  it had converged in the recorded runs on Windows, and that the dipole fits
  of MNE-Python are a second source of differences of this size.

## 1.0.0

Original public release.
