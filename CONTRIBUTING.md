# Contributing to spherical_eeg_sim

Contributions are welcome, whether bug reports, new montages, additional source
or noise models, documentation, or validation experiments. This guide describes
how to set up a development environment and the conventions the project follows.

## Development setup

```bash
git clone https://github.com/Amjad862/spherical_eeg_sim.git
cd spherical_eeg_sim
pip install -e ".[all]"      # editable install with test and optional extras
pytest                        # run the full test suite
mypy spherical_eeg_sim       # verify the public inline type information
```

The core package depends on NumPy, SciPy, and h5py. MATLAB v7.3 output needs
`hdf5storage`; the MNE export path and inverse-validation script need `mne`
(and pandas for epoch metadata); the baseline benchmark needs `scikit-learn`;
figure scripts need `matplotlib`. These extras are imported lazily.

## Conventions

These conventions are what keep the package reproducible and trustworthy as a
benchmark generator. Please follow them in any contribution.

1. **Every scientific/model choice needs evidence.** No unexplained scientific
   numbers in code defaults or documentation. A default must
   either reproduce a peer-reviewed value or state an explicit derivation or
   benchmark scope grounded in peer-reviewed work; never imply that a paper
   prescribed an exact value when it did not. Verify full bibliographic metadata
   (authors, venue, volume, pages or article number, DOI) from an authoritative
   record rather than memory. Execution-only controls such as seed identifiers,
   write-buffer sizes, and worker counts are not physiological parameters; give
   them a deterministic engineering rationale and tests instead of attaching an
   irrelevant scientific citation.
2. **Empirical verification over assertion.** Any reported number — in a
   docstring, the docs, or a validation script — must come from an actual run,
   not an estimate.
3. **Byte-identical reproducibility is a hard requirement.** Output must be
   independent of `cfg.output.chunk_samples` and `cfg.output.n_workers`. Seeding
   is per sample, derived from `(sample_seed, split_id, global_sample_index)`;
   a null `sample_seed` resolves to `master_seed`.
   Any change that ties generated data to chunk boundaries or worker count is a
   correctness bug, not a tuning issue. New generation code must preserve this,
   and reproducibility tests must cover full splits, not just the first chunk.
4. **Keep code and public documentation in lockstep.** A change to configurable
   behaviour or a default should be reflected in `docs/parameters.md`, the
   `README`, examples, and citation metadata where relevant.
5. **Keep the evidence catalog internally consistent.** Changes to scientific
   claims or defaults must update `docs/references.bib`; do not add duplicate,
   orphaned, or unverified entries.
6. **Write current-state documentation.** Comments and documentation should
   explain the behavior, assumptions, and scientific or engineering rationale
   that users need to understand the present implementation.
7. **Read, verify, then edit.** Understand the current behaviour before changing
   it.

## Tests

Add tests for any new behaviour under `tests/`. Physics changes should be checked
against an independent reference (e.g. the forward model against MNE-Python's
independent sphere solver, as in `scripts/validate_inverse_mne.py`) rather than
by a same-code round trip, which does not constitute validation.

## Submitting changes

1. Open an issue describing the change before large work, so design can be
   discussed.
2. Create a feature branch, keep commits focused, and ensure `pytest` passes.
3. Update the affected public documentation and examples.
4. Open a pull request describing the motivation, the change, and how it was
   verified.

## Reporting bugs

Please include the package version, Python version, a minimal `Config` that
reproduces the issue, and the observed versus expected behaviour. Reproducibility
issues (output depending on chunk size or worker count) are treated as
high-priority correctness bugs.
