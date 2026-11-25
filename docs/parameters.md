# Parameter reference

This document explains every configuration field in `spherical_eeg_sim`,
its default value, what it does, and the literature backing the default.
Use it when deciding what to tune for your specific use case.

Evidence is reported at three levels. Tissue properties and physiological
quantities use peer-reviewed measurements directly. Benchmark ranges may be
derived explicitly from such measurements (for example, moment density times a
published active area), with the derivation stated rather than presenting the
range as an observed population distribution. Execution controls such as a
random-seed identifier, write-buffer size, and worker count are operational
parameters: they do not model physiology, are excluded from the canonical
scientific configuration where appropriate, and are justified by deterministic
tests rather than by attributing them to an unrelated neuroscience paper.

The configuration is organised into sub-configs, accessed as
`cfg.head`, `cfg.sources`, `cfg.dipole`, etc. Top-level fields
(`source_mode`, `electrode_file`, `output_dir`, `master_seed`, optional seed
domains) live directly on `cfg`.

---

## Top-level switches

### `cfg.electrode_file`
- **Type**: string (path to CSV, or a bundled-montage basename)
- **Default**: `'standard_1010_69.csv'`
- **What**: Electrode-position CSV, or the basename of a bundled montage.
  Required columns: `Name, x, y, z, R, theta, phi`. Coordinates in meters,
  `theta` and `phi` in degrees. The Cartesian and spherical entries must
  agree (the loader checks this and auto-detects between the standard
  convention and a swapped one where the theta/phi labels are interchanged).
  Bundled montages are placed on the model scalp sphere and are
  **radius-independent**: their positions are rescaled to `head.radii_m[-1]`
  at load time, so a montage is not tied to the radius its CSV is stored at.
  (A custom CSV whose radii are implausible for a head, e.g. supplied in
  millimeters, is rejected.)
- **When to change**: To use any other electrode layout. Bundled options:
  - Standard family (Oostenveld & Praamstra 2001; Jurcak et al. 2007):
    `standard_1020_19.csv`, `standard_1020_20.csv`, `standard_1010_32.csv`,
    `standard_1010_61.csv`, `standard_1010_69.csv` (~10-10),
    `standard_1005_336.csv` (full 10-05, high density).
  - Vendor nets, spherical approximations (`_spherical` suffix):
    `biosemi_16/32/64/128/256_spherical.csv`,
    `gsn_hydrocel_32/64/128/256/257_spherical.csv` (EGI/Philips).

  The nominal HydroCel-32 template contains 32 numbered sensors plus Cz, so its
  CSV has 33 recording channels. BioSemi Iz is retained; only LPA, RPA, and Nz
  fiducials are excluded. These counts are checked against the attributed
  templates in the test suite.

  See `spherical_eeg_sim/montages/` for the full set and
  `montages/_templates/NOTICE` for provenance and license attribution.

### `cfg.output_dir`
- **Type**: string (path)
- **Default**: `'output_eeg_dataset'`
- **What**: Dedicated canonical dataset directory. The pipeline generates into
  a sibling staging directory and publishes only the complete result by a
  directory rename. Initial publication is atomic; explicit replacement first
  moves the previous directory aside, installs the completed staging directory,
  and rolls back if installation fails.
  It refuses a non-empty target unless `cfg.output.overwrite=True`. Runtime
  timestamp, output path, worker count, generation chunk size, and wall time
  are written separately to `<output_dir>.runtime.json`. Stable interpreter,
  platform, package, and numerical-library versions are also written inside
  the canonical directory as `environment.json`.

### `cfg.master_seed`
- **Type**: int
- **Default**: `862`
- **What**: Master RNG seed and fallback for the two named seed domains below.
  When the named seeds are null, setup stages and per-sample generation derive
  their streams from this value.
  The same seed and scientific config produce byte-identical canonical output,
  and
  because seeding is per-sample, that output is **independent of
  `cfg.output.chunk_samples` and `cfg.output.n_workers`. Complete-directory
  SHA-256 tests cover HDF5, NPZ, and MATLAB v7.3 backends. The numeric seed is
  an arbitrary reproducibility identifier, not a scientific parameter.

### `cfg.position_seed` / `cfg.sample_seed`
- **Type**: optional non-negative integers; both default to `None`.
- **What**: A null value falls back to `master_seed`. `position_seed` controls
  the source pool, the `position` and `shared` assignments, and the background
  projection pool. The `region` assignment has its own `region_rotation_seed`.
  `sample_seed` controls per-window source/signal/noise streams, with each
  window derived from `(sample_seed, split_id, global_sample_index)` via
  `SeedSequence`. Separate domains support paired or factorial acquisition
  designs with the same position assignment and independently generated windows.
  Both values are canonical scientific provenance, and output remains independent
  of `cfg.output.chunk_samples` and `cfg.output.n_workers`.

### `cfg.source_mode`
- **Type**: string
- **Default**: `'point'`
- **Options**: `'point' | 'multi_point' | 'patch'`
- **What**: What each sample contains.
  - `'point'`: single dipole per sample. Ground truth is a single
    `(position, orientation, magnitude)` triple.
  - `'multi_point'`: 1..N dipoles per sample, padded ground truth with complete
    per-source time courses and their spectral/blending construction.
  - `'patch'`: spatially-extended cortical patch with finite support and a
    configurable spatial weighting profile.
    Like ConvDip's patch generator and DeepSIF's distributed sources.

---

## Head model: `cfg.head`

### `radii_m`
- **Default**: `[0.080, 0.081, 0.086, 0.092]` (brain, CSF, skull, scalp)
- **What**: Layer radii in meters, innermost to outermost.
- **Provenance**: This is a declared benchmark geometry close to the
  80/82/87/92 mm four-shell proportions used by Srinivasan et al. (1998), with
  a 1 mm CSF shell and an 86 mm outer-skull boundary. The exact radii are
  software defaults, not anatomical estimates for a particular individual.
- **When to change**: To test sensitivity to head size or shell thickness. The
  solver accepts exactly four interfaces; the three-tissue preset removes the
  CSF conductivity discontinuity while retaining four numerical radii.

### `conductivities_S_per_m`
- **Default**: `[0.330, 1.790, 0.0042, 0.330]`
- **What**: Conductivities in S/m, same order as `radii_m`. The brain-to-skull
  conductivity ratio is `0.330 / 0.0042` ≈ 78.6 here, near the upper end of the
  reported range.
- **Citation**: Geddes & Baker (1967) for brain and scalp conductivity near
  0.33 S/m; Baumann et al. (1997) for cerebrospinal fluid near 1.79 S/m; Rush &
  Driscoll (1968) for a brain-to-skull ratio near 80.
- **When to change**: Skull conductivity is the most uncertain head-model
  parameter, with reported brain-to-skull ratios spanning roughly 8:1 to 80:1
  (Gonçalves et al., 2003; Oostendorp et al., 2000). Treat it as a sensitivity
  variable rather than an established constant.
- **Presets**: `HeadModelConfig` provides ready-made head models for the two
  common ablations, assignable to `cfg.head`:
  - `HeadModelConfig.three_shell()` — a brain/skull/scalp three-shell model
    (CSF conductivity collapsed onto the brain value, so there is no CSF
    conductivity jump; Rush & Driscoll 1968).
  - `HeadModelConfig.with_skull_ratio(r)` — the default four-shell model with
    skull conductivity set to `0.330 / r`, for a brain-to-skull ratio sweep.
  - `HeadModelConfig.four_shell_default()` — the default, named explicitly.

### `n_legendre`
- **Default**: `80`
- **What**: Maximum order of the Legendre expansion; it controls truncation
  error and computation cost, with superficial sources converging slowest.
- **Citation**: de Munck & Peters (1993) for the series formulation.
- **Convergence**: The supported float64 range is 40 through 120.
  `scripts/legendre_convergence.py` measures the relative topography error
  against an arbitrary-precision reference (mpmath at 120 decimal digits, summed
  to order 400), not against another float64 order. Before reporting errors, the
  script requires its order-200 and order-400 reference topographies to agree.
  A high float64 order is not a converged reference but the point at which
  round-off begins to overtake truncation. At the default order of 80 the
  archived measurement gives a
  relative topography error of 1.22e-05 for a 79.5 mm radial source,
  8.84e-07 for a 79.5 mm tangential source, and 3.36e-14 at 55 mm; the
  superficial cases improve at order 100 and worsen again at 120, which is
  what fixes the upper boundary.
- **Accuracy is not monotone in this setting**: because truncation error falls
  with order while round-off in the transfer recurrence grows, the total error
  has an interior minimum instead of decreasing throughout the supported range.
  For the default geometry that minimum is near order 100: the archived
  superficial radial error is 7.197e-07 at order 100 and 5.017e-06 at order 120,
  so raising the order to the top of the range makes the most demanding case
  about seven times worse. `compute_transfer_coefficients` therefore emits a
  `RuntimeWarning` above order 100. The crossover moves with the radii and
  conductivities, so the warning is guidance rather than a limit; rerun
  `scripts/legendre_convergence.py` to locate it for a non-default head model.
- **Coefficient conditioning**: truncation error and round-off error move in
  opposite directions with order. The scalp potential is protected from the
  round-off tail because order n reaches the scalp weighted by
  `(r_source / R_scalp) ** n`, but the transfer coefficients themselves are not:
  the recurrence closes the layered system through a subtraction whose two terms
  grow like `(R_scalp / R_brain) ** (2n)` while their difference does not, so at
  high order a coefficient can lose its sign or cancel to zero. With the default
  geometry that begins near order 91. `compute_transfer_coefficients` raises a
  `RuntimeWarning` naming the first affected order, and
  `estimate_transfer_coefficient_accuracy(radii, conductivities, n_legendre)`
  returns a per-order conditioning indicator. It is useful for locating
  cancellation-prone orders but is not a rigorous error bound. Use a lower
  `n_legendre` if the coefficients are needed as reference values; the default
  80 is below the warning threshold.

### `truncation_tail_tolerance`
- **Default**: `0.05`
- **What**: Maximum accepted value of the dimensionless geometric-tail
  indicator `q ** (n_legendre + 1) / (1 - q)`, where `q` is the largest
  configured signal or background-dipole radius divided by the scalp radius.
  Configuration validation rejects a source domain whose geometric weighting
  decays too slowly at the selected expansion order.
- **Interpretation**: This indicator is a conservative domain check, not a
  bound on voltage or topography error. It admits the complete shipped source
  domain at every supported order. For a non-default geometry, run
  `scripts/legendre_convergence.py --config my_config.json` before deliberately
  relaxing the tolerance. The audit script still evaluates a configuration
  that fails only this publication gate; data generation remains blocked until
  the declared tolerance or source domain is changed.

---

## Sources: `cfg.sources`

### `cortical_radius_range_mm`
- **Default**: `[70.5, 79.5]`
- **What**: Radial range for cortical sources, relative to brain center.
  At brain radius 80 mm, this corresponds to 0.5-9.5 mm depth from the
  brain surface.
- **Citation**: The shell is a controlled spherical source space, not a literal
  cortical-thickness model. Spherical and surface-constrained ESI source spaces
  are discussed by de Munck & Peters (1993) and Dale & Sereno (1993); the exact
  radial span is a benchmark range and must not be interpreted anatomically.
- **Radial distribution**: Radius is uniform within this interval, giving equal
  probability to radial depth rather than uniform probability per unit volume.
  This is a declared benchmark distribution; it is not a cortical-volume model.

### `subcortical_radius_range_mm`
- **Default**: `[30.0, 65.0]`
- **What**: Radial range for the model's deep-source population. It is a
  spherical benchmark volume, not a segmentation or coordinate claim for the
  thalamus, basal ganglia, or hippocampus.
- **Citation and scope**: Pizzo et al. (2019) and Seeber et al. (2019) motivate
  including deep generators; they do not establish these spherical radial
  bounds. The exact interval is a declared, configurable benchmark choice.
- **Overlap with the cortical shell**: This interval is validated independently
  of `cortical_radius_range_mm`, so the two may overlap. Overlap is permitted
  because `population` is a declared sampling label rather than an anatomical
  assignment: a position drawn from the subcortical pool is labelled
  subcortical even if its radius also lies inside the cortical shell, and it
  then takes the subcortical magnitude range. Keep the intervals disjoint
  unless the two populations are meant to be distinguished by their magnitude
  and sampling rules rather than by depth.

### `subcortical_fraction`
- **Default**: `0.15`
- **What**: Fraction of training samples drawn from subcortical positions.
  This is a **simulation-design choice**, not an anatomical prevalence claim;
  it deliberately gives the deep population training exposure.
- **Citation**: The choice is supported by:
  1. Subcortical localization is the harder problem.
  2. HD-EEG can recover subcortical signals (Seeber et al., 2019;
     Pizzo et al., 2019).
  3. Precedent in deep-learning ESI work targeting subcortex (Yokoyama
     et al., 2024) uses simulated cortical and subcortical training sources.
- **Range**: 0.0 (cortex-only) to 1.0 (subcortex-only).
- **Split coupling**: draws use the population's eligible positions in each
  split. The default assignment stratifies cortical and subcortical positions
  separately. A buffered region assignment raises if it leaves any split
  without an enabled population. An enabled population needs at least three
  pool positions for separate assignments, or one for a fully shared pool.

### `n_cortical_positions` / `n_subcortical_positions`
- **Defaults**: `35_000` / `6_500`
- **What**: Number of unique positions in each population. Each
  position becomes one row in the lead-field tensor. Different samples
  using the same position differ in orientation, magnitude, signal,
  and noise.
- **Memory cost**: For float64, the basis lead field uses
  `n_positions * n_channels * 3 * 8` bytes. Compute the value for the selected
  montage and position counts rather than relying on a fixed estimate.
- **Basis and scope**: Quasi-uniform spherical discretization is supported by
  Gonzalez (2010), and large simulated source spaces are used in supervised
  ESI (Hecker et al., 2021; Sun et al., 2022; Reynaud et al., 2024). Those
  publications do not prescribe 35,000 or 6,500. The exact counts are declared
  benchmark-resolution choices; convergence or task sensitivity should be
  measured when they are changed.

### `cortical_sampling`
- **Default**: `'fibonacci'`
- **Options**: `'fibonacci' | 'uniform_random'`
- **What**: How positions are placed on the cortical shell. Fibonacci
  lattice gives quasi-uniform angular coverage; uniform_random is less
  even at the same point count.

### `subcortical_sampling`
- **Default**: `'uniform_volume'`
- **Options**: `'uniform_volume' | 'fibonacci_shells'`
- **What**: How positions fill the subcortical volume. Volume-uniform
  uses inverse-CDF sampling on `r ~ r^2 dr`. Fibonacci shells stack
  Fibonacci lattices at evenly spaced radii. The shell count is derived as
  `max(3, floor(cuberoot(n_subcortical_positions)))`; the cube-root rule and
  minimum of three are declared discretization choices, while the angular
  lattice follows Gonzalez (2010).

---

## Dipole: `cfg.dipole`

### `orientation_mode`
- **Default**: `'uniform_s2'`
- **Options**: `'uniform_s2' | 'codebook' | 'cortex_normal'`
- **What**: How dipole orientations are sampled.
  - `uniform_s2`: uniform on the unit sphere from normalized isotropic
    Gaussian draws.
    Most general; allows any orientation.
  - `codebook`: each sample is purely radial or purely tangential. Tangential
    directions are drawn from a fixed finite golden-angle codebook controlled
    by `codebook_tangential_directions`. One third are radial and two thirds
    tangential, assigning equal aggregate probability to the three local
    moment-component dimensions; this is a declared codebook composition, not
    a physiological orientation ratio.
  - `cortex_normal`: aligned with the local radial direction with a random
    inward/outward sign. Cheapest in training samples per orientation.
- **Citation**: Cortex-normal constraints are common for folded cortical
  source spaces (Dale & Sereno, 1993) and appear in supervised ESI generators
  such as ConvDip (Hecker et al., 2021). Cortical/subcortical learning with
  unconstrained components is represented by Yokoyama et al. (2024). Uniform
  sphere sampling is a robustness-oriented benchmark choice, not a claim about
  the orientation distribution of neural generators. Golden-angle coverage is
  motivated by Fibonacci spherical sampling (Gonzalez, 2010); that work does
  not prescribe the radial/tangential mixture.

### `codebook_tangential_directions`
- **Default**: `256`
- **What**: Number of fixed azimuths in the local tangential codebook used by
  `orientation_mode='codebook'`. Indices are uniformly sampled, and azimuths
  follow the golden-angle sequence without a per-call random rotation.

### `cortical_magnitude_range_nAm`
- **Default**: `[5.0, 300.0]`
- **What**: Cortical dipole moment range in nA·m, log-uniform.
- **Citation and derivation**: Hämäläinen et al. (1993) give approximately
  10 nA·m as a typical focal evoked-response moment. Murakami & Okada (2015)
  report 0.16–0.77 nA·m/mm² for human neocortex and an approximate 1–2
  nA·m/mm² maximum across structures and species. Goldenholz et al. (2009)
  analyze 3 and 8 cm² extended cortical sources. Density multiplied by those
  published areas spans tens to several hundred nA·m (human-neocortex values
  alone give 48–616 nA·m). The configured 5–300 nA·m interval is therefore a
  broad benchmark spanning below-focal stress cases through extended-source
  moments; it is not a reported empirical ECD distribution.

### `subcortical_magnitude_range_nAm`
- **Default**: `[30.0, 500.0]`
- **What**: Subcortical dipole moment range in nA·m, log-uniform.
- **Citation and scope**: Attal & Schwartz (2013) motivate accounting for
  deep-source attenuation. The 30–500 nA·m benchmark stays within the
  moment-density-times-extended-area scale derived above and emphasizes
  detectable deep examples. It is a simulation stress range, not evidence that
  deep structures have higher physiological moment density (Murakami & Okada,
  2015).

### `magnitude_distribution`
- **Default**: `'log_uniform'`
- **Options**: `'log_uniform' | 'uniform' | 'fixed'`
- **What**: Distribution within the magnitude range. The cited literature
  guides the configurable range; log-uniform sampling is a benchmark choice
  that spreads examples across that range and is not presented as an empirical
  dipole-moment distribution (Hämäläinen et al., 1993; Murakami & Okada, 2015).

---

## Multi-point: `cfg.multi_point` (only used when `source_mode='multi_point'`)

### `n_sources_min` / `n_sources_max`
- **Defaults**: `1` / `5`
- **What**: Number of dipoles per sample is uniform from this range.
- **Citation**: ConvDip uses 1-5; Pantazis & Adler (2021) train one
  model per fixed Q ∈ {1, 2, 3}.

### `min_separation_mm`
- **Default**: `20.0`
- **What**: Minimum 3D Euclidean separation between any two dipoles in
  the same sample. Prevents degenerate near-colocated configurations.
  Rejection sampling enforces this. Under the default
  `separation_failure_policy='raise'`, exhausting the attempt budget aborts with
  the sample/source index; a published corpus therefore cannot violate its
  advertised separation. Exploratory runs may explicitly select
  `accept_and_flag`, which then accepts one further draw from the pool
  positions not yet used in the sample and writes `separation_violated=true`.
- **Feasibility check**: configuration validation rejects a separation larger
  than the diameter of the configured source space, since no pair of positions
  could ever satisfy it. That check runs before the position pool and lead-field
  tensor are built, so an unsatisfiable setting fails immediately rather than
  part-way into a run. A separation that is merely difficult rather than
  impossible is still resolved by the attempt budget above.
- **Basis and scope**: Multi-source separation is motivated by simultaneous-
  source ESI benchmarks (Hecker et al., 2021; Pantazis & Adler, 2021). The
  exact 20 mm guard is a declared algorithmic setting,
  not physiological separation estimates; the measured violation rate is
  reported.

### `max_separation_attempts`
- **Default**: `100`
- **What**: Maximum candidate draws for each additional dipole before the last
  draw either raises or is accepted and flagged according to
  `separation_failure_policy`.
- **Basis and scope**: Rejection sampling implements the multi-source separation
  benchmark motivated above. One hundred is a declared compute-versus-acceptance
  budget, not a scientific constant; the realized violation count and rate are
  emitted so adequacy is measured for each configuration.

### `time_coupling`
- **Default**: `0.0`
- **Range**: `[0.0, 1.0]`
- **What**: Temporal coupling between additional dipoles and the primary
  dipole. This is a linear **blend weight**, not a Pearson correlation
  coefficient. Each additional dipole's time course is mixed toward the
  primary as `coupling*primary + (1-coupling)*independent`, then
  renormalized to unit variance.
  - 0.0 = additional dipoles use independently generated time courses
  - 1.0 = additional dipoles reuse the primary course (fully coherent)
  - Independently generated finite windows need not have zero sample
    correlation. If their Pearson correlation is `rho`, the blend's
    correlation with the primary is
    `(c + (1-c)*rho) / sqrt(c^2 + (1-c)^2 + 2*c*(1-c)*rho)`.
    Thus the blend weight is not itself the realized correlation. The stored
    base and final courses let consumers measure the realized value exactly.
- **Basis and scope**: Independent and coherent simultaneous sources are
  relevant ESI test cases (Pantazis & Adler, 2021). The default zero blend is
  a declared independent-course baseline, not a population estimate.
- **Serialized construction**: The final padded `source_time_courses` and every
  base band, frequency, phase, component amplitude, component count, blend
  weight, and pre-normalization blend standard deviation are stored. Unused
  slots use documented padding sentinels.

### `time_coupling_values`
- **Default**: `None`
- **What**: Optional non-empty list of blend weights in `[0, 1]`. When present,
  one value is sampled uniformly and independently for every window and stored
  in `ground_truth/time_coupling`; it supersedes the scalar `time_coupling`.
  This supports factorial designs in which temporal coupling varies
  independently of acquisition settings.

### `separation_failure_policy`
- **Default**: `'raise'`
- **Options**: `'raise' | 'accept_and_flag'`
- **What**: Whether rejection-budget exhaustion is a hard corpus error or an
  explicitly flagged exploratory fallback.

---

## Patch: `cfg.patch` (only used when `source_mode='patch'`)

Patch mode is cortical-only. It does not construct a subcortical position pool,
so `n_subcortical_positions` and `subcortical_fraction` have no effect in this
mode. The canonical source-pool metadata records zero subcortical positions.

### `diameter_range_mm`
- **Default**: `[10.0, 60.0]`
- **What**: Patch diameter range. ConvDip used 21-58 mm; the benchmark range is
  widened to include smaller supports. The exact default is a simulation range,
  not an anatomical prevalence claim (Hecker et al., 2021).

### `falloff_profile`
- **Default**: `'gaussian'`
- **Options**: `'gaussian' | 'uniform' | 'cosine'`
- **What**: How activation decays from patch center.
  - `gaussian`: Gaussian decay with σ = diameter/4, truncated at the configured
    patch radius. Its unnormalized boundary value is `exp(-2)`, followed by a
    hard zero outside; it is therefore not continuous across the boundary.
  - `uniform`: step function (within = 1, outside = 0)
  - `cosine`: cosine taper (smooth, finite support)
- **Basis and scope**: Distributed supports follow patch-based ESI simulation
  precedent (Hecker et al., 2021). The truncated Gaussian with
  `sigma = diameter/4` is a declared benchmark profile, not an empirical
  cortical activation kernel.

### `dipoles_per_mm2`
- **Default**: `1.0`
- **What**: Target density of within-patch dipoles per mm² of patch area. It
  sets an upper bound on how many pool positions are recruited into a patch of
  a given diameter (`ceil(dipoles_per_mm2 * patch_area_mm2)`).
- **Interaction with the position pool**: the *actual* within-patch dipoles are
  whichever split-local cortical pool positions fall within the configured
  **great-circle surface radius**, so the realized density is
  governed by the cortical pool density (`n_cortical_positions` over the
  cortical shell). Raise
  `n_cortical_positions` for genuinely denser patches; raising
  `dipoles_per_mm2` alone cannot exceed the available pool positions. Ground
  truth records the active count, global source-pool indices, and signed radial
  dipole moments in padding-free CSR row encoding. A `source_pool.*` file maps
  those indices to coordinates and split labels. Patch mode assigns equal cortical
  position counts to the three support meshes so their density is independent
  of the requested sample counts.
- **Sizing the pool**: each of the three support meshes holds about
  `n_cortical_positions / 3` positions spread over a shell of area
  `4 * pi * r_mean ** 2`, so the expected active count for a patch of diameter
  `D` is roughly `(n_cortical_positions / 3) * (D / (4 * r_mean)) ** 2`. At the
  default 35,000 positions and a 75 mm mean radius that is about 13 dipoles for
  a 10 mm patch and about 470 for a 60 mm patch, so the 1.0 per mm² cap never
  binds; the pool is the constraint at every practical setting. Below roughly
  10 dipoles in the smallest configured patch the activation is closer to a
  small cluster of point sources than a distributed source, so `cfg.validate()`
  raises a `RuntimeWarning` reporting the realized density and the
  `n_cortical_positions` needed. The realized support of every sample is stored
  in `ground_truth/n_active_per_patch`.
- **Geometry**: Support and falloff use angular separation multiplied by the
  selected center's radius. Active pool points retain their own radii from
  `cortical_radius_range_mm`; a patch is therefore an angular support over the
  configured superficial shell, not a constant-depth anatomical surface.
- **Basis and scope**: Distributed-patch simulation follows the benchmark class
  used by Hecker et al. (2021). The default `1.0` is a computational support cap,
  not a measured physiological dipole density; realized support is serialized
  and therefore directly auditable.

---

### `report_overlap`

Defaults to `True`. After generation, `validation_report.json` summarizes
the nearest-centre distances and best spherical-cap intersection over union
across splits. The caps use the saved centre and diameter and describe the
nominal continuous footprints. Very large workloads can use a deterministic
subset of query patches. Each scored query searches the full reference split.
Elapsed time is recorded in the runtime sidecar.

## Signal: `cfg.signal`

### `sampling_freq_hz`
- **Default**: `500.0`
- **What**: Sampling rate in Hz. The Nyquist frequency limits the
  highest band you can model. The value is a common EEG acquisition rate and
  supports the configured 45 Hz upper edge (Babiloni et al., 2020).

### `window_samples`
- **Default**: `100`
- **What**: Number of time samples per window. At 500 Hz this is 200 ms.
- **Limitation**: Delta-band oscillations (0.5-4 Hz) fit poorly in
  200 ms (0.1 to 0.8 cycles per window). For better delta coverage,
  increase window_samples.
- **Basis**: The compact default is useful for quick generation and model
  prototyping; the explicit resolvability policy below prevents the duration
  from silently contradicting a low-frequency label.

### `n_components_min` / `n_components_max`
- **Defaults**: `3` / `3`
- **What**: Number of sinusoidal components per sample, uniform from
  this range. Default fixes at 3 for richer within-window patterns.
  Set both to 1 for a single-sinusoid model.
- **Basis**: Published EEG simulation frameworks motivate composite oscillatory
  source signals (Krol et al., 2018; Barzegaran et al., 2019). Three components
  are a declared benchmark-complexity choice rather than a value prescribed by
  those papers; the final course and all component parameters are serialized,
  so alternative counts remain auditable.

### `bands_hz`
- **Default**: IFCN clinical neurophysiology bands:
  - `delta`: [0.5, 4.0] Hz
  - `theta`: [4.0, 8.0] Hz
  - `alpha`: [8.0, 13.0] Hz
  - `beta`: [14.0, 30.0] Hz
  - `gamma`: [30.0, 45.0] Hz
- **Citation**: Babiloni et al. (2020) IFCN recommendations for
  standardised band definitions in clinical and research EEG.

### `band_order`
- **Default**: `['delta', 'theta', 'alpha', 'beta', 'gamma']`
- **What**: Explicit semantic order used by `band_idx` and
  `source_base_band_idx`. It must contain every `bands_hz` key exactly once.
  Serializing the order separately prevents JSON key sorting from remapping
  integer labels when a manifest is loaded as a new configuration.

### `band_probabilities`
- **Default**: alpha-enriched categorical sampling probabilities:
  - `delta`: 0.20, `theta`: 0.15, `alpha`: 0.35, `beta`: 0.20, `gamma`: 0.10
- **Citation and scope**: Buzsáki (2006) and Babiloni et al. (2020) support the
  rhythm categories, not this probability vector. These values are
  probabilities of choosing a generating band, not empirical proportions of
  total EEG power. The alpha enrichment is a transparent benchmark choice.
- **Normalisation**: The values are relative weights. They are divided by their
  own sum before sampling, so a vector that does not add to one is accepted and
  rescaled rather than rejected. After the resolvability policy has run, the
  weights of any dropped band are redistributed over the surviving bands, and
  the realised vector is written to
  `metadata/band_resolvability/effective_band_probabilities` in every split.
- **When to change**: Use target-protocol literature or measured pilot data to
  define task-specific generating-band probabilities.

### `component_amplitude_jitter`
- **Default**: `True`
- **What**: If True, components have random Dirichlet-distributed
  relative amplitudes (sum normalised to 1). If False, all components
  have equal amplitude. The realized amplitudes are stored with frequency,
  phase, component count, and the final source time course.

### `component_dirichlet_concentration`
- **Default**: `1.0`
- **What**: Positive symmetric Dirichlet concentration used for relative
  component amplitudes when `component_amplitude_jitter=True`. One produces a
  uniform distribution over the amplitude simplex; smaller positive values
  favor sparse weights and larger values favor similar weights.
- **Basis and scope**: Composite oscillatory simulation is motivated by Krol et
  al. (2018) and Barzegaran et al. (2019). The Dirichlet family and default
  concentration are declared benchmark choices, not empirical component-power
  distributions. Every realized weight is serialized.

### `min_band_cycles` / `low_freq_band_policy`
- **Defaults**: `1.0` / `'restrict'`
- **What**: A short analysis window cannot resolve a full cycle of the lowest
  bands. A band's low edge completes `f_low * window_samples / sampling_freq`
  cycles in the window; at the default 200 ms window the delta band (0.5 Hz)
  spans only ~0.1 cycle and theta (4 Hz) ~0.8 cycle, so those "bands" are closer
  to a slowly-varying offset than a resolved oscillation.
  - `min_band_cycles` is the resolvability threshold (cycles of the low edge).
  - `low_freq_band_policy='restrict'` (default) **drops** under-resolved bands
    and redistributes their probability over the resolvable bands, so every
    generated band actually oscillates within the window. The decision (which
    bands were dropped, the per-band cycle counts, the effective probabilities)
    is logged and written to the run manifest and each split's metadata.
    A configuration with no surviving band is rejected rather than silently
    retaining labels that violate the requested restriction.
  - `low_freq_band_policy='keep'` keeps all bands unchanged but logs a warning
    and records in metadata that under-resolved band labels denote the
    **dominant spectral component of the generating process**, not a resolved
    in-window oscillation.
- **Guidance**: lengthen `window_samples` if you need the low bands resolved;
  increasing sampling rate without adding samples shortens elapsed time and
  cannot improve low-frequency resolution. Lower `min_band_cycles` to keep
  marginally-resolved bands; or set `'keep'` if you deliberately want low-frequency,
  near-DC content with the caveat above.

### `zero_mean_signal`
- **Default**: `True`
- **What**: Removes the per-sample temporal mean from each source time course
  before scaling, so each window is a zero-mean fluctuation rather than a
  fluctuation riding on a DC offset. This matters most for the lowest bands,
  whose short-window segments otherwise carry a large DC term that survives
  unit-variance normalization (and would propagate to the scalp EEG and into the
  SNR power). The default is consistent with standard EEG high-pass handling
  (Babiloni et al., 2020). Set False to preserve the raw (possibly DC-heavy)
  sinusoid sum.

---

## Noise: `cfg.noise`

### `noise_model`
- **Default**: `'absolute_floor'`
- **What**: Selects how the mixed noise is scaled relative to the signal.
  - `'absolute_floor'` (default) — the combined noise is scaled to a **fixed
    physical RMS** (`absolute_noise_rms_uV`) on the raw sensor montage,
    independent of the signal. The per-sample SNR is **not** prescribed; it
    *emerges* from forward gain and source magnitude. Absolute voltage scale is
    retained; `snr_db_range` is unused. Source depth is stored explicitly and
    should be analyzed while controlling magnitude and orientation.
  - `'snr_target'` — the noise is scaled so each sample hits a target SNR drawn
    from `snr_db_range` on the `snr_definition` montage. The referenced sample
    and optional components remain in physical volts unless the independent
    `output.eeg_normalization` transform is enabled (target-SNR precedent: Sun
    et al., 2022).
- **Why it matters**: Use `absolute_floor` when physical input scale belongs in
  the noise definition. Use `snr_target` when SNR should be controlled. Choose
  output normalization separately according to whether absolute scale belongs
  in the learning task.

### `absolute_noise_rms_uV`
- **Default**: `1.00` µV
- **What**: The fixed combined-noise RMS in microvolts, applied on the raw
  scalp montage. Used only when `noise_model == 'absolute_floor'`.
- **Rationale**: Goldenholz et al. (2009) and Piastra et al. (2021) use a
  10 µV² spontaneous-baseline variance, corresponding to approximately
  3.16 µV RMS. The package uses 1.00 µV RMS by default so its broad default
  source and montage settings retain a more useful emergent-SNR range; this is
  a declared benchmark choice, not a literature-prescribed noise estimate.
  Set `3.16` explicitly for the reference calibration. At fixed signal power,
  `20 log10(3.16 / 1.00) = 9.99 dB`, so the reference setting reduces SNR by
  approximately 10 dB relative to the package default.

### `n_background_sources`
- **Default**: `10`
- **What**: Number of background dipoles for the source-projected background
  proxy. Higher → more isotropic noise; lower → more low-rank
  spatial structure.
- **Basis and scope**: Spatially correlated source-generated background models
  are motivated by de Munck et al. (2002), Bijma et al. (2003), and de Munck et
  al. (2008). The exact count of ten is a declared finite-rank benchmark choice,
  not an estimate of the number of concurrent physiological generators.

### `background_pool_size`
- **Default**: `2000`
- **What**: Number of distinct random spherical source projections from which
  each sample selects `n_background_sources` without replacement. It must be
  at least `n_background_sources`; increasing it increases the diversity of
  available spatial background patterns and the one-time forward computation.
- **Basis and scope**: Source-generated correlated background is motivated by
  de Munck et al. (2002) and Bijma et al. (2003). The exact pool size is a
  declared Monte Carlo support choice, not a physiological generator count;
  exposing it prevents an unrecorded implementation constant from affecting
  the dataset.

### `background_radius_range_mm`
- **Default**: `[70.5, 79.5]`
- **What**: Radial interval for the spherical dipoles whose projections form
  the source-background pool. It is independent of the task-source population
  ranges, so a source-space sensitivity experiment does not silently redefine
  the noise model.
- **Basis and scope**: Cortical generators dominate conventional EEG source
  modeling (Hämäläinen et al., 1993), and source-projected correlated background
  is motivated by de Munck et al. (2002). The exact interval is a declared
  superficial spherical benchmark range, not an anatomical background-source
  distribution.

### `background_amplitude_concentration`
- **Default**: `1.0`
- **What**: Positive symmetric Dirichlet concentration for the relative weights
  of the selected source-projected background dipoles. One is uniform over the
  simplex; changing it alters the effective spatial rank per sample.
- **Basis and scope**: The cited correlated-background literature motivates the
  model class, not this amplitude distribution. The Dirichlet family and unit
  concentration are declared benchmark choices rather than empirical
  generator-strength statistics.

### `mix_physiological` / `mix_pink` / `mix_white`
- **Defaults**: `0.50` / `0.35` / `0.15`
- **What**: Intended relative-variance weights of the three noise components.
  They are normalized internally and their square roots multiply the
  standardized components. Finite-window cross-covariances mean these are not
  exact realized variance shares; the combined trace is standardized afterward.
- **Rationale**: Published work motivates correlated background activity and
  colored spectra (de Munck et al., 2002; Bijma et al., 2003; de Munck et al.,
  2008). The exact 50/35/15 weights are a declared benchmark mixture, not
  an empirical decomposition of all EEG recordings. The independent pink and
  white terms are spectral/statistical proxies rather than assignments to
  particular instrument mechanisms.

### `snr_db_range`
- **Default**: `[0.0, 20.0]`
- **Applies when**: `noise_model == 'snr_target'` (ignored under `absolute_floor`).
- **What**: Per-sample target SNR drawn uniformly from this range. SNR is
  10*log10(signal_power / noise_power), averaged across channels and time.
  The target is achieved on the montage chosen by `snr_definition` (below);
  the achieved SNR on *both* the raw and the referenced montage is recorded
  in ground truth as `snr_db_raw` and `snr_db_referenced`.
- **Basis**: The interval is a controlled synthetic benchmark range, not a
  claim that it spans all clinical recordings (Krol et al., 2018; Sun et al.,
  2022).

### `snr_definition`
- **Default**: `'referenced'`
- **What**: Which montage the reported `snr_db` refers to.
  - `'referenced'` — `snr_db` refers to the **final, referenced** `/eeg`
    (the data a model trains on). Recommended, because real EEG is referenced.
  - `'raw'` — `snr_db` refers to the raw, pre-reference forward potentials.
  Under `snr_target` this is the montage on which the target SNR is achieved.
  Under `absolute_floor` the fixed noise floor is always imposed on the **raw**
  montage, and this setting only selects which montage's *emergent* SNR is
  reported as `snr_db`.
- **Why it matters**: referencing redistributes signal and noise power
  unequally across channels, so the SNR of the referenced `/eeg` differs from
  the SNR of the raw montage in general. This setting decides which one equals
  `snr_db`. When `reference_mode='none'` the two
  montages coincide and this setting has no effect. Either way, `snr_db_raw`
  and `snr_db_referenced` are both stored so you always know both.

### `background_freq_alpha`
- **Default**: `1.0`
- **What**: Spectral exponent of background noise (1/f^α). α=1 is pink noise;
  other exponents are configurable and should be justified for the target data.
- **Citation**: Donoghue et al. (2020) on the 1/f component of EEG.

---

## Split: `cfg.split`

### `policy`

- `position` is the default. Each source position has exactly one split.
  Patches use equally sized cortical support meshes across splits. This
  policy prevents exact index reuse across splits and can leave nearby
  positions or overlapping continuous patch footprints.
- `region` is available in every source mode. Set `region_count`
  to an integer of at least three, `buffer_mm` to a finite non-negative
  Euclidean clearance, and `region_rotation_seed` to a non-negative integer.
  The configuration must leave at least one sector for each split. Whole
  rotated spherical Voronoi sectors are assigned with latitude coverage.
  Test positions stay intact, validation is buffered from test, and training
  is buffered from both across both populations. Generation stops if a split
  loses every position of an enabled population.

  In patch mode the policy also contains every patch in its own split. A
  patch is centred only where its whole footprint, the cap of directions
  within `diameter / 2` of the centre measured on the centre's radius, lies
  inside the sectors of its split and contains no position of another split
  or of the buffer. The planner uses a lower bound on the distance to
  sectors of other splits. Near a sector corner it can exclude a centre
  even if the footprint would fit. The diameter is drawn first and the
  centre uniformly from the admitted positions, so every split keeps the
  configured diameter distribution. Footprints of different splits are
  disjoint and their active dipoles are at least `buffer_mm` apart.
  Planning stops if a split has no admitted centre for the largest
  configured diameter and warns if it has fewer than 50.
  Six to eight sectors suit the default 10 to 60 mm diameters. Each split
  keeps the full pool density inside its region, where the default policy
  gives each split one third of the pool; use one third of the pool size
  for the same support density. `source_pool.*` stores the planner's radius
  limit at each position as `patch_containment_radius_mm`. The report field
  `largest_admissible_diameter_mm` is twice the largest such limit in the split.
- `shared` is available for point and multi-point modes, not for patches. Set
  `shared_position_fraction` in `(0, 1]`. For each population, the common core
  contains the requested fraction of positions rounded down to a whole
  position. Other positions retain their disjoint default assignments. A
  fraction of 1 gives all three splits the same finite pool. Each split has
  independent per-window seeds. The observed share of windows using a
  previously used location depends on the draws and the number of windows.

`plan_position_split(cfg)` returns the geometry before generation. Its
`positions_xyz` are in metres and its `indices` mapping gives eligible pool
rows for `train`, `val`, and `test`. `source_pool.*` and optional `leadfield.*`
store the same global indexing. Their `split` codes are 0/1/2 for train/val/test,
with −1 for a buffer exclusion and −2 for a shared position. Shared-policy
files also include `split_membership`, with one 0/1 column per split in that order.

After writing, the pipeline verifies that serialized source-pool indices are
nonempty and eligible for their split. For `position` and `region`, it also
verifies disjointness. The report includes all three cross-split nearest-
neighbour comparisons in millimetres, using both eligible positions and the
distinct positions actually used. Each population-specific result restricts
both sides of the comparison to that population. Region reports include
pre-buffer and retained counts, radial bin counts, sector assignments, and
the verified clearance. Shared reports separate eligibility from observed reuse.

### `stratify_by_depth`
- **Default**: `True`
- **What**: Stratify the exact-position assignment by radial depth. The
  split-specific part of a partially shared assignment uses the same rule.

### `depth_bins`
- **Default**: `10`
- **What**: Number of equal-count quantile bins for depth stratification.
- **Basis and scope**: Held-out-source evaluation is important for supervised
  synthetic ESI benchmarks (Hecker et al., 2021; Sun et al., 2022). Those works
  do not prescribe ten bins. Ten is a declared stratification granularity, and
  the realized population/radius coverage is reported rather than assumed.

---

## Dataset: `cfg.dataset`

### `n_train` / `n_val` / `n_test`
- **Defaults**: `300_000` / `30_000` / `30_000`
- **What**: Number of independently seeded windows per split. The split
  policy determines which source locations the windows can reuse.
- **Basis and scope**: Large simulated datasets are established practice in
  supervised ESI (Hecker et al., 2021; Sun et al., 2022; Reynaud et al., 2024).
  The exact 300,000/30,000/30,000 allocation is a declared benchmark-capacity
  choice, not a sample size prescribed by those papers or a statistical power
  calculation. Learning curves should be reported for task-specific use.

---

## Output: `cfg.output`

The canonical dataset directory contains deterministic scientific artifacts.
Operational values (`output_dir`, generation chunk size, worker count,
overwrite policy, timestamp, and wall time) are excluded from those artifacts
and written to the sibling runtime-provenance JSON. Stable interpreter,
platform, package, and numerical-library versions are stored canonically in
`environment.json`. EEG arrays are in volts by default under both noise
conventions.

Each split embeds electrode coordinates and names, position/electrode units,
EEG units, magnitude units, source-mode and reference metadata. With the
default centered unit-standard-deviation source courses,
`magnitude_convention='rms'`: `magnitudes_nAm` is the RMS dipole moment. If
`signal.zero_mean_signal=False`, the same courses retain unit standard
deviation but need not have unit RMS; the serialized convention is then
`'waveform_scale'`, meaning that the instantaneous moment is
`magnitudes_nAm * source_time_course` (or the corresponding row of
`source_time_courses` in multi-point mode).

### `eeg_normalization`
- **Default**: `'none'`
- **Options**: `'none' | 'per_sample_rms'`
- **What**: An output transform independent of `noise_model`. `'none'` preserves
  physical volts and stores `eeg_normalization_scale_V = 1`. `'per_sample_rms'`
  divides each referenced sample and its optional signal/noise components by
  the sample's total EEG RMS. The arrays are then dimensionless, and multiplying
  them by the stored per-sample `eeg_normalization_scale_V` restores volts.
- **Why it matters**: Unit-RMS output can prevent absolute sensor scale from
  directly carrying a magnitude label. Keeping this choice separate from SNR
  targeting avoids silently discarding physically meaningful amplitude.

Every run also writes `source_pool.*`, containing the global coordinates,
population labels, and split assignments referenced by sample truth, plus
`checksums.sha256`, which hashes every other canonical artifact in the dataset
directory. Verification requires the checksum members, metadata inventory, and
actual canonical files to agree exactly.

### `save_components`
- **Default**: `False`
- **What**: If True, also save signal-only and noise-only EEG
  separately (in addition to the mixed signal+noise EEG). This adds two more
  EEG-sized arrays and therefore approximately triples the EEG-tensor portion
  before compression; measured file-size impact depends on the data/backend.
- **Effect on canonical bytes**: with `dtype='float32'`, enabling this flag also
  changes how `/eeg` is rounded. Components are stored so that
  `eeg_signal_only + eeg_noise_only` reconstructs `/eeg` exactly, which requires
  forming `/eeg` as `float32(signal) + float32(noise)` rather than
  `float32(signal + noise)`; the two differ by one rounding step. The canonical
  directory hash of a float32 dataset therefore depends on `save_components`.
  At the default `dtype='float64'` both paths are bit-identical.

### `save_leadfield`
- **Default**: `False`
- **What**: If True, also write the position-pool lead-field operator to
  `leadfield.h5`, `leadfield.npz`, or MATLAB v7.3 `leadfield.mat`: the forward map
  `L` (shape `P x n_channels x 3`, local spherical frame, V per A·m) evaluated
  once at every source position, plus the positions, population labels, and the
  per-position train/val/test split. The configured sensor reference is already
  applied; any dipole moment projects as `L @ [p_r, p_theta, p_phi]`. Written once 
  per run (shared across splits). Off by default because the tensor can be large. 
  See `examples/export_leadfield.py`.

### `save_graph`
- **Default**: `False`
- **What**: If True, also write the electrode adjacency graph as `graph.h5`,
  `graph.npz`, or MATLAB v7.3 `graph.mat` for graph-neural-network models: a
  bidirectional `edge_index` (2, E), Gaussian-of-distance `edge_weight` (E,),
  raw `edge_length_m` (E,), node `degree` (N,), and the embedded electrode block
  (node order matches). Depends only on the montage, so written once per run.
  See `examples/export_graph.py`.
- **Basis and scope**: Graph-structured ESI motivates carrying sensor adjacency
  with the data (Jiao et al., 2022). The Gaussian edge weight with scale equal
  to mean retained-edge length is a declared, montage-adaptive engineering
  choice rather than a physiological coupling model.

### `graph_method`
- **Default**: `'knn'`
- **Options**: `'knn' | 'delaunay' | 'distance'`
- **What**: How the electrode graph is built (used when `save_graph=True`).
  `'knn'` connects each electrode to its `graph_k` nearest neighbours
  (symmetrised); `'delaunay'` triangulates the electrodes on the scalp sphere
  (convex hull of the 3-D points); `'distance'` connects all pairs within
  `graph_distance_threshold_m`.

### `graph_k`
- **Default**: `6`
- **What**: Number of nearest neighbours for `graph_method='knn'`. It must be
  smaller than the number of electrodes; invalid values are rejected rather
  than silently clamped.
- **Basis and scope**: Graph-structured sensor processing has ESI precedent
  (Jiao et al., 2022). Six is a declared sparse-connectivity choice, not a
  neighbour count prescribed by that work; graph sensitivity can be assessed
  by changing this value or selecting Delaunay construction.

### `graph_distance_threshold_m`
- **Default**: `None`
- **What**: Connection radius in metres for `graph_method='distance'` (required
  in that mode; ignored otherwise).

### `diagnostic_plots`
- **Default**: `False`
- **What**: Auto-generate figures: position scatter, radial depth
  distribution, SNR histogram, magnitude distribution, and sample EEG
  traces. Requires the `plot` extra. An explicit request fails clearly when
  matplotlib is absent rather than silently changing the canonical directory.

### `diagnostic_sample_count`
- **Default**: `1000`
- **What**: Number of leading training samples used for diagnostic SNR,
  magnitude, and trace plots (or the full split when it is smaller). It is
  independent of `chunk_samples`, preventing an operational write-buffer size
  from changing canonical figures.
- **Basis and scope**: One thousand is a declared visualization support size,
  not a statistical sample-size recommendation. The plots are descriptive
  audits of the generated configuration, not inferential validation.

### `dtype`
- **Default**: `'float64'`
- **Options**: `'float64' | 'float32'`
- **What**: Floating dtype used for the written EEG arrays. Ground-truth arrays
  retain their semantic dtypes. Float32 halves the uncompressed EEG-array byte
  width; its task-level effect should be measured. If `save_components=True`,
  `eeg_signal_only` and `eeg_noise_only` use the same dtype as `/eeg`, and the
  rounding of `/eeg` itself changes as described under `save_components`.

### `format`
- **Default**: `'hdf5'`
- **Options**: `'hdf5' | 'npz' | 'mat'`
- **What**: HDF5 supports streaming and partial loads. NPZ and genuine MATLAB
  v7.3 output are finalized from in-memory split arrays, so they are intended
  for datasets that fit in memory. MATLAB output requires `hdf5storage`; its
  wall-clock `Created on` user-block field is canonicalized before publication
  so operational time does not change the artifact hash.

### `compress`
- **Default**: `True`
- **What**: gzip compression for native HDF5, deterministic deflate for NPZ,
  and MATLAB v7.3 compression. Runtime and size changes are data-dependent and
  should be measured.

### `chunk_samples`
- **Default**: `1000`
- **What**: Number of samples generated and written per streaming chunk.
  Lower it on memory-constrained machines for HDF5. NPZ and MAT still retain
  each complete split until finalization. Because every sample is seeded
  individually (see `cfg.sample_seed` / `cfg.master_seed`), this controls peak
  memory and progress granularity only -- the produced data are identical for
  any `chunk_samples` value at fixed scientific seeds. Must be a positive int.

### `hdf5_chunk_rows`
- **Default**: `64`
- **What**: On-disk chunk size (samples along the leading axis) for native
  HDF5 `eeg` and ground-truth datasets. Larger values can favor
  sequential minibatch reads; smaller values speed up shuffled single-sample
  access. Use `1` for predominantly shuffled single-sample reads unless a
  measured access trace justifies a larger value.
  It is independent of `chunk_samples`: the writer stores blocks of complete
  storage chunks in a fixed order and keeps the rows of an unfinished block in
  memory, so generation chunk boundaries do not alter the HDF5 layout. A block
  is one storage chunk, or the smallest whole number of storage chunks that
  reaches 64 rows when this value is smaller, so small values do not multiply
  the number of write calls. It affects the canonical storage layout and read
  performance, not generated array values. Must be a positive int.

### `n_workers`
- **Default**: `1`
- **What**: Number of per-sample worker processes; `-1` requests all available
  CPUs. This is operational provenance, excluded from canonical artifacts.
  Per-sample seeding makes canonical output independent of worker count.
  Workers hold process-local data, so memory use must be measured.

### `overwrite`
- **Default**: `False`
- **What**: A non-empty target directory raises `FileExistsError` unless this is
  true. Even then, replacement is allowed only when package metadata identifies
  the target as a prior `spherical_eeg_sim` dataset and every existing file is
  listed in its canonical inventory. Replacement publishes a fully generated
  staging directory and removes stale members of the previous dataset;
  partially generated output is not installed as the target. On startup,
  marker-verified generator-owned incomplete staging trees are discarded,
  staging-like directories without a valid ownership marker are never deleted,
  the internal marker is removed before publication, checksum-complete
  matching staging trees are published, and a sole checksum-complete rollback
  directory with a valid package/target/metadata marker is restored if the
  target name is absent. Rollback-like directories without that marker are
  never used or deleted. Ambiguous states fail closed.
  Concurrent generation or recovery for the same target raises
  `BlockingIOError`. The sibling `.<target>.generation.lock` stays in place
  between runs. Its OS lock is released when the process closes it or exits.
  Portable
  replacement is a rename-and-rollback transaction, not an atomic exchange of
  two non-empty directories.

### `reference_mode` / `reference_channel` / `reference_weights`
- **Defaults**: `'none'` / `'Cz'` / `None`
- **What**: EEG reference convention applied to the written `/eeg`. Real EEG is
  reference-dependent; a per-time-point constant (or weighted channel
  combination) is subtracted from every channel.
  - `'none'` — no discrete electrode reference (best for physics validation and
    for users who reference downstream). Because the analytic series begins at
    Legendre order `n=1`, it fixes the otherwise arbitrary additive potential
    constant by the zero-spherical-mean gauge (de Munck & Peters, 1993; Nunez &
    Srinivasan, 2006).
  - `'average'` — common average reference (subtract the per-sample
    across-channel mean at each time point).
  - `'cz'` — subtract the `reference_channel` trace (default `'Cz'`).
  - `'custom'` — subtract `reference_channel` by name, or a weighted channel
    combination given by `reference_weights` (one entry per channel, normalized
    to sum 1).
- **Note**: By default (`snr_definition='referenced'`) the per-sample target
  SNR is achieved on this referenced `/eeg`, so the `snr_db` label matches the
  stored data. Set `snr_definition='raw'` to instead define SNR on
  the pre-reference potentials. Either way, both the raw-montage and
  referenced-montage SNR are written to ground truth (`snr_db_raw`,
  `snr_db_referenced`). The chosen reference mode and the electrode
  names/coordinates are written into every output file (a native HDF5 group,
  MATLAB struct, or `electrode_*` NPZ arrays), so reference choices are
  reproducible without the original montage CSV.

---

## Tuning by use case

### "I want to compare against ConvDip / DeepSIF"
Set `subcortical_fraction=0`, `orientation_mode='cortex_normal'`, and
optionally `source_mode='patch'` to approximate selected cortex-constrained
benchmark factors in ConvDip and DeepSIF (Hecker et al., 2021; Sun et al.,
2022). This is not a reproduction of either method: use their reported source,
head, signal, and preprocessing models for a direct comparison.

### "I want to validate channel-count generalisation"
Generate the dataset once with a high-channel montage (e.g. a 256-channel
custom CSV). Use random channel-subsetting at training time. Electrode-count
robustness is motivated by Reynaud et al. (2024); 256 is an example supported
montage size, not a prescribed study design.

### "I want a small smoke-test dataset"
Use `quickstart_config()` (800/100/100 samples with a 500/100 position pool),
or adapt `examples/quickstart_small.py` for a different smoke-test size. These
are operational test counts chosen to exercise every split, not scientific
sample-size recommendations.

### "I want to study deep-source localisation specifically"
Set `subcortical_fraction=0.5`, increase `n_subcortical_positions` to
20,000, and set `subcortical_radius_range_mm=[20, 50]` to focus on
the deepest model radii. Deep-source evaluation is motivated by Seeber et al.
(2019), Pizzo et al. (2019), and Yokoyama et al. (2024); these exact tutorial
values are declared stress-test starting points, not anatomical bounds or
prevalence estimates.

### "I want high-SNR ceiling-performance data"
Set `noise_model='snr_target'`, `snr_db_range=[40, 50]`, and
`mix_white=0`, `mix_pink=0`, `mix_physiological=1` (still some
volume-conducted background to avoid degenerate zero-noise inputs). Controlled
SNR simulation is supported by Krol et al. (2018) and Sun et al. (2022); the
40-to-50 dB interval and pure-background mixture are declared ceiling-test
settings, not empirical recording statistics.

### "I want my model to handle multiple simultaneous sources"
Set `source_mode='multi_point'`, `n_sources_min=1`, `n_sources_max=5`
for ConvDip-compatible multi-source training.

### "I want spatially extended (patch) sources"
Set `source_mode='patch'`. Defaults give 10-60 mm great-circle-diameter
supports with a truncated Gaussian profile, informed by the patch extents in
ConvDip (Hecker et al., 2021).
