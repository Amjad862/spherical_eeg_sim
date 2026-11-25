"""
Configuration module for the EEG Source Localization dataset generator.

All user-facing settings live here as nested dataclasses with documented defaults.
The defaults reflect the choices defended in our methodology (see README.md and
docs/parameters.md for citations).

Quick start:

    from spherical_eeg_sim.config import Config
    cfg = Config()                          # Use all defaults
    cfg.dataset.n_train = 50_000           # Override individual fields
    cfg.source_mode = 'multi_point'

Or load from a JSON file:

    cfg = Config.from_json('my_config.json')
    cfg.to_json('my_config_saved.json')
"""

from __future__ import annotations
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json
import math
import numbers
import warnings


# Fewest active dipoles the smallest configured patch should recruit before the
# activation stops being meaningfully distributed. Below this the support is a
# small cluster of point sources, so the run is flagged rather than silently
# producing patch-labelled samples with near-point extent.
MIN_EXPECTED_PATCH_SUPPORT = 10


class _StrictFields:
    """Base class that rejects assignment to undeclared configuration fields.

    Every setting that reaches the generator is declared as a dataclass field.
    A name that is not declared cannot influence the simulation, so accepting it
    silently would let a run report a configuration it never used. Assignment to
    an unknown public name therefore raises instead of creating a dead attribute.
    Private names (leading underscore) are left to normal Python semantics.
    """

    __slots__ = ()

    def __setattr__(self, name: str, value: Any) -> None:
        fields = getattr(type(self), '__dataclass_fields__', {})
        if not name.startswith('_') and name not in fields:
            valid = ', '.join(sorted(fields))
            raise AttributeError(
                f"{type(self).__name__} has no configuration field {name!r}. "
                f"Valid fields are: {valid}"
            )
        object.__setattr__(self, name, value)


# =============================================================================
# Head model
# =============================================================================

@dataclass
class HeadModelConfig(_StrictFields):
    """4-layer spherical head model.

    The default radii define a declared benchmark geometry close to the
    80/82/87/92 mm four-shell proportions used by Srinivasan et al. (1998),
    with a 1 mm CSF shell and an 86 mm outer-skull boundary. Brain and scalp
    conductivity follow Geddes & Baker (1967), cerebrospinal fluid follows
    Baumann et al. (1997), and the skull value gives a brain-to-skull ratio
    near 80 (Rush & Driscoll 1968). Order is brain -> CSF -> skull -> scalp.

    Modify these to model alternative head geometries (e.g. ratio-3-shell
    by removing the CSF layer, or different skull conductivity values
    to study sensitivity to that parameter).
    """
    # Layer radii in meters (brain, CSF, skull, scalp).
    radii_m: List[float] = field(default_factory=lambda: [0.080, 0.081, 0.086, 0.092])

    # Conductivities in S/m for each layer (same order as radii_m).
    conductivities_S_per_m: List[float] = field(
        default_factory=lambda: [0.330, 1.790, 0.0042, 0.330]
    )

    # Maximum order of Legendre expansion. The default 80 and the supported
    # range are audited empirically by scripts/legendre_convergence.py across
    # superficial, mid-depth, and deep radial/tangential cases.
    # Orders above 120 are rejected because the current float64 transfer-matrix
    # recurrence becomes round-off dominated before underflow; the empirical
    # convergence audit is implemented by scripts/legendre_convergence.py.
    # Accuracy is not monotone in this setting. Truncation error falls with
    # order while round-off in the transfer recurrence grows, so the total error
    # has an interior minimum near order 100 for the default geometry and
    # degrades again by order 120. Raising n_legendre past that optimum is
    # accepted but warned about, because the crossover moves with the radii and
    # conductivities.
    n_legendre: int = 80

    # Maximum accepted value of the geometric tail indicator
    #
    #     q ** (n_legendre + 1) / (1 - q),
    #
    # where q is the largest configured signal or background-dipole radius
    # divided by the scalp radius. This is a dimensionless convergence
    # diagnostic, not an error bound: it prevents a finite expansion from being
    # used in a source domain whose geometric weighting decays too slowly. The
    # default admits the full shipped source domain throughout the supported
    # order range.
    truncation_tail_tolerance: float = 0.05

    # ---- Presets for head-model ablation studies ----
    # These return ready-made HeadModelConfig instances for the two most common
    # sensitivity analyses. Assign the result to cfg.head, e.g.
    #     cfg.head = HeadModelConfig.three_shell()
    #     cfg.head = HeadModelConfig.with_skull_ratio(20.0)
    # so common head-model sensitivity analyses are turnkey rather than
    # requiring the user to look up conductivity values.

    @classmethod
    def four_shell_default(cls) -> 'HeadModelConfig':
        """The default four-shell brain/CSF/skull/scalp model (Srinivasan et al. 1998).

        Returned explicitly so ablation drivers can name the baseline alongside
        the alternative presets below.
        """
        return cls()

    @classmethod
    def three_shell(cls, n_legendre: int = 80) -> 'HeadModelConfig':
        """A classic three-tissue brain/skull/scalp model (Rush & Driscoll 1968).

        Realized within the four-layer solver by setting the CSF conductivity
        equal to the brain conductivity: with no conductivity jump at the
        brain/CSF boundary the medium is homogeneous out to the skull, so this
        is a genuine three-shell model (not an approximation) that omits the
        CSF compartment. Useful for isolating the effect of the CSF layer, which
        is the single largest difference between three- and four-shell spherical
        heads. Radii are unchanged from the default; only the CSF conductivity
        is collapsed onto the brain value.
        """
        return cls(
            radii_m=[0.080, 0.081, 0.086, 0.092],
            conductivities_S_per_m=[0.330, 0.330, 0.0042, 0.330],
            n_legendre=n_legendre,
        )

    @classmethod
    def with_skull_ratio(cls, brain_to_skull_ratio: float,
                         n_legendre: int = 80) -> 'HeadModelConfig':
        """Four-shell default with the skull conductivity set by a brain:skull ratio.

        Skull conductivity is the most uncertain head-model parameter, with
        reported brain-to-skull conductivity ratios spanning roughly 8 to 80
        (Goncalves et al. 2003; Oostendorp et al. 2000). This preset fixes the
        skull conductivity at ``sigma_brain / brain_to_skull_ratio`` and leaves
        every other layer at the default, so a ratio sweep (e.g. 15, 40, 80)
        maps the sensitivity of the forward solution and of downstream
        localization to that single parameter. The default four-shell model
        corresponds to a ratio of about 78.6 (0.330 / 0.0042).
        """
        if (isinstance(brain_to_skull_ratio, bool)
                or not isinstance(brain_to_skull_ratio, numbers.Real)
                or not math.isfinite(float(brain_to_skull_ratio))
                or brain_to_skull_ratio <= 0):
            raise ValueError("brain_to_skull_ratio must be a finite value > 0")
        if (isinstance(n_legendre, bool)
                or not isinstance(n_legendre, numbers.Integral)):
            raise ValueError("n_legendre must be an integer")
        sigma_brain = 0.330
        return cls(
            radii_m=[0.080, 0.081, 0.086, 0.092],
            conductivities_S_per_m=[
                sigma_brain, 1.790, sigma_brain / brain_to_skull_ratio, 0.330
            ],
            n_legendre=n_legendre,
        )


# =============================================================================
# Source populations
# =============================================================================

@dataclass
class SourcesConfig(_StrictFields):
    """Source location distributions for cortical and subcortical populations.

    Cortical and subcortical labels denote superficial and deep radial source
    spaces in the spherical model; they are not anatomical segmentations.
    Deep-source detectability motivates including the second population
    (Seeber et al. 2019; Pizzo et al. 2019).

    The subcortical_fraction is a SIMULATION-DESIGN choice: real EEG is
    dominated by cortical sources, but for training a generic source
    localizer we oversample subcortical positions to ensure the model
    learns to handle them. Set to 0.0 for cortex-only training.

    Position counts control how dense the spatial sampling grid is.
    More positions = finer localization but larger lead-field tensor.
    """
    # Cortical shell radial range (in mm from brain center).
    # Default 70.5-79.5 mm corresponds to ~0.5-9.5 mm depth from brain
    # surface (radius 80 mm). This is a benchmark shell, not a literal
    # cortical-thickness model (de Munck & Peters 1993; Dale & Sereno 1993).
    cortical_radius_range_mm: List[float] = field(default_factory=lambda: [70.5, 79.5])

    # Subcortical radial range (in mm from brain center).
    # 30-65 mm is a configurable deep radial benchmark. It does not identify
    # specific nuclei or hippocampal anatomy (Seeber et al. 2019; Pizzo et al.
    # 2019 motivate deep-source inclusion, not these spherical coordinates).
    subcortical_radius_range_mm: List[float] = field(default_factory=lambda: [30.0, 65.0])

    # Fraction of training positions drawn from the subcortical population.
    # 0.0 = superficial-only; 0.15 = deep-source exposure with superficial dominance;
    # 0.5 = balanced; 1.0 = subcortex-only (rarely useful).
    subcortical_fraction: float = 0.15

    # Number of unique superficial positions in the lead-field grid. The
    # default is a configurable benchmark-resolution choice. Quasi-uniform
    # spherical discretization follows Gonzalez (2010), while the exact count
    # is declared rather than attributed to that paper; see docs/parameters.md.
    n_cortical_positions: int = 35_000

    # Number of unique deep positions in the lead-field grid. The default is a
    # configurable benchmark-resolution choice. Published supervised ESI work
    # motivates dense simulated source spaces, not this exact count (Hecker et
    # al. 2021; Sun et al. 2022); see docs/parameters.md.
    n_subcortical_positions: int = 6_500

    # Sampling pattern on the cortical shell.
    # 'fibonacci': quasi-uniform via Fibonacci lattice (recommended)
    # 'uniform_random': independent uniform samples (less even coverage)
    cortical_sampling: str = 'fibonacci'

    # Sampling pattern in the subcortical volume.
    # 'uniform_volume': uniform within the radial shell
    # 'fibonacci_shells': layered Fibonacci lattices at multiple radii
    subcortical_sampling: str = 'uniform_volume'


# =============================================================================
# Dipole properties
# =============================================================================

@dataclass
class DipoleConfig(_StrictFields):
    """Dipole orientation and magnitude distributions.

    Orientation modes:
      'uniform_s2'    -- Uniform on the unit sphere from normalized isotropic
                         Gaussian draws.
                         RECOMMENDED. Allows the model to handle arbitrary
                         orientations rather than memorising a discrete codebook.
      'codebook'      -- Each sample is either purely radial OR
                         purely tangential, with the tangent direction drawn
                         from a finite golden-angle codebook. A discrete
                         comparison mode with equal aggregate probability
                         across the three local basis dimensions.
      'cortex_normal' -- Orientations along the local radial direction (i.e.
                         normal to a notional spherical cortex). Cheapest in
                         training samples per orientation; biologically motivated
                         for cortical pyramidal-cell ensembles.

    The configurable ranges are literature-guided (Hamalainen et al. 1993;
    Murakami & Okada 2015; Attal & Schwartz 2013). Log-uniform sampling is a
    declared benchmark choice that distributes examples across those ranges;
    it is not claimed to be an empirical dipole-moment distribution.
    """
    # 'uniform_s2' | 'codebook' | 'cortex_normal'
    orientation_mode: str = 'uniform_s2'

    # Number of discrete golden-angle directions in the local tangent plane
    # when orientation_mode='codebook'.
    codebook_tangential_directions: int = 256

    # Superficial-population benchmark range (nA*m), spanning the ~10 nA*m
    # focal scale and moment-density-times-active-area scales reported in the
    # literature (Hamalainen et al. 1993; Murakami & Okada 2015; Goldenholz et
    # al. 2009); see docs/parameters.md for the explicit derivation and scope.
    cortical_magnitude_range_nAm: List[float] = field(default_factory=lambda: [5.0, 300.0])

    # Deep-population benchmark range (nA*m), within the extended-source scale
    # derived in docs/parameters.md and emphasized to expose attenuated deep
    # sources. This is not a claim of higher physiological moment density
    # (Attal & Schwartz 2013; Murakami & Okada 2015).
    subcortical_magnitude_range_nAm: List[float] = field(default_factory=lambda: [30.0, 500.0])

    # 'log_uniform' | 'uniform' | 'fixed' (uses range[0] as the constant)
    magnitude_distribution: str = 'log_uniform'


# =============================================================================
# Source mode (single-point / multi-point / patch)
# =============================================================================

@dataclass
class MultiPointConfig(_StrictFields):
    """Multi-source generation options (only used if source_mode='multi_point').

    Each sample contains 1..N dipoles at random positions, each
    with its own orientation and magnitude. Useful for training multi-source
    detectors or evaluating how methods scale with source count (Hecker et al.
    2021; Pantazis & Adler 2021). Exact separation and coupling defaults are
    declared benchmark settings rather than values prescribed by those papers.
    """
    n_sources_min: int = 1
    n_sources_max: int = 5

    # Minimum 3D Euclidean separation (mm) between any two dipoles in the
    # same sample. Prevents degenerate near-colocated configurations. The exact
    # 20 mm threshold is a declared benchmark guard.
    min_separation_mm: float = 20.0

    # Candidate draws allowed per added dipole before accepting and flagging a
    # separation violation. One hundred is a declared rejection-sampling
    # budget; the realized violation count is reported.
    max_separation_attempts: int = 100

    # Temporal coupling between additional dipoles and the primary dipole.
    # This is a linear BLEND WEIGHT, not a Pearson correlation coefficient.
    # The primary dipole always uses time_courses[i]; each additional dipole
    # gets a freshly generated course, then linearly mixed toward the primary:
    #     course = time_coupling*primary + (1-time_coupling)*new
    # and renormalized to unit std. Independently generated finite windows need
    # not have zero sample correlation. If their correlation is rho, the blend's
    # correlation with the primary is
    #   (c + (1-c)*rho) / sqrt(c**2 + (1-c)**2 + 2*c*(1-c)*rho).
    # The stored base and final courses make the realized value auditable.
    #   0.0 -- additional dipoles use independently generated time courses
    #   1.0 -- additional dipoles reuse the primary course (fully coherent)
    time_coupling: float = 0.0

    # Optional finite design set sampled independently for every generated
    # window. When provided it supersedes ``time_coupling`` and supports
    # factorial designs in which acquisition settings and temporal coupling
    # vary independently.
    time_coupling_values: Optional[List[float]] = None

    # ``raise`` enforces the advertised separation as a hard corpus invariant.
    # ``accept_and_flag`` permits explicitly exploratory generation while
    # recording each accepted failure.
    separation_failure_policy: str = 'raise'


@dataclass
class PatchConfig(_StrictFields):
    """Distributed cortical patch options (only used if source_mode='patch').

    A patch is a finite-support region of cortex with a center, an extent, and
    a configurable activation profile. It provides a controlled
    spherical approximation to distributed-source benchmarks (Hecker et al.
    2021).

    Patches always use 'cortex_normal' orientations regardless of the
    DipoleConfig.orientation_mode setting. Each active point follows its own
    local normal on the spherical surface; this is a controlled surface-normal
    model and does not represent cortical folding.
    """
    # Patch diameter range in mm. ConvDip used 21-58 mm; we widen slightly.
    diameter_range_mm: List[float] = field(default_factory=lambda: [10.0, 60.0])

    # Activation falloff from patch center.
    # 'gaussian' -- Gaussian decay truncated at the patch radius
    #               (sigma = diameter/4; nonzero limiting boundary weight)
    # 'uniform'  -- step function (within = 1, outside = 0)
    # 'cosine'   -- cosine taper (smooth, finite support)
    falloff_profile: str = 'gaussian'

    # Maximum density of active dipoles inside a patch (dipoles per mm^2 of
    # disk area). Acts as an upper bound: if the cortical position pool has
    # more candidates within the patch radius than this density implies,
    # the candidates are randomly subsampled to match. If the pool has
    # fewer (i.e. the natural pool density is below this cap), every pool
    # candidate within the radius is used. Larger value -> denser patches
    # but slower generation; smaller value -> sparser patches. Patch-source
    # simulation is motivated by Hecker et al. (2021); 1.0 is a declared
    # discretization cap rather than a reported physiological density.
    dipoles_per_mm2: float = 1.0

    # Report cross-split overlap of the nominal spherical-cap footprints.
    report_overlap: bool = True


# =============================================================================
# Signal model
# =============================================================================

@dataclass
class SignalConfig(_StrictFields):
    """Source time-course generation.

    The default time course is a sum of n_components band-limited sinusoids
    with random frequencies, phases, and (optionally) amplitudes. Each
    sample's generating band is drawn according to band_probabilities. The
    band definitions follow standard EEG rhythm categories (Buzsaki 2006;
    Babiloni et al. 2020); the categorical probabilities are a declared,
    alpha-enriched benchmark distribution rather than empirical power shares.

    Setting n_components=1 reduces to a single-sinusoid time course.
    Increasing n_components produces richer within-window patterns and
    forces the temporal encoder to do meaningful denoising/separation.
    """
    sampling_freq_hz: float = 500.0
    window_samples: int = 100  # 200 ms at 500 Hz

    # Number of sinusoidal components in each sample's time course. Composite
    # oscillatory simulation follows Krol et al. (2018) and Barzegaran et al.
    # (2019); three is a declared benchmark-complexity choice. Drawn uniformly
    # from [min, max]. min=max=1 for a single-sinusoid model.
    n_components_min: int = 3
    n_components_max: int = 3

    # IFCN-standard frequency bands (Hz).
    bands_hz: Dict[str, List[float]] = field(default_factory=lambda: {
        'delta': [0.5, 4.0],
        'theta': [4.0, 8.0],
        'alpha': [8.0, 13.0],
        'beta':  [14.0, 30.0],
        'gamma': [30.0, 45.0],
    })

    # Explicit semantic order for band indices. Keeping this as data rather
    # than relying on mapping insertion order makes JSON/manifest round trips
    # unambiguous.
    band_order: List[str] = field(default_factory=lambda: [
        'delta', 'theta', 'alpha', 'beta', 'gamma',
    ])

    # Alpha-enriched benchmark probability for selecting a generating band.
    # These are not empirical proportions of total EEG power.
    # The index recorded in /ground_truth/band_idx follows band_order.
    band_probabilities: Dict[str, float] = field(default_factory=lambda: {
        'delta': 0.20,
        'theta': 0.15,
        'alpha': 0.35,
        'beta':  0.20,
        'gamma': 0.10,
    })

    # Per-component amplitude variation within a sample.
    # If True, components have random relative amplitudes (Dirichlet on simplex);
    # if False, all components have equal amplitude.
    component_amplitude_jitter: bool = True

    # Symmetric Dirichlet concentration used when amplitude jitter is enabled.
    # A value of one gives a uniform simplex draw. Dirichlet-weighted composite
    # signals are a declared simulation choice; realized amplitudes are stored.
    component_dirichlet_concentration: float = 1.0

    # ---- Window-vs-band resolvability policy ----
    # A short analysis window cannot resolve a full cycle of the lowest bands:
    # a band's lowest edge f_low completes f_low * window_samples / sampling_freq
    # cycles within the window. At the default 200 ms window the delta band
    # (0.5 Hz low edge) spans only ~0.1 cycle, so a "delta" window is closer to
    # a slowly-varying offset than a resolved oscillation.
    #
    # min_band_cycles : a band is considered "resolvable" only if its low edge
    #   completes at least this many cycles within the window.
    # low_freq_band_policy controls what happens to bands that fall below it:
    #   'restrict' (default) -- drop under-resolved bands (their probability is
    #                           redistributed over the resolvable bands) and log
    #                           which bands were dropped. Guarantees every
    #                           generated band actually oscillates in-window;
    #                           configurations with no surviving band are rejected.
    #   'keep'               -- keep all bands unchanged, but log a warning and
    #                           record in metadata that under-resolved band
    #                           labels denote the DOMINANT SPECTRAL COMPONENT of
    #                           the generating process, not a resolved in-window
    #                           oscillation.
    # Either way the decision and the effective band probabilities are written
    # to metadata.json for full transparency.
    min_band_cycles: float = 1.0
    low_freq_band_policy: str = 'restrict'

    # Remove the per-sample temporal mean from each source time course before
    # it is scaled (so the window is a zero-mean fluctuation rather than a
    # fluctuation riding on a DC offset). With a short window the lowest bands
    # otherwise retain a large DC term after unit-variance normalization; mean
    # removal makes the signal-only EEG average-free in time, consistent with
    # standard EEG high-pass handling (Babiloni et al. 2020). Set False to preserve the raw
    # (possibly DC-heavy) sinusoid sum.
    zero_mean_signal: bool = True


# =============================================================================
# Noise model
# =============================================================================

@dataclass
class NoiseConfig(_StrictFields):
    """Three-component noise model.

    Recorded EEG contains spatially correlated and uncorrelated variability.
    This benchmark mixes:
      'physiological' -- N spherical background dipoles with 1/f time courses,
                         forward-projected through the head model. Naturally
                         spatially correlated through volume conduction; this
                         is a background proxy, not a physiological recording.
      'pink'          -- Independent per-channel 1/f noise.
      'white'         -- Independent per-channel flat-spectrum noise.

    All three are mixed with square-root coefficients derived from weights that
    sum to one. The weights are intended relative variance contributions;
    finite-window cross-covariances mean they are not exact realized shares.
    How the mixed noise is then scaled relative to the signal is set by
    ``noise_model`` (see below).

    Bijma et al. (2003), de Munck et al. (2002, 2008), and Donoghue et al.
    (2020) motivate correlated source background and aperiodic spectra, but do
    not prescribe this decomposition. Sun et al. (2022) provides controlled-SNR
    simulation precedent. Goldenholz et al. (2009) and Piastra et al. (2021)
    support the default absolute-floor calibration; see docs/parameters.md for
    scope and limitations.
    """
    # ---- Noise scaling model ----
    # Selects how the mixed noise is scaled relative to the signal.
    #   'absolute_floor' (default) -- fix the combined-noise RMS at a physical
    #       level (absolute_noise_rms_uV) on the raw sensor montage, independent
    #       of the signal. The per-sample SNR is NOT prescribed; it emerges from
    #       source amplitude x forward gain. Absolute voltage scale is retained
    #       rather than erased, but identifiability is not guaranteed because
    #       position, orientation, and magnitude remain confounded. Physical
    #       source radius and depth are stored explicitly, and
    #       signal_topographic_concentration is a descriptive sensor-map
    #       statistic, not an anatomical depth estimator.
    #       snr_db_range is unused in this mode.
    #   'snr_target' -- scale the noise so each sample hits a target SNR drawn
    #       uniformly from snr_db_range on the montage named by snr_definition.
    #       This controlled-SNR convention appears in deep-learning source
    #       imaging (e.g. Sun et al. 2022). Signal voltages are preserved;
    #       output.eeg_normalization independently controls any stored-array
    #       normalization.
    noise_model: str = 'absolute_floor'

    # Fixed combined-noise RMS at the raw scalp montage, in microvolts; used
    # only when noise_model == 'absolute_floor'.
    #   Goldenholz et al. (2009) and Piastra et al. (2021) provide a 3.16 uV
    #   RMS reference calibration (variance ~= 10 uV^2). The package default
    #   is 1.00 uV RMS so the broad default source/montage distribution retains
    #   a more useful emergent-SNR range. Set 3.16 explicitly when reproducing
    #   the reference-calibrated floor.
    absolute_noise_rms_uV: float = 1.00

    # Number of background dipoles for the source-projected background proxy.
    # Higher -> more isotropic noise; lower -> more low-rank spatial structure.
    # Published models motivate spatially correlated background activity (de
    # Munck et al. 2002; Bijma et al. 2003); 10 is a declared finite-rank
    # benchmark choice, not an empirical generator count.
    n_background_sources: int = 10

    # Number of distinct source projections available to the background proxy.
    # Correlated source-background models are motivated by de Munck et al.
    # (2002) and Bijma et al. (2003); 2000 is a declared Monte Carlo support
    # choice, not a physiological source count. Must be >= n_background_sources.
    background_pool_size: int = 2000

    # Radial range of source projections used by the background proxy. The
    # superficial default reflects cortical dominance in EEG source modeling
    # (Hamalainen et al. 1993; de Munck et al. 2002); the exact spherical bounds
    # are declared benchmark coordinates, not anatomy.
    background_radius_range_mm: List[float] = field(
        default_factory=lambda: [70.5, 79.5]
    )

    # Symmetric Dirichlet concentration for relative background-source weights.
    # One is a declared uniform-simplex benchmark choice, not an empirical
    # distribution of concurrent generator strengths.
    background_amplitude_concentration: float = 1.0

    # Intended relative-variance weights (non-negative; normalized internally).
    mix_physiological: float = 0.50
    mix_pink: float = 0.35
    mix_white: float = 0.15

    # Per-sample SNR in dB drawn uniformly from this range. Used only when
    # noise_model == 'snr_target'.
    snr_db_range: List[float] = field(default_factory=lambda: [0.0, 20.0])

    # Which montage the target SNR is defined on.
    #   'referenced' (default) -- the drawn snr_db is achieved on the FINAL,
    #                             referenced /eeg (i.e. the data a model trains
    #                             on). This is the common use case because real
    #                             EEG is always referenced.
    #   'raw'                  -- the drawn snr_db is achieved on the raw,
    #                             pre-reference forward potentials.
    # Referencing is linear, so it commutes with signal+noise superposition; the
    # only difference between the two settings is which montage's signal/noise
    # powers set the noise scale factor. When reference_mode='none' the two
    # montages coincide and this setting has no effect. Either way BOTH the
    # raw-montage SNR and the referenced-montage SNR are written to ground truth
    # (snr_db_raw, snr_db_referenced); snr_db equals whichever one this setting
    # selects.
    snr_definition: str = 'referenced'

    # Spectral exponent for both physiological and pink noise (1/f^alpha).
    # alpha=1 denotes pink noise; target-specific exponents should be measured
    # or justified (Donoghue et al. 2020).
    background_freq_alpha: float = 1.0


# =============================================================================
# Train/val/test split
# =============================================================================

@dataclass
class SplitConfig(_StrictFields):
    """Source-position eligibility for independent train/val/test windows."""
    # Stratify the split by radial depth bins to ensure each split has
    # balanced superficial/deep representation.
    stratify_by_depth: bool = True

    # Number of equal-count depth bins for stratification. Ten is a declared
    # benchmark granularity; the resulting split balance is measured and
    # stored rather than assumed. Position-held-out evaluation is motivated by
    # the synthetic supervised-ESI setting (Hecker et al. 2021; Sun et al.
    # 2022), but those papers do not prescribe this count.
    depth_bins: int = 10

    # 'position' assigns each position to one split. 'region' holds out
    # buffered spatial sectors and, in patch mode, admits a patch only where
    # its whole footprint stays inside its own split region. 'shared' makes a
    # common fraction of the finite position pool eligible for all three
    # splits (point and multi-point modes).
    policy: str = 'position'
    region_count: Optional[int] = None
    buffer_mm: Optional[float] = None
    region_rotation_seed: Optional[int] = None
    shared_position_fraction: Optional[float] = None


# =============================================================================
# Dataset size and output
# =============================================================================

@dataclass
class DatasetConfig(_StrictFields):
    """Overall dataset volumes and split sizes.

    Large simulated training sets are established in supervised ESI (Hecker et
    al. 2021; Sun et al. 2022; Reynaud et al. 2024). The exact defaults below
    are declared benchmark-capacity choices rather than literature-prescribed
    sample counts and remain configurable.
    """
    n_train: int = 300_000
    n_val: int = 30_000
    n_test: int = 30_000


@dataclass
class OutputConfig(_StrictFields):
    """Output format and diagnostics."""
    # Save signal-only and noise-only components alongside mixed EEG (debug).
    #
    # Enabling this changes how the stored EEG is rounded when dtype='float32'.
    # With components saved, eeg is formed as float32(signal) + float32(noise)
    # so that the two stored components reconstruct it exactly; without them it
    # is float32(signal + noise), which differs by a single rounding step. The
    # canonical hash of a float32 dataset therefore depends on this flag. At the
    # default dtype='float64' the two paths are bit-identical.
    save_components: bool = False

    # Also write the position-pool basis lead-field operator to disk as an
    # auxiliary artifact (leadfield.h5, or leadfield.npz when format='npz').
    # This is the forward operator L (shape P x n_channels x 3, local spherical
    # frame) evaluated once at every source position, plus the positions,
    # population labels, and per-position train/val/test split assignment. It
    # lets downstream methods -- classical inverse operators, and physics-
    # informed models that need the forward map explicitly -- reuse the exact
    # operator the dataset was generated from without recomputing it. Off by
    # default because the tensor can be large (P x n_channels x 3 float64).
    save_leadfield: bool = False

    # Also write the electrode adjacency graph to disk as an auxiliary artifact
    # (graph.h5, or graph.npz when format='npz') for graph-neural-network
    # workflows. The graph depends only on the montage, so it is built once per
    # run. It stores a bidirectional edge_index (2, E), Gaussian-of-distance
    # edge weights, raw edge lengths, node degrees, and the self-describing
    # electrode block. See spherical_eeg_sim.io.graph_export. Off by default.
    save_graph: bool = False

    # Construction method for the electrode graph (used when save_graph=True).
    #   'knn'      -- symmetric k-nearest-neighbour graph (default)
    #   'delaunay' -- Delaunay triangulation on the scalp sphere (parameter-free)
    #   'distance' -- all electrode pairs within graph_distance_threshold_m
    graph_method: str = 'knn'

    # Number of nearest neighbours for graph_method='knn'. Graph-based ESI
    # motivates exporting sensor adjacency (Jiao et al. 2022); six is a
    # declared sparse-connectivity choice, not a value prescribed there.
    graph_k: int = 6

    # Connection radius in metres for graph_method='distance'. Ignored otherwise.
    graph_distance_threshold_m: Optional[float] = None

    # Auto-generate diagnostic figures (position scatter, SNR histogram, etc).
    # Opt-in because matplotlib is an optional dependency; an explicit request
    # fails clearly if the plotting extra is unavailable.
    diagnostic_plots: bool = False

    # Leading samples used for diagnostic histograms/traces. This is separate
    # from chunk_samples so canonical plots cannot depend on write buffering.
    # One thousand is a declared visualization support size, not a statistical
    # sample-size recommendation.
    diagnostic_sample_count: int = 1000

    # Storage backend.
    # 'hdf5' -- single .h5 file per split (recommended; cross-language)
    # 'npz'  -- numpy .npz archives
    # 'mat'  -- genuine MATLAB v7.3 .mat container; requires hdf5storage
    format: str = 'hdf5'

    # Compress output (HDF5 gzip, deterministic NPZ deflate, MAT compression).
    compress: bool = True

    # Numeric dtype used for EEG arrays written to disk. Ground-truth arrays
    # keep their natural precision and integer types. Use 'float32' for large
    # ML datasets to reduce disk use and loading time; use 'float64' for
    # physics validation or maximum numerical precision. Component arrays saved
    # by output.save_components use the same dtype as /eeg.
    dtype: str = 'float64'

    # Optional stored-array normalization, independent of the noise model.
    #   'none'           -- write the referenced EEG in physical volts.
    #   'per_sample_rms' -- divide each complete referenced sample by its RMS;
    #                       eeg_normalization_scale_V stores the exact inverse
    #                       physical scale for that sample.
    eeg_normalization: str = 'none'

    # Per-chunk sample count for streaming split generation. Each split is
    # generated and written in chunks of this many samples; lower it on
    # memory-constrained machines. Because every sample is seeded individually
    # from (master_seed, split_id, global_sample_index), this knob affects only
    # peak memory and progress granularity -- the generated data are identical
    # for any chunk_samples value at a fixed master_seed. Must be a positive int.
    chunk_samples: int = 1000

    # On-disk chunk size (samples along the leading axis) for native HDF5
    # datasets. Larger values speed up minibatch reads; smaller values speed
    # up single-sample random access. It is independent of the generation
    # chunk size: the writer stores complete storage chunks only. Affects
    # storage layout and read performance only, never the generated arrays.
    # Must be a positive int.
    hdf5_chunk_rows: int = 64

    # ---- EEG reference convention ----
    # Real EEG is reference-dependent: an additive, per-time-point constant (or
    # weighted combination of channels) is subtracted from every channel. The
    # generator can apply a reference to the written /eeg so datasets match the
    # convention a downstream model expects. The reference is applied to the
    # final (signal+noise) montage. cfg.noise.snr_definition selects whether
    # the primary SNR label is computed on raw or referenced components; both
    # values are stored because referencing can change their power ratio.
    #   'none'    -- no discrete electrode reference (default; the analytic
    #                n>=1 series uses the zero-spherical-mean potential gauge)
    #   'average' -- common average reference (subtract the per-sample mean
    #                across channels at each time point)
    #   'cz'      -- subtract the reference_channel trace (default 'Cz')
    #   'custom'  -- subtract reference_channel by name, or a weighted channel
    #                combination given by reference_weights (normalized to sum 1)
    # The chosen mode and the electrode names/coordinates are written into every
    # output file so reference choices are reproducible.
    reference_mode: str = 'none'
    reference_channel: str = 'Cz'
    reference_weights: Optional[List[float]] = None

    # Number of worker processes for per-sample generation. 1 (default) runs
    # serially. Values > 1 parallelize sample generation across CPU cores
    # within each write chunk. Because every sample is seeded independently
    # from (master_seed, split_id, global_sample_index), the output is
    # byte-identical regardless of n_workers -- this knob trades CPU/RAM for
    # wall-clock time only, never the data. Use -1 for "all available cores".
    # Parallel workers each hold a copy of the split's lead-field pool, so
    # raise this with available memory in mind. Must be -1 or a positive int.
    n_workers: int = 1

    # Refuse to write into a non-empty dataset directory unless replacement is
    # explicitly requested, preventing unrelated or stale target-directory
    # files from being included in the generated dataset.
    overwrite: bool = False


# =============================================================================
# Top-level config
# =============================================================================

@dataclass
class Config(_StrictFields):
    """Top-level configuration object aggregating all sub-configs.

    Fields here are TOP-LEVEL switches that affect the simulation as a whole
    (source mode, RNG seed, electrode file, output directory).

    Sub-configs are nested: cfg.head, cfg.sources, cfg.dipole, cfg.signal,
    cfg.noise, cfg.split, cfg.dataset, cfg.output.

    Mode-specific configs (cfg.multi_point, cfg.patch) are only consulted
    when the corresponding source_mode is selected.
    """
    # ------------------- Top-level switches -------------------

    # Path to an electrode CSV, OR the basename of a bundled montage.
    # Bundled montages (all placed on the model scalp sphere; positions
    # adapt to head.radii_m[-1] automatically):
    #   Standard 10-20/10-10/10-05 family:
    #     'standard_1020_19.csv', 'standard_1020_20.csv',
    #     'standard_1010_32.csv', 'standard_1010_61.csv',
    #     'standard_1010_69.csv', 'standard_1005_336.csv'
    #   Vendor nets (spherical approximations, '_spherical' suffix):
    #     'biosemi_16/32/64/128/256_spherical.csv',
    #     'gsn_hydrocel_32/64/128/256/257_spherical.csv'
    # See spherical_eeg_sim.montages for the full list. Any value not
    # resolvable to a file raises a ValueError during Config.validate().
    electrode_file: str = 'standard_1010_69.csv'

    # Output directory. Will be created if it doesn't exist.
    output_dir: str = 'output_eeg_dataset'

    # Master RNG seed. Position split, orientations, magnitudes, signals,
    # and noise all derive from this via independent sub-streams.
    master_seed: int = 862

    # Optional independent seed domains. ``position_seed`` controls the source
    # pool, its train/val/test assignment, and the background projection pool;
    # ``sample_seed`` controls per-window sources, signals, and noise. A null
    # value falls back to ``master_seed``. Separate domains support paired or
    # factorial acquisition designs without duplicating sample streams.
    position_seed: Optional[int] = None
    sample_seed: Optional[int] = None

    # Source mode determines what each sample contains.
    # 'point'        -- single dipole per sample
    # 'multi_point'  -- 1..N dipoles per sample (ConvDip-style)
    # 'patch'        -- spatially extended cortical patch (DeepSIF-style)
    source_mode: str = 'point'

    # ------------------- Sub-configs -------------------
    head: HeadModelConfig = field(default_factory=HeadModelConfig)
    sources: SourcesConfig = field(default_factory=SourcesConfig)
    dipole: DipoleConfig = field(default_factory=DipoleConfig)
    multi_point: MultiPointConfig = field(default_factory=MultiPointConfig)
    patch: PatchConfig = field(default_factory=PatchConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # ------------------- Serialization -------------------

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, path: Union[str, Path], indent: int = 2) -> None:
        """Save the configuration to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', newline='\n') as f:
            json.dump(self.to_dict(), f, indent=indent, sort_keys=True)
            f.write('\n')

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Config':
        """Reconstruct a Config from a (possibly partial) dict.

        Unknown top-level and nested keys are rejected to surface user typos.
        """
        if not isinstance(d, dict):
            raise ValueError("Configuration must be a JSON object / dictionary")
        sub_keys = {
            'head', 'sources', 'dipole', 'multi_point', 'patch',
            'signal', 'noise', 'split', 'dataset', 'output',
        }
        top_keys = {
            'electrode_file', 'output_dir', 'master_seed', 'position_seed',
            'sample_seed', 'source_mode',
        }
        unknown = set(d.keys()) - sub_keys - top_keys
        if unknown:
            raise ValueError(
                f"Unknown top-level config key(s): {sorted(unknown)}. "
                f"Expected one of: {sorted(top_keys | sub_keys)}"
            )

        def construct(section: str, section_type):
            values = d.get(section, {})
            if not isinstance(values, dict):
                raise ValueError(f"Config section {section!r} must be an object")
            allowed = {f.name for f in fields(section_type)}
            nested_unknown = set(values) - allowed
            if nested_unknown:
                raise ValueError(
                    f"Unknown config key(s) in {section!r}: "
                    f"{sorted(nested_unknown)}. Expected one of: "
                    f"{sorted(allowed)}"
                )
            return section_type(**values)

        head = construct('head', HeadModelConfig)
        sources = construct('sources', SourcesConfig)
        dipole = construct('dipole', DipoleConfig)
        multi_point = construct('multi_point', MultiPointConfig)
        patch = construct('patch', PatchConfig)
        signal = construct('signal', SignalConfig)
        noise = construct('noise', NoiseConfig)
        split = construct('split', SplitConfig)
        dataset = construct('dataset', DatasetConfig)
        output = construct('output', OutputConfig)

        top = {k: v for k, v in d.items() if k in top_keys}

        return cls(
            head=head, sources=sources, dipole=dipole,
            multi_point=multi_point, patch=patch,
            signal=signal, noise=noise, split=split,
            dataset=dataset, output=output,
            **top
        )

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> 'Config':
        """Load configuration from a JSON file."""
        with Path(path).open() as f:
            return cls.from_dict(json.load(f))

    # ------------------- Validation -------------------

    def validate(self) -> None:
        """Sanity checks on configuration consistency.

        Raises ValueError on first detected problem with a helpful message.
        """
        def require_finite(name: str, value: Any) -> float:
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise ValueError(f"{name} must be a finite real number")
            out = float(value)
            if not math.isfinite(out):
                raise ValueError(f"{name} must be finite")
            return out

        def require_int(name: str, value: Any, minimum: Optional[int] = None) -> int:
            if isinstance(value, bool) or not isinstance(value, numbers.Integral):
                raise ValueError(f"{name} must be an integer")
            out = int(value)
            if minimum is not None and out < minimum:
                raise ValueError(f"{name} must be >= {minimum}")
            return out

        def require_pair(name: str, value: Any) -> tuple[float, float]:
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError(f"{name} must be a two-value [low, high] pair")
            return require_finite(f"{name}[0]", value[0]), require_finite(
                f"{name}[1]", value[1]
            )

        # Top-level
        if self.source_mode not in ('point', 'multi_point', 'patch'):
            raise ValueError(
                f"source_mode must be 'point', 'multi_point', or 'patch'; "
                f"got {self.source_mode!r}"
            )
        if not isinstance(self.electrode_file, str) or not self.electrode_file:
            raise ValueError("electrode_file must be a non-empty string")
        require_int("master_seed", self.master_seed, 0)
        if self.position_seed is not None:
            require_int("position_seed", self.position_seed, 0)
        if self.sample_seed is not None:
            require_int("sample_seed", self.sample_seed, 0)
        if not isinstance(self.output_dir, str) or not self.output_dir:
            raise ValueError("output_dir must be a non-empty string")
        # Resolve electrode_file (filesystem path or bundled montage name)
        # and verify it exists.
        from .core.electrodes import resolve_electrode_path
        resolved = resolve_electrode_path(self.electrode_file)
        if not Path(resolved).is_file():
            raise ValueError(
                f"electrode_file does not resolve to an existing CSV: "
                f"{self.electrode_file!r}. Provide a filesystem path or one "
                f"of the bundled montage names (e.g. 'standard_1010_69.csv', "
                f"'standard_1005_336.csv', 'biosemi_64_spherical.csv', "
                f"'gsn_hydrocel_128_spherical.csv'); see "
                f"spherical_eeg_sim.montages for the full list."
            )

        # Head model
        if len(self.head.radii_m) != 4 or len(self.head.conductivities_S_per_m) != 4:
            raise ValueError(
                "The analytic forward model requires exactly four radii and "
                "four conductivities (brain, CSF, skull, scalp)"
            )
        for i, r in enumerate(self.head.radii_m):
            if require_finite(f"head.radii_m[{i}]", r) <= 0:
                raise ValueError("All radii must be positive")
        for i, s in enumerate(self.head.conductivities_S_per_m):
            if require_finite(f"head.conductivities_S_per_m[{i}]", s) <= 0:
                raise ValueError("All conductivities must be positive")
        if any(r1 >= r2 for r1, r2 in zip(self.head.radii_m, self.head.radii_m[1:])):
            raise ValueError("radii_m must be strictly increasing (innermost to outermost)")
        require_int("head.n_legendre", self.head.n_legendre)
        if self.head.n_legendre < 40:
            raise ValueError(
                f"n_legendre={self.head.n_legendre} is below the audited "
                f"supported range; use >= 40 and run "
                f"scripts/legendre_convergence.py for the configured geometry"
            )
        from .core.forward import MAX_NUMERIC_LEGENDRE_ORDER
        if self.head.n_legendre > MAX_NUMERIC_LEGENDRE_ORDER:
            raise ValueError(
                f"n_legendre={self.head.n_legendre} exceeds the float64 "
                f"recurrence bound {MAX_NUMERIC_LEGENDRE_ORDER}. The unscaled "
                "transfer recurrence is round-off dominated above this order; "
                f"use n_legendre <= {MAX_NUMERIC_LEGENDRE_ORDER} and see "
                "scripts/legendre_convergence.py for the realised error of the "
                "configured geometry."
            )
        tail_tolerance = require_finite(
            "head.truncation_tail_tolerance",
            self.head.truncation_tail_tolerance,
        )
        if tail_tolerance <= 0.0:
            raise ValueError("head.truncation_tail_tolerance must be > 0")

        # Sources
        brain_r_mm = self.head.radii_m[0] * 1000
        cmin, cmax = require_pair(
            "sources.cortical_radius_range_mm",
            self.sources.cortical_radius_range_mm,
        )
        if cmin >= cmax:
            raise ValueError("cortical_radius_range_mm[0] must be < [1]")
        if cmin <= 0:
            raise ValueError("cortical_radius_range_mm[0] must be > 0")
        if cmax >= brain_r_mm:
            raise ValueError(
                f"cortical_radius_range_mm[1]={cmax} mm must be < brain radius "
                f"{brain_r_mm:.1f} mm"
            )
        smin, smax = require_pair(
            "sources.subcortical_radius_range_mm",
            self.sources.subcortical_radius_range_mm,
        )
        if smin >= smax:
            raise ValueError("subcortical_radius_range_mm[0] must be < [1]")
        if smax >= brain_r_mm:
            raise ValueError(
                f"subcortical_radius_range_mm[1]={smax} mm must be < brain radius "
                f"{brain_r_mm:.1f} mm"
            )
        if smin <= 0:
            raise ValueError("subcortical_radius_range_mm[0] must be > 0")
        require_finite("sources.subcortical_fraction", self.sources.subcortical_fraction)
        if not (0.0 <= self.sources.subcortical_fraction <= 1.0):
            raise ValueError("subcortical_fraction must be in [0, 1]")
        require_int("sources.n_cortical_positions", self.sources.n_cortical_positions, 0)
        require_int("sources.n_subcortical_positions", self.sources.n_subcortical_positions, 0)
        if (self.sources.n_cortical_positions + self.sources.n_subcortical_positions) < 1:
            raise ValueError("Sum of cortical + subcortical positions must be >= 1")
        if self.source_mode == 'patch' and self.sources.n_cortical_positions < 3:
            raise ValueError(
                "Patch mode requires at least 3 cortical positions so each "
                "position-level split has cortical support"
            )
        if self.source_mode != 'patch':
            minimum_positions = (
                1 if self.split.policy == 'shared'
                and self.split.shared_position_fraction == 1.0 else 3
            )
            if (self.sources.subcortical_fraction < 1.0
                    and self.sources.n_cortical_positions < minimum_positions):
                raise ValueError(
                    f"n_cortical_positions must be >= {minimum_positions} when "
                    "subcortical_fraction < 1 so every position split has "
                    "an eligible superficial source"
                )
            if (self.sources.subcortical_fraction > 0.0
                    and self.sources.n_subcortical_positions < minimum_positions):
                raise ValueError(
                    f"n_subcortical_positions must be >= {minimum_positions} when "
                    "subcortical_fraction > 0 so every position split has "
                    "an eligible deep source"
                )

        constructed_maxima_mm = []
        if self.sources.n_cortical_positions > 0:
            constructed_maxima_mm.append(cmax)
        if (self.source_mode != 'patch'
                and self.sources.n_subcortical_positions > 0):
            constructed_maxima_mm.append(smax)
        if self.sources.cortical_sampling not in ('fibonacci', 'uniform_random'):
            raise ValueError(
                f"cortical_sampling must be 'fibonacci' or 'uniform_random'; "
                f"got {self.sources.cortical_sampling!r}"
            )
        if self.sources.subcortical_sampling not in ('uniform_volume', 'fibonacci_shells'):
            raise ValueError(
                f"subcortical_sampling must be 'uniform_volume' or "
                f"'fibonacci_shells'; got {self.sources.subcortical_sampling!r}"
            )

        # Dipole
        if self.dipole.orientation_mode not in ('uniform_s2', 'codebook', 'cortex_normal'):
            raise ValueError(
                f"orientation_mode must be 'uniform_s2', 'codebook', or "
                f"'cortex_normal'; got {self.dipole.orientation_mode!r}"
            )
        require_int(
            "dipole.codebook_tangential_directions",
            self.dipole.codebook_tangential_directions,
            1,
        )
        if self.dipole.magnitude_distribution not in ('log_uniform', 'uniform', 'fixed'):
            raise ValueError(
                f"magnitude_distribution must be 'log_uniform', 'uniform', "
                f"or 'fixed'; got {self.dipole.magnitude_distribution!r}"
            )
        for label, rng in (
            ('cortical_magnitude_range_nAm', self.dipole.cortical_magnitude_range_nAm),
            ('subcortical_magnitude_range_nAm', self.dipole.subcortical_magnitude_range_nAm),
        ):
            low, high = require_pair(f"dipole.{label}", rng)
            if low <= 0:
                raise ValueError(f"{label} low must be > 0")
            if high < low:
                raise ValueError(f"{label} must satisfy low <= high")

        # Multi-point
        require_int("multi_point.n_sources_min", self.multi_point.n_sources_min, 1)
        require_int("multi_point.n_sources_max", self.multi_point.n_sources_max, 1)
        if self.multi_point.n_sources_max < self.multi_point.n_sources_min:
            raise ValueError("multi_point.n_sources_max must be >= n_sources_min")
        require_finite("multi_point.min_separation_mm", self.multi_point.min_separation_mm)
        if self.multi_point.min_separation_mm < 0:
            raise ValueError("multi_point.min_separation_mm must be >= 0")
        # The widest separation any two sources can reach is the diameter of the
        # outermost sampled shell. A larger requirement is unsatisfiable for
        # every draw, so reject it here rather than after the position pool and
        # lead-field tensor have been built.
        if (self.source_mode == 'multi_point'
                and self.multi_point.n_sources_max > 1):
            reachable_radius_mm = max(
                self.sources.cortical_radius_range_mm[1],
                self.sources.subcortical_radius_range_mm[1],
            )
            if self.multi_point.min_separation_mm > 2.0 * reachable_radius_mm:
                raise ValueError(
                    f"multi_point.min_separation_mm="
                    f"{self.multi_point.min_separation_mm} mm exceeds the "
                    f"{2.0 * reachable_radius_mm:g} mm diameter of the "
                    f"configured source space, so no pair of positions can "
                    f"satisfy it. Lower min_separation_mm, widen "
                    f"sources.cortical_radius_range_mm / "
                    f"subcortical_radius_range_mm, or set "
                    f"multi_point.n_sources_max=1."
                )
        require_int(
            "multi_point.max_separation_attempts",
            self.multi_point.max_separation_attempts,
            1,
        )
        require_finite("multi_point.time_coupling", self.multi_point.time_coupling)
        if not (0.0 <= self.multi_point.time_coupling <= 1.0):
            raise ValueError("multi_point.time_coupling must be in [0, 1]")
        if self.multi_point.time_coupling_values is not None:
            if not isinstance(self.multi_point.time_coupling_values, list) or not self.multi_point.time_coupling_values:
                raise ValueError("multi_point.time_coupling_values must be null or a non-empty list")
            for i, value in enumerate(self.multi_point.time_coupling_values):
                require_finite(f"multi_point.time_coupling_values[{i}]", value)
                if not 0.0 <= value <= 1.0:
                    raise ValueError("multi_point.time_coupling_values entries must be in [0, 1]")
        if self.multi_point.separation_failure_policy not in ('raise', 'accept_and_flag'):
            raise ValueError(
                "multi_point.separation_failure_policy must be 'raise' or 'accept_and_flag'"
            )

        # Patch
        if self.patch.falloff_profile not in ('gaussian', 'uniform', 'cosine'):
            raise ValueError(
                f"patch.falloff_profile must be 'gaussian', 'uniform', or "
                f"'cosine'; got {self.patch.falloff_profile!r}"
            )
        dlow, dhigh = require_pair(
            "patch.diameter_range_mm", self.patch.diameter_range_mm
        )
        if dlow <= 0 or dhigh < dlow:
            raise ValueError("patch.diameter_range_mm must be (low > 0, high >= low)")
        require_finite("patch.dipoles_per_mm2", self.patch.dipoles_per_mm2)
        if self.patch.dipoles_per_mm2 <= 0:
            raise ValueError("patch.dipoles_per_mm2 must be > 0")
        if not isinstance(self.patch.report_overlap, bool):
            raise ValueError("patch.report_overlap must be boolean")
        if self.source_mode == 'patch':
            self._warn_if_patch_support_is_sparse(dlow)

        # Signal
        require_finite("signal.sampling_freq_hz", self.signal.sampling_freq_hz)
        if self.signal.sampling_freq_hz <= 0:
            raise ValueError("sampling_freq_hz must be > 0")
        require_int("signal.window_samples", self.signal.window_samples)
        if self.signal.window_samples < 10:
            raise ValueError("window_samples must be >= 10")
        require_int("signal.n_components_min", self.signal.n_components_min, 1)
        require_int("signal.n_components_max", self.signal.n_components_max, 1)
        if self.signal.n_components_max < self.signal.n_components_min:
            raise ValueError("n_components_max must be >= n_components_min")
        for name in ('component_amplitude_jitter', 'zero_mean_signal'):
            if not isinstance(getattr(self.signal, name), bool):
                raise ValueError(f"signal.{name} must be boolean")
        require_finite(
            "signal.component_dirichlet_concentration",
            self.signal.component_dirichlet_concentration,
        )
        if self.signal.component_dirichlet_concentration <= 0:
            raise ValueError(
                "signal.component_dirichlet_concentration must be > 0"
            )
        if set(self.signal.bands_hz.keys()) != set(self.signal.band_probabilities.keys()):
            raise ValueError(
                "Keys of bands_hz and band_probabilities must match. "
                f"bands_hz: {sorted(self.signal.bands_hz.keys())}, "
                f"probabilities: {sorted(self.signal.band_probabilities.keys())}"
            )
        order = self.signal.band_order
        if (not isinstance(order, (list, tuple)) or not order
                or not all(isinstance(name, str) and name for name in order)
                or len(set(order)) != len(order)):
            raise ValueError(
                "signal.band_order must be a non-empty sequence of unique names"
            )
        if set(order) != set(self.signal.bands_hz.keys()):
            raise ValueError(
                "signal.band_order must contain every bands_hz key exactly once; "
                f"got {list(order)!r}"
            )
        for name, p in self.signal.band_probabilities.items():
            require_finite(f"signal.band_probabilities[{name!r}]", p)
        if any(p < 0 for p in self.signal.band_probabilities.values()):
            raise ValueError("All band probabilities must be non-negative")
        if sum(self.signal.band_probabilities.values()) <= 0:
            raise ValueError("Band probabilities must sum to a positive value")
        from .signals.source_signals import _validate_band_upper_edge

        for name, lh in self.signal.bands_hz.items():
            low, high = require_pair(f"signal.bands_hz[{name!r}]", lh)
            if low >= high or low < 0:
                raise ValueError(f"bands_hz[{name!r}] must be [low, high] with 0 <= low < high")
            try:
                _validate_band_upper_edge(name, high, self.signal.sampling_freq_hz)
            except ValueError as error:
                raise ValueError(f"signal.bands_hz[{name!r}]: {error}") from error
        if self.signal.low_freq_band_policy not in ('restrict', 'keep'):
            raise ValueError(
                "signal.low_freq_band_policy must be 'restrict' or 'keep'; "
                f"got {self.signal.low_freq_band_policy!r}"
            )
        require_finite("signal.min_band_cycles", self.signal.min_band_cycles)
        if self.signal.min_band_cycles <= 0:
            raise ValueError("signal.min_band_cycles must be > 0")
        if self.signal.low_freq_band_policy == 'restrict':
            window_seconds = (
                float(self.signal.window_samples)
                / float(self.signal.sampling_freq_hz)
            )
            if not any(
                float(self.signal.bands_hz[name][0]) * window_seconds
                >= float(self.signal.min_band_cycles)
                for name in self.signal.band_order
            ):
                raise ValueError(
                    "signal.low_freq_band_policy='restrict' leaves no "
                    "resolvable bands; lengthen window_samples, lower "
                    "min_band_cycles, change bands_hz, or use 'keep'"
                )

        # Noise
        for label, w in (
            ('mix_physiological', self.noise.mix_physiological),
            ('mix_pink', self.noise.mix_pink),
            ('mix_white', self.noise.mix_white),
        ):
            require_finite(f"noise.{label}", w)
            if w < 0:
                raise ValueError(f"noise.{label} must be >= 0 (got {w})")
        mix_sum = self.noise.mix_physiological + self.noise.mix_pink + self.noise.mix_white
        if mix_sum <= 0:
            raise ValueError("Noise mix proportions must sum to a positive value")
        snr_low, snr_high = require_pair("noise.snr_db_range", self.noise.snr_db_range)
        if snr_low >= snr_high:
            raise ValueError("snr_db_range must be [low, high] with low < high")
        if self.noise.snr_definition not in ('referenced', 'raw'):
            raise ValueError(
                "noise.snr_definition must be 'referenced' or 'raw'; "
                f"got {self.noise.snr_definition!r}"
            )
        if self.noise.noise_model not in ('snr_target', 'absolute_floor'):
            raise ValueError(
                "noise.noise_model must be 'snr_target' or 'absolute_floor'; "
                f"got {self.noise.noise_model!r}"
            )
        require_finite("noise.absolute_noise_rms_uV", self.noise.absolute_noise_rms_uV)
        if self.noise.absolute_noise_rms_uV <= 0:
            raise ValueError(
                "noise.absolute_noise_rms_uV must be > 0 (got "
                f"{self.noise.absolute_noise_rms_uV})"
            )
        require_int("noise.n_background_sources", self.noise.n_background_sources, 1)
        require_int("noise.background_pool_size", self.noise.background_pool_size, 1)
        if self.noise.background_pool_size < self.noise.n_background_sources:
            raise ValueError(
                "noise.background_pool_size must be >= n_background_sources"
            )
        bg_low, bg_high = require_pair(
            "noise.background_radius_range_mm",
            self.noise.background_radius_range_mm,
        )
        if bg_low <= 0 or bg_high <= bg_low or bg_high >= brain_r_mm:
            raise ValueError(
                "noise.background_radius_range_mm must satisfy 0 < low < high "
                "< brain radius"
            )
        maximum_source_radius_m = max(
            *constructed_maxima_mm,
            bg_high,
        ) * 1e-3
        scalp_radius_m = float(self.head.radii_m[-1])
        q = maximum_source_radius_m / scalp_radius_m
        tail_indicator = q ** (self.head.n_legendre + 1) / (1.0 - q)
        if tail_indicator > tail_tolerance:
            raise ValueError(
                "The configured Legendre expansion converges too slowly over "
                "the requested source domain: with maximum signal/background "
                f"dipole radius {maximum_source_radius_m * 1000.0:g} mm, scalp "
                f"radius {scalp_radius_m * 1000.0:g} mm, and "
                f"n_legendre={self.head.n_legendre}, the geometric-tail "
                f"indicator q**(N+1)/(1-q) is {tail_indicator:.6g}, which "
                "exceeds head.truncation_tail_tolerance="
                f"{tail_tolerance:.6g}. Increase n_legendre within the "
                f"supported range, reduce the maximum source radius, increase "
                "the scalp radius, or explicitly raise the tolerance after "
                "auditing the configured geometry with "
                "scripts/legendre_convergence.py."
            )
        require_finite(
            "noise.background_amplitude_concentration",
            self.noise.background_amplitude_concentration,
        )
        if self.noise.background_amplitude_concentration <= 0:
            raise ValueError(
                "noise.background_amplitude_concentration must be > 0"
            )
        require_finite("noise.background_freq_alpha", self.noise.background_freq_alpha)
        if self.noise.background_freq_alpha < 0:
            raise ValueError("background_freq_alpha must be >= 0")

        # Split
        require_int("split.depth_bins", self.split.depth_bins, 1)
        if not isinstance(self.split.stratify_by_depth, bool):
            raise ValueError("split.stratify_by_depth must be boolean")
        if self.split.policy not in ('position', 'region', 'shared'):
            raise ValueError("split.policy must be 'position', 'region', or 'shared'")
        if self.source_mode == 'patch' and self.split.policy == 'shared':
            raise ValueError("Patch mode supports split.policy='position' or 'region'")
        if self.split.policy == 'region':
            require_int("split.region_count", self.split.region_count, 3)
            require_int("split.region_rotation_seed", self.split.region_rotation_seed, 0)
            if require_finite("split.buffer_mm", self.split.buffer_mm) < 0:
                raise ValueError("split.buffer_mm must be >= 0")
        elif any(getattr(self.split, name) is not None for name in (
            'region_count', 'buffer_mm', 'region_rotation_seed',
        )):
            raise ValueError("Region settings require split.policy='region'")
        if self.split.policy == 'shared':
            fraction = require_finite(
                "split.shared_position_fraction", self.split.shared_position_fraction,
            )
            if not 0 < fraction <= 1:
                raise ValueError("split.shared_position_fraction must be in (0, 1]")
        elif self.split.shared_position_fraction is not None:
            raise ValueError("split.shared_position_fraction requires split.policy='shared'")

        # Dataset sizing
        for name in ("n_train", "n_val", "n_test"):
            require_int(f"dataset.{name}", getattr(self.dataset, name), 1)

        # Output
        if self.output.dtype not in ('float32', 'float64'):
            raise ValueError(
                f"output.dtype must be 'float32' or 'float64'; got {self.output.dtype!r}"
            )
        if self.output.format not in ('hdf5', 'npz', 'mat'):
            raise ValueError(
                f"output.format must be 'hdf5', 'npz', or 'mat'; "
                f"got {self.output.format!r}"
            )
        if self.output.eeg_normalization not in ('none', 'per_sample_rms'):
            raise ValueError(
                "output.eeg_normalization must be 'none' or 'per_sample_rms'; "
                f"got {self.output.eeg_normalization!r}"
            )
        require_int("output.chunk_samples", self.output.chunk_samples, 1)
        require_int("output.hdf5_chunk_rows", self.output.hdf5_chunk_rows, 1)
        require_int("output.n_workers", self.output.n_workers)
        if self.output.n_workers != -1 and self.output.n_workers < 1:
            raise ValueError("output.n_workers must be -1 (all cores) or a positive int")
        for name in ("save_components", "save_leadfield", "save_graph",
                     "diagnostic_plots", "compress", "overwrite"):
            if not isinstance(getattr(self.output, name), bool):
                raise ValueError(f"output.{name} must be boolean")
        require_int(
            "output.diagnostic_sample_count",
            self.output.diagnostic_sample_count,
            1,
        )
        if self.output.reference_mode not in ('none', 'average', 'cz', 'custom'):
            raise ValueError(
                "output.reference_mode must be 'none', 'average', 'cz', or "
                f"'custom'; got {self.output.reference_mode!r}"
            )
        if not isinstance(self.output.reference_channel, str) or not self.output.reference_channel:
            raise ValueError("output.reference_channel must be a non-empty string")
        if self.output.reference_weights is not None:
            weights = self.output.reference_weights
            if not isinstance(weights, (list, tuple)) or len(weights) == 0:
                raise ValueError(
                    "output.reference_weights must be a non-empty list or tuple"
                )
            if not all(isinstance(x, numbers.Real) and not isinstance(x, bool)
                       and math.isfinite(float(x)) for x in weights):
                raise ValueError("output.reference_weights must contain finite numbers")
            if abs(float(sum(weights))) < 1e-12:
                raise ValueError("output.reference_weights must not sum to zero")
        if self.output.graph_method not in ('knn', 'delaunay', 'distance'):
            raise ValueError(
                "output.graph_method must be 'knn', 'delaunay', or 'distance'; "
                f"got {self.output.graph_method!r}"
            )
        require_int("output.graph_k", self.output.graph_k, 1)
        if self.output.save_graph and self.output.graph_method == 'knn':
            # The exporter enforces this too, but only once the montage is
            # loaded. Checking it here means a standalone cfg.validate() and a
            # pipeline preflight agree, instead of the constraint surfacing
            # only from the deeper call.
            from .core.electrodes import resolve_electrode_path, load_electrodes
            try:
                _n_ch = len(load_electrodes(
                    resolve_electrode_path(self.electrode_file)
                ).names)
            except Exception:  # pragma: no cover - montage errors reported above
                _n_ch = None
            if _n_ch is not None and self.output.graph_k >= _n_ch:
                raise ValueError(
                    f"output.graph_k={self.output.graph_k} must be smaller "
                    f"than the montage electrode count ({_n_ch}) for "
                    "graph_method='knn'"
                )
        if self.output.graph_distance_threshold_m is not None:
            require_finite(
                "output.graph_distance_threshold_m",
                self.output.graph_distance_threshold_m,
            )
            if self.output.graph_distance_threshold_m <= 0:
                raise ValueError(
                    "output.graph_distance_threshold_m must be a positive "
                    "distance in metres"
                )
        if self.output.graph_method == 'distance':
            thr = self.output.graph_distance_threshold_m
            if thr is None:
                raise ValueError(
                    "output.graph_distance_threshold_m must be a positive "
                    "distance in metres for graph_method='distance'"
                )

    def _warn_if_patch_support_is_sparse(self, min_diameter_mm: float) -> None:
        """Warn when the position pool cannot resolve the smallest patch.

        Patch support is recruited from the split-local cortical mesh, so the
        realised density of active dipoles is set by
        ``sources.n_cortical_positions`` divided across the three splits, not by
        ``patch.dipoles_per_mm2``. Under the region policy each split keeps every
        pool position inside its own sectors, so its mesh has the full pool
        density instead of one third of it. That parameter is an upper cap: it binds only
        when the pool is denser than the cap. A pool that leaves the smallest
        configured patch with a handful of dipoles produces something closer to
        a small cluster of point sources than a distributed activation, so the
        expected count is reported before a long run begins.
        """
        r_lo, r_hi = self.sources.cortical_radius_range_mm
        mean_radius_mm = 0.5 * (float(r_lo) + float(r_hi))
        shell_area_mm2 = 4.0 * math.pi * mean_radius_mm ** 2
        # The default policy splits the position pool into three equal support
        # meshes that each cover the sphere; the region policy keeps the pool
        # density inside every split region.
        meshes_per_pool = 1.0 if self.split.policy == 'region' else 3.0
        per_split_positions = self.sources.n_cortical_positions / meshes_per_pool
        realised_density = per_split_positions / shell_area_mm2
        smallest_patch_area_mm2 = math.pi * (float(min_diameter_mm) / 2.0) ** 2
        expected_active = realised_density * smallest_patch_area_mm2

        if expected_active < MIN_EXPECTED_PATCH_SUPPORT:
            needed = math.ceil(
                meshes_per_pool * MIN_EXPECTED_PATCH_SUPPORT * shell_area_mm2
                / smallest_patch_area_mm2
            )
            warnings.warn(
                f"Patch support is sparse: with "
                f"sources.n_cortical_positions={self.sources.n_cortical_positions} "
                f"the split-local cortical mesh realises about "
                f"{realised_density:.3f} positions/mm^2, so the smallest "
                f"configured patch ({float(min_diameter_mm):.1f} mm diameter) "
                f"recruits roughly {expected_active:.0f} active dipole(s) rather "
                f"than a distributed activation. The realised density is set by "
                f"the position pool, not by patch.dipoles_per_mm2="
                f"{self.patch.dipoles_per_mm2:g}, which is an upper cap and does "
                f"not bind here. Raise sources.n_cortical_positions to about "
                f"{needed} for roughly {MIN_EXPECTED_PATCH_SUPPORT} dipoles in "
                f"the smallest patch, or raise "
                f"patch.diameter_range_mm[0]. The realised support of every "
                f"sample is stored in ground_truth/n_active_per_patch.",
                RuntimeWarning,
                stacklevel=3,
            )


# Convenience: a reduced configuration for installation smoke tests.
def quickstart_config() -> Config:
    """Reduced 800/100/100-sample configuration for smoke tests."""
    cfg = Config()
    cfg.dataset.n_train = 800
    cfg.dataset.n_val = 100
    cfg.dataset.n_test = 100
    cfg.sources.n_cortical_positions = 500
    cfg.sources.n_subcortical_positions = 100
    cfg.head.n_legendre = 40
    cfg.output.diagnostic_plots = False
    cfg.output.chunk_samples = 500
    return cfg
