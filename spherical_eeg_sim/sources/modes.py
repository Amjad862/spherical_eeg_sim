"""
Source-mode dispatchers and per-sample pipeline orchestration.

Three source modes are implemented:

  1. 'point'        -- single dipole per sample. Ground truth is a single
                       (position, orientation, magnitude) triple.
  2. 'multi_point'  -- 1..N dipoles per sample, each with its own
                       position/orientation/magnitude/time-course. Ground
                       truth includes the complete padded temporal construction.
  3. 'patch'        -- spatially extended cortical patch with finite support.
                       Ground truth includes center, extent, magnitude,
                       and exact position-pool support indices and weights. The
                       forward EEG is the superposition of active dipoles.

All three modes share infrastructure (lead-field tensor, electrode array,
noise generation, SNR application, output writing). All three modes
populate ``extras['position_idx_in_pool']`` so the pipeline can record
per-sample provenance in the HDF5 output (``/ground_truth/position_pool_idx``).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence
import numpy as np

from ..config import Config
from ..core.forward import forward_potential_basis, potentials_from_basis
from ..ragged import RaggedArray
from .sampling import (
    sample_unit_vectors_s2,
    sample_orientations,
    sample_magnitudes,
    world_orientation_to_spherical_frame,
    cartesian_to_spherical,
)


# =============================================================================
# Shared sample-batch struct
# =============================================================================

@dataclass
class SampleBatch:
    """A batch of generated samples (clean EEG, ground truth, metadata).

    The 'clean' EEG is signal-only (no noise applied). Noise is added in a
    later pipeline stage.

    Ground-truth fields are kept as numpy arrays; mode-specific extras
    (e.g. multi-point counts, patch diameters) are in 'extras'.
    """
    eeg_signal: np.ndarray           # (B, N_channels, T)
    positions_xyz: np.ndarray        # (B, 3) for point/patch; (B, max_K, 3) for multi
    orientations_world: np.ndarray   # same shape as positions_xyz
    magnitudes_nAm: np.ndarray       # (B,) or (B, max_K)
    population: np.ndarray           # (B,) or (B, max_K), 0=cortical 1=subcortical
    extras: Dict[str, Any]


# =============================================================================
# Helper: synthesize EEG from (position-index, world-orientation, magnitude)
# =============================================================================

def _eeg_from_dipole(
    pos_xyz: np.ndarray,                # (3,)
    L_basis: np.ndarray,                # (N_channels, 3) basis lead field at this position
    orientation_world: np.ndarray,      # (3,) unit
    magnitude_Am: float,
    time_course: np.ndarray,            # (T,) standard-normalized
) -> np.ndarray:
    """Compute scalp EEG (N_channels, T) for a single dipole."""
    local = world_orientation_to_spherical_frame(orientation_world, pos_xyz)
    scalp_unit = potentials_from_basis(L_basis, local[0], local[1], local[2])
    return magnitude_Am * scalp_unit[:, None] * time_course[None, :]


def _draw_population_index(
    rng: np.random.Generator,
    f_sub: float,
    cort_pool_idx: np.ndarray,
    sub_pool_idx: np.ndarray,
) -> int:
    """Draw one position-pool index, mixing cortical/subcortical per f_sub.

    Falls back to whichever population has positions when the other is empty,
    so callers never see an empty-array crash.
    """
    want_sub = rng.uniform() < f_sub
    if want_sub and sub_pool_idx.size > 0:
        return int(rng.choice(sub_pool_idx))
    if cort_pool_idx.size > 0:
        return int(rng.choice(cort_pool_idx))
    if sub_pool_idx.size > 0:
        return int(rng.choice(sub_pool_idx))
    raise ValueError("Both cortical and subcortical pools are empty")


# =============================================================================
# Mode: 'point' -- single dipole per sample
# =============================================================================

def generate_point_samples(
    cfg: Config,
    n_samples: int,
    position_pool_xyz: np.ndarray,      # (P, 3) all eligible positions for this split
    leadfield_pool: np.ndarray,         # (P, N_channels, 3) basis lead fields
    population_pool: np.ndarray,        # (P,) 0=cortical 1=subcortical
    time_courses: np.ndarray,           # (n_samples, T) standard-normalized
    rng: np.random.Generator,
) -> SampleBatch:
    """Generate single-dipole samples.

    Each sample picks one position (with subcortical_fraction mixing),
    one orientation, one magnitude, and uses the precomputed time course.
    """
    N_channels = leadfield_pool.shape[1]
    T = time_courses.shape[1]

    cort_pool_idx = np.where(population_pool == 0)[0]
    sub_pool_idx = np.where(population_pool == 1)[0]
    if cort_pool_idx.size == 0 and sub_pool_idx.size == 0:
        raise ValueError("Point mode requires at least one population to be non-empty")

    f_sub = cfg.sources.subcortical_fraction
    # Per-sample stochastic population assignment: each sample is independently
    # subcortical with probability f_sub (the same scheme multi-point mode
    # uses). This keeps the realized subcortical fraction correct in
    # expectation for any sample count -- including a single sample -- so the
    # configured fraction is respected regardless of how the split is divided
    # into streaming chunks. If the drawn population has no positions, the
    # sample falls back to the other population.
    want_sub = rng.random(n_samples) < f_sub
    if sub_pool_idx.size == 0:
        want_sub[:] = False
    if cort_pool_idx.size == 0:
        want_sub[:] = True

    pos_idx = np.empty(n_samples, dtype=np.int64)
    n_want_sub = int(want_sub.sum())
    n_want_cort = n_samples - n_want_sub
    if n_want_cort > 0:
        pos_idx[~want_sub] = rng.choice(cort_pool_idx, size=n_want_cort, replace=True)
    if n_want_sub > 0:
        pos_idx[want_sub] = rng.choice(sub_pool_idx, size=n_want_sub, replace=True)

    positions = position_pool_xyz[pos_idx]                       # (B, 3)
    L = leadfield_pool[pos_idx]                                  # (B, N, 3)
    populations = population_pool[pos_idx]                        # (B,)

    orientations_world, _ = sample_orientations(
        n_samples,
        cfg.dipole.orientation_mode,
        positions,
        rng,
        codebook_tangential_directions=cfg.dipole.codebook_tangential_directions,
    )

    magnitudes_nAm = np.empty(n_samples)
    is_cort_sample = (populations == 0)
    if is_cort_sample.any():
        magnitudes_nAm[is_cort_sample] = sample_magnitudes(
            int(is_cort_sample.sum()),
            cfg.dipole.cortical_magnitude_range_nAm,
            cfg.dipole.magnitude_distribution,
            rng,
        )
    if (~is_cort_sample).any():
        magnitudes_nAm[~is_cort_sample] = sample_magnitudes(
            int((~is_cort_sample).sum()),
            cfg.dipole.subcortical_magnitude_range_nAm,
            cfg.dipole.magnitude_distribution,
            rng,
        )

    magnitudes_Am = magnitudes_nAm * 1e-9

    eeg = np.zeros((n_samples, N_channels, T), dtype=np.float64)
    for i in range(n_samples):
        eeg[i] = _eeg_from_dipole(
            positions[i], L[i], orientations_world[i],
            magnitudes_Am[i], time_courses[i]
        )

    return SampleBatch(
        eeg_signal=eeg,
        positions_xyz=positions,
        orientations_world=orientations_world,
        magnitudes_nAm=magnitudes_nAm,
        population=populations,
        extras={
            'position_idx_in_pool': pos_idx,
        },
    )


# =============================================================================
# Mode: 'multi_point' -- 1..N dipoles per sample
# =============================================================================

def generate_multi_point_samples(
    cfg: Config,
    n_samples: int,
    position_pool_xyz: np.ndarray,
    leadfield_pool: np.ndarray,
    population_pool: np.ndarray,
    time_courses: np.ndarray,
    rng: np.random.Generator,
    *,
    band_probabilities: Optional[Dict[str, float]] = None,
    zero_mean_signal: Optional[bool] = None,
    primary_signal_metadata: Optional[Dict[str, np.ndarray]] = None,
) -> SampleBatch:
    """Generate multi-dipole samples.

    Each sample has K ~ Uniform[n_min, n_max] dipoles. Each dipole has its
    own position, orientation, magnitude, and source time-course. The primary
    dipole uses ``time_courses[i]`` and its metadata is supplied by the caller.
    Each additional dipole gets a freshly generated time course using the same
    effective frequency-band policy and zero-mean policy as the primary course,
    then is linearly blended toward the primary by the sample's configured
    coupling value and renormalized to unit standard deviation.  A finite set
    in ``time_coupling_values`` is sampled per window when present; otherwise
    the scalar ``time_coupling`` is used.

    Ground-truth tensors are padded to max_K with NaN/-1/0 for unused slots.
    The actual normalized source courses and the independent sinusoid bases
    used to construct them are stored for every active dipole.  Blend weights
    and the post-blend standard-deviation divisor make temporally coupled
    courses exactly reconstructable, rather than assigning a single frequency
    label to a mixed-band course.

    ``extras['separation_violated']`` is a per-sample boolean flag that is True
    when the min-separation constraint could not be satisfied for at least one
    dipole after the rejection-sampling attempts were exhausted.
    """
    from ..signals.source_signals import generate_signal_batch

    if band_probabilities is None:
        band_probabilities = cfg.signal.band_probabilities
    if zero_mean_signal is None:
        zero_mean_signal = cfg.signal.zero_mean_signal

    N_channels = leadfield_pool.shape[1]
    T = time_courses.shape[1]
    n_min = cfg.multi_point.n_sources_min
    n_max = cfg.multi_point.n_sources_max
    n_comp_max = cfg.signal.n_components_max
    min_sep_m = cfg.multi_point.min_separation_mm * 1e-3

    if n_min == n_max:
        K_per = np.full(n_samples, n_min, dtype=np.int32)
    else:
        K_per = rng.integers(n_min, n_max + 1, size=n_samples).astype(np.int32)

    positions_padded = np.full((n_samples, n_max, 3), np.nan)
    orientations_padded = np.full((n_samples, n_max, 3), np.nan)
    magnitudes_padded = np.full((n_samples, n_max), np.nan)
    population_padded = np.full((n_samples, n_max), -1, dtype=np.int8)
    pos_idx_padded = np.full((n_samples, n_max), -1, dtype=np.int64)

    # Per-source temporal metadata. Active sources are filled below; unused
    # source slots remain padded with -1/NaN/0.
    source_time_courses = np.full((n_samples, n_max, T), np.nan, dtype=np.float64)
    source_base_band_idx = np.full((n_samples, n_max), -1, dtype=np.int32)
    source_base_frequencies_hz = np.full(
        (n_samples, n_max, n_comp_max), np.nan, dtype=np.float64
    )
    source_base_phases_rad = np.full(
        (n_samples, n_max, n_comp_max), np.nan, dtype=np.float64
    )
    source_base_component_amplitudes = np.full(
        (n_samples, n_max, n_comp_max), np.nan, dtype=np.float64
    )
    source_base_n_components = np.zeros((n_samples, n_max), dtype=np.int32)
    source_primary_blend_weight = np.zeros((n_samples, n_max), dtype=np.float64)
    source_independent_blend_weight = np.zeros((n_samples, n_max), dtype=np.float64)
    source_blend_std = np.full((n_samples, n_max), np.nan, dtype=np.float64)

    eeg = np.zeros((n_samples, N_channels, T), dtype=np.float64)

    is_cortical = (population_pool == 0)
    is_subcortical = (population_pool == 1)
    cort_pool_idx = np.where(is_cortical)[0]
    sub_pool_idx = np.where(is_subcortical)[0]
    if cort_pool_idx.size == 0 and sub_pool_idx.size == 0:
        raise ValueError("Multi-point mode requires at least one population to be non-empty")
    eligible_pool_size = int(cort_pool_idx.size + sub_pool_idx.size)
    if eligible_pool_size < n_max:
        raise ValueError(
            "Multi-point mode requires at least "
            f"multi_point.n_sources_max={n_max} distinct eligible positions; "
            f"got {eligible_pool_size}"
        )

    f_sub = cfg.sources.subcortical_fraction
    max_attempts = cfg.multi_point.max_separation_attempts
    separation_violated = np.zeros(n_samples, dtype=bool)
    coupling_values = cfg.multi_point.time_coupling_values
    if coupling_values is None:
        coupling_per_sample = np.full(
            n_samples, cfg.multi_point.time_coupling, dtype=np.float64
        )
    else:
        coupling_per_sample = rng.choice(
            np.asarray(coupling_values, dtype=np.float64),
            size=n_samples,
            replace=True,
        )

    for i in range(n_samples):
        K = int(K_per[i])
        chosen_idx: List[int] = []
        chosen_xyz: List[np.ndarray] = []
        for source_index in range(K):
            for _ in range(max_attempts):
                candidate_idx = _draw_population_index(
                    rng, f_sub, cort_pool_idx, sub_pool_idx,
                )
                cand_xyz = position_pool_xyz[candidate_idx]
                # A pool position is never reused inside one sample, even when
                # min_separation_mm is zero: two dipoles at an identical
                # location are one dipole, and the separation test alone would
                # admit them because 0.0 >= 0.0.
                if candidate_idx not in chosen_idx and (
                    not chosen_xyz or all(
                        np.linalg.norm(cand_xyz - existing) >= min_sep_m
                        for existing in chosen_xyz
                    )
                ):
                    chosen_idx.append(candidate_idx)
                    chosen_xyz.append(cand_xyz)
                    break
            else:
                if cfg.multi_point.separation_failure_policy == 'raise':
                    raise RuntimeError(
                        "Could not satisfy multi_point.min_separation_mm="
                        f"{cfg.multi_point.min_separation_mm} after {max_attempts} "
                        f"attempts for sample {i}, source {source_index + 1}/{K}. Increase the "
                        "position pool, lower the source count/separation, or explicitly "
                        "set separation_failure_policy='accept_and_flag'."
                    )
                unused_cort = cort_pool_idx[
                    ~np.isin(cort_pool_idx, chosen_idx, assume_unique=True)
                ]
                unused_sub = sub_pool_idx[
                    ~np.isin(sub_pool_idx, chosen_idx, assume_unique=True)
                ]
                if unused_cort.size == 0 and unused_sub.size == 0:
                    raise RuntimeError(
                        "Could not choose a distinct position for multi-point "
                        f"sample {i}, source {source_index + 1}/{K}"
                    )
                candidate_idx = _draw_population_index(
                    rng, f_sub, unused_cort, unused_sub,
                )
                cand_xyz = position_pool_xyz[candidate_idx]
                chosen_idx.append(candidate_idx)
                chosen_xyz.append(cand_xyz)
                separation_violated[i] = True

        idx_arr = np.array(chosen_idx)
        positions_K = position_pool_xyz[idx_arr]                 # (K, 3)
        L_K = leadfield_pool[idx_arr]                            # (K, N, 3)
        populations_K = population_pool[idx_arr]                 # (K,)

        orient_world_K, _ = sample_orientations(
            K,
            cfg.dipole.orientation_mode,
            positions_K,
            rng,
            codebook_tangential_directions=cfg.dipole.codebook_tangential_directions,
        )

        mags_K = np.empty(K)
        for k in range(K):
            if populations_K[k] == 0:
                mags_K[k] = sample_magnitudes(
                    1, cfg.dipole.cortical_magnitude_range_nAm,
                    cfg.dipole.magnitude_distribution, rng,
                )[0]
            else:
                mags_K[k] = sample_magnitudes(
                    1, cfg.dipole.subcortical_magnitude_range_nAm,
                    cfg.dipole.magnitude_distribution, rng,
                )[0]

        primary_course = time_courses[i]                          # (T,)
        courses_K = np.empty((K, T), dtype=np.float64)
        courses_K[0] = primary_course

        # Primary-source temporal metadata is passed in from the split runner
        # because the primary source course is generated before source-mode
        # dispatch. Direct calls without metadata leave primary metadata padded.
        if primary_signal_metadata is not None:
            source_base_band_idx[i, 0] = int(primary_signal_metadata['band'][i])
            source_base_frequencies_hz[i, 0] = primary_signal_metadata['frequencies_hz'][i]
            source_base_phases_rad[i, 0] = primary_signal_metadata['phases_rad'][i]
            source_base_component_amplitudes[i, 0] = \
                primary_signal_metadata['component_amplitudes'][i]
            source_base_n_components[i, 0] = int(primary_signal_metadata['n_components'][i])
        source_primary_blend_weight[i, 0] = 1.0
        source_independent_blend_weight[i, 0] = 0.0
        source_blend_std[i, 0] = 1.0

        if K > 1:
            coupling = float(coupling_per_sample[i])
            extra_sig = generate_signal_batch(
                n_samples=K - 1,
                window_samples=T,
                sampling_freq_hz=cfg.signal.sampling_freq_hz,
                bands_hz=cfg.signal.bands_hz,
                band_probabilities=band_probabilities,
                n_components_min=cfg.signal.n_components_min,
                n_components_max=cfg.signal.n_components_max,
                component_amplitude_jitter=cfg.signal.component_amplitude_jitter,
                component_dirichlet_concentration=(
                    cfg.signal.component_dirichlet_concentration
                ),
                rng=rng,
                zero_mean=bool(zero_mean_signal),
                band_order=cfg.signal.band_order,
            )
            extra_courses = extra_sig['signal']
            blended = coupling * primary_course[None, :] + (1 - coupling) * extra_courses
            sd = blended.std(axis=1, keepdims=True)
            sd = np.where(sd > 0, sd, 1.0)
            blended = blended / sd
            courses_K[1:] = blended

            source_base_band_idx[i, 1:K] = extra_sig['band'].astype(np.int32)
            source_base_frequencies_hz[i, 1:K] = extra_sig['frequencies_hz']
            source_base_phases_rad[i, 1:K] = extra_sig['phases_rad']
            source_base_component_amplitudes[i, 1:K] = \
                extra_sig['component_amplitudes']
            source_base_n_components[i, 1:K] = extra_sig['n_components'].astype(np.int32)
            source_primary_blend_weight[i, 1:K] = coupling
            source_independent_blend_weight[i, 1:K] = 1.0 - coupling
            source_blend_std[i, 1:K] = sd[:, 0]

        eeg_i = np.zeros((N_channels, T), dtype=np.float64)
        for k in range(K):
            eeg_i += _eeg_from_dipole(
                positions_K[k], L_K[k], orient_world_K[k],
                mags_K[k] * 1e-9, courses_K[k]
            )
        eeg[i] = eeg_i

        positions_padded[i, :K] = positions_K
        orientations_padded[i, :K] = orient_world_K
        magnitudes_padded[i, :K] = mags_K
        population_padded[i, :K] = populations_K
        pos_idx_padded[i, :K] = idx_arr
        source_time_courses[i, :K] = courses_K

    return SampleBatch(
        eeg_signal=eeg,
        positions_xyz=positions_padded,
        orientations_world=orientations_padded,
        magnitudes_nAm=magnitudes_padded,
        population=population_padded,
        extras={
            'n_dipoles_per_sample': K_per,
            'separation_violated': separation_violated,
            'time_coupling': coupling_per_sample,
            'position_idx_in_pool': pos_idx_padded,
            'source_time_courses': source_time_courses,
            'source_base_band_idx': source_base_band_idx,
            'source_base_frequencies_hz': source_base_frequencies_hz,
            'source_base_phases_rad': source_base_phases_rad,
            'source_base_component_amplitudes': source_base_component_amplitudes,
            'source_base_n_components': source_base_n_components,
            'source_primary_blend_weight': source_primary_blend_weight,
            'source_independent_blend_weight': source_independent_blend_weight,
            'source_blend_std': source_blend_std,
        },
    )


# =============================================================================
# Mode: 'patch' -- spatially extended cortical patch with finite support
# =============================================================================

def generate_patch_samples(
    cfg: Config,
    n_samples: int,
    position_pool_xyz: np.ndarray,
    leadfield_pool: np.ndarray,
    population_pool: np.ndarray,
    time_courses: np.ndarray,
    rng: np.random.Generator,
) -> SampleBatch:
    """Generate patch samples (extended cortical activations).

    Each sample picks a CORTICAL center position and a patch diameter, then
    activates cortical positions within that diameter with the configured
    finite-support profile. Dipoles in the patch use coherent local cortex-normal orientations:
    each active dipole is aligned with its own local radial normal on the
    spherical cortical shell, while all dipoles share the same sign and
    common time course (the patch is one coherent activation).

    The maximum number of active dipoles per patch is capped at
    ``pi * (D/2)^2 * cfg.patch.dipoles_per_mm2``; if more candidates lie
    within the radius than this cap, they are randomly subsampled.

    Patch extent is defined by surface-geodesic distance: angular separation
    on the spherical cortex multiplied by the center radius.  Candidate source
    depths remain those of the configured cortical shell.

    Ground truth: (center_position, diameter_mm, orientation, total_magnitude).
    ``magnitudes_nAm`` for a patch is the TOTAL dipole moment of the patch --
    the sum of the falloff-weighted per-dipole moments (Sigma_k q_k), in nA*m --
    NOT a fitted equivalent point dipole. Because the patch's dipoles are spread
    over the cortical shell, its scalp topography is not identical to a single
    dipole of that magnitude at the center; the label is the total injected
    current moment, which differs in meaning from the point/multi-point single-
    dipole ``magnitudes_nAm``. ``extras['position_idx_in_pool']`` stores the
    pool index of the chosen patch center for downstream provenance. Padding-free
    CSR-style ``patch_active_position_idx_in_pool`` and
    ``patch_active_moments_nAm`` arrays store the exact support and signed
    radial moment allocation; the pipeline maps the indices to the exported
    global source pool.
    """
    N_channels = leadfield_pool.shape[1]
    T = time_courses.shape[1]

    cortical_mask = (population_pool == 0)
    cort_pool_idx = np.where(cortical_mask)[0]
    if cort_pool_idx.size == 0:
        raise ValueError("Patch mode requires cortical positions in the pool")

    cort_xyz = position_pool_xyz[cort_pool_idx]   # (Pc, 3)
    # Patch support is an angular neighbourhood on the notional cortical
    # sphere, so every candidate enters the geometry only through its
    # direction. The directions depend on the pool alone, not on the sample.
    cort_unit = cort_xyz / np.linalg.norm(cort_xyz, axis=1, keepdims=True)

    diameter_low_mm, diameter_high_mm = cfg.patch.diameter_range_mm
    falloff = cfg.patch.falloff_profile
    dipoles_per_mm2 = cfg.patch.dipoles_per_mm2

    centers_xyz = np.zeros((n_samples, 3))
    diameters_mm = np.zeros(n_samples)
    orientations_world = np.zeros((n_samples, 3))
    magnitudes_nAm = np.zeros(n_samples)
    populations = np.zeros(n_samples, dtype=np.int8)  # always 0 for patches
    n_active_per = np.zeros(n_samples, dtype=np.int32)
    center_pool_idx = np.zeros(n_samples, dtype=np.int64)
    active_index_rows: List[np.ndarray] = []
    active_moment_rows: List[np.ndarray] = []

    eeg = np.zeros((n_samples, N_channels, T), dtype=np.float64)

    for i in range(n_samples):
        # Choose patch center
        center_local = int(rng.integers(0, cort_pool_idx.size))
        center_global = int(cort_pool_idx[center_local])
        center_xyz = position_pool_xyz[center_global]
        center_pool_idx[i] = center_global

        # Choose diameter and compute its geodesic radius.
        D_mm = rng.uniform(diameter_low_mm, diameter_high_mm)
        radius_mm = D_mm / 2.0

        # Great-circle separation on the notional spherical cortical surface.
        # Multiplying the angular distance by the center radius gives the
        # surface-geodesic distance used for both support and falloff.
        #
        # Support is selected on the cosine rather than the angle. The
        # criterion d_mm <= radius_mm is equivalent to cos(gamma) >= cos(
        # radius_mm / center_radius_mm) because the arccosine is monotonically
        # decreasing on [-1, 1], so the two select the same neighbourhood while
        # the cosine form evaluates one dot product instead of a transcendental
        # function over the whole pool. The angle itself is then needed only for
        # the falloff weights of the points actually recruited.
        center_radius_m = float(np.linalg.norm(center_xyz))
        center_unit = center_xyz / max(center_radius_m, 1e-15)
        cos_gamma = cort_unit @ center_unit
        # A patch wider than the sphere covers it entirely; cap the half-angle
        # at pi so the cosine threshold stays monotone in the diameter.
        angular_radius = float(radius_mm) / (center_radius_m * 1000.0)
        if angular_radius > np.pi:
            angular_radius = float(np.pi)
        candidates_local = np.flatnonzero(cos_gamma >= np.cos(angular_radius))
        if candidates_local.size == 0:
            candidates_local = np.array([center_local])

        # Cap density via random subsampling if the pool is denser than the cap
        area_mm2 = np.pi * (D_mm / 2.0) ** 2
        max_dipoles = max(1, int(np.ceil(area_mm2 * dipoles_per_mm2)))
        if candidates_local.size > max_dipoles:
            # The labeled center is part of the activation by definition.
            # Preserve it while randomly subsampling the remaining candidates.
            others = candidates_local[candidates_local != center_local]
            n_other = max_dipoles - 1
            if n_other > 0:
                selected = rng.choice(others, size=n_other, replace=False)
                candidates_local = np.concatenate((
                    np.array([center_local], dtype=candidates_local.dtype),
                    np.asarray(selected, dtype=candidates_local.dtype),
                ))
            else:
                candidates_local = np.array(
                    [center_local], dtype=candidates_local.dtype
                )

        # Serialize the support in ascending pool order so a patch's CSR row
        # has the same layout however its members were recruited.
        candidates_local = np.sort(candidates_local)
        in_patch_global = cort_pool_idx[candidates_local]
        in_patch_d_mm = (
            np.arccos(np.clip(cos_gamma[candidates_local], -1.0, 1.0))
            * center_radius_m * 1000.0
        )

        # Falloff weights. The Gaussian option is explicitly truncated at the
        # patch radius; with sigma=radius/2 its boundary value before truncation
        # is exp(-2), so it is not continuous across the support boundary.
        if falloff == 'gaussian':
            sigma_mm = max(radius_mm / 2.0, 1e-9)
            weights = np.exp(-0.5 * (in_patch_d_mm / sigma_mm) ** 2)
        elif falloff == 'uniform':
            weights = np.ones_like(in_patch_d_mm)
        elif falloff == 'cosine':
            weights = np.cos(
                (np.pi / 2.0) * (in_patch_d_mm / max(radius_mm, 1e-12))
            )
            weights = np.clip(weights, 0, 1)
        else:
            raise ValueError(f"Unknown patch.falloff_profile: {falloff!r}")

        # Coherent cortex-normal orientation field. The recorded patch
        # orientation is the signed center normal, while each active dipole
        # uses its own signed local radial normal when projected.
        center_norm = np.linalg.norm(center_xyz)
        e_r_center = center_xyz / max(center_norm, 1e-12)
        sign = float(rng.choice([-1.0, 1.0]))
        orientation_world_i = sign * e_r_center

        # Sample total patch magnitude from the cortical range. The recorded
        # ground-truth magnitude is this TOTAL moment (sum of within-patch
        # per-dipole moments), not an equivalent point-dipole fit.
        total_mag_nAm = sample_magnitudes(
            1, cfg.dipole.cortical_magnitude_range_nAm,
            cfg.dipole.magnitude_distribution, rng,
        )[0]
        weights_sum = weights.sum()
        if weights_sum <= 0:
            weights_sum = 1.0
        per_dipole_nAm = total_mag_nAm * weights / weights_sum
        per_dipole_Am = per_dipole_nAm * 1e-9

        # Superpose the scalp signatures of the patch's active dipoles.
        #
        # Every active dipole points along its own signed radial normal, so in
        # the local spherical frame at its own position its moment direction is
        # exactly (sign, 0, 0): the radial basis vector is the orientation
        # itself, and the two tangential basis vectors are orthogonal to it.
        # The projection therefore reduces to the radial column of each
        # position's basis lead field, and the whole superposition is a single
        # matrix-vector product.
        scalp_sum = sign * (per_dipole_Am @ leadfield_pool[in_patch_global][:, :, 0])

        eeg[i] = scalp_sum[:, None] * time_courses[i][None, :]
        centers_xyz[i] = center_xyz
        diameters_mm[i] = D_mm
        orientations_world[i] = orientation_world_i
        magnitudes_nAm[i] = total_mag_nAm
        n_active_per[i] = candidates_local.size
        active_index_rows.append(in_patch_global.astype(np.int64, copy=False))
        active_moment_rows.append((sign * per_dipole_nAm).astype(np.float64))

    row_splits = np.zeros(n_samples + 1, dtype=np.int64)
    if n_samples:
        row_splits[1:] = np.cumsum([row.size for row in active_index_rows])
    active_index_values = (
        np.concatenate(active_index_rows)
        if active_index_rows else np.empty(0, dtype=np.int64)
    )
    active_moment_values = (
        np.concatenate(active_moment_rows)
        if active_moment_rows else np.empty(0, dtype=np.float64)
    )

    return SampleBatch(
        eeg_signal=eeg,
        positions_xyz=centers_xyz,
        orientations_world=orientations_world,
        magnitudes_nAm=magnitudes_nAm,
        population=populations,
        extras={
            'patch_diameter_mm': diameters_mm,
            'n_active_per_patch': n_active_per,
            'position_idx_in_pool': center_pool_idx,
            'patch_active_position_idx_in_pool': RaggedArray(
                active_index_values, row_splits,
            ),
            'patch_active_moments_nAm': RaggedArray(
                active_moment_values, row_splits,
            ),
        },
    )


# =============================================================================
# Background lead-field pool (for the source-projected noise proxy)
# =============================================================================

def compute_background_leadfield_pool(
    n_pool: int,
    background_radius_range_mm: Sequence[float],
    elec_theta: np.ndarray,
    elec_phi: np.ndarray,
    M_tilde: np.ndarray,
    brain_radius_m: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Compute a pool of background-dipole scalp projections for noise.

    Each row of the returned matrix is the scalp signature of one
    background dipole at a random brain position with random orientation.
    Used by the source-projected noise proxy to generate spatially correlated
    background arrays by mixing time courses across these projections.

    Parameters
    ----------
    n_pool : int
        Number of background dipoles in the pool.
    background_radius_range_mm : (low, high)
    elec_theta, elec_phi : ndarray
    M_tilde : ndarray
    brain_radius_m : float
    rng : numpy Generator

    Returns
    -------
    pool : ndarray, shape (n_pool, N_channels)
        Each row standard-normalized to unit norm; the time-course
        amplitude controls actual scale at noise-generation time.
    """
    radii_m = rng.uniform(
        background_radius_range_mm[0] * 1e-3,
        background_radius_range_mm[1] * 1e-3,
        size=n_pool,
    )
    directions = sample_unit_vectors_s2(n_pool, rng)
    positions = directions * radii_m[:, None]

    orientations_world = sample_unit_vectors_s2(n_pool, rng)

    N_channels = elec_theta.shape[0]
    pool = np.zeros((n_pool, N_channels), dtype=np.float64)
    for i in range(n_pool):
        rtp = cartesian_to_spherical(positions[i])
        L = forward_potential_basis(
            rtp[0], rtp[1], rtp[2], elec_theta, elec_phi, M_tilde,
            brain_radius_m=brain_radius_m,
        )
        local = world_orientation_to_spherical_frame(orientations_world[i], positions[i])
        pool[i] = potentials_from_basis(L, local[0], local[1], local[2])

    norms = np.linalg.norm(pool, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    return pool / norms
