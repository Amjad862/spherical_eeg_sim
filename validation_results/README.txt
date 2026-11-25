Archived validation results of SphericalEEGSim 1.1.0

Every file in this folder is listed with its SHA-256 in SHA256SUMS. Run

    python scripts/make_validation_manifest.py --check

from the source root to verify the files, the bound source inventory and the
relationships between the result bundles. No validation command writes into
this folder unless it is given --write-canonical: by default each one writes
to a new folder under validation_runs/.

Numerical validation of the generator

  transfer_coefficients.json      scripts/validate_transfer_coefficients.py
      Transfer coefficients against an independent boundary-value solution.
  legendre_convergence.json       scripts/legendre_convergence.py
      Truncation and round-off of the Legendre series by expansion order.
  ecd_69ch_35db_summary.json, ecd_69ch_35db_per_sample.csv, ecd_69ch_35db.png
                                  scripts/validate_inverse_mne.py
      Dipole recovery with MNE-Python and the SNR sweep.
  patch_forward_mne_summary.json  scripts/validate_patch_forward_mne.py
      Patch forward fields against MNE-Python.
  split_spacing.json              scripts/measure_split_spacing.py
      Nearest-neighbour distances between the splits of the default pool.
  baseline_benchmark_summary.json, baseline_benchmark.png
                                  scripts/train_baseline.py
      The supervised demonstration of release 1.0.0, kept as a historical
      record; the paper does not use it.

Each summary records the software environment of its run. The summaries of
the numerical validation were recorded with version 1.0.0 on Linux. Version
1.1.0 generates the same arrays under the default split policy:
tests/test_release_arrays.py compares small point, multi-point and patch
datasets with arrays recorded from 1.0.0 (by digest in the recorded
reference environment, to a relative 1e-10 elsewhere), and for the ten
datasets of the two comparisons with MNE-Python every array is identical
under both versions, as are the results of the four validation scripts when
both versions run on one machine (folder validation-rerun-v11 of the paper's
evidence deposit).

The statistics of the two comparisons with MNE-Python are not identical
between platforms. On Windows, with the library versions of the archived
run, the same 300 dipoles give a median position error of 0.27 mm with a
90th percentile of 0.84 mm and a maximum of 2.48 mm, where the archived
Linux run has 0.27, 0.88 and 2.41 mm, and the patch comparison gives a
median relative field difference of 2.4e-3 against 2.7e-3 for point sources,
where the archived run has 2.8e-3 against 3.0e-3. Differences of this size
arise in two fits of MNE-Python, which scripts/run_sphere_fit_convergence.py
examines (folder sphere-fit-convergence-v11 of the evidence deposit).

MNE-Python fits the Berg-Scherg parameters of its sphere model with SciPy's
COBYLA optimizer and sets no evaluation limit. For the head model of this
package and for the four other spheres of the paper, MNE-Python 1.12.1 with
SciPy 1.17.1 reached SciPy's default of 1,000 function evaluations before
the fit had converged, in every run on the Windows machine of these
records. Where the unfinished fit ends depends on rounding and
on the kernel of the linear-algebra library. Over eleven runs on one
machine, eight of
them with the objective changed in its last bits and two with other kernels
of OpenBLAS, the median relative field difference of the patch comparison
ranged from 1.8e-3 to 4.4e-3. With the limit raised the fit converged in
every run, after 1,672 to 3,199 evaluations, and the field difference
was 5.0e-4 for patches and 5.3e-4 for point sources, with less than 1% of
variation. For the sphere that MNE-Python builds when no head model is
given, the outcome also depended on the kernel: the fit stopped by itself
after 87 evaluations with the kernel that OpenBLAS selects on the machine of
these runs (Haswell) and reached the limit with the two others (Sandybridge,
Nehalem).

The dipole fits of MNE-Python end when the trust-region radius of the
optimizer has shrunk to 0.05 mm. A last-bit change of their objective moves
the fitted position of a dipole by a median of 0.04 mm and by up to
1.12 mm, also with a converged parameter fit. Over 19 runs on one machine
the median position error was 0.26 to 0.29 mm with the default limit
(0.26 to 0.28 mm with the raised limit), the 90th percentile
0.80 to 0.88 mm (0.67 to 0.74 mm) and the maximum 2.39 to 3.03 mm
(2.31 to 3.20 mm). With the parameter fit converged and the dipole fits
ended at a radius of 1e-7 m (argument tol of mne.fit_dipole), the three
statistics agreed within 0.01 mm over eleven runs (0.25, 0.69 and 2.47 mm
without changes), although single fitted positions still differed from
those of the run without changes by up to 0.15 mm. These runs are examples;
their ranges are not bounds for other platforms.

The experiments of the paper compare their networks with dipole fits that
use the same sphere model. scripts/run_dipole_comparators_converged.py
repeated these fits on their own windows with a converged parameter fit
(folder dipole-comparators-converged-v11 of the evidence deposit): the fits
at the default limit were those of the recorded experiments, and with the
converged fit the twelve median errors changed by -0.08 to +0.03 mm.

The numbers above belong to MNE-Python 1.12.1. MNE-Python 1.13 ends the
parameter fit at a final trust-region radius of 1e-4 in place of 1e-5.
scripts/check_installed_sphere_fit.py counted the fit as the installed
library makes it, for the five spheres of the paper under three kernels of
OpenBLAS (folder sphere-fit-installed-v11 of the evidence deposit). With
MNE-Python 1.12.1 all 15 fits reached the limit of 1,000 evaluations and,
with the limit raised, stopped by themselves at objectives of 3.9e-6 to
6.0e-6. With MNE-Python 1.13.2 the fits of the default head model and of
the size-matched sphere reached the limit with the kernel of the machine
(objectives 7.8e-5 and 3.7e-5); the 13 other fits stopped by themselves
after 52 to 326 evaluations, at objectives of 5.8e-4 to 1.2e-3, about 100
to 210 times those at which the same fits stopped with MNE-Python 1.12.1
and the raised limit. A fit that stops by itself is therefore not
necessarily converged. With MNE-Python 1.13.2 on Windows the
300 dipoles gave a median position error of 0.27 mm with a 90th percentile
of 0.81 mm and a maximum of 2.30 mm, and the patch comparison a median
relative field difference of 1.8e-3 against 1.9e-3 for point sources.

The entries of legendre_convergence.json at orders 100 and 120, which are
limited by round-off, differ by as much as 2.4% between the two platforms.

Figures of the paper

  Figure 1  pipeline_schematic.png, .pdf          scripts/make_validation_figures.py
  Figure 2  topographies.png, .pdf                scripts/make_validation_figures.py
  Figure 3  region_split.png, .pdf                scripts/plot_region_split.py
  Figure 4  distributions.png, .pdf               scripts/make_validation_figures.py
  Figure 5  split_localization_error.png, .pdf    scripts/plot_split_study.py
  Figure 6  head_adaptation_finetuning.png, .pdf  scripts/plot_head_adaptation.py

publication_figures.json binds Figures 1, 2 and 4 to their generation
environment. region_split_source.json, split_localization_error_source.json
and head_adaptation_finetuning_source.json record what Figures 3, 5 and 6
plot and the hashes of the summaries they were drawn from.

Experiments of the paper

  revision_tables.json       scripts/make_revision_tables.py
      The tables and inline statistics of the experiments of the paper and
      of its supplementary material (generation benchmark, split comparisons,
      template-head transfer and adaptation, validation rerun, parameter
      fit of MNE-Python), rebuilt from the evidence archive. The driver
      recomputes the error statistics from the per-window files of that
      archive and compares each with the recorded summary; its description
      lists the numbers that it reads without recomputation.
  generation_benchmark.json  scripts/benchmark_generation.py
      The generation benchmark: time, memory and output size by case, with
      the host, the software environment and every repetition.

The per-window predictions, dataset configurations, summaries and provenance
records of the experiments (split comparison, training-location control,
patch split comparison, shared positions, boundary-element transfer,
head-model adaptation, generation benchmark, validation rerun, parameter fit
of MNE-Python) are deposited separately as the
evidence archive of the paper, which the data availability statement of the
paper cites. docs/revision_experiments.md gives the protocols and README.md
the commands. Generated EEG datasets, serialized models and the fsaverage
template are not part of either archive.
