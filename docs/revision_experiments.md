# Experiment protocols

These are the protocols of the experiments reported in the paper. Each one is run by a script in `scripts/`, writes to a new folder, and records the commit, the script and package hashes, the interpreter, the resolved packages and the thread settings. Recorded runs need a clean committed checkout. Every `run_*.py` driver has a `--smoke` option for a small end-to-end check that does not reproduce the reported estimates; the generation benchmark has none.

## Common settings

The supervised experiments use point sources unless stated, the 69-channel montage, 60 Legendre terms, 200 ms windows at 500 Hz, target SNR drawn from 10 to 25 dB on the average-referenced montage, and per-window RMS normalization. Datasets have 20,000 training, 2,000 validation and 3,000 test windows. Dataset seeds are 862, 863 and 864 and model seeds are 862 and 863.

The regressor is a multilayer perceptron with hidden layers of 256 and 128 units. Its input is the average-referenced scalp map at the sample of peak global field power, and its target is the source position in millimetres. Inputs and targets are standardized with training statistics. Training runs for at most 300 epochs and keeps the epoch with the lowest validation position error, stopping after 15 epochs without improvement.

Summaries report the median, mean and 90th percentile of the localization error. Intervals come from 2,000 bootstrap draws that resample source positions, keeping all windows at a position and both model-seed predictions together; differences of medians use paired draws. The headline is the mean of the three dataset-seed medians with their range. Strata with fewer than 30 distinct positions are descriptive only. Intervals are conditional on the generated datasets, the region assignments and the fitted models.

## Split comparison (`run_split_study.py`)

Point sources with 8,000 cortical and 1,500 subcortical pool positions, generated under the exact-position policy and under the region policy with 24 sectors, 15 mm clearance and rotation seeds 24000, 24001 and 24002. The two policies share the position pool and the background-noise pool.

The region assignment must pass adequacy checks that are evaluated from the geometry alone, before any data are generated. Test and validation shares before buffering must lie within three percentage points of their targets. After buffering, training and validation must each retain at least 60% of each population and 50% of every radius band, and test needs at least 30 eligible positions in every band. The bands are 30 to 45, 45 to 55 and 55 to 65 mm for subcortical positions and 70.5 to 75 and 75 to 79.5 mm for cortical positions.

Comparators, all on the same test windows: a lead-field dictionary that maximizes the fraction of map power explained by the three orientation columns at the training positions; nearest-neighbour regression on absolute map correlation with one and ten neighbours; and the mean training position. A matched-sphere dipole fit is run on a fixed subset of 500 test windows per dataset, chosen by a separate random stream before fitting, and the other methods are also reported on that subset.

The two training regimes are compared on the same test windows: the exact-position test windows whose positions lie in the regional test sectors. Strata are source population, radius band, SNR band, distance to the nearest training position, and whether the source lies below the montage. Both common-window prediction files include the mean training position for their own policy. The archived regional file predates this diagnostic. Its inclusion in new runs leaves the reported common-window MLP comparison unchanged.

## Training-location count (`run_training_count_control.py`)

Region holdout leaves fewer distinct training locations than exact-position holdout. For each dataset seed, the exact-position training set is reduced to the number of distinct locations that the regional training set used, separately in each population and radius band, with the random stream `[dataset seed, 101]`; all windows at a chosen location are kept. The first model seed is refitted on the reduced set and scored on the same test windows as the original fit, and on the test windows that the two training regimes of the split comparison have in common.

## Patch split comparison (`run_patch_split_study.py`)

Cortical patches of 10 to 60 mm diameter with the default Gaussian profile. The exact-position dataset uses a pool of 35,001 cortical positions, which gives each split a support mesh of 11,667 positions. The region dataset uses a pool of 11,667 positions, so its support density is the same, with eight sectors, 15 mm clearance, rotation seeds 24000 to 24002 and support containment.

The region assignment must let the largest patch fit at 50 or more centres in every split and must retain at least half of the training region after buffering. The script records, from each dataset's validation report, the nearest-centre distances, the best footprint overlap of each test patch with any training patch, the smallest footprint gap, the minimum distance between supports of different splits, and the containment verification.

The regressor predicts the patch centre. Comparators are the centre of the best-correlated training patch, the mean centre of the ten best-correlated training patches, and the mean training centre. Strata are diameter band and SNR band.

## Coordinate reuse under shared positions (`run_shared_position_demo.py`)

The seed-862 point dataset of the split comparison is generated under the shared policy with 1% and with 100% of the position pool shared. The second case is a full independent redraw of test locations from the training pool. The script reports, from the dataset's validation report, how many test windows fall on a location that training also used.

## Transfer to a boundary-element head (`run_head_transfer.py`)

The networks trained under the exact-position policy are scored, without retraining, on windows generated with the fsaverage three-layer boundary-element model in MNE-Python 1.12.1. The model has brain, skull and scalp compartments with the sphere's conductivities and no separate cerebrospinal fluid.

Each test source keeps its position, orientation, moment and time course and is placed rigidly in the template: the sphere centre is the least-squares centre of the 336 template electrodes, the vertical axis passes through Cz and the anterior axis points to Fpz. The template's 69 same-named electrodes are projected to the boundary-element scalp. Only sources at least 5 mm inside the inner skull are scored. The saved sphere noise of each window is rescaled to keep its referenced SNR, and the window is normalized to unit RMS. Sphere and boundary-element results are compared on the same eligible windows.

Five checks precede the comparison: a rigidly transformed sphere must reproduce the original fields within a relative 1e-6; registered and native electrode directions must agree within 1.5 degrees; replaying the stored lead fields must reproduce the stored signal, noise scale and normalized window within 1e-10; an independent sphere implementation must agree with a median relative residual of at most 0.005 and no window above 0.05; and the networks' median errors on the two sphere implementations must agree within 0.5 mm.

Dipole fits with the matched forward model of each condition are run on 500 eligible seed-862 windows.

## Head-model adaptation (`run_head_adaptation.py`)

All arms are scored on the same eligible boundary-element test windows as the transfer experiment.

Sphere variants. New networks are trained on spheres matched to the template while the source pools and every per-window draw stay the same: the default radii with the cerebrospinal-fluid layer merged into the brain; all four radii scaled so that the scalp radius equals the median distance of the projected template electrodes from the registered centre, with and without the cerebrospinal-fluid layer; and radii set to the median distance of the inner-skull, outer-skull and scalp surfaces in the upper half of the head, with the cerebrospinal-fluid layer merged.

Dipole fits. The 500 boundary-element windows of the transfer experiment are fitted with each sphere and its own montage, which is the classical counterpart of a network trained on that sphere.

Training on the realistic head. Boundary-element windows are built for the eligible training and validation windows in the same way as for the test windows. One arm trains the network on all of them. For 100, 300, 1,000 and 3,000 training windows and for all of them, drawn with the random stream `[dataset seed, model seed, size, 605]` together with a validation subset of one quarter of that size (at least 25 windows), one arm continues training the sphere-trained network with the same early-stopping rule and one arm trains a new network on the same windows. Fine-tuning starts with the weights from the best sphere-validation epoch, the sphere-trained input and target scalers, and Adam’s first and second moments and update count from the last pretraining epoch.

## Convergence of the Berg–Scherg parameter fit in MNE-Python (`run_sphere_fit_convergence.py`)

The two comparisons with MNE-Python (`validate_inverse_mne.py` and `validate_patch_forward_mne.py`) depend on the Berg–Scherg parameters that MNE-Python fits for its sphere model. MNE-Python 1.12.1 calls SciPy's COBYLA optimizer for this fit without an evaluation limit of its own, so SciPy's default of 1,000 function evaluations applies. The script repeats the fit through the same optimizer call, once with that limit and once with a limit of 100,000, and records the arguments of the call, the number of evaluations, the reason for stopping, the value of the objective and the first four evaluations. It stops if the first of the two fits does not reproduce the parameters of the installed MNE-Python exactly. It records the same for the sphere that MNE-Python builds when no head model is given.

With each fit the script runs the patch comparison (60 patches and 60 point sources), the dipole recovery at 35 dB (300 dipoles) and the SNR sweep (150 dipoles per level) on the datasets of the two validation scripts, generated with their seed and default settings. Thread settings are left unchanged, as in the two validation scripts.

Four further parts vary the arithmetic on one machine. First, the patch comparison and the dipole recovery at 35 dB are repeated with the objective of the parameter fit multiplied by 1 + j·eps for j = 1 to 8, where eps is the machine precision, with either evaluation limit. Second, the dipole recovery is repeated with the objective of the dipole fits multiplied by the same factors, with the sphere model of either limit; the script stops if its instrumented dipole fit with a factor of one does not reproduce the fitted positions of the installed MNE-Python. For every trial of these two parts the fitted position and the errors of every dipole are recorded. Third, the dipole recovery is repeated with the dipole fits ended at a trust-region radius of 1e-7 m in place of the 5e-5 m of MNE-Python (argument `tol` of `mne.fit_dipole`): with unchanged objectives and with the factors for j = 1 to 4 on either objective. Fourth, both fits, their comparisons, the fit of the sphere that MNE-Python builds by default and the dipole recovery with the small radius are run in child processes with other kernels of OpenBLAS, selected with the environment variable `OPENBLAS_CORETYPE` (by default Haswell, Sandybridge and Nehalem; a kernel that the processor does not support, or that is the one in use, is recorded as not available), and the parameter fit of the main process is repeated with its objective at the second trial point replaced by the value of each other kernel.

The optimizer starts at μ = 0 with a first step of 0.5. At each of its first four trial points at least two of the three Berg–Scherg terms coincide, so the least-squares problem inside the objective is rank-deficient and the value that MNE-Python computes depends on singular vectors that the problem does not determine. At three of these points the degenerate columns of the model matrix are exactly zero; at the second, μ = (0.5, 0, 0), two columns are equal and nonzero, and the value differs between kernels. These parts show that the outcome depends on such details, and what remains when the parameter fit has converged and the dipole fits end at the smaller radius. They are examples from one machine and give no bounds for other platforms. The SNR sweep is run with the two unchanged fits only.

With `--geometry`, the script takes the spheres of a run of `run_head_adaptation.py` from its `geometry.json` and records, for each sphere and each evaluation limit, how the parameter fit ends, the relative field difference of 60 point sources, and the fitted position and position error of 300 point sources at 10 to 25 dB that were generated with that sphere.

## The parameter fit of the installed MNE-Python (`check_installed_sphere_fit.py`)

`run_sphere_fit_convergence.py` replaces the optimizer call inside MNE-Python 1.12.1 and does not run with a version that looks the optimizer up elsewhere. This script changes nothing in the fit. It counts the function evaluations of SciPy's COBYLA while the installed MNE-Python builds its sphere model, once with SciPy's default limit of 1,000 evaluations, which MNE-Python uses, and once with a limit of 100,000. It stops if the fit that it counted at the default limit is not the fit of the installed MNE-Python.

It records the arguments of the optimizer call, the number of evaluations, whether the optimizer stopped before the limit, the objective at the end and the fitted parameters: for the default head model of the package, for the sphere that MNE-Python builds when no head model is given, and with `--geometry` for the spheres of a run of `run_head_adaptation.py`. The same is recorded in child processes with other kernels of OpenBLAS, selected with `OPENBLAS_CORETYPE` as in `run_sphere_fit_convergence.py`. No dipole is fitted and no dataset is generated.

The script was run with MNE-Python 1.12.1 and with MNE-Python 1.13.2, in both cases with SciPy 1.17.1. Version 1.13 ends the fit at a final trust-region radius of 1e-4 in place of 1e-5, computes its weights from the relative radii, and keeps the result of a fit. With version 1.13.2 the two validation scripts `validate_inverse_mne.py` and `validate_patch_forward_mne.py` were run as well, with their default settings.

## Dipole-fit comparators with a converged parameter fit (`run_dipole_comparators_converged.py`)

The dipole fits with a sphere model that the paper reports from the split comparison, the transfer experiment and the adaptation experiment (Tables 3 and 5) use the sphere model of MNE-Python and therefore a parameter fit that ended at the evaluation limit in the recorded runs (MNE-Python 1.12.1 with SciPy 1.17.1). This script repeats them on the windows of the recorded experiments, with the same calls, once with the limit of 1,000 evaluations and once with a limit of 100,000. It repeats three groups of fits: the matched-sphere fit of the split comparison on the dipole-fit subset of each of the six datasets (500 windows, random stream `[dataset seed, 601]`); the matched-sphere fit of the transfer experiment on its 500 sphere windows of the seed-862 dataset (stream `[dataset seed, 602]`); and the fits of each of the five spheres of the adaptation experiment to the same 500 boundary-element windows. It needs the datasets of a recorded run of `run_split_study.py` and the fsaverage template.

The fit with the boundary-element model uses no sphere model and is not repeated. The records of the experiments hold two further dipole fits with a sphere model that the paper does not report and the script does not repeat: the fit with the automatically fitted sphere of the transfer experiment, and the matched-sphere fit on the shared test windows of the split comparison.

With `--recorded`, the script compares every fitted position at the limit of 1,000 evaluations with the per-window files of the three recorded experiments and stops if one differs. It records, for every fit and both limits, how the parameter fit ended, the fitted position and position error of every window, the median, mean and 90th percentile of the error, and the median, 90th percentile and maximum of the distance between the two fitted positions of a window. The dipole fits run with one thread per library, as in the experiments.

## Generation benchmark (`benchmark_generation.py`)

Each case runs the generator in a child process on an otherwise idle machine. Wall time covers the whole process including validation and publication; memory is the largest sampled sum of resident memory over the process and its workers; size is the total of the published files. Fixed-size cases generate 10,000 training, 1,000 validation and 1,000 test windows with the default pools and are repeated three times; the full default configuration (360,000 windows) is run once with one worker and once with six. Stage timers separate the lead-field setup, the background-noise pool, window synthesis and writing.
