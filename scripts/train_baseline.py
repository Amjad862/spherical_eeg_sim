"""Baseline ML benchmark: does the generated data train a source localizer?

This script is a machine-learning usability demonstration. Its purpose is NOT
to claim state-of-the-art localization,
but to show end-to-end that a dataset produced by spherical_eeg_sim trains a
supervised model that recovers source position with sensible, depth-dependent
accuracy -- i.e. that the outputs are usable ML training data.

Pipeline
--------
1. Generate a single-dipole (``point``-mode) dataset with the package.
2. Featurize each sample as the average-referenced scalp topography at the
   sample of peak global field power (GFP). For a single dipole the scalp
   pattern is, up to noise, a fixed spatial map scaled by the time course, so
   the peak-GFP topography is the natural, low-variance spatial estimator. This
   gives an ``n_channels``-dimensional input per sample.
3. Train a small multilayer perceptron (scikit-learn ``MLPRegressor``) to
   predict the source (x, y, z) position, in millimetres, from that topography.
4. Report median / mean / depth-stratified localization error on the held-out
   test split, alongside a predict-the-training-mean baseline so the learned
   error is interpretable.

The test positions are drawn from a held-out position pool (position-level
split), so the reported error reflects generalization to source locations not
seen in training, not memorization of a position dictionary.

Only numpy, scipy, h5py, and scikit-learn are required (no GPU / deep-learning
framework); a PyTorch equivalent is straightforward and is sketched in
``examples/pytorch_dataloader.py``.

Usage
-----
    python scripts/train_baseline.py
    python scripts/train_baseline.py --n-train 20000 --n-test 3000 --figure baseline.png
"""

import argparse
import copy
import hashlib
import json
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts._validation_outputs import (
    ValidationOutputs, add_output_options, check_work_directory, relative_reference,
)
from spherical_eeg_sim import Config, run_pipeline
from spherical_eeg_sim.provenance import environment_snapshot


def average_reference(topo):
    """Common-average-reference a (samples, channels) topography array."""
    return topo - topo.mean(axis=1, keepdims=True)


def peak_gfp_topography(eeg):
    """Average-referenced scalp topography at the peak-GFP time sample.

    Parameters
    ----------
    eeg : ndarray (B, C, T)

    Returns
    -------
    ndarray (B, C) -- one average-referenced topography per sample.
    """
    car = eeg - eeg.mean(axis=1, keepdims=True)         # (B, C, T)
    gfp = car.std(axis=1)                               # (B, T) spatial SD
    peak_t = np.argmax(gfp, axis=1)                     # (B,)
    idx = np.arange(eeg.shape[0])
    return car[idx, :, peak_t]                          # (B, C)


def load_split(path):
    import h5py
    with h5py.File(path, 'r') as f:
        eeg = f['eeg'][:]
        pos_mm = f['ground_truth/positions_xyz'][:] * 1000.0   # m -> mm
    return peak_gfp_topography(eeg), pos_mm


def depth_stratified(errors_mm, pos_mm):
    """Median error within deep / mid / superficial radial bands."""
    radius = np.linalg.norm(pos_mm, axis=1)
    bands = [('deep (30-50 mm)', radius < 50),
             ('mid (50-65 mm)', (radius >= 50) & (radius < 65)),
             ('superficial (>=65 mm)', radius >= 65)]
    out = []
    for label, mask in bands:
        if mask.sum() > 0:
            out.append((label, int(mask.sum()), float(np.median(errors_mm[mask]))))
    return out


def generate_dataset(args):
    cfg = Config()
    cfg.source_mode = 'point'
    cfg.dataset.n_train = args.n_train
    cfg.dataset.n_val = args.n_val
    cfg.dataset.n_test = args.n_test
    cfg.sources.n_cortical_positions = args.n_cortical
    cfg.sources.n_subcortical_positions = args.n_subcortical
    cfg.head.n_legendre = args.n_legendre
    cfg.electrode_file = args.montage
    cfg.master_seed = args.seed
    cfg.output.diagnostic_plots = False
    # Control SNR independently, then remove total amplitude so the baseline
    # cannot use absolute sensor scale as a shortcut for source magnitude.
    cfg.noise.noise_model = 'snr_target'
    cfg.noise.snr_db_range = [args.snr_low, args.snr_high]
    cfg.output.eeg_normalization = 'per_sample_rms'
    cfg.output_dir = args.output_dir
    cfg.output.overwrite = getattr(args, 'overwrite', False)
    run_pipeline(cfg)
    return cfg


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('--n-train', type=int, default=20000)
    ap.add_argument('--n-val', type=int, default=2000)
    ap.add_argument('--n-test', type=int, default=3000)
    ap.add_argument('--n-cortical', type=int, default=8000)
    ap.add_argument('--n-subcortical', type=int, default=1500)
    ap.add_argument('--n-legendre', type=int, default=60)
    ap.add_argument('--montage', type=str, default='standard_1010_69.csv')
    ap.add_argument('--snr-low', type=float, default=10.0)
    ap.add_argument('--snr-high', type=float, default=25.0)
    ap.add_argument('--seed', type=int, default=862)
    ap.add_argument('--hidden', type=int, nargs='+', default=[256, 128])
    ap.add_argument('--max-iter', type=int, default=300)
    ap.add_argument('--output-dir', type=str,
                    help='retain the generated dataset at this path (default: temporary)')
    ap.add_argument('--figure', type=str, default=None)
    ap.add_argument('--summary-json', type=str, default=None,
                    help='path for the measured metrics and optional figure hash')
    add_output_options(ap)
    args = ap.parse_args()

    explicit = {}
    if args.figure:
        explicit['figure'] = Path(args.figure).expanduser()
        explicit['summary'] = explicit['figure'].parent / 'baseline_benchmark_summary.json'
    if args.summary_json:
        explicit['summary'] = Path(args.summary_json)
    outputs = ValidationOutputs(
        ap, args, name='baseline',
        canonical={'figure': 'baseline_benchmark.png', 'summary': 'baseline_benchmark_summary.json'},
        settings={
            'n_train': 20000, 'n_val': 2000, 'n_test': 3000,
            'n_cortical': 8000, 'n_subcortical': 1500, 'n_legendre': 60,
            'montage': 'standard_1010_69.csv', 'snr_low': 10.0, 'snr_high': 25.0,
            'seed': 862, 'hidden': [256, 128], 'max_iter': 300,
        },
        explicit=explicit,
        ordinary={'summary': 'baseline_benchmark_summary.json'},
    )
    if args.output_dir:
        args.output_dir = str(check_work_directory(
            ap, args.output_dir, overwrite=args.overwrite,
            result_paths=outputs.paths.values(),
        ))
    with outputs.stage() as paths:
        if args.output_dir:
            _run(args, paths, outputs.paths)
        else:
            with tempfile.TemporaryDirectory(
                prefix='.baseline-data-', dir=paths['summary'].parent.parent,
            ) as temporary:
                args.output_dir = str(Path(temporary) / 'dataset')
                _run(args, paths, outputs.paths)
    for path in outputs.paths.values():
        print(f" Written to {path}")


def _run(args, paths, final_paths):
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.exceptions import ConvergenceWarning

    t0 = time.time()
    print("Generating dataset ...")
    generate_dataset(args)

    out = Path(args.output_dir)
    Xtr, Ytr = load_split(out / 'train.h5')
    Xva, Yva = load_split(out / 'val.h5')
    Xte, Yte = load_split(out / 'test.h5')
    print(f"  train {Xtr.shape}  val {Xva.shape}  test {Xte.shape}")

    print("Training MLP ...")
    x_scaler = StandardScaler().fit(Xtr)
    y_scaler = StandardScaler().fit(Ytr)
    Xtr_scaled = x_scaler.transform(Xtr)
    Xva_scaled = x_scaler.transform(Xva)
    Xte_scaled = x_scaler.transform(Xte)
    Ytr_scaled = y_scaler.transform(Ytr)

    model = MLPRegressor(
        hidden_layer_sizes=tuple(args.hidden),
        activation='relu',
        solver='adam',
        early_stopping=False,
        max_iter=1,
        random_state=args.seed,
    )
    patience = 15
    best_val_mse = np.inf
    best_epoch = 0
    best_weights = None
    epochs_without_improvement = 0
    epochs_run = 0
    for epoch in range(1, args.max_iter + 1):
        epochs_run = epoch
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.partial_fit(Xtr_scaled, Ytr_scaled)
        val_pred = y_scaler.inverse_transform(model.predict(Xva_scaled))
        val_mse = float(np.mean((val_pred - Yva) ** 2))
        if val_mse < best_val_mse - 1e-8:
            best_val_mse = val_mse
            best_epoch = epoch
            best_weights = (
                copy.deepcopy(model.coefs_),
                copy.deepcopy(model.intercepts_),
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_weights is None:
        raise RuntimeError("MLP training did not produce a finite validation score")
    model.coefs_, model.intercepts_ = best_weights

    pred = y_scaler.inverse_transform(model.predict(Xte_scaled))
    err = np.linalg.norm(pred - Yte, axis=1)             # mm

    # Declared constant reference: predict the training-position mean.
    mean_pos = Ytr.mean(axis=0, keepdims=True)
    base_err = np.linalg.norm(np.repeat(mean_pos, len(Yte), axis=0) - Yte, axis=1)

    print()
    print("=" * 64)
    print(f" Baseline source-localization benchmark ({args.montage})")
    print(f" SNR band {args.snr_low:.0f}-{args.snr_high:.0f} dB, "
          f"{args.n_train} train / {args.n_test} test, MLP {tuple(args.hidden)}")
    print("=" * 64)
    print(f" Median localization error : {np.median(err):6.2f} mm")
    print(f" Mean   localization error : {np.mean(err):6.2f} mm")
    print(f" 90th percentile error     : {np.percentile(err, 90):6.2f} mm")
    print(f" Predict-mean baseline med : {np.median(base_err):6.2f} mm  "
          f"({np.median(base_err) / np.median(err):.1f}x worse than the MLP)")
    print(" Depth-stratified median error:")
    depth_results = depth_stratified(err, Yte)
    for label, n, med in depth_results:
        print(f"   {label:24s} n={n:5d}   {med:6.2f} mm")
    print(f"\n Total wall time: {time.time() - t0:.1f}s")

    if 'figure' in paths and not _make_figure(paths['figure'], err, Yte):
        raise RuntimeError("Install matplotlib to write the benchmark figure.")

    if 'summary' in paths:
        summary = {
            'method': 'scikit-learn MLPRegressor on peak-GFP topography',
            'montage': args.montage,
            'seed': args.seed,
            'dataset': {
                'n_train': args.n_train,
                'n_val': args.n_val,
                'n_test': args.n_test,
                'n_cortical_positions': args.n_cortical,
                'n_subcortical_positions': args.n_subcortical,
                'n_legendre': args.n_legendre,
                'noise_model': 'snr_target',
                'eeg_normalization': 'per_sample_rms',
                'target_snr_db_range': [args.snr_low, args.snr_high],
            },
            'hidden_layer_sizes': args.hidden,
            'model_selection': {
                'validation_data': 'position-held-out val.h5',
                'criterion': 'minimum validation mean squared position error',
                'patience_epochs': patience,
                'best_epoch': best_epoch,
                'epochs_run': epochs_run,
                'best_validation_mse_mm2': best_val_mse,
            },
            'software_environment': environment_snapshot(),
            'localization_error_mm': {
                'median': float(np.median(err)),
                'mean': float(np.mean(err)),
                'p90': float(np.percentile(err, 90)),
            },
            'predict_training_mean_baseline_median_mm': float(
                np.median(base_err)
            ),
            'depth_stratified': [
                {'band': label, 'n': n, 'median_error_mm': med}
                for label, n, med in depth_results
            ],
        }
        if 'figure' in paths:
            figure_path = paths['figure']
            summary['figure'] = relative_reference(final_paths['figure'], final_paths['summary'])
            summary['figure_sha256'] = hashlib.sha256(
                figure_path.read_bytes()
            ).hexdigest()
        summary_path = paths['summary']
        with summary_path.open('w', newline='\n') as f:
            json.dump(summary, f, indent=2, sort_keys=True)
            f.write('\n')


def _make_figure(path, err, pos_mm):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available; skipping figure.")
        return False
    radius = np.linalg.norm(pos_mm, axis=1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

    ax[0].hist(err, bins=40, color='#4477aa', alpha=0.9)
    ax[0].axvline(np.median(err), color='k', ls='--',
                  label=f'median {np.median(err):.2f} mm')
    ax[0].set_xlabel('localization error (mm)')
    ax[0].set_ylabel('test samples')
    ax[0].set_title('Error distribution')
    ax[0].legend()

    ax[1].scatter(radius, err, s=6, alpha=0.35, color='#4477aa')
    ax[1].set_xlabel('source radius (mm)  [smaller = deeper]')
    ax[1].set_ylabel('localization error (mm)')
    ax[1].set_title('Error vs. source depth')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


if __name__ == '__main__':
    main()
