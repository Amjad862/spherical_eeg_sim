"""Generate deterministic descriptive validation figures.

Produces three reusable figures:

  * ``pipeline_schematic.pdf`` / ``.png`` -- the end-to-end generation pipeline
    as a labelled block diagram (data-independent).
  * ``topographies.pdf`` / ``.png`` -- representative average-referenced scalp topographies
    for equal radial dipoles at superficial, middle, and deep model radii.
  * ``distributions.pdf`` / ``.png`` -- a compact multi-panel summary of a generated
    dataset: source radial distribution by population, dipole-magnitude
    distribution, emergent-SNR distribution, orientation uniformity, frequency-
    band counts, and the train/val/test radial separation (position-level
    split).

Everything is generated deterministically (fixed master seed) so the figures
are reproducible. matplotlib is required.

Usage
-----
    python scripts/make_validation_figures.py
    python scripts/make_validation_figures.py --outdir figures
"""

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts._validation_outputs import (
    ValidationOutputs, add_output_options, check_work_directory,
)
from spherical_eeg_sim import Config, run_pipeline
from spherical_eeg_sim.core.electrodes import load_electrodes
from spherical_eeg_sim.core.forward import (
    compute_transfer_coefficients, forward_potential,
)
from spherical_eeg_sim.sources.sampling import (
    cartesian_to_spherical, world_orientation_to_spherical_frame,
)
from spherical_eeg_sim.provenance import environment_snapshot


# ---------------------------------------------------------------------------
# Figure 1: pipeline schematic
# ---------------------------------------------------------------------------

def fig_pipeline_schematic(outdir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    steps = [
        "Configuration",
        "Electrode\nmontage",
        "Four-sphere\nforward model",
        "Source position\npools",
        "Split policy:\nexact, region\nor shared",
        "Per-window\nrandom streams",
        "Orientation,\nmagnitude,\ntime course",
        "Lead-field\nprojection",
        "Noise mixture\nand scaling",
        "Reference and\nground truth",
        "HDF5, NPZ or\nMATLAB output",
        "Validation and\npublication",
    ]
    # Drawn at the width of the journal text block so that the 8 pt labels
    # print at 8 pt.
    fig, ax = plt.subplots(figsize=(6.3, 3.0))
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 3)
    ax.axis('off')

    # 4 columns x 3 rows, boustrophedon flow.
    ncol, nrow = 4, 3
    positions = []
    for r in range(nrow):
        row_cols = range(ncol) if r % 2 == 0 else reversed(range(ncol))
        for c in row_cols:
            positions.append((c + 0.5, nrow - 0.5 - r))

    boxw, boxh = 0.84, 0.70
    colors = ['#e8f0fa'] * len(steps)
    for (cx, cy), label, col in zip(positions, steps, colors):
        box = FancyBboxPatch(
            (cx - boxw / 2, cy - boxh / 2), boxw, boxh,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=0.9, edgecolor='#33517a', facecolor=col)
        ax.add_patch(box)
        ax.text(cx, cy, label, ha='center', va='center', fontsize=8.0, linespacing=1.15)

    for i in range(len(positions) - 1):
        x0, y0 = positions[i]
        x1, y1 = positions[i + 1]
        dx, dy = x1 - x0, y1 - y0
        if abs(dx) >= abs(dy):
            direction = 1.0 if dx > 0 else -1.0
            start = (x0 + direction * boxw / 2.0, y0)
            end = (x1 - direction * boxw / 2.0, y1)
        else:
            direction = 1.0 if dy > 0 else -1.0
            start = (x0, y0 + direction * boxh / 2.0)
            end = (x1, y1 - direction * boxh / 2.0)
        arr = FancyArrowPatch(
            start, end,
            arrowstyle='-|>', mutation_scale=8, linewidth=0.9,
            color='#33517a',
            shrinkA=1.5, shrinkB=1.5,
            connectionstyle="arc3,rad=0.0")
        ax.add_patch(arr)

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    _save_figure(fig, outdir, 'pipeline_schematic', dpi=600)
    plt.close(fig)


def _save_figure(fig, outdir, stem, dpi):
    """Write a raster and a vector version with reproducible PDF metadata."""
    fig.savefig(Path(outdir) / f'{stem}.png', dpi=dpi)
    fig.savefig(Path(outdir) / f'{stem}.pdf', metadata={'CreationDate': None, 'ModDate': None})


# ---------------------------------------------------------------------------
# Figure 2: representative topographies
# ---------------------------------------------------------------------------

def _azimuthal_xy(spherical):
    """Azimuthal-equidistant projection of electrode positions for plotting."""
    theta = spherical[:, 1]     # polar angle from +z
    phi = spherical[:, 2]
    r = theta                   # radial coord = polar angle
    return r * np.cos(phi), r * np.sin(phi)


def fig_topographies(outdir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cfg = Config()
    el = load_electrodes(cfg.electrode_file, target_radius_m=cfg.head.radii_m[-1])
    M = compute_transfer_coefficients(
        cfg.head.radii_m, cfg.head.conductivities_S_per_m, cfg.head.n_legendre)
    ex, ey = _azimuthal_xy(el.spherical)

    # One representative radial dipole at three depths (same angular position).
    depths_mm = [78.0, 60.0, 40.0]
    labels = ['Superficial (78 mm)', 'Middle (60 mm)', 'Deep (40 mm)']
    theta0, phi0 = 0.6, 0.4
    ori_world = np.array([np.sin(theta0) * np.cos(phi0),
                          np.sin(theta0) * np.sin(phi0),
                          np.cos(theta0)])   # radial orientation

    maps_uV = []
    for r_mm in depths_mm:
        pos = np.array([r_mm / 1000.0 * np.sin(theta0) * np.cos(phi0),
                        r_mm / 1000.0 * np.sin(theta0) * np.sin(phi0),
                        r_mm / 1000.0 * np.cos(theta0)])
        rtp = cartesian_to_spherical(pos)
        loc = world_orientation_to_spherical_frame(ori_world, pos)
        V = forward_potential(rtp[0], rtp[1], rtp[2],
                              30e-9 * loc[0], 30e-9 * loc[1], 30e-9 * loc[2],
                              el.theta_rad, el.phi_rad, M,
                              brain_radius_m=cfg.head.radii_m[0])
        maps_uV.append((V - V.mean()) * 1e6)

    vmax = max(float(np.max(np.abs(v))) for v in maps_uV)
    levels = np.linspace(-vmax, vmax, 15)
    # Journal text width; the caption carries the title.
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.3), constrained_layout=True)
    tc = None
    for ax, V_uV, lab in zip(axes, maps_uV, labels):
        tc = ax.tricontourf(ex, ey, V_uV, levels=levels, cmap='RdBu_r',
                            extend='both')
        ax.scatter(ex, ey, s=2.5, c='k', alpha=0.35, linewidths=0)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(lab, fontsize=8.5)
    assert tc is not None
    bar = fig.colorbar(tc, ax=axes.tolist(), fraction=0.035, pad=0.03)
    bar.set_label('Potential (µV)', fontsize=8.5)
    bar.ax.tick_params(labelsize=8)
    _save_figure(fig, outdir, 'topographies', dpi=600)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: dataset distributions
# ---------------------------------------------------------------------------

def fig_distributions(outdir, workdir, *, overwrite=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import h5py

    cfg = Config()
    cfg.source_mode = 'point'
    cfg.dataset.n_train = 6000
    cfg.dataset.n_val = 1500
    cfg.dataset.n_test = 1500
    cfg.sources.n_cortical_positions = 3000
    cfg.sources.n_subcortical_positions = 700
    cfg.head.n_legendre = 50
    cfg.master_seed = 862
    cfg.output.diagnostic_plots = False
    cfg.output_dir = str(Path(workdir) / 'figdata')
    cfg.output.overwrite = overwrite
    run_pipeline(cfg, log_level=40)

    d = Path(workdir) / 'figdata'
    def read(split):
        with h5py.File(d / f'{split}.h5', 'r') as f:
            return {k: f[f'ground_truth/{k}'][:] for k in
                    ('positions_xyz', 'magnitudes_nAm', 'snr_db',
                     'orientations_world', 'band_idx', 'population')}
    tr = read('train'); va = read('val'); te = read('test')

    # Drawn at the printed width of the text block; no text is smaller than 8 pt.
    plt.rcParams.update({'font.size': 8.5, 'axes.titlesize': 9, 'axes.labelsize': 8.5,
                         'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8})
    fig, axes = plt.subplots(3, 2, figsize=(6.3, 5.2), layout='constrained')
    ax = axes.ravel()

    # (a) radial distribution by population
    r = np.linalg.norm(tr['positions_xyz'], axis=1) * 1000
    pop = tr['population']
    ax[0].hist(r[pop == 0], bins=40, alpha=0.7, label='cortical', color='#4477aa')
    ax[0].hist(r[pop == 1], bins=40, alpha=0.7, label='subcortical', color='#cc6677')
    ax[0].set(xlabel='Source radius (mm)', ylabel='Count', title='(a) Source radius')
    ax[0].legend()

    # (b) magnitude distribution (log)
    ax[1].hist(np.log10(tr['magnitudes_nAm']), bins=40, color='#228833')
    ax[1].set(xlabel='log₁₀ dipole magnitude (nA·m)', ylabel='Count', title='(b) Dipole magnitude')

    # (c) emergent SNR
    ax[2].hist(tr['snr_db'], bins=40, color='#aa3377')
    ax[2].set(xlabel='SNR (dB)', ylabel='Count', title='(c) Emergent SNR')

    # (d) orientation z-component (uniform on S2 -> Unif[-1,1])
    ax[3].hist(tr['orientations_world'][:, 2], bins=40, color='#66ccee')
    ax[3].set(xlabel='Orientation z-component', ylabel='Count', title='(d) Dipole orientation')

    # (e) frequency-band counts
    bands = np.bincount(tr['band_idx'], minlength=5)
    names = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    ax[4].bar(range(len(bands)), bands, color='#ee7733')
    ax[4].set_xticks(range(len(names)), names)
    ax[4].set(ylabel='Count', title='(e) Frequency band')

    # (f) train/val/test radial separation (position-level split)
    for split, dd, col in (('train', tr, '#4477aa'), ('val', va, '#228833'),
                           ('test', te, '#cc6677')):
        rr = np.linalg.norm(dd['positions_xyz'], axis=1) * 1000
        ax[5].hist(rr, bins=40, histtype='step', density=True, label=split, color=col)
    ax[5].set(xlabel='Source radius (mm)', ylabel='Density', title='(f) Radius by split')
    ax[5].legend()
    _save_figure(fig, outdir, 'distributions', dpi=600)
    plt.close(fig)
    plt.rcdefaults()
    return {
        'train_snr_db': {
            'minimum': float(np.min(tr['snr_db'])),
            'p01': float(np.percentile(tr['snr_db'], 1)),
            'median': float(np.median(tr['snr_db'])),
            'p99': float(np.percentile(tr['snr_db'], 99)),
            'maximum': float(np.max(tr['snr_db'])),
        },
        'train_band_counts': [int(value) for value in bands],
        'noise_model': cfg.noise.noise_model,
        'absolute_noise_rms_uV': float(cfg.noise.absolute_noise_rms_uV),
        'master_seed': int(cfg.master_seed),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--outdir', type=str)
    ap.add_argument(
        '--workdir', type=str, default=None,
        help='optional retained work directory; otherwise a temporary directory is used',
    )
    add_output_options(ap)
    args = ap.parse_args()
    figure_files = (
        'pipeline_schematic.pdf',
        'pipeline_schematic.png',
        'topographies.pdf',
        'topographies.png',
        'distributions.pdf',
        'distributions.png',
    )
    manifest_name = 'publication_figures.json'
    names = (*figure_files, manifest_name)
    outputs = ValidationOutputs(
        ap, args, name='figures',
        canonical={name: name for name in names},
        explicit={
            name: Path(args.outdir) / name for name in names
        } if args.outdir else None,
    )
    workdir = None
    if args.workdir:
        workdir = check_work_directory(
            ap, Path(args.workdir) / 'figdata', overwrite=args.overwrite,
            result_paths=outputs.paths.values(),
        ).parent

    with outputs.stage() as paths:
        outdir = paths[manifest_name].parent
        print("Figure 1/3: pipeline schematic ...")
        fig_pipeline_schematic(outdir)
        print("Figure 2/3: representative topographies ...")
        fig_topographies(outdir)
        print("Figure 3/3: dataset distributions ...")
        if workdir is None:
            with tempfile.TemporaryDirectory(prefix='.figure-data-', dir=outdir.parent) as temporary:
                distribution_statistics = fig_distributions(outdir, temporary)
        else:
            distribution_statistics = fig_distributions(outdir, workdir, overwrite=args.overwrite)
        _write_summary(paths, figure_files, manifest_name, distribution_statistics)
    print(f"Figures written to {outputs.paths[manifest_name].parent}")


def _write_summary(paths, figure_files, manifest_name, distribution_statistics):
    manifest = {
        'software_environment': environment_snapshot(),
        'distribution_statistics': distribution_statistics,
        'files': {
            name: hashlib.sha256(paths[name].read_bytes()).hexdigest()
            for name in figure_files
        },
    }
    manifest_path = paths[manifest_name]
    with manifest_path.open('w', encoding='utf-8', newline='\n') as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write('\n')


if __name__ == '__main__':
    main()
