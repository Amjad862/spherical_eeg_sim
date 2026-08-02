"""Generate deterministic descriptive validation figures.

Produces three reusable figures:

  * ``pipeline_schematic.pdf`` / ``.png`` -- the end-to-end generation pipeline
    as a labelled block diagram (data-independent).
  * ``topographies.png`` -- representative average-referenced scalp topographies
    for equal radial dipoles at superficial, middle, and deep model radii.
  * ``distributions.png`` -- a compact multi-panel summary of a generated
    dataset: source radial distribution by population, dipole-magnitude
    distribution, emergent-SNR distribution, orientation uniformity, frequency-
    band counts, and the train/val/test radial separation (position-level
    split).

Everything is generated deterministically (fixed master seed) so the figures
are reproducible. matplotlib is required.

Usage
-----
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
        "Config\n(dataclasses)",
        "Electrode\nmontage",
        "Four-sphere\nforward model\n(transfer coeffs)",
        "Source position\npools\n(cortical / subcortical)",
        "Position-level\ntrain / val / test\nsplit",
        "Per-sample seeds\n(sample, split, index)",
        "Orientation +\nmagnitude +\ntime course",
        "Lead-field\nprojection",
        "Three-component\nnoise + SNR /\nabsolute floor",
        "Reference +\nground truth",
        "HDF5 / NPZ / MAT\n+ metadata",
        "Validation +\ndiagnostics",
    ]
    fig, ax = plt.subplots(figsize=(12, 5.2))
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

    boxw, boxh = 0.86, 0.66
    colors = ['#e8f0fa'] * len(steps)
    for (cx, cy), label, col in zip(positions, steps, colors):
        box = FancyBboxPatch(
            (cx - boxw / 2, cy - boxh / 2), boxw, boxh,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=1.2, edgecolor='#33517a', facecolor=col)
        ax.add_patch(box)
        ax.text(cx, cy, label, ha='center', va='center', fontsize=8.2)

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
            arrowstyle='-|>', mutation_scale=12, linewidth=1.1,
            color='#33517a',
            shrinkA=2, shrinkB=2,
            connectionstyle="arc3,rad=0.0")
        ax.add_patch(arr)

    ax.set_title("spherical_eeg_sim generation pipeline", fontsize=12, pad=8)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        output_path = Path(outdir) / f'pipeline_schematic.{ext}'
        if ext == 'pdf':
            fig.savefig(
                output_path,
                dpi=200,
                bbox_inches='tight',
                metadata={'CreationDate': None, 'ModDate': None},
            )
        else:
            fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


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
    labels = ['superficial (78 mm)', 'mid (60 mm)', 'deep (40 mm)']
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
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), constrained_layout=True)
    tc = None
    for ax, V_uV, lab in zip(axes, maps_uV, labels):
        tc = ax.tricontourf(ex, ey, V_uV, levels=levels, cmap='RdBu_r',
                            extend='both')
        ax.scatter(ex, ey, s=6, c='k', alpha=0.35)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(lab, fontsize=10)
    assert tc is not None
    fig.colorbar(
        tc, ax=axes.tolist(), fraction=0.035, pad=0.03,
        label='potential (µV)',
    )
    fig.suptitle("Average-referenced scalp maps for a 30 nA·m radial dipole",
                 fontsize=12)
    fig.savefig(Path(outdir) / 'topographies.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: dataset distributions
# ---------------------------------------------------------------------------

def fig_distributions(outdir, workdir):
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
    cfg.output.overwrite = True
    run_pipeline(cfg, log_level=40)

    d = Path(workdir) / 'figdata'
    def read(split):
        with h5py.File(d / f'{split}.h5', 'r') as f:
            return {k: f[f'ground_truth/{k}'][:] for k in
                    ('positions_xyz', 'magnitudes_nAm', 'snr_db',
                     'orientations_world', 'band_idx', 'population')}
    tr = read('train'); va = read('val'); te = read('test')

    fig, ax = plt.subplots(2, 3, figsize=(13, 7.5))

    # (a) radial distribution by population
    r = np.linalg.norm(tr['positions_xyz'], axis=1) * 1000
    pop = tr['population']
    ax[0, 0].hist(r[pop == 0], bins=40, alpha=0.7, label='cortical', color='#4477aa')
    ax[0, 0].hist(r[pop == 1], bins=40, alpha=0.7, label='subcortical', color='#cc6677')
    ax[0, 0].set_xlabel('source radius (mm)'); ax[0, 0].set_ylabel('count')
    ax[0, 0].set_title('(a) source radial distribution'); ax[0, 0].legend()

    # (b) magnitude distribution (log)
    ax[0, 1].hist(np.log10(tr['magnitudes_nAm']), bins=40, color='#228833')
    ax[0, 1].set_xlabel('log10 dipole magnitude (nA·m)'); ax[0, 1].set_ylabel('count')
    ax[0, 1].set_title('(b) magnitude (log-uniform)')

    # (c) emergent SNR
    ax[0, 2].hist(tr['snr_db'], bins=40, color='#aa3377')
    ax[0, 2].set_xlabel('emergent SNR (dB)'); ax[0, 2].set_ylabel('count')
    ax[0, 2].set_title('(c) emergent SNR (absolute floor)')

    # (d) orientation z-component (uniform on S2 -> Unif[-1,1])
    ax[1, 0].hist(tr['orientations_world'][:, 2], bins=40, color='#66ccee')
    ax[1, 0].set_xlabel('orientation z-component'); ax[1, 0].set_ylabel('count')
    ax[1, 0].set_title('(d) orientation uniformity (uniform S²)')

    # (e) frequency-band counts
    bands = np.bincount(tr['band_idx'], minlength=5)
    names = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    ax[1, 1].bar(range(len(bands)), bands, color='#ee7733')
    ax[1, 1].set_xticks(range(len(names))); ax[1, 1].set_xticklabels(names, rotation=30)
    ax[1, 1].set_ylabel('count'); ax[1, 1].set_title('(e) frequency band (window-resolvable)')

    # (f) train/val/test radial separation (position-level split)
    for split, dd, col in (('train', tr, '#4477aa'), ('val', va, '#228833'),
                           ('test', te, '#cc6677')):
        rr = np.linalg.norm(dd['positions_xyz'], axis=1) * 1000
        ax[1, 2].hist(rr, bins=40, histtype='step', density=True, label=split, color=col)
    ax[1, 2].set_xlabel('source radius (mm)'); ax[1, 2].set_ylabel('density')
    ax[1, 2].set_title('(f) split radial coverage'); ax[1, 2].legend()

    fig.suptitle(
        "Generated-dataset distributions (deterministic reduced-size run)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(Path(outdir) / 'distributions.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
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
    ap.add_argument('--outdir', type=str, default='figures')
    ap.add_argument(
        '--workdir', type=str, default=None,
        help='optional retained work directory; otherwise a temporary directory is used',
    )
    args = ap.parse_args()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    print("Figure 1/3: pipeline schematic ...")
    fig_pipeline_schematic(args.outdir)
    print("Figure 2/3: representative topographies ...")
    fig_topographies(args.outdir)
    print("Figure 3/3: dataset distributions ...")
    if args.workdir is None:
        with tempfile.TemporaryDirectory(prefix='spherical_eeg_sim_figures_') as workdir:
            distribution_statistics = fig_distributions(args.outdir, workdir)
    else:
        Path(args.workdir).mkdir(parents=True, exist_ok=True)
        distribution_statistics = fig_distributions(args.outdir, args.workdir)
    outdir = Path(args.outdir)
    figure_files = (
        'pipeline_schematic.pdf',
        'pipeline_schematic.png',
        'topographies.png',
        'distributions.png',
    )
    manifest = {
        'software_environment': environment_snapshot(),
        'distribution_statistics': distribution_statistics,
        'files': {
            name: hashlib.sha256((outdir / name).read_bytes()).hexdigest()
            for name in figure_files
        },
    }
    manifest_path = outdir / 'publication_figures.json'
    with manifest_path.open('w', encoding='utf-8', newline='\n') as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write('\n')
    print(f"Figures written to {args.outdir}/")


if __name__ == '__main__':
    main()
