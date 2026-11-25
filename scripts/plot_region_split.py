"""Draw the region split of the two supervised studies on a map of source directions.

Panel (a) shows the cortical positions of the point-source split study (24
sectors, 15 mm clearance); panel (b) shows the patch split study (8 sectors,
15 mm clearance) and marks the positions that can hold the largest patch.
Only the split geometry is computed; no lead field or EEG is generated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spherical_eeg_sim import plan_position_split
import run_patch_split_study
import run_split_study

COLORS = {'train': '#4477AA', 'val': '#CCBB44', 'test': '#EE6677', 'excluded': '#C8C8C8'}
LABELS = {'train': 'Training', 'val': 'Validation', 'test': 'Test', 'excluded': 'Excluded by the buffer'}


def directions(positions):
    radius = np.linalg.norm(positions, axis=1)
    return np.arctan2(positions[:, 1], positions[:, 0]), np.arcsin(positions[:, 2] / radius)


def draw(ax, plan, title, largest_radius_mm=None):
    cortical = plan.population == 0
    longitude, latitude = directions(plan.positions_xyz)
    for code, name in ((-1, 'excluded'), (0, 'train'), (1, 'val'), (2, 'test')):
        mask = cortical & (plan.split_labels == code)
        ax.scatter(longitude[mask], latitude[mask], s=1.6, color=COLORS[name], linewidths=0,
                   alpha=0.45 if largest_radius_mm is not None and code >= 0 else 1.0, rasterized=True)
        if largest_radius_mm is not None and code >= 0:
            holds = mask & (plan.patch_containment_radius_mm >= largest_radius_mm)
            ax.scatter(longitude[holds], latitude[holds], s=1.6, color=COLORS[name], linewidths=0,
                       rasterized=True)
    ax.set_title(title, fontsize=8.5, loc='left')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, linewidth=0.3, color='#888888', alpha=0.5)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output-dir', required=True, type=Path, help='new folder for the figure')
    parser.add_argument('--seed', type=int, default=run_split_study.DATA_SEEDS[0],
                        help='dataset seed whose region assignment is drawn')
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('choose a new output directory')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    point_cfg = run_split_study.config_for(args.seed, 'region', args.output_dir / 'unused')
    patch_cfg = run_patch_split_study.config_for(args.seed, 'region', args.output_dir / 'unused')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        point, patch = plan_position_split(point_cfg), plan_position_split(patch_cfg)
    largest = patch_cfg.patch.diameter_range_mm[1]

    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.35), subplot_kw=dict(projection='mollweide'))
    draw(axes[0], point, f'(a) Point sources: {point_cfg.split.region_count} sectors')
    draw(axes[1], patch, f'(b) Patches: {patch_cfg.split.region_count} sectors', largest / 2)
    handles = [Line2D([], [], marker='o', linestyle='', markersize=4, color=COLORS[name], label=LABELS[name])
               for name in ('train', 'val', 'test', 'excluded')]
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=8, frameon=False, handletextpad=0.2,
               columnspacing=1.4)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.14, wspace=0.05)
    args.output_dir.mkdir(parents=True)
    fig.savefig(args.output_dir / 'region_split.png', dpi=600)
    fig.savefig(args.output_dir / 'region_split.pdf', dpi=600, metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)

    def counts(plan):
        return {name: int(np.sum(plan.split_labels == code))
                for code, name in ((0, 'train'), (1, 'val'), (2, 'test'), (-1, 'excluded'))}

    record = dict(
        dataset_seed=args.seed,
        point=dict(region_count=point_cfg.split.region_count, buffer_mm=point_cfg.split.buffer_mm,
                   rotation_seed=point_cfg.split.region_rotation_seed, positions=counts(point),
                   drawn='cortical positions'),
        patch=dict(region_count=patch_cfg.split.region_count, buffer_mm=patch_cfg.split.buffer_mm,
                   rotation_seed=patch_cfg.split.region_rotation_seed, positions=counts(patch),
                   largest_diameter_mm=largest,
                   positions_holding_largest_patch={
                       name: int(np.sum((patch.split_labels == code)
                                        & (patch.patch_containment_radius_mm >= largest / 2)))
                       for code, name in ((0, 'train'), (1, 'val'), (2, 'test'))}),
        projection='Mollweide map of source directions; full-strength colour in panel (b) marks positions '
                   'that can hold the largest patch',
        files={name: hashlib.sha256((args.output_dir / name).read_bytes()).hexdigest()
               for name in ('region_split.png', 'region_split.pdf')},
    )
    with (args.output_dir / 'region_split_source.json').open('w', encoding='utf-8', newline='\n') as file:
        json.dump(record, file, indent=2)
        file.write('\n')
    print(f'Figure written to {args.output_dir}')


if __name__ == '__main__':
    main()
