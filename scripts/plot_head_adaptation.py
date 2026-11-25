"""Plot localization error against the number of boundary-element training windows.

Reads the summary of a ``run_head_adaptation.py`` run and draws, for networks
fine-tuned from the sphere-trained network and for networks trained on the
boundary-element windows alone, the mean of the dataset-seed median errors on
the boundary-element test windows, with the range over dataset seeds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('run', type=Path, help='run_head_adaptation.py output directory')
    parser.add_argument('--output-dir', required=True, type=Path, help='new folder for the figure')
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('choose a new output directory')
    summary_path = args.run / 'summary.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    headline = summary['bem_windows_headline']
    seeds = summary['by_dataset_seed']
    all_windows = float(np.mean([seed['bem_training_eligibility']['train']['eligible_windows']
                                 for seed in seeds.values()]))
    labels = [arm.split('_', 1)[1] for arm in headline if arm.startswith('finetuned_')]
    sizes = [all_windows if label == 'all' else float(label) for label in labels]

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.4, 2.9))
    styles = (('finetuned', 'Pretrained on the sphere, fine-tuned', '#4477AA', 'o', '-'),
              ('scratch', 'Trained on these windows alone', '#EE6677', 's', '--'))
    for arm, label, color, marker, line in styles:
        entries = [headline[f'{arm}_{size}'] for size in labels]
        mean = np.array([entry['mean_of_dataset_seed_medians'] for entry in entries])
        low = np.array([entry['range'][0] for entry in entries])
        high = np.array([entry['range'][1] for entry in entries])
        ax.errorbar(sizes, mean, yerr=[mean - low, high - mean], color=color, marker=marker, linestyle=line,
                    markersize=4.5, linewidth=1.2, capsize=2.5, label=label)
    frozen = headline['default']['mean_of_dataset_seed_medians']
    sphere = summary['own_sphere_eligible_headline']['default']['mean_of_dataset_seed_medians']
    ax.axhline(frozen, color='#555555', linewidth=0.9, linestyle=':')
    ax.axhline(sphere, color='#555555', linewidth=0.9, linestyle='-.')
    ax.text(sizes[0] * 0.92, frozen * 1.04, 'Sphere-trained, not adapted', fontsize=8, color='#333333')
    ax.text(sizes[0] * 0.92, sphere * 0.86, 'Sphere-trained, sphere windows', fontsize=8, color='#333333')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks(sizes)
    # The last point uses every eligible training window; their number differs
    # slightly between dataset seeds, so the tick shows the rounded mean.
    ax.set_xticklabels([f'all\n(about {int(round(size, -2)):,})' if label == 'all' else f'{int(round(size)):,}'
                        for label, size in zip(labels, sizes)], fontsize=8)
    ax.minorticks_off()
    ticks = [4, 5, 7, 10, 15, 20, 25]
    ax.set_yticks(ticks)
    ax.set_yticklabels([str(tick) for tick in ticks], fontsize=8)
    ax.set_ylim(3.6, 30)
    ax.set_xlim(sizes[0] * 0.8, sizes[-1] * 1.3)
    ax.set_xlabel('Boundary-element training windows', fontsize=8.5)
    ax.set_ylabel('Median localization error (mm)', fontsize=8.5)
    ax.legend(fontsize=8, frameon=False, loc='upper right')
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    fig.tight_layout(pad=0.4)
    args.output_dir.mkdir(parents=True)
    fig.savefig(args.output_dir / 'head_adaptation_finetuning.png', dpi=600)
    fig.savefig(args.output_dir / 'head_adaptation_finetuning.pdf', metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)
    record = dict(
        summary_sha256=hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        training_windows={label: size for label, size in zip(labels, sizes)},
        plotted={arm: {size: headline[f'{arm}_{size}'] for size in labels} for arm in ('finetuned', 'scratch')},
        frozen_network_on_bem_windows_mm=frozen, frozen_network_on_sphere_windows_mm=sphere,
        files={name: hashlib.sha256((args.output_dir / name).read_bytes()).hexdigest()
               for name in ('head_adaptation_finetuning.png', 'head_adaptation_finetuning.pdf')},
    )
    with (args.output_dir / 'head_adaptation_finetuning_source.json').open('w', encoding='utf-8', newline='\n') as file:
        json.dump(record, file, indent=2)
        file.write('\n')
    print(f'Figure written to {args.output_dir}')


if __name__ == '__main__':
    main()
