"""Plot error by distance to the nearest observed training location."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    runs = json.loads((args.study / 'summary.json').read_text(encoding='utf-8'))
    if set(runs) != {'862', '863', '864'}:
        parser.error('the figure needs results from all three declared dataset seeds')
    output = args.output_dir.resolve()
    canonical = Path(__file__).resolve().parents[1] / 'validation_results'
    if output == canonical or canonical in output.parents:
        parser.error('choose a new run folder outside validation_results')
    output.mkdir(parents=True, exist_ok=False)
    bins = ('0_5', '5_10', '10_20', '20_30', '30_inf')
    labels = ('0–5', '5–10', '10–20', '20–30', '≥30')
    methods = (('mlp_pooled', 'MLP', '#4477aa', 'o'),
               ('dictionary', 'Lead-field dictionary', '#cc6677', 's'),
               ('knn10', '10-nearest neighbors', '#228833', '^'))
    # Drawn at the printed width of the text block; no text is smaller than 8 pt.
    plt.rcParams.update({'font.size': 8.5, 'axes.titlesize': 9, 'axes.labelsize': 8.5,
                         'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8.5,
                         'lines.linewidth': 1.2, 'lines.markersize': 4.5})
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.0), sharey=True, layout='constrained')
    for ax, policy, title in zip(axes, ('position', 'region'),
                                 ('(a) Exact-position split', '(b) Buffered region split')):
        for method, label, color, marker in methods:
            medians = []
            for interval in bins:
                statistics = [run[policy]['full'][method]['strata'].get(f'distance_{interval}')
                              for run in runs.values()]
                values = [item['median_mm'] for item in statistics
                          if item is not None and item['positions'] >= 30]
                medians.append(float(np.mean(values)) if values else np.nan)
            ax.plot(range(len(bins)), medians, color=color, marker=marker, label=label)
        ax.set_title(title)
        ax.set_xticks(range(len(bins)), labels)
        ax.set_xlabel('Nearest training position (mm)')
        ax.grid(axis='y', alpha=.25)
        ax.set_xlim(-.25, len(bins) - .75)
    axes[0].set_ylim(bottom=0)
    axes[0].set_ylabel('Localization error (mm)')
    axes[0].legend(loc='best')
    fig.savefig(output / 'split_localization_error.png', dpi=600)
    fig.savefig(output / 'split_localization_error.pdf',
                metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    record = dict(
        summary_sha256=digest(args.study / 'summary.json'),
        plotted=dict(methods=[method for method, *_ in methods], distance_bins_mm=list(labels),
                     rule='mean over dataset seeds with at least 30 distinct test positions in the bin'),
        files={name: digest(output / name)
               for name in ('split_localization_error.png', 'split_localization_error.pdf')},
    )
    with (output / 'split_localization_error_source.json').open('w', encoding='utf-8', newline='\n') as file:
        json.dump(record, file, indent=2)
        file.write('\n')


if __name__ == '__main__':
    main()
