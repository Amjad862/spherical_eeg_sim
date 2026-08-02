"""
Quickstart example: a reduced dataset for a smoke test.

Use this to verify an installation or sanity-check parameter changes with a
reduced workload. It exercises the full pipeline but is not presented as a
training benchmark.

Run from the repository root:
    python examples/quickstart_small.py

Generates 800 train / 100 val / 100 test; output is a few MB. This is the
smoke test: every other example generates a real dataset.
"""

from spherical_eeg_sim import quickstart_config, run_pipeline


def main():
    # The shared preset uses 800/100/100 samples, a 500/100 position pool, and
    # the supported lower Legendre order. These are operational smoke-test
    # settings, not scientific sample-size recommendations.
    cfg = quickstart_config()

    # Use the bundled 69-channel montage
    cfg.electrode_file = 'standard_1010_69.csv'
    cfg.output_dir = 'output_quickstart'
    summary = run_pipeline(cfg)
    print()
    print('Output:', summary['output_dir'])


if __name__ == '__main__':
    main()
