"""
Patch-mode dataset: distributed cortical sources with finite support.

Each sample is a cortical patch (10-60 mm diameter) with a Gaussian profile
truncated at the patch radius. Useful for source-imaging methods that target
spatially extended generators. Membership uses angular separation multiplied
by the selected center radius; active pool points retain their configured
radial depths. The exact global source-pool indices and signed radial moments
are stored without padding in CSR row encoding.

Patches are cortex-only by design.

Generates 50,000 train; output is a few GB.
"""

from spherical_eeg_sim import Config, run_pipeline


def main():
    cfg = Config()
    cfg.source_mode = 'patch'
    cfg.sources.n_subcortical_positions = 0
    cfg.patch.diameter_range_mm = [10.0, 60.0]
    cfg.patch.falloff_profile = 'gaussian'

    cfg.dataset.n_train = 50_000
    cfg.dataset.n_val = 5_000
    cfg.dataset.n_test = 5_000

    cfg.electrode_file = 'standard_1010_69.csv'
    cfg.output_dir = 'output_patch_mode'

    summary = run_pipeline(cfg)
    print()
    print('Output:', summary['output_dir'])


if __name__ == '__main__':
    main()
