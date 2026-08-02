"""Head-model ablation: swap the head model with one line using presets.

The spherical head model has two parameters that a reviewer will (rightly) ask
about: whether the CSF layer matters, and how sensitive results are to the
poorly-constrained skull conductivity. ``HeadModelConfig`` ships presets so both
ablations are turnkey -- assign the preset to ``cfg.head`` and regenerate:

  * ``HeadModelConfig.three_shell()`` -- a classic brain/skull/scalp model
    (Rush & Driscoll 1968), realized within the four-layer solver by collapsing
    the CSF conductivity onto the brain value (no CSF conductivity jump => a
    genuine three-shell model, not an approximation).
  * ``HeadModelConfig.with_skull_ratio(r)`` -- the default four-shell model with
    skull conductivity set to sigma_brain / r, for a brain:skull ratio sweep
    across the reported ~8-80 range (Goncalves et al. 2003; Oostendorp 2000).

This script generates one small dataset per head model with a shared seed, so the
sources and splits are identical and only the forward physics differs.

Run from the repository root:
    python examples/head_model_presets.py

Generates 3,000 train per preset; output is well under a GB.
"""

from spherical_eeg_sim import Config, run_pipeline
from spherical_eeg_sim.config import HeadModelConfig


def base_cfg():
    cfg = Config()
    cfg.source_mode = 'point'
    cfg.dataset.n_train = 3000
    cfg.dataset.n_val = 500
    cfg.dataset.n_test = 500
    cfg.electrode_file = 'standard_1010_69.csv'
    cfg.master_seed = 862      # shared => identical sources/splits across heads
    return cfg


def main():
    heads = {
        'four_shell_default': HeadModelConfig.four_shell_default(),
        'three_shell_no_csf': HeadModelConfig.three_shell(),
        'skull_ratio_15':     HeadModelConfig.with_skull_ratio(15.0),
        'skull_ratio_40':     HeadModelConfig.with_skull_ratio(40.0),
        'skull_ratio_80':     HeadModelConfig.with_skull_ratio(80.0),
    }
    for name, head in heads.items():
        cfg = base_cfg()
        cfg.head = head
        cfg.output_dir = f'output_head_{name}'
        run_pipeline(cfg)
        print(f"generated {name}: sigma = {head.conductivities_S_per_m}")

    print()
    print("Train/evaluate a model on each to quantify sensitivity to the CSF "
          "layer and to skull conductivity.")


if __name__ == '__main__':
    main()
