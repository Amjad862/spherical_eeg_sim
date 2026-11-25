"""Generate one benchmark case and record nested stage wall times."""

import json
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spherical_eeg_sim import Config, pipeline
from spherical_eeg_sim.io.writer import ChunkedHDF5Writer


def main():
    times = {}

    def timed(name, function):
        def invoke(*args, **kwargs):
            start = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                times[name] = times.get(name, 0.0) + time.perf_counter() - start
        return invoke

    stages = (
        (pipeline, 'forward_potential_batch', 'basis_forward'),
        (pipeline, 'compute_background_leadfield_pool', 'background_pool'),
        (pipeline, '_run_one_split', 'split_generation_and_writing'),
        (ChunkedHDF5Writer, 'append', 'hdf5_parent_append'),
        (ChunkedHDF5Writer, 'finalize', 'hdf5_finalize'),
    )
    with ExitStack() as patches:
        for owner, attribute, name in stages:
            patches.enter_context(patch.object(owner, attribute, timed(name, getattr(owner, attribute))))
        pipeline.run_pipeline(Config.from_json(sys.argv[1]))
    Path(sys.argv[2]).write_text(json.dumps(times, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
