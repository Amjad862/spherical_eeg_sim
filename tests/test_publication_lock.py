"""Independent runs may share a parent directory, but never a target."""
import logging
from pathlib import Path
import subprocess
import sys
from threading import Event, Thread

import pytest

from spherical_eeg_sim import pipeline
from spherical_eeg_sim._output_lock import lock_output
from spherical_eeg_sim.config import quickstart_config


def test_second_run_cannot_recover_active_staging(tmp_path, monkeypatch):
    cfg = quickstart_config()
    target = tmp_path / "dataset"
    cfg.output_dir = str(target)
    active, done = Event(), Event()
    writer_errors = []

    def write_stage(config, directory, *args, **kwargs):
        partial = directory / "partial"
        partial.write_bytes(b"still writing")
        active.set()
        assert done.wait(20)
        assert partial.read_bytes() == b"still writing"
        raise InterruptedError("simulated generation failure")

    def first_run():
        try:
            pipeline.run_pipeline(cfg, logging.CRITICAL)
        except BaseException as error:
            writer_errors.append(error)

    monkeypatch.setattr(pipeline, "_run_pipeline_impl", write_stage)
    worker = Thread(target=first_run)
    worker.start()
    try:
        assert active.wait(20)
        with pytest.raises(BlockingIOError, match="generation is active"):
            pipeline.run_pipeline(cfg, logging.CRITICAL)
    finally:
        done.set()
        worker.join(20)
    assert not worker.is_alive()
    assert len(writer_errors) == 1
    assert isinstance(writer_errors[0], InterruptedError)
    with lock_output(target):
        pass


def test_process_lock_is_per_target_and_released_after_termination(tmp_path):
    target = tmp_path / "dataset"
    # Acquiring in this independent process exercises the kernel lock, even
    # when a test runner imports the library in the parent.
    script = """
from pathlib import Path
import sys
from spherical_eeg_sim._output_lock import lock_output
with lock_output(Path(sys.argv[1])):
    print('locked', flush=True)
    sys.stdin.readline()
"""
    with subprocess.Popen(
        [sys.executable, "-c", script, str(target)],
        cwd=Path(__file__).resolve().parents[1],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
    ) as child:
        try:
            assert child.stdout.readline().strip() == "locked"
            with pytest.raises(BlockingIOError, match="generation is active"):
                with lock_output(target):
                    pass
            with lock_output(tmp_path / "another"):
                pass
        finally:
            child.kill()
            child.wait(timeout=20)
    with lock_output(target):
        pass
