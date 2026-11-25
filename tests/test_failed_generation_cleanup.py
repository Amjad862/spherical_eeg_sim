"""Preserve the first error and recover owned staging after failed writes."""

from pathlib import Path

import pytest

from spherical_eeg_sim import pipeline, run_pipeline
from spherical_eeg_sim.io.writer import ChunkedHDF5Writer
from test_release_arrays import tiny_config


@pytest.mark.parametrize('phase', ['generate', 'append', 'finalize'])
def test_failed_patch_run_closes_hdf5_and_ragged_spools(tmp_path, monkeypatch, phase):
    cfg = tiny_config(tmp_path / 'dataset', 'patch', 'hdf5')
    cfg.output.chunk_samples = 1
    opened = []
    real_open = pipeline.open_split_writer
    real_generate = pipeline._generate_one_sample
    real_append = ChunkedHDF5Writer.append

    def capture_writer(*args, **kwargs):
        writer = real_open(*args, **kwargs)
        opened.append(writer)
        return writer

    def fail_generate(*args, **kwargs):
        if args[3] == 1:
            raise RuntimeError('injected generate failure')
        return real_generate(*args, **kwargs)

    def fail_append(self, *args, **kwargs):
        if self._written:
            raise RuntimeError('injected append failure')
        return real_append(self, *args, **kwargs)

    def fail_finalize(self, *args, **kwargs):
        raise RuntimeError('injected finalize failure')

    monkeypatch.setattr(pipeline, 'open_split_writer', capture_writer)
    if phase == 'generate':
        monkeypatch.setattr(pipeline, '_generate_one_sample', fail_generate)
    elif phase == 'append':
        monkeypatch.setattr(ChunkedHDF5Writer, 'append', fail_append)
    else:
        monkeypatch.setattr(ChunkedHDF5Writer, 'finalize', fail_finalize)
    with pytest.raises(RuntimeError, match=f'injected {phase} failure'):
        run_pipeline(cfg)
    assert len(opened) == 1
    assert not opened[0]._f.id.valid
    spools = [entry['spool'] for entry in opened[0]._gt_dsets.values() if isinstance(entry, dict)]
    assert spools and all(spool.closed for spool in spools)
    assert not Path(cfg.output_dir).exists()
    assert not list(tmp_path.glob('.dataset.staging-*'))


def test_cleanup_failure_preserves_error_and_staging_ownership(tmp_path, monkeypatch):
    target = tmp_path / 'dataset'
    cfg = tiny_config(target, 'point', 'hdf5')
    original_unlink = Path.unlink

    def fail_generate(*args, **kwargs):
        staging = next(tmp_path.glob('.dataset.staging-*'))
        (staging / 'locked.txt').write_text('injected', encoding='utf-8')
        raise RuntimeError('injected original failure')

    def fail_unlink(self, *args, **kwargs):
        if self.name == 'locked.txt':
            raise PermissionError('injected sharing violation')
        return original_unlink(self, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, 'unlink', fail_unlink)
        patch.setattr(pipeline.time, 'sleep', lambda seconds: None)
        patch.setattr(pipeline, '_generate_one_sample', fail_generate)
        with pytest.raises(RuntimeError, match='injected original failure') as failure:
            run_pipeline(cfg)
        if hasattr(failure.value, '__notes__'):
            assert 'injected sharing violation' in failure.value.__notes__[0]
        staging = next(tmp_path.glob('.dataset.staging-*'))
        assert (staging / '.spherical_eeg_sim.transaction.json').is_file()
        assert not target.exists()
    run_pipeline(cfg)
    assert target.exists()
    assert not staging.exists()


def test_published_replacement_survives_partial_backup_cleanup(tmp_path, monkeypatch):
    target = tmp_path / 'dataset'
    cfg = tiny_config(target, 'point', 'hdf5')
    run_pipeline(cfg)
    cfg.output.overwrite = True
    original_unlink = Path.unlink

    def locked_backup(self, *args, **kwargs):
        if self.name == 'train.h5' and '.previous-' in self.parent.name:
            raise PermissionError('injected backup sharing violation')
        return original_unlink(self, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, 'unlink', locked_backup)
        patch.setattr(pipeline.time, 'sleep', lambda seconds: None)
        run_pipeline(cfg)
        backup = next(tmp_path.glob('.dataset.previous-*'))
        assert pipeline._directory_checksum_complete(target)
        assert pipeline._is_owned_cleanup_backup(backup, target)
        assert not pipeline._is_owned_rollback(backup, target)
        pipeline._recover_interrupted_publish(target, cfg)
        assert (backup / pipeline._ROLLBACK_MARKER).is_file()
    pipeline._recover_interrupted_publish(target, cfg)
    assert not backup.exists()
    assert pipeline._directory_checksum_complete(target)
