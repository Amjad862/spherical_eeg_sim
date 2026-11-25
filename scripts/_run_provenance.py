"""Shared preflight and source records of the experiment drivers."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

from threadpoolctl import threadpool_info

from spherical_eeg_sim.provenance import environment_snapshot


def hash_file(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def check_new_output(path, root):
    output = path.resolve()
    canonical = root / 'validation_results'
    if output == canonical or canonical in output.parents:
        raise ValueError('Choose a new run folder outside validation_results')
    if output.exists():
        raise ValueError(f'Output already exists. Choose a new run folder: {output}')
    return output


def collect_provenance(root, sources, arguments, *, require_clean):
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()

    try:
        commit = git('rev-parse', 'HEAD')
        dirty = bool(git('status', '--porcelain', '--untracked-files=all'))
        package_tree = git('rev-parse', f'{commit}:spherical_eeg_sim')
    except (OSError, subprocess.CalledProcessError):
        if require_clean:
            raise ValueError('Recorded runs record the commit and require a clean Git checkout of the '
                             'repository; a source archive without Git metadata is not enough.')
        commit, package_tree, dirty = None, None, None
    if require_clean and dirty:
        raise ValueError('Recorded runs require a clean checkout. Commit the intended inputs before running.')

    package_files = [path for path in (root / 'spherical_eeg_sim').rglob('*')
                     if path.is_file() and '__pycache__' not in path.parts]
    files = sorted(set([*package_files, *sources, Path(__file__)]))
    return dict(
        commit=commit, package_tree=package_tree, working_tree_dirty=dirty,
        arguments=list(arguments), source_sha256={path.relative_to(root).as_posix(): hash_file(path) for path in files},
        environment=environment_snapshot(include_paths=True), threads=threadpool_info(),
        thread_variables={key: os.getenv(key) for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
        resolved_packages=subprocess.check_output([sys.executable, '-m', 'pip', 'freeze', '--all'], text=True),
    )
