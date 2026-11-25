"""Output routing shared by the validation command-line scripts."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4


REPO_ROOT = Path(__file__).resolve().parent.parent
ARCHIVE = REPO_ROOT / "validation_results"
RUNS = REPO_ROOT / "validation_runs"


def add_output_options(parser):
    parser.add_argument(
        "--write-canonical", action="store_true",
        help="regenerate the archived files using the publication settings",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="replace existing files at explicitly selected noncanonical paths",
    )


def new_run_directory(name):
    return RUNS / f"{name}-{uuid4().hex[:12]}"


def relative_reference(path, summary):
    """Refer to a companion file from its final summary location."""
    try:
        return Path(os.path.relpath(path, summary.parent)).as_posix()
    except ValueError:  # Windows paths on different drives.
        return path.as_posix()


def _check_destination(path, *, overwrite, canonical=False, directory=False):
    original = Path(os.path.abspath(Path(path).expanduser()))
    resolved = original.resolve()
    if not canonical and (
        original.is_relative_to(ARCHIVE)
        or resolved.is_relative_to(ARCHIVE.resolve())
    ):
        raise ValueError("validation_results/ is protected. Use --write-canonical.")
    if directory and (
        ARCHIVE.is_relative_to(original) or ARCHIVE.resolve().is_relative_to(resolved)
    ):
        raise ValueError(f"Dataset directory contains the validation archive: {resolved}")
    if original.is_symlink():
        raise ValueError(f"Output destination is a symbolic link: {original}")
    if resolved.exists():
        if not overwrite:
            raise ValueError(f"Output already exists: {resolved}. Use --overwrite.")
        if directory != resolved.is_dir() or (not directory and not resolved.is_file()):
            raise ValueError(f"Output destination has the wrong type: {resolved}")
    for parent in resolved.parents:
        if parent.exists():
            if not parent.is_dir():
                raise ValueError(f"Output parent is not a directory: {parent}")
            break
    return resolved


def check_work_directory(parser, path, *, overwrite=False, result_paths=()):
    """Preflight a generated dataset path before an experiment begins."""
    try:
        directory = _check_destination(path, overwrite=overwrite, directory=True)
        if any(
            path.is_relative_to(directory) or directory.is_relative_to(path)
            for path in result_paths
        ):
            raise ValueError(f"Dataset directory overlaps a result path: {directory}")
        return directory
    except ValueError as error:
        parser.error(str(error))


class ValidationOutputs:
    """Preflight destinations and stage all files before publishing the run."""

    def __init__(
        self, parser, args, *, name, canonical, settings=None, explicit=None,
        ordinary=None,
    ):
        explicit = explicit or {}
        self.run_directory = None
        self.overwrite = args.overwrite or args.write_canonical
        if args.write_canonical:
            if explicit:
                parser.error("--write-canonical selects the archived output paths.")
            changed = [
                field for field, expected in (settings or {}).items()
                if getattr(args, field) != expected
            ]
            if changed:
                parser.error(
                    "--write-canonical requires the publication settings. "
                    "Conflicting options: "
                    + ", ".join("--" + field.replace("_", "-") for field in changed)
                )
            paths = {key: ARCHIVE / filename for key, filename in canonical.items()}
        elif explicit:
            paths = explicit
        else:
            self.run_directory = new_run_directory(name)
            paths = {
                key: self.run_directory / filename
                for key, filename in (ordinary or canonical).items()
            }

        try:
            self.paths = {
                key: _check_destination(
                    path, overwrite=self.overwrite, canonical=args.write_canonical,
                )
                for key, path in paths.items()
            }
            if len(set(self.paths.values())) != len(self.paths):
                raise ValueError("Each result requires a distinct output path.")
            if any(
                parent in self.paths.values()
                for path in self.paths.values() for parent in path.parents
            ):
                raise ValueError("One result path cannot be another result's directory.")
        except ValueError as error:
            parser.error(str(error))

    @contextmanager
    def stage(self):
        """Yield writable paths and publish them only after successful generation."""
        if self.run_directory is not None:
            run_directory = self.run_directory
            run_directory.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(
                prefix=".validation-stage-", dir=run_directory.parent,
            ) as temporary:
                stage = Path(temporary)
                paths = {
                    key: stage / target.name for key, target in self.paths.items()
                }
                yield paths
                self._require_files(paths)
                if run_directory.exists() or run_directory.is_symlink():
                    raise FileExistsError(run_directory)
                stage.rename(run_directory)
            return

        stages = {}
        keep_backups = False
        try:
            paths = {}
            for key, target in self.paths.items():
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.parent not in stages:
                    stages[target.parent] = Path(tempfile.mkdtemp(
                        prefix=".validation-stage-", dir=target.parent,
                    ))
                paths[key] = stages[target.parent] / target.name
            yield paths
            self._require_files(paths)
            backups = {}
            published = []
            try:
                for key, target in self.paths.items():
                    if self.overwrite:
                        if target.exists():
                            # Keep the previous file in the staging directory so
                            # a later publication failure can restore the bundle.
                            backup = paths[key].parent / f".previous-{uuid4().hex}"
                            shutil.copy2(target, backup)
                            backups[key] = backup
                        paths[key].replace(target)
                    elif os.name == "nt":
                        # Windows rename fails if a destination already exists.
                        paths[key].rename(target)
                    else:
                        # A hard link gives POSIX the same no-clobber behavior.
                        os.link(paths[key], target)
                    published.append(key)
            except BaseException:
                for key in reversed(published):
                    target = self.paths[key]
                    try:
                        if key in backups:
                            backups[key].replace(target)
                        else:
                            target.unlink()
                    except OSError as error:
                        keep_backups = True
                        print(
                            f"Could not roll back {target}: {error}. "
                            f"Staged files kept in {paths[key].parent}.",
                            file=sys.stderr,
                        )
                raise
        finally:
            if not keep_backups:
                for stage in stages.values():
                    shutil.rmtree(stage, ignore_errors=True)

    @staticmethod
    def _require_files(paths):
        for path in paths.values():
            if not path.is_file():
                raise RuntimeError(f"Validation did not produce {path.name}")
