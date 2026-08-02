#!/usr/bin/env python
"""Regenerate ``validation_results/SHA256SUMS``.

The manifest binds the executed results, the parameter documentation, and the
scripts that produced them, so a reader can verify the numbers, the code that
generated them, and the prose that describes them as one unit. This script
writes that binding and is itself covered by the manifest it produces.

Paths are recorded relative to the repository root, with POSIX separators and a
sorted order, so the manifest is byte-identical across platforms and reruns.
Verification is the standard ``sha256sum -c`` format::

    cd <repo root> && sha256sum -c validation_results/SHA256SUMS

Usage::

    python scripts/make_validation_manifest.py [--check]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Iterator, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = REPO_ROOT / "validation_results" / "SHA256SUMS"

# Directories whose entire contents are bound, and the extensions that count.
# Everything that determines a reported number, or describes one, belongs here.
BOUND_TREES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("validation_results", (".json", ".csv", ".png", ".pdf", ".txt")),
    ("scripts", (".py",)),
    ("docs", (".md", ".bib")),
    (
        "spherical_eeg_sim",
        (".py", ".csv", ".txt", ".sfp", ".elc", ".csd", ".typed"),
    ),
)

EXCLUDED_NAMES = frozenset({"SHA256SUMS"})
BOUND_FILES = (
    "pyproject.toml",
    "validation-requirements.txt",
    "README.md",
    "CITATION.cff",
    "MANIFEST.in",
)


def _iter_bound_files() -> Iterator[Path]:
    for directory, extensions in BOUND_TREES:
        root = REPO_ROOT / directory
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if path.name in EXCLUDED_NAMES:
                continue
            if "__pycache__" in path.parts:
                continue
            if path.suffix.lower() not in extensions and path.name != "NOTICE":
                continue
            yield path
    for name in BOUND_FILES:
        path = REPO_ROOT / name
        if path.is_file():
            yield path


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _entries() -> List[Tuple[str, str]]:
    rows = [
        (_digest(path), path.relative_to(REPO_ROOT).as_posix())
        for path in _iter_bound_files()
    ]
    # Sort on the recorded name so the manifest does not depend on filesystem
    # traversal order.
    return sorted(rows, key=lambda row: row[1])


def _render(rows: List[Tuple[str, str]]) -> str:
    return "".join(f"{digest}  {name}\n" for digest, name in rows)


def check() -> int:
    if not MANIFEST.is_file():
        print(f"missing manifest: {MANIFEST}", file=sys.stderr)
        return 1
    recorded = {}
    for line in MANIFEST.read_text().splitlines():
        if not line.strip():
            continue
        digest, _, name = line.partition("  ")
        recorded[name.strip()] = digest.strip()

    actual = {name: digest for digest, name in _entries()}
    missing = sorted(set(recorded) - set(actual))
    unlisted = sorted(set(actual) - set(recorded))
    changed = sorted(
        name for name in set(recorded) & set(actual)
        if recorded[name] != actual[name]
    )

    for name in missing:
        print(f"MISSING   {name}")
    for name in unlisted:
        print(f"UNLISTED  {name}")
    for name in changed:
        print(f"CHANGED   {name}")

    if missing or unlisted or changed:
        print(
            f"\n{len(recorded)} listed, {len(missing)} missing, "
            f"{len(unlisted)} unlisted, {len(changed)} changed"
        )
        return 1
    print(f"OK: {len(recorded)} files match the manifest")
    return 0


def write() -> int:
    rows = _entries()
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    # Written through an explicit LF newline so the manifest is byte-identical
    # whatever platform produced it, and so external checkers such as
    # sha256sum(1) do not read a trailing carriage return as part of the path.
    with MANIFEST.open('w', encoding='utf-8', newline='\n') as handle:
        handle.write(_render(rows))
    by_dir: dict[str, int] = {}
    for _, name in rows:
        by_dir[name.split("/", 1)[0]] = by_dir.get(name.split("/", 1)[0], 0) + 1
    print(f"wrote {MANIFEST.relative_to(REPO_ROOT)} with {len(rows)} entries")
    for directory in sorted(by_dir):
        print(f"  {directory}: {by_dir[directory]}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the manifest instead of rewriting it (non-zero exit on drift)",
    )
    args = parser.parse_args()
    return check() if args.check else write()


if __name__ == "__main__":
    raise SystemExit(main())
