#!/usr/bin/env python
"""Verify or explicitly regenerate ``validation_results/SHA256SUMS``.

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
    python scripts/make_validation_manifest.py --write-canonical
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from statistics import median
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
    "validation-windows-lock.txt",
    "validation-windows-requirements.txt",
    "CHANGELOG.md",
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


def _read_summary(name: str) -> dict:
    return json.loads((MANIFEST.parent / name).read_text(encoding="utf-8"))


def _check_companion(name: str, expected_digest: str) -> None:
    if _digest(MANIFEST.parent / name) != expected_digest:
        raise ValueError(f"{name} disagrees with its summary checksum")


def _verify_result_bundles() -> bool:
    """Check relationships a fresh manifest of individual files could hide."""
    try:
        ecd = _read_summary("ecd_69ch_35db_summary.json")
        expected = {
            "n_test": 300,
            "montage": "standard_1010_69.csv",
            "target_snr_db": 35.0,
            "reference_mode": "none",
            "seed": 20260703,
            "metrics_csv": "ecd_69ch_35db_per_sample.csv",
            "figure": "ecd_69ch_35db.png",
        }
        for key, value in expected.items():
            if ecd.get(key) != value:
                raise ValueError(f"archived ECD summary has unexpected {key}")
        sweep = [
            (entry["snr_db"], entry["n_test"]) for entry in ecd["snr_degradation_sweep"]
        ]
        if sweep != [(snr, 150) for snr in range(5, 36, 5)]:
            raise ValueError("archived ECD summary lacks the full publication SNR sweep")
        _check_companion(ecd["metrics_csv"], ecd["metrics_csv_sha256"])
        _check_companion(ecd["figure"], ecd["figure_sha256"])
        with (MANIFEST.parent / ecd["metrics_csv"]).open(newline="", encoding="utf-8") as handle:
            samples = list(csv.DictReader(handle))
        if [int(row["sample_index"]) for row in samples] != list(range(ecd["n_test"])):
            raise ValueError("archived ECD CSV does not contain 300 indexed samples")
        # The CSV stores position errors to four decimal places.
        reported = ecd["position_error_mm"]["median"]
        measured = median(float(row["err_mm"]) for row in samples)
        if not abs(reported - measured) <= 0.00005 + 1e-12:
            raise ValueError("archived ECD summary median disagrees with the CSV")

        baseline = _read_summary("baseline_benchmark_summary.json")
        if baseline["figure"] != "baseline_benchmark.png":
            raise ValueError("archived baseline summary names an unexpected figure")
        _check_companion(baseline["figure"], baseline["figure_sha256"])

        figures = _read_summary("publication_figures.json")["files"]
        if set(figures) != {
            "pipeline_schematic.pdf", "pipeline_schematic.png",
            "topographies.pdf", "topographies.png",
            "distributions.pdf", "distributions.png",
        }:
            raise ValueError("archived publication figure summary names an unexpected file set")
        for name, digest in figures.items():
            _check_companion(name, digest)

        # Figures 3, 5 and 6 of the paper: each record names its image files.
        for record in ("region_split_source.json", "split_localization_error_source.json",
                       "head_adaptation_finetuning_source.json"):
            for name, digest in _read_summary(record)["files"].items():
                _check_companion(name, digest)

        # The tables, the benchmark record and Figures 5 and 6 must come from
        # the same evidence: the table driver recorded the hash of every
        # evidence file it read.
        evidence = _read_summary("revision_tables.json")["provenance"]["evidence_sha256"]
        _check_companion("generation_benchmark.json",
                         evidence["generation-benchmark-v11/generation_benchmark.json"])
        for record, source in (
            ("split_localization_error_source.json", "split-study-v11/summary.json"),
            ("head_adaptation_finetuning_source.json", "head-adaptation-v11/summary.json"),
        ):
            if _read_summary(record)["summary_sha256"] != evidence[source]:
                raise ValueError(f"{record} was drawn from a different summary than the tables")
    except (KeyError, OSError, TypeError, ValueError, csv.Error) as error:
        print(f"Inconsistent archived validation results: {error}", file=sys.stderr)
        return False
    return True


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
    if not _verify_result_bundles():
        return 1
    print(f"OK: {len(recorded)} files match the manifest and the result bundles are consistent")
    return 0


def write() -> int:
    if not _verify_result_bundles():
        return 1
    rows = _entries()
    # Written through an explicit LF newline so the manifest is byte-identical
    # whatever platform produced it, and so external checkers such as
    # sha256sum(1) do not read a trailing carriage return as part of the path.
    with tempfile.TemporaryDirectory(prefix=".manifest-stage-", dir=MANIFEST.parent) as temporary:
        staged = Path(temporary) / MANIFEST.name
        with staged.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(_render(rows))
        staged.replace(MANIFEST)
    by_dir: dict[str, int] = {}
    for _, name in rows:
        by_dir[name.split("/", 1)[0]] = by_dir.get(name.split("/", 1)[0], 0) + 1
    print(f"wrote {MANIFEST.relative_to(REPO_ROOT)} with {len(rows)} entries")
    for directory in sorted(by_dir):
        print(f"  {directory}: {by_dir[directory]}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check",
        action="store_true",
        help="verify the manifest (the default; non-zero exit on drift)",
    )
    mode.add_argument(
        "--write-canonical",
        action="store_true",
        help="replace the manifest after checking consistency of the archived results",
    )
    args = parser.parse_args()
    return write() if args.write_canonical else check()


if __name__ == "__main__":
    raise SystemExit(main())
