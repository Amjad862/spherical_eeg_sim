"""Stable software-environment metadata for datasets and validations."""

from __future__ import annotations

import platform
import sys
from importlib.metadata import PackageNotFoundError, version as distribution_version
from typing import Dict, Iterable


CORE_DISTRIBUTIONS = (
    "spherical_eeg_sim",
    "numpy",
    "scipy",
    "h5py",
    "hdf5storage",
    "matplotlib",
    "mne",
    "pandas",
    "scikit-learn",
)


def software_versions(
    extra_distributions: Iterable[str] = (),
) -> Dict[str, str]:
    """Return installed versions relevant to numerical artifacts."""
    versions: Dict[str, str] = {}
    names = dict.fromkeys((*CORE_DISTRIBUTIONS, *extra_distributions))
    for distribution in names:
        try:
            versions[distribution] = distribution_version(distribution)
        except PackageNotFoundError:
            if distribution == "spherical_eeg_sim":
                from . import __version__

                versions[distribution] = f"{__version__}+source"

    try:
        import h5py

        versions["libhdf5"] = h5py.version.hdf5_version
    except (ImportError, AttributeError):
        pass
    return versions


def environment_snapshot(
    extra_distributions: Iterable[str] = (),
) -> Dict[str, object]:
    """Return deterministic interpreter, platform, and package provenance."""
    return {
        "package": "spherical_eeg_sim",
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "byteorder": sys.byteorder,
        "software_versions": software_versions(extra_distributions),
    }
