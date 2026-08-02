"""Narrow compatibility helpers for optional third-party dependencies."""

from __future__ import annotations

import importlib
import warnings


def import_hdf5storage():
    """Import hdf5storage without exposing its NumPy 2.5 chararray warning.

    hdf5storage 0.2.2 references ``np.char.chararray`` while registering its
    own marshallers. NumPy 2.5 deprecates that class, although the code path
    used here neither creates nor returns a chararray. The filter is scoped to
    this single import and exact upstream message.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            message=(
                'The chararray class is deprecated and will be removed in a '
                'future release.*'
            ),
            category=DeprecationWarning,
        )
        return importlib.import_module('hdf5storage')
