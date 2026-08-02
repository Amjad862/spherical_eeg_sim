"""Compact row-oriented arrays for variable-size per-sample ground truth."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class RaggedArray:
    """Values plus CSR-style row boundaries.

    ``row_splits`` has length ``n_rows + 1``, starts at zero, is
    non-decreasing, and ends at ``values.shape[0]``. Any trailing dimensions
    of ``values`` belong to each item rather than to the row structure.
    """

    values: np.ndarray
    row_splits: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        row_splits = np.asarray(self.row_splits, dtype=np.int64)
        if values.ndim < 1:
            raise ValueError("RaggedArray values must have at least one dimension")
        if row_splits.ndim != 1 or row_splits.size < 1:
            raise ValueError("RaggedArray row_splits must be a non-empty 1-D array")
        if row_splits[0] != 0 or np.any(np.diff(row_splits) < 0):
            raise ValueError(
                "RaggedArray row_splits must start at zero and be non-decreasing"
            )
        if int(row_splits[-1]) != int(values.shape[0]):
            raise ValueError(
                "RaggedArray row_splits[-1] must equal values.shape[0]"
            )
        object.__setattr__(self, 'values', values)
        object.__setattr__(self, 'row_splits', row_splits)

    @property
    def n_rows(self) -> int:
        """Number of logical rows."""
        return int(self.row_splits.size - 1)

    def row(self, index: int) -> np.ndarray:
        """Return one logical row as a view of ``values``."""
        start, stop = self.row_splits[index:index + 2]
        return self.values[start:stop]


def concatenate_ragged(items: Sequence[RaggedArray]) -> RaggedArray:
    """Concatenate compatible ragged arrays along their row axis."""
    if not items:
        return RaggedArray(
            np.empty((0,), dtype=np.float64), np.array([0], dtype=np.int64)
        )
    trailing_shape = items[0].values.shape[1:]
    dtype = items[0].values.dtype
    for item in items:
        if not isinstance(item, RaggedArray):
            raise TypeError("All items must be RaggedArray instances")
        if item.values.shape[1:] != trailing_shape:
            raise ValueError("RaggedArray trailing shapes do not match")
        if item.values.dtype != dtype:
            raise ValueError("RaggedArray dtypes do not match")

    values = np.concatenate([item.values for item in items], axis=0)
    row_splits = [0]
    value_offset = 0
    for item in items:
        row_splits.extend((item.row_splits[1:] + value_offset).tolist())
        value_offset += item.values.shape[0]
    return RaggedArray(values, np.asarray(row_splits, dtype=np.int64))
