"""
I/O: dataset writers (HDF5/NPZ/MAT), lead-field operator writer, run manifest,
README writer. MNE-export helpers live in ``spherical_eeg_sim.io.mne_export``
(imported lazily; MNE is an optional dependency).
"""

from .writer import (
    save_split,
    open_split_writer,
    ChunkedHDF5Writer,
    write_leadfield_operator,
    write_source_pool,
    write_checksum_manifest,
    write_run_manifest,
    write_run_readme,
)
from .graph_export import build_electrode_graph, write_electrode_graph

__all__ = [
    'save_split',
    'open_split_writer',
    'ChunkedHDF5Writer',
    'write_leadfield_operator',
    'write_source_pool',
    'write_checksum_manifest',
    'write_run_manifest',
    'write_run_readme',
    'build_electrode_graph',
    'write_electrode_graph',
]
