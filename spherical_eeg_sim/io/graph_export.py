"""Electrode-graph construction and export.

Graph neural networks for EEG source localization operate on a graph whose
nodes are the recording electrodes and whose edges encode spatial adjacency on
the scalp. The forward operator (``save_leadfield``) gives such models the
physics; this module gives them the *graph*: an electrode adjacency built
directly from the montage that travels with the dataset, so a GNN consumer does
not have to reconstruct it from the embedded coordinates.

The graph depends only on the electrode montage (not on any sample), so it is
built once per run. Three construction methods are provided:

  'knn'      -- symmetric k-nearest-neighbour graph (each electrode connected to
                its ``k`` nearest neighbours by Euclidean distance; the union is
                taken so the adjacency is symmetric). This is the declared
                default sparse graph.
  'delaunay' -- the Delaunay triangulation of the electrodes on the scalp
                sphere, obtained as the convex hull of the 3-D electrode points
                (for points on a sphere the convex-hull faces are exactly the
                spherical Delaunay triangulation). Yields a planar, parameter-
                free neighbourhood graph.
  'distance' -- every electrode pair within ``distance_threshold_m`` metres is
                connected (a fixed-radius graph).

Edge weights are a Gaussian function of edge length,
``w_ij = exp(-d_ij^2 / (2 * sigma^2))`` with ``sigma`` the mean kept-edge
length, so nearer electrodes are coupled more strongly and the weights are
scale-adaptive to the montage. This is an engineering edge encoding, not a
physiological coupling model. Both the weight and the raw edge length (metres)
are stored. Graph-based ESI precedent is described by Jiao et al. (2022).

Outputs mirror the lead-field export: an ``edge_index`` of shape ``(2, E)`` in
the bidirectional form PyTorch Geometric expects (both ``(i, j)`` and
``(j, i)`` present), a matching ``edge_weight`` and ``edge_length_m`` of shape
``(E,)``, the node coordinates, and a self-describing electrode block.
"""

from __future__ import annotations
import json
import numbers
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from .._compat import import_hdf5storage


def build_electrode_graph(
    cartesian_xyz: np.ndarray,
    method: str = 'knn',
    k: int = 6,
    distance_threshold_m: Optional[float] = None,
) -> Dict[str, Any]:
    """Build an electrode adjacency graph from 3-D electrode coordinates.

    Parameters
    ----------
    cartesian_xyz : ndarray (N, 3)
        Electrode Cartesian coordinates in metres.
    method : {'knn', 'delaunay', 'distance'}
    k : int
        Number of nearest neighbours (``method='knn'``).
    distance_threshold_m : float or None
        Connection radius in metres (``method='distance'``).

    Returns
    -------
    dict with keys:
        'edge_index'    : int64 ndarray (2, E), bidirectional (i->j and j->i).
        'edge_weight'   : float64 ndarray (E,), Gaussian-of-distance weights.
        'edge_length_m' : float64 ndarray (E,), Euclidean edge lengths (metres).
        'degree'        : int64 ndarray (N,), node degrees (out = in, symmetric).
        'n_nodes'       : int64 scalar ndarray.
    """
    xyz = np.asarray(cartesian_xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"cartesian_xyz must be (N, 3); got {xyz.shape}")
    if not np.all(np.isfinite(xyz)):
        raise ValueError("cartesian_xyz must contain only finite coordinates")
    n = xyz.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 electrodes to build a graph")
    if np.unique(xyz, axis=0).shape[0] != n:
        raise ValueError("cartesian_xyz contains duplicate electrode coordinates")

    # Collect an undirected edge set as a set of ordered (i < j) pairs so each
    # physical edge is represented once before symmetrisation.
    undirected: set[Tuple[int, int]] = set()

    if method == 'knn':
        if (isinstance(k, bool) or not isinstance(k, numbers.Integral)
                or k < 1 or k >= n):
            raise ValueError(
                "graph_k must be a positive integer smaller than the number "
                "of electrodes for method='knn'"
            )
        k_eff = int(k)
        # Pairwise Euclidean distances; self-distance set to +inf so a node is
        # never its own neighbour.
        d2 = np.sum((xyz[:, None, :] - xyz[None, :, :]) ** 2, axis=2)
        np.fill_diagonal(d2, np.inf)
        # k nearest neighbours per node.
        nn = np.argpartition(d2, kth=k_eff - 1, axis=1)[:, :k_eff]
        for i in range(n):
            for j in nn[i]:
                a, b = (i, int(j)) if i < j else (int(j), i)
                if a != b:
                    undirected.add((a, b))

    elif method == 'delaunay':
        from scipy.spatial import ConvexHull
        if n < 4:
            raise ValueError("Delaunay graph construction requires at least 4 electrodes")
        # For points on a sphere the convex-hull triangulation coincides with
        # the spherical Delaunay triangulation. Each hull simplex is a triangle;
        # its three edges become graph edges.
        hull = ConvexHull(xyz)
        for simplex in hull.simplices:
            s = [int(v) for v in simplex]
            for a_idx in range(len(s)):
                for b_idx in range(a_idx + 1, len(s)):
                    a, b = s[a_idx], s[b_idx]
                    if a != b:
                        undirected.add((a, b) if a < b else (b, a))

    elif method == 'distance':
        if (distance_threshold_m is None
                or isinstance(distance_threshold_m, bool)
                or not isinstance(distance_threshold_m, numbers.Real)
                or not np.isfinite(distance_threshold_m)
                or distance_threshold_m <= 0):
            raise ValueError(
                "graph_distance_threshold_m must be a positive distance in "
                "metres for method='distance'")
        assert distance_threshold_m is not None
        d2 = np.sum((xyz[:, None, :] - xyz[None, :, :]) ** 2, axis=2)
        thr2 = float(distance_threshold_m) ** 2
        iu, ju = np.where(np.triu(d2 <= thr2, k=1))
        for a, b in zip(iu.tolist(), ju.tolist()):
            undirected.add((a, b))

    else:
        raise ValueError(
            f"Unknown graph method {method!r}; expected 'knn', 'delaunay', "
            f"or 'distance'")

    if not undirected:
        raise ValueError(
            f"Graph construction produced no edges (method={method!r}). "
            f"Check k / distance_threshold_m against the montage.")

    edges = np.array(sorted(undirected), dtype=np.int64)          # (E0, 2), i<j
    lengths = np.linalg.norm(xyz[edges[:, 0]] - xyz[edges[:, 1]], axis=1)
    if not np.all(np.isfinite(lengths)) or np.any(lengths <= 0.0):
        raise ValueError(
            "Graph edges must have finite positive length; check for duplicate "
            "electrode coordinates"
        )

    # Scale-adaptive Gaussian edge weights (sigma = mean kept-edge length).
    sigma = float(np.mean(lengths))
    weights = np.exp(-(lengths ** 2) / (2.0 * sigma ** 2))

    # Symmetrise into the bidirectional edge_index PyG expects.
    src = np.concatenate([edges[:, 0], edges[:, 1]])
    dst = np.concatenate([edges[:, 1], edges[:, 0]])
    edge_index = np.stack([src, dst], axis=0).astype(np.int64)     # (2, 2*E0)
    edge_weight = np.concatenate([weights, weights]).astype(np.float64)
    edge_length_m = np.concatenate([lengths, lengths]).astype(np.float64)

    degree = np.bincount(src, minlength=n).astype(np.int64)

    return {
        'edge_index': edge_index,
        'edge_weight': edge_weight,
        'edge_length_m': edge_length_m,
        'degree': degree,
        'n_nodes': np.int64(n),
    }


def write_electrode_graph(
    output_dir: Union[str, Path],
    electrodes: Dict[str, Any],
    method: str = 'knn',
    k: int = 6,
    distance_threshold_m: Optional[float] = None,
    fmt: str = 'hdf5',
    compress: bool = True,
) -> Path:
    """Build and write the electrode-adjacency graph as an auxiliary file.

    Parameters
    ----------
    output_dir : str or Path
    electrodes : dict
        Self-describing electrode block (names, cartesian_xyz,
        spherical_r_theta_phi, reference_mode, reference_info) -- the same block
        embedded in each dataset split.
    method, k, distance_threshold_m
        Passed to :func:`build_electrode_graph`.
    fmt : 'hdf5' | 'npz' | 'mat'
        Writes ``graph.h5``, ``graph.npz``, or a MATLAB v7.3 ``graph.mat``.
    compress : bool

    Returns
    -------
    Path to the written file.
    """
    from .writer import (
        _atomic_replace,
        _canonicalize_mat_userblock,
        _deterministic_savez,
        _write_electrodes_group,
        _write_metadata_to_h5,
        _jsonify,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cartesian = np.asarray(electrodes.get('cartesian_xyz'), dtype=np.float64)
    graph = build_electrode_graph(
        cartesian, method=method, k=k,
        distance_threshold_m=distance_threshold_m,
    )

    n_edges = int(graph['edge_index'].shape[1])
    meta = {
        'description': (
            'Electrode adjacency graph for GNN workflows. edge_index is '
            '(2, E) in bidirectional (i->j and j->i) form; edge_weight[e] is a '
            'Gaussian function of edge length (exp(-d^2 / (2 sigma^2)), sigma = '
            'mean edge length) and edge_length_m[e] is the Euclidean edge '
            'length in metres. Node order matches /electrodes.'
        ),
        'method': method,
        'k': int(k) if method == 'knn' else None,
        'distance_threshold_m': (
            float(distance_threshold_m)
            if method == 'distance' and distance_threshold_m is not None
            else None
        ),
        'n_nodes': int(graph['n_nodes']),
        'n_edges_directed': n_edges,
        'n_edges_undirected': n_edges // 2,
        'edge_index_convention': 'bidirectional (both i->j and j->i present)',
        'weight': 'exp(-d^2 / (2 sigma^2)), sigma = mean edge length (metres)',
    }

    if fmt == 'npz':
        path = output_dir / 'graph.npz'
        save_dict = {
            'edge_index': graph['edge_index'],
            'edge_weight': graph['edge_weight'],
            'edge_length_m': graph['edge_length_m'],
            'degree': graph['degree'],
            'electrode_names': np.array(list(electrodes.get('names', []))),
            'electrode_cartesian_xyz': cartesian,
            'electrode_spherical_r_theta_phi': np.asarray(
                electrodes.get('spherical_r_theta_phi')),
            '_metadata_json': np.array(json.dumps(_jsonify(meta))),
        }
        temp = output_dir / '.graph.writing.npz'
        _deterministic_savez(temp, save_dict, compress)
        _atomic_replace(temp, path)
        return path

    if fmt == 'mat':
        try:
            hdf5storage = import_hdf5storage()
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "MATLAB v7.3 output requires hdf5storage; install "
                "spherical_eeg_sim[mat]."
            ) from exc
        path = output_dir / 'graph.mat'
        temp = output_dir / '.graph.writing.mat'
        payload = {
            **graph,
            'electrodes': {
                'names': np.asarray(list(electrodes.get('names', []))),
                'cartesian_xyz': cartesian,
                'spherical_r_theta_phi': np.asarray(
                    electrodes.get('spherical_r_theta_phi')
                ),
            },
            'metadata_json': json.dumps(_jsonify(meta), sort_keys=True),
        }
        hdf5storage.savemat(
            temp, payload, fmt='7.3', appendmat=False,
            store_python_metadata=False, matlab_compatible=True,
            compress=compress, truncate_existing=True,
        )
        _canonicalize_mat_userblock(temp)
        _atomic_replace(temp, path)
        return path

    # Native HDF5 container.
    import h5py
    path = output_dir / 'graph.h5'
    temp = output_dir / '.graph.writing.h5'
    comp = 'gzip' if compress else None
    with h5py.File(temp, 'w') as f:
        f.create_dataset(
            'edge_index', data=graph['edge_index'], compression=comp,
            track_times=False,
        )
        f.create_dataset(
            'edge_weight', data=graph['edge_weight'], compression=comp,
            track_times=False,
        )
        f.create_dataset(
            'edge_length_m', data=graph['edge_length_m'], compression=comp,
            track_times=False,
        )
        f.create_dataset('degree', data=graph['degree'], track_times=False)
        _write_electrodes_group(f, electrodes)
        mg = f.create_group('metadata')
        _write_metadata_to_h5(mg, meta)
    _atomic_replace(temp, path)
    return path
