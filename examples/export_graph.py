"""Export the electrode adjacency graph for graph-neural-network workflows.

A GNN inverse model needs a graph over the electrodes. Setting
``cfg.output.save_graph = True`` writes ``graph.h5``, ``graph.npz``, or a
MATLAB v7.3 ``graph.mat`` alongside the dataset: a bidirectional
``edge_index`` (2, E),
Gaussian-of-distance ``edge_weight`` (E,), raw ``edge_length_m`` (E,), node
``degree``, and the self-describing electrode block, with node order matching
the ``/electrodes`` group in every split file.

Three construction methods are available via ``cfg.output.graph_method``:
'knn' (default), 'delaunay' (triangulation on the scalp sphere), and 'distance'
(fixed-radius). This script writes a kNN graph and shows how to load it into a
PyTorch Geometric ``Data`` object (falling back to a raw-array print if PyG is
not installed).

Run from the repository root:
    python examples/export_graph.py

Generates 2,000 train; output is well under a GB.
"""

from spherical_eeg_sim import Config, run_pipeline


def main():
    cfg = Config()
    cfg.source_mode = 'point'
    cfg.dataset.n_train = 2000
    cfg.dataset.n_val = 400
    cfg.dataset.n_test = 400
    cfg.electrode_file = 'standard_1010_69.csv'
    cfg.output_dir = 'output_graph_example'
    cfg.output.save_graph = True
    cfg.output.graph_method = 'knn'     # or 'delaunay' / 'distance'
    cfg.output.graph_k = 6
    run_pipeline(cfg)

    import h5py
    with h5py.File('output_graph_example/graph.h5', 'r') as f:
        edge_index = f['edge_index'][:]           # (2, E)
        edge_weight = f['edge_weight'][:]         # (E,)
        node_xyz = f['electrodes/cartesian_xyz'][:]
    print(f"graph: {node_xyz.shape[0]} nodes, {edge_index.shape[1]} directed edges")

    # Optional: wrap as a PyTorch Geometric graph (per-sample EEG becomes x).
    try:
        import torch
        from torch_geometric.data import Data
        edge_index_t = torch.as_tensor(edge_index, dtype=torch.long)
        edge_weight_t = torch.as_tensor(edge_weight, dtype=torch.float32)
        pos_t = torch.as_tensor(node_xyz, dtype=torch.float32)
        # For one EEG window of shape (n_channels, n_timepoints), node features
        # x would be that window (channels as nodes). Here we just show the wiring.
        data = Data(edge_index=edge_index_t, edge_attr=edge_weight_t, pos=pos_t)
        print("PyG Data:", data)
    except Exception:
        print("(install torch + torch_geometric to build a PyG Data object; "
              "the raw arrays above are all a GNN needs)")


if __name__ == '__main__':
    main()
