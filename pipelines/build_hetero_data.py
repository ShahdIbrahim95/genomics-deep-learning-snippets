"""
Constructs a heterogeneous PyG graph (HeteroData) from multi-modal genomic data.
Partitions global node embeddings (BioBERT priors) and routes directed edge
attributes across 6 biological node classes and 18 regulatory interaction types.
"""

from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import torch
from torch_geometric.data import Data, HeteroData


def convert_to_hetero_data(
    homogeneous_data: Data,
    node_type_names: List[str],
    num_type_classes: int = 7,
) -> HeteroData:
    """
    Splits a single monolithic graph into a typed HeteroData structure.
    """
    # Recover categorical node type from one-hot suffix
    node_type_ids = torch.argmax(homogeneous_data.x[:, -num_type_classes:], dim=1)
    node_types = [node_type_names[i] for i in node_type_ids.tolist()]

    # Index partitioning
    type_to_global_indices = defaultdict(list)
    for idx, n_type in enumerate(node_types):
        type_to_global_indices[n_type].append(idx)

    global_to_local = {
        (n_type, g_idx): l_idx
        for n_type, indices in type_to_global_indices.items()
        for l_idx, g_idx in enumerate(indices)
    }

    hetero = HeteroData()

    # Add node features without the one-hot type vector
    for n_type, indices in type_to_global_indices.items():
        idx_tensor = torch.tensor(indices, dtype=torch.long)
        hetero[n_type].x = homogeneous_data.x[idx_tensor, :-num_type_classes]

    # Map edges to typed relations
    edge_groups = defaultdict(list)
    edge_attrs = defaultdict(list)

    for i in range(homogeneous_data.edge_index.shape[1]):
        src = homogeneous_data.edge_index[0, i].item()
        dst = homogeneous_data.edge_index[1, i].item()
        rel_key = (node_types[src], "regulates", node_types[dst])

        edge_groups[rel_key].append((
            global_to_local[(node_types[src], src)],
            global_to_local[(node_types[dst], dst)],
        ))
        edge_attrs[rel_key].append(homogeneous_data.edge_attr[i])

    for rel_key, edges in edge_groups.items():
        src_local, dst_local = zip(*edges)
        hetero[rel_key].edge_index = torch.tensor([src_local, dst_local], dtype=torch.long)
        hetero[rel_key].edge_attr = torch.stack(edge_attrs[rel_key])

    return hetero