"""
Heterogeneous Graph Attention Network (GATv2) Encoder for Genomic Interactomes.
Supports multi-relational biological entities with edge attribute integration.
"""

from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, HeteroConv


class HeteroGNNEncoder(nn.Module):
    """
    Two-layer Heterogeneous Graph Neural Network encoder utilizing GATv2Conv
    wrapped in HeteroConv for edge-attribute-aware message passing.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        metadata: Tuple[List[str], List[Tuple[str, str, str]]],
        edge_dim: int = 13,
        heads: int = 2,
    ):
        super().__init__()

        node_types, edge_types = metadata

        # Assign a GATv2Conv operator to each relation type
        self.convs = nn.ModuleDict({
            "__".join(edge_type): GATv2Conv(
                in_channels=in_channels,
                out_channels=out_channels,
                edge_dim=edge_dim,
                heads=heads,
                concat=False,
                add_self_loops=False,
            )
            for edge_type in edge_types
        })

        # Map back to tuple keys expected by HeteroConv
        conv_dict = {
            edge_type: self.convs["__".join(edge_type)]
            for edge_type in edge_types
        }
        self.hetero_conv = HeteroConv(conv_dict, aggr="sum")

        # Layer normalization per node type
        self.norms = nn.ModuleDict({
            node_type: nn.LayerNorm(out_channels)
            for node_type in node_types
        })

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        edge_attr_dict: Dict[Tuple[str, str, str], torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Forward message-passing step across heterogeneous node and edge types.
        """
        out_dict = self.hetero_conv(x_dict, edge_index_dict, edge_attr_dict)

        # Apply node-specific normalization
        return {
            node_type: self.norms[node_type](x)
            for node_type, x in out_dict.items()
        }