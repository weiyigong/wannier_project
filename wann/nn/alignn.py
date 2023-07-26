from typing import Optional
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter

RESOURCE_PATH = Path(__file__).parent.parent.parent / 'resources'


class RBFExpansion(nn.Module):
    """Expand interatomic distances with radial basis functions."""

    def __init__(
            self,
            vmin: float = 0,
            vmax: float = 8,
            bins: int = 40,
            lengthscale: Optional[float] = None,
    ):
        """Register torch parameters for RBF expansion."""
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins
        self.register_buffer(
            "centers", torch.linspace(self.vmin, self.vmax, self.bins)
        )

        if lengthscale is None:
            # SchNet-style
            # set lengthscales relative to granularity of RBF expansion
            self.lengthscale = np.diff(self.centers).mean()
            self.gamma = 1 / self.lengthscale

        else:
            self.lengthscale = lengthscale
            self.gamma = 1 / (lengthscale ** 2)

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        """Apply RBF expansion to interatomic distance tensor."""
        return torch.exp(
            -self.gamma * (distance.unsqueeze(1) - self.centers) ** 2
        )


class MLPLayer(nn.Module):
    """Multilayer perceptron layer helper."""

    def __init__(self, in_features: int, out_features: int):
        """Linear, Batchnorm, SiLU layer."""
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.SiLU(),
        )

    def forward(self, x):
        """Linear, Batchnorm, silu layer."""
        return self.layer(x)


class EdgeGatedGraphConvPyG(MessagePassing):
    def __init__(self, input_features: int, output_features: int, residual: bool = True):
        super().__init__(node_dim=0, flow='target_to_source')

        self.residual = residual

        self.src_gate = nn.Linear(input_features, output_features)
        self.dst_gate = nn.Linear(input_features, output_features)
        self.edge_gate = nn.Linear(input_features, output_features)
        self.bn_edges = nn.BatchNorm1d(output_features)

        self.src_update = nn.Linear(input_features, output_features)
        self.dst_update = nn.Linear(input_features, output_features)
        self.bn_nodes = nn.BatchNorm1d(output_features)

    def forward(self, edge_index, x, edge_attr):
        edge_update, x_update = self.update_graph(edge_index, x, edge_attr)
        x = x + x_update
        edge_attr = edge_attr + edge_update
        return x, edge_attr

    def update_graph(self, edge_index, x, edge_attr):
        edge_update, x_update = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        edge_update = F.silu(self.bn_edges(edge_update))

        x_update = self.src_update(x) + x_update
        x_update = F.silu(self.bn_nodes(x_update))
        return edge_update, x_update

    def message(self, x_i, x_j, edge_attr, index):
        edge_message = self.src_gate(x_i) + self.dst_gate(x_j) + self.edge_gate(edge_attr)

        sigmoid_e = torch.sigmoid(edge_message)
        sigmoid_e_sum = scatter(sigmoid_e, index, dim=0, reduce='sum') + 1e-9
        sigmoid_e_sum = sigmoid_e_sum.index_select(0, index)
        node_message = sigmoid_e * self.dst_update(x_j) / sigmoid_e_sum
        return edge_message, node_message

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        edge_message, node_message = inputs
        node_message = scatter(node_message, index, dim=0, reduce='sum')
        return edge_message, node_message


class ALIGNNConvPyG(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.node_update = EdgeGatedGraphConvPyG(in_features, out_features)
        self.edge_update = EdgeGatedGraphConvPyG(out_features, out_features)

    def forward(self, g, lg, x, y, z):
        x, m = self.node_update(g, x, y)
        y, z = self.edge_update(lg, m, z)
        return x, y, z


class ALIGNNPyG(nn.Module):
    def __init__(self, atom_fea_dim, bond_len_fea_dim, bond_cos_fea_dim, angle_fea_dim,
                 embedding_features, hidden_features, alignn_layers, gcn_layers, output_features):
        super().__init__()

        self.atom_embedding = nn.Sequential(
            nn.Embedding.from_pretrained(torch.load(RESOURCE_PATH / 'atom_init.pt'), freeze=True),
            MLPLayer(atom_fea_dim, hidden_features)
        )

        self.bond_len_embedding = RBFExpansion(0, 8, bond_len_fea_dim)
        self.bond_cos_embedding = RBFExpansion(0, 1, bond_cos_fea_dim)

        self.bond_embedding = nn.Sequential(
            MLPLayer(bond_len_fea_dim + 3 * bond_cos_fea_dim, embedding_features),
            MLPLayer(embedding_features, hidden_features),
        )

        self.angle_embedding = nn.Sequential(
            RBFExpansion(-1, 1, angle_fea_dim),
            MLPLayer(angle_fea_dim, embedding_features),
            MLPLayer(embedding_features, hidden_features),
        )

        self.alignn_layers = nn.ModuleList(
            [ALIGNNConvPyG(hidden_features, hidden_features) for _ in range(alignn_layers)]
        )
        self.gcn_layers = nn.ModuleList(
            [EdgeGatedGraphConvPyG(hidden_features, hidden_features) for _ in range(gcn_layers)]
        )

        self.output_head = nn.Linear(hidden_features, output_features)

    def forward(self, data):
        x = self.atom_embedding(data.x)
        y = self.edge_embed(data.edge_attr)
        z = self.angle_embedding(data.lg_edge_attr)
        g = data.edge_index
        lg = data.lg_edge_index_batch

        # ALIGNN updates: update node, edge, triplet features
        for alignn_layer in self.alignn_layers:
            x, y, z = alignn_layer(g, lg, x, y, z)

        # gated GCN updates: update node, edge features
        for gcn_layer in self.gcn_layers:
            x, y = gcn_layer(g, x, y)

        y = self.output_head(y)
        return y

    def edge_embed(self, edge_attr):
        bond_len = edge_attr[..., 0]
        bond_cos = edge_attr[..., 1:]

        bond_len = self.bond_len_embedding(bond_len)
        bond_cos = self.bond_cos_embedding(bond_cos.flatten()).reshape(bond_cos.size(0), -1)
        return self.bond_embedding(torch.cat([bond_len, bond_cos], dim=-1))
