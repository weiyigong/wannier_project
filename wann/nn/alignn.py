"""Atomistic LIne Graph Neural Network.

A prototype crystal line graph network dgl implementation.
"""
from typing import Tuple, Union, Optional

import dgl
import dgl.function as fn
import numpy as np
import torch
from dgl.nn import AvgPooling
from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter
import torch_geometric.transforms as T

# from dgl.nn.functional import edge_softmax
from pydantic.typing import Literal
from torch import nn
from torch.nn import functional as F

from pydantic import BaseSettings as PydanticBaseSettings


class BaseSettings(PydanticBaseSettings):
    """Add configuration to default Pydantic BaseSettings."""

    class Config:
        """Configure BaseSettings behavior."""

        extra = "forbid"
        use_enum_values = True
        env_prefix = "jv_"


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


class ALIGNNConfig(BaseSettings):
    """Hyperparameter schema for jarvisdgl.models.alignn."""

    name: Literal["alignn"]
    alignn_layers: int = 4
    gcn_layers: int = 4
    atom_input_features: int = 92
    edge_input_features: int = 80
    triplet_input_features: int = 40
    embedding_features: int = 64
    hidden_features: int = 256
    # fc_layers: int = 1
    # fc_features: int = 64
    output_features: int = 1

    # if link == log, apply `exp` to final outputs
    # to constrain predictions to be positive
    link: Literal["identity", "log", "logit"] = "identity"
    zero_inflated: bool = False
    classification: bool = False
    num_classes: int = 2

    class Config:
        """Configure model settings behavior."""

        env_prefix = "jv_model"


class EdgeGatedGraphConv(nn.Module):
    """Edge gated graph convolution from arxiv:1711.07553.

    see also arxiv:2003.0098.

    This is similar to CGCNN, but edge features only go into
    the soft attention / edge gating function, and the primary
    node update function is W cat(u, v) + b
    """

    def __init__(
            self, input_features: int, output_features: int, residual: bool = True
    ):
        """Initialize parameters for ALIGNN update."""
        super().__init__()
        self.residual = residual
        # CGCNN-Conv operates on augmented edge features
        # z_ij = cat(v_i, v_j, u_ij)
        # m_ij = σ(z_ij W_f + b_f) ⊙ g_s(z_ij W_s + b_s)
        # coalesce parameters for W_f and W_s
        # but -- split them up along feature dimension
        self.src_gate = nn.Linear(input_features, output_features)
        self.dst_gate = nn.Linear(input_features, output_features)
        self.edge_gate = nn.Linear(input_features, output_features)
        self.bn_edges = nn.BatchNorm1d(output_features)

        self.src_update = nn.Linear(input_features, output_features)
        self.dst_update = nn.Linear(input_features, output_features)
        self.bn_nodes = nn.BatchNorm1d(output_features)

    def forward(
            self,
            g: dgl.DGLGraph,
            node_feats: torch.Tensor,
            edge_feats: torch.Tensor,
    ) -> torch.Tensor:
        """Edge-gated graph convolution.

        h_i^l+1 = ReLU(U h_i + sum_{j->i} eta_{ij} ⊙ V h_j)
        """
        g = g.local_var()

        # instead of concatenating (u || v || e) and applying one weight matrix
        # split the weight matrix into three, apply, then sum
        # see https://docs.dgl.ai/guide/message-efficient.html
        # but split them on feature dimensions to update u, v, e separately
        # m = BatchNorm(Linear(cat(u, v, e)))

        # compute edge updates, equivalent to:
        # Softplus(Linear(u || v || e))
        g.ndata["e_src"] = self.src_gate(node_feats)
        g.ndata["e_dst"] = self.dst_gate(node_feats)
        g.apply_edges(fn.u_add_v("e_src", "e_dst", "e_nodes"))
        m = g.edata.pop("e_nodes") + self.edge_gate(edge_feats)

        g.edata["sigma"] = torch.sigmoid(m)
        g.ndata["Bh"] = self.dst_update(node_feats)
        g.update_all(
            fn.u_mul_e("Bh", "sigma", "m"), fn.sum("m", "sum_sigma_h")
        )
        g.update_all(fn.copy_e("sigma", "m"), fn.sum("m", "sum_sigma"))
        g.ndata["h"] = g.ndata["sum_sigma_h"] / (g.ndata["sum_sigma"] + 1e-6)
        x = self.src_update(node_feats) + g.ndata.pop("h")

        # node and edge updates
        x = F.silu(self.bn_nodes(x))
        y = F.silu(self.bn_edges(m))

        if self.residual:
            x = node_feats + x
            y = edge_feats + y

        return x, y


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
        sigmoid_e_sum = scatter(sigmoid_e, index, dim=0, reduce='sum') + 1e-16
        sigmoid_e_sum = sigmoid_e_sum.index_select(0, index)
        node_message = sigmoid_e * self.dst_update(x_j) / sigmoid_e_sum
        return edge_message, node_message

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        edge_message, node_message = inputs
        node_message = scatter(node_message, index, dim=0, reduce='sum')
        return edge_message, node_message


class ALIGNNConv(nn.Module):
    """Line graph update."""

    def __init__(
            self, in_features: int, out_features: int,
    ):
        """Set up ALIGNN parameters."""
        super().__init__()
        self.node_update = EdgeGatedGraphConv(in_features, out_features)
        self.edge_update = EdgeGatedGraphConv(out_features, out_features)

    def forward(
            self,
            g: dgl.DGLGraph,
            lg: dgl.DGLGraph,
            x: torch.Tensor,
            y: torch.Tensor,
            z: torch.Tensor,
    ):
        """Node and Edge updates for ALIGNN layer.

        x: node input features
        y: edge input features
        z: edge pair input features
        """
        g = g.local_var()
        lg = lg.local_var()
        # Edge-gated graph convolution update on crystal graph
        x, m = self.node_update(g, x, y)

        # Edge-gated graph convolution update on crystal graph
        y, z = self.edge_update(lg, m, z)

        return x, y, z


class ALIGNNConvPyG(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.node_update = EdgeGatedGraphConvPyG(in_features, out_features)
        self.edge_update = EdgeGatedGraphConvPyG(out_features, out_features)

    def forward(self, g, lg, x, y, z):
        x, m = self.node_update(g, x, y)
        y, z = self.edge_update(lg, m, z)
        return x, y, z


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


class ALIGNN(nn.Module):
    """Atomistic Line graph network.

    Chain alternating gated graph convolution updates on crystal graph
    and atomistic line graph.
    """

    def __init__(self, config: ALIGNNConfig = ALIGNNConfig(name="alignn")):
        """Initialize class with number of input features, conv layers."""
        super().__init__()
        # print(config)
        self.classification = config.classification

        self.atom_embedding = MLPLayer(
            config.atom_input_features, config.hidden_features
        )

        self.edge_embedding = nn.Sequential(
            RBFExpansion(vmin=0, vmax=8.0, bins=config.edge_input_features, ),
            MLPLayer(config.edge_input_features, config.embedding_features),
            MLPLayer(config.embedding_features, config.hidden_features),
        )
        self.angle_embedding = nn.Sequential(
            RBFExpansion(
                vmin=-1, vmax=1.0, bins=config.triplet_input_features,
            ),
            MLPLayer(config.triplet_input_features, config.embedding_features),
            MLPLayer(config.embedding_features, config.hidden_features),
        )

        self.alignn_layers = nn.ModuleList(
            [
                ALIGNNConv(config.hidden_features, config.hidden_features, )
                for idx in range(config.alignn_layers)
            ]
        )
        self.gcn_layers = nn.ModuleList(
            [
                EdgeGatedGraphConv(
                    config.hidden_features, config.hidden_features
                )
                for idx in range(config.gcn_layers)
            ]
        )

        self.readout = AvgPooling()

        if self.classification:
            self.fc = nn.Linear(config.hidden_features, config.num_classes)
            self.softmax = nn.LogSoftmax(dim=1)
        else:
            self.fc = nn.Linear(config.hidden_features, config.output_features)
        self.link = None
        self.link_name = config.link
        if config.link == "identity":
            self.link = lambda x: x
        elif config.link == "log":
            self.link = torch.exp
            avg_gap = 0.7  # magic number -- average bandgap in dft_3d
            self.fc.bias.data = torch.tensor(
                np.log(avg_gap), dtype=torch.float
            )
        elif config.link == "logit":
            self.link = torch.sigmoid

    def forward(
            self, g: Union[Tuple[dgl.DGLGraph, dgl.DGLGraph], dgl.DGLGraph]
    ):
        """ALIGNN : start with `atom_features`.

        x: atom features (g.ndata)
        y: bond features (g.edata and lg.ndata)
        z: angle features (lg.edata)
        """
        if len(self.alignn_layers) > 0:
            g, lg = g
            lg = lg.local_var()

            # angle features (fixed)
            z = self.angle_embedding(lg.edata.pop("h"))

        g = g.local_var()

        # initial node features: atom feature network...
        x = g.ndata.pop("atom_features")
        x = self.atom_embedding(x)

        # initial bond features
        bondlength = torch.norm(g.edata.pop("r"), dim=1)
        y = self.edge_embedding(bondlength)

        # ALIGNN updates: update node, edge, triplet features
        for alignn_layer in self.alignn_layers:
            x, y, z = alignn_layer(g, lg, x, y, z)

        # gated GCN updates: update node, edge features
        for gcn_layer in self.gcn_layers:
            x, y = gcn_layer(g, x, y)

        # norm-activation-pool-classify
        h = self.readout(g, x)
        out = self.fc(h)

        if self.link:
            out = self.link(out)

        if self.classification:
            # out = torch.round(torch.sigmoid(out))
            out = self.softmax(out)
        return torch.squeeze(out)


class ALIGNNPyG(nn.Module):
    def __init__(self, atom_input_features, edge_input_features, triplet_input_features,
                 embedding_features, hidden_features, alignn_layers, gcn_layers, output_features):
        super().__init__()

        self.atom_embedding = MLPLayer(atom_input_features, hidden_features)

        self.edge_embedding = nn.Sequential(
            RBFExpansion(vmin=0, vmax=8.0, bins=edge_input_features, ),
            MLPLayer(edge_input_features, embedding_features),
            MLPLayer(embedding_features, hidden_features),
        )
        self.angle_embedding = nn.Sequential(
            RBFExpansion(vmin=-1, vmax=1.0, bins=triplet_input_features),
            MLPLayer(triplet_input_features, embedding_features),
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
        y = self.edge_embedding(data.edge_attr)
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


if __name__ == '__main__':
    from wann.data import CrystalGraphPreprocessedDataset, get_train_val_dataloader

    dataset = CrystalGraphPreprocessedDataset('../../dataset/MoS2_96/processed_dataset')
    train_loader, val_loader = get_train_val_dataloader(dataset, batch_size=2, train_ratio=0.8, seed=0, num_workers=16, drop_last=False, pin_memory=True)
    data = next(iter(train_loader))

    model = ALIGNNPyG(
        alignn_layers=4,
        gcn_layers=4,
        atom_input_features=92,
        edge_input_features=80,
        triplet_input_features=40,
        embedding_features=64,
        hidden_features=256,
        output_features=64
    )

    pred = model(data)
