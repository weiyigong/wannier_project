import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import CGConv, TransformerConv


class GATLayer(nn.Module):
    def __init__(self, d_model, n_heads, edge_dim, beta, dropout):
        super().__init__()

        self.d_model = d_model
        assert d_model % n_heads == 0, 'd_model must be divisible by n_heads'
        self.d_head = d_model // n_heads

        self.conv = TransformerConv(
            in_channels=d_model,
            out_channels=self.d_head,
            heads=n_heads,
            concat=True,
            beta=beta,
            dropout=dropout,
            edge_dim=edge_dim
        )
        self.bn = nn.BatchNorm1d(d_model)
        self.leakyrelu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv(x, edge_index, edge_attr)
        x = self.bn(x)
        x = self.leakyrelu(x)
        x = self.dropout(x)
        return x


class GraphEncoder(nn.Module):
    def __init__(self, d_model, edge_dim, num_layers, conv_type, n_heads=None):
        super().__init__()

        if conv_type == 'cgcnn':
            conv = CGConv(d_model, edge_dim, batch_norm=True)
        elif conv_type == 'transformer':
            conv = GATLayer(d_model, n_heads, edge_dim, beta=False, dropout=0.0)
        else:
            raise ValueError

        self.conv_layers = nn.ModuleList([copy.deepcopy(conv) for _ in range(num_layers)])

    def forward(self, x, edge_index, edge_attr):
        for layer in self.conv_layers:
            x = layer(x, edge_index, edge_attr)
        return x


class Model(nn.Module):
    def __init__(self, node_dim, edge_dim, d_model, h_dim, num_layers, conv_type, n_heads=None):
        super().__init__()
        self.embed_node = nn.Linear(node_dim, d_model)

        self.graph_encoder = GraphEncoder(d_model, edge_dim, num_layers, conv_type, n_heads)

        self.fc1 = nn.Linear(d_model * 2 + edge_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, 1)

    def forward(self, data):
        x = self.embed_node(data.x)

        x = self.graph_encoder(x,  data.edge_index, data.edge_attr)

        x = torch.cat([x.index_select(0, data.edge_index[0]), x.index_select(0, data.edge_index[1]), data.edge_attr], dim=-1)

        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)

        return x.squeeze(-1)

