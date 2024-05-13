# from paper: GATED GRAPH SEQUENCE NEURAL NETWORKS
# "The biggest modification of GNNs is that the authors use Gated Recurrent Units
# (Cho et al., 2014) and unroll the recurrence for a fixed number of steps T
# and use backpropagation through time in order to compute gradients."

# Based on the following tutorial: https://github.com/AntonioLonga/PytorchGeometricTutorial/tree/main/Tutorial9

import os
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn.inits import uniform
from torch_geometric.nn.conv import MessagePassing
from torch.nn import Parameter as Param
from torch import Tensor
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import time
from datetime import datetime

torch.manual_seed(42)


# region Gated Graph Conv
# BASE Class for information propagation and update amongst nodes
class GatedGraphConv(MessagePassing):

    def __init__(self, dataset, out_channels, num_layers, aggr="add", bias=True, **kwargs):
        super(GatedGraphConv, self).__init__(aggr=aggr, **kwargs)

        self.out_channels = out_channels
        self.num_layers = num_layers
        self.data = dataset[0]

        self.rnn = torch.nn.GRUCell(self.out_channels, self.out_channels, bias=bias)
        self.weight = Param(
            Tensor(self.num_layers, self.out_channels, self.out_channels)
        )

        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.out_channels, self.weight)
        self.rnn.reset_parameters()

    def forward(self, x):

        edge_index = self.data.edge_index
        edge_weight = self.data.edge_attr

        if x.size(-1) > self.out_channels:
            raise ValueError(
                "The number of input channels is not allowed to "
                "be larger than the number of output channels"
            )

        # Create padding in case input is smaller than output
        if x.size(-1) < self.out_channels:
            zero = x.new_zeros(x.size(0), self.out_channels - x.size(-1))
            x = torch.cat([x, zero], dim=1)

        for i in range(self.num_layers):
            m = torch.matmul(x, self.weight[i])

            # Propagation model based on point 3.2 of paper
            m = self.propagate(edge_index, x=m, edge_weight=edge_weight, size=None)
            x = self.rnn(m, x)

        return x

    def __repr__(self):
        return "{}({}, num_layers={})".format(
            self.__class__.__name__, self.out_channels, self.num_layers
        )


# endregion
# region MLP
class MLP(nn.Module):
    def __init__(self, input_dim, hid_dims):
        super(MLP, self).__init__()

        # Create the sequential model
        self.mlp = nn.Sequential()
        # List of dimensions
        dims = [input_dim] + hid_dims
        for i in range(len(dims) - 1):
            self.mlp.add_module(
                "lay_{}".format(i),
                nn.Linear(in_features=dims[i], out_features=dims[i + 1]),
            )
            if i + 1 < len(dims):
                self.mlp.add_module("act_{}".format(i), nn.Tanh())

    def reset_parameters(self):
        for i, l in enumerate(self.mlp):
            if type(l) == nn.Linear:
                nn.init.xavier_normal_(l.weight)

    def forward(self, x):
        return self.mlp(x)


# endregion


# region GGNN
class GGNN(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        dataset,
        out_channels=None,
        num_conv=3,
        hidden_dim=32,
        aggr="add",
        mlp_hdim=32,
        mlp_hlayers=3,
        
    ):
        super(GGNN, self).__init__()
        self.emb = lambda x: x
        self.best_acc = -1

        if out_channels is None:
            out_channels = in_channels

        if in_channels != out_channels:
            print("mismatch, operating a reduction")
            self.emb = nn.Linear(in_channels, hidden_dim, bias=False)

        self.conv = GatedGraphConv(dataset=dataset, out_channels=out_channels, num_layers=num_conv)

        self.mlp = MLP(input_dim=out_channels, hid_dims=[mlp_hdim] * mlp_hlayers)

        self.out_layer = nn.Linear(mlp_hdim, dataset.num_classes)

    def forward(self, data):

        # Linear Encoding / Feature Reduction
        x = self.emb(data.x)

        # Propagation and GRU
        x = self.conv(x)

        # MLP
        x = self.mlp(x)

        # Linear Decoding
        x_emb = self.out_layer(x)

        # Prediction
        return x_emb, F.log_softmax(x_emb, dim=-1)
