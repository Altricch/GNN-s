# GAT = Graph Attention Network --> let to learn automatically the importance of each node in the graph

# Graph attention layer
# 1. Apply a shared linear transformation to all nodes
# 2. Self attention e_ij = a(W * h_i, W * h_j) specifies the importance of node j's features to node i
# 3. Normalize the attention scores using the softmax function
# 4. the attention mechanism 'a' is a single layer feedforward neural network
# activation function: LeakyReLU
# output layer: softmax
# 5. Massage passing
# 6. Multi-head attention
# 7. averaging the output of all heads

# Based on the following tutorial: https://github.com/ebrahimpichka/GAT-pt/blob/main/models.py
# and https://github.com/AntonioLonga/PytorchGeometricTutorial/blob/main/Tutorial3/Tutorial3.ipynb

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

# Visualization
from datetime import datetime

# nn module for Neural Network model
# Implement graph conv, gene conf etc.
import torch_geometric.nn as pyg_nn

# Performs some graph utlity functions
import torch_geometric.utils as pyg_utils
from torch_geometric.utils import add_self_loops, degree

# For graph visualization
import networkx as nx
import torch_geometric.transforms as T

# Way to track our training and how well we perform over time
from tensorboardX import SummaryWriter

# to get node embeddings into 2d representation
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Datasets we will use
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader

# GATConv from torch_geometric
from torch_geometric.nn import GATConv

import argparse


class GAT(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_dim,
        heads=5,
        dropout=0.6,
        concat=True,
    ):
        super(GAT, self).__init__()

        self.n_features = in_channels
        self.n_classes = out_channels
        self.n_hidden = hidden_dim
        self.n_heads = heads
        self.dropout = dropout
        self.concat = concat

        # Linear transformation from input to hidden
        self.emb = nn.Linear(self.n_features, self.n_hidden)

        # GAT layer from hidden to hidden
        self.conv1 = GATConv(
            self.n_hidden,
            self.n_hidden,
            heads=self.n_heads,
            dropout=self.dropout,
            concat=self.concat,
        )

        # GATConv from hidden to output
        self.conv2 = GATConv(
            self.n_hidden * self.n_heads,
            self.n_classes,
            heads=self.n_heads,
            dropout=self.dropout,
            concat=False,
        )

    def forward(self, data):

        # Get the node features and edge index
        x, edge_index = data.x, data.edge_index

        # Apply the linear transformation
        x = self.emb(x)
        emb = x

        x = F.dropout(x, p=self.dropout, training=self.training)

        # Apply the GAT layer 1 and the activation function
        x = F.elu(self.conv1(x, edge_index))

        x = F.dropout(x, p=self.dropout, training=self.training)

        # Apply the GAT layer 2
        x = self.conv2(x, edge_index)

        return emb, F.log_softmax(x, dim=1)

