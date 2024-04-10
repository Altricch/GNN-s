# https://www.youtube.com/watch?v=-UjytpbqX4A&t=2221s

# !pip install torch-scatter
# !pip install torch-cluster
# !pip install torch-sparse
# !pip install torch-geometric
# !pip install tensorboardX

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

# nn module for Neural Network model
# Implement graph conv, gene conf etc. 
import torch_geometric.nn as pyg_nn

# Performs some graph utlity functions
import torch_geometric.utils as pyg_utils

# For graph visualization
import networkx as nx
import torch_geometric.transforms as T
# Way to track our training and how well we perform over time
from tensorboardX import SummaryWriter
# to get node embeddings into 2d representation
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Datasets we will use
from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader

class CustomConv(pyg_nn.MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(CustomConv, self).__init__(aggr='add') # "Add" aggregation
        self.linear = nn.Linear(in_channels, out_channels)
        self.lin_self = nn.Linear(in_channels, out_channels)
        
    def forward(self, x, edge_index):
        # Feature matrix is x, adjecency list is edge index
        # x has shape [N, in_channels]
        # edge index has shape [2, E] (as every edge has two nodes)
        
        # Add self-loops to the adjacency matrix.
        # Add self loop -> also want to myself in the convolution
        # We can add a lot of custmization here
        # By adding self loop, we enable skip conections
        edge_index , _ = pyg_utils.add_self_loops(edge_index, num_nodes=x.size(0))
        
        # Transform node feature matrix
        # skip connection: 
        self_x = self.lin_self(x)
        
        # x = self.linear(x)
        
        # propagation calls message, computes message for all nodes 
        # and then does the aggregation
    
        return self_x + self.propagate(edge_index, size=(x.size(0), x.size(0)), x = self.skip(x))
    
    def message(self, x_j, edge_index, size):
        # Compute messages
        # x_j has shape [E, outchannels]
        
        row, col = edge_index
        deg = pyg_utils.degree(row, size[0], dtype=x_j.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        # Here we add outselves back in
        return norm.view(-1, 1) * x_j
    
    # After message passing, do we have additional layers
    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]
        
        return aggr_out    

#TODO: Understand forward funciton (not clear)  
        
        
        