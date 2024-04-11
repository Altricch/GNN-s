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
import torch.nn.init as init

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

class ADGNConv(pyg_nn.MessagePassing):
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 gamma: float = 0.1, 
                 epsilon : float = 0.1, 
                 antisymmetry = True,
                 ):
        super(ADGNConv, self).__init__(aggr='add') # "Add" aggregation (can alternatively use mean or max)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.gamma = gamma
        self.epsilon = epsilon
        self.act_func = nn.Tanh()  
        self.antisymmetry = antisymmetry    
        
        # Defines learnable weighs and biases, where W has (nxn) and bias is (n)
        self.Weights = nn.Parameter(torch.zeros((self.in_channels, self.in_channels)), requires_grad=True)
        # As presented in the paper, bias is always learned from the depiction, thus we always set bias
        self.bias = nn.Parameter(torch.zeros((self.in_channels)), requires_grad=True)
        # Identity as per formula of paper
        self.Identity = nn.Parameter(torch.eye(self.in_channels), requires_grad=False)
        
        # Aggregation function NEVER has learnable parameters
        self.linear = nn.Linear(self.in_channels, self.out_channels, bias=False)
        
        # Reset parameters Kaiming takes into account activation function, Xavier does not
        init.kaiming_uniform(self.Weights, np.sqrt(5))
        # fan_in and fan_out = number of neurons in and number of neurons out
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.W)
        bound = np.clip(1 / np.sqrt(fan_in), 0, np.inf)
        init.uniform_(self.bias, -bound, bound)
        self.linear.reset_parameters()

        # # Conv block
        # self.conv = pyg_nn.GCNConv(in_channels, in_channels, bias=False)
        
    def forward(self, x, edge_index):
        # Antisymmetric formulation (paper formula 5)
        W = ((self.Weights - self.Weights.T) - (self.gamma * self.Identity)) if self.antisymmetry else self.Weights
        
        # Convolution of neighbors of previous layer PHI*(X(l-1), N_u)
        # Do forward pass for backpropp to learn weights of Linear Layer
        aggr_x = self.linear(x)
        aggr_x = self.propagate(aggr_x, edge_index)
        
        # Store previous x
        x_prev = x
        
        # Apply function of paper in the forward pass
        x = (W * x_prev.T + aggr_x + self.bias)
        x = self.epsilon*(self.act_func(x))
        x = x_prev + x
        
        return x
    
    def message(self, x_j, edge_index, size):
        # Compute messages
        # x_j has shape [E, outchannels]
        
        row, col = edge_index
        deg = pyg_utils.degree(row, size[0], dtype=x_j.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        # Here we add outselves back in
        return norm.view(-1, 1) * x_j

class ADGN(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels, num_layers, epsilon, gamma, antisymmetric = True):
        super(ADGN, self).__init__()
        
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.epsilon = epsilon
        self.gamma = gamma
        self.antisymmetric = antisymmetric
        
        self.emb = None
        if self.hidden_dim is not None:
             self.emb = nn.Linear(self.in_channels, hidden_dim, bias=False)
        
        self.conv = nn.ModuleList()
        
        
        # Get 
        self.conv.append(ADGNConv(
            in_channels = in_channels,
            
        ))
        
        # Apply hidden dimensions in conv block
        for _ in range(1, num_layers):
            self.conv.append((ADGNConv(
                in_channels=self.hidden_dim,
            )))
            
        
        
            
        

#TODO: Why multiple times per hidden layer (e.g. num_iterations not clear)
        
        
        