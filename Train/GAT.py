# https://www.youtube.com/watch?v=CwsPoa7z2c8
# https://github.com/Altricch/GNN-s/blob/main/Train/GAT.py

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

# region: GATConv old
# class GATConv(nn.Module):
#     def __init__(self, in_channels, out_channels, heads, 
#                 negative_slope = 0.2, dropout=0.6, concat=True):
        
#         super(GATConv, self).__init__()

#         self.dropout = dropout
#         self.in_features = in_channels
#         self.out_features = out_channels
#         self.heads = heads
#         self.negative_slope = negative_slope # LeakyReLU
#         self.concat = concat  # concat = True for all layers except the output layer

#         # Xavier Initialization of Weights
#         # self.W = nn.Parameter(torch.zeros(size=(in_channels, self.heads * out_channels)))
#         self.W = nn.Parameter(torch.Tensor(in_channels, heads * out_channels))
#         self.a = nn.Parameter(torch.Tensor(1, heads, 2 * out_channels))
#         # nn.init.xavier_uniform_(self.W.data, gain=1.414)

#         # Attention Weights
#         # self.a = nn.Parameter(torch.zeros(size=(1, heads, 2 * out_channels)))
#         # nn.init.xavier_uniform_(self.a.data, gain=1.414)
#         self.reset_parameters()
        
#         # LeakyReLU
#         self.leakyrelu = nn.LeakyReLU(self.negative_slope)
    
#     def reset_parameters(self):
#         init.xavier_uniform_(self.W)
#         init.xavier_uniform_(self.a)

#     def forward(self, x, edge_index):

#         # Linear Transformation
#         # x = torch.mm(x, self.W)
#         # # breakpoint()
#         # # 
#         # wh1 = torch.matmul(x, self.a[:self.out_features, :])
#         # wh2 = torch.matmul(x, self.a[self.out_features:, :])
        
#         # # 
#         # e = self.leakyrelu(wh1 + wh2.T)
        
#         # # 
#         # attention = F.softmax(e, dim=1)
        
#         # # 
#         # # attention = F.softmax(e, dim=1)
        
#         # # 
#         # h_prime = torch.matmul(attention, x)
        
#         # # 
#         # if self.concat:
#         #     return F.elu(h_prime)
#         # else:
#         #     return h_prime
#         # breakpoint()
#         x = torch.mm(x, self.W).view(-1, self.heads, self.out_features)
#         edge_index_i, edge_index_j = edge_index
        
#         # Calculate attention coefficients
#         x_i = x[edge_index_i]
#         x_j = x[edge_index_j]
        
#         a_input = torch.cat([x_i, x_j], dim=-1).unsqueeze(0) * self.a
        
#         e = self.leakyrelu(a_input).sum(dim=-1)
        
#         alpha = F.softmax(e, dim=1)
        
#         out = (x_j * alpha.unsqueeze(-1)).sum(dim=1)
        
#         if self.concat:
#             return out.view(-1, self.heads * self.out_features)
#         else:
#             return out.mean(dim=1)
# endregion 


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

        self.emb = nn.Linear(in_channels, hidden_dim)
        
        self.conv1 = GATConv(hidden_dim, hidden_dim, heads=self.n_heads, dropout=dropout, concat=self.concat)
        self.conv2 = GATConv(hidden_dim * heads, out_channels, heads=self.n_heads, dropout=dropout, concat=False)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.emb(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = F.elu(self.conv1(x, edge_index))
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv2(x, edge_index)
        
        return F.log_softmax(x, dim=1)


def train(dataset, hidden_dim, writer, epochs, heads):
    
    test_loader = loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Build model
    # input_dim, output_dim, hidden_dim, num_heads
    model = GAT(dataset.num_node_features, dataset.num_classes, hidden_dim, heads)
    opt = optim.Adam(model.parameters(), lr=0.003)
    loss_fn = nn.NLLLoss()
    test_accuracies = []

    print(
        "#" * 20
        + f" Running GAT, with {str(epochs)} epochs, {str(heads)} heads "
        + "#" * 20
    )

    for epoch in range(0, epochs):
        total_loss = 0
        model.train()

        for batch in loader:
            opt.zero_grad()
            pred = model(batch)
            label = batch.y

            pred = pred[batch.train_mask]
            label = label[batch.train_mask]

            loss = loss_fn(pred, label)

            loss.backward()

            opt.step()

            total_loss += loss.item() * batch.num_graphs
        total_loss /= len(loader.dataset)
        # tensorboard
        writer.add_scalar("Loss", total_loss, epoch)

        if epoch % 10 == 0:
            test_acc = test(test_loader, model)
            test_accuracies.append(test_acc)
            print(
                "Epoch {}. Loss {:.4f}. Test accuracy {:.4f}".format(
                    epoch, total_loss, test_acc
                )
            )
            model.best_accuracy = max(test_accuracies)
            print("best accuracy is", max(test_accuracies))
            writer.add_scalar("test accuracy", test_acc, epoch)

    return model


def test(loader, model, is_validation=False):
    model.eval()

    correct = 0
    for data in loader:
        with torch.no_grad():
            pred = model(data)
            pred = pred.argmax(dim=1)
            label = data.y

        mask = data.val_mask if is_validation else data.test_mask

        # Node classification: only evaluate in test set
        pred = pred[mask]
        label = data.y[mask]

        correct += pred.eq(label).sum().item()

    else:
        total = 0
        for data in loader.dataset:
            total += torch.sum(data.test_mask).item()
    return correct / total


### Flags Areas ###
parser = argparse.ArgumentParser(description="Process some inputs.")
parser.add_argument("--epoch", type=int, help="Epoch Amount", default=100)
parser.add_argument("--hidden", type=int, help="Hidden Dimension", default=10)
parser.add_argument("--heads", type=int, help="Use number of heads", default=30)


if __name__ == "__main__":

    args = parser.parse_args()
    # Node classification
    writer = SummaryWriter("./PubMed/" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    dataset = Planetoid(root="/tmp/PubMed", name="PubMed")
    heads = args.heads
    epochs = args.epoch
    hidden_dim = args.hidden
    model = train(dataset, hidden_dim, writer, epochs, heads)
