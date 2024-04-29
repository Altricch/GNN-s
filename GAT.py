# https://www.youtube.com/watch?v=CwsPoa7z2c8

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

import argparse

class GATConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout = 0.6, alpha = 0.2, concat=True):
        super(GATConv, self).__init__()
        
        self.dropout = dropout        
        self.in_features = in_channels    
        self.out_features = out_channels 
        self.alpha = alpha          # LeakyReLU with negative input slope, alpha = 0.2
        self.concat = concat         # conacat = True for all layers except the output layer.

        
        # Xavier Initialization of Weights
        # Alternatively use weights_init to apply weights of choice 
        self.W = nn.Parameter(torch.zeros(size=(in_channels, out_channels)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        self.a = nn.Parameter(torch.zeros(size=(1, 2*out_channels)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        # LeakyReLU
        self.leakyrelu = nn.LeakyReLU(self.alpha)
    
    
    def forward(self, x, edge_index):
        
        # Linear Transformation
        # breakpoint()
        wh = torch.mm(x, self.W)
        
        
        # Attention Mechanism
        a_input = torch.cat([wh[edge_index[0]], wh[edge_index[1]]], dim=1)
        # print("Shape of a_input before attention:", a_input.shape)
        # e = self.leakyrelu(self.a @ a_input.t())
        e = self.leakyrelu((self.a * a_input).sum(dim=1, keepdim=True))
        # print("Shape of e after LeakyReLU:", e.shape)
        
        # Masked Attention
        # zero_vec is used to mask out the elements that should be ignored during the attention calculation.
        zero_vec = -9e15*torch.ones_like(e)
        # masking out the elements that should be ignored during the attention calculation.
        attention = torch.where(e > 0, e, zero_vec)
        
        
        attention = F.softmax(attention, dim=0)
        # attention = F.dropout(attention, self.dropout, training=self.training)
        
        # print("Shape of attention after softmax:", attention.shape)
        # print("Shape of wh after attention:", wh.shape)
        # breakpoint()
        # h_prime = torch.matmul(attention, wh)
        
        # Manual feature aggregation
        h_prime = torch.zeros_like(wh)
        # Gather contributions from source nodes to target nodes
        for i in range(wh.size(0)):
            # Find edges where the current node is the target
            mask = (edge_index[1] == i)
            if mask.any():
                # Aggregate the weighted features
                h_prime[i] = (attention[mask] * wh[edge_index[0][mask]]).sum(dim=0)
        
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


class GAT(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim, n_layers, dropout = 0.6, alpha = 0.2, concat=True):
        super(GAT, self).__init__()
        
        self.n_features = in_channels
        self.n_classes = out_channels
        self.n_hidden = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        
        self.emb = nn.Linear(in_channels, hidden_dim)
        
        self.layers = nn.ModuleList()
        for _ in range(1, n_layers):
            self.layers.append(GATConv
                               (hidden_dim,
                                hidden_dim,))
        
        # Output layer
        self.out_att = nn.Linear(hidden_dim, out_channels)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = self.emb(x)
        
        for layer in self.layers:
            x = layer(x, edge_index)
        
        x = self.out_att(x)
        
        x = F.elu(x)
        
        return F.log_softmax(x, dim=1)

def train(dataset, conv_layer, writer,  epochs, anti_symmetric =True):
    test_loader = loader =  DataLoader(dataset, batch_size = 1, shuffle = True)

    # Build model
    # input_dim, hidden_dim, out_dim
    model = GAT(dataset.num_node_features, dataset.num_classes, 32, conv_layer)
    opt = optim.Adam(model.parameters(), lr = 0.01)
    loss_fn = nn.NLLLoss()
    test_accuracies = []
    
    print("#"*20 + f" Running GAT, with {str(epochs)} epochs, {str(conv_layer)} convs " + "#"*20)

    for epoch in range(0,epochs):
        print("Epoch", epoch)
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
            print("Epoch {}. Loss {:.4f}. Test accuracy {:.4f}".format(epoch, total_loss, test_acc))
            model.best_accuracy = max(test_accuracies)
            print("best accuracy is", max(test_accuracies))
            writer.add_scalar("test accuracy", test_acc, epoch)
            
    return model

def test(loader, model, is_validation = False):
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
parser = argparse.ArgumentParser(description='Process some inputs.')
parser.add_argument('--epoch', type=int, help='Epoch Amount', default=100)
parser.add_argument('--conv', type=int, help='Conv Amount', default=3)
parser.add_argument('--asym', type=bool, help='Use AntiSymmetric Weights', default=1)


if __name__ == "__main__":

    args = parser.parse_args()
    # Node classification
    writer = SummaryWriter("./PubMed/" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    dataset = Planetoid(root='/tmp/PubMed', name = 'PubMed')
    conv_layer = args.conv
    model = train(dataset, conv_layer, writer, args.epoch)  
