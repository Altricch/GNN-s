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


# Train the model
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

            # Node classification: only compute loss for nodes in the training set
            pred = pred[batch.train_mask]
            label = label[batch.train_mask]

            loss = loss_fn(pred, label)

            loss.backward()

            opt.step()

            # Accumulate the loss
            total_loss += loss.item() * batch.num_graphs

        total_loss /= len(loader.dataset)

        # Write the loss to tensorboard
        writer.add_scalar("Loss", total_loss, epoch)

        # Evaluate the model every 10 epochs on the test set
        if epoch % 10 == 0:
            test_acc = test(test_loader, model)
            test_accuracies.append(test_acc)
            print(
                "Epoch {}. Loss {:.4f}. Test accuracy {:.4f}".format(
                    epoch, total_loss, test_acc
                )
            )
            # Best accuracy so far
            model.best_accuracy = max(test_accuracies)
            print("best accuracy is", max(test_accuracies))
            # Write the accuracy to tensorboard
            writer.add_scalar("test accuracy", test_acc, epoch)

    return model


# Test the model
def test(loader, model, is_validation=False):
    model.eval()

    correct = 0
    for data in loader:
        with torch.no_grad():

            pred = model(data)

            pred = pred.argmax(dim=1)

            label = data.y

        # # Get the mask for the validation or test set
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
    # Parse the arguments and run the model
    args = parser.parse_args()

    writer = SummaryWriter("./PubMed/" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    dataset = Planetoid(root="/tmp/PubMed", name="PubMed")
    heads = args.heads
    epochs = args.epoch
    hidden_dim = args.hidden
    model = train(dataset, hidden_dim, writer, epochs, heads)
