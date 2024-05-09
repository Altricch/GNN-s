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

import os
import json

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


torch.manual_seed(42)
np.random.seed(42)


class ADGNConv(pyg_nn.MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        gamma: float = 0.1,
        epsilon: float = 0.1,
        antisymmetry=True,
    ):
        super(ADGNConv, self).__init__(
            aggr="add"
        )  # "Add" aggregation (can alternatively use mean or max)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.gamma = gamma
        self.epsilon = epsilon
        self.act_func = nn.Tanh()
        self.antisymmetry = antisymmetry
        self.best_accuracy = -1

        # Defines learnable weighs and biases, where W has (nxn) and bias is (n)
        self.Weights = nn.Parameter(
            torch.zeros((self.in_channels, self.in_channels)), requires_grad=True
        )
        # As presented in the paper, bias is always learned from the depiction, thus we always set bias
        self.bias = nn.Parameter(torch.zeros((self.in_channels)), requires_grad=True)
        # Identity as per formula of paper
        self.Identity = nn.Parameter(torch.eye(self.in_channels), requires_grad=False)

        # Aggregation function NEVER has learnable parameters
        self.linear = nn.Linear(self.in_channels, self.out_channels, bias=False)

        # Reset parameters Kaiming takes into account activation function, Xavier does not
        init.kaiming_uniform(self.Weights, np.sqrt(5))
        # fan_in and fan_out = number of neurons in and number of neurons out
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.Weights)
        bound = np.clip(1 / np.sqrt(fan_in), 0, np.inf)
        init.uniform_(self.bias, -bound, bound)
        self.linear.reset_parameters()

    def forward(self, x, edge_index):
        # Antisymmetric formulation (paper formula 5)
        W = (
            ((self.Weights - self.Weights.T) - (self.gamma * self.Identity))
            if self.antisymmetry
            else self.Weights
        )

        # Convolution of neighbors of previous layer PHI*(X(l-1), N_u)
        # Do forward pass for backprop to learn weights of Linear Layer
        aggr_x = self.linear(x)

        # Add self loops to edge index
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Split edge index into row and column
        row, col = edge_index

        # Calculate the degree of each node
        deg = pyg_utils.degree(row, aggr_x.size()[0])
        deg_inv_sqrt = deg.pow(-0.5)

        # Formula 7 of paper, normalization
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Apply message passing by aggregating neighbors
        aggr_x = self.propagate(edge_index, x=aggr_x, norm=norm)

        x_prev = x

        # Apply the function of the paper
        x = x_prev @ W + aggr_x + self.bias
        x = self.epsilon * (self.act_func(x))
        x = x_prev + x

        return x

    def message(self, x_j, norm):
        # Compute messages
        # x_j has shape [E, outchannels]
        return norm.view(-1, 1) * x_j


class ADGN(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_dim,
        out_channels,
        num_layers,
        epsilon=0.1,
        gamma=0.1,
        antisymmetric=True,
    ):
        super(ADGN, self).__init__()

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.epsilon = epsilon
        self.gamma = gamma
        self.antisymmetric = antisymmetric
        self.best_accuracy = -1

        # Embedding layer to reduce dimensionality of input to hidden_dim
        self.emb = None
        if self.hidden_dim is not None:
            self.emb = nn.Linear(self.in_channels, self.hidden_dim, bias=False)

        # Convolutional layers
        self.conv = nn.ModuleList()

        # Apply hidden dimensions in conv block
        for _ in range(1, num_layers):
            self.conv.append(
                (
                    ADGNConv(
                        in_channels=self.hidden_dim,
                        out_channels=self.hidden_dim,
                    )
                )
            )

        # Output layer to map hidden_dim to out_channels
        self.linear = nn.Linear(self.hidden_dim, self.out_channels)

    def forward(self, x):
        # Get node features and edge index
        x, edge_idx = (
            x.x,
            x.edge_index,
        )

        emb = None

        # Apply embedding layer (Linear layer)
        x = self.emb(x)

        # Apply convolutional layers called conv (ModuleList)
        for conv in self.conv:
            x = conv(x, edge_idx)
            emb = x

        # Apply output layer (Linear layer)
        x = self.linear(x)
        return emb, x


# Train the model
def train(
    dataset, conv_layer, writer, epochs, lr=0.01, hidden_layer=32, anti_symmetric=True
):
    test_loader = loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Build the model
    # self, in_channels, hidden_dim, out_channels, num_layers
    model = ADGN(
        max(dataset.num_node_features, 1),
        hidden_layer,
        dataset.num_classes,
        num_layers=conv_layer,
        antisymmetric=anti_symmetric,
    )

    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    test_accuracies = []

    print(
        "#" * 20
        + f" Running ADGN, with {str(epochs)} epochs, {str(conv_layer)} convs and antisymmetric {model.antisymmetric} "
        + "#" * 20
    )

    for epoch in range(0, epochs):
        total_loss = 0
        model.train()

        for batch in loader:

            opt.zero_grad()

            emb, pred = model(batch)
            label = batch.y

            pred = pred[batch.train_mask]
            label = label[batch.train_mask]

            loss = loss_fn(pred, label)

            loss.backward()

            opt.step()

            # Accumulate loss
            total_loss += loss.item() * batch.num_graphs

        # Average loss
        total_loss /= len(loader.dataset)

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

    # Mean average accuracy
    mean_avg_accuracy = sum(test_accuracies) / len(test_accuracies)

    return model, mean_avg_accuracy


# Test the model
def test(loader, model, is_validation=False):
    model.eval()

    correct = 0
    for data in loader:
        with torch.no_grad():

            emb, pred = model(data)

            pred = pred.argmax(dim=1)

            label = data.y

        # Get the mask for the validation or test set
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

# Grid search for hyperparameters
def hyperparameter_search():

    all_best_acc = float("-inf")
    all_best_lr = 0
    all_best_hidden = 0

    current_filename = os.path.abspath(__file__).split("/")[-1]

    configs = {}

    # Possible hyperparameters
    convs = [1, 2, 3, 5, 10, 12, 20, 30]
    learning_rates = [0.1, 0.01, 10e-3, 10e-4, 10e-5]
    hidden_layers = [4, 8, 12, 24, 48, 64, 128]

    dataset = Planetoid(root="/tmp/PubMed", name="PubMed")

    for conv in convs:

        # Reset best accuracy, learning rate and hidden layer
        all_best_acc = float("-inf")
        all_best_lr = 0
        all_best_hidden = 0

        # Iterate over possible learning rates and hidden layers
        for lr in learning_rates:
            for lay in hidden_layers:

                # Train the model and get the best accuracy
                model, best_accuracy = train(
                    dataset, conv, writer=None, epochs=100, lr=lr, hidden_layer=lay
                )

                # Update best hyperparameters
                if all_best_acc < best_accuracy:
                    all_best_lr = lr
                    all_best_hidden = lay
                    all_best_acc = best_accuracy

        # Store the best hyperparameters
        configs[conv] = {
            "LR": all_best_lr,
            "Hidden": all_best_hidden,
            "ACC": all_best_acc,
        }

    # Dump into a json file to be retrieved when testing
    with open(current_filename + "_config.json", "w") as json_file:
        json.dump(configs, json_file, indent=4)


### Flags Areas ###
import argparse

parser = argparse.ArgumentParser(description="Process some inputs.")
parser.add_argument("--epoch", type=int, help="Epoch Amount", default=100)
parser.add_argument("--conv", type=int, help="Conv Amount", default=3)
parser.add_argument("--asym", type=bool, help="Use AntiSymmetric Weights", default=1)

if __name__ == "__main__":

    # args = parser.parse_args()

    hyperparameter_search()
