# !pip install torch-scatter
# !pip install torch-cluster
# !pip install torch-sparse
# !pip install torch-geometric
# !pip install tensorboardX

import numpy as np
import pandas as pd
import seaborn as sns
import os
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

from scipy.linalg import eig

torch.manual_seed(42)
np.random.seed(42)

# List to store the max eigenvalues of the Jacobian matrix
max_eigenvalues_list_true = []
max_eigenvalues_list_false = []

# Global variable to keep track of the epoch
iter = 0


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

        global iter

        # Antisymmetric formulation (paper formula 5)
        W = (
            ((self.Weights - self.Weights.T) - (self.gamma * self.Identity))
            if self.antisymmetry
            else self.Weights
        )

        self.curr_W = W

        # Convolution of neighbors of previous layer PHI*(X(l-1), N_u)
        # Do forward pass for backpropp to learn weights of Linear Layer
        aggr_x = self.linear(x)

        # Add self loops to the edge index
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        row, col = edge_index

        # Calculate the degree of the nodes
        deg = pyg_utils.degree(row, aggr_x.size()[0])
        deg_inv_sqrt = deg.pow(-0.5)

        # Formula 7 of paper, Normalization
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Propagate the messages via aggregation function
        aggr_x = self.propagate(edge_index, x=aggr_x, norm=norm)
        self.curr_aggr_x = aggr_x

        x_prev = x

        # Apply function of paper in the forward pass
        x = x_prev @ W + aggr_x + self.bias

        # For sanity check
        inter_x = self.act_func(x)[iter]

        # Skip Connection and Epsilon scaling
        x = self.epsilon * (self.act_func(x))
        x = x_prev + x

        # Paper claims that the Jacobian matrix should be non-positive
        def jacobian_fun(x_in):
            return self.act_func(
                x_in @ self.curr_W + self.curr_aggr_x[iter] + self.bias
            )

        y = jacobian_fun(x_prev[iter])

        # Perform the Jacobian
        jacobian = torch.autograd.functional.jacobian(jacobian_fun, inter_x).numpy()

        # Extract the real part of the eigenvalues
        eigenvalues, _ = eig(jacobian)

        # Obtain Real Part
        real_part = eigenvalues.real

        # Get the max eigenvalue and store it
        if self.antisymmetry:
            max_eigenvalues_list_true.append(np.max(real_part))
        else:
            max_eigenvalues_list_false.append(np.max(real_part))

        # Check if the Jacobian matrix is non-positive for all eigenvalues of the Jacobian matrix
        # If not, raise an error
        if np.any(real_part > 0) and self.antisymmetry:
            raise ValueError(
                "[FAILED JACOBIAN TEST]\nShould be non-positive for all eigenvalues of Jacobian matrix"
            )

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
        antisymmetry=True,
    ):
        super(ADGN, self).__init__()

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.epsilon = epsilon
        self.gamma = gamma
        self.antisymmetry = antisymmetry

        # Embedding layer (Linear Layer) to reduce the dimensionality of the input to the hidden dimension
        self.emb = None
        if self.hidden_dim is not None:
            self.emb = nn.Linear(self.in_channels, hidden_dim, bias=False)

        # Module list for the convolutional layers
        self.conv = nn.ModuleList()

        # Apply hidden dimensions in conv block
        for _ in range(1, num_layers):
            self.conv.append(
                (
                    ADGNConv(
                        in_channels=self.hidden_dim,
                        out_channels=self.hidden_dim,
                        antisymmetry=self.antisymmetry,
                    )
                )
            )

        # Linear layer to map the hidden dimension to the output dimension
        self.linear = nn.Linear(self.hidden_dim, self.out_channels)

    def forward(self, x):

        # Get the node features and edge index
        x, edge_idx = (
            x.x,
            x.edge_index,
        )

        # Apply the embedding layer (Linear Layer)
        x = self.emb(x)

        # Apply the convolutional layers
        for conv in self.conv:
            x = conv(x, edge_idx)
            emb = x

        # Apply the linear layer
        x = self.linear(x)

        return emb, x


# Cluster the node embeddings
def visualization_nodembs(dataset, model):

    color_list = ["red", "orange", "green", "blue", "purple", "brown", "black"]

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    embs = []
    colors = []

    for batch in loader:

        emb, pred = model(batch)

        embs.append(emb)

        colors += [color_list[y] for y in batch.y]

    embs = torch.cat(embs, dim=0)

    # Get the 2D representation of the embeddings
    xs, ys = zip(*TSNE(random_state=42).fit_transform(embs.detach().numpy()))

    # Plot the 2D representation of the embeddings
    plt.scatter(xs, ys, color=colors)
    plt.title(
        f"ADGN, #epoch:{str(args.epoch)}, #conv:{str(args.conv)}\n accuracy:{model.best_accuracy*100}%, symmetry {model.antisymmetry}"
    )
    plt.show()


# Train the model
def train(dataset, conv_layer, hidden_dim, writer, epochs, antisymmetry=True):

    # Global variable to keep track of the epoch
    global iter

    test_loader = loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Build model
    # self, in_channels, hidden_dim, out_channels, num_layers
    model = ADGN(
        max(dataset.num_node_features, 1),
        hidden_dim,
        dataset.num_classes,
        num_layers=conv_layer,
        antisymmetry=antisymmetry,
    )

    opt = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    test_accuracies = []

    print(
        "#" * 20
        + f" Running ADGN, with {str(epochs)} epochs, {str(conv_layer)} convs and antisymmetric {model.antisymmetry} "
        + "#" * 20
    )

    for epoch in range(0, epochs):
        total_loss = 0
        model.train()
        iter = epoch
        for batch in loader:
            opt.zero_grad()

            emb, pred = model(batch)

            label = batch.y

            # Filter training mask and labels only for node classification
            pred = pred[batch.train_mask]
            label = label[batch.train_mask]

            loss = loss_fn(pred, label)

            loss.backward()

            opt.step()

            # Accumulate the loss
            total_loss += loss.item() * batch.num_graphs

        # Average loss
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
            # Write the test accuracy to tensorboard
            writer.add_scalar("test accuracy", test_acc, epoch)

    return model


# Test the model
def test(loader, model, is_validation=False):
    model.eval()

    correct = 0
    for data in loader:
        with torch.no_grad():
            emb, pred = model(data)

            pred = pred.argmax(dim=1)

            label = data.y

        # Filter the mask for validation or test set
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


# Plot the max eigenvalues
def plot_eigenvalues(path):

    data = pd.read_csv(path)

    fig, ax = plt.subplots()
    sns.lineplot(data=data, ax=ax)
    plt.axhline(0, color="black", linewidth=1)
    ax.set_ylim(-0.5, 3)
    plt.title("Max Eigenvalues of Jacobian Matrix")
    plt.ylabel("Max Eigenvalue")
    plt.xlabel("Epoch")
    plt.savefig(os.path.join(os.getcwd(), "Jacobian/max_eigenvalues.png"))
    plt.show()


### Flags Areas ###
import argparse

parser = argparse.ArgumentParser(description="Process some inputs.")
parser.add_argument("--epoch", type=int, help="Epoch Amount", default=100)
parser.add_argument("--conv", type=int, help="Conv Amount", default=2)
parser.add_argument("--hidden", type=int, help="Hidden Dimension", default=32)
parser.add_argument("--asym", type=bool, help="Use AntiSymmetric Weights", default=1)

if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    writer = SummaryWriter("./PubMed/" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    dataset = Planetoid(root="/tmp/PubMed", name="PubMed")

    args = parser.parse_args()

    # Check if the csv file exists
    csv_path = os.path.join(os.getcwd(), "Jacobian/max_eigenvalues.csv")
    if os.path.exists(csv_path):
        plot_eigenvalues(csv_path)

    else:
        # Otherwise, train the model with the given hyperparameters
        epochs = args.epoch
        conv_layer = args.conv
        hidden_dim = args.hidden
        antisymmetry = True if args.asym == 1 else False
        model = train(
            dataset, conv_layer, hidden_dim, writer, epochs, antisymmetry=antisymmetry
        )
        model = train(
            dataset, conv_layer, hidden_dim, writer, epochs, antisymmetry=False
        )
        # visualization_nodembs(dataset, model)

        # Store the lists in a dataframe and save it to a csv file
        df = pd.DataFrame()
        df["True"] = max_eigenvalues_list_true
        df["False"] = max_eigenvalues_list_false
        df.to_csv(csv_path, index=False)
