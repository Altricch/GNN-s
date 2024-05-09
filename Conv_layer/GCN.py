
# !pip install torch-scatter
# !pip install torch-cluster
# !pip install torch-sparse
# !pip install torch-geometric
# !pip install tensorboardX

import numpy as np
from datetime import datetime
import torch


import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import json


import os

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


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, conv_layers=2, doutrate=0.4):
        super().__init__()

        # Dropout rate
        self.doutrate = doutrate

        # Number of convolutional layers
        self.conv_layers = conv_layers

        self.best_accuracy = -1
        
        # Module list for convolutional layers
        self.convs = nn.ModuleList()
        self.convs.append(self.build_conv_model(input_dim, hidden_dim))
        for i in range(conv_layers):
            self.convs.append(self.build_conv_model(hidden_dim, hidden_dim))

        # Post message passing defined by MLP
        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(doutrate),
            nn.Linear(hidden_dim, out_dim),
        )

        # Number of total layers
        self.num_layers = self.conv_layers + 1

    # Build convolutional blocks
    def build_conv_model(self, input_dim, hidden_dim):
        # If we perform node classification, we use simple graph convolution
        return pyg_nn.GCNConv(input_dim, hidden_dim)

    def forward(self, data):
        # Data consists of: data.x (feature matrix), data.edge_index (adjecency matrix, what are the edges),
        # data.batch (which node belongs to which graph)
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Sanity check:
        if data.num_node_features == 0:
            # if no features, we use constant feature
            x = torch.ones(data.num_nodes, 1)

        for i in range(self.num_layers):
            # Input to GCN is (node features (|V|, F_in), edge indices (2,E))
            # Output: node features (|V|, F_out)
            
            # Call the convolutional layer
            x = self.convs[i](x, edge_index)

            # Store the embedding
            emb = x

            # Apply activation function and dropout
            x = F.tanh(x)
            x = F.dropout(x, p=self.doutrate, training=self.training)

        # Apply post message passing
        x = self.post_mp(x)

        # Do log softmax for crossentropy
        # return embedding for visualization
        return emb, F.log_softmax(x, dim=1)

    def loss(self, pred, label):
        # Since we return log softmax we need to return negative log likelihood
        return F.nll_loss(pred, label)


def train(dataset, conv_layer, writer, epochs, lr=0.01, hidden_layer=32):

    # Data loader definition
    test_loader = loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Build the model
    model = GCN(
        max(dataset.num_node_features, 1),
        hidden_layer,
        dataset.num_classes,
        conv_layers=conv_layer,
    )

    # Define optimizer
    opt = optim.Adam(model.parameters(), lr=lr)

    # Accuracy list
    test_accuracies = []

    print(
        "#" * 10,
        "Epoch: ",
        epochs,
        " Convs: ",
        conv_layer,
        " LR: ",
        lr,
        " Hidd. Lay: ",
        hidden_layer,
        "#" * 10,
    )

    for epoch in range(0, epochs):
        total_loss = 0
        model.train()

        for batch in loader:
            # Reset gradients
            opt.zero_grad()

            # Forward pass
            embedding, pred = model(batch)

            # Extract labels
            label = batch.y

            # Filter training mask and labels
            pred = pred[batch.train_mask]
            label = label[batch.train_mask]

            # Calculate loss
            loss = model.loss(pred, label)

            # Backward pass
            loss.backward()

            # Model update weights
            opt.step()

            # Accumulate loss
            total_loss += loss.item() * batch.num_graphs

        # Average loss
        total_loss /= len(loader.dataset)

        # Write loss to tensorboard
        if writer is not None:
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

    # Mean average accuracy
    mean_avg_accuracy = sum(test_accuracies) / len(test_accuracies)

    return model, mean_avg_accuracy


def test(loader, model, is_validation=False):
    model.eval()

    correct = 0
    for data in loader:
        with torch.no_grad():
            # Forward pass
            embeddings, pred = model(data)

            # Get the class with highest probability
            pred = pred.argmax(dim=1)

            # Get the label from the ground truth
            label = data.y

        # Get the mask for the validation or test set
        mask = data.val_mask if is_validation else data.test_mask

        # Node classification: only evaluate in test set
        pred = pred[mask]
        label = data.y[mask]

        # Count correct predictions
        correct += pred.eq(label).sum().item()

    else:
        total = 0
        for data in loader.dataset:
            # Total number of nodes in the test set
            total += torch.sum(data.test_mask).item()
    return correct / total


def conv():

    # Variables to store the best accuracy and hyperparameters
    all_best_acc = float("-inf")
    all_best_lr = 0
    all_best_hidden = 0

    # Get the current filename
    current_filename = os.path.abspath(__file__).split("/")[-1]

    # Configs dictionary to store the best hyperparameters
    configs = {}

    # Tensorboard writer
    writer = SummaryWriter("./PubMed/" + datetime.now().strftime("%Y%m%d-%H%M%S"))

    # Hyperparameters to iterate over
    convs = [1, 5, 10, 20]
    learning_rates = [0.003]
    hidden_layers = [10, 20, 30]

    # Load the dataset
    dataset = Planetoid(root="/tmp/PubMed", name="PubMed")

    for conv in convs:

        # Reset the best accuracy, learning rate and hidden layer
        all_best_acc = float("-inf")
        all_best_lr = 0
        all_best_hidden = 0

        # Iterate over the hyperparameters
        for lr in learning_rates:
            for lay in hidden_layers:

                # Train the model and get the best accuracy
                model, best_accuracy = train(
                    dataset, conv, writer=writer, epochs=100, lr=lr, hidden_layer=lay
                )

                # Update the best hyperparameters
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


if __name__ == "__main__":

    conv()
