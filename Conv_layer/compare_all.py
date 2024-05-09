from ADGN_Message import ADGN
from GCN import GCN

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


def train(dataset, conv_layer, writer, epochs, model, lr, hidden_layer):
    # Data loader definition
    test_loader = loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Define optimizer and loss function
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    # Accuracy list
    test_accuracies = []

    print("MODEL IS", model)
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
            loss = loss_fn(pred, label)

            # Backward pass
            loss.backward()

            # Update model weights
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


def test(loader, model, is_validation=False):
    model.eval()

    correct = 0
    for data in loader:
        with torch.no_grad():

            # Forward pass
            embeddings, pred = model(data)

            # Get the class with the highest probability
            pred = pred.argmax(dim=1)

            # Extract label from ground truth
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

    # Get the current filename
    current_filename = os.path.abspath(__file__).split("/")[-1]

    # Define the writer for tensorboard
    writer = SummaryWriter("./Conv_layer/runs/" + "convcomp")

    # Define the hyperparameters: number of conv layers, learning rate, hidden layers
    convs = [1, 5, 10, 20]
    learning_rates = [0.003]
    hidden_layers = [10, 20, 30]

    # Define the models to compare
    models = ["ADGNT", "ADGNF", "GCN"]

    # Load the dataset
    dataset = Planetoid(root="/tmp/PubMed", name="PubMed")

    # Initialize the mean accuracy dictionary
    conv_mean_acc = {}

    # Empty list for each model
    for i in models:
        conv_mean_acc[i] = []

    # Iterate over the hyperparameters
    for conv in convs:
        for lr in learning_rates:
            for lay in hidden_layers:

                for m in models:

                    # Based on the model, create the corresponding model
                    if m == "ADGNT":
                        model = ADGN(
                            max(dataset.num_node_features, 1),
                            lay,
                            dataset.num_classes,
                            num_layers=conv,
                            antisymmetric=True,
                        )

                    elif m == "ADGNF":
                        model = ADGN(
                            max(dataset.num_node_features, 1),
                            lay,
                            dataset.num_classes,
                            num_layers=conv,
                            antisymmetric=False,
                        )

                    else:
                        model = GCN(
                            max(dataset.num_node_features, 1),
                            lay,
                            dataset.num_classes,
                            conv_layers=conv,
                        )

                    # Train the model and get the mean accuracy
                    model, mean_accuracy = train(
                        dataset,
                        conv,
                        writer=writer,
                        epochs=100,
                        model=model,
                        lr=lr,
                        hidden_layer=lay,
                    )
                    conv_mean_acc[m].append(mean_accuracy)

    # Write to tensorboard the mean test accuracy for each model
    for j, conv in enumerate(convs):
        writer.add_scalars(
            "test accuracy",
            {
                "ADGNT": conv_mean_acc["ADGNT"][j],
                "ADGNF": conv_mean_acc["ADGNF"][j],
                "GCN": conv_mean_acc["GCN"][j],
            },
            conv,
        )

    # Average accuracy for plotting
    averaged_data = {}

    # Process each entry in the original data
    for key, values in mean_accuracy.items():

        # Compute averages of consecutive groups of three values
        averaged_values = [sum(values[i : i + 3]) / 3 for i in range(0, len(values), 3)]

        # Save the averaged values
        averaged_data[key] = averaged_values

    # Dump the averaged data to a JSON file
    with open("Conv_layer/averaged_data.json", "w") as file:
        json.dump(averaged_data, file, indent=4)

    # Plot the averaged data
    for model, y_values in averaged_data.items():
        plt.plot(convs, y_values, label=model)

    # Add labels and legend
    plt.xlabel("Nr. of conv layers")
    plt.ylabel("Accuracy")
    plt.title("Accuracy with increasing number of layers for each model")
    plt.legend()

    # Save the plot as an image file
    plt.savefig("Conv_layer/conv_mean_acc_plot.png", bbox_inches="tight")

    # Show the plot
    plt.show()


if __name__ == "__main__":

    conv()
