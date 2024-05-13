from ADGN_Message import ADGN
from GCN import GCN
from GAT import GAT
from GGNN import GGNN

import numpy as np
from datetime import datetime
import torch

# print(torch.__file__)
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


import psutil
import sys
from time import sleep, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ComputationRequirements.compstats import computeStatsModel

# Train
def train(dataset, conv_layer, writer, epochs, model, lr, hidden_layer, loss_func=None):
    test_loader = loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Build model
    # model = GCN(max(dataset.num_node_features, 1), hidden_layer, dataset.num_classes, conv_layers=conv_layer)
    opt = optim.Adam(model.parameters(), lr=lr)

    loss_fn = loss_func

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
            # breakpoint()
            opt.zero_grad()
            embedding, pred = model(batch)
            label = batch.y

            pred = pred[batch.train_mask]
            label = label[batch.train_mask]

            loss = loss_fn(pred, label)
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch.num_graphs
        total_loss /= len(loader.dataset)

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

    mean_avg_accuracy = sum(test_accuracies) / len(test_accuracies)

    return model, mean_avg_accuracy


# Test
def test(loader, model, is_validation=False):
    model.eval()

    correct = 0
    for data in loader:
        with torch.no_grad():
            embeddings, pred = model(data)
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


# Compare different models with different hyperparameters
def conv():

    current_filename = os.path.abspath(__file__).split("/")[-1]
    
    requirements = {}
    
    file_path = 'comp_per_model/compare_all.py_requirements.json'
    if os.path.exists(file_path):
        print("[WARNING]\n Importing an already existing json file for the requirements dictionary")
        print(file_path)
        # Open the file and load the data
        with open(file_path, 'r') as file:
            requirements = json.load(file)
    else:
        print(current_filename)
        current_filename + "_requirements.json"

    writer = SummaryWriter("./Conv_layer/runs/" + "convcomp")

    convs = [1, 5, 10, 20]
    learning_rates = [0.003]
    hidden_layers = [10, 20, 30]
    models = ["ADGNT", "ADGNF", "GCN", "GAT", "GGNN"]
    
    dataset = Planetoid(root="/tmp/PubMed", name="PubMed")
    
    # Get the current process ID
    pid = os.getpid()
    # Get the psutil Process object using the PID
    current_process = psutil.Process(pid)
    num_cpus = psutil.cpu_count()
    
    start_time = time()

    #Prevent CPU 0%
    computeStatsModel(0, -1, -1, -1, current_process, -1, requirements, "None")
    counter_conf = 0 

    conv_mean_acc = {}

    for i in models:
        conv_mean_acc[i] = []

    for conv in convs:
        for lr in learning_rates:
            for lay in hidden_layers:

                for m in models:
                    if m == "ADGNT":
                        loss_fn = nn.CrossEntropyLoss()
                        model = ADGN(
                            max(dataset.num_node_features, 1),
                            lay,
                            dataset.num_classes,
                            num_layers=conv,
                            antisymmetric=True,
                        )
                    elif m == "ADGNF":
                        loss_fn = nn.CrossEntropyLoss()
                        model = ADGN(
                            max(dataset.num_node_features, 1),
                            lay,
                            dataset.num_classes,
                            num_layers=conv,
                            antisymmetric=False,
                        )
                    # H
                    elif m == "GAT":
                        loss_fn = nn.NLLLoss()
                        model = GAT(
                            dataset.num_node_features, dataset.num_classes, lay, conv
                        )

                    elif m == "GGNN":
                        loss_fn = nn.CrossEntropyLoss()
                        model = GGNN(
                            in_channels=dataset.x.shape[-1],
                            out_channels=lay,
                            num_conv=conv,
                            hidden_dim=lay,
                            mlp_hdim=lay,
                        )

                    # else m == 'GCN':
                    else:
                        loss_fn = nn.CrossEntropyLoss()
                        model = GCN(
                            max(dataset.num_node_features, 1),
                            lay,
                            dataset.num_classes,
                            conv_layers=conv,
                        )

                    conf_start_time = time()
                    
                    model, mean_accuracy = train(
                        dataset,
                        conv,
                        writer=writer,
                        epochs=100,
                        model=model,
                        lr=lr,
                        hidden_layer=lay,
                        loss_func=loss_fn,
                    )
                    conv_mean_acc[m].append(mean_accuracy)
                    
                    computeStatsModel(conf_start_time, conv, lr, lay, current_process, num_cpus, requirements, m, True, file_path, counter_conf)
                    counter_conf += 1

    # NEXT LINE ONLY FOR CPU AND MEMORY REQUIREMENTS
    writer.close()
    
    '''
    for j, conv in enumerate(convs):
        writer.add_scalars(
            "test accuracy",
            {
                "ADGNT": conv_mean_acc["ADGNT"][j],
                "ADGNF": conv_mean_acc["ADGNF"][j],
                "GCN": conv_mean_acc["GCN"][j],
                "GAT": conv_mean_acc["GAT"][j],
                "GGNN": conv_mean_acc["GGNN"][j],
            },
            conv,
        )
        
    
    averaged_data = {}

    # Process each entry in the original data
    for key, values in conv_mean_acc.items():
        # Compute averages of consecutive groups of three values
        averaged_values = [sum(values[i : i + 3]) / 3 for i in range(0, len(values), 3)]
        averaged_data[key] = averaged_values

    with open("Conv_layer/comp_per_model/averaged_data.json", "w") as file:
        json.dump(averaged_data, file, indent=4)

    for model, y_values in averaged_data.items():
        plt.plot(convs, y_values, label=model)

    # Add labels and legend
    plt.xlabel("Nr. of conv layers")
    plt.ylabel("Accuracy")
    plt.title("Accuracy with increasing number of layers for each model")
    plt.legend()

    # Save the plot as an image file
    plt.savefig("Conv_layer/conv_mean_acc_plot.png", bbox_inches="tight")

    # Show plot
    plt.show()
    
    '''


if __name__ == "__main__":

    conv()
