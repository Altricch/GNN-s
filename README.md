### Introduction
This repository has the goal to reproduce the Anti-Symmetric DGN architecture and compare it with other standard convolutional graph model such as DGN, GAT, GCN and GGNN. A-DGN was developed by Alessio Gravina, Davide Bacciu and Claudio Gallicchio (https://openreview.net/forum?id=J3Y7cgZOOS).

### Requirements
We assume Miniconda/Anaconda is installed. 

1. Create a conda environment named adgn:
    `conda create -n adgn python=3.12.x`

2. Install the required libraries using the requirements file:
    `conda install --file requirements.txt`

### Repository Structure
The repository is structured as follows:

    ├── README.md                                  <- The top-level README.
    │
    ├── requirements.txt                           <- The conda environment requirements.
    |
    ├── ComputationRequirements                    <-  Folder that contains the script for calculating CPU, Memory and Time consumption.
    │    |
    |    ├── compstats.py                          <- Script that calculates CPU, Memory and Time consumption, storing the data in a json.
    │
    ├── Conv_layer                                 <-  Folder to prof or disprove whether a increase in depth in the model architecture leads to a decay in accuracy of not.
    |    |
    |    ├──comp_per_model                         <- Folder that contains the script for producing plots, json file (for CPU, Memory and Time)
    |    |    |
    |    |    ├── averaged_data.json               <- Average accuracies across all epochs and hidden layers.
    |    |    |
    |    |    ├── compare_all.py_requirements.json <- CPU, Memory and Time consumption json file over convolutional layers
    |    |    |
    |    |    ├── CPU_vs_Convolutions.png          <- CPU consumption over convolutional layers.
    |    |    |
    |    |    ├── Memory_vs_Convolution.png        <- Memory consumption over convolutional layers.
    |    |    |
    |    │    ├── requirements_plot.py             <- Script that produces the requirements consumption's visualizations.
    |    |    |
    |    |    ├── Time_vs_Convolutions.png         <- Time consumption over convolutional layers.
    │    |
    |    ├── ADGN_Message.py                       <- A-DGN architecture.
    │    |
    │    ├── compare_all.py                        <- Script that runs the comparison, store the accuracies, store the requirements, and produces the visualization conv_mean_acc_plot.png.
    |    ├── conv_mean_acc_plot.png                <- Accuracy plot with an increase number of layers for each model.
    │    |
    │    ├── GAT.py                                <- GAT architecture.
    |    |
    │    ├── GCN.py                                <- GCN architecture.
    |    |
    │    ├── GGNN.py                               <- GGNN architecture.
    │
    ├── GNN_Basics                                 <- Basic examples for our understanding.
    |   |
    |   ├── CustomConv.py                          <- Standard implementation for a graph convolution.
    │   |
    │   ├── GNN.py                                 <- Standard implementation of a Graph Neural Network.
    |   


    ├── Hyperparameter                  <- Folder for grid-search and corresponding visualization/results.
    |   |
    |   ├── ADGN_Message                <- Script that performs grid-search for ADGN.
    |   |
    |   ├── ADGN_MEssage.py_config.json <- JSON file for ADGN.py that stores for each number of convolutional layers the best learning rate, hidden dimension and accuracy.
    |   |
    |   ├── ADGN.png                    <- Plot of ADGN_MEssage.py_config.json.
    |   |
    |   ├── GCN.png                     <- Plot of GCN.py_config.json.
    |   |
    |   ├── GCN.py                      <- Script that performs grid-search for GCN.
    |   |
    |   ├── GCN.py_config.json          <- JSON file for GCN.py that stores for each number of convolutional layers the best learning rate, hidden dimension and accuracy.
    |   |
    |   ├── visualization.py            <- Script that generates both GCN.png and ADGN.png.
    |








    ├── PDFs                             <- PDFs Folder.
    |   |
    |   ├── ADGN.pdf                    <- ADGN paper.
    |   |
    |   ├── GAT.pdf                     <- GAT paper.
    |   |
    |   ├── GGNN.pdf                    <- GGNN paper.
    |
    ├── Train                           <- Folder with our models for simple training and script correctness.
    |   |
    |   ├── ADGN_Message.py             <- ADGN.
    |   |
    |   ├── GAT.py                      <- GAT.
    |   |
    |   ├── GCN.py                      <- GCN.
    |   |
    |   ├── GGNN.py                     <- GGNN.

Since time constraints, note that