### Basics: 
In GNN we have node level predictions, Edge level predictions (also called Link Predictions) and even graph level predictions

#### Difficulty of GNN: 
Size and shape of graphs change and cannot just be resized. Also, two different graphs can be structurally the same, thus they are the same -> Permutation invariance (thus we cannot used Adjacency matrix as input) Last, graph exists in non-euclidean space.

Here, we use representation learning hrough node level emebdding (each node knows something about the other nodes). This is learned through message passing layers

#### Graph Convolution / Message Passing:
Use neighborhood of node and combine info into new embedding. Message passing can happen through various aggregation methods to combine information. Of course (as in CNNs) the convolution can change the size of the embeddings.

Stacking too many message passing layers can lead to oversmoothing (GNNs exponentially loose expressive power for node classifications (source: https://disco.ethz.ch/courses/fs21/seminar/talks/GNN_Oversmoothing.pdf)). This means that we will ultimately. Update functions can be mean, max, NN or RNN's and Aggregate functions can be Mean Max Normalized Sum and NNs.

Advanced methods uses aggregation as a normalized sum of the neighboring states or use MLP to do the aggregate of neighboring states (which then becomes learnable). Lastly, Graph attention networks are important as well.

#### GCNConv & GINConv
GCNConv and GINConv are instances 

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


    ├── README.md                       <- The top-level README.
    │
    ├── requirements.txt                <- The conda environment requirements.
    │
    ├── Conv_layer                      <-  Folder to prof or disprove whether a increase in depth in the model architecture leads to a decay in accuracy of not
    │    |
    |    ├── ADGN_Message.py            <- A-DGN architecture
    │    |
    │    ├── averaged_data.json         <- Average accuracies across all epochs and hidden layers
    │    |
    │    ├── compare_all.py             <- Script that runs the comparison, store the accuracies, and produces the visualization conv_mean_acc_plot.png
    │    |
    │    ├── conv_mean_acc_plot.png     <- Accuracy plot with an increase number of layers for each model
    │    |
    │    ├── GCN.py                     <- GCN (Graph Convolution Network) architecture
    │
    ├── GNN_Basics                      <- 
    |   |
    |   ├── CustomConv.py               <- 
    │   |
    │   ├── GNN.py                      <-
    |   
    ├── Hyperparameter                  <- 
    |   |
    |   ├── ADGN_Message                <-
    |   |
    |   ├── ADGN_MEssage.py_config.json <-
    |   |
    |   ├── ADGN.png                    <-
    |   |
    |   ├── GCN.png                     <-
    |   |
    |   ├── GCN.py                      <-
    |   |
    |   ├── GCN.py_config.json          <-
    |   |
    |   ├── visualization.py            <-
    |
    ├── PDF                             <- PDF Folder
    |   |
    |   ├── ADGN.pdf                    <- ADGN paper
    |   |
    |   ├── GAT.pdf                     <- GAT paper
    |   |
    |   ├── GGNN.pdf                    <- GGNN paper
    |
    ├── Train                           <- Train Folder
    |   |
    |   ├── ADGN_Message.py             <- ADGN 
    |   |
    |   ├── GAT.py                      <- GAT 
    |   |
    |   ├── GCN.py                      <- GCN
    |   |
    |   ├── GGNN.py                     <- GGNN