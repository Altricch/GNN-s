### Introduction
This repository has the goal to reproduce the Anti-Symmetric DGN architecture and compare it with other standard convolutional graph model such as DGN, GAT, GCN and GGNN. A-DGN was developed by Alessio Gravina, Davide Bacciu and Claudio Gallicchio (https://openreview.net/forum?id=J3Y7cgZOOS).

### Requirements
We assume Miniconda/Anaconda is installed. 

1. Create a conda environment named adgn:
    `conda create -n adgn python=3.12.x`

2. Install the required libraries using the requirements file:
    `conda install --file requirements.txt`

### Repository Structure
Since time constraints, please note that our code has a lot of content reduncency which has to be clean up in future iterations (e.g, having one file for each model, one train/test loop, etc...).
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
    |    |
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
    ├── Hyperparameter                             <- Folder for hyperparameter space search and corresponding visualization/results.
    |   |
    |   ├── comp_per_model                         <- Folder that contains the script that procedus json, plots.
    |   |   |
    |   |   ├── ADGN_Message.py_config.json        <- JSON file with best configuration per convolution layer amount.
    |   |   |
    |   |   ├── ADGN_Message.py_requirements.json  <- Resources utilization per config.
    |   |   |
    |   |   ├── ADGN.png                           <- Plot of ADGN_MEssage.py_config.json.
    |   |   |
    |   |   ├── CPU.png                            <- CPU usage.
    |   |   |
    |   |   ├── GCN.png                            <- Plot of GCN.py_config.json.
    |   |   |
    |   |   ├── GCN.py_config.json                 <- JSON file with best configuration per convolution layer amount.
    |   |   |
    |   |   ├── gridgrid_search_plotter.py         <- Script that plots hardware utilization.
    |   |   |
    |   |   ├── Memory.png                         <- Memory usage.
    |   |   |
    |   |   ├── Time.png                           <- Time exectuion time.
    |   |   |
    |   |   ├── visualization.py                   <- Script that plots hyperparameter space search.
    |   |   
    |   ├── ADGN_Message.py                        <- ADGN architecture.
    |   |
    |   ├── GCN.py                                 <- GCN architecture.
    |
    ├── Model_Papers_PDF                           <- PDFs Folder.
    |   |
    |   ├── ADGN.pdf                               <- ADGN paper.
    |   |
    |   ├── GAT.pdf                                <- GAT paper.
    |   |
    |   ├── GGNN.pdf                               <- GGNN paper.
    |
    ├── Train                                      <- Folder with our models for simple training and script correctness.
    |   |
    |   ├── comp_per_model                         <- Folder that contains comparison JSON files and plots.
    |   |  |
    |   |  ├── ADGN_Message.py_requirements.json   <- ADGN JSON file for CPU, Memory and Time consumption per epoch.
    |   |  |
    |   |  ├── cpu_comparison.png                  <- Comparative CPU usage.
    |   |  |
    |   |  ├── GAT.py_requirements.json            <- GAT JSON file for CPU, Memory and Time consumption per epoch.
    |   |  |
    |   |  ├── GCN.py_requirements.json            <- GCN JSON file for CPU, Memory and Time consumption per epoch.
    |   |  |
    |   |  ├── GGNN.py_requirements.json           <- GGNN JSON file for CPU, Memory and Time consumption per epoch.
    |   |  |
    |   |  ├── memory_comparison.png               <- Comparative Memory usage.
    |   |  |
    |   |  ├── time_comparison.png                 <- Comparative execution time.
    |   |  |
    |   |  ├── visualize.py                        <- Plotting script.
    |   |
    |   ├── ADGN_Message.py                        <- ADGN architecture.
    |   |
    |   ├── compstats.py                           <- Script that calculates CPU, Memory and Time consumption, storing the data in a JSON
    |   |
    |   ├── GAT.py                                 <- GAT architecture.
    |   |
    |   ├── GCN.py                                 <- GCN architecture.
    |   |
    |   ├── GGNN.py                                <- GGNN architecture.