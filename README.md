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
For each runnable file, we assume you are in the root of the project.
The repository is structured as follows:

    ├── README.md                                  <- The top-level README.
    │
    ├── requirements.txt                           <- The conda environment requirements.
    |
    ├── ComputationRequirements                    <-  Folder that contains the script for calculating CPU, Memory and Time consumption.
    │    |
    |    ├── compstats.py (not runnable)           <- Script that calculates CPU, Memory and Time consumption, storing the data in a json.
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
    |    |    |                                       (python3 Conv_layer/comp_per_model/requirements_plot.py)             
    |    |    |
    |    |    ├── Time_vs_Convolutions.png         <- Time consumption over convolutional layers.
    │    |
    |    ├── ADGN_Message.py (not runnable)        <- A-DGN architecture.
    │    |
    │    ├── compare_all.py                        <- Script that runs the comparison, store the accuracies, store the requirements, and produces the visualization 
    |    |                                             (python3 Conv_layer/compare_all.py)
    |    |
    |    ├── conv_mean_acc_plot.png                <- Accuracy plot with an increase number of layers for each model.
    │    |
    │    ├── GAT.py (not runnable)                 <- GAT architecture.
    |    |
    │    ├── GCN.py (not runnable)                 <- GCN architecture.
    |    |
    │    ├── GGNN.py (not runnable)                <- GGNN architecture.
    │    
    ├── GNN_Basics                                 <- Basic examples for our understanding.
    |   |
    |   ├── CustomConv.py (not runnable)           <- Standard implementation for a graph convolution.
    │   |
    │   ├── GNN.py                                 <- Standard implementation of a Graph Neural Network on Cora.
    |                                                 (python3 GNN_basics/GNN.py)
    |   
    ├── Hyperparameter                             <- Folder for hyperparameter space search and corresponding visualization/results.
    |   |
    |   ├── comp_per_model                         <- Folder that contains the script that procedus json, plots.
    |   |   |
    |   |   ├── ADGN_Hyper.py_config.json          <- JSON file with best configuration per convolution layer amount.
    |   |   |
    |   |   ├── ADGN_Hyper.py_requirements.json    <- Resources utilization per config.
    |   |   |
    |   |   ├── ADGN.png                           <- Plot of ADGN_MEssage.py_config.json.
    |   |   |
    |   |   ├── CPU.png                            <- CPU usage.
    |   |   |
    |   |   ├── GCN.png                            <- Plot of GCN.py_config.json.
    |   |   |
    |   |   ├── GCN_Hyper.py_config.json           <- JSON file with best configuration per convolution layer amount.
    |   |   |
    |   |   ├── gridgrid_search_plotter.py         <- Script that plots hardware utilization.
    |   |   |                                         (python3 Hyperparameter/comp_per_model/grid_search_plotter.py)
    |   |   |
    |   |   ├── Memory.png                         <- Memory usage.
    |   |   |
    |   |   ├── Time.png                           <- Time exectuion time.
    |   |   |
    |   |   ├── visualization.py                   <- Script that plots hyperparameter space search.
    |   |                                             (python3 Hyperparameter/comp_per_model/visualization.py)
    |   |   
    |   ├── ADGN_Hyper.py                          <- ADGN architecture.
    |   |                                             (python3 Hyperparameter/ADGN_Hyper.py)
    |   |
    |   ├── GCN_Hyper.py                           <- GCN architecture.
    |                                                 (python3 Hyperparameter/GCN_Hyper.py)
    |
    ├── Jacobian                                   <- Folder that tests the claims of the paper.
    |   |
    |   ├── ADGN_Jacobian.py                       <- Script that calculates the real part of the Jacobian on random nodes.
    |   |                                            (python3 Jacobian/ADGN_Jacobian.py)
    |   |
    |   ├── max_eigenvalues.csv                    <- CSV file with the max eigenvalues over epochs.
    |   |
    |   ├── max_eigenvalues.png                    <- Plot of the max eigenvalues over epochs with anti symmetric both True and False
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
    |   |                                             (python3 Train/comp_per_model/visualize.py)
    |   |
    |   ├── ADGN_Train.py                          <- ADGN simple train + visualization of clustering of model output.
    |   |                                             (python3 Train/ADGN_Train.py)
    |   |
    |   ├── compstats.py                           <- Script that computes CPU, Memory and Time and storing those in a JSON
    |   |                                             (python3 Train/compostats.py)
    |   |
    |   ├── GAT_Train.py                           <- GAT simple train + visualization of clustering of model output.
    |   |                                             (python3 Train/GAT_Train.py)
    |   |
    |   ├── GCN_Train.py                           <- GCN simple train + visualization of clustering of model output.
    |   |                                             (python3 Train/GCN_Train.py)
    |   |
    |   ├── GGNN_Train.py                           <- GGNN simple train + visualization of clustering of model output.
    |   |                                             (python3 Train/GGNN_Train.py)