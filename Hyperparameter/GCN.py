# https://www.youtube.com/watch?v=-UjytpbqX4A&t=2221s

# !pip install torch-scatter
# !pip install torch-cluster
# !pip install torch-sparse
# !pip install torch-geometric
# !pip install tensorboardX

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


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, conv_layers = 2, doutrate = 0.4):
        super().__init__()
        self.doutrate = doutrate
        self.conv_layers = conv_layers
        # print("conv layer initialized to", conv_layers)
        # test = getattr(pyg_nn, "GCNConv")
        # breakpoint()
        # Put all our convolution operations
        self.best_accuracy = -1
        self.convs = nn.ModuleList()
        self.convs.append(self.build_conv_model(input_dim, hidden_dim))
        for i in range(conv_layers):
            self.convs.append(self.build_conv_model(hidden_dim, hidden_dim))

        # Post message passing defined by MLP
        self.post_mp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                     nn.Dropout(doutrate),
                                     nn.Linear(hidden_dim, out_dim))
        
        self.num_layers = self.conv_layers + 1
        
    # Build convolutional blocks
    def build_conv_model(self, input_dim, hidden_dim):
        # If we perform node classification, we use simple graph convolution        
        return pyg_nn.GCNConv(input_dim, hidden_dim)
    
    

    def forward(self, data):
        # Data consists of: data.x (feature matrix), data.edge_index (adjecency matrix, what are the edges),
        # data.batch (which node belongs to which graph)
        # print(data)
        # breakpoint()
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # print("X is", x.shape)
        # print("edge is", edge_index.shape)
        # Sanity check:
        if data.num_node_features == 0:
            # if no features, we use constant feature
            x = torch.ones(data.num_nodes, 1)
            
        for i in range(self.num_layers):
            # Input to GCN is (node features (|V|, F_in), edge indices (2,E))
            # Output: node features (|V|, F_out)
            x = self.convs[i](x, edge_index)
            emb = x
            x = F.tanh(x)
            x = F.dropout(x, p = self.doutrate, training=self.training)

        x = self.post_mp(x)
        
        # Do log softmax for crossentropy
        # return embedding for visualization
        return emb, F.log_softmax(x, dim=1)
    
    def loss(self, pred, label):
        # Since we return log softmax we need to return negative log likelihood
        return F.nll_loss(pred, label)
    
    

def train(dataset, conv_layer, writer,  epochs, lr=0.01, hidden_layer = 32):
    test_loader = loader =  DataLoader(dataset, batch_size = 64, shuffle = True)

    # Build model
    model = GCN(max(dataset.num_node_features, 1), hidden_layer, dataset.num_classes, conv_layers=conv_layer)
    opt = optim.Adam(model.parameters(), lr = lr)
    
    test_accuracies = []

    print("#"*10, "Epoch: ", epochs, " Convs: ", conv_layer, " LR: ", lr, " Hidd. Lay: ", hidden_layer,  "#"*10)
    
    for epoch in range(0,epochs):
        total_loss = 0
        model.train()
        
        for batch in loader:
            # breakpoint()
            opt.zero_grad()
            embedding, pred = model(batch)
            label = batch.y
        
            pred = pred[batch.train_mask]
            label = label[batch.train_mask]
                
            loss = model.loss(pred, label)
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch.num_graphs
        total_loss /= len(loader.dataset)
        # tensorboard
        if writer is not None:
            writer.add_scalar("Loss", total_loss, epoch)
        
        if epoch % 10 == 0:
            test_acc = test(test_loader, model)
            test_accuracies.append(test_acc)
            print("Epoch {}. Loss {:.4f}. Test accuracy {:.4f}".format(epoch, total_loss, test_acc))
            model.best_accuracy = max(test_accuracies)
            print("best accuracy is", max(test_accuracies))
            
            if writer is not None:
                writer.add_scalar("test accuracy", test_acc, epoch)
            
    return model, model.best_accuracy


def test(loader, model, is_validation = False):
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


def visualization_nodembs(dataset, model):
    color_list = ["red", "orange", "green", "blue", "purple", "brown", "black"]
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    embs = []
    colors = []
    for batch in loader:
        # print("batch is", batch)
        emb, pred = model(batch)
        embs.append(emb)
        # for elem in batch.y:
            # print(elem)
        colors += [color_list[y] for y in batch.y]
    embs = torch.cat(embs, dim=0)

    xs, ys = zip(*TSNE().fit_transform(embs.detach().numpy()))
    # print("xs shape is", xs)
    # print("ys shape is", len(xs))
    
    plt.scatter(xs, ys, color=colors)
    plt.title(f"GCN, #epoch:{str(args.epoch)}, #conv:{str(args.conv)}\n accuracy:{model.best_accuracy*100}%")
    plt.show()
    
    
def hyperparameter_search():
    all_best_acc = float('-inf')
    all_best_lr = 0
    all_best_hidden = 0
    current_filename = os.path.abspath(__file__).split("/")[-1]
    
    configs = {}
    
    convs = [1,2,3,5,10,12,20,30]
    learning_rates = [0.1, 0.01, 10e-3, 10e-4, 10e-5]
    hidden_layers = [4, 8, 12, 24, 48, 64, 128]
    dataset = Planetoid(root='/tmp/PubMed', name = 'PubMed')
    for conv in convs:  
        
        all_best_acc = float('-inf')
        all_best_lr = 0
        all_best_hidden = 0
        
        for lr in learning_rates: 
            for lay in hidden_layers:
                
                model, best_accuracy = train(dataset, conv, writer=None, epochs=100, lr= lr, hidden_layer=lay)   
                
                if all_best_acc<best_accuracy:
                    all_best_lr = lr
                    all_best_hidden = lay
                    all_best_acc = best_accuracy
                    
        configs[conv] = {"LR": all_best_lr, "Hidden":all_best_hidden, "ACC": all_best_acc}
                    
    # Dump into a json file to be retrieved when testing
    with open(current_filename+'_config.json', 'w') as json_file:
        json.dump(configs, json_file, indent=4)
                    




### Flags Areas ###
import argparse
parser = argparse.ArgumentParser(description='Process some inputs.')
parser.add_argument('--epoch', type=int, help='Epoch Amount', default=100)
parser.add_argument('--conv', type=int, help='Conv Amount', default=3)

if __name__ == "__main__":

    hyperparameter_search()
    
 
# For report: 
# We changed from ReLU to tanh (as in their implementation)
# We use a dropout in training 
# We use two hidden layers atm

