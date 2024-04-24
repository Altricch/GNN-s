#https://medium.com/stanford-cs224w/wikinet-an-experiment-in-recurrent-graph-neural-networks-3f149676fbf3
#https://colab.research.google.com/drive/1geXYFIopoh7W4bLFrICqdkh1HuVySLQn?usp=sharing

import torch
import torch.nn as nn
from torch_geometric.nn import GCN  # import any GNN -- we'll use GCN in this example
from torch_geometric.utils import from_networkx
import torch.optim as optim
from datetime import datetime

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

import torch.functional as F
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

# convert our networkx graph into a PyTorch Geometric (PyG) graph
#pyg_graph = from_networkx(networkx_graph, group_node_attrs=NODE_ATTRIBUTE_NAMES)

# define our PyTorch model class
class GRNN(nn.Module):
    #def __init__(self, pyg_graph):   
    def __init__(self, data, sequence_path_length=32, gnn_hidden_size=128, node_embed_size=64, lstm_hidden_size=32, conv_l=3):
        super().__init__()
        self.gnn = GCN(in_channels=data.x.shape[1], #num features, 
                       hidden_channels=gnn_hidden_size, 
                       num_layers=conv_l, 
                       out_channels=node_embed_size)
        self.batch_norm_lstm = nn.BatchNorm1d(sequence_path_length)
        self.batch_norm_linear = nn.BatchNorm1d(lstm_hidden_size)
        self.lstm_in_size = node_embed_size



        self.lstm = nn.LSTM(input_size=node_embed_size,
                            hidden_size=lstm_hidden_size,
                            batch_first=True)
        self.pred_head = nn.Linear(lstm_hidden_size, data.x.shape[0]) #num nodes)

    def forward(self, data, indices):
        print("merdone nel forward")
        breakpoint()
        print(data.size())
        print(data)

        x , edge_index = data.x, data.edge_index

        node_emb = self.gnn(x, edge_index)
        node_emb_with_padding = torch.cat([node_emb, torch.zeros((1, self.lstm_in_size))])
        paths = node_emb_with_padding[indices]
        paths = self.batch_norm_lstm(paths)
        _, (h_n, _) = self.lstm(paths)
        h_n = self.batch_norm_linear(torch.squeeze(h_n))
        predictions = self.pred_head(h_n)
        return F.log_softmax(predictions, dim=1)
    
def train(dataset, conv_layer, writer,  epochs):
    test_loader = loader =  DataLoader(dataset, batch_size = 64, shuffle = True)

    # Build model
    model = GRNN(data=dataset.data, 
                sequence_path_length=32, 
                gnn_hidden_size=128, 
                node_embed_size=64,
                lstm_hidden_size=32,
                conv_l=3)
    opt = optim.Adam(model.parameters(), lr = 0.01)
    
    test_accuracies = []

    print("#"*20 + " Running GRNN, with " + str(epochs) + " epochs " + "and "+ str(conv_layer)+" convs " +"#"*20)
    
    for epoch in range(0,epochs):
        total_loss = 0
        model.train()
        
        for batch in loader:
            # breakpoint()
            opt.zero_grad()
            embedding, pred = model(batch, np.arange(0,100))
            label = batch.y
        
            pred = pred[batch.train_mask]
            label = label[batch.train_mask]
                
            loss = model.loss(pred, label)
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch.num_graphs
        total_loss /= len(loader.dataset)
        # tensorboard
        writer.add_scalar("Loss", total_loss, epoch)
        
        if epoch % 10 == 0:
            test_acc = test(test_loader, model)
            test_accuracies.append(test_acc)
            print("Epoch {}. Loss {:.4f}. Test accuracy {:.4f}".format(epoch, total_loss, test_acc))
            model.best_accuracy = max(test_accuracies)
            print("best accuracy is", max(test_accuracies))
            writer.add_scalar("test accuracy", test_acc, epoch)
            
    return model

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




### Flags Areas ###
import argparse
parser = argparse.ArgumentParser(description='Process some inputs.')
parser.add_argument('--epoch', type=int, help='Epoch Amount', default=100)
parser.add_argument('--conv', type=int, help='Conv Amount', default=3)
parser.add_argument('--asym', type=bool, help='Use AntiSymmetric Weights', default=1)


if __name__ == '__main__':

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    writer = SummaryWriter("./PubMed/" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    dataset = Planetoid(root='/tmp/PubMed', name = 'PubMed')
    
    epochs = args.epoch
    conv_layer = args.conv

    print("li mortacci tua: ", dataset.data)

    model = train(dataset.data, conv_layer, writer, epochs)   
    
