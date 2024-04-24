# https://www.youtube.com/watch?v=-UjytpbqX4A&t=2221s

# !pip install torch-scatter
# !pip install torch-cluster
# !pip install torch-sparse
# !pip install torch-geometric
# !pip install tensorboardX

import numpy as np
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


torch.manual_seed(42)
np.random.seed(42)

class ADGNConv(pyg_nn.MessagePassing):
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 gamma: float = 0.1, 
                 epsilon : float = 0.1, 
                 antisymmetry = True,
                 ):
        super(ADGNConv, self).__init__(aggr='add') # "Add" aggregation (can alternatively use mean or max)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.gamma = gamma
        self.epsilon = epsilon
        self.act_func = nn.Tanh()  
        self.antisymmetry = antisymmetry   
        self.best_accuracy = -1 
        
        # Defines learnable weighs and biases, where W has (nxn) and bias is (n)
        self.Weights = nn.Parameter(torch.zeros((self.in_channels, self.in_channels)), requires_grad=True)
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

        # # Conv block
        # self.conv = pyg_nn.GCNConv(in_channels, in_channels, bias=False)
        
    def forward(self, x, edge_index):
        # Antisymmetric formulation (paper formula 5)
        W = ((self.Weights - self.Weights.T) - (self.gamma * self.Identity)) if self.antisymmetry else self.Weights
        
        # Convolution of neighbors of previous layer PHI*(X(l-1), N_u)
        # Do forward pass for backpropp to learn weights of Linear Layer
        aggr_x = self.linear(x)
        # breakpoint()
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        row, col = edge_index
        # breakpoint()
        deg = pyg_utils.degree(row, aggr_x.size()[0])
        deg_inv_sqrt = deg.pow(-0.5)
        # Formula 7 of paper
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        aggr_x = self.propagate(edge_index, x = aggr_x, norm=norm)
        
        
        # Store previous x
        x_prev = x
        
        # Apply function of paper in the forward pass
        x = (x_prev @ W + aggr_x + self.bias)
        x = self.epsilon*(self.act_func(x))
        x = x_prev + x
        
        return x
    
    def message(self, x_j, norm):
        # Compute messages
        # x_j has shape [E, outchannels]
        
        
        
        # Here we add outselves back in
        # breakpoint()
        return norm.view(-1, 1) * x_j

class ADGN(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels, num_layers, epsilon = 0.1, gamma = 0.1, antisymmetric = True):
        super(ADGN, self).__init__()
        
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.epsilon = epsilon
        self.gamma = gamma
        self.antisymmetric = antisymmetric
        
        
        self.emb = None
        if self.hidden_dim is not None:
            self.emb = nn.Linear(self.in_channels, hidden_dim, bias=False)
        
        self.conv = nn.ModuleList()
        
        # Apply hidden dimensions in conv block
        for _ in range(1, num_layers):
            self.conv.append((ADGNConv(
                in_channels=self.hidden_dim,
                out_channels=self.hidden_dim,
            )))
        
        self.linear = nn.Linear(self.hidden_dim, self.out_channels)

    def forward(self, x):
        x, edge_idx = x.x, x.edge_index, 
        
        x = self.emb(x)
        
        for conv in self.conv:
            x = conv(x, edge_idx)
            emb = x
        
        x = self.linear(x)
        # print("X is", x.)
        return emb, x
    
    
def visualization_nodembs(dataset, model):
    color_list = ["red", "orange", "green", "blue", "purple", "brown", "black"]
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    embs = []
    colors = []
    for batch in loader:
        # print("batch is", batch)
        emb, pred = model(batch)
        # print(emb.shape)
        embs.append(emb)
        
        # for elem in batch.y:
            # print(elem)
        colors += [color_list[y] for y in batch.y]
    embs = torch.cat(embs, dim=0)

    xs, ys = zip(*TSNE(random_state=42).fit_transform(embs.detach().numpy()))
    # print("xs shape is", xs)
    # print("ys shape is", len(xs))
    
    # max_distance = calculate_diameter(xs, ys)
    # eccentricity = calculate_eccentricity(xs, ys)
    
    # print(f"Max distance: {max_distance}")
    # print(f"Eccentricity: {eccentricity}")
    
    plt.scatter(xs, ys, color=colors)
    plt.title(f"ADGN, #epoch:{str(args.epoch)}, #conv:{str(args.conv)}\n accuracy:{model.best_accuracy*100}%, symmetry {model.antisymmetry}")
    plt.show()    

def train(dataset, conv_layer, writer,  epochs, anti_symmetric =True):
    test_loader = loader =  DataLoader(dataset, batch_size = 1, shuffle = True)
    # print("LOADER IS", loader)

    # Build model
    # self, in_channels, hidden_dim, out_channels, num_layers
    model = ADGN(max(dataset.num_node_features, 1), 32, dataset.num_classes, num_layers=conv_layer, antisymmetric=anti_symmetric)
    opt = optim.Adam(model.parameters(), lr = 0.01)
    loss_fn = nn.CrossEntropyLoss()
    test_accuracies = []
    
    print("#"*20 + f" Running ADGN, with {str(epochs)} epochs, {str(conv_layer)} convs and antisymmetric {model.antisymmetric} " + "#"*20)

    for epoch in range(0,epochs):
        total_loss = 0
        model.train()
        
        for batch in loader:
            # print(model(batch)[0])
            # breakpoint()
            opt.zero_grad()
            emb, pred = model(batch)
            label = batch.y
            
            # print("batch mask", np.where(batch.train_mask == True))
            # print(pred[batch.train_mask])
            
            pred = pred[batch.train_mask]
            label = label[batch.train_mask]
                
            loss = loss_fn(pred, label)
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
            emb, pred = model(data)
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

    #args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    writer = SummaryWriter("./PubMed/" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    dataset = Planetoid(root='/tmp/PubMed', name = 'PubMed')
    #model = ADGN(max(dataset.num_node_features, 1), 32, dataset.num_classes, 2, antisymmetric=args.asym)
    # print(model)
    # emb,x = dataset
    # print(model(x).shape)

    args = parser.parse_args()
    
    epochs = args.epoch
    conv_layer = args.conv
    antisymmetry = True if args.asym == 1 else False 
    print(antisymmetry)
    model = train(dataset, conv_layer, writer, epochs, anti_symmetric = antisymmetry)   
    visualization_nodembs(dataset, model)
    




#TODO: Why multiple times per hidden layer (e.g. num_iterations not clear)
        
        
        