
# from paper: GATED GRAPH SEQUENCE NEURAL NETWORKS

import os
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
import torch_geometric
from torch_geometric.datasets import Planetoid, TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn.inits import uniform
from torch_geometric.nn.conv import MessagePassing
from torch.nn import Parameter as Param
from torch import Tensor
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import time
from datetime import datetime
torch.manual_seed(42)

#region Gated Graph Conv
#BASE Class for information propagation and update amongst nodes
class GatedGraphConv(MessagePassing):

    def __init__(self, out_channels, num_layers, aggr = 'add',
                 bias = True, **kwargs):
        super(GatedGraphConv, self).__init__(aggr=aggr, **kwargs)

        self.out_channels = out_channels
        self.num_layers = num_layers

        self.rnn = torch.nn.GRUCell(out_channels, out_channels, bias=bias)
        self.weight = Param(Tensor(num_layers, out_channels, out_channels))
        #self.rnn = torch.nn.GRUCell(out_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.out_channels, self.weight)
        self.rnn.reset_parameters()

    def forward(self, x):
        """"""
        #x = data.x
        edge_index = data.edge_index
        edge_weight = data.edge_attr

        #breakpoint()

        if x.size(-1) > self.out_channels:
            raise ValueError('The number of input channels is not allowed to '
                             'be larger than the number of output channels')

        if x.size(-1) < self.out_channels:
            zero = x.new_zeros(x.size(0), self.out_channels - x.size(-1))
            x = torch.cat([x, zero], dim=1)

        for i in range(self.num_layers):
            m = torch.matmul(x, self.weight[i])

            #breakpoint()
            m = self.propagate(edge_index, x=m, edge_weight=edge_weight,
                               size=None)
            x = self.rnn(m, x)

        return x

    #def message(self, x_j, edge_weight):
    #    return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    # def message_and_aggregate(self, adj_t, x):
    #     return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, num_layers={})'.format(self.__class__.__name__,
                                              self.out_channels,
                                              self.num_layers)
#endregion
#region MLP
class MLP(nn.Module):
    def __init__(self, input_dim, hid_dims):
        super(MLP, self).__init__()

        self.mlp = nn.Sequential()
        dims = [input_dim] + hid_dims
        for i in range(len(dims)-1):
            self.mlp.add_module('lay_{}'.format(i),nn.Linear(in_features=dims[i], out_features=dims[i+1]))
            if i+1 < len(dims):
                self.mlp.add_module('act_{}'.format(i), nn.Tanh())

    def reset_parameters(self):
        for i, l in enumerate(self.mlp):
            if type(l) == nn.Linear:
                nn.init.xavier_normal_(l.weight)

    def forward(self, x):
        return self.mlp(x)
#endregion
#region GGNN
class GGNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels=None, num_conv=3, hidden_dim = 32, aggr = 'add', mlp_hdim=32, mlp_hlayers=3):
        super(GGNN, self).__init__()
        self.emb = lambda x : x 
        self.best_acc = -1

        if out_channels is None:
            out_channels = in_channels

        if in_channels != out_channels:
            print("mismatch, operating a reduction")
            self.emb = nn.Linear(in_channels, hidden_dim, bias=False)

        self.conv = GatedGraphConv(out_channels=out_channels,
                                   num_layers=num_conv)

        self.mlp = MLP(input_dim=out_channels,
                       hid_dims=[mlp_hdim]*mlp_hlayers)
        
        self.out_layer = nn.Linear(mlp_hdim, dataset.num_classes)


    def forward(self, data):
        #print("0, x forward shape:", data.x.shape)
        x = self.emb(data.x) # Linear reduction
        #print("1, x after emb shape:", x.shape)
        x = self.conv(x)
        #print("2, x after conv shape:", x.shape)
        x = self.mlp(x)
        #print("3, x after mlp shape:", x.shape)
        x_emb = self.out_layer(x)
        #print("4, x after out shape:", x_emb.shape)
        #x_emb
        return x_emb, F.log_softmax(x_emb, dim=-1)
#endregion 

def train(dataset, epochs=100, num_conv=3, learning_rate=0.001):
    ###### SETUP ######
    start_time = time.time()
    test_loader = loader = DataLoader(dataset, batch_size=1, shuffle=True)
    model = GGNN(in_channels=dataset.x.shape[-1], out_channels=32, num_conv=num_conv).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    tot_loss = 0
    best_acc = [0,0,0]

    print("#"*100+"\n")
    print("[MODEL REPR]", repr(model))
    print("#" * 20 + f" Running Gated GNN, with {str(epochs)} epochs, {str(num_conv)} convs " + "#" * 20)
    ####################

    for epoch in range(1, epochs+1):
        print(f'Processing Epoch {epoch}', end="\r")
        epoch_start_time = datetime.now()
        model.train()

        for batch in loader:
            optimizer.zero_grad()

            emb_x, pred = model(batch)

            pred = pred[batch.train_mask]
            label = batch.y[batch.train_mask]

            loss = loss_fn(pred, label)
            loss_fn(pred, label).backward()

            optimizer.step()
            tot_loss += loss.item() # TODO verify need for batch.num_graphs

        tot_loss /= len(loader.dataset)
            ###

        ### Test
        # MODIFICATION, evaluation now every 10 steps rather than natively 1to1
        if epoch % 10 == 0:
            model.eval()
            for data in test_loader:
                with torch.no_grad():
                    accs = []
                    emb, logits = model(data)
                    masks = [data.train_mask, data.val_mask, data.test_mask]
                    for mask in masks:
                        pred = logits[mask].max(1)[1]
                        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
                        accs.append(acc)

                train_acc = accs[0]
                val_acc = accs[1]
                test_acc = accs[2]

                best_acc[0] = max(best_acc[0],train_acc)
                best_acc[1] = max(best_acc[1],val_acc)
                best_acc[2] = max(best_acc[2],test_acc)

                model.best_acc = best_acc[2]

            print('Epoch: {:03d}, Train Acc: {:.0%}, '
                  'Val Acc: {:.0%}, Test Acc: {:.0%}, time for 10 epochs {:.2f}'.format(epoch, train_acc,
                                                             val_acc, test_acc, epoch_start_time))

    print("Training Completed in {:.2f} seconds".format(time.time() - start_time))
    print("Best Accuracies Train Acc: {:.0%}, Val Acc: {:.0%}, Test Acc: {:.0%}".format(best_acc[0], best_acc[1],  best_acc[2]))

    return model

#region Visualisation
def visualization_nodembs(dataset, model):
    color_list = ["red", "orange", "green", "blue", "purple", "brown", "black"]
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    embs = []
    colors = []
    for batch in loader:
        print("batch is", batch)
        emb, pred = model(batch)
        embs.append(emb)

        colors += [color_list[y] for y in batch.y]
    embs = torch.cat(embs, dim=0)

    xs, ys = zip(*TSNE(random_state=42).fit_transform(embs.detach().numpy()))

    plt.scatter(xs, ys, color=colors)
    plt.title(
        f"GGNN, #epoch:{str(args.epoch)}, #conv:{str(args.conv)}\n accuracy:{model.best_acc*100}%"
    )
    plt.show()
#endregion

### Flags Areas ###
import argparse
parser = argparse.ArgumentParser(description='Process some inputs.')
parser.add_argument('--epoch', type=int, help='Epoch Amount', default=100)
parser.add_argument('--conv', type=int, help='Conv Amount', default=3)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = "cpu"

dataset = 'PubMed'
transform = T.Compose([T.TargetIndegree(),
])
path = osp.join('data', dataset)
dataset = Planetoid(path, dataset, transform=transform)

data = dataset[0]

print("[DATA],", data)

if __name__ == '__main__':

    test_dataset = dataset[:len(dataset) // 10]
    train_dataset = dataset[len(dataset) // 10:]
    test_loader = DataLoader(test_dataset)
    train_loader = DataLoader(train_dataset)

    args = parser.parse_args()

    epochs = args.epoch
    convs = args.conv

    model = train(dataset, epochs=epochs, num_conv=convs, learning_rate=0.001)
    model.__repr__()
    visualization_nodembs(dataset, model)

    #TODO:
    # ADD LINER REDUCTION to 32 [DONE]
    # ADD droput to prevent Overfitting