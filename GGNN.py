# Gated Graph nn
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

torch.manual_seed(42)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = "cpu"

dataset = 'PubMed'
transform = T.Compose([T.TargetIndegree(),
])
path = osp.join('data', dataset)
dataset = Planetoid(path, dataset, transform=transform)

#dataset = Planetoid(root='/tmp/PubMed', name = 'PubMed')
data = dataset[0]

print(data)

class MLP(nn.Module):
    def __init__(self, input_dim, hid_dims, out_dim):
        super(MLP, self).__init__()

        self.mlp = nn.Sequential()
        dims = [input_dim] + hid_dims + [out_dim]
        for i in range(len(dims)-1):
            self.mlp.add_module('lay_{}'.format(i),nn.Linear(in_features=dims[i], out_features=dims[i+1]))
            if i+2 < len(dims):
                self.mlp.add_module('act_{}'.format(i), nn.Tanh())
    def reset_parameters(self):
        for i, l in enumerate(self.mlp):
            if type(l) == nn.Linear:
                nn.init.xavier_normal_(l.weight)

    def forward(self, x):
        return self.mlp(x)


#BASE Class for information propagation and update amongst nodes
class GatedGraphConv(MessagePassing):

    def __init__(self, out_channels, num_layers, aggr = 'add',
                 bias = True, **kwargs):
        super(GatedGraphConv, self).__init__(aggr=aggr, **kwargs)

        self.out_channels = out_channels
        self.num_layers = num_layers

        self.weight = Param(Tensor(num_layers, out_channels, out_channels))
        self.rnn = torch.nn.GRUCell(out_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.out_channels, self.weight)
        self.rnn.reset_parameters()

    def forward(self, data):
        """"""
        x = data.x

        edge_index = data.edge_index
        edge_weight = data.edge_attr
        if x.size(-1) > self.out_channels:
            raise ValueError('The number of input channels is not allowed to '
                             'be larger than the number of output channels')

        if x.size(-1) < self.out_channels:
            zero = x.new_zeros(x.size(0), self.out_channels - x.size(-1))
            x = torch.cat([x, zero], dim=1)

        for i in range(self.num_layers):
            m = torch.matmul(x, self.weight[i])
            m = self.propagate(edge_index, x=m, edge_weight=edge_weight,
                               size=None)
            x = self.rnn(m, x)

        return x

    def message(self, x_j, edge_weight):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    # def message_and_aggregate(self, adj_t, x):
    #     return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, num_layers={})'.format(self.__class__.__name__,
                                              self.out_channels,
                                              self.num_layers)

class GGNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_conv=3, hidden_dim = 32, aggr = 'add'):
        super(GGNN, self).__init__()

        self.conv = GatedGraphConv(out_channels=out_channels,
                                   num_layers=num_conv)

        self.mlp = MLP(input_dim= in_channels,
                       hid_dims= [32,32,32],
                       out_dim= dataset.num_classes)

    def forward(self):
        x = self.conv(data)
        x = self.mlp(x)
        return F.log_softmax(x, dim=-1)

# def train():
#     model.train()
#     optimizer.zero_grad()
#     loss_fn(model()[data.train_mask], data.y[data.train_mask]).backward()
#     optimizer.step()
#
# def test(model):
#     model.eval()
#     logits, accs = model(), []
#     for _, mask in data('train_mask', 'val_mask', 'test_mask'):
#         pred = logits[mask].max(1)[1]
#         acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
#         accs.append(acc)
#     return accs

def train(dataset, epochs=100):
    start_time = time.time()
    test_loader = loader = DataLoader(dataset, batch_size=1, shuffle=True)
    model = GGNN(in_channels=500, out_channels=500).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    tot_loss = 0

    best_acc = [0,0,0]

    print("#" * 20 + f" Running Gated GNN, with {str(epochs)} epochs, {None} convs" + "#" * 20)

    for epoch in range(1, epochs+1):
        print(f'Processing Epoch {epoch}', end="\r")
        epoch_start_time = time.time()
        ### Train
        model.train()

        for batch in loader:

            optimizer.zero_grad()

            pred = model()[batch.train_mask]
            label = batch.y[batch.train_mask]

            loss = loss_fn(pred, label)
            loss_fn(pred, label).backward()

            optimizer.step()
            tot_loss += loss.item() # TODO verify need for batch.num_graphs

        tot_loss /= len(loader.dataset)
            ###

        ### Test
        #MODIFICATION, evaluation now every 10 steps rather than natively 1to1
        if epoch % 10 == 0:
            model.eval()
            logits, accs = model(), []
            #for _, mask in data('train_mask', 'val_mask', 'test_mask'):

            for data in test_loader:
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

            print('Epoch: {:03d}, Train Acc: {:.0%}, '
                  'Val Acc: {:.0%}, Test Acc: {:.0%}, time for 10 epochs {:.2f}'.format(epoch, train_acc,
                                                             val_acc, test_acc, time.time() - epoch_start_time))

    print("Training Completed in {:.2f} seconds".format(time.time() - start_time))
    print("Best Accuracies Train Acc: {:.0%}, Val Acc: {:.0%}, Test Acc: {:.0%}".format(best_acc[0], best_acc[1],  best_acc[2]))

    return model

def visualization_nodembs(dataset, model):
    color_list = ["red", "orange", "green", "blue", "purple", "brown", "black"]
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    embs = []
    colors = []
    for batch in loader:
        print("batch is", batch)
        emb, pred = model(batch)
        # print(emb.shape)
        embs.append(emb)

        # for elem in batch.y:
        # print(elem)
        colors += [color_list[y] for y in batch.y]
    embs = torch.cat(embs, dim=0)

    xs, ys = zip(*TSNE(random_state=42).fit_transform(embs.detach().numpy()))

    plt.scatter(xs, ys, color=colors)
    plt.title( "swag")
        #f"ADGN, #epoch:{str(args.epoch)}, #conv:{str(args.conv)}\n accuracy:{model.best_accuracy * 100}%, symmetry {model.antisymmetry}")
    plt.show()


### Flags Areas ###
import argparse
parser = argparse.ArgumentParser(description='Process some inputs.')
parser.add_argument('--epoch', type=int, help='Epoch Amount', default=100)
parser.add_argument('--conv', type=int, help='Conv Amount', default=3)


if __name__ == '__main__':

    test_dataset = dataset[:len(dataset) // 10]
    train_dataset = dataset[len(dataset) // 10:]
    test_loader = DataLoader(test_dataset)
    train_loader = DataLoader(train_dataset)

    args = parser.parse_args()

    epochs = args.epoch
    model = train(dataset, epochs=epochs)

    model.__repr__()
    #visualization_nodembs(dataset, model)