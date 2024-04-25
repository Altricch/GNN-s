# Gated Graph nn
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
from torch.nn import Parameter as Param
from torch import Tensor

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = "cpu"
from torch_geometric.nn.conv import MessagePassing

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
    def __init__(self):
        super(GGNN, self).__init__()

        self.conv = GatedGraphConv(1433, 3)
        self.mlp = MLP(1433, [32,32,32], dataset.num_classes)

    def forward(self):
        x = self.conv(data)
        x = self.mlp(x)
        return F.log_softmax(x, dim=-1)


def train():
    model.train()
    optimizer.zero_grad()
    loss_fn(model()[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()


def test(): 
    model.eval()
    logits, accs = model(), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


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
    # print("xs shape is", xs)
    # print("ys shape is", len(xs))

    # max_distance = calculate_diameter(xs, ys)
    # eccentricity = calculate_eccentricity(xs, ys)

    # print(f"Max distance: {max_distance}")
    # print(f"Eccentricity: {eccentricity}")

    plt.scatter(xs, ys, color=colors)
    plt.title( "swag")
        #f"ADGN, #epoch:{str(args.epoch)}, #conv:{str(args.conv)}\n accuracy:{model.best_accuracy * 100}%, symmetry {model.antisymmetry}")
    plt.show()




if __name__ == '__main__':
    model = GGNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    test_dataset = dataset[:len(dataset) // 10]
    train_dataset = dataset[len(dataset) // 10:]
    test_loader = DataLoader(test_dataset)
    train_loader = DataLoader(train_dataset)

    for epoch in range(1, 2):
        train()
        accs = test()
        train_acc = accs[0]
        val_acc = accs[1]
        test_acc = accs[2]
        print('Epoch: {:03d}, Train Acc: {:.5f}, '
              'Val Acc: {:.5f}, Test Acc: {:.5f}'.format(epoch, train_acc,
                                                         val_acc, test_acc))

    visualization_nodembs(dataset, model)