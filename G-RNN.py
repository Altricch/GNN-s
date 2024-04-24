import torch
import torch.nn as nn
from torch_geometric.nn import GCN  # import any GNN -- we'll use GCN in this example
from torch_geometric.utils import from_networkx

# convert our networkx graph into a PyTorch Geometric (PyG) graph
pyg_graph = from_networkx(networkx_graph, group_node_attrs=NODE_ATTRIBUTE_NAMES)

# define our PyTorch model class
class Model(torch.nn.Module):
    def __init__(self, pyg_graph):
        super().__init__()
        self.graphX = pyg_graph.x
        self.graphEdgeIndex = pyg_graph.edge_index
        self.gnn = GCN(in_channels=NODE_FEATURE_SIZE, 
                       hidden_channels=GNN_HIDDEN_SIZE, 
                       num_layers=NUM_GNN_LAYERS, 
                       out_channels=NODE_EMBED_SIZE)
        self.batch_norm_lstm = nn.BatchNorm1d(SEQUENCE_PATH_LENGTH)
        self.batch_norm_linear = nn.BatchNorm1d(LSTM_HIDDEN_SIZE)
        self.lstm = nn.LSTM(input_size=NODE_EMBED_SIZE,
                            hidden_size=LSTM_HIDDEN_SIZE,
                            batch_first=True)
        self.pred_head = nn.Linear(LSTM_HIDDEN_SIZE, NUM_GRAPH_NODES)

    def forward(self, indices):
        node_emb = self.gnn(self.graphX, self.graphEdgeIndex)
        node_emb_with_padding = torch.cat([node_emb, torch.zeros((1, NODE_EMBED_SIZE))])
        paths = node_emb_with_padding[indices]
        paths = self.batch_norm_lstm(paths)
        _, (h_n, _) = self.lstm(paths)
        h_n = self.batch_norm_linear(torch.squeeze(h_n))
        predictions = self.pred_head(h_n)
        return F.log_softmax(predictions, dim=1)