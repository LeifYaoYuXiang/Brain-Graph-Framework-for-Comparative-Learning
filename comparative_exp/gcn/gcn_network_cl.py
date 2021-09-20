from dgl.nn.pytorch import GraphConv
import torch.nn as nn
from torch.nn import Linear


class GCN_CL(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers,
                 node_each_graph,
                 activation, dropout):
        super(GCN_CL, self).__init__()
        self.layers = nn.ModuleList()
        self.node_each_graph = node_each_graph
        self.n_hidden = n_hidden
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))

        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))

        self.classifier = Linear(self.node_each_graph * self.n_hidden, n_classes)
        self.dropout = nn.Dropout(p=dropout)


    def forward(self, input):
        features = input[0]
        graph = input[1]
        batch_size = input[2]
        for i, layer in enumerate(self.layers):
            if i != 0:
                features = self.dropout(features)
            features = layer(graph, features)
        features = features.reshape(batch_size, self.node_each_graph * self.n_hidden)
        features = self.classifier(features)
        return features

