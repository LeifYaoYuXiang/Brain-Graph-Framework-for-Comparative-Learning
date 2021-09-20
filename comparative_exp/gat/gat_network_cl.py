import torch.nn as nn
from dgl.nn.pytorch.conv.gatconv import GATConv
from torch.nn import Linear


class GAT_CL(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers,
                 node_each_graph,
                 heads, activation, feat_drop, attn_drop, dropout, negative_slope, residual):
        heads = [64, 64, 64, 64, 1]
        super(GAT_CL, self).__init__()
        self.layers = nn.ModuleList()
        self.node_each_graph = node_each_graph
        self.n_hidden = n_hidden
        self.layers.append(GATConv(in_feats, n_hidden, heads[0], feat_drop, attn_drop, negative_slope, False, activation))
        for i in range(n_layers-1):
            self.layers.append(
                GATConv(n_hidden * heads[i], n_hidden, heads[i+1], feat_drop, attn_drop, negative_slope, residual, activation)
            )

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
            features = features.flatten(1)
        features = features.reshape(batch_size, self.node_each_graph * self.n_hidden)
        features = self.classifier(features)
        return features
