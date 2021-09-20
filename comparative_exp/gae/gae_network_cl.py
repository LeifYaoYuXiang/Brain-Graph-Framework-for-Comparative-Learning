import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F


class NodeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        feat = self.linear(node.data['feat'])
        feat = self.activation(feat)
        return {'feat': feat}


gcn_msg = fn.copy_src(src='feat', out='m')
gcn_reduce = fn.sum(msg='m', out='feat')  # sum aggregation


class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)

    def forward(self, g, feature):
        g.ndata['feat'] = feature
        g.update_all(gcn_msg, gcn_reduce)
        g.apply_nodes(func=self.apply_mod)
        feat = g.ndata.pop('feat')
        return feat


class GAE_CL(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers,
                 node_each_graph):
        super(GAE_CL, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GCN(in_feats, n_hidden, lambda x:x))

        for i in range(n_layers - 1):
            if i != n_layers - 2:
                self.layers.append(GCN(n_hidden, n_hidden, F.relu))
            else:
                self.layers.append(GCN(n_hidden, n_hidden, lambda x: x))

        self.decoder = InnerProductDecoder(activation=lambda x: x)
        self.classifier = nn.Linear(node_each_graph * n_hidden, n_classes)

    def encode(self, g, feat):
        for conv in self.layers:
            feat = conv(g, feat)
        return feat

    def forward(self, input):
        features = input[0]
        graph = input[1]
        batch_size = input[2]
        embedding = self.encode(graph, features)
        embedding = embedding.reshape(batch_size, -1)
        logits = self.classifier(embedding)
        return logits


class InnerProductDecoder(nn.Module):
    def __init__(self, activation=torch.sigmoid, dropout=0.1):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.activation = activation

    def forward(self, z):
        z = F.dropout(z, self.dropout)
        adj = self.activation(torch.mm(z, z.t()))
        return adj