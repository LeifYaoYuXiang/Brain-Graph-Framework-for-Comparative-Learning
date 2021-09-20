from dgl.nn.pytorch import GraphConv
import torch
import torch.nn as nn
import torch.nn.functional as F


class VGAEModel_CL(nn.Module):
    def __init__(self, in_feats, n_hidden_1, n_hidden_2, n_classes,
                 node_each_graph):
        super(VGAEModel_CL, self).__init__()
        self.in_dim = in_feats
        self.hidden1_dim = n_hidden_1
        self.hidden2_dim = n_hidden_2
        self.node_each_graph = node_each_graph

        layers = [GraphConv(self.in_dim, self.hidden1_dim, activation=F.relu, allow_zero_in_degree=True),
                  GraphConv(self.hidden1_dim, self.hidden2_dim, activation=lambda x: x, allow_zero_in_degree=True),
                  GraphConv(self.hidden1_dim, self.hidden2_dim, activation=lambda x: x, allow_zero_in_degree=True)]

        self.layers = nn.ModuleList(layers)
        self.classifier = nn.Linear(self.node_each_graph*self.hidden2_dim, n_classes)

    def encoder(self, g, features):
        h = self.layers[0](g, features)
        self.mean = self.layers[1](g, h)
        self.log_std = self.layers[2](g, h)
        gaussian_noise = torch.randn(features.size(0), self.hidden2_dim)
        sampled_z = self.mean + gaussian_noise * torch.exp(self.log_std)
        return sampled_z

    def decoder(self, z):
        adj_rec = torch.sigmoid(torch.matmul(z, z.t()))
        return adj_rec

    def forward(self, input):
        features = input[0]
        graph = input[1]
        batch_size = input[2]
        embedding = self.encoder(graph, features)
        embedding = embedding.reshape(batch_size, -1)
        logits = self.classifier(embedding)
        return logits