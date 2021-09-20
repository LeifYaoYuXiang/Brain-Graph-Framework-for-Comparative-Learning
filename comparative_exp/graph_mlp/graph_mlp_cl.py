import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Dropout, Linear, LayerNorm


class MLP(nn.Module):
    def __init__(self, input_dim, hid_dim, dropout):
        super(MLP, self).__init__()
        self.fc1 = Linear(input_dim, hid_dim)
        self.fc2 = Linear(hid_dim, hid_dim)
        self.act_fn = torch.nn.functional.gelu
        self._init_weights()

        self.dropout = Dropout(dropout)
        self.layernorm = LayerNorm(hid_dim, eps=1e-6)

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.layernorm(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def get_feature_dis(x):
    x_dis = x@x.T
    mask = torch.eye(x_dis.shape[0])
    x_sum = torch.sum(x**2, 1).reshape(-1, 1)
    x_sum = torch.sqrt(x_sum).reshape(-1, 1)
    x_sum = x_sum @ x_sum.T
    x_dis = x_dis*(x_sum**(-1))
    x_dis = (1-mask) * x_dis
    return x_dis


class GMLP_CL(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes,
                 node_each_graph,
                 dropout):
        super(GMLP_CL, self).__init__()
        self.n_hidden = n_hidden
        self.node_each_graph = node_each_graph
        self.mlp = MLP(in_feats, self.n_hidden, dropout)
        self.classifier = Linear(self.node_each_graph * self.n_hidden, n_classes)

    def forward(self, input):
        features = input[0]
        batch_size = input[2]

        # features_temp = features

        features = self.mlp(features)
        features_temp = features

        if self.training:
            features_dis = get_feature_dis(features_temp)

        features = features.reshape(batch_size, self.node_each_graph * self.n_hidden)
        class_feature = self.classifier(features)

        class_logits = F.log_softmax(class_feature, dim=1)

        if self.training:
            return class_logits, features_dis
        else:
            return class_logits