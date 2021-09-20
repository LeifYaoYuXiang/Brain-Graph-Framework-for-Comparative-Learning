import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GINConv
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling


class ApplyNodeFunc(nn.Module):
    def __init__(self, mlp):
        super(ApplyNodeFunc, self).__init__()
        self.mlp = mlp
        self.bn = nn.BatchNorm1d(self.mlp.output_dim)

    def forward(self, h):
        h = self.mlp(h)
        h = self.bn(h)
        h = F.relu(h)
        return h


class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.output_dim = output_dim

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            return self.linear(x)
        else:
            h = x
            for i in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[i](self.linears[i](h)))
            return self.linears[-1](h)


class GIN_CL(nn.Module):
    def __init__(self, n_layers, n_mlp_layers,
                 in_feats, n_hidden, n_classes,
                 node_each_graph,
                 final_dropout, learn_eps, graph_pooling_type,
                 neighbor_pooling_type):
        super(GIN_CL, self).__init__()
        self.n_layers = n_layers
        self.learn_eps = learn_eps

        self.ginlayers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.n_layers - 1):
            if layer == 0:
                mlp = MLP(n_mlp_layers, in_feats, n_hidden, n_hidden)
            else:
                mlp = MLP(n_mlp_layers, n_hidden, n_hidden, n_hidden)
            self.ginlayers.append(GINConv(ApplyNodeFunc(mlp), neighbor_pooling_type, 0, self.learn_eps))
            self.batch_norms.append(nn.BatchNorm1d(n_hidden))
        self.linears_prediction = torch.nn.ModuleList()

        for layer in range(n_layers):
            if layer == 0:
                self.linears_prediction.append(nn.Linear(node_each_graph*node_each_graph, n_classes))
            else:
                self.linears_prediction.append(nn.Linear(node_each_graph*n_hidden, n_classes))
        self.drop = nn.Dropout(final_dropout)
        if graph_pooling_type == 'sum':
            self.pool = SumPooling()
        elif graph_pooling_type == 'mean':
            self.pool = AvgPooling()
        elif graph_pooling_type == 'max':
            self.pool = MaxPooling()
        else:
            raise NotImplementedError

    def forward(self, input):
        h = input[0]
        g = input[1]
        batch_size = input[2]
        hidden_rep = [h]

        for i in range(self.n_layers - 1):
            h = self.ginlayers[i](g, h)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            hidden_rep.append(h)
        score_over_layer = 0

        for i, h in enumerate(hidden_rep):
            pooled_h = h.reshape(batch_size, -1)
            predictions = self.linears_prediction[i](pooled_h)
            score_over_layer += self.drop(predictions)
        return score_over_layer
