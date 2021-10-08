from dgl.nn.pytorch import GraphConv
from torch.nn import Linear
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GINConv
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling

# Encoder 选择一：GCN
from comparative_exp.graph_mlp.graph_mlp_cl import get_feature_dis


class GCN(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers,
                 node_each_graph,
                 activation, dropout):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.node_each_graph = node_each_graph
        self.n_hidden = n_hidden
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
        self.read_out_layer = Linear(in_features=n_hidden, out_features=1, bias=False)

        self.classifier = Linear(self.node_each_graph * self.n_hidden, n_classes)

        self.proj_head = nn.Sequential(
            nn.Linear(self.node_each_graph * self.n_hidden, self.node_each_graph * self.n_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.node_each_graph * self.n_hidden, self.node_each_graph * self.n_hidden)
        )
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

    def get_embedding(self, input):
        features = input[0]
        graph = input[1]
        batch_size = input[2]
        for i, layer in enumerate(self.layers):
            if i != 0:
                features = self.dropout(features)
            features = layer(graph, features)
        features = features.reshape(batch_size, self.node_each_graph * self.n_hidden)
        return features


# Encoder 选择二：GIN
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


class GIN(nn.Module):
    def __init__(self, n_layers, n_mlp_layers,
                 in_feats, n_hidden, n_classes,
                 node_each_graph,
                 final_dropout, learn_eps, graph_pooling_type,
                 neighbor_pooling_type):
        super(GIN, self).__init__()
        self.n_layers = n_layers
        self.learn_eps = learn_eps
        self.node_each_graph = node_each_graph
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

        if not self.training:
            for i, h in enumerate(hidden_rep):
                pooled_h = h.reshape(batch_size, -1)
                # print(pooled_h.shape)
                score_over_layer += self.drop(self.linears_prediction[i](pooled_h))
            # print(score_over_layer.shape)
            return score_over_layer
        else:
            for i, h in enumerate(hidden_rep):
                pooled_h = h.reshape(batch_size, -1)
                pooled_h_temp = pooled_h
                score_over_layer += self.drop(self.linears_prediction[i](pooled_h))

                pooled_h_temp = pooled_h_temp.reshape(batch_size * self.node_each_graph, -1)
                if i == 0:
                    features = pooled_h_temp
                else:
                    features = torch.cat((features, pooled_h_temp), 1)
            features_dis = get_feature_dis(features)
            return score_over_layer, features_dis


    def get_embedding(self, input):
        h = input[0]
        g = input[1]
        batch_size = input[2]
        hidden_rep = [h]

        for i in range(self.n_layers - 1):
            h = self.ginlayers[i](g, h)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            hidden_rep.append(h)

        for i, h in enumerate(hidden_rep):
            pooled_h = h.reshape(batch_size, -1)
            if i == 0:
                embedding = pooled_h
            else:
                embedding = torch.cat((embedding,pooled_h), 1)
        return embedding


