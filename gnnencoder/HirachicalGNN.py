import torch
from torch import nn


class hirachicalGNN(nn.Module):
    """
    Hierarchical Graph Neural Network (GNN) class for processing edge features and decision features.
    :param env: (object) TEEnvMultiTopo Environment object containing network parameters.
    :param decision_unit: (int) Number of decision units.
    :param hidden_dim: (int) Dimension of the hidden layers.
    :param device: (torch.device) Specifies the computation device (CPU/GPU).
    """

    def __init__(self, env, decision_unit, hidden_dim, device):
        super(hirachicalGNN, self).__init__()

        self.device = device

        self.n_paths = env.num_path
        feature_size = 5

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.decision_unit = decision_unit

        self.fc_egnn = nn.Linear(feature_size, hidden_dim)
        self.fc_egnn2 = nn.Linear(hidden_dim, hidden_dim)

        self.fc_p2egnn = nn.Linear(hidden_dim, hidden_dim)

        self.fc = nn.Linear(self.decision_unit*self.n_paths*(hidden_dim+1),self.decision_unit*self.n_paths*hidden_dim)

    def edgegnn(self, edge_features, adj):
        """
        Process edge features using a two-layer GNN.
        :param edge_features: (torch.Tensor) Edge features
        :param adj: (torch.Tensor) Adjacency tensor

        :return out_features: (torch.Tensor) Processed edge features.
        """

        adj = adj.to(self.device)

        temp = torch.matmul(adj.unsqueeze(0), edge_features)

        out_features1 = self.fc_egnn(temp)
        out_features1 = self.relu(out_features1)

        out_features = self.fc_egnn2(torch.matmul(adj.unsqueeze(0), out_features1))

        return out_features

    def forward(self, tm_orin, edge_features, p2e_adj):
        """
        Forward pass of the hierarchical GNN.
        :param tm_orin: (torch.Tensor) Original traffic matrix.
        :param edge_features: (torch.Tensor) Edge features.
        :param p2e_adj: (torch.Tensor) Path-to-edge adjacency tensor.

        :return hidden: (torch.Tensor) Output features.
        """

        edge_features = edge_features.to(self.device)
        p2e_adj = p2e_adj.to(self.device)

        p2egnn= self.fc_p2egnn(torch.matmul(p2e_adj,edge_features.unsqueeze(1)))
        p2egnn = self.tanh(p2egnn)

        tm_path = tm_orin.repeat(1,1,self.n_paths)
        tm = tm_path.reshape(tm_path.shape[0],-1,self.decision_unit*self.n_paths)

        decision_features = torch.cat([p2egnn.reshape(tm.shape[0],tm.shape[1],tm.shape[2],p2egnn.shape[3]),tm.unsqueeze(3)],dim=3)
        decision_features = decision_features.reshape(tm.shape[0],tm.shape[1],-1)

        hidden= self.fc(decision_features)

        return hidden
