import os
import pickle

import torch
from tqdm import tqdm
import numpy as np


class SubgraphGenerator:
    """
    Generates subgraph-related information
    for a given network topology in TE environment.
    :param Gedges: (EdgeView or list) Edges of the graph.
    :param capacity: (torch.Tensor) Capacities of each edge.
    :param path_num: (int) Number of paths per demand.
    :param paths_to_edges: (torch.Tensor) Binary matrix indicating which edges belong to which paths.
    :param topo: (str) Name of the topology.
    """

    def __init__(self, Gedges, capacity,path_num,paths_to_edges, topo):
        self.G_edges =Gedges
        self.capacity = capacity
        self.path_num = path_num
        self.paths_to_edges = paths_to_edges
        self.topo = topo
    def edge_features(self, tms):
        """
        Compute edge features for each traffic matrix in tms. If cached data exists, load it;
        otherwise, compute and save for future use.

        :param tms: (list or np.array) A collection of traffic matrices.
        :return:
          - edge_features: (np.array) Computed edge features.
          - edge_adj_matrix: (torch.Tensor) Binary adjacency matrix of edges.
        """

        edge_feature_path = f"./env/edge_features/{self.topo}_edge_feature.pkl"
        try:
            with open(edge_feature_path, 'rb') as f:
                edge_features = pickle.load(f)
                print("edge_features size:", len(edge_features))
        except FileNotFoundError:
            print("Creating edge features {}".format(edge_feature_path))
            edges2path = self.paths_to_edges.transpose(0, 1)#edge,paths
            path_counts = torch.sum(edges2path, dim=1)
            edge_features = []
            for tm in tqdm(tms, desc="Processing TMs"):
                tm = torch.FloatTensor(
                    [[ele] for i, ele in enumerate(tm.flatten())
                     if i % len(tm) != i // len(tm)]).flatten()  # n_agents
                extended_demand =  tm.unsqueeze(1).repeat(1,self.path_num).flatten()
                edge_demands = extended_demand.unsqueeze(0) * edges2path
                edge_sum = torch.sum(edge_demands, dim=1)
                edge_mean = edge_sum / path_counts.clamp(min=1)
                edge_max = torch.max(edge_demands, dim=1).values
                edge_min = torch.min(edge_demands, dim=1).values

                edge_feature = torch.stack([path_counts.float(), edge_mean, edge_max, edge_min], dim=1)
                edge_feature = torch.cat([edge_feature,self.capacity.unsqueeze(1)],dim=1)
                edge_features.append(edge_feature)
            edge_features = torch.stack(edge_features, dim=0).numpy()
            print("Saving paths to pickle file")
            with open(edge_feature_path, "wb") as w:
                pickle.dump(edge_features, w)
            assert len(edge_features) == len(tms)

        edge_adj_matrix = torch.zeros((len(self.capacity), len(self.capacity)), dtype=torch.float32)
        for i, edge1 in enumerate(self.G_edges):
            for j, edge2 in enumerate(self.G_edges):
                if edge1[0] in edge2 or edge1[1] in edge2:
                    edge_adj_matrix[i, j] = 1
        return edge_features, edge_adj_matrix
