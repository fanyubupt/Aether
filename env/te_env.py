import json
import math
import os
import pickle
import random

import numpy as np
import torch
import torch_scatter
from networkx.readwrite import json_graph
from env.generatepath_utils import find_paths, graph_copy_with_edge_weights, remove_cycles
import torch.nn.functional as F
from env.subgraph_generate import SubgraphGenerator

source_root = "../Aether/datasets"

# Legacy class; the single topology handling is now integrated into the TEEnvMultiTopo class
class TEEnv:
    def __init__(self, args, logger):
        self.topo = args.topo
        self.obj = args.obj
        self.logger = logger
        self.G = self._read_graph_json()
        self.n_nodes = len(self.G.nodes)
        self.n_agents = self.n_nodes * (self.n_nodes - 1)
        self.num_path = args.num_path
        edge_disjoint = True
        dist_metric = 'min-hop'
        self.capacity = torch.FloatTensor(
            [float(c_e) for u, v, c_e in self.G.edges.data('capacity')])
        self.num_edge_node = len(self.G.edges)
        self.num_path_node = self.num_path * self.G.number_of_nodes() \
                             * (self.G.number_of_nodes() - 1)
        self.edge_index, self.edge_index_values, self.p2e = \
            self.get_topo_matrix(self.num_path, edge_disjoint, dist_metric)
        self.pathdict(self.num_path, edge_disjoint, dist_metric)
        self.time_window = 1
        #if train all topo
        split = {"Abilene": 48383, "Geant": 10773, 'B4': 36, 'Brain': 9723, 'Kdl': 500, 'Germany50': 288}[args.topo]
        self.generator = SubgraphGenerator(self.G.edges, self.capacity,self.num_path,self.paths_to_edges,self.topo)
        self.edge_adj = None
        self.edgefeature_data = None
        self.generate_lpalldata(split)
        self.obs_dim = 1
        self.share_obs_dim =  self.obs_dim
        self.action_dim = self.num_path
        self.train_cursor = 0
        self.test_cursor = 0
        self.edge_feature_dim=args.edge_feature_dim
        self.decision_unit =args.decision_unit

    def _read_graph_json(self):
        """Return network topo from json file."""
        with open(source_root + '/{}/{}.json'.format(self.topo, self.topo)) as f:
            data = json.load(f)
        return json_graph.node_link_graph(data)

    def get_topo_matrix(self, num_path, edge_disjoint, dist_metric):
        """
        Return matrices related to topology.
        edge_index, edge_index_values: index and value for matrix
        D^(-0.5)*(adjacent)*D^(-0.5) without self-loop
        p2e: [path_node_idx, edge_nodes_inx]
        """
        # get regular path dict
        path_dict = self.get_regular_path(num_path, edge_disjoint, dist_metric)

        # edge nodes' degree, index lookup
        self.edge2idx_dict = {edge: idx for idx, edge in enumerate(self.G.edges)}
        node2degree_dict = {}
        edge_num = len(self.G.edges)

        # build edge_index
        src, dst, path_i = [], [], 0
        for s in range(len(self.G)):
            for t in range(len(self.G)):
                if s == t:
                    continue
                for path in path_dict[(s, t)]:
                    for (u, v) in zip(path[:-1], path[1:]):
                        src.append(edge_num + path_i)
                        dst.append(self.edge2idx_dict[(u, v)])

                        if src[-1] not in node2degree_dict:
                            node2degree_dict[src[-1]] = 0
                        node2degree_dict[src[-1]] += 1
                        if dst[-1] not in node2degree_dict:
                            node2degree_dict[dst[-1]] = 0
                        node2degree_dict[dst[-1]] += 1
                    path_i += 1

        # edge_index is D^(-0.5)*(adj)*D^(-0.5) without self-loop
        edge_index_values = torch.tensor(
            [1 / math.sqrt(node2degree_dict[u] * node2degree_dict[v])
             for u, v in zip(src + dst, dst + src)])
        edge_index = torch.tensor(
            [src + dst, dst + src], dtype=torch.long)
        p2e = torch.tensor([src, dst], dtype=torch.long)
        p2e[0] -= len(self.G.edges)

        return edge_index, edge_index_values, p2e

    def compute_path(self, num_path, edge_disjoint, dist_metric):
        """Return path dictionary through computation."""

        path_dict = {}
        G = graph_copy_with_edge_weights(self.G, dist_metric)
        for s_k in G.nodes:
            for t_k in G.nodes:
                if s_k == t_k:
                    continue
                paths = find_paths(G, s_k, t_k, num_path, edge_disjoint)
                paths_no_cycles = [remove_cycles(path) for path in paths]
                path_dict[(s_k, t_k)] = paths_no_cycles
        return path_dict

    def get_regular_path(self, num_path, edge_disjoint, dist_metric):
        """Return path dictionary with the same number of paths per demand.
        Fill with the first path when number of paths is not enough.
        """
        self.path_fname = source_root + '/{}/{}.json-4-paths_edge-disjoint-True_dist-metric-min-hop-dict.pkl'.format(
            self.topo, self.topo)
        print("Loading paths from pickle file", self.path_fname)
        try:
            with open(self.path_fname, 'rb') as f:
                path_dict = pickle.load(f)
                print("path_dict size:", len(path_dict))
                # return path_dict
        except FileNotFoundError:
            print("Creating paths {}".format(self.path_fname))
            path_dict = self.compute_path(num_path, edge_disjoint, dist_metric)
            print("Saving paths to pickle file")
            with open(self.path_fname, "wb") as w:
                pickle.dump(path_dict, w)

        for (s_k, t_k) in path_dict:
            if len(path_dict[(s_k, t_k)]) < self.num_path:
                path_dict[(s_k, t_k)] = [
                                            path_dict[(s_k, t_k)][0] for _
                                            in range(self.num_path - len(path_dict[(s_k, t_k)]))] \
                                        + path_dict[(s_k, t_k)]
            elif len(path_dict[(s_k, t_k)]) > self.num_path:
                path_dict[(s_k, t_k)] = path_dict[(s_k, t_k)][:self.num_path]
        return path_dict

    def create_sliding_windows(self, data_x, data_tm):
        windows = []
        tm = []
        for i in range(len(data_x) - self.time_window + 1):
            windows.append(data_x[i + self.time_window - 1])
            tm.append(data_tm[i:i + self.time_window])
        return np.array(windows), np.array(tm)

    def generate_lpalldata(self, split):
        algorithm_dict = {}

        def read_and_load(file_path):
            if file_path.endswith('.pkl'):
                with open(file_path, 'rb') as file:
                    return pickle.load(file, fix_imports=True, encoding='bytes')
            elif file_path.endswith('.npz'):
                import scipy.sparse as sp
                loaded = sp.load_npz(file_path).tocsr()
                num_nodes = loaded.shape[1]
                num_matrices = loaded.shape[0] // num_nodes
                matrices = [loaded[i * num_nodes:(i + 1) * num_nodes, :].toarray() for i in range(num_matrices)]
                return matrices

        tm_file_path = os.path.join(source_root, self.topo, f'{self.topo}_tm.pkl') if self.topo not in ["Brain", "Kdl", "Germany50"] else os.path.join(source_root, self.topo, f'{self.topo}_tm.npz')
        tm_data = read_and_load(tm_file_path)
        edgefeature_data, self.edge_adj = self.generator.edge_features(tm_data)


        algorithm_dict = {'x': edgefeature_data[:split], # example[0]
                                    'tm': tm_data[:split]} # example[1]
        train_set = {}
        test_set = {}
        for data in algorithm_dict.items():
            x, y, tm, reward = self.create_sliding_windows(data['x'], data['tm'])

            combined_data = list(zip(x, tm, reward))
            train_size = int(0.8 * len(data['x']))
            test_size = len(data['x']) - train_size
            train_size = len(data['x']) - test_size
            train_set = combined_data[:train_size]
            test_set = combined_data[-test_size:]
        self.train_set = train_set['lp_all']
        self.test_set = test_set['lp_all']


    def obs_agent(self, agent_obs, agent_ids):
        return agent_obs

    def reset(self, mode='train'):
        if mode == 'train':
            self.train_cursor = 0
            example = self.train_set[self.train_cursor]
        else:
            self.test_cursor = 0
            example = self.test_set[self.test_cursor]

        tm = torch.FloatTensor(
                [[ele] for i, ele in enumerate(example[1][-1].flatten())
                 if i % len(example[1][-1]) != i // len(example[1][-1])]).flatten()
        obs = share_obs = tm.unsqueeze(0).unsqueeze(2)
        edge_feature = np.expand_dims(example[0],axis=0)
        indices = (obs > 2).long().squeeze(2)
        
        indices = np.expand_dims(indices,axis=0)
        return obs.numpy(), share_obs.numpy(), edge_feature,indices

    def round_action(
            self, demand, action, round_demand=True, round_capacity=True,
            num_round_iter=2):
        """Return rounded action.
        Action can still violate constraints even after ADMM fine-tuning.
        This function rounds the action through cutting flow.

        Args:
            action: input action
            round_demand: whether to round action for demand constraints
            round_capacity: whether to round action for capacity constraints
            num_round_iter: number of rounds when iteratively cutting flow
        """
        demand = demand[0::self.num_path]
        action = torch.from_numpy(action)
        capacity = self.capacity

        # reduce action proportionally if action exceed demand
        if round_demand:
            action = action.reshape(-1, self.num_path)
            ratio = action.sum(-1) / demand
            action[ratio > 1, :] /= ratio[ratio > 1, None]
            action = action.flatten()

        # iteratively reduce action proportionally if action exceed capacity
        if round_capacity:
            path_flow = action
            path_flow_allocated_total = torch.zeros(path_flow.shape)
            for round_iter in range(num_round_iter):
                # flow on each edge
                edge_flow = torch_scatter.scatter(
                    path_flow[self.p2e[0]], self.p2e[1])
                # util of each edge
                util = 1 + (edge_flow / capacity - 1).relu()
                # propotionally cut path flow by max util
                util = torch_scatter.scatter(
                    util[self.p2e[1]], self.p2e[0], reduce="max")
                path_flow_allocated = path_flow / util
                # update total allocation, residual capacity, residual flow
                path_flow_allocated_total += path_flow_allocated
                if round_iter != num_round_iter - 1:
                    capacity = (capacity - torch_scatter.scatter(
                        path_flow_allocated[self.p2e[0]], self.p2e[1])).relu()
                    path_flow = path_flow - path_flow_allocated
            action = path_flow_allocated_total
        return action

    def step(self, actions,cur_indices, mode='train'):
        if mode == 'train':
            example = self.train_set[self.train_cursor]
            self.train_cursor = (self.train_cursor + 1) % len(self.train_set)
        else:
            if self.test_cursor + 1 >= len(self.test_set):
                return None
            example = self.test_set[self.test_cursor]
            self.test_cursor = self.test_cursor + 1
        edge_feature = np.expand_dims(example[0],axis=0)
        tm_matrix = example[1][-1]
        tm = torch.FloatTensor(
                [[ele] * self.num_path for i, ele in enumerate(tm_matrix.flatten())
                 if i % len(tm_matrix) != i // len(tm_matrix)]).flatten()
        temp = torch.FloatTensor(
                [[ele] for i, ele in enumerate(example[1][-1].flatten())
                 if i % len(example[1][-1]) != i // len(example[1][-1])]).flatten()#n_agents
        obs = share_obs = temp.unsqueeze(0).unsqueeze(2)

        combined_actions = torch.zeros(1, self.n_agents, self.num_path)
        demand_split = F.softmax(torch.from_numpy(actions),dim=2)
        demands = tm

        cur_indices = torch.tensor(cur_indices).squeeze(1).unsqueeze(2)
        cur_indices_inv = 1 -cur_indices
        nn_demand_split = torch.mul(demand_split, cur_indices_inv)
        combined_actions = torch.add(combined_actions,nn_demand_split)
        actions = torch.mul(demand_split.flatten(), demands)


        if self.obj == "total_flow":
            actions = self.round_action(tm, actions.numpy())
            rewards = np.broadcast_to(np.asarray(actions.sum()), (1, self.n_agents, 1))
        else:
            # Differential traffic strategy
            nn_action = torch.mul(nn_demand_split.flatten(),demands).reshape(1,-1)
            tmperedges = self.paths_to_edges.transpose(0, 1).matmul(torch.transpose(nn_action, 0, 1))  # dim edge_numbers
            link_util = tmperedges.divide(self.capacity.unsqueeze(1))
            temp_util = torch.sum(torch.mul(self.paths_to_edges,link_util.transpose(0,1)),dim=1).view(demand_split.shape)
            min_temp_util_indices = torch.argsort(temp_util,dim=2)
            min_temp_util_indices = min_temp_util_indices[:,:, :2]
            one_hot = torch.zeros_like(temp_util).scatter_(2, min_temp_util_indices, 1.0 / 2)
            updated_split_ratio = one_hot * cur_indices
            combined_actions = torch.add(combined_actions,updated_split_ratio)
            y_pred = torch.mul(combined_actions.flatten(), demands).reshape(1,-1)

            tmperedges = self.paths_to_edges.transpose(0, 1).matmul(torch.transpose(y_pred, 0, 1))  # dim edge_numbers
            temp = tmperedges.divide(self.capacity.unsqueeze(1))
            alpha = torch.max(temp, dim=0)[0]
            rewards = np.broadcast_to(np.asarray(-alpha), (1, self.n_agents, 1))
        compare_dict = None
        if self.obj == "total_flow":
            reward = actions.sum()
            compare_dict = {"reward":reward.item()}
        else:
            compare_dict = {"reward": -alpha.item()}
        indices = (obs > 2).long().squeeze()
        indices = np.expand_dims(indices,axis=0)
        return (obs.numpy(), share_obs.numpy(), edge_feature,indices, rewards,
                np.zeros((1, self.n_agents), dtype=bool), None, np.ones((1, self.n_agents, self.action_dim), dtype=bool),actions.numpy().reshape(1, -1, 4), compare_dict)

    def pathdict(self, num_path, edge_disjoint, dist_metric):
        path_dict = self.get_regular_path(num_path, edge_disjoint, dist_metric)
        path_edgeset = {}
        for key, paths in path_dict.items():
            path_edgeset[key] = []
            for path in paths:
                edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
                path_edgeset[key].append(edges)

        edge_num = len(self.G.edges)
        paths_arr = []
        commodities = []
        _path_to_commodity = dict()
        _path_to_idx = dict()
        cntr = 0
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if i == j:
                    continue
                idx = 0
                for p in path_edgeset[(i, j)]:
                    p_ = [self.edge2idx_dict[e] for e in p]
                    p__ = np.zeros((int(edge_num),))
                    for k in p_:
                        p__[k] = 1
                    paths_arr.append(p__)
                    _path_to_commodity[cntr] = (i, j)
                    _path_to_idx[cntr] = idx
                    cntr += 1
                    idx += 1
                    commodities.append((i, j))
        self.paths_to_edges= torch.Tensor(np.stack(paths_arr))

    def close(self):
        self.dataset_file.close()

