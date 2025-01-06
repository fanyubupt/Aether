import json
import os

import math
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import torch_scatter
from networkx.readwrite import json_graph
from tqdm import tqdm

from env.subgraph_generate import SubgraphGenerator

source_root = "../Aether/datasets"

def read_and_load(file_path):
    """Read and load data from a file based on its extension (.pkl, .json or .npz)."""
    if file_path.endswith(".pkl"):
        with open(file_path, 'rb') as file:
            return pickle.load(file, fix_imports=True, encoding='bytes')
    if file_path.endswith(".json"):
        with open(file_path) as file:
            return json.load(file)
    if file_path.endswith('.npz'):
        import scipy.sparse as sp
        loaded = sp.load_npz(file_path).tocsr()
        num_nodes = loaded.shape[1]
        num_matrices = loaded.shape[0] // num_nodes
        matrices = [loaded[i * num_nodes:(i + 1) * num_nodes, :].toarray() for i in range(num_matrices)]
        return matrices
    raise RuntimeError

class TEEnvMultiTopo:
    """
    Multi-topology environment for Traffic Engineering (TE) tasks. This class
    manages training and testing across both single and multiple topologies,
    loading different datasets and providing environment transitions for reinforcement learning.

    :param args: (argparse.Namespace) Arguments containing environment and model configurations.
    :param logger: (object) Logger for tracking and debugging.
    """

    def __init__(self, args, logger):
        """
        Initialize the multi-topology TE environment.
        :param args: (Namespace) Command-line arguments or config settings.
        :param logger: (object) Logger for tracking and recording information.
        """

        # Store training and testing topologies
        self.train_topos = args.train_topos
        self.test_topos = args.test_topos
        # Consolidate all topologies
        self.all_topos = list(set(args.train_topos + args.test_topos))

        # Create dictionaries to store data and parameters for each topology
        self._topo_data_dict = {topo :dict() for topo in self.all_topos}
        self.topo_kwargs = {topo :dict() for topo in self.all_topos}

        # Core environment parameters
        self.obj = args.obj
        self.num_path = args.num_path
        self.logger = logger
        self.time_window = 1
        self.episode_length = args.episode_length
        self.use_next_sample = args.use_next_sample

        # Threshold and constants for traffic and path selection
        self.traffic_threshold = 2
        self.candidate_path = 2

        # Predefined dataset sizes (split points) for different topologies
        split = {"Abilene": 48383, "Geant": 10773, 'B4': 36, 'Brain': 9723, "Kdl": 500, "Germany50": 288}

        # Keep track of topology complexity for ordering/hardness
        hardness_list = []

        # Load data for each topology
        for topo in self.all_topos:
            # Set up file paths
            topo_root = source_root + f'/{topo}'
            tm_path = topo_root + f'/{topo}_tm.pkl' if topo not in ["Brain", "Kdl", "Germany50"] else topo_root + f'/{topo}_tm.npz'
            topo_path = topo_root + f'/{topo}.json'
            path_fname = topo_root +f'/{topo}.json-4-paths_edge-disjoint-True_dist-metric-min-hop-dict.pkl'

            # Load traffic matrix data, topology, capacity, path dictionary, etc.
            tm_data = read_and_load(tm_path)[:split[topo]]
            G = json_graph.node_link_graph(read_and_load(topo_path))
            capacity = torch.FloatTensor([float(c_e) for u, v, c_e in G.edges.data('capacity')])
            path_dict = self.get_regular_path(path_fname)

            # Construct path-edge adjacency / subgraph generator for features
            n_nodes = len(G.nodes)
            n_agents = n_nodes * (n_nodes - 1)
            paths_to_edges = self.pathdict_opt(path_dict, G, n_nodes, topo)
            generator = SubgraphGenerator(G.edges, capacity, self.num_path, paths_to_edges, topo)
            edgefeature_data, edge_adj = generator.edge_features(tm_data)
            edgefeature_data = edgefeature_data[:split[topo]]

            # Store environment parameters for external usage
            num_edge_node = len(G.edges)
            num_path_node = self.num_path * G.number_of_nodes() * (G.number_of_nodes() - 1)


            # Prepare path-edge mapping for GNN usage
            edge_index, edge_index_values, p2e, edge2idx_dict = self.get_topo_matrix(path_dict, G)

            # Store environment config and data
            self.topo_kwargs[topo] = {'n_nodes': n_nodes, 'n_agents': n_agents, 'edge_adj': edge_adj, 'paths_to_edges': paths_to_edges, 'num_edge': num_edge_node}
            self._topo_data_dict[topo] = {'tm': tm_data, 'G': G, 'capacity': capacity, 'path_dict': path_dict, 'edge_feature': edgefeature_data,'p2e': p2e}

            # Gather hardness metric: node count + edge count
            hardness_list.append((topo, n_nodes + num_edge_node))

        # for topo in self.topo_kwargs.keys():
        #     self.topo_kwargs[topo]['n_agents'] = max_n_agents
        #     self.topo_kwargs[topo]['num_edge'] = max_num_edge_node
        #     self.topo_kwargs[topo]['edge_adj'] = pad_array_to_shape(self.topo_kwargs[topo]['edge_adj'], (max_num_edge_node, max_num_edge_node))
        #     self.topo_kwargs[topo]['paths_to_edges'] = pad_array_to_shape(self.topo_kwargs[topo]['paths_to_edges'], (`max_n_agents`*self.num_path, max_num_edge_node))

        # Sort topologies by complexity/hardness
        self.topo_hardness_order = [x[0] for x in sorted(hardness_list, key=lambda x: x[1])]

        # Generate datasets for training/testing
        self.train_set, self.train_indices, self.test_set = self.generate_dataset()
        self.train_cursor = 0
        self.test_cursor = 0

        # Exposed environment parameters that can change
        self.n_nodes = None
        self.n_agents = None
        self.num_edge_node = None
        self.edge_adj = None
        self.paths_to_edges = None

        # Fixed parameters
        self.n_agents = -1
        self.num_edge_node = -1

        # Topology-independent dimensions
        self.obs_dim = 1
        self.share_obs_dim =  self.obs_dim
        self.action_dim = self.num_path
        self.edge_feature_dim=args.edge_feature_dim
        self.decision_unit =args.decision_unit

    def get_topo_matrix(self, path_dict, G):
        """
        Build and return topology matrices for path-edge relationships.
        :param path_dict: (dict) Dictionary mapping (s,t) to lists of paths.
        :param G: (networkx.Graph) Graph with capacity data and edges.
        :return: (edge_index, edge_index_values, p2e, edge2idx_dict)
                 where edge_index and edge_index_values can be used for
                 graph-based processing, and p2e maps path indices to edges.
        """

        # Map each edge to an index
        edge2idx_dict = {edge: idx for idx, edge in enumerate(G.edges)}
        node2degree_dict = {}
        edge_num = len(G.edges)
        src, dst, path_i = [], [], 0

        # Build adjacency list from path dictionary
        for s in range(len(G)):
            for t in range(len(G)):
                if s == t:
                    continue
                for path in path_dict[(s, t)]:
                    for (u, v) in zip(path[:-1], path[1:]):
                        src.append(edge_num + path_i)
                        dst.append(edge2idx_dict[(u, v)])
                        if src[-1] not in node2degree_dict:
                            node2degree_dict[src[-1]] = 0
                        node2degree_dict[src[-1]] += 1
                        if dst[-1] not in node2degree_dict:
                            node2degree_dict[dst[-1]] = 0
                        node2degree_dict[dst[-1]] += 1
                    path_i += 1

        edge_index_values = torch.tensor(
            [1 / math.sqrt(node2degree_dict[u] * node2degree_dict[v])
             for u, v in zip(src + dst, dst + src)])
        edge_index = torch.tensor(
            [src + dst, dst + src], dtype=torch.long)

        # p2e tracks path nodes and corresponding edges
        p2e = torch.tensor([src, dst], dtype=torch.long)
        p2e[0] -= len(G.edges)

        return edge_index, edge_index_values, p2e, edge2idx_dict

    def get_regular_path(self, path_fname):
        """
        Load a path dictionary from a file and ensure each (s, t) has a consistent
        number of paths. Paths are trimmed or padded to match self.num_path.
        :param path_fname: (str) File name for the precomputed paths.
        :return: (dict) Updated path dictionary with exactly self.num_path paths per demand.
        """
        print("Loading paths from pickle file", path_fname)
        with open(path_fname, 'rb') as f:
            path_dict = pickle.load(f)
            print("path_dict size:", len(path_dict))

        # Pad or trim paths as needed
        for (s_k, t_k) in tqdm(path_dict.keys(), desc="Padding path_dict"):
            if len(path_dict[(s_k, t_k)]) < self.num_path:
                path_dict[(s_k, t_k)] = [path_dict[(s_k, t_k)][0] for _
                                            in range(self.num_path - len(path_dict[(s_k, t_k)]))] \
                                        + path_dict[(s_k, t_k)]
            elif len(path_dict[(s_k, t_k)]) > self.num_path:
                path_dict[(s_k, t_k)] = path_dict[(s_k, t_k)][:self.num_path]
        return path_dict

    def pathdict(self, path_dict, G, n_nodes):
        """
        Convert a path dictionary into a path-to-edges matrix.
        :param path_dict: (dict) Mapping (s, t) to paths.
        :param G: (networkx.Graph) Graph object containing edges.
        :param n_nodes: (int) Number of nodes in the graph.
        :return paths_to_edges: (torch.Tensor) A matrix indicating which edges
                                belong to which paths.
        """

        path_edgeset = {}
        for key, paths in path_dict.items():
            path_edgeset[key] = []
            for path in paths:
                edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
                path_edgeset[key].append(edges)

        edge2idx_dict = {edge: idx for idx, edge in enumerate(G.edges)}
        edge_num = len(G.edges)
        paths_arr = []

        for i in tqdm(range(n_nodes), desc="Building edge2idx"):
            for j in range(n_nodes):
                if i == j:
                    continue
                for p in path_edgeset[(i, j)]:
                    p_ = [edge2idx_dict[e] for e in p]
                    p__ = np.zeros((int(edge_num),))
                    for k in p_:
                        p__[k] = 1
                    paths_arr.append(p__)

        paths_to_edges = torch.Tensor(np.stack(paths_arr))
        return paths_to_edges

    def pathdict_opt(self, path_dict, G, n_nodes, topo):
        """
        Optimized version of pathdict. If the paths_to_edges result is
        precomputed and saved, load it. Otherwise, compute and cache it.
        :param path_dict: (dict) Mapping (s, t) to paths.
        :param G: (networkx.Graph) Graph object.
        :param n_nodes: (int) Number of nodes.
        :param topo: (str) Topology name for file saving/loading.
        :return: (torch.Tensor) paths_to_edges matrix.
        """

        filename = f'./env/paths_to_edges/paths_to_edges_{topo}.pkl'
        directory = os.path.dirname(filename)
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"The directory '{directory}' does not exist.")

        # Check if already computed
        if os.path.exists(filename):
            print(f'Loading pre-computed result from {filename}')
            with open(filename, 'rb') as f:
                return pickle.load(f)

        # Otherwise, compute and save
        paths_to_edges = self.pathdict(path_dict, G, n_nodes)
        with open(filename, 'wb') as f:
            pickle.dump(paths_to_edges, f)
        return paths_to_edges

    def create_sliding_windows(self, data_x, data_tm, data_topo):
        """Generate sliding windows for data sequences:"""

        windows = []
        tm = []
        topos = []

        for i in range(len(data_x) - self.time_window + 1):
            windows.append(data_x[i + self.time_window - 1])
            tm.append(data_tm[i:i + self.time_window])
            topos.append(data_topo[i + self.time_window - 1])


        if isinstance(windows[0], torch.Tensor):
            stacked_tensor = torch.stack(windows)
            windows_array = stacked_tensor.numpy()
        else:
            windows_array = np.array(windows)

        return windows_array,np.array(tm), topos

    def generate_dataset(self):
        """
        Construct a unified dataset for training and testing by:
        1. Creating sliding windows.
        2. Splitting data into independent training and testing sets for
           'uniq_train_topos', 'uniq_test_topos', and 'common_topos'.
        3. Merging and interleaving them into one train_set and one test_set.

        :return train_set, train_indices, test_set
        """

        datasets = {}
        for topo, data in self._topo_data_dict.items():
            # Create repeated list of topology names
            topos = [topo for _ in range(len(data['tm']))]
            x, tm, topos = self.create_sliding_windows(data['edge_feature'], data['tm'],  topos)
            combined_data = list(zip(x, tm, topos))
            datasets[topo] = combined_data

        # Prepare separate lists for training and testing sets
        ori_train_sets = []
        ori_test_sets = []

        # Collect unique training topologies (not in test_topos)
        uniq_train_topos = list(set(self.train_topos) - set(self.test_topos))
        print("Unique training topologies:", uniq_train_topos)
        for topo in uniq_train_topos:
            ori_train_sets.append((topo, datasets[topo]))

        # Collect unique testing topologies (not in train_topos)
        uniq_test_topos = list(set(self.test_topos) - set(self.train_topos))
        print("Unique testing topologies:", uniq_test_topos)
        for topo in uniq_test_topos:
            test_size = int(0.2 * len(datasets[topo]))
            datasets[topo] = datasets[topo][:test_size]
            ori_test_sets.append((topo, datasets[topo]))

        # Collect common topologies that appear in both train and test
        common_topos = list(set(self.train_topos) & set(self.test_topos))
        print("Common topologies", common_topos)
        for topo in common_topos:
            dataset = datasets[topo]
            train_size = int(0.95 * len(dataset))
            test_size = len(dataset) - train_size
            # Partition data for training and testing
            ori_test_sets.append((topo, dataset[:test_size]))
            ori_train_sets.append((topo, dataset[-train_size:]))

        # Order the training sets by topology hardness
        order_map = {value: index for index, value in enumerate(self.topo_hardness_order)}
        sorted_train_sets = sorted(ori_train_sets, key=lambda x: order_map[x[0]])

        # Create an indexing scheme (train_indices) to interleave training data for episodes
        topo_switch_interval = self.episode_length
        lengths = [len(data) for _, data in sorted_train_sets]
        max_length = max(lengths)
        total_repeat = int((max_length + topo_switch_interval - 1) / topo_switch_interval)
        while int((max_length + total_repeat + topo_switch_interval - 1) / topo_switch_interval) > total_repeat:
            total_repeat += 1

        start_idx = [0]
        for i in range(1, len(lengths)):
            start_idx.append(start_idx[-1] + lengths[i-1])

        dataset_cursors = [0 for _ in lengths]
        train_indices = []

        # Interleave data from each topology for episodes
        for i in range(total_repeat):
            for j in range(len(lengths)):
                for k in range(topo_switch_interval - 1):
                    train_indices.append(start_idx[j] + dataset_cursors[j])
                    dataset_cursors[j] = (dataset_cursors[j] + 1) % lengths[j]
                    if k == topo_switch_interval - 2:
                        train_indices.append(start_idx[j] + dataset_cursors[j])

        # Combine all sorted training data
        train_set = []
        for name, data in sorted_train_sets:
            train_set.extend(data)

        # Combine all test data
        test_set = []
        for name, data in ori_test_sets:
            test_set.extend(data)


        return train_set, train_indices, test_set

    def obs_agent(self, agent_obs, agent_ids):
        """Not used"""
        return agent_obs

    def reset(self, mode='train'):
        """
        Reset the environment to the first sample of the training or testing set.
        :param mode: (str) Indicates 'train' or 'test' mode.
        :return: Tuple containing initial observations, share_obs, edge features, indices,
                 placeholder for actions, and topology ID.
        """
        if mode == 'train':
            self.train_cursor = 0
            example = self.train_set[self.train_indices[self.train_cursor]]
        else:
            self.test_cursor = 0
            example = self.test_set[self.test_cursor]

        topo = example[3]
        edge_feature = np.expand_dims(example[0],axis=0)

        # Prepare traffic matrix for n_agents
        tm = torch.FloatTensor(
                [[ele] for i, ele in enumerate(example[1][-1].flatten())
                 if i % len(example[1][-1]) != i // len(example[1][-1])]).flatten() # n_agents
        obs = share_obs = tm.unsqueeze(0).unsqueeze(2)

        # Indices for traffic above the threshold
        indices = (obs > self.traffic_threshold).long().squeeze(2)
        indices = np.expand_dims(indices,axis=0)

        # Update environment parameters according to the current topology
        self.n_agents = self.topo_kwargs[topo]['n_agents']
        self.edge_adj = self.topo_kwargs[topo]['edge_adj']
        self.paths_to_edges = self.topo_kwargs[topo]['paths_to_edges']
        self.num_edge_node = self.topo_kwargs[topo]['num_edge']
        self.n_nodes = self.topo_kwargs[topo]['n_nodes']

        topo = np.array(topo).reshape(1,1)

        return obs.numpy(), share_obs.numpy(), edge_feature,indices, topo

    def step(self, last_actions, mode='train'):
        """
        Execute a step in the environment. Retrieves the next example from the dataset,
        calculates rewards based on the provided actions, and returns the new observation.
        :param last_actions: (np.array) Actions from the policy/agent.
        :param mode: (str) 'train' or 'test' mode to select the corresponding dataset.
        :return: A tuple of (next_obs, next_share_obs, next_edge_feature, next_indices,
                 rewards, dones, info, available_actions, round_actions,
                 compare_dict, next_topo). Returns None if test set is exhausted.
        """

        if mode == 'train':
            last_example = self.train_set[self.train_indices[self.train_cursor - 1]] if self.train_cursor > 0 else None
            current_example = self.train_set[self.train_indices[self.train_cursor]]
            self.train_cursor = (self.train_cursor + 1) % len(self.train_set)
            next_example = self.train_set[self.train_indices[self.train_cursor]]
        else:
            if self.test_cursor + 1 >= len(self.test_set):
                return None
            last_example = self.test_set[self.test_cursor - 1] if self.test_cursor > 0 else None
            current_example = self.test_set[self.test_cursor]
            self.test_cursor = self.test_cursor + 1
            next_example = self.test_set[self.test_cursor]

        # Retrieve current topology info
        current_topo = current_example[3]
        capacity = self._topo_data_dict[current_topo]['capacity']
        paths_to_edges = self.topo_kwargs[current_topo]["paths_to_edges"]
        tm_matrix = current_example[1][-1]

        # Prepare observation/tensors
        temp = torch.FloatTensor(
                [[ele] for i, ele in enumerate(tm_matrix.flatten())
                 if i % len(tm_matrix) != i // len(tm_matrix)]).flatten()#n_agents
        current_obs = current_share_obs = temp.unsqueeze(0).unsqueeze(2)
        current_edge_feature = np.expand_dims(current_example[0], axis=0)
        current_indices = (current_obs > self.traffic_threshold).long().squeeze(2)
        current_indices = np.expand_dims(current_indices, axis=0)

        # If not using next sample but the topology switches, skip reward
        if not self.use_next_sample and last_example is not None and last_example[3] != current_example[3]:
            rewards = None
            compare_dict = None
            last_actions = torch.zeros(4)
        else:
            # Construct the demands and handle partial allocations
            current_tm = torch.FloatTensor(
                    [[ele] * self.num_path for i, ele in enumerate(tm_matrix.flatten())
                     if i % len(tm_matrix) != i // len(tm_matrix)]).flatten()
            combined_demand_splits = torch.zeros(last_actions.shape)
            demands = current_tm
            demand_split = F.softmax(torch.from_numpy(last_actions), dim=2)
            cur_indices = torch.tensor(current_indices).squeeze(1).unsqueeze(2)
            cur_indices_inv = 1 - cur_indices

            nn_demand_split = torch.mul(demand_split, cur_indices_inv)
            combined_demand_splits = torch.add(combined_demand_splits, nn_demand_split)

            last_actions = torch.mul(demand_split.flatten(), demands)
            nn_action = torch.mul(nn_demand_split.flatten(), demands).reshape(1, -1)

            tmperedges = paths_to_edges.transpose(0, 1).matmul(torch.transpose(nn_action, 0, 1))
            link_util = tmperedges.divide(capacity.unsqueeze(1))
            temp_util = torch.sum(torch.mul(paths_to_edges, link_util.transpose(0, 1)), dim=1).view(demand_split.shape)

            min_temp_util_indices = torch.argsort(temp_util, dim=2)
            min_temp_util_indices = min_temp_util_indices[:, :, :self.candidate_path]
            one_hot = torch.zeros_like(temp_util).scatter_(2, min_temp_util_indices, 1.0 / self.candidate_path)
            updated_split_ratio = one_hot * cur_indices

            combined_demand_splits = torch.add(combined_demand_splits, updated_split_ratio)

            y_pred = torch.mul(combined_demand_splits.flatten(), demands).reshape(1, -1)

            assert -1 < y_pred.sum() - temp.sum() < 1

            tmperedges = self.topo_kwargs[current_topo]['paths_to_edges'].transpose(0, 1).matmul(torch.transpose(y_pred, 0, 1))  # dim edge_numbers
            temp = tmperedges.divide(capacity.unsqueeze(1))
            alpha = torch.max(temp, dim=0)[0]

            # Standard reward calculation
            if self.obj == "total_flow":
                actions = self.round_action(current_tm, y_pred.numpy() , self._topo_data_dict[current_topo]['capacity'], self._topo_data_dict[current_topo]['p2e'])
                reward = actions.sum()
                rewards = np.broadcast_to(np.asarray(reward), (1, self.topo_kwargs[current_topo]['n_agents'], 1))
                compare_dict = {"reward": reward.item()}
            else:
                reward = -alpha
                rewards = np.broadcast_to(np.asarray(reward), (1, self.topo_kwargs[current_topo]['n_agents'], 1))
                compare_dict = {"reward": reward.item()}

        # Prepare the next sample
        next_topo = next_example[3]
        next_temp = torch.FloatTensor(
                [[ele] for i, ele in enumerate(next_example[1][-1].flatten())
                 if i % len(next_example[1][-1]) != i // len(next_example[1][-1])]).flatten()#n_agents
        next_obs = next_share_obs = next_temp.unsqueeze(0).unsqueeze(2)
        next_edge_feature = np.expand_dims(next_example[0],axis=0)
        next_indices = (next_obs > self.traffic_threshold).long().squeeze(2)
        next_indices = np.expand_dims(next_indices,axis=0)

        if not self.use_next_sample:
            # Remain on the current topology
            self.n_agents = self.topo_kwargs[current_topo]['n_agents']
            self.edge_adj = self.topo_kwargs[current_topo]['edge_adj']
            self.paths_to_edges = self.topo_kwargs[current_topo]['paths_to_edges']
            self.num_edge_node = self.topo_kwargs[current_topo]['num_edge']
            self.n_nodes = self.topo_kwargs[current_topo]['n_nodes']
            current_topo = np.array(current_topo).reshape(1, 1)

            return (current_obs.numpy(), current_share_obs.numpy(), current_edge_feature, current_indices, rewards,
                    np.zeros((1, self.n_agents), dtype=bool), None, np.ones((1, self.n_agents, self.action_dim), dtype=bool), None, last_actions.numpy().reshape(1, -1, 4), compare_dict, current_topo)
        else:
            # Switch to next topology
            self.n_agents = self.topo_kwargs[next_topo]['n_agents']
            self.edge_adj = self.topo_kwargs[next_topo]['edge_adj']
            self.paths_to_edges = self.topo_kwargs[next_topo]['paths_to_edges']
            self.num_edge_node = self.topo_kwargs[next_topo]['num_edge']
            self.n_nodes = self.topo_kwargs[next_topo]['n_nodes']
            next_topo = np.array(next_example[3]).reshape(1,1)

            return (next_obs.numpy(), next_share_obs.numpy(), next_edge_feature, next_indices, rewards,
                    np.zeros((1, self.n_agents), dtype=bool), None, np.ones((1, self.n_agents, self.action_dim), dtype=bool), None, last_actions.numpy().reshape(1, -1, 4), compare_dict, next_topo)

    def get_topo_next_sample(self, topo):
        """Advance the training cursor by one to retrieve the next example."""
        self.train_cursor = (self.train_cursor + 1) % len(self.train_set)

    def close(self):
        """Not used"""
        pass

    def round_action(
            self, demand, action, capacity, p2e, round_demand=True, round_capacity=True,
            num_round_iter=2):
        """
        Return rounded action.
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
                    path_flow[p2e[0]], p2e[1])
                # util of each edge
                util = 1 + (edge_flow / capacity - 1).relu()
                # propotionally cut path flow by max util
                util = torch_scatter.scatter(
                    util[p2e[1]], p2e[0], reduce="max")
                path_flow_allocated = path_flow / util
                # update total allocation, residual capacity, residual flow
                path_flow_allocated_total += path_flow_allocated
                if round_iter != num_round_iter - 1:
                    capacity = (capacity - torch_scatter.scatter(
                        path_flow_allocated[p2e[0]], p2e[1])).relu()
                    path_flow = path_flow - path_flow_allocated
            action = path_flow_allocated_total
        return action