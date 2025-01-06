import torch
import numpy as np
from algorithms.util import update_linear_schedule
from algorithms.util import get_shape_from_obs_space
from algorithms.util_algo import check


class TransformerPolicy:
    """
    A policy class that uses the Aether model to select actions.
    Integrates both actor (policy) and critic (value) functionality.
    """

    def __init__(self, env,args, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):
        """
        Initialize the TransformerPolicy.
        :param env: Environment object containing topologies, n_agents, etc.
        :param args: Arguments or configuration containing hyperparameters (e.g., n_embd).
        :param obs_space: Observation space.
        :param cent_obs_space: Central observation space (shared).
        :param act_space: Action space.
        :param device: Device to run the model on (CPU or GPU).
        """

        self.device = device
        self.algorithm_name = args.algorithm_name
        self.lr = args.lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay
        self._use_policy_active_masks = args.use_policy_active_masks
        if act_space.__class__.__name__ == 'Box':
            self.action_type = 'Continuous'
        else:
            self.action_type = 'Discrete'

        self.obs_dim = get_shape_from_obs_space(obs_space)[0]
        self.share_obs_dim = get_shape_from_obs_space(cent_obs_space)[0]
        self.num_agents = env.n_agents
        self.num_edge = env.num_edge_node
        self.edge_dim = env.edge_feature_dim

        if self.action_type == 'Discrete':
            self.act_dim = act_space.n
            self.act_num = 1
        else:
            print("act high: ", act_space.high)
            self.act_dim = act_space.shape[0]
            self.act_num = self.act_dim

        print("obs_dim: ", self.obs_dim)
        print("share_obs_dim: ", self.share_obs_dim)
        print("act_dim: ", self.act_dim)

        self.tpdv = dict(dtype=torch.float32, device=device)

        if self.algorithm_name in ["mat", "mat_dec"]:
            from algorithms.EMAGT import MultiAgentTransformer as MAT
        else:
            raise NotImplementedError

        self.transformer = MAT(env,  self.act_dim,
                               n_block=args.n_block, n_embd=args.n_embd, n_head=args.n_head,
                               encode_state=args.encode_state, device=device,
                               action_type=self.action_type, dec_actor=args.dec_actor,
                               share_actor=args.share_actor, use_normal_attn=args.use_normal_attn)
        if args.env_name == "hands":
            self.transformer.zero_std()

        self.optimizer = torch.optim.Adam(self.transformer.parameters(),
                                          lr=self.lr, eps=self.opti_eps,
                                          weight_decay=self.weight_decay)

    def lr_decay(self, episode, episodes):
        """
        Linearly decay the learning rate for each episode to zero by the end of training.
        :param episode: Current episode index.
        :param episodes: Total number of episodes for training.
        """

        update_linear_schedule(self.optimizer, episode, episodes, self.lr)

    def get_actions(self, cent_obs, obs, edge_features, indices, masks, deterministic=False, **kwargs):
        """
        Generate actions, their log probabilities, and estimated values from the policy.

        :param cent_obs: Central (shared) observations.
        :param obs: Local observations.
        :param edge_features: Edge features.
        :param indices: Traffic threshold-based masks.
        :param masks: Not used here, kept for compatibility.
        :param deterministic: If True, select actions deterministically (e.g., argmax). Otherwise, sample stochastically.
        :param kwargs: Topology-specific arguments.

        :return values: Estimated state values.
        :return actions: Actions selected by the policy.
        :return action_log_probs: Log probabilities of the selected actions.
        """

        num_agents = self.num_agents if "n_agents" not in kwargs else kwargs["n_agents"]
        num_edge = self.num_edge if "num_edge" not in kwargs else kwargs["num_edge"]
        edge_dim = self.edge_dim
        cent_obs = cent_obs.reshape(-1, num_agents, self.share_obs_dim)
        obs = obs.reshape(-1, num_agents, self.obs_dim)
        edge_features = edge_features.reshape(-1, num_edge, edge_dim)
        indices = indices.reshape(-1,obs.shape[1])

        actions, action_log_probs, values = self.transformer.get_actions(cent_obs,
                                                                         obs,
                                                                         edge_features,
                                                                         indices,
                                                                         deterministic,
                                                                         **kwargs)

        actions = actions.view(-1, self.act_num)
        action_log_probs = action_log_probs.view(-1, self.act_num)
        values = values.view(-1, 1)

        return values, actions, action_log_probs

    def get_values(self, cent_obs, obs, edge_features, indices, masks, **kwargs):
        """
        Compute only the value from the policy (critic).
        :param cent_obs: Central observations.
        :param obs: Local observations.
        :param edge_features: Edge features.
        :param indices: Traffic threshold-based masks.
        :param masks: Not used, kept for compatibility.
        :param kwargs: Topology-specific arguments.

        :return: Estimated values.
        """

        num_agents = self.num_agents if "n_agents" not in kwargs else kwargs["n_agents"]
        num_edge = self.num_edge if "num_edge" not in kwargs else kwargs["num_edge"]
        edge_dim = self.edge_dim
        cent_obs = cent_obs.reshape(-1, num_agents, self.share_obs_dim)
        obs = obs.reshape(-1, num_agents, self.obs_dim)
        edge_features = edge_features.reshape(-1, num_edge, edge_dim)
        indices = indices.reshape(-1, obs.shape[1])

        values = self.transformer.get_values(cent_obs, obs,edge_features,indices, **kwargs)

        values = values.view(-1, 1)

        return values

    def evaluate_actions(self, cent_obs, obs,edge_features,indices, actions, masks, active_masks=None, **kwargs):
        """
        Evaluate the log probabilities, values, and entropies of given actions for policy optimization.

        :param cent_obs: Central observations.
        :param obs: Local observations.
        :param edge_features: Edge features.
        :param indices: Traffic threshold-based masks.
        :param actions: Actions that were taken, used to compute log probabilities.
        :param masks: Unused placeholder for compatibility.
        :param active_masks: If not None, indicates active agents for masking out inactive ones.
        :param kwargs: Topology-specific arguments.

        :return values: Estimated values.
        :return action_log_probs: Log probabilities
        :return entropy: Entropy for loss calculation.
        """

        num_agents = self.num_agents if "n_agents" not in kwargs else kwargs["n_agents"]
        num_edge = self.num_edge if "num_edge" not in kwargs else kwargs["num_edge"]
        edge_dim = self.edge_dim
        cent_obs = cent_obs.reshape(-1, num_agents, self.share_obs_dim)
        obs = obs.reshape(-1, num_agents, self.obs_dim)
        actions = actions.reshape(-1, num_agents, self.act_num)
        edge_features = edge_features.reshape(-1,num_edge,edge_dim)
        indices = indices.reshape(-1,obs.shape[1])
        action_log_probs, values, entropy = self.transformer(cent_obs, obs, actions, edge_features,indices, **kwargs)

        action_log_probs = action_log_probs.view(-1, self.act_num)
        values = values.view(-1, 1)
        entropy = entropy.view(-1, self.act_num)

        if self._use_policy_active_masks and active_masks is not None:
            entropy = (entropy*active_masks).sum()/active_masks.sum()
        else:
            entropy = entropy.mean()

        return values, action_log_probs, entropy

    def act(self, cent_obs, obs,edge_feature,indices,masks, deterministic=True, **kwargs):
        """
        A wrapper to directly get actions for rollout, ignoring value/log_probs outputs.
        :param cent_obs: Central observations.
        :param obs: Local observations.
        :param edge_feature: Edge features.
        :param indices: Traffic threshold-based masks.
        :param masks: Unused placeholder for compatibility.
        :param deterministic: If True, use deterministic action selection.
        :param kwargs: Additional arguments.

        :return: Actions returned on the CPU.
        """

        _, actions, _ = self.get_actions(cent_obs,obs,edge_feature,indices,masks,deterministic, **kwargs)

        return actions.cpu()

    def save(self, save_dir, episode):
        """
        Save the transformer model parameters to disk.
        :param save_dir: Directory path to save the model file.
        :param episode: Current training episode or iteration for checkpointing.
        """

        torch.save(self.transformer.state_dict(), str(save_dir) + "/transformer_" + str(episode) + ".pt")

    def restore(self, model_dir):
        """
        Load model parameters from a saved state dictionary.
        :param model_dir: File path to the saved model parameters.
        """

        transformer_state_dict = torch.load(model_dir)
        self.transformer.load_state_dict(transformer_state_dict)

    def train(self):
        """
        Switch the transformer to training mode.
        """

        self.transformer.train()

    def eval(self):
        """
        Switch the transformer to evaluation mode.
        """

        self.transformer.eval()

