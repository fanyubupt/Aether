import torch
import numpy as np
from algorithms.util import get_shape_from_obs_space, get_shape_from_act_space

def _flatten(T, N, x):
    """Reshapes the input tensor x from shape [T, N, ...] into a flat shape [T*N, ...]."""

    return x.reshape(T * N, *x.shape[2:])

def _cast(x):
    """Rearranges the axes of input tensor x from [T, N, Agents, Features] to [T*N, Agents, Features]."""

    return x.transpose(1, 2, 0, 3).reshape(-1, *x.shape[3:])

def _shuffle_agent_grid(x, y):
    """Generates a pair of coordinate grids (row, col) for a grid of size x by y."""

    rows = np.indices((x, y))[0]
    cols = np.stack([np.arange(y) for _ in range(x)])
    return rows, cols

class SharedReplayBuffer(object):
    """
    A shared replay buffer designed for reinforcement training.
    Stores observations, actions, rewards, etc., and supports generating mini-batches
    for policy optimization.
    """

    def __init__(self, args, num_agents,num_edges, obs_space, cent_obs_space, act_space, env_name):
        """
        Initializes the shared replay buffer with dimensions based on the environment,
        number of agents, and configuration parameters.
        :param args: (argparse.Namespace) Arguments containing buffer and training settings.
        :param num_agents: (int) Number of agents in the environment.
        :param num_edges: (int) Number of edges in the topology or graph.
        :param obs_space: (gym.Space) Observation space of a single agent.
        :param cent_obs_space: (gym.Space) Central observation space (for shared/global observations).
        :param act_space: (gym.Space) Action space for a single agent.
        :param env_name: (str) Name of the environment.
        """

        self.episode_length = args.episode_length-1 if not args.use_single_topo else args.episode_length
        self.n_rollout_threads = args.n_rollout_threads
        self.hidden_size = args.hidden_size
        self.recurrent_N = args.recurrent_N
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self._use_gae = args.use_gae
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_proper_time_limits = args.use_proper_time_limits
        self.algo = args.algorithm_name
        self.env_name = env_name
        self.edge_feature_dim = args.edge_feature_dim
        self.num_edges = num_edges

        obs_shape = get_shape_from_obs_space(obs_space)
        share_obs_shape = get_shape_from_obs_space(cent_obs_space)
        if type(obs_shape[-1]) == list:
            obs_shape = obs_shape[:1]
        if type(share_obs_shape[-1]) == list:
            share_obs_shape = share_obs_shape[:1]

        self.share_obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, *share_obs_shape),dtype=np.float32)
        self.obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, *obs_shape), dtype=np.float32)
        self.edge_features = np.zeros((self.episode_length + 1, self.n_rollout_threads, self.num_edges , self.edge_feature_dim), dtype=np.float32)
        self.indices = np.zeros((self.episode_length + 1, self.n_rollout_threads,num_agents), dtype=np.int32)
        self.topos = np.zeros((self.episode_length + 1, self.n_rollout_threads, 1), dtype=np.dtype('U10'))
        self.value_preds = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, num_agents, 1), dtype=np.float32)
        self.returns = np.zeros_like(self.value_preds)
        self.advantages = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, 1), dtype=np.float32)
        self.available_actions = None
        act_shape = get_shape_from_act_space(act_space)
        self.actions = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, act_shape), dtype=np.float32)
        self.action_log_probs = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, act_shape), dtype=np.float32)
        self.rewards = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, 1), dtype=np.float32)

        self.masks = np.ones((self.episode_length + 1, self.n_rollout_threads, num_agents, 1), dtype=np.float32)
        self.bad_masks = np.ones_like(self.masks)
        self.active_masks = np.ones_like(self.masks)

        self.step = 0

    def insert(self, share_obs, obs,edge_features,indices, rnn_states_actor, rnn_states_critic, actions, action_log_probs,
               value_preds, rewards, masks, topo, bad_masks=None, active_masks=None, available_actions=None):
        """
        Inserts data from the current timestep into the replay buffer.
        The next observation has already been placed into share_obs, obs, etc.

        :param share_obs: (np.array) Central or shared observation at the next timestep.
        :param obs: (np.array) Local observation at the next timestep.
        :param edge_features: (np.array) Edge features at the next timestep.
        :param indices: (np.array) Traffic threshold indices for the next timestep.
        :param rnn_states_actor: (Unused placeholder) LSTM/GRU states for the actor.
        :param rnn_states_critic: (Unused placeholder) LSTM/GRU states for the critic.
        :param actions: (np.array) Actions taken at the current timestep.
        :param action_log_probs: (np.array) Log probabilities of the current actions.
        :param value_preds: (np.array) Predicted values for the current timestep.
        :param rewards: (np.array) Rewards obtained after the current actions.
        :param masks: (np.array) Unused masks.
        :param topo: (np.array) Topology identifier for the next timestep.
        :param bad_masks: (np.array) Additional masks to handle partial episodes or timeouts.
        :param active_masks: (np.array) Indicates which agents are still active.
        :param available_actions: (Unused placeholder) Available actions for discrete spaces.
        """

        if share_obs is not None: self.share_obs[self.step + 1] = share_obs.copy()
        if obs is not None: self.obs[self.step + 1] = obs.copy()
        if edge_features is not None: self.edge_features[self.step + 1] =edge_features.copy()
        if indices is not None: self.indices[self.step + 1] = indices.copy()
        if topo is not None: self.topos[self.step + 1] = topo.copy()
        if masks is not None: self.masks[self.step + 1] = masks.copy()
        if bad_masks is not None: self.bad_masks[self.step + 1] = bad_masks.copy()
        if active_masks is not None: self.active_masks[self.step + 1] = active_masks.copy()

        if actions is not None: self.actions[self.step] = actions.copy()
        if action_log_probs is not None: self.action_log_probs[self.step] = action_log_probs.copy()
        if value_preds is not None: self.value_preds[self.step] = value_preds.copy()
        if rewards is not None: self.rewards[self.step] = rewards.copy()
        self.step = (self.step + 1) % self.episode_length

    def after_update(self):
        """
        Resets the buffer by moving the final stored observations into the zero-th slot.
        This ensures continuity in the next episode.
        """

        self.share_obs[0] = self.share_obs[-1].copy()
        self.obs[0] = self.obs[-1].copy()
        self.edge_features[0] = self.edge_features[-1].copy()
        self.indices[0] = self.indices[-1].copy()
        self.topos[0] = self.topos[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()
        self.active_masks[0] = self.active_masks[-1].copy()

    def compute_returns(self, next_value, value_normalizer=None):
        """
        Computes returns (and optionally advantages) for each timestep using GAE (Generalized Advantage Estimation).
        :param next_value: (torch.Tensor or np.array) Estimated value for the terminal state.
        :param value_normalizer: (ValueNorm or None) Normalization function for value targets if used.
        """

        self.value_preds[-1] = next_value
        gae = 0
        for step in reversed(range(self.rewards.shape[0])):
            if self._use_popart or self._use_valuenorm:
                delta = self.rewards[step] + self.gamma * value_normalizer.denormalize(
                    self.value_preds[step + 1]) * self.masks[step + 1] \
                        - value_normalizer.denormalize(self.value_preds[step])
                gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae

                # here is a patch for mpe, whose last step is timeout instead of terminate
                if self.env_name == "MPE" and step == self.rewards.shape[0] - 1:
                    gae = 0

                self.advantages[step] = gae
                self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
            else:
                delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * \
                        self.masks[step + 1] - self.value_preds[step]
                gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae

                # here is a patch for mpe, whose last step is timeout instead of terminate
                if self.env_name == "MPE" and step == self.rewards.shape[0] - 1:
                    gae = 0

                self.advantages[step] = gae
                self.returns[step] = gae + self.value_preds[step]

    def feed_forward_generator_transformer(self, advantages, num_mini_batch=None, mini_batch_size=None):
        """
        Generator function that yields mini-batches of data for training.

        :param advantages: (np.array) Computed advantages.
        :param num_mini_batch: (int) Number of mini-batches to split the data into.
        :param mini_batch_size: (int) If not None, uses this size for each mini-batch instead of auto-calculation.

        :yield: Batches of experience (obs, share_obs, actions, returns, etc.) for optimization steps.
        """

        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        batch_size = n_rollout_threads * episode_length
        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(n_rollout_threads, episode_length,
                          n_rollout_threads * episode_length,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch
        rand = torch.randperm(batch_size).numpy()
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]
        rows, cols = _shuffle_agent_grid(batch_size, num_agents)

        share_obs = self.share_obs[:-1].reshape(-1, *self.share_obs.shape[2:])
        share_obs = share_obs[rows, cols]
        obs = self.obs[:-1].reshape(-1, *self.obs.shape[2:])
        obs = obs[rows, cols]
        edge_features = self.edge_features[:-1].reshape(-1, *self.edge_features.shape[2:])
        cur_indices = self.indices[:-1].reshape(-1,*self.indices.shape[2:])
        topos = self.topos[:-1].reshape(-1,*self.topos.shape[2:])

        actions = self.actions.reshape(-1, *self.actions.shape[2:])
        actions = actions[rows, cols]
        value_preds = self.value_preds[:-1].reshape(-1, *self.value_preds.shape[2:])
        value_preds = value_preds[rows, cols]
        returns = self.returns[:-1].reshape(-1, *self.returns.shape[2:])
        returns = returns[rows, cols]
        masks = self.masks[:-1].reshape(-1, *self.masks.shape[2:])
        masks = masks[rows, cols]
        active_masks = self.active_masks[:-1].reshape(-1, *self.active_masks.shape[2:])
        active_masks = active_masks[rows, cols]
        action_log_probs = self.action_log_probs.reshape(-1, *self.action_log_probs.shape[2:])
        action_log_probs = action_log_probs[rows, cols]
        advantages = advantages.reshape(-1, *advantages.shape[2:])
        advantages = advantages[rows, cols]

        for indices in sampler:
            indice_batch = cur_indices[indices].reshape(-1, *cur_indices.shape[2:])
            share_obs_batch = share_obs[indices].reshape(-1, *share_obs.shape[2:])
            obs_batch = obs[indices].reshape(-1, *obs.shape[2:])
            edge_feature_batch = edge_features[indices].reshape(-1, *edge_features.shape[2:])
            topo_batch = topos[indices].reshape(-1, *topos.shape[2:])
            actions_batch = actions[indices].reshape(-1, *actions.shape[2:])
            available_actions_batch = None
            value_preds_batch = value_preds[indices].reshape(-1, *value_preds.shape[2:])
            return_batch = returns[indices].reshape(-1, *returns.shape[2:])
            masks_batch = masks[indices].reshape(-1, *masks.shape[2:])
            active_masks_batch = active_masks[indices].reshape(-1, *active_masks.shape[2:])
            old_action_log_probs_batch = action_log_probs[indices].reshape(-1, *action_log_probs.shape[2:])
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages[indices].reshape(-1, *advantages.shape[2:])
            yield share_obs_batch, obs_batch,edge_feature_batch,indice_batch, None, None, actions_batch, \
                  value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
                  adv_targ, available_actions_batch, topo_batch