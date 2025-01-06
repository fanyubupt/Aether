import torch
from torch.distributions import Categorical, Normal
from torch.nn import functional as F

def continuous_autoregreesive_act(decoder,decision_unit, obs_rep, obs, batch_size, action_dim, tpdv,
                                  deterministic=False):
    """
    Generates continuous actions autoregressively based on the given observations.

    :param decoder: (nn.Module) Decoder component used to generate action means.
    :param decision_unit: (int) Number of decision units.
    :param obs_rep: (torch.Tensor) Encoded observation representations.
    :param obs: (torch.Tensor) Original observations.
    :param batch_size: (int) Number of sequences in the batch.
    :param action_dim: (int) Dimension of each action.
    :param tpdv: (dict) Dictionary containing default tensor dtype and device.
    :param deterministic: (bool) If True, outputs the mean of the distribution; otherwise samples from Normal.

    :return:
        - output_action: (torch.Tensor) The generated actions.
        - output_action_log: (torch.Tensor) The log probabilities of generated actions.
    """

    shifted_action = torch.zeros((batch_size, obs_rep.shape[1], action_dim*decision_unit)).to(**tpdv)
    output_action = torch.zeros((batch_size, obs_rep.shape[1], action_dim*decision_unit), dtype=torch.float32)
    output_action_log = torch.zeros_like(output_action, dtype=torch.float32)
    for i in range(obs_rep.shape[1]):
        act_mean = decoder(shifted_action, obs_rep, obs)[:, i, :]
        action_std = torch.sigmoid(decoder.log_std) * 0.5

        distri = Normal(act_mean, action_std)
        action = act_mean if deterministic else distri.sample()
        action_log = distri.log_prob(action)

        output_action[:, i, :] = action
        output_action_log[:, i, :] = action_log
        if i + 1 < obs_rep.shape[1]:
            shifted_action[:, i + 1, :] = action

    return output_action, output_action_log

def find_factor(col_len):
    """Finds the smallest integer factor > 8 of the given number `col_len`."""

    factor = 8
    # If 8 is already a factor, then return directly
    if col_len % 8 == 0:
        return 8
    # Otherwise, start searching upwards from 8
    while True:
        factor += 1
        if col_len % factor == 0:
            return factor

def continuous_autoregressive_act_approx(decoder, decision_unit, obs_rep, obs, batch_size, action_dim, tpdv, n_nodes, deterministic=False):
    """
    Generates continuous actions autoregressively using segmented attention for large node counts.
    The sequence is split into segments (if needed), and the decoder processes these segments one by one.

    :param decoder: (nn.Module) Decoder component used to generate action means.
    :param decision_unit: (int) Number of decision units in each action output.
    :param obs_rep: (torch.Tensor) Encoded observation representations.
    :param obs: (torch.Tensor) Original observations.
    :param batch_size: (int) Number of sequences in the batch.
    :param action_dim: (int) Dimension of each action.
    :param tpdv: (dict) Dictionary containing default tensor dtype and device.
    :param n_nodes: (int) Number of nodes (may trigger segment-based processing if too large).
    :param deterministic: (bool) If True, uses the mean of the distribution; otherwise samples from Normal.

    :return:
        - output_action: (torch.Tensor) Generated actions.
        - output_action_log: (torch.Tensor) Log probabilities for each generated action.
    """

    if n_nodes > 200:
        total_segments = find_factor(obs_rep.shape[1])
        segment_size = obs_rep.shape[1] // total_segments
    else:
        segment_size = n_nodes - 1
        total_segments = obs_rep.shape[1] // segment_size

    shifted_action = torch.zeros((batch_size, obs_rep.shape[1], action_dim*decision_unit)).to(**tpdv)
    output_action = torch.zeros((batch_size, obs_rep.shape[1], action_dim*decision_unit), dtype=torch.float32)
    output_action_log = torch.zeros_like(output_action, dtype=torch.float32)

    for segment in range(total_segments):
        start_idx = segment * segment_size
        end_idx = start_idx + segment_size

        act_mean = decoder(shifted_action, obs_rep, obs)[:, start_idx:end_idx, :]
        action_std = torch.sigmoid(decoder.log_std) * 0.5

        distri = Normal(act_mean, action_std)
        action = act_mean if deterministic else distri.sample()
        action_log = distri.log_prob(action)

        output_action[:, start_idx:end_idx, :] = action
        output_action_log[:, start_idx:end_idx, :] = action_log

        if segment < total_segments - 1:
            shifted_action[:, end_idx:end_idx + segment_size, :] = action

    return output_action, output_action_log

def continuous_parallel_act(decoder,decision_unit, obs_rep, obs, action, batch_size, action_dim, tpdv):
    """
    Generates continuous actions in a parallel manner. The previous actions are prepared
    in a 'shifted' buffer to emulate autoregressive conditioning, and then the model
    generates outputs all at once.

    :param decoder: (nn.Module) Decoder component used to generate action means.
    :param decision_unit: (int) Number of decision units in each action.
    :param obs_rep: (torch.Tensor) Encoded observation representations.
    :param obs: (torch.Tensor) Original observations.
    :param action: (torch.Tensor) Input actions.
    :param batch_size: (int) Number of sequences in the batch.
    :param action_dim: (int) Dimension of each action.
    :param tpdv: (dict) Dictionary containing default tensor dtype and device.
    :param label_action: (torch.Tensor or None) Unused Optional labeled action data.

    :return:
       - action_log: (torch.Tensor) Log probabilities for each action in the batch.
       - entropy: (torch.Tensor) Entropy of the action distribution.
    """

    shifted_action = torch.zeros((batch_size, obs_rep.shape[1], action_dim*decision_unit)).to(**tpdv)
    action = action.reshape(action.shape[0],-1,action.shape[2]*decision_unit)
    shifted_action[:, 1:, :] = action[:, :-1, :]
    act_mean = decoder(shifted_action, obs_rep, obs)
    action_std = torch.sigmoid(decoder.log_std) * 0.5
    distri = Normal(act_mean, action_std)
    action_log = distri.log_prob(action)
    entropy = distri.entropy()
    return action_log, entropy
