#!/usr/bin/env python
import pickle
import sys

import numpy as np
import torch
from gym.spaces import Box
from tqdm import tqdm

from env.te_env_multitopo import TEEnvMultiTopo

sys.path.append("/")

from env.experiment_utils import ExperimentLoggingManager, Timer
from env.te_env import TEEnv
from algorithms.transformer_policy import TransformerPolicy
from trainer import MATTrainer
from algorithms.shared_buffer import SharedReplayBuffer
from config import get_config

"""Train script for TE."""

def _t2n(x):
    """Convert a tensor to a numpy array."""

    return x.detach().cpu().numpy()

@torch.no_grad()
def compute(env, model, trainer, buffer, config):
    """Calculate returns for the collected data."""

    # Switch model to the evaluation mode
    trainer.prep_rollout()

    # Get the current topology information
    topo = buffer.topos[-1].item()
    kwargs = {} if config.use_single_topo else {key: value for key, value in env.topo_kwargs[topo].items()}

    # Get the values for the next state using the model
    next_values = trainer.policy.get_values(np.concatenate(buffer.share_obs[-1]),
                                            np.concatenate(buffer.obs[-1]),
                                            np.concatenate(buffer.edge_features[-1]),
                                            np.concatenate(buffer.indices[-1]),
                                            np.concatenate(buffer.masks[-1]),
                                            **kwargs)
    next_values = np.array(np.split(_t2n(next_values), 1))

    # Calculate advantages and returns for loss computation using next_values
    buffer.compute_returns(next_values, trainer.value_normalizer)

@torch.no_grad()
def eval(all_args, eval_envs, trainer, total_num_steps, logger, writer):
    """Evaluate the model on the test set and calculate rewards."""

    # Initialize evaluation metrics
    eval_mat_reward = 0 # Multi-agent transformer reward (main evaluation metric)
    test_num = 0 # Number of evaluation steps

    if not all_args.use_single_topo:
        # Reset environment evaluation and initialize tracking variables
        eval_obs, eval_share_obs, eval_edge_feature,eval_indices, topo1 = eval_envs.reset('test')
        topo_reward = {topo: 0 for topo in eval_envs.test_topos} # Cumulative reward for each topology
        topo_rewards = {topo: [] for topo in eval_envs.test_topos} # List of rewards for each topology
        topo_step = {topo: 0 for topo in eval_envs.test_topos} # Number of steps per topology
    else:
        # Legacy option for single topology; now integrated into the multi-topology process
        eval_obs, eval_share_obs, eval_edge_feature, eval_indices = eval_envs.reset('test')

    # Initialize masks for evaluation (No masking)
    eval_masks = np.ones((all_args.eval_episodes, eval_envs.n_agents, 1), dtype=np.float32)

    # Placeholder for actions taken during evaluation
    round_eval_actions = None

    # Progress bar for evaluation steps
    tbar = tqdm(total=len(eval_envs.test_set), smoothing=0)

    while True:
        # Update the progress bar description with the current evaluation example number and topology
        tbar.set_description(f"test cursor:{eval_envs.test_cursor} topo:{topo1.item()} ")

        # Switch the model to the evaluation mode
        trainer.prep_rollout()

        # Set topology-specific parameters
        kwargs = {} if all_args.use_single_topo else eval_envs.topo_kwargs[topo1.item()]

        # Generate actions using the model policy in deterministic mode
        eval_actions = \
            trainer.policy.act(np.concatenate(eval_share_obs),
                                    np.concatenate(eval_obs),
                                    np.concatenate(eval_edge_feature),
                                    np.concatenate(eval_indices),
                                    np.concatenate(eval_masks),
                                    deterministic=True,
                                    **kwargs)
        # Expand the action tensor
        eval_actions = np.expand_dims(eval_actions, axis=0)

        # Execute the action in the environment and step
        A = eval_envs.step(eval_actions, mode='test')
        # Break the loop at the end of the testset
        if A is None:
            break

        if not all_args.use_single_topo:
            # Unpack the environment step results
            eval_obs, eval_share_obs,eval_edge_feature, eval_indices, eval_rewards, _, eval_infos, _, round_eval_actions, compare_dict, topo2 = A
            if eval_rewards is not None:
                eval_rewards = eval_rewards[0, 0, 0]
                topo1 = topo1.item()
                # Accumulate rewards for the current topology
                topo_reward[topo1] += eval_rewards
                topo_rewards[topo1].append(eval_rewards)
                # Increment step count for the topology
                topo_step[topo1] += 1
            # Update topology for the next step
            topo1 = topo2
        else:
            # Legacy option for single topology; now integrated into the multi-topology process
            eval_obs, eval_share_obs,eval_edge_feature, eval_indices, eval_rewards, _, eval_infos, _, round_eval_actions, compare_dict = A
            eval_rewards = eval_rewards[0, 0, 0]
            eval_mat_reward += eval_rewards

        # Increment evaluation step count
        test_num += 1

        # Update progress bar
        tbar.update()

    if not all_args.use_single_topo:
        # Compute average rewards and log metrics
        for topo in topo_step.keys():
            # Compute average rewards and baseline-aligned rewards
            topo_reward[topo] = topo_reward[topo] / topo_step[topo]

            # Log evaluation information and metrics
            logger.log("【Eval】 {}: total num steps is {}.".format(topo, total_num_steps))
            logger.log(f"Eval output action example: {str(round_eval_actions[0, 0, :])}")
            logger.log(f"avg_mat_reward: {topo_reward[topo]}")

            # Draw Plots
            writer.log_metric(f"eval/{topo}_reward", topo_reward[topo], total_num_steps)

            # Save the evaluation reward file
            save_path = f"./log/dataset_{topo}.json_mat-{all_args.obj}-reward.pkl"
            with open(save_path, "wb") as f:
                pickle.dump(topo_rewards[topo] if all_args.obj == "total_flow" else [-reward for reward in topo_rewards[topo]], f)
                print(f"save dataset reward {save_path}")
    else:
        # Legacy option for single topology; now integrated into the multi-topology process
        eval_mat_reward = eval_mat_reward / test_num
        logger.log("【Eval】 total num steps is {}, eval_reward is {}, target_reward is {}.".format(total_num_steps, eval_mat_reward, 1))
        logger.log(f"Eval output action example: {str(round_eval_actions[0,0,:])}")
        writer.log_metric("eval/reward", eval_mat_reward, total_num_steps)

    # Return the evaluation reward (legacy return, unused)
    return eval_mat_reward

def train(env, model, trainer, buffer_or_dict, config, logger, writer, model_saver):
    """Train function: Execute training and validation processes"""

    # # Initialize environment and replay buffer
    if not config.use_single_topo:
        # Environment reset; fetch initial observations
        obs, share_obs, edge_features, indices, topo1 = env.reset()
        buffer_dict = buffer_or_dict
        # Initialize replay buffer for the current topology
        buffer_dict[topo1.item()].topos[0] = topo1.copy()
        buffer_dict[topo1.item()].share_obs[0] = share_obs.copy()
        buffer_dict[topo1.item()].obs[0] = obs.copy()
        buffer_dict[topo1.item()].edge_features[0] = edge_features.copy()
        buffer_dict[topo1.item()].indices[0] = indices.copy()
        topo2 = topo1
    else:
        # Legacy option for single topology; now integrated into the multi-topology process
        buffer = buffer_or_dict
        obs, share_obs, edge_features, indices = env.reset()
        topo2 = topo1 = np.array(config.topo).reshape(1,1)
        buffer.topos[0] = topo1.reshape(1,1).copy()
        buffer.share_obs[0] = share_obs.copy()
        buffer.obs[0] = obs.copy()
        buffer.edge_features[0] = edge_features.copy()
        buffer.indices[0] = indices.copy()

    # Initialize masks for agent availability (No masking, all actions available)
    masks = np.ones((1, env.n_agents, 1), dtype=np.float32)
    active_masks = np.ones((1, env.n_agents, 1), dtype=np.float32)

    # Calculate the number of episodes based on the number of training steps and episode length
    episodes = int(config.num_env_steps) // (config.episode_length-1) // 1
    episode_length = config.episode_length

    # Initialize timer for performance tracking
    timer = Timer()

    for episode in range(episodes):
        print(f"\nepisode {episode}\n")
        timer.start()
        total_num_steps = episode * episode_length
        total_rewards = []  # Track total rewards
        train_bar = tqdm(range(episode_length))
        episode_topos = set() # Track visited topologies in the episode
        with torch.no_grad():
            for step in train_bar:
                # Perform action selection
                topo1 = topo2
                train_bar.set_description(f"train cursor:{env.train_cursor} topo:{topo1.item()} ")
                trainer.prep_rollout()
                episode_topos.add(topo1.item())
                if config.use_single_topo:
                    # Legacy option for single topology; now integrated into the multi-topology process
                    value, action, action_log_prob \
                        = model.get_actions(np.concatenate(buffer.share_obs[step]),
                                                    np.concatenate(buffer.obs[step]),
                                                    np.concatenate(buffer.edge_features[step]),
                                                    np.concatenate(buffer.indices[step]),
                                                    np.concatenate(buffer.masks[step]))
                else:
                    # Select action by the multi-agent transformer model for the current observation
                    kwargs = {key: value for key, value in env.topo_kwargs[topo1.item()].items()}
                    value, action, action_log_prob \
                        = model.get_actions(np.concatenate(buffer_dict[topo1.item()].share_obs[step]),
                                                    np.concatenate(buffer_dict[topo1.item()].obs[step]),
                                                    np.concatenate(buffer_dict[topo1.item()].edge_features[step]),
                                                    np.concatenate(buffer_dict[topo1.item()].indices[step]),
                                                    np.concatenate(buffer_dict[topo1.item()].masks[step]),
                                                    **kwargs)
                # Convert result tensors to numpy arrays
                values = np.array(np.split(_t2n(value), 1))
                actions = np.array(np.split(_t2n(action), 1))
                action_log_probs = np.array(np.split(_t2n(action_log_prob), 1))

                if not config.use_single_topo:
                    # Execute action in environment, update environment state, retrieve results and new observation
                    obs, share_obs,edge_features, indices, rewards, dones, infos, available_actions, round_actions, compare_dict, topo2 = env.step(actions)
                    if topo1 == topo2:
                        # Topology remains the same in next step; update the same buffer
                        buffer_dict[topo1.item()].insert(share_obs, obs, edge_features, indices, None, None, round_actions,
                                      action_log_probs, values, rewards, masks, topo1, None, active_masks, None)
                    else:
                        # Topology changes in next step; update the corresponding buffer
                        buffer_dict[topo2.item()].step = 0
                        buffer_dict[topo2.item()].topos[0] = topo2.copy()
                        buffer_dict[topo2.item()].share_obs[0] = share_obs.copy()
                        buffer_dict[topo2.item()].obs[0] = obs.copy()
                        buffer_dict[topo2.item()].edge_features[0] = edge_features.copy()
                        buffer_dict[topo2.item()].indices[0] = indices.copy()
                        active_masks = np.ones((1, env.topo_kwargs[topo2.item()]['n_agents'], 1), dtype=np.float32)
                        masks = np.ones((1, env.topo_kwargs[topo2.item()]['n_agents'], 1), dtype=np.float32)
                        buffer_dict[topo2.item()].active_masks[0] = active_masks.copy()
                        buffer_dict[topo2.item()].masks[0] = masks.copy()
                else:
                    # Legacy option for single topology; now integrated into the multi-topology process
                    obs, share_obs,edge_features,indices, rewards, dones, infos, available_actions, round_actions, compare_dict = env.step(actions, mode='train')
                    buffer.insert(share_obs, obs, edge_features, indices, None, None, round_actions,
                                  action_log_probs, values, rewards, masks, np.array(topo1).reshape((1,1)), None, active_masks, None)

                # Update metrics
                total_num_steps += 1
                if rewards is not None:
                    total_rewards.append(np.mean(rewards))

        if config.use_single_topo:
            # Legacy option for single topology; now integrated into the multi-topology process
            compute(env, model, trainer, buffer, config)
            trainer.prep_training()
            train_infos = trainer.train(buffer)
            if topo1 == topo2:
                buffer.after_update()
        else:
            for topo, buffer in buffer_dict.items():
                # Perform model training for each topology in the episode using the corresponding buffer
                # Topology is set to change at the end of the episode, so this loop will be executed once
                if topo not in episode_topos:
                    continue
                # Compute advantage, value, return
                compute(env, model, trainer, buffer, config)
                # Switch the model to training mode
                trainer.prep_training()
                # Perform model training on the current topology's buffer
                train_infos = trainer.train(buffer)
                # Reset the buffer
                buffer.after_update()
        # Stop the timer
        timer.stop()

        if episode % config.log_interval == 0:
            # Log the training information
            logger.log("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}, time_each_eposide: {:.2f}s.\n"
                  .format(config.topo,
                          config.algorithm_name,
                          config.experiment_name,
                          episode,
                          episodes,
                          total_num_steps,
                          config.num_env_steps,
                          1,
                          timer.avg()))
            # Reset the timer for the next interval
            timer.reset()

            # Log the average reward ratio and the total loss
            train_infos["loss"] = train_infos["value_loss"] * config.value_loss_coef + train_infos["policy_loss"] - train_infos["dist_entropy"] * config.entropy_coef
            logger.log(train_infos)

            # Draw plots
            writer.log_metric('train/loss', train_infos["loss"], total_num_steps)
            writer.log_metric('train/value_loss', train_infos["value_loss"], total_num_steps)
            writer.log_metric('train/policy_loss', train_infos["policy_loss"], total_num_steps)
            writer.log_metric('train/dist_entropy', train_infos["dist_entropy"], total_num_steps)
            writer.log_metric('train/gnn_grad_norm', train_infos["gnn_grad_norm"], total_num_steps)
            writer.log_metric('train/critic_grad_norm', train_infos["critic_grad_norm"], total_num_steps)
            writer.log_metric('train/actor_grad_norm', train_infos["actor_grad_norm"], total_num_steps)
            writer.log_metric('train/average_step_reward_ratio', train_infos["average_step_reward_ratio"], total_num_steps)

    # Evaluate the model after the training process
    eval(config, env, trainer, total_num_steps, logger, writer)

def main():
    """Main Function: Train the Aether model and evaluate its performance"""

    # Initialize a logging tool for experiment tracking
    logging_manager = ExperimentLoggingManager()
    logger = logging_manager.get_logger("te")
    # Set up a TensorBoard writer for visualizing training metrics
    writer = logging_manager.get_writer("te")
    # Initialize a model checkpoint saver to manage and save model states
    model_saver = logging_manager.get_model_saver("mat")

    # Parse command-line arguments and configuration settings
    all_args = get_config()

    # Add an extra episode for shared buffer topology switching if multiple topologies are being used
    if not all_args.use_single_topo:
        all_args.episode_length = all_args.episode_length + 1

    # Log the configuration settings for reference
    logger.log(all_args)

    # Select computation device: CPU or GPU
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")

        # Enable deterministic behavior if specified
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")

    # Set the random seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # Create a simulation environment
    if all_args.use_single_topo:
        env = TEEnv(all_args, logger)
    else:
        env = TEEnvMultiTopo(all_args, logger)

    # Define model input dimensions
    observation_space = Box(low=0, high=100000, shape=(env.obs_dim,))
    share_observation_space = Box(low=0, high=100000, shape=(env.share_obs_dim,))
    action_space = Box(low=0, high=100000, shape=(env.action_dim,))

    # Initialize the multi-agent transformer model
    model = TransformerPolicy(env,all_args,
                     observation_space,
                     share_observation_space,
                     action_space,
                     device=device)

    # Create the Aether trainer object for managing training processes
    trainer = MATTrainer(all_args, model, env, device=device)

    if all_args.use_single_topo:
        # Legacy option for single topology; now integrated into the multi-topology process
        buffer = SharedReplayBuffer(all_args,
                                         env.n_agents,
                                         env.num_edge_node,
                                         observation_space,
                                         share_observation_space,
                                         action_space,
                                         all_args.env_name)
        train(env, model, trainer, buffer, all_args, logger, writer, model_saver)
    else:
        # Initialize the replay buffer
        buffer_dict = {topo: SharedReplayBuffer(all_args,
                                         env.topo_kwargs[topo]['n_agents'],
                                         env.topo_kwargs[topo]['num_edge'],
                                         observation_space,
                                         share_observation_space,
                                         action_space,
                                         all_args.env_name) for topo in env.all_topos}
        # Execute training and validation processes
        train(env, model, trainer, buffer_dict, all_args, logger, writer, model_saver)

    # Close the environment and logging tools
    env.close()
    logging_manager.close_all()


if __name__ == "__main__":
    # Run the main function
    main()