import numpy as np
import torch
import os

import gym
from gym.spaces import Discrete
from general import get_logger, Progbar, export_plot
from baseline_network import BaselineNetwork
from network_utils import build_mlp, device, np2torch
from policy import CategoricalPolicy, GaussianPolicy


class PolicyGradient(object):
    """
    Class for implementing a policy gradient algorithm
    """

    def __init__(self, env, config, seed, logger=None):
        """
        Initialize Policy Gradient Class

        Args:
                env: an OpenAI Gym environment
                config: class with hyperparameters
                logger: logger instance from the logging module
        """
        # directory for training outputs
        if not os.path.exists(config.output_path):
            os.makedirs(config.output_path)

        # store hyperparameters
        # TODO - move seed to config
        self.config = config
        self.seed = seed

        self.logger = logger
        if logger is None:
            self.logger = get_logger(config.log_path)
        self.env = env
        self.env.seed(self.seed)

        # discrete vs continuous action space
        self.discrete = isinstance(env.action_space, Discrete)
        # TODO - should fix things here for continuous action space
        self.observation_dim = self.env.observation_space.shape[0]
        self.action_dim = (
            self.env.action_space.n if self.discrete else self.env.action_space.shape[0]
        )

        self.lr = self.config.learning_rate

        self.init_policy()

        if config.use_baseline:
            self.baseline_network = BaselineNetwork(env, config)

    def init_policy(self):
        # 1. Create a neural network
        self.network = build_mlp(
            input_size=self.observation_dim,
            output_size=self.action_dim,
            n_layers=self.config.n_layers,
            size=self.config.layer_size
        ).to(device)

        # 2. Instantiate the correct policy
        if self.discrete:
            self.policy = CategoricalPolicy(self.network)
        else:
            self.policy = GaussianPolicy(self.network, self.action_dim)

        # 3. Create an Adam optimizer
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

    def init_averages(self):
        self.avg_reward = 0.0
        self.max_reward = 0.0
        self.std_reward = 0.0
        self.eval_reward = 0.0

    def update_averages(self, rewards, scores_eval):
        """
        Update the averages.

        Args:
            rewards: deque
            scores_eval: list
        """
        self.avg_reward = np.mean(rewards)
        self.max_reward = np.max(rewards)
        self.std_reward = np.sqrt(np.var(rewards) / len(rewards))

        if len(scores_eval) > 0:
            self.eval_reward = scores_eval[-1]

    def record_summary(self, t):
        pass

    # TODO - remove this
    # def collect_episode(self, env):
    #     state = env.reset()
    #     done = False
    #     states = []
    #     actions = []
    #     rewards = []
    #     episode_len = 0  # TODO - remove this
    #
    #     while not done:
    #         episode_len += 1  # TODO - remove this
    #         if episode_len > 10:
    #             action = 0
    #         else:
    #             action = self.get_action(state)
    #         next_state, reward, done, _ = env.step(action)
    #         states.append(state)
    #         actions.append(action)
    #         rewards.append(reward)
    #         state = next_state
    #
    #     return states, actions, rewards

    def sample_single_episode(self, env):
        state = env.reset()
        states, actions, rewards = [], [], []
        episode_reward = 0
        done = False

        while not done:
            states.append(state)
            action = self.policy.act(states[-1]).item()
            actions.append(action)
            state, reward, done, _ = env.step(action)
            rewards.append(reward)
            episode_reward += reward

        path = {
            "observation": torch.stack(states),
            "reward": torch.tensor(rewards),
            "action": torch.tensor(actions),
        }
        return path, episode_reward

    # TODO - rename to sample_episodes
    # TODO - thing if I can generate a batch of episodes at once
    # TODO - remove this finalize option
    # TODO - info is actually the generated answer
    def sample_paths(self, env, num_episodes=None):
        episode = 0
        episode_rewards = []
        paths = []
        t = 0

        while num_episodes or t < self.config.batch_size:
            path, episode_reward = self.sample_single_episode(env)
            t += len(path["observation"])  # Update the count with the number of steps in the episode
            paths.append(path)
            episode_rewards.append(episode_reward)
            episode += 1
            if num_episodes and episode >= num_episodes:
                break

        return paths, episode_rewards

    def get_returns(self, paths):
        """
        Calculate the returns G_t for each timestep

        Args:
            paths: recorded sample paths. See sample_paths() for details.

        Return:
            returns: return G_t for each timestep
        """

        all_returns = []
        for path in paths:
            rewards = path["reward"]
            returns = []

            # Initialize G_t as 0
            G_t = 0
            for r in reversed(rewards):
                # Calculate G_t using the formula
                G_t = r + self.config.gamma * G_t
                returns.insert(0, G_t)

            all_returns.append(returns)

        returns = np.concatenate(all_returns)
        return returns

    def normalize_advantage(self, advantages):
        """
        Args:
            advantages: np.array of shape [batch size]
        Returns:
            normalized_advantages: np.array of shape [batch size]
        """
        mean_advantage = np.mean(advantages)
        std_advantage = np.std(advantages)
        # Adding a small epsilon to avoid division by zero
        normalized_advantages = (advantages - mean_advantage) / (std_advantage + 1e-8)
        return normalized_advantages

    def calculate_advantage(self, returns, observations):
        """
        Calculates the advantage for each of the observations
        Args:
            returns: np.array of shape [batch size]
            observations: np.array of shape [batch size, dim(observation space)]
        Returns:
            advantages: np.array of shape [batch size]
        """
        if self.config.use_baseline:
            # override the behavior of advantage by subtracting baseline
            advantages = self.baseline_network.calculate_advantage(
                returns, observations
            )
        else:
            advantages = returns

        if self.config.normalize_advantage:
            advantages = self.normalize_advantage(advantages)

        return advantages

    def update_policy(self, observations, actions, advantages):
        """
        Args:
            observations: np.array of shape [batch size, dim(observation space)]
            actions: np.array of shape
                [batch size, dim(action space)] if continuous
                [batch size] (and integer type) if discrete
            advantages: np.array of shape [batch size]
        """
        observations = np2torch(observations)
        actions = np2torch(actions)
        advantages = np2torch(advantages)

        # Get log probabilities of the actions
        action_dists = self.policy.action_distribution(observations)
        log_probs = action_dists.log_prob(actions)

        # Zero out the gradients from the previous pass
        self.optimizer.zero_grad()

        # Compute the loss function
        loss = -(log_probs * advantages).mean()

        # Backward pass to compute gradients and update the policy
        loss.backward()
        self.optimizer.step()

    def train(self):
        last_record = 0

        self.init_averages()
        all_total_rewards = (
            []
        )  # the returns of all episodes samples for training purposes
        averaged_total_rewards = []  # the returns for each iteration

        for t in range(self.config.num_batches):

            # collect a minibatch of samples
            # TODO - currently num_episodes=10 until figure out a way to parallel sampling path
            paths, total_rewards = self.sample_paths(self.env, num_episodes=10)
            all_total_rewards.extend(total_rewards)
            observations = np.concatenate([path["observation"] for path in paths])
            actions = np.concatenate([path["action"] for path in paths])
            # TODO - maybe remove this
            # rewards = np.concatenate([path["reward"] for path in paths])
            # compute Q-val estimates (discounted future returns) for each time step
            returns = self.get_returns(paths)

            # advantage will depend on the baseline implementation
            advantages = self.calculate_advantage(returns, observations)

            # run training operations
            if self.config.use_baseline:
                self.baseline_network.update_baseline(returns, observations)
            self.update_policy(observations, actions, advantages)

            # logging
            if t % self.config.summary_freq == 0:
                self.update_averages(total_rewards, all_total_rewards)
                self.record_summary(t)

            # compute reward statistics for this batch and log
            avg_reward = np.mean(total_rewards)
            sigma_reward = np.sqrt(np.var(total_rewards) / len(total_rewards))
            msg = "[ITERATION {}]: Average reward: {:04.2f} +/- {:04.2f}".format(
                t, avg_reward, sigma_reward
            )
            averaged_total_rewards.append(avg_reward)
            self.logger.info(msg)

            if self.config.record and (last_record > self.config.record_freq):
                self.logger.info("Recording...")
                last_record = 0
                self.record()

        self.logger.info("- Training done.")
        np.save(self.config.scores_output, averaged_total_rewards)
        export_plot(
            averaged_total_rewards,
            "Score",
            self.config.env_name,
            self.config.plot_output,
        )

    def evaluate(self, env=None, num_episodes=1):
        if env == None:
            env = self.env
        paths, rewards = self.sample_paths(env, num_episodes)
        avg_reward = np.mean(rewards)
        sigma_reward = np.sqrt(np.var(rewards) / len(rewards))
        msg = "Average reward: {:04.2f} +/- {:04.2f}".format(avg_reward, sigma_reward)
        self.logger.info(msg)
        return avg_reward

    def record(self):
        """
        Recreate an env and record a video for one episode
        """
        env = gym.make(self.config.env_name)
        env.seed(self.seed)
        env = gym.wrappers.Monitor(
            env, self.config.record_path, video_callable=lambda x: True, resume=True
        )
        self.evaluate(env, 1)

    def run(self):
        # TODO think of removing record option
        """
        Apply procedures of training for a PG.
        """
        # record one game at the beginning
        if self.config.record:
            self.record()
        # model
        print("Training started...")
        self.train()
        print("Training finished...")
        # record one game at the end
        if self.config.record:
            self.record()
