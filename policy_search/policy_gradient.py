import logging
import os

import numpy as np
import torch

from baseline_network import BaselineNetwork
from network_utils import build_mlp, device, np2torch
from policy import CategoricalPolicy
from policy_search.episode import Episode
from torchsummary import summary

from utils.utils import CaptureStdout

logger = logging.getLogger('root')


# TODO - remove unneeded imports
class PolicyGradient(object):
    """
    Class for implementing a policy gradient algorithm
    """

    def __init__(self, env, config, logger):
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
        self.config = config

        # TODO - check this logger, use my own logger instead
        self.logger = logger
        self.env = env

        # discrete vs continuous action space
        # TODO - should fix things here for continuous action space
        self.observation_dim, self.action_dim = self.env.observation_space.shape[0], self.env.action_space.n
        self.lr = self.config.learning_rate

        self.init_policy()

        if config.baseline:
            self.baseline_network = BaselineNetwork(self.env, config)

        with CaptureStdout() as capture:
            summary(self.policy.network, input_size=(self.observation_dim,))

        logger.info(f"Policy initialized with:"
                    f"\n{capture.get_output()}"
                    f"{self.optimizer=})"
                    f"\n{self.observation_dim=}, "
                    f"{self.action_dim=}, "
                    f"{self.lr=}")

    def init_policy(self):
        self._network = build_mlp(
            input_size=self.observation_dim,
            output_size=self.action_dim,
            n_layers=self.config.n_layers,
            size=self.config.layer_size
        ).to(device)

        self.policy = CategoricalPolicy(self._network).to(device)

        # TODO - used to have GaussianPolicy here

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

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

    # TODO - remove this method
    def record_summary(self, t):
        pass

    # TODO - maybe create an episode class
    def sample_episode(self):
        observation = self.env.reset()
        episode = Episode()
        done = False

        # TODO - this have to be batched and the episode.add should be a numpy operation
        while not done:
            action, _ = self.policy.act(observation.reshape(1, -1))
            next_observation, reward, done, _ = self.env.step(action.item())
            episode.add(observation, action.item(), reward)
            observation = next_observation

        return episode

    # TODO - thing if I can generate a batch of episodes at once
    # TODO - info is actually the generated answer
    def sample_episodes(self):
        episodes = []
        t = 0

        while t < self.config.batch_size:
            episode = self.sample_episode()
            t += len(episode)
            episodes.append(episode)

        return episodes

    def get_returns(self, episodes):
        """
        Calculate the discounted cumulative returns G_t for each timestep in the provided episodes.

        Args:
            episodes (list): A list of episodes. Each episode is expected to have a 'rewards' attribute
                             which is a np.array of the corresponding rewards for each timestep in the episode.

        Returns:
            np.array: A np.array containing the discounted cumulative returns G_t for each timestep
                          across all episodes. The array shape is (total_timesteps), where
                          total_timesteps is the sum of the number of timesteps across all episodes.
        """

        all_returns = []
        for episode in episodes:
            rewards = episode.rewards
            returns = np.zeros_like(rewards)

            G_t = 0
            for t in reversed(range(len(rewards))):
                G_t = rewards[t] + self.config.gamma * G_t
                returns[t] = G_t
            all_returns.append(returns)

        # Stack all the returns into a single tensor
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
        if self.config.baseline:
            # override the behavior of advantage by subtracting baseline
            advantages = self.baseline_network.calculate_advantage(
                returns, observations
            )
        else:
            advantages = returns  # baseline is 0 in case of no baseline

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

    def merge_episodes_to_batch(self, episodes):
        observations = np.concatenate([episode.observations for episode in episodes])
        actions = np.concatenate([episode.actions for episode in episodes])

        # compute Q-val estimates (discounted future returns) for each time step
        returns = self.get_returns(episodes)

        # advantage will depend on the baseline implementation
        advantages = self.calculate_advantage(returns, observations)

        return observations, actions, returns, advantages

    # TODO - add checkpoint logic and save model every x timestamps
    def train(self):
        for t in range(self.config.num_batches):
            episodes = self.sample_episodes()
            observations, actions, returns, advantages = self.merge_episodes_to_batch(episodes)

            # run training operations
            if self.config.baseline:
                self.baseline_network.update_baseline(returns, observations)

            self.update_policy(observations, actions, advantages)

    def evaluate(self, env=None, num_episodes=1):
        pass
        # if env == None:
        #     env = self.env
        # paths, rewards = self.sample_paths(env, num_episodes)
        # avg_reward = np.mean(rewards)
        # sigma_reward = np.sqrt(np.var(rewards) / len(rewards))
        # msg = "Average reward: {:04.2f} +/- {:04.2f}".format(avg_reward, sigma_reward)
        # self.logger.info(msg)
        # return avg_reward

    def record(self):
        pass
        # """
        # Recreate an env and record a video for one episode
        # """
        # env = gym.make(self.config.env_name)
        # env.seed(self.seed)
        # env = gym.wrappers.Monitor(
        #     env, self.config.record_path, video_callable=lambda x: True, resume=True
        # )
        # self.evaluate(env, 1)

    def run(self):
        """
        Apply procedures of training for a PG.
        """
        # record one game at the beginning
        # TODO think of removing record option
        # if self.config.record:
        #     self.record()
        # model
        logger.info("Training started...")
        self.train()
        logger.info("Training completed...")
        # record one game at the end
        # TODO think of removing record option

        # if self.config.record:
        #     self.record()
