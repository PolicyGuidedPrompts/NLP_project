import logging

import numpy as np
import torch

from utils.network_utils import np2torch
from policy_search.policy_gradient import PolicyGradient
from policy_search.ppo_episode import PPOEpisode

logger = logging.getLogger('root')


class PPO(PolicyGradient):

    def __init__(self, env, config):
        super(PPO, self).__init__(env, config)

    def update_policy(self, observations, actions, advantages, old_logprobs):
        """
        Args:
            observations: np.array of shape [batch size, dim(observation space)]
            actions: np.array of shape
                [batch size, dim(action space)] if continuous
                [batch size] (and integer type) if discrete
            advantages: np.array of shape [batch size]
            old_logprobs: np.array of shape [batch size]
        """
        observations = np2torch(observations)
        actions = np2torch(actions)
        advantages = np2torch(advantages)
        old_logprobs = np2torch(old_logprobs)

        # Get the distribution of actions under the current policy
        dist = self.policy.action_distribution(observations)

        # Compute log probabilities for the actions
        new_logprobs = dist.log_prob(actions).squeeze()

        # Compute the ratio between new policy and old policy
        ratio = (new_logprobs - old_logprobs).exp()

        # Compute clipped objective
        clipped_advantage = torch.clamp(ratio, 1.0 - self.config.eps_clip, 1.0 + self.config.eps_clip) * advantages

        # Compute the PPO objective function
        loss = -torch.min(ratio * advantages, clipped_advantage).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def merge_episodes_to_batch(self, episodes):
        observations = np.concatenate([episode.observations for episode in episodes])
        old_logprobs = np.concatenate([episode.old_logprobs for episode in episodes])
        actions = np.concatenate([episode.actions for episode in episodes])

        # compute Q-val estimates (discounted future returns) for each time step
        returns = self.get_returns(episodes)

        # advantage will depend on the baseline implementation
        advantages = self.calculate_advantage(returns, observations)

        batch_rewards = np.array([episode.total_reward for episode in episodes])

        return observations, actions, returns, advantages, batch_rewards, old_logprobs

    def train(self):
        averaged_total_rewards = []

        for t in range(self.config.num_batches):
            episodes = self.sample_episodes()
            observations, actions, returns, advantages, batch_rewards, old_logprobs = self.merge_episodes_to_batch(
                episodes)

            # run training operations
            for k in range(self.config.update_freq):
                self.baseline_network.update_baseline(returns, observations)
                self.update_policy(observations, actions, advantages,
                                   old_logprobs)

            avg_batch_reward = batch_rewards.mean()
            msg = "[ITERATION {}]: Average reward: {:04.2f} +/- {:04.2f}".format(
                t, avg_batch_reward, batch_rewards.std()
            )
            averaged_total_rewards.append(avg_batch_reward)
            logger.info(msg)

    def sample_episode(self):
        observation = self.env.reset()
        episode = PPOEpisode()
        done = False

        # TODO - this have to be batched and the episode.add should be a numpy operation
        while not done:
            action, old_logprob = self.policy.act(observation.reshape(1, -1), return_log_prob=True)
            next_observation, reward, done, _ = self.env.step(action.item())
            episode.add(observation, action.item(), reward, old_logprob.item())
            observation = next_observation

        return episode

    # TODO - use Episode class
    # TODO - multiple places using env instead of self.env
    def sample_episodes(self):
        episodes = []
        t = 0

        while t < self.config.batch_size:
            episode = self.sample_episode()
            t += len(episode)
            episodes.append(episode)

        return episodes
