import logging

import numpy as np
import torch
import wandb

from utils.network_utils import np2torch
from policy_search.policy_gradient import PolicyGradient
from policy_search.ppo_episode import PPOEpisode

logger = logging.getLogger('root')


class PPO(PolicyGradient):

    def __init__(self, env, config):
        super(PPO, self).__init__(env, config)

    def update_policy(self, observations, actions, advantages, old_logprobs, current_batch):
        """
        Args:
            observations: np array of shape [batch size, dim(observation space)]
            actions: np array of shape
                [batch size, dim(action space)] if continuous
                [batch size] (and integer type) if discrete
            advantages: np array of shape [batch size]
            old_logprobs: np array of shape [batch size]
        """
        observations = np2torch(observations)
        actions = np2torch(actions)
        advantages = np2torch(advantages)
        old_logprobs = np2torch(old_logprobs)

        dist = self.policy.action_distribution(observations, current_batch)

        new_logprobs = dist.log_prob(actions).squeeze()

        # Compute the ratio between new policy and old policy
        ratio = (new_logprobs - old_logprobs).exp()

        clipped_advantage = torch.clamp(ratio, 1.0 - self.config.eps_clip, 1.0 + self.config.eps_clip) * advantages

        loss = -torch.min(ratio * advantages, clipped_advantage).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def merge_episodes_to_batch(self, episodes):
        observations = np.concatenate([episode.observations for episode in episodes])
        old_logprobs = np.concatenate([episode.old_logprobs for episode in episodes])
        actions = np.concatenate([episode.actions for episode in episodes])

        logger.debug(f"chosen actions in entire batch: {actions}")

        # compute Q-val estimates (discounted future returns) for each time step
        returns = self.get_returns(episodes)

        advantages = self.calculate_advantage(returns, observations)

        batch_rewards = np.array([episode.total_reward for episode in episodes])

        return observations, actions, returns, advantages, batch_rewards, old_logprobs

    def train(self):
        for t in range(self.config.num_batches):
            episodes = self.sample_episodes(current_batch=t)
            observations, actions, returns, advantages, batch_rewards, old_logprobs = self.merge_episodes_to_batch(
                episodes)

            for k in range(self.config.update_freq):
                self.baseline_network.update_baseline(returns, observations)
                self.update_policy(observations, actions, advantages, old_logprobs, current_batch=t)

            avg_batch_reward = batch_rewards.mean()
            std_batch_reward = batch_rewards.std()
            msg = "[ITERATION {}]: Average reward: {:04.2f} +/- {:04.2f}".format(
                t+1, avg_batch_reward, std_batch_reward
            )
            logger.info(msg)

            if self.config.run_name:
                wandb.log({"avg_batch_reward": avg_batch_reward, "std_batch_reward": std_batch_reward})

            # test logic
            if t % self.config.test_every == 0:
                self.evaluate()

    def sample_episode(self, current_batch):
        observation = self.env.reset()
        episode = PPOEpisode()
        done = False

        while not done:
            action, old_logprob = self.policy.act(observation.reshape(1, -1), current_batch, return_log_prob=True)
            next_observation, reward, done = self.env.step(action.item())
            episode.add(observation, action.item(), reward, old_logprob.item())
            observation = next_observation

        return episode
