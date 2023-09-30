import torch
import torch.nn as nn
import numpy as np

from utils.network_utils import np2torch


class BasePolicy:
    def action_distribution(self, observations, current_batch):
        """
        Args:
            observations: torch.Tensor of shape [batch size, dim(observation space)]
        Returns:
            distribution: instance of a subclass of torch.distributions.Distribution

        See https://pytorch.org/docs/stable/distributions.html#distribution

        This is an abstract method and must be overridden by subclasses.
        It will return an object representing the policy's conditional
        distribution(s) given the observations. The distribution will have a
        batch shape matching that of observations, to allow for a different
        distribution for each observation in the batch.
        """
        raise NotImplementedError

    def act(self, observation, current_batch, return_log_prob=False):
        """
        Args:
            observation: np array of shape (1, dim(observation space))
        Returns:
            sampled_action: np array of shape (1, *shape of action)
            log_probs: np array of shape [batch size] (optionally, if return_log_prob)
        """
        observation = np2torch(observation)
        action_distribution = self.action_distribution(observation, current_batch)

        sampled_action = action_distribution.sample()

        # Used only to collect log_probs for old policy, not for the one being trained, hence detach().numpy()
        log_probs = action_distribution.log_prob(sampled_action).detach().cpu().numpy() if return_log_prob else None

        return sampled_action.detach().cpu().numpy(), log_probs

    def test_act(self, observation):
        """
        Args:
            observation: np array of shape (1, dim(observation space))
        Returns:
            sampled_action: np array of shape (1, *shape of action)
        """
        observation = np2torch(observation)

        logits = self.network(observation)
        action_distribution = torch.distributions.Categorical(logits=logits)

        sampled_action = action_distribution.sample()

        return sampled_action.detach().cpu().numpy()


class CategoricalPolicy(BasePolicy, nn.Module):
    def __init__(self, network, config):
        nn.Module.__init__(self)
        self.config = config
        self.network = network
        self.epsilon = 1.0

    def action_distribution(self, observations, current_batch):
        """
        Args:
            observations: torch.Tensor of shape [batch size, dim(observation space)]
        Returns:
            distribution: torch.distributions.Categorical where the logits
                are computed by self.network
        """
        logits = self.network(observations)
        softmax_temperature = self._get_softmax_temperature(current_batch)
        distribution = torch.distributions.Categorical(logits=logits / softmax_temperature)
        return distribution

    def _get_softmax_temperature(self, current_batch):
        if self.config.policy_exploration_logic == 'linear_temperature_decay':
            return self._get_linear_softmax_temperature(current_batch)
        elif self.config.policy_exploration_logic == 'exponential_temperature_decay':
            return self._get_exp_softmax_temperature(current_batch)
        elif self.config.policy_exploration_logic == 'epsilon_greedy':
            return self._get_epsilon_greedy_softmax_temperature(current_batch)
        else:
            raise NotImplementedError

    def _get_exp_softmax_temperature(self, current_batch):
        return self.config.initial_temperature * (self.config.exploration_decay_factor ** current_batch) + 1.0

    def _get_linear_softmax_temperature(self, current_batch):
        return self.config.initial_temperature + (self.config.end_temperature - self.config.initial_temperature) * (
                current_batch / self.config.num_batches)

    def _get_epsilon_greedy_softmax_temperature(self, current_batch):
        self.epsilon = self.config.exploration_decay_factor**current_batch
        if np.random.rand() < self.epsilon:
            return float('inf')
        else:
            return self.config.end_temperature
