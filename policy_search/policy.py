import torch
import torch.nn as nn

from utils.network_utils import np2torch


class BasePolicy:
    def action_distribution(self, observations):
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

    def act(self, observation, return_log_prob=False):
        """
        Args:
            observation: np.array of shape (1, dim(observation space))
        Returns:
            sampled_action: np.array of shape (1, *shape of action)
            log_probs: np.array of shape [batch size] (optionally, if return_log_prob)
        """
        observation = np2torch(observation)
        action_distribution = self.action_distribution(observation)

        sampled_action = action_distribution.sample()

        # Used only to collect log_probs for old policy, not for the one being trained, hence detach().numpy()
        log_probs = action_distribution.log_prob(sampled_action).detach().cpu().numpy() if return_log_prob else None

        return sampled_action.detach().cpu().numpy(), log_probs


# TODO - think if this CategoricalPolicy is even required
class CategoricalPolicy(BasePolicy, nn.Module):
    def __init__(self, network):
        nn.Module.__init__(self)
        self.network = network

    def action_distribution(self, observations):
        """
        Args:
            observations: torch.Tensor of shape [batch size, dim(observation space)]
        Returns:
            distribution: torch.distributions.Categorical where the logits
                are computed by self.network
        """
        logits = self.network(observations)
        distribution = torch.distributions.Categorical(logits=logits)
        return distribution


class GaussianPolicy(BasePolicy, nn.Module):
    def __init__(self, network, action_dim):
        nn.Module.__init__(self)
        self.network = network
        self.log_std = torch.nn.Parameter(torch.zeros(action_dim))

    def std(self):
        """
        Returns:
            std: torch.Tensor of shape [dim(action space)]
        """
        std = torch.exp(self.log_std)
        return std

    def action_distribution(self, observations):
        """
        Args:
            observations: torch.Tensor of shape [batch size, dim(observation space)]
        Returns:
            distribution: an instance of a subclass of
                torch.distributions.Distribution representing a diagonal
                Gaussian distribution whose mean (loc) is computed by
                self.network and standard deviation (scale) is self.std()
        """
        mean = self.network(observations)
        std = self.std()
        covariance_matrix = torch.diag(torch.square(std))
        distribution = torch.distributions.MultivariateNormal(mean, covariance_matrix)
        return distribution
