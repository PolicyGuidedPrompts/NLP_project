import torch
import torch.nn as nn
import torch.distributions as ptd

from network_utils import np2torch, device


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

    def act(self, observations, return_log_prob = False):
        """
        Args:
            observations: np.array of shape [batch size, dim(observation space)]
        Returns:
            sampled_actions: np.array of shape [batch size, *shape of action]
            log_probs: np.array of shape [batch size] (optionally, if return_log_prob)

        TODO:
        Call self.action_distribution to get the distribution over actions,
        then sample from that distribution. Compute the log probability of
        the sampled actions using self.action_distribution. You will have to
        convert the actions and log probabilities to a numpy array, via numpy(). 

        You may find the following documentation helpful:
        https://pytorch.org/docs/stable/distributions.html
        """
        # Convert observations to torch.Tensor
        # TODO - probably need to remove this
        # observations = np2torch(observations)

        # TODO - can't generate same action more than once
        # TODO - episode has max_length, last action must be special action

        # Get action distribution based on observations
        action_distribution = self.action_distribution(observations)

        # Sample actions from the distribution
        sampled_actions = action_distribution.sample()

        if return_log_prob:
            # Calculate log probabilities of the sampled actions
            log_probs = action_distribution.log_prob(sampled_actions).detach().numpy()
            sampled_actions = sampled_actions.detach().numpy()

            return sampled_actions, log_probs

        # Convert sampled actions to numpy array
        sampled_actions = sampled_actions.detach().numpy()

        return sampled_actions


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

        See https://pytorch.org/docs/stable/distributions.html#categorical
        """
        logits = self.network(observations)
        distribution = torch.distributions.Categorical(logits=logits)
        return distribution


class GaussianPolicy(BasePolicy, nn.Module):
    def __init__(self, network, action_dim):
        """
        After the basic initialization, you should create a nn.Parameter of
        shape [dim(action space)] and assign it to self.log_std.
        A reasonable initial value for log_std is 0 (corresponding to an
        initial std of 1), but you are welcome to try different values.
        """
        nn.Module.__init__(self)
        self.network = network
        self.log_std = torch.nn.Parameter(torch.zeros(action_dim))

    def std(self):
        """
        Returns:
            std: torch.Tensor of shape [dim(action space)]

        The return value contains the standard deviations for each dimension
        of the policy's actions. It can be computed from self.log_std
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

        Note: PyTorch doesn't have a diagonal Gaussian built in, but you can
            fashion one out of
            (a) torch.distributions.MultivariateNormal
            or
            (b) A combination of torch.distributions.Normal
                             and torch.distributions.Independent
        """
        mean = self.network(observations)
        std = self.std()
        covariance_matrix = torch.diag(torch.square(std))
        distribution = torch.distributions.MultivariateNormal(mean, covariance_matrix)
        return distribution
