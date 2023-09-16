import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.network_utils import build_mlp, np2torch, device
from torchsummary import summary

from utils.utils import CaptureStdout

logger = logging.getLogger("root")


class BaselineNetwork(nn.Module):
    def __init__(self, env, config):
        super().__init__()
        self.config = config
        self.env = env
        self.baseline = self.config.baseline
        self.lr = self.config.learning_rate
        self.observation_dim = self.env.observation_space

        # TODO - baseline network should have different parameters than policy network
        # Create the neural network baseline
        self.network = build_mlp(
            input_size=self.observation_dim,
            output_size=1,
            n_layers=self.config.n_layers,
            size=self.config.first_layer_size,
            config=self.config,
        ).to(device)

        with CaptureStdout() as capture:
            summary(self.network, input_size=(self.observation_dim,))

        # Define the optimizer for the baseline
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        logger.info(
            f"Baseline initialized with:"
            f"\n{capture.get_output()}"
            f"{self.optimizer=}"
            f"\n{self.observation_dim=}"
            f"\n{self.lr=}"
        )

    def forward(self, observations):
        """
        Args:
            observations: torch.Tensor of shape [batch size, dim(observation space)]
        Returns:
            output: torch.Tensor of shape [batch size]
        """
        # Pass the observations through the network
        output = self.network(observations)
        output = output.squeeze()

        assert output.ndim == 1
        return output

    def calculate_advantage(self, returns, observations):
        """
        Args:
            returns: np.array of shape [batch size]
                all discounted future returns for each step
            observations: np.array of shape [batch size, dim(observation space)]
        Returns:
            advantages: np.array of shape [batch size]
        """
        observations = np2torch(observations)

        # Use the forward pass of the baseline network to get the predicted value of the state
        predicted_values = self(observations)

        # Compute the advantage estimates
        advantages = returns - predicted_values.detach().cpu().numpy()

        return advantages

    def update_baseline(self, returns, observations):
        """
        Args:
            returns: np.array of shape [batch size], containing all discounted
                future returns for each step
            observations: np.array of shape [batch size, dim(observation space)]
        """
        returns = np2torch(returns)
        observations = np2torch(observations)

        self.optimizer.zero_grad()

        predicted_values = self(observations)

        loss = F.mse_loss(predicted_values, returns)

        loss.backward()

        self.optimizer.step()
