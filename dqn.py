import torch.nn as nn
import torch
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, layer_sizes: list[int]):
        """
        DQN initialisation

        Args:
            layer_sizes: list with size of each layer as elements
        """
        super(DQN, self).__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(layer_sizes[i], layer_sizes[i + 1]) for i in
             range(len(layer_sizes) - 1)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the DQN

        Args:
            x: input to the DQN

        Returns:
            outputted value by the DQN
        """
        for layer in self.layers:
            x = F.relu(layer(x))
        return x
