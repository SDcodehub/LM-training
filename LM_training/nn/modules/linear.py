"""
Linear module.
"""
import torch
import torch.nn as nn
import torch.nn.init as nn_init

from LM_training.nn import functional as F

class Linear(nn.Module):
    """
    Stateful linear wrapper 
    creates, initializes and stores the learnable params
    """
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # create the weight matrix as learnable params
        self.W = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))

        # initialise the parameters
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialise the weight matrix using truncated normal distribution
        """
        std = (2 / (self.in_features + self.out_features)) ** 0.5
        nn_init.trunc_normal_(self.W, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        In forward pass, we call the stateless function linear from functional.py
        passing the input tensor x and the weight matrix w
        """
        return F.linear(x, self.W)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias=False"