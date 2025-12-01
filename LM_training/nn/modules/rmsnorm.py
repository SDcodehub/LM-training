"""
RMSNorm module.
"""
import torch
import torch.nn as nn
import torch.nn.init as nn_init

# import our functional kernel
from LM_training.nn import functional as F

class RMSNorm(nn.Module):
    """
    a stateful wrapper for the stateless rmsnorm kernel
    this module creates, initializes, and stores the learnable gain parameter
    """
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps

        # create the learnable gain parameter (gamma)
        self.weight = nn.Parameter(torch.empty(d_model, device=device, dtype=dtype))

        # initialize the gain parameters
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize the gain parameter to ones
        """
        nn_init.ones_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        calls the stateless rmsnorm functional implementation
        """
        return F.rms_norm(x, self.weight, self.eps)

    def extra_repr(self) -> str:
        return f'{self.d_model}, eps={self.eps}'