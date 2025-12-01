import torch
import torch.nn as nn

# we can now import and reuse our custom linear layer
from LM_training.nn.modules import Linear
from LM_training.nn import functional as F


class SwiGLUFFN(nn.Module):
    """
    a stateful module for the SwiGLU feed forward network
    this module creates and manages the three linear layer required
    """
    def __init__(self, d_module: int, d_ff: int, device=None, dtype=None):
        super().__init__()

        # the three linear layers required for the SwiGLU FFN
        self.w1 = Linear(d_module, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_module, device=device, dtype=dtype)
        self.w3 = Linear(d_module, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        this forward pass we call the swiglu function
        passing it the input tensor x and our stored weight parameters
        """
        return F.swiglu_ffn(x, self.w1.W, self.w2.W, self.w3.W) 