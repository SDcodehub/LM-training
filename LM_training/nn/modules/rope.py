import torch
import torch.nn as nn

from LM_training.nn import functional as F

class RotaryPositionEmbedding(nn.Module):
    """
    a stateful module for rotary position embedding RoPE
    this module pre-computes and buffers the sine and cosine values required for rotation
    """
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None, dtype=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # pre-compute the inverse frequencies (thetas_k in the formula)
        # shape (d_k/2)
        thetas_k = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device, dtype=dtype) / d_k))

        # pre-comute the position indices
        # shape (max_seq_len)
        positions = torch.arange(max_seq_len, device=device, dtype=dtype)

        #calculate the arguments for sin and cos
        # shape (max_seq_len, d_k/2)
        angles = torch.outer(positions, thetas_k)

        # interleave each angle value twice to match (pair, 2) pairing
        # shape (max_seq_len, d_k) with pattern [θ0, θ0, θ1, θ1, ...]
        angles_repeated = torch.repeat_interleave(angles, repeats=2, dim=-1)
        
        # register sin and cos values as non-learnable buffers
        self.register_buffer('cos_cached', angles_repeated.cos(), persistent=False)
        self.register_buffer('sin_cached', angles_repeated.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        applies RoPE to the input tensor using the pre-computed values.

        Args:
            x (torch.Tensor): input tensor of shape (..., seq_len, d_k)
            token_positions (torch.Tensor): a tensor specifying the absolute positions
                                            of tokens in the sequence, shape (..., seq_len)
        Returns:
            torch.Tensor: tensor with RoPE applied , same shape as x
        """
        # slice the pre-computed cos and sin tensors using the provided token positions
        # this allows handling sequences shorter than max_seq_len and enables featues 
        # like kv caching during inference
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        # the input x can have arbitrary batch dimensions so we need to make sure 
        # cos and sin are broadcastable to the shape of x
        # we add unsqueezed dimensions to cos/sin to match the batch dimensions of x
        # example x shape (B, H, T, D), pos shape (B, T) -> cos/sin shape (B, H, T, D)
        while cos.dim() < x.dim():
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
        # apply the RoPE function
        return F.rope(x, cos, sin)
