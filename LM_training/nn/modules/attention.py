import torch
import torch.nn as nn

from LM_training.nn.modules import Linear
from LM_training.nn.modules import RotaryPositionEmbedding
from LM_training.nn import functional as F

class MultiHeadAttention(nn.Module):
    """
    implements a stateful causal multi-head self-attention module
    """
    def __init__(self, 
    d_model: int, 
    num_heads: int, 
    max_seq_len: int, 
    rope_theta: float = 10000.0,
    device=None,
    dtype=None):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Linear layers for Q, K, V and output projections
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.out_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        # Rotary positional embedding
        self.rope = RotaryPositionEmbedding(
            theta=rope_theta, 
            d_k=self.head_dim, 
            max_seq_len=max_seq_len, 
            device=device, 
            dtype=dtype)

        # causal mask buffer
        # the mask is created once and reused, not learnable parameter.
        causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len, device=device, dtype=dtype))
        self.register_buffer('causal_mask', causal_mask, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # project inputs to QKV
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # split heads
        # (B, T, D) -> (B, T, H, h_dim) -> (B, H, T, h_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)

        # optionally apply RoPE to Q and K
        if token_positions is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        # apply scaled dot-product attention with causal mask
        # the causal mask is (T, T), we need to slice it for the current seq_len
        mask = self.causal_mask[:seq_len, :seq_len]
        attention_output = F.scaled_dot_product_attention(q, k, v, mask=mask)

        # combine heads
        # (B, H, T, h_dim) -> (B, T, H, h_dim) -> (B, T, D)
        attention_output = attention_output.transpose(1,2).contiguous().view(batch_size, seq_len, self.d_model)

        # final output projection
        return self.out_proj(attention_output)