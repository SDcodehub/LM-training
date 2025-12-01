import torch
import torch.nn as nn

from LM_training.nn.modules import MultiHeadAttention, SwiGLUFFN, RMSNorm, Embedding, Linear

class Transformer(nn.Module):
    """
    implements a single pre-norm transformer block as a stateful module
    this block consists of a multi-head self-attention layer and a feed-forward network
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        rope_theta: float = 10000.0,
        device=None,
        dtype=None,
    ):
        super().__init__()

        # first sub-layer components
        self.attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
            device=device,
            dtype=dtype
        )

        self.norm1 = RMSNorm(d_model, device=device, dtype=dtype)

        # second sub-layer components
        self.ffn = SwiGLUFFN(d_model, d_ff, device=device, dtype=dtype)
        self.norm2 = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        defines the forward pass for the pre-norm Transformer block
        """
        # first sub-layer: multi-head self-attention
        # residual connection starts from the original input 'x'
        # the equation is: x + attention(Norm(x))

        attention_output = self.attn(self.norm1(x), token_positions)
        x = x + attention_output

        # second sub-layer feed forward network
        # residual connection starts from the output of the first sub-layer
        # the equation is x + ffn(norm(x))
        ffn_output = self.ffn(self.norm2(x))
        x = x + ffn_output

        return x


class TransformerLM(nn.Module):
    """
    a complete transformer language model
    this module stacks transformer blocks and adds the embedding layers
    """
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float = 10000.0,
        device=None,
        dtype=None,
    ):
        super().__init__()

        # token embedding layer
        self.token_embedding = Embedding(
            num_embeddings=vocab_size, 
            embedding_dim=d_model, 
            device=device, 
            dtype=dtype)

        # stack of transformer blocks
        # we use nn.ModuleList to propertly register all blocks
        self.blocks = nn.ModuleList(
            [
                Transformer(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    rope_theta=rope_theta,
                    device=device,
                    dtype=dtype
                )
                for _ in range(num_layers)

            ]
        )

        # final normalization layer
        self.final_norm = RMSNorm(d_model, device=device, dtype=dtype)

        # output embedding
        self.lm_head = Linear(in_features=d_model, out_features=vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        defines the forward pass for the entire transformer LM
        """
        _, seq_len = token_ids.shape

        # get token embeddings
        x = self.token_embedding(token_ids)

        # we need token positions for the RoPE embedding, we can generate them on the fly
        # shape (seq_len, )
        token_positions = torch.arange(seq_len, device=token_ids.device, dtype=token_ids.dtype)

        # pass through all transformer blocks
        for block in self.blocks:
            x = block(x, token_positions)

        # apply final normalization
        x = self.final_norm(x)

        # apply LM head to get logits
        logits = self.lm_head(x)

        return logits

