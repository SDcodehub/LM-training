import torch
import torch.nn as nn
import torch.nn.init as nn_init

# import our functional kernel
from LM_training.nn import functional as F

class Embedding(nn.Module):
    """
    a stateful wrapper for the stateless embedding kernel
    this module creates, initializes and stoed the learnable embedding matrix
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings # vocab size
        self.embedding_dim = embedding_dim # d model/hidden size

        # create the embedding matrix as a learnable parameter
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))

        # initialize the parameters
        self.reset_parameters()

    def reset_parameters(self):
        """
        initialize the embedding matrix using a truncated normal distribution
        """
        # for embeddings, the spec is N(0,1) truncated to p-3,3]
        std = 1
        nn_init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        forward pass, calls the stateless functional implementation
        """
        return F.embedding(token_ids, self.weight)

    # Expose a compatibility alias so callers can use either `.W` or `.weight`
    @property
    def weight(self):
        return self.W

    def extra_repr(self) -> str:
        return f"{self.num_embeddings}, {self.embedding_dim}"