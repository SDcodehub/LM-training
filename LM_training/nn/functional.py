"""
stateless kernels: linear, softmax, silu, rmsnorm, sdpa, rope, embedding_lookup
"""

import torch
import math
from einops import einsum

def linear(input_tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Applies linear transformation to incoming data.
    Stateless function.

    Args:
        input(torch.Tensor): input tensor of shape (...., in_features)
        weight(torch.Tensor): weight matrix of shape (out_features, in_features)

    Returns:
    torch.Tensor: output tensor of shape (..., out_features)
    """
    # explicit einsum notation; supports arbitrary leading/batch dims
    return einsum(input_tensor, weight, "... in_features, out_features in_features -> ... out_features")
    

def embedding(input_ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Performs a lookup in an embedding matrix
    this is stateless function

    Args:
        input_ids(torch.Tensor): a tensor of integer token IDs of shape (...)
        weight (torch.Tensor): the embedding matrix of shape (vocab_size, embedding_dim)

    Returns:
        torch.Tensor: a looked up embedding vectors of shap (..., embedding_dim)
    """
    # Pytorchs indexing is higly optimised for this operation
    return weight[input_ids]


def rms_norm(input_tensor: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    applies rms layer normalisation
    stateless function
    
    Args:
        input (torch.Tensor): input tensor of shape (..., d_model)
        weight (torch.Tensor): learnable gain parameter (gamma) of shape (d_model, )
        eps (float): a small value added for numerical stability

    Returns:
        torch.Tensor: normalised tensor of the same shape as input
    """
    # store original dtype to cast back at the end
    input_dtype = input_tensor.dtype

    # upcast to float32 for stable computation of squares
    x = input_tensor.to(torch.float32)

    # calculate the mean of the squares of the input along the last dimension
    variance = x.pow(2).mean(dim=-1, keepdim=True)

    # calculate the reciprocal fo the square root
    rsqrt = torch.rsqrt(variance + eps)

    # normalize the input and apply the learnable gain (weight)
    normalized_x = x * rsqrt

    # apply the gain and cast back to the original dtype
    return (weight * normalized_x ).to(input_dtype)


def silu(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    applies the sigmoid weighted linear unit activation function
    also known as swish, x * sigmoid(x)

    Args:
        input (torch.Tensor): input tensor

    Returns:
        torch.Tensor: output tensor, of same as input
    """
    # addignment allows using torch.sigmoid for numerical stability
    return input_tensor * torch.sigmoid(input_tensor)

def swiglu_ffn(
    input_tensor: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
) -> torch.Tensor:
    """
    implementation of swiglu ffn 
    stateless function

    Args:
        input (torch.Tensor): input tensor of shape (...,d_model)
        w1 (torch.Tensor): weight matric for the first projection, shape (d_ff, d_model)
        w2 (torch.Tensor): weight matric for the output projection, shape (d_model, d_ff)
        w3 (torch.Tensor): weight matric for the gate projection, shape (d_ff, d_model)

    Returns:
        torch.Tensor: output tensor of shape (..., d_model)
    """
    # project up using w1, w3
    # einsum: "... d_model, d_ff d_model -> ... d_ff"
    x1 = linear(input_tensor, w1)
    # einsum: "... d_model, d_ff d_model -> ... d_ff"
    x3 = linear(input_tensor, w3)

    # apply silu activation and the gating mechanism (elementwise multiplication)
    gated_x = silu(x1) * x3

    # project back down using w2
    # einsum: "... d_ff, d_model d_ff -> ... d_model"
    return linear(gated_x, w2)


def rope(
    input_tensor: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    applies rotary position embedding RoPE to the input tensor
    this is a stateless function

    Args:
        input (torch.Tensor): input tensor of shape (..., seq_len, d_k)
        cos (torch.Tensor): Pre-computed cosine values of shape (..., seq_len, d_k )
        sin (torch.Tensor): Pre-computed sine values of shape (..., seq_len, d_k)

    Returns:
        torch.Tensor: Tensor with RoPE applied, of the same shape as input
    """
    # reshape input to view the last dimension as pairs of features
    # (..., seq_len, d_k) -> (..., seq_len, d_k/2, 2)
    x_pairs = input_tensor.unflatten(-1, (-1,2))

    # get the two components of each pair
    x1 = x_pairs[..., 0]
    x2 = x_pairs[..., 1]

    # Reshape sin and cos to match the paired shape
    cos = cos.unflatten(-1, (-1, 2))[..., 0]
    sin = sin.unflatten(-1, (-1, 2))[..., 0]

    # apply the 2d rotation matrix formula:
    # y1 = x1*cos(theta) - x2*sin(theta)
    # y2 = x1*sin(theta) + x2*cos(theta)
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos

    # stack the roatated pairs back together
    # (..., seq_len, d_k/2), (..., seq_len, d_k/2) -> (..., seq_len, d_k/2, 2)
    y_pairs = torch.stack((y1, y2), dim=-1)

    # flatten the last two dimensions to restore the original shape
    # (..., seq_len, d_k/2, 2) -> (..., seq_len, d_k)
    return y_pairs.flatten(-2)


def softmax(input_tensor: torch.Tensor, dim: int) -> torch.Tensor:
    """
    applies numerically stable softmax function
    this is a stateless function

    Args: 
        input (torch.Tensor): input tensor of logits of shape
        dim (int): dimension along which softmax will be applied

    Returns:
        torch.Tensor: Tensor of probabilities, same shape as input.
    """
    # subtract the max for numerical stability
    # we use keepdim=true to ensure the result is broadccastable
    max_vals, _ = torch.max(input_tensor, dim=dim, keepdim=True)
    shifted_logits = input_tensor - max_vals

    # exponentiate
    exps = torch.exp(shifted_logits)

    # sum the exponents and divide
    sum_exps = torch.sum(exps, dim=dim, keepdim=True)
    
    return exps / sum_exps


def scaled_dot_product_attention(
    query: torch.Tensor,
    key:torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    computes scaled dot product attenstion as a stateless function.
    formula: softmax( (query @ key.T) / sqrt(d_key) ) @ value

    Args:
        query (torch.Tensor): query tensor fo shape (..., seq_len, d_k)
        key (torch.Tensor): key tensor of shape (..., seq_len, d_k)
        value (torch.Tensor): value tensor of shape (..., seq_len, d_v)
        mask (torch.Tensor, optional): boolean mask of shape (..., seq_len_q, seq_len_k)
                                        if a value is flase, the corresponding attention 
                                        score is set to -inf. defaults to none
        Returns:
            torch.Tensor: the output of the attention mechanism, shape (..., seq_len, d_v)
    """
    # d_k is the dimension fo the key/query vectors
    d_k = key.shape[-1]

    # compute raw scores with matrix multiplication (Q @ K.T)
    scores = torch.einsum("...qd, ...kd-> ...qk", query, key)

    # scale the scores
    scaled_scores = scores / math.sqrt(d_k)

    # apply mask if provided
    if mask is not None:
        # we need to ensure the mask can be broadcasted to the scores shape
        # this is usually handled by how the mask is constructed, but a view can make is robust
        # for a mask of shape (T,T ) we might need to add batch/head dimensions
        while mask.dim() < scaled_scores.dim():
            mask = mask.unsqueeze(0)

        # set scores for a very large negative number where the mask is false
        scaled_scores = scaled_scores.masked_fill(mask == False, -torch.finfo(scaled_scores.dtype).max)

    # compute the attention weights using softmax
    # the softmax is applied on the last dimension (the keys)
    attention_weights = softmax(scaled_scores, dim=-1)

    # compute the weighted sum of values
    return torch.einsum("...qk, ...kd-> ...qd", attention_weights, value)
    
    
def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    computes the cross-entropy loss in a numerically stable way.
    This is a stateless function.

    Args:
        logits (torch.Tensor): the raw logits from the model, of shape (..., vocab_size)
        targets (torch.Tensor): the target token IDs, of shape (...)
    Returns:
        torch.Tensor: A single scalar tensor representing the average cross-entropy loss.
    """
    # 1. find the max logit for stability
    max_logits, _ = torch.max(logits, dim=-1, keepdim=True)

    # 2. Substract max logit before exponentiating
    stable_logits = logits - max_logits

    # 3. compute the log sum exp term
    # log(sum(exp(o_i -c))) + c
    log_sum_exp = torch.log(torch.sum(torch.exp(stable_logits), dim=-1)) + max_logits.squeeze(-1)

    # 4. get the logit score for the correct target token
    # we need to gather the logits corresponding to the target indices
    # targets.unsqueeze(-1) adds a dim for gather: (...) -> (..., 1)
    # .squeeze(-1) removes it: (..., 1) -> (...)
    # example
    #     import torch

    # # logits: (batch=2, seq=2, vocab=3)
    # logits = torch.tensor([
    #     [[1.0, 2.0, 3.0],
    #     [4.0, 5.0, 6.0]],
    #     [[7.0, 8.0, 9.0],
    #     [10.0,11.0,12.0]],
    # ])

    # # targets: (batch=2, seq=2)
    # targets = torch.tensor([
    #     [2, 0],
    #     [1, 2],
    # ])

    # print("logits.shape:", logits.shape)    # (2, 2, 3)
    # print("targets.shape:", targets.shape)  # (2, 2)

    # idx = targets.unsqueeze(-1)
    # print("\n1) targets.unsqueeze(-1):")
    # print("   shape:", idx.shape)           # (2, 2, 1)
    # print(idx)

    # gathered = logits.gather(-1, idx)
    # print("\n2) logits.gather(-1, idx):")
    # print("   shape:", gathered.shape)      # (2, 2, 1)
    # print(gathered)

    # target_logits = gathered.squeeze(-1)
    # print("\n3) ... .squeeze(-1):")
    # print("   shape:", target_logits.shape) # (2, 2)
    # print(target_logits)

    # logits.shape: torch.Size([2, 2, 3])
    # targets.shape: torch.Size([2, 2])

    # 1) targets.unsqueeze(-1):
    # shape: torch.Size([2, 2, 1])
    # tensor([[[2],
    #         [0]],

    #         [[1],
    #         [2]]])

    # 2) logits.gather(-1, idx):
    # shape: torch.Size([2, 2, 1])
    # tensor([[[ 3.],
    #         [ 4.]],

    #         [[ 8.],
    #         [12.]]])

    # 3) ... .squeeze(-1):
    # shape: torch.Size([2, 2])
    # tensor([[ 3.,  4.],
    #         [ 8., 12.]])
    target_logits = logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    # 5. Compute the loss for each token
    # l_i = log_sum_exp(o_i) - o_i(x_{i+1})
    token_loss = log_sum_exp - target_logits

    # 6. return the average loss over the entire batch 
    return token_loss.mean()

