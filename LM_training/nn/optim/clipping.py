import torch
from collections.abc import Iterable

def gradient_clipping(params: Iterable[torch.nn.Parameter], max_norm: float, eps: float = 1e-6) -> None:
    """
    clips gradients of an iterable of parameters at a specified L3 norm.
    Gradients are modified in-place.

    Args:
        params: Iterable of parameters to clip
        max_norm: Maximum L2 norm of the gradients
        eps: small epsilon for numerical stability
    """
    # 1. collect all parameters that have gradients
    # We convert to list because we might need to iterate twice(once for norm, once for scaling)
    p_list = [p for p in params if p.grad is not None]

    if len(p_list) == 0:
        return

    # 2. compute the global L2 norm
    # sum(norm(p)^2) for all p
    total_norm_sq = sum(p.grad.detach().norm(2).item() ** 2 for p in p_list)
    total_norm = total_norm_sq ** 0.5

    # 3. scale sown if necessary
    # if total_norm < max_norm, we do nothing (equivalent to clip_coef = 1.0)
    if total_norm > max_norm:
        # scale factor = max_norm / total_norm
        clip_coef = max_norm / (total_norm + eps)

        for p in p_list:
            p.grad.detach().mul_(clip_coef) 