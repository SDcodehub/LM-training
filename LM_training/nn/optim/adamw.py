import torch
import math
from collections.abc import Callable
from typing import Optional

class AdamW(torch.optim.Optimizer):
    """
    Implements the AdamW optimizer
    """
    def __init__(self, params, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999), eps:float = 1e-8, weight_decay: float = 1e-2):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1 parameter: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2 parameter: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, beta1=betas[0], beta2=betas[1], eps=eps, weight_decay=weight_decay)

        super().__init__(params, defaults)


    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """
        Performs a single optimization step
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            # get hyperparameters for this parameter group
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # lazy state initialization
                if len(state) == 0:
                    # t start at 1 in the algorith
                    state["t"] = 0
                    # m <- 
                    state["m"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # v < - 0
                    state["v"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                m, v = state["m"], state["v"]

                # Increment step t (starts at 1)
                t = state["t"] + 1
                state["t"] = t

                # Decoupled weight decay (AdamW)
                if weight_decay != 0:
                    p.add_(p, alpha=-lr * weight_decay)

                # Update biased first and second moment estimates
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias corrections
                bias_corr1 = 1.0 - beta1**t
                bias_corr2 = 1.0 - beta2**t
                step_size = lr * math.sqrt(bias_corr2) / bias_corr1

                # Denominator: sqrt(v) + eps (PyTorch-style)
                denom = v.sqrt().add_(eps)

                # Parameter update
                p.addcdiv_(m, denom, value=-step_size)

        return loss
