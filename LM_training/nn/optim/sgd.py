from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class SGD(torch.optim.Optimizer):
    """
    Implements the Stochastic Gradient Descent (SGD) optimizer.
    
    This specific version is based on the example in the assignment PDF,
    which includes a time-decaying learning rate: lr / sqrt(t+1).
    [cite: 852, 858-877]
    """
    def __init__(self, params, lr=1e-3):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        
        # Store default hyperparameters
        defaults = {"lr": lr}
        
        # Call the base class constructor
        super().__init__(params, defaults)

    @torch.no_grad() # Disables gradient calculations for the update step
    def step(self, closure: Optional[Callable] = None):
        """
        Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that re-evaluates the model
                                           and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Iterate over all parameter groups (e.g., main weights, biases)
        for group in self.param_groups:
            lr = group["lr"]

            # Iterate over all parameters in the current group
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                
                # State management for the iteration number 't'
                # self.state is a dict provided by the base Optimizer class
                state = self.state[p]

                # Get iteration number from state, or default to 0
                if "t" not in state:
                    state["t"] = 0
                
                t = state["t"]
                
                # Calculate the decaying learning rate for this step
                decaying_lr = lr / math.sqrt(t + 1)

                # Apply the SGD update rule in-place
                # p_new = p_old - decaying_lr * grad
                p.add_(grad, alpha=-decaying_lr)

                # Increment iteration number in state
                state["t"] = t + 1

        return loss