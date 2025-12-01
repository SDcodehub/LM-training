import math

def get_lr_cosine_schedule(t: int, alpha_max: float, alpha_min: float, Tw: int, Tc: int) -> float:
    """
    calculates the learning rate at step t using cosine schedule with warmup

    args:
        t: current step number
        alpha_max: maximum learning rate (peak value after warmup)
        alpha_min: minimum learning rate (final value)
        Tw: Number of warmup steps
        Tc: total number of cosine annealing steps (typically total training steps)

    returns:
        the learning rate alpha_t for the current step.
    """
    # 1. warmup phase
    if t < Tw:
        return (t / Tw) * alpha_max
    
    # 2. cosine annealing phase
    elif t <= Tc:
        # calculate progress fraction within the cosine phase
        progress = (t - Tw) / (Tc - Tw)

        # calculate the cosine decay factor (ranges from 1.0 down to 0.0)
        # cos(theta) = 1, cos(pi) = -1
        # 0.5 * (1 + 1) = 1, 0.5 * (1 - 1) = 0
        decay_factor = 0.5 * (1 + math.cos(progress * math.pi))

        return alpha_min + decay_factor * (alpha_max - alpha_min)

    # 3. post-annealing phase
    else:
        return alpha_min