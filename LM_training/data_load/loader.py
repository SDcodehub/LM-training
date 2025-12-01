import numpy as np
import torch

def get_batch(data: np.ndarray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Samples a batch of input and target sequences from the data

    Args:
        data: A numpy array of integer token IDs
        batch_size: The number of sequences to sample
        context_length: The lenght of each sequence
        device: The PyTorch device string (e.g. 'cpu', 'cuda:0', 'mps')

    Returns:
        x: Input sequences of shape (batch_size, context_length)
        y: Target sequences of shape (batch_size, context_length)
    """
    # we need to pick random starting indices.
    # the last possible starting index id len(data) - context_length -1
    # The -1 is necessary because we need tohe NEXT token for the target
    # torch.randint high is exclusive, so we use len(data) - context_length
    ix = torch.randint(0, len(data) - context_length, (batch_size,))

    # Slice the numpy array for each random index.
    # We convert to int64 (long) which is standard for token IDs in PyTorch.
    x_batch = torch.stack([torch.from_numpy((data[i : i + context_length]).astype(np.int64)) for i in ix])
    y_batch = torch.stack([torch.from_numpy((data[i + 1 : i + 1 + context_length]).astype(np.int64)) for i in ix])
    
    # Move to the requested device
    # Note: We create tensors on CPU first and then move them. This is often
    # faster/safer when dealing with numpy arrays, especially memmapped ones.
    if device is not None:
        device = torch.device(device)
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
        
    return x_batch, y_batch