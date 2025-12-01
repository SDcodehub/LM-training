import torch
import os
import typing

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: typing.Union[str, os.PathLike, typing.BinaryIO, typing.IO[bytes]]
) -> None:
    """
    Saves the model parameters, optimizer state, and current iteration to a file.

    Args:
        model: The PyTorch model to save.
        optimizer: The optimizer whose state needs saving.
        iteration: The current training step count.
        out: A file path or file-like object to write the checkpoint to.
    """
    # Create a dictionary to hold all necessary state
    checkpoint_state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    
    # Use torch.save to serialize the dictionary
    torch.save(checkpoint_state, out)

def load_checkpoint(
    src: typing.Union[str, os.PathLike, typing.BinaryIO, typing.IO[bytes]],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Loads a checkpoint and restores the model and optimizer states.

    Args:
        src: A file path or file-like object to read the checkpoint from.
        model: The model to load weights into.
        optimizer: The optimizer to load state into.

    Returns:
        The iteration number retrieved from the checkpoint.
    """
    # Load the dictionary from the file
    # Note: map_location can be added here if moving between GPU/CPU, 
    # but standard loading is usually sufficient for this assignment.
    checkpoint_state = torch.load(src)

    # Restore states
    model.load_state_dict(checkpoint_state["model"])
    optimizer.load_state_dict(checkpoint_state["optimizer"])

    # Return the iteration so the training loop knows where to resume
    return checkpoint_state["iteration"]