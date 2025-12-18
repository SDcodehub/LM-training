"""
Count tokens in a tokenized .npy dataset file and estimate training epochs.

This utility helps understand:
- Total number of tokens in the dataset
- Number of training steps given batch size and context length
- Approximate number of epochs (passes over the data) for a given max_iters

Usage:
    uv run python -m LM_training.tokenizer.cli.count_tokens \
        --data ./output/file/npy/TinyStoriesV2-GPT4-train-10k.npy \
        --batch_size 32 \
        --context_length 256 \
        --max_iters 5000
"""

import argparse
import os
import sys
from typing import Optional

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]


def _ensure_package_import() -> None:
    """
    Ensure the assignment root is on sys.path so `LM_training.*` imports work.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    package_dir = os.path.dirname(current_dir)
    assignment_root = os.path.dirname(package_dir)
    if assignment_root not in sys.path:
        sys.path.append(assignment_root)


_ensure_package_import()
try:
    from LM_training.utils.logging_config import get_logger  # type: ignore  # noqa: E402
except Exception:  # pragma: no cover
    import logging  # noqa: E402

    def get_logger() -> "logging.Logger":  # type: ignore[name-defined]
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)s:%(name)s:%(message)s")
            )
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
            logger.propagate = False
        return logger


LOGGER = get_logger()


def count_tokens_in_npy(filepath: str) -> int:
    """
    Count the number of tokens in a tokenized .npy file.
    
    Args:
        filepath: Path to the .npy file
    
    Returns:
        Number of tokens in the file
    """
    if np is None:
        raise RuntimeError("NumPy is required but not installed. Please install numpy.")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Use memory mapping to avoid loading the entire file
    data = np.load(filepath, mmap_mode='r')
    num_tokens = len(data)
    return num_tokens


def calculate_training_metrics(
    num_tokens: int,
    batch_size: int,
    context_length: int,
    max_iters: Optional[int] = None
) -> dict:
    """
    Calculate training metrics based on dataset size and hyperparameters.
    
    Args:
        num_tokens: Total number of tokens in dataset
        batch_size: Batch size for training
        context_length: Context length (sequence length)
        max_iters: Maximum training iterations (optional)
    
    Returns:
        Dictionary containing training metrics
    """
    tokens_per_batch = batch_size * context_length
    max_possible_steps = num_tokens // tokens_per_batch
    
    metrics = {
        "total_tokens": num_tokens,
        "tokens_per_batch": tokens_per_batch,
        "max_possible_steps": max_possible_steps,
    }
    
    if max_iters is not None:
        total_tokens_seen = tokens_per_batch * max_iters
        epochs = total_tokens_seen / num_tokens
        metrics["max_iters"] = max_iters
        metrics["total_tokens_seen"] = total_tokens_seen
        metrics["epochs"] = epochs
    
    return metrics


def format_number(num: float) -> str:
    """Format large numbers with commas for readability."""
    if isinstance(num, float) and num != int(num):
        return f"{num:,.2f}"
    return f"{int(num):,}"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Count tokens in a tokenized .npy file and calculate training metrics. "
            "Helps understand dataset size and estimate training epochs."
        )
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to tokenized .npy file"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for training (optional, for calculating metrics)"
    )
    parser.add_argument(
        "--context_length",
        type=int,
        default=None,
        help="Context length / sequence length (optional, for calculating metrics)"
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=None,
        help="Maximum training iterations (optional, for calculating epochs)"
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    
    # Validate file exists
    if not os.path.isfile(args.data):
        LOGGER.error(f"Data file not found: {args.data}")
        sys.exit(1)
    
    # Count tokens
    LOGGER.info(f"Analyzing file: {args.data}")
    num_tokens = count_tokens_in_npy(args.data)
    
    # Print basic info
    print("\n" + "=" * 70)
    print(f"Dataset: {os.path.basename(args.data)}")
    print("=" * 70)
    print(f"Total tokens: {format_number(num_tokens)}")
    
    # Calculate additional metrics if batch_size and context_length provided
    if args.batch_size is not None and args.context_length is not None:
        metrics = calculate_training_metrics(
            num_tokens=num_tokens,
            batch_size=args.batch_size,
            context_length=args.context_length,
            max_iters=args.max_iters
        )
        
        print(f"\nTraining Configuration:")
        print(f"  Batch size: {format_number(args.batch_size)}")
        print(f"  Context length: {format_number(args.context_length)}")
        print(f"  Tokens per batch: {format_number(metrics['tokens_per_batch'])}")
        print(f"  Max possible steps (1 epoch): {format_number(metrics['max_possible_steps'])}")
        
        if args.max_iters is not None:
            print(f"\nTraining Plan (for {format_number(args.max_iters)} iterations):")
            print(f"  Total tokens seen: {format_number(metrics['total_tokens_seen'])}")
            print(f"  Approximate epochs: {metrics['epochs']:.4f}")
            
            if metrics['epochs'] < 1.0:
                print(f" Warning: Less than 1 epoch - will not see all data")
            elif metrics['epochs'] > 10.0:
                print(f" Note: More than 10 epochs - data will be repeated significantly")
        else:
            print(f"\n💡 Tip: Add --max_iters to calculate epochs")
    else:
        print(f"\n💡 Tip: Add --batch_size and --context_length to calculate training metrics")
    
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()



