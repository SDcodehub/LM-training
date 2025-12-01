"""
Tokenize a text dataset with the BPE tokenizer and save token ids to a .npy file.

Usage (10k TinyStories tokenizer on the VALID set):

As a module (recommended):
    LOG_LEVEL=INFO python -m LM_training.scripts.tokenize_dataset \
      --input ./data/TinyStoriesV2-GPT4-valid.txt \
      --vocab ./bpe_tokenizer/tinystories-10k_vocab.json \
      --merges ./bpe_tokenizer/tinystories-10k_merges.txt \
      --output ./data/TinyStoriesV2-GPT4-valid_ids_10k.npy

With uv:
    LOG_LEVEL=INFO uv run python -m LM_training.scripts.tokenize_dataset \
      --input ./data/TinyStoriesV2-GPT4-valid.txt \
      --vocab ./bpe_tokenizer/tinystories-10k_vocab.json \
      --merges ./bpe_tokenizer/tinystories-10k_merges.txt \
      --output ./data/TinyStoriesV2-GPT4-valid_ids_10k.npy

Direct script invocation:
    LOG_LEVEL=INFO python ./LM_training/scripts/tokenize_dataset.py \
      --input ./data/TinyStoriesV2-GPT4-valid.txt \
      --vocab ./bpe_tokenizer/tinystories-10k_vocab.json \
      --merges ./bpe_tokenizer/tinystories-10k_merges.txt \
      --output ./data/TinyStoriesV2-GPT4-valid_ids_10k.npy
With uv:
    LOG_LEVEL=INFO uv run python ./LM_training/scripts/tokenize_dataset.py \
      --input ./data/TinyStoriesV2-GPT4-valid.txt \
      --vocab ./bpe_tokenizer/tinystories-10k_vocab.json \
      --merges ./bpe_tokenizer/tinystories-10k_merges.txt \
      --output ./data/TinyStoriesV2-GPT4-valid_ids_10k.npy
"""

import argparse
import os
import sys
from typing import Any, Iterable, List

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - fallback for lint environments without numpy
    np = None  # type: ignore[assignment]


def _ensure_package_import() -> None:
    """
    Ensure the assignment root is on sys.path so `LM_training.*` imports
    work when running this file directly.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    package_dir = os.path.dirname(current_dir)
    assignment_root = os.path.dirname(package_dir)
    if assignment_root not in sys.path:
        sys.path.append(assignment_root)


_ensure_package_import()
try:  # defer optional logging import with safe fallback
    from LM_training.utils.logging_config import get_logger  # type: ignore  # noqa: E402
except Exception:  # pragma: no cover - fallback if package import fails in lint env
    import logging  # noqa: E402

    def get_logger() -> "logging.Logger":  # type: ignore[name-defined]
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s:%(name)s:%(message)s"))
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
            logger.propagate = False
        return logger


LOGGER = get_logger()


def build_arg_parser() -> argparse.ArgumentParser:
    assignment_root = "."
    default_input = os.path.join(assignment_root, "data", "TinyStoriesV2-GPT4-train.txt")
    default_vocab = os.path.join(assignment_root, "bpe_tokenizer", "tinystories_vocab.json")
    default_merges = os.path.join(assignment_root, "bpe_tokenizer", "tinystories_merges.txt")

    parser = argparse.ArgumentParser(
        description=(
            "Tokenize a text dataset using the BPE tokenizer and save token ids to .npy"
        )
    )
    parser.add_argument("--input", type=str, default=default_input, help="Path to input text file")
    parser.add_argument("--vocab", type=str, default=default_vocab, help="Path to vocab JSON")
    parser.add_argument("--merges", type=str, default=default_merges, help="Path to merges TXT")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output .npy path (default: derive from input as <name>_ids.npy)",
    )
    return parser


def load_tokenizer(vocab_filepath: str, merges_filepath: str) -> Any:
    LOGGER.info("Loading tokenizer")
    # Import locally to avoid import errors in static lint environments
    from LM_training.tokenizer import Tokenizer  # type: ignore

    return Tokenizer.from_files(
        vocab_filepath=vocab_filepath,
        merges_filepath=merges_filepath,
        special_tokens=["<|endoftext|>"],
    )


def tokenize_lines(lines: Iterable[str], tokenizer: Any) -> List[int]:
    """
    Encode an iterable of lines into a flat list of token ids.
    """
    token_ids_iter = tokenizer.encode_iterable(lines)
    return list(token_ids_iter)


def tokens_to_array_uint16(token_ids: List[int]) -> Any:
    """
    Convert list of token ids to a numpy array with dtype uint16.
    Note: callers must ensure vocabulary size fits uint16.
    """
    if np is None:
        raise RuntimeError("NumPy is required but not installed. Please install numpy.")
    return np.array(token_ids, dtype=np.uint16)


def derive_output_path(input_path: str, explicit_output: str | None) -> str:
    if explicit_output:
        return explicit_output
    base, _ = os.path.splitext(os.path.basename(input_path))
    return os.path.join(os.path.dirname(input_path), f"{base}_ids.npy")


def save_numpy_array(array: np.ndarray, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.save(output_path, array)


def main() -> None:
    args = build_arg_parser().parse_args()

    if not os.path.isfile(args.input):
        LOGGER.error("Input file not found: %s", args.input)
        sys.exit(1)
    if not os.path.isfile(args.vocab):
        LOGGER.error("Vocab file not found: %s", args.vocab)
        sys.exit(1)
    if not os.path.isfile(args.merges):
        LOGGER.error("Merges file not found: %s", args.merges)
        sys.exit(1)

    output_path = derive_output_path(args.input, args.output)

    tokenizer = load_tokenizer(args.vocab, args.merges)

    LOGGER.info("Opening input and starting tokenization: %s", args.input)
    with open(args.input, "r", encoding="utf-8") as file_handle:
        all_token_ids = tokenize_lines(file_handle, tokenizer)

    LOGGER.info("Finished tokenizing. Total tokens: %d", len(all_token_ids))
    LOGGER.info("Converting tokens to uint16 array")
    token_array = tokens_to_array_uint16(all_token_ids)

    LOGGER.info("Saving token ids to: %s", output_path)
    save_numpy_array(token_array, output_path)
    LOGGER.info("All done. Dataset ready: %s", output_path)


if __name__ == "__main__":
    main()