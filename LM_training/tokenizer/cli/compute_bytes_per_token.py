import argparse
import os
import random
import sys
from typing import List


def _ensure_package_import() -> None:
    """
    Ensure the parent directory (assignment root) is on sys.path so that
    `LM_training.tokenizer` can be imported when this file is run as a script.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    assignment_root = os.path.dirname(current_dir)
    if assignment_root not in sys.path:
        sys.path.append(assignment_root)


_ensure_package_import()
from LM_training.tokenizer import Tokenizer  # noqa: E402


def load_docs_from_text_file(text_file_path: str) -> List[str]:
    """
    Load a text file and split it into documents using either the special token
    <|endoftext|> if present, or blank-line separation as a fallback.
    """
    with open(text_file_path, "r", encoding="utf-8") as file_handle:
        full_text = file_handle.read()

    if "<|endoftext|>" in full_text:
        raw_docs = [document.strip() for document in full_text.split("<|endoftext|>")]
    else:
        raw_docs = [document.strip() for document in full_text.split("\n\n")]

    return [document for document in raw_docs if document]


def sample_documents(documents: List[str], sample_size: int, seed: int) -> List[str]:
    """
    Deterministically sample up to `sample_size` documents from the provided list.
    """
    if not documents:
        return []
    random.seed(seed)
    actual_sample_size = min(sample_size, len(documents))
    return random.sample(documents, actual_sample_size)


def compute_bytes_per_token_for_docs(documents: List[str], tokenizer: Tokenizer) -> float:
    """
    Compute bytes/token as total UTF-8 bytes divided by total number of tokens
    across all provided documents.
    """
    total_bytes = 0
    total_tokens = 0
    for document in documents:
        total_bytes += len(document.encode("utf-8"))
        token_ids = tokenizer.encode(document)
        total_tokens += len(token_ids)

    if total_tokens == 0:
        return float("inf")
    return total_bytes / total_tokens


def build_arg_parser() -> argparse.ArgumentParser:
    assignment_root = "/Users/sagdesai/Desktop/work/building-transformer-lm/assignment1-basics"
    default_ts_text = os.path.join(
        assignment_root, "data", "TinyStoriesV2-GPT4-valid.txt"
    )
    default_ts_vocab = os.path.join(
        assignment_root, "bpe_tokenizer", "tinystories-5k_vocab.json"
    )
    default_ts_merges = os.path.join(
        assignment_root, "bpe_tokenizer", "tinystories-5k_merges.txt"
    )

    parser = argparse.ArgumentParser(
        description=(
            "Compute bytes/token for TinyStories (10K) and optionally OpenWebText (32K) "
            "by sampling N documents and encoding them with the provided tokenizers."
        )
    )

    parser.add_argument(
        "--ts-text",
        type=str,
        default=default_ts_text,
        help="Path to TinyStories text file (default: TinyStoriesV2-GPT4-valid.txt)",
    )
    parser.add_argument(
        "--ts-vocab",
        type=str,
        default=default_ts_vocab,
        help="Path to TinyStories 10K vocab JSON",
    )
    parser.add_argument(
        "--ts-merges",
        type=str,
        default=default_ts_merges,
        help="Path to TinyStories 10K merges TXT",
    )
    parser.add_argument(
        "--owt-text",
        type=str,
        default=None,
        help="Path to OpenWebText concatenated text file (optional)",
    )
    parser.add_argument(
        "--owt-vocab",
        type=str,
        default=None,
        help="Path to OpenWebText 32K vocab JSON (optional)",
    )
    parser.add_argument(
        "--owt-merges",
        type=str,
        default=None,
        help="Path to OpenWebText 32K merges TXT (optional)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Number of documents to sample from each corpus (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling (default: 42)",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    # TinyStories 10K
    ts_tokenizer = Tokenizer.from_files(
        vocab_filepath=args.ts_vocab,
        merges_filepath=args.ts_merges,
        special_tokens=["<|endoftext|>"],
    )
    ts_documents = sample_documents(
        load_docs_from_text_file(args.ts_text), sample_size=args.n, seed=args.seed
    )
    ts_bytes_per_token = compute_bytes_per_token_for_docs(ts_documents, ts_tokenizer)
    print(f"TinyStories tokenizer (10K) bytes/token: {ts_bytes_per_token:.3f}")

    # OpenWebText 32K (optional; only compute if all paths provided)
    if args.owt_text and args.owt_vocab and args.owt_merges:
        owt_tokenizer = Tokenizer.from_files(
            vocab_filepath=args.owt_vocab,
            merges_filepath=args.owt_merges,
            special_tokens=["<|endoftext|>"],
        )
        owt_documents = sample_documents(
            load_docs_from_text_file(args.owt_text), sample_size=args.n, seed=args.seed
        )
        owt_bytes_per_token = compute_bytes_per_token_for_docs(owt_documents, owt_tokenizer)
        print(f"OpenWebText tokenizer (32K) bytes/token: {owt_bytes_per_token:.3f}")
    else:
        print(
            "OpenWebText paths not fully provided; skipping OpenWebText computation. "
            "Provide --owt-text, --owt-vocab, and --owt-merges to compute."
        )


if __name__ == "__main__":
    main()


