"""
Tokenize a text dataset with the BPE tokenizer and save token ids to a .npy file.
Optimized with Multiprocessing.
"""

import argparse
import os
import sys
import concurrent.futures
import multiprocessing
from typing import Any, Iterable, List, Iterator

# --- Safe Imports ---
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs): return iterable

try:
    import numpy as np  # type: ignore
except Exception:
    np = None  # type: ignore[assignment]


def _ensure_package_import() -> None:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    package_dir = os.path.dirname(current_dir)
    assignment_root = os.path.dirname(package_dir)
    if assignment_root not in sys.path:
        sys.path.append(assignment_root)

_ensure_package_import()

# Import Logger safely
try:
    from LM_training.utils.logging_config import get_logger  # type: ignore
except Exception:
    import logging
    def get_logger() -> "logging.Logger":
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s:%(name)s:%(message)s"))
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

LOGGER = get_logger()

# --- Worker Global State ---
# This global variable exists separately in each child process
# It prevents us from reloading the JSON/TXT files for every chunk
WORKER_TOKENIZER = None 

def build_arg_parser() -> argparse.ArgumentParser:
    assignment_root = "."
    default_input = os.path.join(assignment_root, "data", "TinyStoriesV2-GPT4-train.txt")
    default_vocab = os.path.join(assignment_root, "bpe_tokenizer", "tinystories_vocab.json")
    default_merges = os.path.join(assignment_root, "bpe_tokenizer", "tinystories_merges.txt")

    parser = argparse.ArgumentParser(description="Parallel BPE Tokenization")
    parser.add_argument("--input", type=str, default=default_input)
    parser.add_argument("--vocab", type=str, default=default_vocab)
    parser.add_argument("--merges", type=str, default=default_merges)
    parser.add_argument("--output", type=str, default=None)
    # New argument for tuning performance
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of parallel processes")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Number of lines per chunk")
    return parser

def load_tokenizer(vocab_filepath: str, merges_filepath: str) -> Any:
    # Local import to avoid top-level issues
    from LM_training.tokenizer import Tokenizer  # type: ignore
    return Tokenizer.from_files(
        vocab_filepath=vocab_filepath,
        merges_filepath=merges_filepath,
        special_tokens=["<|endoftext|>"],
    )

def _worker_init(vocab_path: str, merges_path: str):
    """
    Initializer for worker processes. Loads the tokenizer once per process.
    """
    global WORKER_TOKENIZER
    WORKER_TOKENIZER = load_tokenizer(vocab_path, merges_path)

def _worker_process_chunk(lines: List[str]) -> List[int]:
    """
    Function running inside the child process.
    """
    global WORKER_TOKENIZER
    if WORKER_TOKENIZER is None:
        raise RuntimeError("Worker tokenizer not initialized!")
    
    # encode_iterable returns an iterator, we convert to list here
    return list(WORKER_TOKENIZER.encode_iterable(lines))

def read_in_chunks(file_path: str, chunk_size: int) -> Iterator[List[str]]:
    """
    Generator that yields lists of strings (chunks) from the file.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        chunk = []
        for line in f:
            chunk.append(line)
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

def tokens_to_array_uint16(token_ids: List[int]) -> Any:
    if np is None:
        raise RuntimeError("NumPy required.")
    return np.array(token_ids, dtype=np.uint16)

def derive_output_path(input_path: str, explicit_output: str | None) -> str:
    if explicit_output: return explicit_output
    base, _ = os.path.splitext(os.path.basename(input_path))
    return os.path.join(os.path.dirname(input_path), f"{base}_ids.npy")

def save_numpy_array(array: np.ndarray, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.save(output_path, array)

def main() -> None:
    args = build_arg_parser().parse_args()
    output_path = derive_output_path(args.input, args.output)

    if not os.path.isfile(args.input):
        LOGGER.error("Input not found: %s", args.input)
        sys.exit(1)

    LOGGER.info(f"Starting parallel tokenization with {args.workers} workers.")
    LOGGER.info(f"Chunk size: {args.lines_per_chunk if hasattr(args, 'lines_per_chunk') else args.chunk_size} lines")

    all_token_ids = []
    
    # We use ProcessPoolExecutor to manage the workers
    # max_workers=None defaults to the number of processors on the machine
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_worker_init,
        initargs=(args.vocab, args.merges)
    ) as executor:
        
        # 1. Create a generator for chunks
        chunk_generator = read_in_chunks(args.input, args.chunk_size)
        
        # 2. Map the chunks to the workers
        # We wrap chunk_generator in a list or keep it as generator? 
        # Keeping as generator is memory efficient, but tqdm needs to know count for %
        # For simplicity/speed, we won't pre-count lines, so tqdm will just show iteration count
        
        futures = []
        for chunk in chunk_generator:
            futures.append(executor.submit(_worker_process_chunk, chunk))
            
        LOGGER.info(f"Submitted {len(futures)} chunks to workers.")

        # 3. Gather results as they complete
        # as_completed yields futures as they finish
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing Chunks"):
            chunk_token_ids = future.result()
            all_token_ids.extend(chunk_token_ids)

    LOGGER.info("Tokenization complete. Total tokens: %d", len(all_token_ids))
    
    LOGGER.info("Converting to uint16 numpy array...")
    token_array = tokens_to_array_uint16(all_token_ids)

    LOGGER.info("Saving to %s", output_path)
    save_numpy_array(token_array, output_path)
    LOGGER.info("Done.")

if __name__ == "__main__":
    # Windows requires multiprocessing code to be guarded by if __name__ == "__main__"
    multiprocessing.freeze_support() 
    main()