"""
Train a BPE model on a text file
"""
import os
import logging
import json
import time
from binascii import b2a_hex
from heapq import nlargest
import regex as re
from tqdm import tqdm
from LM_training.utils.logging_config import get_logger

log = get_logger()
# GPT 2 tokenizer pattern
# This regex splits the text into chunks of letters numbers or punctuations
# Its designed to keep the spaces attached to the words that follow them
# Compile pattern once for performance
SPLIT_PATTERN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def pretokenise_text(input_path, special_tokens=None):

    if special_tokens is None:
        special_tokens = []

    log.info(f"Reading {input_path}...")
    with open(input_path, "r", encoding="utf-8") as read_file:
        text = read_file.read()
   

    # Build a regex pattern to split the text by any of the special tokens.
    # re.escape is used in case a special token contains characters with special
    # meaning in regex, like '|'.
    if special_tokens:
        special_pattern = "|".join(re.escape(token) for token in special_tokens)
        text_chunks = re.split(f"({special_pattern})", text)
    else:
        text_chunks = [text]

    # pre tokenize the text chunks seperately
    word_counts = {}

    log.info("Pre-tokenizing text...")
    for chunk in tqdm(text_chunks, desc="Chunking"):
        # Ignore the special tokens
        # handles in the vocab seperately
        if chunk in special_tokens:
            continue

        # find all pre-tokens in the chunk
        for word in SPLIT_PATTERN.findall(chunk):
            word_counts[word] = word_counts.get(word, 0) + 1

    # BPE generally works on the byte sequences to converting the strings into byte sequences
    splits = {word.encode("utf-8"): count for word, count in word_counts.items()}
    return splits

def initialise_vocab(special_tokens):
    # vocab is a mapping from the integer ID to the byte sequence
    vocab = {i: bytes([i]) for i in range(256)}

    #add special tokens
    next_id = 256
    for token in special_tokens:
        vocab[next_id] = token.encode("utf-8")
        next_id += 1
    
    return vocab

def get_stats(splits):
    """
    Give n splits pre tokenized, return a dictionary of pairs of byte sequences and their counts
    """
    stats = {}
    for word_part, count in splits.items():
        # A word is represented as the byte sequences
        for i in range(len(word_part)-1):
            # form the pair of adjacent tokens
            pair = (word_part[i], word_part[i+1])
            # increment the count for the pair
            stats[pair] = stats.get(pair, 0) + count
    return stats

def merge_splits(splits, pair, new_token):
    """Replaces all the occuraces of pair in the splits with new_token"""
    p0, p1 = pair
    new_splits = {}
    for words_parts, count in splits.items():
        # Optimization: If the pair isn't in this word, skip the heavy logic
        # Note: This is a heuristic check; p0 might exist without p1 following it.
        # But it saves time for words that contain neither byte.
        if p0 not in words_parts:
             new_splits[words_parts] = count
             continue

        new_words_parts = []
        i = 0
        n = len(words_parts)

        while i < n:
            # Optimized Check: Direct index access is faster than slicing [i:i+2]
            if i < n - 1 and words_parts[i] == p0 and words_parts[i+1] == p1:
                new_words_parts.append(new_token)
                i += 2
            else:
                new_words_parts.append(words_parts[i])
                i += 1
        new_splits[tuple(new_words_parts)] = count
    return new_splits


def save_tokenizer(vocab, merges, prefix):
    """Saves the vocabulary and merges to files."""
    vocab_file = f"{prefix}_vocab.json"
    merges_file = f"{prefix}_merges.txt"

    # 1. Save the vocabulary
    # We need to convert bytes to a JSON-serializable format (list of ints)
    serializable_vocab = {
        token_id: list(byte_sequence) for token_id, byte_sequence in vocab.items()
    }
    with open(vocab_file, "w", encoding="utf-8") as f:
        json.dump(serializable_vocab, f, ensure_ascii=False, indent=2)
    log.info(f"Vocabulary saved to {vocab_file}")

    # 2. Save the merges
    # We save as hex to avoid any issues with special characters or spaces
    with open(merges_file, "w", encoding="utf-8") as f:
        for p1, p2 in merges:
            p1_hex = b2a_hex(p1).decode('ascii')
            p2_hex = b2a_hex(p2).decode('ascii')
            f.write(f"{p1_hex} {p2_hex}\n")
    log.info(f"Merges saved to {merges_file}")


def train_bpe(input_path, vocab_size, special_tokens, save_prefix=None):
    """Main function for training BPE model"""

    start_time = time.time()
    vocab_map = initialise_vocab(special_tokens)
    log.info("vocab size: %d", len(vocab_map))

    raw_splits = pretokenise_text(input_path, special_tokens)
    log.info("unique pretokenized byte-sequences: %d", len(raw_splits))

    # Convert raw bytes keys to tuple of bytes for mutability simulation
    splits = {tuple(bytes([b]) for b in word): count for word, count in raw_splits.items()}


    merges = []
    num_merges = vocab_size - len(vocab_map)

    log.info(f"Starting BPE training. Target merges: {num_merges}")

    # WRAPPER: tqdm for progress bar
    progress_bar = tqdm(range(num_merges), desc="Training BPE")
    
    for i in progress_bar:
        # Get the stats of the splits
        pair_stats = get_stats(splits)

        if not pair_stats:
            # If there are no more adjacent-byte pairs to merge, break
            log.info("No more adjacent-byte pairs to merge")
            break

        log.info("unique adjacent-byte pairs: %d", len(pair_stats))

        # Debug-only: top-K pairs
        if log.isEnabledFor(logging.DEBUG):
            top_pairs = nlargest(20, pair_stats.items(), key=lambda kv: kv[1])
            for (a, b), count in top_pairs:
                log.debug("pair (%d,%d) [%02x %02x] -> %d", a[0], b[0], a[0], b[0], count)

        # Get the top pair by the count        
        best_pair = max(pair_stats, key=lambda pair: (pair_stats[pair], pair))

        # Create new token and perform the merge
        p1, p2 = best_pair
        new_token_bytes = p1 + p2
        new_token_id = len(vocab_map)

        # Upsatte vocab, merges, splits
        vocab_map[new_token_id] = new_token_bytes
        merges.append(best_pair)
        splits = merge_splits(splits, best_pair, new_token_bytes)

        # Update tqdm description with current stats rarely (to save rendering time)
        if i % 10 == 0:
            progress_bar.set_postfix({"Best Pair": f"{p1}+{p2}", "Count": pair_stats[best_pair]})
        
        # LOGGING STRATEGY: Only log to console every X steps
        if i % 100 == 0:
             if log.isEnabledFor(logging.DEBUG):
                 log.debug(f"Merge {i+1}: {best_pair} -> {new_token_bytes}")

    total_time = time.time() - start_time
    log.info(f"Finished training in {total_time:.2f}s. Final vocab size: {len(vocab_map)}")

    if save_prefix:
        save_tokenizer(vocab_map, merges, save_prefix)

    return vocab_map, merges

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Train a Byte-Pair Encoding (BPE) tokenizer on a text file."
    )

    # 1. Input File (Positional Argument - Required)
    parser.add_argument(
        "input_path", 
        type=str, 
        help="Path to the training text file (e.g., data/corpus.txt)"
    )

    # 2. Output Directory (Optional)
    parser.add_argument(
        "--output_dir", "-o", 
        type=str, 
        default="bpe_tokenizer",
        help="Directory to save the vocab and merges files (default: bpe_tokenizer)"
    )

    # 3. Vocab Size (Optional)
    parser.add_argument(
        "--vocab_size", "-v", 
        type=int, 
        default=5000,
        help="Target vocabulary size (default: 5000)"
    )

    # 4. Filename Prefix (Optional)
    parser.add_argument(
        "--prefix", "-p", 
        type=str, 
        default="tokenizer",
        help="Prefix for the saved files (default: tokenizer)"
    )

    # 5. Special Tokens (Optional - List)
    parser.add_argument(
        "--special_tokens", "-s", 
        nargs="*", 
        default=["<|endoftext|>"],
        help="List of special tokens to include (default: <|endoftext|>)"
    )

    args = parser.parse_args()

    # Validation: Check if input file exists
    if not os.path.exists(args.input_path):
        log.error(f"Input file not found: {args.input_path}")
        sys.exit(1)

    # Prepare output directory
    os.makedirs(args.output_dir, exist_ok=True)
    full_save_prefix = os.path.join(args.output_dir, args.prefix)

    log.info(f"Training BPE with vocab_size={args.vocab_size} on {args.input_path}")
    log.info(f"Special tokens: {args.special_tokens}")

    # Run Training
    train_bpe(
        args.input_path, 
        args.vocab_size, 
        args.special_tokens, 
        save_prefix=full_save_prefix
    )