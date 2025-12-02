"""
Train a BPE model on a text file
"""
import os
import logging
import json
from binascii import b2a_hex
from heapq import nlargest
import regex as re
from LM_training.utils.logging_config import get_logger

log = get_logger()
# GPT 2 tokenizer pattern
# This regex splits the text into chunks of letters numbers or punctuations
# Its designed to keep the spaces attached to the words that follow them
split_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def pretokenise_text(input_path, special_tokens=None):

    if special_tokens is None:
        special_tokens = []

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
    for chunk in text_chunks:
        # Ignore the special tokens
        # handles in the vocab seperately
        if chunk in special_tokens:
            continue

        # find all pre-tokens in the chunk
        for word in re.findall(split_pattern, chunk):
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
    new_splits = {}
    for words_parts, count in splits.items():
        new_words_parts = []
        i = 0
        while i < len(words_parts):
            if words_parts[i:i+2] == pair:
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

    vocab_map = initialise_vocab(special_tokens)
    log.info("vocab size: %d", len(vocab_map))

    raw_splits = pretokenise_text(input_path, special_tokens)
    log.info("unique pretokenized byte-sequences: %d", len(raw_splits))
    splits = {tuple(bytes([b]) for b in word): count for word, count in raw_splits.items()}

    # Debug-only: top-K splits
    if log.isEnabledFor(logging.DEBUG):
        top_splits = nlargest(10, splits.items(), key=lambda kv: kv[1])
        for byte_seq, count in top_splits:
            hex_bytes = " ".join(f"{x[0]:02x}" for x in byte_seq)
            log.debug("split %r [%s] -> %d", byte_seq, hex_bytes, count)

    merges = []
    num_merges = vocab_size - len(vocab_map)
    
    for i in range(num_merges):
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

        log.info(f"Merge {i+1}/{num_merges}: {best_pair} -> {new_token_bytes}")

    log.info(f"Finished training. Final vocab size: {len(vocab_map)}")

    if save_prefix:
        save_tokenizer(vocab_map, merges, save_prefix)

    return vocab_map, merges

if __name__ == "__main__":
    # TODO remove this hard coding and make it a command line argument
    output_dir = "bpe_tokenizer"
    os.makedirs(output_dir, exist_ok=True)
    # sample_text = "Hello, world! It's a test — with numbers 123 and spaces.  New line?\nYes!"
    # temp_path = "/tmp/pretokenise_sample.txt"
    # with open(temp_path, "w", encoding="utf-8") as f:
    #     f.write(sample_text)
    input_path = "./data/TinyStoriesV2-GPT4-train.txt"
    vocab_size = 5000
    special_tokens = ["<|endoftext|>"]
    train_bpe(
        input_path, 
        vocab_size, 
        special_tokens, 
        save_prefix=os.path.join(output_dir, "tinystories-5k")
    )