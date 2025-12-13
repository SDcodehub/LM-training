"""
Train a BPE model on a text file (Optimized with Parallel Processing & Inverted Index)
"""
import os
import logging
import json
import time
import regex as re
import multiprocessing
import heapq
from typing import BinaryIO, List
from binascii import b2a_hex
from collections import defaultdict, Counter
from tqdm import tqdm
from LM_training.utils.logging_config import get_logger

log = get_logger()

# -----------------------------------------------------------------------------
# Helper: Parallel Chunk Boundary Finder
# -----------------------------------------------------------------------------
def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> List[int]:
    """
    Chunk the file into parts that can be counted independently.
    Ensures boundaries align with the special token (e.g., space) to avoid cutting words.
    """
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096 

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                # Set boundary strictly AFTER the token to ensure the token 
                # stays with the previous chunk (or is the split point)
                chunk_boundaries[bi] = initial_position + found_at
                break
            
            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))

# -----------------------------------------------------------------------------
# Helper: Worker for Parallel Processing
# -----------------------------------------------------------------------------
def _process_chunk_worker(args):
    """Worker function to process a single file chunk."""
    filename, start, end, special_pattern_str = args
    local_counts = Counter()
    
    # GPT-2 Split Pattern (compiled locally for the worker)
    # Note: We rely on the byte-level processing, so we assume UTF-8 text.
    local_split_pattern = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
    
    with open(filename, "rb") as f:
        f.seek(start)
        text_bytes = f.read(end - start)
        # Decode efficiently, ignoring errors at boundaries
        text = text_bytes.decode("utf-8", errors="ignore")

    if special_pattern_str:
        chunks = re.split(special_pattern_str, text)
    else:
        chunks = [text]

    for chunk in chunks:
        if not chunk: continue
        for word in local_split_pattern.findall(chunk):
            local_counts[word] += 1
            
    return local_counts

# -----------------------------------------------------------------------------
# Core Functions
# -----------------------------------------------------------------------------
def initialise_vocab(special_tokens):
    vocab = {i: bytes([i]) for i in range(256)}
    next_id = 256
    for token in special_tokens:
        vocab[next_id] = token.encode("utf-8")
        next_id += 1
    return vocab

def save_tokenizer(vocab, merges, prefix):
    vocab_file = f"{prefix}_vocab.json"
    merges_file = f"{prefix}_merges.txt"

    serializable_vocab = {
        token_id: list(byte_sequence) for token_id, byte_sequence in vocab.items()
    }
    with open(vocab_file, "w", encoding="utf-8") as f:
        json.dump(serializable_vocab, f, ensure_ascii=False, indent=2)
    log.info(f"Vocabulary saved to {vocab_file}")

    with open(merges_file, "w", encoding="utf-8") as f:
        for p1, p2 in merges:
            p1_hex = b2a_hex(p1).decode('ascii')
            p2_hex = b2a_hex(p2).decode('ascii')
            f.write(f"{p1_hex} {p2_hex}\n")
    log.info(f"Merges saved to {merges_file}")

# -----------------------------------------------------------------------------
# Main Training Logic
# -----------------------------------------------------------------------------
def train_bpe(input_path, vocab_size, special_tokens, save_prefix=None):
    start_time = time.time()
    
    # 1. Initialize Vocab
    vocab_map = initialise_vocab(special_tokens)
    log.info("Initial vocab size (bytes + special): %d", len(vocab_map))

    # 2. Parallel Pre-tokenization
    log.info(f"Chunking {input_path}...")
    
    # Prepare regex for special tokens
    special_pattern_str = None
    if special_tokens:
        escaped = [re.escape(t) for t in special_tokens]
        special_pattern_str = f"({'|'.join(escaped)})"

    # Find boundaries using 'space' as a safe split point for text
    num_processes = max(1, os.cpu_count() - 1)
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b" ") 

    tasks = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        tasks.append((input_path, start, end, special_pattern_str))

    log.info(f"Pre-tokenizing with {num_processes} cores...")
    global_counts = Counter()
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        for local_count in tqdm(pool.imap_unordered(_process_chunk_worker, tasks), total=len(tasks), desc="Pre-tokenizing"):
            global_counts.update(local_count)
            
    # Convert string words to byte tuples for BPE
    splits = {tuple(bytes([b]) for b in word.encode("utf-8")): count 
              for word, count in global_counts.items()}
    
    log.info(f"Unique words found: {len(splits)}")

    # 3. Build Inverted Index & Initial Stats
    # token_to_words: token -> set of words containing that token
    token_to_words = defaultdict(set)
    pair_stats = defaultdict(int)

    log.info("Building inverted index and stats...")
    for word, count in splits.items():
        # Indexing
        for token in word:
            token_to_words[token].add(word)
        # Stats
        for i in range(len(word) - 1):
            pair = (word[i], word[i+1])
            pair_stats[pair] += count

    # 4. Training Loop
    merges = []
    num_merges = vocab_size - len(vocab_map)
    
    # Heap for O(1) access to best pair. (-count, pair) for Min-Heap simulating Max-Heap
    stats_heap = []
    for pair, count in pair_stats.items():
        heapq.heappush(stats_heap, (-count, pair))

    log.info(f"Starting BPE training. Target merges: {num_merges}")
    
    progress_bar = tqdm(range(num_merges), desc="Training BPE")
    
    for i in progress_bar:
        # A. Get Best Pair (Lazy removal from heap)
        best_pair = None
        current_count = 0
        
        while stats_heap:
            neg_count, pair = heapq.heappop(stats_heap)
            real_count = pair_stats.get(pair, 0)
            # If heap count matches real count, it's valid. Else it's stale.
            if -neg_count == real_count:
                best_pair = pair
                current_count = real_count
                break
        
        if not best_pair:
            log.info("No more pairs to merge.")
            break

        # B. Create New Token
        p0, p1 = best_pair
        new_token_bytes = p0 + p1
        new_token_id = len(vocab_map)
        vocab_map[new_token_id] = new_token_bytes
        merges.append(best_pair)
        
        # Clean up stats for the merged pair itself
        del pair_stats[best_pair]

        # C. Merge in Splits (Using Inverted Index)
        # We only look at words containing p0
        words_to_check = list(token_to_words[p0])
        updates = defaultdict(int) # Track changes to update heap later

        for word in words_to_check:
            # 1. Validation checks
            if word not in splits: continue
            if p1 not in word: continue # Heuristic: p1 must also be in word

            # 2. Rebuild the word merging p0+p1
            new_word_list = []
            i_idx = 0
            changed = False
            n = len(word)
            
            while i_idx < n:
                if i_idx < n - 1 and word[i_idx] == p0 and word[i_idx+1] == p1:
                    new_word_list.append(new_token_bytes)
                    i_idx += 2
                    changed = True
                else:
                    new_word_list.append(word[i_idx])
                    i_idx += 1
            
            if changed:
                new_word = tuple(new_word_list)
                count = splits[word]
                
                # 3. Update Stats: Remove old pairs
                del splits[word]
                for j in range(len(word) - 1):
                    old_pair = (word[j], word[j+1])
                    pair_stats[old_pair] -= count
                    updates[old_pair] = pair_stats[old_pair]

                # 4. Update Stats: Add new pairs
                splits[new_word] = count
                for j in range(len(new_word) - 1):
                    new_pair = (new_word[j], new_word[j+1])
                    pair_stats[new_pair] += count
                    updates[new_pair] = pair_stats[new_pair]

                # 5. Update Index (Add new word to relevant buckets)
                # We don't remove `word` from p0/p1 buckets to save time (lazy removal)
                for token in new_word:
                    token_to_words[token].add(new_word)

        # D. Push updates to heap
        for pair, count in updates.items():
            if count > 0:
                heapq.heappush(stats_heap, (-count, pair))

        # Update progress bar occasionally
        if i % 100 == 0:
            progress_bar.set_postfix({"Best Count": current_count})

    total_time = time.time() - start_time
    log.info(f"Finished training in {total_time:.2f}s")

    if save_prefix:
        save_tokenizer(vocab_map, merges, save_prefix)

    return vocab_map, merges

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Train BPE tokenizer.")
    parser.add_argument("input_path", type=str, help="Path to training text")
    parser.add_argument("--output_dir", "-o", type=str, default="bpe_tokenizer")
    parser.add_argument("--vocab_size", "-v", type=int, default=5000)
    parser.add_argument("--prefix", "-p", type=str, default="tokenizer")
    parser.add_argument("--special_tokens", "-s", nargs="*", default=["<|endoftext|>"])

    args = parser.parse_args()

    if not os.path.exists(args.input_path):
        log.error(f"Input file not found: {args.input_path}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    full_save_prefix = os.path.join(args.output_dir, args.prefix)

    train_bpe(
        args.input_path, 
        args.vocab_size, 
        args.special_tokens, 
        save_prefix=full_save_prefix
    )