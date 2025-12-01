# LM Training repository
Training and model architecture ablations

## Env setup
Run uv tests to setup the environment
```
uv run pytest
```

## data download

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
```

Run the BPE preprocessing sample with logging from the repository root.

### Default run (INFO level)
```bash
cd assignment1-basics
uv run python -u LM_training/train_bpe.py | cat
```

### Debug run (verbose top‑K details)
```bash
cd assignment1-basics
LOG_LEVEL=DEBUG uv run python -u LM_training/train_bpe.py | cat
```

Notes:
- INFO shows high‑level counts. Set `LOG_LEVEL=DEBUG` to include top‑K splits and pair stats.
- Outputs go to stdout; no files are written during normal debugging.




### Tokenization script
- `LM_training/scripts/tokenize_dataset.py`: tokenizes a text file with the TinyStories BPE and writes a `.npy` of uint16 token ids. See the script header for usage examples (including `uv run`).

### Benchmark (timing)
You can use [`hyperfine`](https://github.com/sharkdp/hyperfine) to benchmark the runtime of the BPE training script.
cd assignment1-basics
```bash
hyperfine --warmup 1 --runs 2 --show-output 'uv run -- python -u LM_training/train_bpe.py'
```

Note the commit id using following command
```bash
git rev-parse --short HEAD
```

### Performance results
Dataset used is `TinyStoriesV2-GPT4-train.txt`

| Commit  | Runs | Warmup | Mean ± σ              | Range (min…max)           | Command                                       |
|---------|------|--------|-----------------------|---------------------------|-----------------------------------------------|
| 00312fb | 2    | 1      | 166.923 s ± 0.600 s   | 166.498 s … 167.347 s     | `uv run -- python -u LM_training/train_bpe.py` |

#### TODO Using above setup think of optimising the train_BPE code

### High‑throughput tokenization (chunked streaming)

- **Why not line‑by‑line?**
  - **I/O overhead:** many tiny reads slow overall throughput
  - **CPU underutilization:** compute waits on disk between short lines

- **Do this instead:** read fixed‑size chunks (e.g., 1–4 MB) and feed them to `encode_iterable`.

- **Helper: chunked reader**

```python
def read_in_chunks(file_object, chunk_size: int = 1024 * 1024):
    """Yield text in fixed-size chunks (default 1 MB)."""
    while True:
        data = file_object.read(chunk_size)
        if not data:
            break
        yield data
```

- **Use with `encode_iterable`:**

```python
# tokenizer: your initialized Tokenizer
from LM_training.tokonizer import Tokenizer  # if needed

with open("data/TinyStoriesV2-GPT4-train.txt", "r", encoding="utf-8") as fh:
    chunk_gen = read_in_chunks(fh, chunk_size=4 * 1024 * 1024)  # 4 MB
    for token_id in tokenizer.encode_iterable(chunk_gen):
        # process token_id (streaming)
        ...
```

- **Tips:**
  - **Chunk size:** start with 1–4 MB; tune for your disk/CPU.
  - **Memory‑safe:** streaming keeps peak memory low while saturating compute.

### Compression ratio (bytes/token)

- **Definition:** bytes/token = total UTF‑8 bytes ÷ total token count, averaged over sampled docs.

- **TinyStories (10K) over 10 docs:**
```bash
cd assignment1-basics
uv run python LM_training/compute_bytes_per_token.py | cat
```

- **Add OpenWebText (32K) paths:**
```bash
cd assignment1-basics
uv run python LM_training/compute_bytes_per_token.py \
  --owt-text /absolute/path/to/openwebtext.txt \
  --owt-vocab /absolute/path/to/openwebtext-32k_vocab.json \
  --owt-merges /absolute/path/to/openwebtext-32k_merges.txt | cat
```

- **Options:**
  - `--n 10` to change the number of sampled documents
  - `--seed 42` for deterministic sampling

- **Deliverable (example, 1–2 sentences):**
  - "TinyStories 10K: X.XXX bytes/token; OpenWebText 32K: Y.YYY bytes/token (10 sampled docs)."

#### Reference results

- TinyStories 10K: 4.058 bytes/token (10 sampled docs, seed 42)
- TinyStories 32K: 4.072 bytes/token (10 sampled docs, seed 42)
