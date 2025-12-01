## Transformer LM Training

A compact, educational codebase for building and training a Transformer language model end‑to‑end:
from byte-pair encoding (BPE) tokenization, through a minimal Transformer implementation
with RoPE and RMSNorm, to training, evaluation, and text generation. The repo is organized
for readability and testability, with lightweight dependencies managed via `uv`.

### What’s inside
- **Tokenizer (BPE)**: Train a vocabulary and merges, tokenize raw text, and compute bytes-per-token.
- **Model**: Attention, RoPE, RMSNorm, FFN, embeddings, and a Transformer LM composed from simple modules.
- **Training**: Scripts for training, checkpointing, logging, and generation.
- **Optimizers**: SGD, AdamW, schedulers, and gradient clipping.
- **Utilities**: Reproducible env via `uv`, logging config, and checkpoint serialization.
- **Tests**: Unit tests and reference snapshots to validate each component.

### Quickstart

```sh
# 1) Ensure uv is installed (https://github.com/astral-sh/uv)
uv --version

# 2) Run inside the managed environment
uv run python -V

# 3) Run tests
uv run pytest
```

Environment is fully managed by `uv`, so dependencies are solved and cached automatically.
You can run any entry point with:

```sh
uv run <python_or_module_path>
```

### Usage and end‑to‑end workflows
For end‑to‑end commands (data download, BPE training, dataset tokenization, model training, and generation),
see the detailed instructions in [USAGE.md](./USAGE.md). That document contains copy‑pasteable commands and options.

### Repository layout

```text
LM_training/
  data_load/           # dataset loading
  nn/                  # core neural modules and optimizers
    functional.py
    modules/
      attention.py, rope.py, rmsnorm.py, ffn.py, transformer.py, embedding.py, linear.py
    optim/
      adamw.py, sgd.py, scheduler.py, clipping.py
  tokenizer/           # BPE training and CLI tools
    bpe/
    cli/
  scripts/
    train.py           # training entry point
  utils/
    checkpointing.py, logging_config.py
tests/                 # unit tests and snapshots
```

### Testing
```sh
uv run pytest
```
If you’re implementing components from scratch, some tests may initially fail with `NotImplementedError`.
Hook your implementation via the adapter functions in [`tests/adapters.py`](./tests/adapters.py).

### Data
TinyStories and a small OpenWebText sample are used in examples. For curated, step‑by‑step
data download and preprocessing instructions, see [USAGE.md](./USAGE.md).

### Notes
- Designed for clarity over raw performance; suitable for learning, prototyping, and experimentation.
- Reproducible environments with `uv` (see `uv.lock`).

For full workflows, tunables, and CLI examples, head to [USAGE.md](./USAGE.md).
