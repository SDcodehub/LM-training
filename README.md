# LLM training

## Setup

### Quickstart (generic)

```sh
# Ensure uv is installed (https://github.com/astral-sh/uv)
uv --version

# Run any script with the managed environment
uv run python -V

# Run tests
uv run pytest
```

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

### General usage
- Use `uv run` to execute any Python entry point in the project. Examples:
  - Run a module or script:
    ```sh
    uv run python -u path/to/script.py
    ```
  - Install or pin additional dependencies (see uv docs) and re-run `uv run ...`.

## Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

## Train the BPE first
```
uv run ./cs336_basics/tokenizer/bpe/training.py
```


## using trained BPE tokenise the dataset
```
LOG_LEVEL=INFO 
```
```
uv run python ./cs336_basics/tokenizer/cli/tokenize_dataset.py \
      --input ./data/TinyStoriesV2-GPT4-train.txt \
      --vocab ./bpe_tokenizer/TinyStoriesV2-GPT4-train-10k_vocab.json \
      --merges ./bpe_tokenizer/TinyStoriesV2-GPT4-train-10k_merges.txt \
      --output ./data/TinyStoriesV2-GPT4-train-GPT4-valid_ids_10k.npy
```
