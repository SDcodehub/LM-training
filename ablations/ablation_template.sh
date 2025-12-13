#!/bin/bash
# ablations/ablation_template.sh
# Template for running a small, reproducible ablation with WandB grouping.
# Copy this file, set GROUP and the args below, and run:
#   bash ablations/ablation_template.sh
set -euo pipefail

# 1) Shared config (change these)
# NOTE: Update dataset paths.
DATA_ARGS="--train_data data/train.npy --val_data data/val.npy"
MODEL_ARGS="--num_layers 4 --d_model 256 --num_heads 8 --d_ff 1024 --context_length 256 --vocab_size 10000"
TRAIN_ARGS="--max_iters 2000 --batch_size 32 --eval_interval 200 --eval_iters 50 --log_interval 10"
PROJECT="LM_training"

# 2) Name this ablation group
GROUP="ablation_TEMPLATE_v1"

echo "Starting Ablation: ${GROUP}"

# --- RUN A: Baseline ---
echo "Running Baseline..."
uv run python LM_training/scripts/train.py \
  ${DATA_ARGS} ${MODEL_ARGS} ${TRAIN_ARGS} \
  --wandb --wandb_project "${PROJECT}" \
  --run_name "baseline" \
  --group "${GROUP}"

# --- RUN B: Variant ---
echo "Running Variant..."
# Example: change one hyperparameter below
uv run python LM_training/scripts/train.py \
  ${DATA_ARGS} \
  --num_layers 4 --d_model 384 --num_heads 8 --d_ff 1536 --context_length 256 --vocab_size 10000 \
  ${TRAIN_ARGS} \
  --wandb --wandb_project "${PROJECT}" \
  --run_name "variant" \
  --group "${GROUP}"

echo "Ablation complete. Check WandB group: ${GROUP}"


