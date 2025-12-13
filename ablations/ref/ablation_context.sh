#!/bin/bash
# ablations/ablation_context.sh
# Compare context lengths under one WandB group.
set -euo pipefail

DATA_ARGS="--train_data data/train.npy --val_data data/val.npy"
PROJECT="LM_training"
GROUP="ablation_context_v1"

TRAIN="--max_iters 2000 --batch_size 32 --eval_interval 200 --eval_iters 50 --log_interval 10"
MODEL="--num_layers 4 --d_model 256 --num_heads 8 --d_ff 1024 --vocab_size 10000"

echo "Starting Context-Length Ablation: ${GROUP}"

for CTX in 128 256 384; do
  echo "Running context_length=${CTX}..."
  uv run python LM_training/scripts/train.py \
    ${DATA_ARGS} ${TRAIN} ${MODEL} \
    --context_length ${CTX} \
    --wandb --wandb_project "${PROJECT}" \
    --run_name "ctx_${CTX}" \
    --group "${GROUP}"
done

echo "Ablation complete. Check WandB group: ${GROUP}"


