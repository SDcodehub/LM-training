#!/bin/bash
# ablations/ablation_lr.sh
# Compare a couple of learning rates under one WandB group.
set -euo pipefail

DATA_ARGS="--train_data data/train.npy --val_data data/val.npy"
MODEL_ARGS="--num_layers 4 --d_model 256 --num_heads 8 --d_ff 1024 --context_length 256 --vocab_size 10000"
TRAIN_STEM="--max_iters 2000 --batch_size 32 --eval_interval 200 --eval_iters 50 --log_interval 10"
PROJECT="LM_training"
GROUP="ablation_lr_v1"

echo "Starting LR Ablation: ${GROUP}"

for LR in 5e-4 1e-3 2e-3; do
  echo "Running lr=${LR}..."
  uv run python LM_training/scripts/train.py \
    ${DATA_ARGS} ${MODEL_ARGS} ${TRAIN_STEM} \
    --lr ${LR} \
    --wandb --wandb_project "${PROJECT}" \
    --run_name "lr_${LR}" \
    --group "${GROUP}"
done

echo "Ablation complete. Check WandB group: ${GROUP}"


