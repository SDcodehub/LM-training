#!/bin/bash
# ablations/ablation_depth.sh
# Compare different numbers of layers under one WandB group.
set -euo pipefail

DATA_ARGS="--train_data data/train.npy --val_data data/val.npy"
PROJECT="LM_training"
GROUP="ablation_depth_v1"

# Keep other dims fixed for fairness
COMMON="--d_model 256 --num_heads 8 --d_ff 1024 --context_length 256 --vocab_size 10000"
TRAIN="--max_iters 2000 --batch_size 32 --eval_interval 200 --eval_iters 50 --log_interval 10"

echo "Starting Depth Ablation: ${GROUP}"

for LAYERS in 2 4 6; do
  echo "Running num_layers=${LAYERS}..."
  uv run python LM_training/scripts/train.py \
    ${DATA_ARGS} ${TRAIN} ${COMMON} \
    --num_layers ${LAYERS} \
    --wandb --wandb_project "${PROJECT}" \
    --run_name "layers_${LAYERS}" \
    --group "${GROUP}"
done

echo "Ablation complete. Check WandB group: ${GROUP}"


