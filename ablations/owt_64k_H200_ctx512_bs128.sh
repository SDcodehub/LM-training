#!/bin/bash
# ablations/ablation_template.sh
# Template for running a small, reproducible ablation with WandB grouping.
# Copy this file, set GROUP and the args below, and run:
#   bash ablations/ablation_template.sh
set -euo pipefail

# 1) Shared config (change these)
# NOTE: Update dataset paths.
# DATA_ARGS="--train_data ./output/file/npy/TinyStoriesV2-GPT4-train-10k.npy --val_data ./output/file/npy/TinyStoriesV2-GPT4-valid-10k.npy"
DATA_ARGS="--train_data ./output/file/npy/owt_train_v2_64k.npy --val_data ./output/file/npy/owt_valid_v2_64k.npy"
MODEL_ARGS="--num_layers 4 --d_model 256 --num_heads 8 --d_ff 1024 --context_length 512 --vocab_size 64000"
TRAIN_ARGS="--max_iters 15000 --batch_size 128 --lr 10e-4 --eval_interval 200 --eval_iters 50 --log_interval 10"
DEVICE_ARGS="--device cuda"
OUT_DIR="runs/owt_64k_H200_ctx512_BS128"
RUN_TAGS="owt_64k H200 BS128 step15k ctx512"
PROJECT="LM_training"

# 2) Name this ablation group
GROUP="owt_64k_H200"

echo "Starting Ablation: ${GROUP}"

# --- RUN A: Baseline ---
# keep the dataset vocabsize HW and variation in the run_name
echo "Running Baseline..."
uv run python ./LM_training/scripts/train.py \
  ${DATA_ARGS} ${MODEL_ARGS} ${TRAIN_ARGS} ${DEVICE_ARGS} \
  --out_dir "${OUT_DIR}" \
  --wandb --wandb_project "${PROJECT}" \
  --run_name "owt_64k_H200_ctx512_BS128" \
  --run_tags ${RUN_TAGS} \
  --group "${GROUP}"


echo "Ablation complete. Check WandB group: ${GROUP}"


