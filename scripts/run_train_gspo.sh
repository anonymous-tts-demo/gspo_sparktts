#!/bin/bash
# Stage 2: GSPO Training for Vi-SparkTTS-0.5B (Multi-GPU)
# Run AFTER Stage 1 SFT is complete (outputs/sft_run/final must exist)
#
# Usage:
#   nohup bash scripts/run_train_gspo.sh > logs/train_gspo_2000_steps.log 2>&1 &

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1
export ACCELERATE_LOG_LEVEL=info

# Kill any leftover distributed processes to free port
pkill -f train_gspo_sparktts 2>/dev/null || true
sleep 2

SFT_CKPT="outputs/sft_run/final"
OUTPUT_DIR="outputs/gspo_run_2000_steps"
DATASET="data/sft/train_sft.jsonl"
REWARD_DEVICE="cuda:0"

echo "[$(date)] Starting GSPO training (multi-GPU)..."

accelerate launch \
    --config_file configs/accelerate_config.yaml \
    --main_process_port 29501 \
    scripts/train_gspo_sparktts.py \
        --sft_checkpoint "$SFT_CKPT" \
        --dataset_path "$DATASET" \
        --output_dir "$OUTPUT_DIR" \
        --per_device_batch_size 2 \
        --num_generations 4 \
        --gradient_accumulation_steps 4 \
        --max_completion_length 1024 \
        --learning_rate 1e-6 \
        --num_train_epochs 3 \
        --max_steps 2000 \
        --temperature 0.9 \
        --epsilon 3e-4 \
        --epsilon_high 4e-4 \
        --reward_device "$REWARD_DEVICE"

echo "[$(date)] GSPO training complete."