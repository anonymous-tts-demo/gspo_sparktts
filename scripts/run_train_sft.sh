#!/bin/bash
# Stage 1: SFT Training Launcher
# This script launches SFT training across available GPUs using accelerate.

set -euo pipefail

cd /data1/speech/nhandt23/06_binh/gspo_sparktts

export CUDA_VISIBLE_DEVICES="0,1" # Assuming 4 GPUs, adjust if needed
export ACCELERATE_LOG_LEVEL=info

echo "Starting SFT Training..."
/home/nhandt23/miniconda3/envs/spark/bin/accelerate launch \
    --config_file configs/accelerate_config.yaml \
    scripts/train_sft.py \
    --output_dir outputs/sft_run \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8

echo "Training Completed!"
