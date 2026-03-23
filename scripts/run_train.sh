#!/bin/bash
# Phase 4: GSPO Training with WER Reward
# Requires: 2 GPUs — training on both, reward models (BiCodec+Whisper) on cuda:0

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1

cd /data1/speech/nhandt23/06_binh/gspo_sparktts

python scripts/train_gspo_sparktts.py \
    --dataset_path data/trl_dataset.jsonl \
    --num_train_epochs 3 \
    --per_device_batch_size 1 \
    --num_generations 4 \
    --gradient_accumulation_steps 8 \
    --max_completion_length 2048 \
    --learning_rate 1e-6 \
    --temperature 0.8 \
    --reward_device cuda:0 \
    --use_peft \
    2>&1 | tee logs/train_wer_$(date +%Y%m%d_%H%M%S).log
