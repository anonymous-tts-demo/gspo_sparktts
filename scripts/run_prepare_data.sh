#!/bin/bash
# Phase 2: Prepare TRL dataset (with voice cloning global tokens)
# Usage: bash scripts/run_prepare_data.sh

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0

cd /data1/speech/nhandt23/06_binh/gspo_sparktts

echo "=========================================="
echo "Preparing TRL Dataset (Voice Cloning)"
echo "=========================================="

python scripts/prepare_data.py \
    --num_samples 5000 \
    --output_path data/trl_dataset.jsonl \
    --checkpoint_every 500 \
    --device cuda \
    2>&1 | tee logs/prepare_data_$(date +%Y%m%d_%H%M%S).log
