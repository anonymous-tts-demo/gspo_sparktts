#!/bin/bash
# Phase 1: Run SparkTTS baseline evaluation
# Usage: bash scripts/run_baseline.sh

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1

cd /data1/speech/nhandt23/06_binh/gspo_sparktts

echo "=========================================="
echo "SparkTTS Baseline Evaluation"
echo "=========================================="

python scripts/run_baseline.py \
    --num_samples 100 \
    --start_idx 0 \
    --device cuda \
    --use_cloning \
    2>&1 | tee logs/baseline_$(date +%Y%m%d_%H%M%S).log
