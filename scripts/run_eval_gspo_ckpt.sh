#!/bin/bash
# Eval GSPO checkpoint-1000 vs Base model
# Run on a free GPU (training must be paused or use separate GPU)
#
# Usage:
#   bash scripts/run_eval_gspo_ckpt.sh
#   nohup bash scripts/run_eval_gspo_ckpt.sh > logs/eval_gspo_1000.log 2>&1 &

set -euo pipefail

export CUDA_VISIBLE_DEVICES=3

CHECKPOINT="outputs/gspo_run/checkpoint-2800"
OUTPUT_DIR="outputs/eval_test_gspo_2800"
NUM_SAMPLES=2251

echo "[$(date)] Starting GSPO-2800 eval..."
echo "  Checkpoint : $CHECKPOINT"
echo "  Samples    : $NUM_SAMPLES"
echo "  Output dir : $OUTPUT_DIR"

/home/nhandt23/miniconda3/envs/spark/bin/python scripts/eval_gspo_ckpt.py \
    --checkpoint "$CHECKPOINT" \
    --num_samples "$NUM_SAMPLES" \
    --output_dir "$OUTPUT_DIR" \
    --device cuda \
    --skip_base

echo "[$(date)] Eval complete. Results in $OUTPUT_DIR"
