#!/bin/bash
# Eval GSPO checkpoint vs Base model on multiple GPUs
#
# Usage:
#   nohup bash scripts/run_eval_multi_gpu.sh > logs/eval_multi_gspo_2800.log 2>&1 &

set -euo pipefail

CHECKPOINT="outputs/gspo_run/checkpoint-2800"
OUTPUT_DIR="outputs/eval_test_gspo_2800"
NUM_SAMPLES=2251
NUM_CHUNKS=2

echo "[$(date)] Starting MULTI-GPU GSPO-2800 eval..."
echo "  Checkpoint : $CHECKPOINT"
echo "  Samples    : $NUM_SAMPLES"
echo "  Output dir : $OUTPUT_DIR"
echo "  Chunks     : $NUM_CHUNKS"

mkdir -p "$OUTPUT_DIR"

# Run 2 jobs in parallel on GPU 2 and 3
for i in {0..1}; do
  GPU_ID=$((i + 2))
  echo "Starting chunk $i on GPU $GPU_ID..."
  CUDA_VISIBLE_DEVICES=$GPU_ID /home/nhandt23/miniconda3/envs/spark/bin/python scripts/eval_gspo_ckpt.py \
      --checkpoint "$CHECKPOINT" \
      --num_samples "$NUM_SAMPLES" \
      --output_dir "$OUTPUT_DIR" \
      --device cuda \
      --skip_base \
      --chunk_idx "$i" \
      --num_chunks "$NUM_CHUNKS" > "logs/eval_chunk_${i}.log" 2>&1 &
done

echo "All 4 background jobs submitted. Check logs/eval_chunk_*.log for individual progress."
echo "Waiting for all background jobs to finish..."
wait

CKPT_STEP=$(basename "$CHECKPOINT" | awk -F'-' '{print $NF}')
echo "[$(date)] All chunks completed! Combining CSVs..."
/home/nhandt23/miniconda3/envs/spark/bin/python scripts/combine_eval_csv.py \
    --output_dir "$OUTPUT_DIR" \
    --prefix "eval_gspo_${CKPT_STEP}"

echo "[$(date)] Multi-GPU Eval Finished!"
