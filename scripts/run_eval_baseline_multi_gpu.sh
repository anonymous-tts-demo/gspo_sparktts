#!/bin/bash
# Eval Base model on multiple GPUs

set -euo pipefail

OUTPUT_DIR="outputs/eval_test_baseline"
NUM_SAMPLES=2251
NUM_CHUNKS=2

echo "[$(date)] Starting MULTI-GPU Baseline eval..."
echo "  Samples    : $NUM_SAMPLES"
echo "  Output dir : $OUTPUT_DIR"
echo "  Chunks     : $NUM_CHUNKS"

mkdir -p "$OUTPUT_DIR"

# Run 2 jobs in parallel on GPU 0 and 1
for i in {0..1}; do
  GPU_ID=$i
  echo "Starting chunk $i on GPU $GPU_ID..."
  CUDA_VISIBLE_DEVICES=$GPU_ID /home/nhandt23/miniconda3/envs/spark/bin/python scripts/eval_gspo_ckpt.py \
      --num_samples "$NUM_SAMPLES" \
      --output_dir "$OUTPUT_DIR" \
      --device cuda \
      --chunk_idx "$i" \
      --num_chunks "$NUM_CHUNKS" > "logs/eval_baseline_chunk_${i}.log" 2>&1 &
done

echo "All 2 background jobs submitted. Check logs/eval_baseline_chunk_*.log for individual progress."
echo "Waiting for all background jobs to finish..."
wait

echo "[$(date)] All chunks completed! Combining CSVs..."
/home/nhandt23/miniconda3/envs/spark/bin/python scripts/combine_eval_csv.py \
    --output_dir "$OUTPUT_DIR" \
    --prefix "eval_base"

echo "[$(date)] Multi-GPU Eval Finished!"
