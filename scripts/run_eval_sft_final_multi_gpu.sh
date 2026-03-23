#!/bin/bash
# Eval multiple SFT checkpoints in parallel on 4 GPUs

set -euo pipefail

BASE_CKPT_DIR="outputs/sft_run"
CHECKPOINTS=("checkpoint-3000" "checkpoint-4000" "checkpoint-5000" "final")
GPUS=(0 1 2 3)
NUM_SAMPLES=100

echo "[$(date)] Starting Parallel Evaluation for 4 checkpoints..."
echo "  Checkpoints : ${CHECKPOINTS[*]}"
echo "  Samples     : $NUM_SAMPLES"

# Tạo thư mục logs nếu chưa có
mkdir -p logs

for i in "${!CHECKPOINTS[@]}"; do
  CKPT_NAME=${CHECKPOINTS[$i]}
  GPU_ID=${GPUS[$i]}
  CKPT_PATH="$BASE_CKPT_DIR/$CKPT_NAME"
  OUTPUT_DIR="outputs/eval_sft_$CKPT_NAME"
  
  mkdir -p "$OUTPUT_DIR"
  
  echo "Launching $CKPT_NAME on GPU $GPU_ID..."
  
  CUDA_VISIBLE_DEVICES=$GPU_ID /home/nhandt23/miniconda3/envs/spark/bin/python scripts/eval_gspo_ckpt.py \
      --checkpoint "$CKPT_PATH" \
      --num_samples "$NUM_SAMPLES" \
      --output_dir "$OUTPUT_DIR" \
      --device cuda > "logs/eval_sft_${CKPT_NAME}.log" 2>&1 &
done

echo "All 4 background jobs submitted. Check logs/eval_sft_*.log for progress."
echo "Waiting for all evaluation jobs to finish..."
wait

echo "[$(date)] All evaluations completed! Combining CSVs..."

for CKPT_NAME in "${CHECKPOINTS[@]}"; do
    OUTPUT_DIR="outputs/eval_sft_$CKPT_NAME"
    /home/nhandt23/miniconda3/envs/spark/bin/python scripts/combine_eval_csv.py \
        --output_dir "$OUTPUT_DIR" \
        --prefix "eval_sft_$CKPT_NAME"
done

echo "[$(date)] Multi-GPU Eval Finished for all checkpoints!"