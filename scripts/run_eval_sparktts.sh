#!/bin/bash
# Evaluate SparkTTS: inference + WER/UTMOS/SIM
# Compares generated audio vs ground truth

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
cd /data1/speech/nhandt23/06_binh/gspo_sparktts

echo "Running SparkTTS evaluation..."
python scripts/eval_sparktts.py \
    --num_samples 15000 \
    --start_idx 0 \
    --device cuda \
    --use_cloning \
    --eval_models_dir download/tts_eval_models \
    --checkpoint_every 50 \
    2>&1 | tee logs/eval_sparktts_$(date +%Y%m%d_%H%M%S).log
    