#!/bin/bash
# Prepares data for SFT Stage 1
# Tokenizes audio to semantic and global tokens

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
cd /data1/speech/nhandt23/06_binh/gspo_sparktts

echo "Extracting tokens for SFT..."
/home/nhandt23/miniconda3/envs/spark/bin/python scripts/prepare_sft_data.py \
    --device cuda \
    2>&1 | tee logs/prepare_sft_data_$(date +%Y%m%d_%H%M%S).log

echo "Done!"
