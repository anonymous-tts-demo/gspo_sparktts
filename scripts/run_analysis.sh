#!/bin/bash
# Analyze PhoAudiobook ground truth: WER + UTMOS
# Step 1: Download eval models (one-time)
# Step 2: Run analysis

set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
cd /data1/speech/nhandt23/06_binh/gspo_sparktts

DOWNLOAD_DIR=download/tts_eval_models

# ── Step 1: Download eval models if not present ──
if [ ! -f "${DOWNLOAD_DIR}/mos/utmos22_strong_step7459_v1.pt" ]; then
    echo "Downloading k2-fsa/TTS_eval_models..."
    mkdir -p ${DOWNLOAD_DIR}
    python -c "
from huggingface_hub import snapshot_download
snapshot_download('k2-fsa/TTS_eval_models', local_dir='${DOWNLOAD_DIR}')
print('Download complete.')
"
    echo "Done downloading."
else
    echo "Eval models already present."
fi

# ── Step 2: Run analysis ──
echo "Running ground truth analysis..."
python scripts/analyze_ground_truth.py \
    --num_samples 0 \
    --device cuda \
    --eval_models_dir ${DOWNLOAD_DIR} \
    --checkpoint_every 1000 \
    2>&1 | tee logs/analysis_gt_$(date +%Y%m%d_%H%M%S).log
