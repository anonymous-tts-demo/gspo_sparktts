"""Phase 2: Convert PhoAudiobook filtered data → TRL-compatible dataset.

Pre-tokenizes reference audio into speaker global tokens using BiCodec,
then formats prompts for GRPOTrainer.

Usage:
    # With voice cloning (extract global tokens from ref audio):
    CUDA_VISIBLE_DEVICES=0 python scripts/prepare_data.py \
        --num_samples 5000 \
        --output_path data/trl_dataset.jsonl

    # With controllable TTS (no ref audio needed, faster):
    CUDA_VISIBLE_DEVICES=0 python scripts/prepare_data.py \
        --num_samples 5000 --skip_tokenize \
        --output_path data/trl_dataset_ctrl.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from loguru import logger

# ════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════
SPARKTTS_MODEL_DIR = Path("/data1/speech/nhandt23/06_binh/spark_based/Vi-SparkTTS-0.5B")
DATA_CSV = Path("/data1/speech/nhandt23/06_binh/chatter_v2v/data/normalized/phoaudiobook/filtered_15k.csv")
WAVS_DIR = Path("/data1/speech/speechData/rawData/Public_Data/phoaudiobook/wavs")
OUTPUT_DIR = Path("/data1/speech/nhandt23/06_binh/gspo_sparktts/data")
SAMPLE_RATE = 16_000


def extract_global_tokens(
    model, processor, wav_path: Path
) -> Optional[list[int]]:
    """Extract speaker global tokens from a wav file using BiCodec.
    
    Uses the processor's voice cloning mode to extract global tokens,
    which represent the speaker identity.
    """
    try:
        inputs = processor(
            text="test",  # dummy text, we only need global tokens
            prompt_speech_path=str(wav_path),
            return_tensors="pt",
        )
        global_ids = inputs.get("global_token_ids_prompt", None)
        if global_ids is not None:
            return global_ids.squeeze(0).tolist()
        return None
    except Exception as e:
        logger.warning(f"Failed to extract global tokens from {wav_path}: {e}")
        return None


def build_cloning_prompt(text: str, global_tokens: list[int]) -> str:
    """Build voice cloning prompt string (matching processor format).
    
    Format: <|task_tts|><|start_content|>{text}<|end_content|>
            <|start_global_token|>{global_tokens}<|end_global_token|>
    
    The model's completion should generate:
        <|start_semantic_token|>{semantic_tokens}<|end_semantic_token|>
    """
    global_str = "".join(f"<|bicodec_global_{t}|>" for t in global_tokens)
    
    return (
        f"<|task_tts|>"
        f"<|start_content|>{text}<|end_content|>"
        f"<|start_global_token|>{global_str}<|end_global_token|>"
    )


def build_controllable_prompt(text: str, gender: str = "female",
                               pitch: str = "moderate", speed: str = "moderate") -> str:
    """Build controllable TTS prompt string (matching processor format).
    
    Format: <|task_controllable_tts|><|start_content|>{text}<|end_content|>
            <|start_style_label|>{attributes}<|end_style_label|>
    """
    gender_map = {"female": 0, "male": 1}
    level_map = {"very_low": 0, "low": 1, "moderate": 2, "high": 3, "very_high": 4}
    
    g_id = gender_map[gender]
    p_id = level_map[pitch]
    s_id = level_map[speed]
    
    return (
        f"<|task_controllable_tts|>"
        f"<|start_content|>{text}<|end_content|>"
        f"<|start_style_label|>"
        f"<|gender_{g_id}|><|pitch_label_{p_id}|><|speed_label_{s_id}|>"
        f"<|end_style_label|>"
    )


def main():
    parser = argparse.ArgumentParser(description="Prepare TRL dataset from PhoAudiobook")
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--output_path", type=str, default="data/trl_dataset.jsonl")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip_tokenize", action="store_true",
                        help="Skip BiCodec tokenization, use controllable TTS prompt instead")
    parser.add_argument("--checkpoint_every", type=int, default=500,
                        help="Save checkpoint every N samples")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path("/data1/speech/nhandt23/06_binh/gspo_sparktts") / args.output_path

    # Load data
    df = pd.read_csv(DATA_CSV)
    df = df.head(args.num_samples)
    logger.info(f"Processing {len(df):,} samples")

    model, processor = None, None
    if not args.skip_tokenize:
        from transformers import AutoModel, AutoProcessor
        logger.info("Loading SparkTTS for global token extraction...")
        # NOTE: Must use float32, BiCodec's cuFFT doesn't support BFloat16
        model = AutoModel.from_pretrained(
            str(SPARKTTS_MODEL_DIR), trust_remote_code=True, torch_dtype=torch.float32,
        )
        model.eval()
        processor = AutoProcessor.from_pretrained(
            str(SPARKTTS_MODEL_DIR), trust_remote_code=True,
        )
        processor.link_model(model)
        logger.success("SparkTTS loaded for tokenization")

    # Process samples
    dataset_rows = []
    failed_count = 0
    for i, row in df.iterrows():
        text = str(row["text"])
        wav_name = row["wav"]
        wav_path = WAVS_DIR / wav_name

        if args.skip_tokenize:
            # Controllable TTS: no reference audio needed
            prompt_str = build_controllable_prompt(text)
        else:
            # Voice cloning: extract global tokens
            if not wav_path.exists():
                logger.warning(f"[{i}] WAV not found: {wav_path}")
                failed_count += 1
                continue

            global_tokens = extract_global_tokens(model, processor, wav_path)
            if global_tokens is None:
                logger.warning(f"[{i}] Failed to extract tokens for {wav_name}")
                failed_count += 1
                continue

            prompt_str = build_cloning_prompt(text, global_tokens)

        # TRL GRPOTrainer expects "prompt" field as list of chat messages
        dataset_rows.append({
            "prompt": [{"role": "user", "content": prompt_str}],
            "reference_text": text,
            "reference_wav": wav_name,
            "speaker": row["speaker"],
        })

        if (i + 1) % args.checkpoint_every == 0:
            logger.info(f"[{i+1:,}/{len(df):,}] processed ({failed_count} failed)")
            # Save checkpoint
            ckpt_path = output_path.parent / f"{output_path.stem}_ckpt.jsonl"
            with open(ckpt_path, "w", encoding="utf-8") as f:
                for r in dataset_rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Save final
    with open(output_path, "w", encoding="utf-8") as f:
        for row in dataset_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    logger.success(
        f"Saved {len(dataset_rows):,} samples to {output_path} "
        f"(failed: {failed_count})"
    )


if __name__ == "__main__":
    main()
