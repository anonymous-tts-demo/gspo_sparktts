"""Extract BiCodec tokens and format prompts for SFT Stage 1.

For each audio file in Train and Val sets:
1. Load audio waveform.
2. Use BiCodec to extract `global_tokens` and `semantic_tokens`.
3. Format exact prompt text combining: text, global, and semantic tokens.
4. Save as JSONL for SFTTrainer.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/prepare_sft_data.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import soundfile
import soxr
from loguru import logger
from tqdm import tqdm

SPARKTTS_MODEL_DIR = Path("/data1/speech/nhandt23/06_binh/spark_based/Vi-SparkTTS-0.5B")
WAVS_DIR = Path("/data1/speech/speechData/rawData/Public_Data/phoaudiobook/wavs")
SPLITS_DIR = Path("/data1/speech/nhandt23/06_binh/gspo_sparktts/data/splits")
OUTPUT_DIR = Path("/data1/speech/nhandt23/06_binh/gspo_sparktts/data/sft")
SAMPLE_RATE = 16_000

def load_audio(adfile: Path, sampling_rate: int = 16000, volume_normalize: bool = True) -> np.ndarray:
    audio, sr = soundfile.read(adfile)
    if len(audio.shape) > 1:
        audio = audio[:, 0]
    if sr != sampling_rate:
        audio = soxr.resample(audio, sr, sampling_rate, quality="VHQ")
    
    if volume_normalize:
        temp = np.sort(np.abs(audio))
        if temp[-1] < 0.1:
            audio = audio / max(temp[-1], 1e-3) * 0.1
        temp = temp[temp > 0.01]
        L = temp.shape[0]
        if L > 10:
            volume = np.mean(temp[int(0.9 * L) : int(0.99 * L)])
            audio = audio * np.clip(0.2 / volume, a_min=0.1, a_max=10)
        max_value = np.max(np.abs(audio))
        if max_value > 1:
            audio = audio / max_value
    return audio

def get_ref_clip(wav: np.ndarray, config) -> np.ndarray:
    ref_segment_length = (
        int(config.sample_rate * config.ref_segment_duration)
        // config.latent_hop_length
        * config.latent_hop_length
    )
    wav_length = len(wav)
    if ref_segment_length > wav_length:
        wav = np.tile(wav, ref_segment_length // wav_length + 1)
    return wav[:ref_segment_length]

def extract_tokens(model, processor, wav_path: Path):
    """Extract global and semantic tokens."""
    wav = load_audio(wav_path, sampling_rate=SAMPLE_RATE, volume_normalize=True)
    
    # Ref clip is required by global token extractor
    wav_ref_np = get_ref_clip(wav, model.config)
    wav_ref = torch.from_numpy(wav_ref_np).unsqueeze(0).float()
    wav_tensor = torch.from_numpy(wav).unsqueeze(0).float()

    device = next(model.parameters()).device
    wav_ref = wav_ref.to(device)
    wav_tensor = wav_tensor.to(device)

    with torch.no_grad():
        global_tokens, semantic_tokens = model.tokenize_audio(wav_tensor, wav_ref)
        
    return global_tokens.squeeze().tolist(), semantic_tokens.squeeze().tolist()

def build_sft_example(text: str, global_tokens: list[int], semantic_tokens: list[int]):
    """Format full generation trace for causal LM."""
    global_str = "".join([f"<|bicodec_global_{t}|>" for t in global_tokens])
    semantic_str = "".join([f"<|bicodec_semantic_{t}|>" for t in semantic_tokens])
    
    # User prompt portion
    prompt = (
        f"<|task_tts|>"
        f"<|start_content|>{text}<|end_content|>"
        f"<|start_global_token|>{global_str}<|end_global_token|>"
    )
    
    # Assistant response portion
    completion = f"<|start_semantic_token|>{semantic_str}<|end_semantic_token|>"
    
    # Full sequence: We train on this entire string, ignoring loss on the prompt.
    # Note: the EOS token is appended by the tokenizer if configured, but we add it explicitly.
    # From logs, eos_token_id is 151645 (for Qwen-based models commonly <|im_end|> or `<|endoftext|>`).
    # We will let SFTTrainer handle EOS token padding normally.
    full_text = prompt + completion + processor.tokenizer.eos_token
    
    return {
        "text": full_text,
        "prompt": prompt,
        "completion": completion
    }

def process_split(split_name: str, csv_path: Path, model, processor, args):
    df = pd.read_csv(csv_path)
    logger.info(f"Processing {split_name} split: {len(df)} samples")
    
    out_path = OUTPUT_DIR / f"{split_name}_sft.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    success = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for i, row in tqdm(df.iterrows(), total=len(df), desc=split_name):
            wav_path = WAVS_DIR / row["wav"]
            if not wav_path.exists():
                continue
                
            try:
                g_toks, s_toks = extract_tokens(model, processor, wav_path)
                example = build_sft_example(str(row["text"]), g_toks, s_toks)
                example["speaker"] = row["speaker"]
                example["wav"] = row["wav"]
                
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
                success += 1
            except Exception as e:
                logger.warning(f"Error on {wav_path}: {e}")
                
    logger.success(f"Saved {success}/{len(df)} {split_name} samples to {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # Load model
    from transformers import AutoModel, AutoProcessor
    logger.info("Loading Vi-SparkTTS model for token extraction...")
    global processor # Needed for eos_token access
    processor = AutoProcessor.from_pretrained(str(SPARKTTS_MODEL_DIR), trust_remote_code=True)
    model = AutoModel.from_pretrained(
        str(SPARKTTS_MODEL_DIR), trust_remote_code=True, torch_dtype=torch.float32
    ).to(args.device).eval()
    
    process_split("val", SPLITS_DIR / "val_2k.csv", model, processor, args)
    process_split("train", SPLITS_DIR / "train_10k.csv", model, processor, args)

if __name__ == "__main__":
    main()
