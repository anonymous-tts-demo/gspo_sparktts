"""Phase 1: Run SparkTTS baseline inference on PhoAudiobook filtered data.

Generates speech for each sample, saves audio, computes WER/SIM/UTMOS metrics.
Results saved to CSV for reward function design.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/run_baseline.py \
        --num_samples 100 \
        --batch_size 1
"""

import argparse
import re
import time
import unicodedata
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from loguru import logger

# ════════════════════════════════════════════════════════════════
# CONFIG — paths
# ════════════════════════════════════════════════════════════════
SPARKTTS_MODEL_DIR = Path("/data1/speech/nhandt23/06_binh/spark_based/Vi-SparkTTS-0.5B")
DATA_CSV = Path("/data1/speech/nhandt23/06_binh/chatter_v2v/data/normalized/phoaudiobook/filtered_15k.csv")
WAVS_DIR = Path("/data1/speech/speechData/rawData/Public_Data/phoaudiobook/wavs")
WHISPER_MODEL_DIR = Path("/data1/speech/nhandt23/06_binh/models/openai--whisper-large-v3")
OUTPUT_DIR = Path("/data1/speech/nhandt23/06_binh/gspo_sparktts/outputs/baseline")

SAMPLE_RATE = 16_000


# ════════════════════════════════════════════════════════════════
# TEXT NORMALIZE
# ════════════════════════════════════════════════════════════════
def normalize_text(text: str) -> str:
    """Chuẩn hóa text: lowercase, bỏ dấu câu, gộp space."""
    text = unicodedata.normalize("NFC", text)
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ════════════════════════════════════════════════════════════════
# LOAD MODELS
# ════════════════════════════════════════════════════════════════
def load_sparktts(device: str = "cuda"):
    """Load SparkTTS model + processor."""
    from transformers import AutoModel, AutoProcessor

    logger.info(f"Loading SparkTTS from {SPARKTTS_MODEL_DIR}")
    # NOTE: Must use float32, not bfloat16!
    # BiCodec's mel_transformer calls cuFFT which doesn't support BFloat16.
    model = AutoModel.from_pretrained(
        str(SPARKTTS_MODEL_DIR),
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(
        str(SPARKTTS_MODEL_DIR),
        trust_remote_code=True,
    )
    processor.link_model(model)

    logger.success(f"SparkTTS loaded on {next(model.parameters()).device}")
    return model, processor


def load_whisper_pipeline(device: str = "cuda"):
    """Load Whisper ASR pipeline for WER evaluation."""
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    logger.info(f"Loading Whisper from {WHISPER_MODEL_DIR}")
    model_path = str(WHISPER_MODEL_DIR) if WHISPER_MODEL_DIR.exists() else "openai/whisper-large-v3"
    compute_dtype = torch.float16 if device == "cuda" else torch.float32

    whisper_processor = AutoProcessor.from_pretrained(model_path)
    whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_path, dtype=compute_dtype, low_cpu_mem_usage=True,
    )
    whisper_model.to(device)

    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model=whisper_model,
        tokenizer=whisper_processor.tokenizer,
        feature_extractor=whisper_processor.feature_extractor,
        dtype=compute_dtype,
        device=device,
        generate_kwargs={"language": "vi"},
    )
    logger.success("Whisper ASR loaded")
    return asr_pipe


# ════════════════════════════════════════════════════════════════
# INFERENCE
# ════════════════════════════════════════════════════════════════
def synthesize_one(
    model,
    processor,
    text: str,
    prompt_wav_path: Optional[str] = None,
    prompt_text: Optional[str] = None,
    device: str = "cuda",
) -> Optional[np.ndarray]:
    """Synthesize speech for one text sample. Returns audio numpy array or None."""
    try:
        if prompt_wav_path and Path(prompt_wav_path).exists():
            # Voice cloning mode
            inputs = processor(
                text=text,
                prompt_speech_path=str(prompt_wav_path),
                prompt_text=prompt_text,
                return_tensors="pt",
            )
        else:
            # Controllable TTS mode (fallback)
            inputs = processor(
                text=text,
                gender="female",
                pitch="moderate",
                speed="moderate",
                return_tensors="pt",
            )

        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        input_len = input_ids.shape[1]

        global_token_ids = inputs.get("global_token_ids_prompt", None)
        if global_token_ids is not None:
            global_token_ids = global_token_ids.to(device)

        # Use model.llm.generate() instead of model.generate()
        # The SparkTTSModel wrapper's config (SparkTTSConfig) lacks
        # num_hidden_layers needed by HF GenerationMixin.
        with torch.no_grad():
            generated = model.llm.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=3000,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
            )

        result = processor.decode(
            generated,
            global_token_ids_prompt=global_token_ids,
            input_ids_len=input_len,
        )
        return result.get("audio", None)

    except Exception as e:
        logger.warning(f"Synthesis failed: {e}")
        return None


def compute_wer(asr_pipe, audio: np.ndarray, ref_text: str) -> float:
    """Compute WER between synthesized audio and reference text."""
    from jiwer import wer

    try:
        result = asr_pipe({"raw": audio, "sampling_rate": SAMPLE_RATE})
        hyp = normalize_text(result["text"])
        ref = normalize_text(ref_text)
        if not ref:
            return 1.0
        return wer(ref, hyp)
    except Exception as e:
        logger.warning(f"WER computation failed: {e}")
        return 1.0


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="SparkTTS baseline evaluation")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to evaluate")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index in dataset")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--skip_whisper", action="store_true", help="Skip WER computation")
    parser.add_argument("--use_cloning", action="store_true", help="Use voice cloning mode")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    audio_dir = OUTPUT_DIR / "audio"
    audio_dir.mkdir(exist_ok=True)

    # Load data
    df = pd.read_csv(DATA_CSV)
    logger.info(f"Loaded {len(df):,} samples from {DATA_CSV}")

    end_idx = min(args.start_idx + args.num_samples, len(df))
    df_subset = df.iloc[args.start_idx:end_idx].reset_index(drop=True)
    logger.info(f"Processing samples [{args.start_idx}:{end_idx}] ({len(df_subset)} samples)")

    # Load models
    model, processor = load_sparktts(args.device)

    asr_pipe = None
    if not args.skip_whisper:
        asr_pipe = load_whisper_pipeline(args.device)

    # Run inference
    results = []
    for i, row in df_subset.iterrows():
        text = str(row["text"])
        wav_name = row["wav"]
        speaker = row["speaker"]
        ref_wav_path = str(WAVS_DIR / wav_name) if args.use_cloning else None

        t0 = time.time()
        audio = synthesize_one(
            model, processor,
            text=text,
            prompt_wav_path=ref_wav_path,
            device=args.device,
        )
        gen_time = time.time() - t0

        if audio is None:
            logger.warning(f"[{i}] FAILED: {wav_name}")
            results.append({
                "idx": args.start_idx + i,
                "wav": wav_name, "speaker": speaker, "text": text,
                "gen_time": gen_time, "audio_duration": 0,
                "wer_baseline": None, "status": "failed",
            })
            continue

        # Save generated audio
        out_path = audio_dir / f"gen_{args.start_idx + i:05d}.wav"
        sf.write(str(out_path), audio, SAMPLE_RATE)
        audio_duration = len(audio) / SAMPLE_RATE

        # Compute WER
        sample_wer = None
        if asr_pipe is not None:
            sample_wer = compute_wer(asr_pipe, audio, text)

        results.append({
            "idx": args.start_idx + i,
            "wav": wav_name, "speaker": speaker, "text": text,
            "gen_time": round(gen_time, 2),
            "audio_duration": round(audio_duration, 2),
            "rtf": round(gen_time / max(audio_duration, 0.01), 3),
            "wer_baseline": round(sample_wer, 4) if sample_wer is not None else None,
            "gen_audio_path": str(out_path),
            "status": "ok",
        })

        if sample_wer is not None:
            logger.info(
                f"[{i:3d}/{len(df_subset)}] {wav_name} | "
                f"WER={sample_wer:.3f} | dur={audio_duration:.1f}s | "
                f"RTF={gen_time/max(audio_duration,0.01):.2f}"
            )
        else:
            logger.info(
                f"[{i:3d}/{len(df_subset)}] {wav_name} | "
                f"dur={audio_duration:.1f}s | gen={gen_time:.1f}s"
            )

    # Save results
    df_results = pd.DataFrame(results)
    results_path = OUTPUT_DIR / f"baseline_results_{args.start_idx}_{end_idx}.csv"
    df_results.to_csv(results_path, index=False)
    logger.success(f"Results saved to {results_path}")

    # Print summary
    ok_mask = df_results["status"] == "ok"
    if ok_mask.any():
        logger.info("=" * 60)
        logger.info("BASELINE SUMMARY")
        logger.info(f"  Total:    {len(df_results)}")
        logger.info(f"  Success:  {ok_mask.sum()}")
        logger.info(f"  Failed:   {(~ok_mask).sum()}")
        if df_results.loc[ok_mask, "wer_baseline"].notna().any():
            mean_wer = df_results.loc[ok_mask, "wer_baseline"].mean()
            logger.info(f"  Mean WER: {mean_wer:.4f} ({mean_wer*100:.2f}%)")
        logger.info(f"  Mean RTF: {df_results.loc[ok_mask, 'rtf'].mean():.3f}")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
