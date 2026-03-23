"""Eval GSPO checkpoint vs Base vs SFT.

Loads the LLM from a GSPO LoRA checkpoint, patches into SparkTTS,
then runs inference + WER/UTMOS/SIM eval on N val samples.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/eval_gspo_ckpt.py \
        --checkpoint outputs/gspo_run/checkpoint-1000 \
        --num_samples 100 \
        --output_dir outputs/eval_gspo_1000
"""

import argparse
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
from loguru import logger
from tqdm import tqdm

ZIPVOICE_ROOT = Path("/data1/speech/nhandt23/06_binh/Zipvoice/ZipVoice-flowgrpo")
PROJECT_ROOT  = Path("/data1/speech/nhandt23/06_binh/gspo_sparktts")
sys.path.insert(0, str(ZIPVOICE_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

SPARKTTS_MODEL_DIR = Path("/data1/speech/nhandt23/06_binh/spark_based/Vi-SparkTTS-0.5B")
LLM_PATH           = SPARKTTS_MODEL_DIR / "LLM"
TEST_CSV           = PROJECT_ROOT / "data/splits/test_2k.csv"
EVAL_MODELS_DIR    = PROJECT_ROOT / "download/tts_eval_models"
WHISPER_DIR        = Path("/data1/speech/nhandt23/06_binh/models/openai--whisper-large-v3")
SAMPLE_RATE        = 16_000


# ════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════

def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_content_text(prompt: str) -> str:
    m = re.search(r"<\|start_content\|>(.*?)<\|end_content\|>", prompt, re.DOTALL)
    return m.group(1).strip() if m else prompt.strip()


def load_test_data(csv_file: Path, n: int) -> list[dict]:
    import pandas as pd
    df = pd.read_csv(csv_file)
    rows = []
    for _, r in df.head(n).iterrows():
        rows.append({
            "prompt": "",
            "ref_text": r["text"],
            "wav": r["wav"],
        })
    logger.info(f"Loaded {len(rows)} test samples from {csv_file}")
    return rows


# ════════════════════════════════════════════════════════
# MODEL LOADING
# ════════════════════════════════════════════════════════

def load_sparktts_with_gspo(checkpoint_path: str | None, device: str):
    """Load SparkTTS + optionally patch LLM with GSPO LoRA checkpoint."""
    from transformers import AutoModel, AutoModelForCausalLM, AutoProcessor
    from peft import PeftModel

    logger.info(f"Loading SparkTTS base from {SPARKTTS_MODEL_DIR}")
    sparktts = AutoModel.from_pretrained(
        str(SPARKTTS_MODEL_DIR), trust_remote_code=True, torch_dtype=torch.float32,
    ).eval().to(device)

    processor = AutoProcessor.from_pretrained(str(SPARKTTS_MODEL_DIR), trust_remote_code=True)
    processor.link_model(sparktts)

    if checkpoint_path and Path(checkpoint_path).exists():
        logger.info(f"Patching LLM with GSPO checkpoint: {checkpoint_path}")
        base_llm = AutoModelForCausalLM.from_pretrained(str(LLM_PATH), torch_dtype=torch.float32)
        peft_llm = PeftModel.from_pretrained(base_llm, checkpoint_path)
        merged   = peft_llm.merge_and_unload()
        # Replace the LLM inside SparkTTS
        sparktts.llm = merged.to(device)
        logger.success("GSPO checkpoint merged into SparkTTS.llm")
    else:
        logger.info("Using base LLM (no checkpoint)")

    return sparktts, processor


# ════════════════════════════════════════════════════════
# INFERENCE
# ════════════════════════════════════════════════════════

def synthesize_one(sparktts, processor, text: str, ref_wav_path: str, device: str) -> np.ndarray | None:
    try:
        if ref_wav_path and Path(ref_wav_path).exists():
            inputs = processor(text=text, prompt_speech_path=ref_wav_path, return_tensors="pt")
        else:
            inputs = processor(text=text, gender="female", pitch="moderate", speed="moderate", return_tensors="pt")

        input_ids   = inputs["input_ids"].to(device)
        attn_mask   = inputs["attention_mask"].to(device)
        input_len   = input_ids.shape[1]
        global_ids  = inputs.get("global_token_ids_prompt", None)
        if global_ids is not None:
            global_ids = global_ids.to(device)

        with torch.no_grad():
            generated = sparktts.llm.generate(
                input_ids=input_ids, attention_mask=attn_mask,
                max_new_tokens=2048, do_sample=True, temperature=0.8, top_p=0.95,
            )
        result = processor.decode(generated, global_token_ids_prompt=global_ids, input_ids_len=input_len)
        return result.get("audio", None)
    except Exception as e:
        logger.warning(f"Synthesis failed: {e}")
        return None


# ════════════════════════════════════════════════════════
# EVAL METRICS
# ════════════════════════════════════════════════════════

def eval_batch(df: pd.DataFrame, asr_pipe, utmos_model, sim_model, device: str) -> pd.DataFrame:
    from jiwer import wer as compute_wer

    df = df.copy()
    wers, utmos_scores, sims = [], [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        gen_wav = row.get("gen_wav", "")
        if not gen_wav or not Path(gen_wav).exists():
            wers.append(float("nan"))
            utmos_scores.append(float("nan"))
            sims.append(float("nan"))
            continue

        audio, sr = sf.read(gen_wav)
        audio = audio.astype(np.float32)

        # WER
        try:
            hyp = normalize_text(asr_pipe({"raw": audio, "sampling_rate": sr})["text"])
            ref = normalize_text(row["ref_text"])
            wers.append(compute_wer(ref, hyp) if ref else 1.0)
        except Exception:
            wers.append(float("nan"))

        # UTMOS
        try:
            wav_t = torch.from_numpy(audio).unsqueeze(0).to(device)
            with torch.no_grad():
                score = utmos_model(wav_t, SAMPLE_RATE)
            utmos_scores.append(score.item())
        except Exception:
            utmos_scores.append(float("nan"))

        # SIM
        try:
            ref_wav_path = row.get("wav", "")
            if ref_wav_path and not str(ref_wav_path).startswith("/"):
                # Handle relative paths from JSONL dataset
                import os
                wav_root = "/data1/speech/speechData/rawData/Public_Data/phoaudiobook/wavs"
                ref_wav_path = os.path.join(wav_root, ref_wav_path)

            if ref_wav_path and Path(ref_wav_path).exists():
                ref_audio_raw, ref_sr = sf.read(ref_wav_path)
                # Ensure mono 1D [T] — same as reward_function.py
                if ref_audio_raw.ndim > 1:
                    ref_audio_raw = ref_audio_raw.mean(axis=1)
                ref_t = torch.from_numpy(ref_audio_raw.astype(np.float32)).to(device)  # [T]
                gen_a = audio if audio.ndim == 1 else audio.mean(axis=1)
                gen_t = torch.from_numpy(gen_a).to(device)  # [T]
                with torch.no_grad():
                    ref_emb = sim_model([ref_t])
                    gen_emb = sim_model([gen_t])
                sims.append(F.cosine_similarity(ref_emb, gen_emb, dim=-1).item())
            else:
                sims.append(float("nan"))
        except Exception as e:
            logger.debug(f"SIM eval error: {e}")
            sims.append(float("nan"))

    df["wer"]   = wers
    df["utmos"] = utmos_scores
    df["sim"]   = sims
    return df


# ════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Eval GSPO checkpoint quality vs base")
    parser.add_argument("--checkpoint",   type=str, default=None,
                        help="Path to GSPO LoRA checkpoint dir, e.g. outputs/gspo_run/checkpoint-1000")
    parser.add_argument("--num_samples",  type=int, default=100)
    parser.add_argument("--output_dir",   type=str, default="outputs/eval_gspo_ckpt")
    parser.add_argument("--device",       type=str, default="cuda")
    parser.add_argument("--skip_base",    action="store_true",
                        help="Skip base model eval (faster if only need GSPO results)")
    parser.add_argument("--chunk_idx",    type=int, default=0,
                        help="Chunk index for multi-GPU processing")
    parser.add_argument("--num_chunks",   type=int, default=1,
                        help="Total number of chunks for multi-GPU processing")
    args = parser.parse_args()

    import math
    from transformers import pipeline as hf_pipeline
    from zipvoice_based.eval.models.utmos import UTMOS22Strong
    from zipvoice_based.eval.models.ecapa_tdnn_wavlm import ECAPA_TDNN_WAVLM

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_dir = output_dir / "wavs"
    wav_dir.mkdir(exist_ok=True)

    samples = load_test_data(TEST_CSV, args.num_samples)

    chunk_size = math.ceil(len(samples) / args.num_chunks)
    start_idx = args.chunk_idx * chunk_size
    end_idx = min(start_idx + chunk_size, len(samples))
    samples = samples[start_idx:end_idx]
    logger.info(f"Processing chunk {args.chunk_idx+1}/{args.num_chunks} (indices {start_idx} to {end_idx-1}, {len(samples)} samples)")

    # ── Load eval models ──
    logger.info("Loading eval models (Whisper, UTMOS, SIM)...")
    asr_pipe = hf_pipeline(
        "automatic-speech-recognition", model=str(WHISPER_DIR),
        device=args.device, torch_dtype=torch.float16,
    )
    asr_pipe.model.config.forced_decoder_ids = asr_pipe.tokenizer.get_decoder_prompt_ids(
        language="vi", task="transcribe"
    )

    utmos_model = UTMOS22Strong()
    utmos_model.load_state_dict(torch.load(
        str(EVAL_MODELS_DIR / "mos/utmos22_strong_step7459_v1.pt"), map_location="cpu"
    ))
    utmos_model = utmos_model.to(args.device).eval()

    sim_model = ECAPA_TDNN_WAVLM(
        feat_dim=1024, channels=512, emb_dim=256, sr=SAMPLE_RATE,
        ssl_model_path=str(EVAL_MODELS_DIR / "speaker_similarity/wavlm_large") + "/",
    )
    state = torch.load(str(EVAL_MODELS_DIR / "speaker_similarity/wavlm_large_finetune.pth"), map_location="cpu")
    sim_model.load_state_dict(state["model"], strict=False)
    sim_model = sim_model.to(args.device).eval()
    logger.success("Eval models ready.")

    results = {}

    ckpt_step = Path(args.checkpoint).name.split("-")[-1] if args.checkpoint and "-" in Path(args.checkpoint).name else "1000"
    for label, ckpt in [("base", None), (f"gspo_{ckpt_step}", args.checkpoint)]:
        if label == "base" and args.skip_base:
            logger.info("Skipping base eval (--skip_base)")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"  Evaluating: {label}")
        logger.info(f"{'='*60}")

        sparktts, processor = load_sparktts_with_gspo(ckpt, args.device)

        rows = []
        wav_subdir = wav_dir / label
        wav_subdir.mkdir(exist_ok=True)

        for i, sample in enumerate(tqdm(samples, desc=f"TTS [{label}]")):
            t0 = time.time()
            # Fix path for reference audio so voice cloning works
            ref_wav_name = sample["wav"]
            if ref_wav_name and not str(ref_wav_name).startswith("/"):
                import os
                wav_root = "/data1/speech/speechData/rawData/Public_Data/phoaudiobook/wavs"
                ref_wav_path = os.path.join(wav_root, ref_wav_name)
            else:
                ref_wav_path = ref_wav_name

            audio = synthesize_one(sparktts, processor, sample["ref_text"], ref_wav_path, args.device)
            elapsed = time.time() - t0

            out_path = wav_subdir / f"chunk{args.chunk_idx}_{i:04d}.wav"
            if audio is not None and len(audio) > 0:
                sf.write(str(out_path), audio, SAMPLE_RATE)
                rows.append({
                    "ref_text": sample["ref_text"],
                    "wav": sample["wav"],
                    "gen_wav": str(out_path),
                    "rtf": round(elapsed / (len(audio) / SAMPLE_RATE), 3),
                })
            else:
                rows.append({"ref_text": sample["ref_text"], "wav": sample["wav"], "gen_wav": "", "rtf": 0})

        del sparktts, processor
        torch.cuda.empty_cache()

        df = pd.DataFrame(rows)
        df = eval_batch(df, asr_pipe, utmos_model, sim_model, args.device)
        csv_name = f"eval_{label}_chunk{args.chunk_idx}.csv" if args.num_chunks > 1 else f"eval_{label}.csv"
        df.to_csv(output_dir / csv_name, index=False)
        results[label] = df

    # ── Summary table ──
    logger.info("\n" + "="*60)
    logger.info("RESULTS SUMMARY")
    logger.info("="*60)
    logger.info(f"{'Model':<15} {'WER%':>8} {'UTMOS':>8} {'SIM':>8} {'RTF':>8}")
    logger.info("-"*50)
    for label, df in results.items():
        ok = df["gen_wav"].ne("")
        wer_m   = df.loc[ok, "wer"].mean() * 100
        utmos_m = df.loc[ok, "utmos"].mean()
        sim_m   = df.loc[ok, "sim"].mean()
        rtf_m   = df.loc[ok, "rtf"].mean()
        logger.info(f"{label:<15} {wer_m:>7.2f}% {utmos_m:>8.4f} {sim_m:>8.4f} {rtf_m:>8.3f}")
    logger.info("="*60)
    logger.success(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
