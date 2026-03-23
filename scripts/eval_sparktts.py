"""Evaluate SparkTTS: inference → WER + UTMOS + SIM comparison vs ground truth.

Pipeline:
  1. SparkTTS voice cloning inference on N samples
  2. Evaluate generated .wav: WER (Whisper), UTMOS (k2-fsa), SIM (WavLM+ECAPA-TDNN)
  3. Compare with ground truth metrics from ground_truth_analysis.csv

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/eval_sparktts.py \
        --num_samples 500 --use_cloning
"""

import argparse
import os
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

# Add ZipVoice eval to path
ZIPVOICE_ROOT = Path("/data1/speech/nhandt23/06_binh/Zipvoice/ZipVoice")
sys.path.insert(0, str(ZIPVOICE_ROOT))

# ════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════
SPARKTTS_MODEL_DIR = Path("/data1/speech/nhandt23/06_binh/spark_based/Vi-SparkTTS-0.5B")
DATA_CSV = Path("/data1/speech/nhandt23/06_binh/chatter_v2v/data/normalized/phoaudiobook/filtered_15k.csv")
WAVS_DIR = Path("/data1/speech/speechData/rawData/Public_Data/phoaudiobook/wavs")
WHISPER_MODEL_DIR = Path("/data1/speech/nhandt23/06_binh/models/openai--whisper-large-v3")
EVAL_MODELS_DIR = Path("/data1/speech/nhandt23/06_binh/gspo_sparktts/download/tts_eval_models")
GT_ANALYSIS_CSV = Path("/data1/speech/nhandt23/06_binh/gspo_sparktts/outputs/analysis/ground_truth_analysis.csv")
OUTPUT_DIR = Path("/data1/speech/nhandt23/06_binh/gspo_sparktts/outputs/sparktts_eval")
SAMPLE_RATE = 16_000


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def load_audio_torch(wav_path: str, sr: int = 16000, device: str = "cuda",
                     max_seconds: float = 120.0) -> torch.Tensor:
    import soxr
    audio, orig_sr = sf.read(wav_path)
    if len(audio.shape) > 1:
        audio = audio[:, 0]
    if orig_sr != sr:
        audio = soxr.resample(audio, orig_sr, sr, quality="VHQ")
    max_samples = int(max_seconds * sr)
    if len(audio) > max_samples:
        audio = audio[:max_samples]
    return torch.from_numpy(audio).float().to(device)


# ════════════════════════════════════════════════════════════════
# MODEL LOADERS
# ════════════════════════════════════════════════════════════════
def load_sparktts(device: str = "cuda"):
    from transformers import AutoModel, AutoProcessor
    logger.info(f"Loading SparkTTS from {SPARKTTS_MODEL_DIR}")
    model = AutoModel.from_pretrained(
        str(SPARKTTS_MODEL_DIR), trust_remote_code=True, torch_dtype=torch.float32,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(str(SPARKTTS_MODEL_DIR), trust_remote_code=True)
    processor.link_model(model)
    logger.success("SparkTTS loaded")
    return model, processor


def load_whisper(device: str = "cuda"):
    from transformers import pipeline
    logger.info(f"Loading Whisper from {WHISPER_MODEL_DIR}")
    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model=str(WHISPER_MODEL_DIR),
        device=device,
        torch_dtype=torch.float16,
    )
    asr_pipe.model.config.forced_decoder_ids = asr_pipe.tokenizer.get_decoder_prompt_ids(
        language="vi", task="transcribe"
    )
    logger.success("Whisper loaded")
    return asr_pipe


def load_utmos(model_dir: Path, device: str = "cuda"):
    from zipvoice.eval.models.utmos import UTMOS22Strong
    ckpt_path = model_dir / "mos" / "utmos22_strong_step7459_v1.pt"
    model = UTMOS22Strong()
    state_dict = torch.load(str(ckpt_path), map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device).eval()
    logger.success("UTMOS loaded")
    return model


def load_sim_model(model_dir: Path, device: str = "cuda"):
    from zipvoice.eval.models.ecapa_tdnn_wavlm import ECAPA_TDNN_WAVLM
    sv_ckpt = model_dir / "speaker_similarity" / "wavlm_large_finetune.pth"
    ssl_dir = model_dir / "speaker_similarity" / "wavlm_large"
    ssl_path_str = str(ssl_dir) + os.sep
    model = ECAPA_TDNN_WAVLM(
        feat_dim=1024, channels=512, emb_dim=256,
        sr=SAMPLE_RATE, ssl_model_path=ssl_path_str,
    )
    state_dict = torch.load(str(sv_ckpt), map_location="cpu")
    model.load_state_dict(state_dict["model"], strict=False)
    model.to(device).eval()
    logger.success("SIM model loaded")
    return model


# ════════════════════════════════════════════════════════════════
# INFERENCE
# ════════════════════════════════════════════════════════════════
def synthesize_one(model, processor, text: str, prompt_wav_path=None, device="cuda"):
    try:
        if prompt_wav_path and Path(prompt_wav_path).exists():
            inputs = processor(text=text, prompt_speech_path=str(prompt_wav_path), return_tensors="pt")
        else:
            inputs = processor(text=text, gender="female", pitch="moderate", speed="moderate", return_tensors="pt")

        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        input_len = input_ids.shape[1]
        global_token_ids = inputs.get("global_token_ids_prompt", None)
        if global_token_ids is not None:
            global_token_ids = global_token_ids.to(device)

        with torch.no_grad():
            generated = model.llm.generate(
                input_ids=input_ids, attention_mask=attention_mask,
                max_new_tokens=3000, do_sample=True, temperature=0.8, top_p=0.95,
            )
        result = processor.decode(generated, global_token_ids_prompt=global_token_ids, input_ids_len=input_len)
        return result.get("audio", None)
    except Exception as e:
        logger.warning(f"Synthesis failed: {e}")
        return None


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="SparkTTS eval: inference + WER/UTMOS/SIM")
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_cloning", action="store_true")
    parser.add_argument("--eval_models_dir", type=str, default=str(EVAL_MODELS_DIR))
    parser.add_argument("--checkpoint_every", type=int, default=50)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    audio_dir = OUTPUT_DIR / "generated_wavs"
    audio_dir.mkdir(exist_ok=True)
    eval_models_dir = Path(args.eval_models_dir)

    # ═══════════════════════════════════════════════════
    # 1. Load data
    # ═══════════════════════════════════════════════════
    df = pd.read_csv(DATA_CSV)
    end_idx = min(args.start_idx + args.num_samples, len(df))
    df_subset = df.iloc[args.start_idx:end_idx].reset_index(drop=True)
    logger.info(f"Evaluating {len(df_subset):,} samples [{args.start_idx}:{end_idx}]")

    # ═══════════════════════════════════════════════════
    # 2. SparkTTS Inference
    # ═══════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("STAGE 1: SparkTTS INFERENCE")
    logger.info("=" * 60)

    model, processor = load_sparktts(args.device)

    gen_results = []
    for i, row in tqdm(df_subset.iterrows(), total=len(df_subset), desc="TTS Inference"):
        text = str(row["text"])
        wav_name = row["wav"]
        ref_wav = str(WAVS_DIR / wav_name) if args.use_cloning else None

        t0 = time.time()
        audio = synthesize_one(model, processor, text=text, prompt_wav_path=ref_wav, device=args.device)
        gen_time = time.time() - t0

        out_path = audio_dir / f"gen_{args.start_idx + i:05d}.wav"
        if audio is not None and len(audio) > 0:
            sf.write(str(out_path), audio, SAMPLE_RATE)
            gen_results.append({
                "idx": i, "wav": wav_name, "speaker": row["speaker"], "text": text,
                "gen_wav": str(out_path), "ref_wav": str(WAVS_DIR / wav_name),
                "gen_time": round(gen_time, 2),
                "audio_duration": round(len(audio) / SAMPLE_RATE, 2),
                "status": "ok",
            })
        else:
            gen_results.append({
                "idx": i, "wav": wav_name, "speaker": row["speaker"], "text": text,
                "gen_wav": "", "ref_wav": str(WAVS_DIR / wav_name),
                "gen_time": round(gen_time, 2), "audio_duration": 0,
                "status": "failed",
            })

        if (i + 1) % args.checkpoint_every == 0:
            logger.info(f"Inference checkpoint: {i+1}/{len(df_subset)}")

    # Free SparkTTS
    del model, processor
    torch.cuda.empty_cache()

    df_gen = pd.DataFrame(gen_results)
    ok_mask = df_gen["status"] == "ok"
    logger.info(f"Inference done: {ok_mask.sum()} OK, {(~ok_mask).sum()} failed")

    # ═══════════════════════════════════════════════════
    # 3. WER Evaluation (Whisper)
    # ═══════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 2: WER EVALUATION (Whisper)")
    logger.info("=" * 60)

    asr_pipe = load_whisper(args.device)
    from jiwer import wer as compute_wer

    wer_scores = []
    for i, row in tqdm(df_gen[ok_mask].iterrows(), total=ok_mask.sum(), desc="WER"):
        try:
            audio, sr = sf.read(row["gen_wav"])
            result = asr_pipe({"raw": audio.astype(np.float32), "sampling_rate": sr})
            hyp = normalize_text(result["text"])
            ref = normalize_text(row["text"])
            w = compute_wer(ref, hyp) if ref else 1.0
        except Exception as e:
            logger.warning(f"WER failed [{i}]: {e}")
            w = float("nan")
        wer_scores.append((i, w))

    for idx, w in wer_scores:
        df_gen.loc[idx, "wer"] = w

    del asr_pipe
    torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════
    # 4. UTMOS Evaluation
    # ═══════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 3: UTMOS EVALUATION")
    logger.info("=" * 60)

    utmos_model = load_utmos(eval_models_dir, args.device)

    utmos_scores = []
    for i, row in tqdm(df_gen[ok_mask].iterrows(), total=ok_mask.sum(), desc="UTMOS"):
        try:
            speech = load_audio_torch(row["gen_wav"], sr=SAMPLE_RATE, device=args.device)
            with torch.no_grad():
                score = utmos_model(speech.unsqueeze(0), SAMPLE_RATE)
            utmos_scores.append((i, score.item()))
        except Exception as e:
            logger.warning(f"UTMOS failed [{i}]: {e}")
            utmos_scores.append((i, float("nan")))

    for idx, s in utmos_scores:
        df_gen.loc[idx, "utmos"] = s

    del utmos_model
    torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════
    # 5. SIM Evaluation (cosine similarity: ref ↔ gen)
    # ═══════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 4: SIM EVALUATION (Speaker Similarity)")
    logger.info("=" * 60)

    sim_model = load_sim_model(eval_models_dir, args.device)

    sim_scores = []
    for i, row in tqdm(df_gen[ok_mask].iterrows(), total=ok_mask.sum(), desc="SIM"):
        try:
            ref_speech = load_audio_torch(row["ref_wav"], sr=SAMPLE_RATE, device=args.device)
            gen_speech = load_audio_torch(row["gen_wav"], sr=SAMPLE_RATE, device=args.device)
            with torch.no_grad():
                ref_emb = sim_model([ref_speech])
                gen_emb = sim_model([gen_speech])
            sim = F.cosine_similarity(ref_emb, gen_emb, dim=-1).item()
            sim_scores.append((i, sim))
        except Exception as e:
            logger.warning(f"SIM failed [{i}]: {e}")
            sim_scores.append((i, float("nan")))

    for idx, s in sim_scores:
        df_gen.loc[idx, "sim"] = s

    del sim_model
    torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════
    # 6. Save results + comparison
    # ═══════════════════════════════════════════════════
    output_csv = OUTPUT_DIR / "sparktts_eval_results.csv"
    df_gen.to_csv(output_csv, index=False)
    logger.success(f"Results saved to {output_csv}")

    # ── Summary ──
    ok_df = df_gen[ok_mask]
    gen_wer = ok_df["wer"].dropna()
    gen_utmos = ok_df["utmos"].dropna()
    gen_sim = ok_df["sim"].dropna()

    logger.info("\n" + "=" * 60)
    logger.info("SPARKTTS EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"  Samples:  {len(ok_df):,} / {len(df_gen):,}")
    if len(gen_wer):
        logger.info(f"  WER  mean:   {gen_wer.mean()*100:.2f}%  median: {gen_wer.median()*100:.2f}%")
    if len(gen_utmos):
        logger.info(f"  UTMOS mean:  {gen_utmos.mean():.4f}  median: {gen_utmos.median():.4f}")
    if len(gen_sim):
        logger.info(f"  SIM  mean:   {gen_sim.mean():.4f}  median: {gen_sim.median():.4f}")

    # ── Compare with ground truth ──
    if GT_ANALYSIS_CSV.exists():
        gt_df = pd.read_csv(GT_ANALYSIS_CSV)
        # Match by wav name
        matched = ok_df.merge(gt_df[["wav", "wer", "utmos"]], on="wav", suffixes=("_gen", "_gt"))

        logger.info("\n" + "=" * 60)
        logger.info("COMPARISON: SparkTTS vs Ground Truth")
        logger.info("=" * 60)
        logger.info(f"  Matched samples: {len(matched):,}")

        if "wer_gt" in matched and "wer_gen" in matched:
            gt_wer = matched["wer_gt"].mean()
            sp_wer = matched["wer_gen"].mean()
            logger.info(f"  WER   GT: {gt_wer*100:.2f}%  →  SparkTTS: {sp_wer*100:.2f}%  (Δ{(sp_wer-gt_wer)*100:+.2f}%)")

        if "utmos_gt" in matched and "utmos_gen" in matched:
            gt_utmos = matched["utmos_gt"].mean()
            sp_utmos = matched["utmos_gen"].mean()
            logger.info(f"  UTMOS GT: {gt_utmos:.4f}  →  SparkTTS: {sp_utmos:.4f}  (Δ{sp_utmos-gt_utmos:+.4f})")

        if len(gen_sim):
            # GT SIM is intra-speaker, generated SIM is ref↔gen (different metrics but comparable)
            logger.info(f"  SIM   SparkTTS (ref↔gen): {gen_sim.mean():.4f}")

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
