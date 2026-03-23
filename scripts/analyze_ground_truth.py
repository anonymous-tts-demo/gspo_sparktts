"""Analyze PhoAudiobook ground truth audio: WER + UTMOS + SIM.

Uses k2-fsa/TTS_eval_models for:
  - UTMOS22Strong: audio quality MOS prediction
  - WavLM + ECAPA-TDNN: speaker embedding for SIM (speaker similarity)
  - WER: already in filtered_15k.csv (Whisper Large V3)

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/analyze_ground_truth.py \
        --num_samples 0  # 0 = all samples
"""

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from loguru import logger
from tqdm import tqdm

# Add ZipVoice eval to path for models
ZIPVOICE_ROOT = Path("/data1/speech/nhandt23/06_binh/Zipvoice/ZipVoice")
sys.path.insert(0, str(ZIPVOICE_ROOT))

# ════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════
DATA_CSV = Path("/data1/speech/nhandt23/06_binh/chatter_v2v/data/normalized/phoaudiobook/filtered_15k.csv")
WAVS_DIR = Path("/data1/speech/speechData/rawData/Public_Data/phoaudiobook/wavs")
EVAL_MODELS_DIR = Path("/data1/speech/nhandt23/06_binh/gspo_sparktts/download/tts_eval_models")
OUTPUT_DIR = Path("/data1/speech/nhandt23/06_binh/gspo_sparktts/outputs/analysis")
SAMPLE_RATE = 16_000


# ════════════════════════════════════════════════════════════════
# MODEL LOADERS
# ════════════════════════════════════════════════════════════════
def load_utmos(model_dir: Path, device: str = "cuda"):
    """Load UTMOS22Strong model from k2-fsa checkpoint."""
    from zipvoice.eval.models.utmos import UTMOS22Strong

    ckpt_path = model_dir / "mos" / "utmos22_strong_step7459_v1.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"UTMOS checkpoint not found at {ckpt_path}")

    model = UTMOS22Strong()
    state_dict = torch.load(str(ckpt_path), map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    logger.success(f"UTMOS loaded from {ckpt_path}")
    return model


def load_sim_model(model_dir: Path, device: str = "cuda"):
    """Load WavLM + ECAPA-TDNN speaker verification model."""
    from zipvoice.eval.models.ecapa_tdnn_wavlm import ECAPA_TDNN_WAVLM

    sv_ckpt = model_dir / "speaker_similarity" / "wavlm_large_finetune.pth"
    ssl_dir = model_dir / "speaker_similarity" / "wavlm_large"
    if not sv_ckpt.exists() or not ssl_dir.exists():
        raise FileNotFoundError(f"SIM model not found at {sv_ckpt} or {ssl_dir}")

    # ECAPA_TDNN_WAVLM.__init__ does:
    #   torch.hub.load(os.path.dirname(ssl_model_path), "wavlm_local",
    #                  source="local", ckpt=os.path.join(ssl_model_path, "wavlm_large.pt"))
    # So ssl_model_path must point to the wavlm_large/ dir, and dirname must yield
    # the dir containing hubconf.py. Since hubconf.py IS in wavlm_large/, we need
    # dirname(ssl_model_path) == wavlm_large/. Append trailing os.sep to ensure this.
    ssl_path_str = str(ssl_dir) + os.sep  # ensures dirname() returns wavlm_large/

    model = ECAPA_TDNN_WAVLM(
        feat_dim=1024, channels=512, emb_dim=256,
        sr=SAMPLE_RATE, ssl_model_path=ssl_path_str,
    )
    state_dict = torch.load(str(sv_ckpt), map_location="cpu")
    model.load_state_dict(state_dict["model"], strict=False)
    model.to(device)
    model.eval()
    logger.success(f"SIM model (WavLM+ECAPA-TDNN) loaded from {sv_ckpt}")
    return model


# ════════════════════════════════════════════════════════════════
# AUDIO + SCORING
# ════════════════════════════════════════════════════════════════
def load_audio_torch(wav_path: str, sr: int = 16000, device: str = "cuda",
                     max_seconds: float = 120.0) -> torch.Tensor:
    """Load audio file and return as torch tensor on device."""
    import soundfile as sf
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


@torch.no_grad()
def compute_utmos_scores(model, wav_paths: list[str], device: str = "cuda",
                         batch_desc: str = "UTMOS") -> list[float]:
    """Compute UTMOS scores for a list of audio files."""
    scores = []
    for wav_path in tqdm(wav_paths, desc=batch_desc):
        try:
            speech = load_audio_torch(wav_path, sr=SAMPLE_RATE, device=device)
            score = model(speech.unsqueeze(0), SAMPLE_RATE)
            scores.append(score.item())
        except Exception as e:
            logger.warning(f"UTMOS failed for {wav_path}: {e}")
            scores.append(float("nan"))
    return scores


@torch.no_grad()
def compute_speaker_embeddings(sim_model, wav_paths: list[str], device: str = "cuda",
                                batch_desc: str = "SIM Embeddings") -> list[torch.Tensor]:
    """Extract speaker embeddings for a list of audio files."""
    embeddings = []
    for wav_path in tqdm(wav_paths, desc=batch_desc):
        try:
            speech = load_audio_torch(wav_path, sr=SAMPLE_RATE, device=device, max_seconds=120)
            emb = sim_model([speech])  # returns (1, emb_dim)
            embeddings.append(emb.cpu())
        except Exception as e:
            logger.warning(f"SIM embedding failed for {wav_path}: {e}")
            embeddings.append(None)
    return embeddings


def compute_intra_speaker_sim(df: pd.DataFrame, embeddings: list,
                               valid_indices: list, max_pairs_per_speaker: int = 50) -> dict:
    """Compute intra-speaker cosine similarity.
    
    For each speaker, sample pairs of audio and compute cosine similarity
    between their embeddings. Returns per-sample SIM and overall stats.
    """
    # Group by speaker
    speaker_embs = defaultdict(list)
    idx_to_emb = {}
    for i, (idx, emb) in enumerate(zip(valid_indices, embeddings)):
        if emb is not None:
            speaker = df.loc[idx, "speaker"]
            speaker_embs[speaker].append(emb)
            idx_to_emb[idx] = emb

    # Compute per-speaker mean embedding and intra-speaker SIM
    speaker_sims = {}
    for speaker, embs in speaker_embs.items():
        if len(embs) < 2:
            speaker_sims[speaker] = 1.0  # Single sample = perfect self-similarity
            continue

        # Stack all embeddings for this speaker
        all_embs = torch.cat(embs, dim=0)  # (N, emb_dim)

        # Compute mean pairwise cosine similarity (sample if too many)
        n = len(all_embs)
        if n > max_pairs_per_speaker:
            # Random sample pairs
            indices = torch.randint(0, n, (max_pairs_per_speaker, 2))
            sims = []
            for a, b in indices:
                if a != b:
                    sim = F.cosine_similarity(all_embs[a].unsqueeze(0), all_embs[b].unsqueeze(0))
                    sims.append(sim.item())
            speaker_sims[speaker] = np.mean(sims) if sims else 1.0
        else:
            # All pairs
            sims = []
            for i in range(n):
                for j in range(i + 1, n):
                    sim = F.cosine_similarity(all_embs[i].unsqueeze(0), all_embs[j].unsqueeze(0))
                    sims.append(sim.item())
            speaker_sims[speaker] = np.mean(sims) if sims else 1.0

    return speaker_sims


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Analyze PhoAudiobook ground truth")
    parser.add_argument("--num_samples", type=int, default=0, help="0 = all")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--eval_models_dir", type=str, default=str(EVAL_MODELS_DIR))
    parser.add_argument("--checkpoint_every", type=int, default=1000)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    eval_models_dir = Path(args.eval_models_dir)

    # ══════════════════════════════════════════════════
    # 1. Load CSV + build wav paths
    # ══════════════════════════════════════════════════
    df = pd.read_csv(DATA_CSV)
    if args.num_samples > 0:
        df = df.head(args.num_samples)
    logger.info(f"Loaded {len(df):,} samples from {DATA_CSV}")

    wav_paths = []
    valid_indices = []
    for i, row in df.iterrows():
        wav_path = WAVS_DIR / row["wav"]
        if wav_path.exists():
            wav_paths.append(str(wav_path))
            valid_indices.append(i)
        else:
            logger.warning(f"[{i}] WAV not found: {wav_path}")
    logger.info(f"Found {len(wav_paths):,} valid audio files")

    # ══════════════════════════════════════════════════
    # 2. WER analysis (already in CSV)
    # ══════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("WER ANALYSIS (from filtered_15k.csv)")
    logger.info("=" * 60)
    wer_col = df["wer"].astype(float)
    logger.info(f"  Count:   {wer_col.count():,}")
    logger.info(f"  Mean:    {wer_col.mean():.4f} ({wer_col.mean()*100:.2f}%)")
    logger.info(f"  Median:  {wer_col.median():.4f} ({wer_col.median()*100:.2f}%)")
    logger.info(f"  Std:     {wer_col.std():.4f}")
    logger.info(f"  Min:     {wer_col.min():.4f}")
    logger.info(f"  Max:     {wer_col.max():.4f}")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        val = wer_col.quantile(p / 100)
        logger.info(f"  P{p:02d}:     {val:.4f} ({val*100:.2f}%)")

    # WER distribution buckets
    bins = [0, 0.001, 0.01, 0.03, 0.05, 0.10, 0.20, 1.0]
    labels = ["=0%", "<1%", "1-3%", "3-5%", "5-10%", "10-20%", ">20%"]
    df["wer_bucket"] = pd.cut(wer_col, bins=bins, labels=labels, include_lowest=True)
    logger.info("\n  WER Distribution:")
    for label in labels:
        count = (df["wer_bucket"] == label).sum()
        pct = count / len(df) * 100
        logger.info(f"    {label:>6s}: {count:>5,} ({pct:5.1f}%)")

    # ══════════════════════════════════════════════════
    # 3. UTMOS analysis
    # ══════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("UTMOS ANALYSIS")
    logger.info("=" * 60)

    utmos_model = load_utmos(eval_models_dir, device=args.device)

    all_utmos = []
    chunk_size = args.checkpoint_every
    for start in range(0, len(wav_paths), chunk_size):
        end = min(start + chunk_size, len(wav_paths))
        chunk_scores = compute_utmos_scores(
            utmos_model, wav_paths[start:end], device=args.device,
            batch_desc=f"UTMOS [{start+1}-{end}/{len(wav_paths)}]"
        )
        all_utmos.extend(chunk_scores)
        logger.info(f"UTMOS checkpoint: {len(all_utmos):,}/{len(wav_paths):,}")

    # Free UTMOS model memory before loading SIM
    del utmos_model
    torch.cuda.empty_cache()

    for idx, score in zip(valid_indices, all_utmos):
        df.loc[idx, "utmos"] = score

    utmos_col = df["utmos"].dropna()
    logger.info(f"\n  UTMOS Statistics:")
    logger.info(f"  Count:   {utmos_col.count():,}")
    logger.info(f"  Mean:    {utmos_col.mean():.4f}")
    logger.info(f"  Median:  {utmos_col.median():.4f}")
    logger.info(f"  Std:     {utmos_col.std():.4f}")
    logger.info(f"  Min:     {utmos_col.min():.4f}")
    logger.info(f"  Max:     {utmos_col.max():.4f}")
    for p in [10, 25, 50, 75, 90, 95]:
        val = utmos_col.quantile(p / 100)
        logger.info(f"  P{p:02d}:     {val:.4f}")

    mos_bins = [0, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    mos_labels = ["<2.0", "2.0-2.5", "2.5-3.0", "3.0-3.5", "3.5-4.0", "4.0-4.5", "4.5-5.0"]
    df["utmos_bucket"] = pd.cut(utmos_col, bins=mos_bins, labels=mos_labels, include_lowest=True)
    logger.info("\n  UTMOS Distribution:")
    for label in mos_labels:
        count = (df["utmos_bucket"] == label).sum()
        pct = count / len(df) * 100
        logger.info(f"    {label:>8s}: {count:>5,} ({pct:5.1f}%)")

    # ══════════════════════════════════════════════════
    # 4. SIM analysis (intra-speaker similarity)
    # ══════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("SIM ANALYSIS (Intra-Speaker Similarity)")
    logger.info("=" * 60)

    sim_model = load_sim_model(eval_models_dir, device=args.device)

    # Extract embeddings in chunks
    all_embeddings = []
    for start in range(0, len(wav_paths), chunk_size):
        end = min(start + chunk_size, len(wav_paths))
        chunk_embs = compute_speaker_embeddings(
            sim_model, wav_paths[start:end], device=args.device,
            batch_desc=f"SIM Emb [{start+1}-{end}/{len(wav_paths)}]"
        )
        all_embeddings.extend(chunk_embs)
        logger.info(f"SIM embedding checkpoint: {len(all_embeddings):,}/{len(wav_paths):,}")

    del sim_model
    torch.cuda.empty_cache()

    # Compute intra-speaker SIM
    logger.info("Computing intra-speaker cosine similarity...")
    speaker_sims = compute_intra_speaker_sim(df, all_embeddings, valid_indices)

    # Add per-speaker SIM to dataframe
    for idx in valid_indices:
        speaker = df.loc[idx, "speaker"]
        df.loc[idx, "speaker_sim"] = speaker_sims.get(speaker, float("nan"))

    sim_values = list(speaker_sims.values())
    logger.info(f"\n  Intra-Speaker SIM Statistics ({len(speaker_sims)} speakers):")
    logger.info(f"  Mean:    {np.mean(sim_values):.4f}")
    logger.info(f"  Median:  {np.median(sim_values):.4f}")
    logger.info(f"  Std:     {np.std(sim_values):.4f}")
    logger.info(f"  Min:     {np.min(sim_values):.4f}")
    logger.info(f"  Max:     {np.max(sim_values):.4f}")
    for p in [10, 25, 50, 75, 90]:
        val = np.percentile(sim_values, p)
        logger.info(f"  P{p:02d}:     {val:.4f}")

    sim_bins = [0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]
    sim_labels = ["<0.5", "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.85", "0.85-0.9", "0.9-0.95", "0.95-1.0"]
    sim_dist = pd.cut(sim_values, bins=sim_bins, labels=sim_labels, include_lowest=True)
    logger.info("\n  SIM Distribution (per speaker):")
    for label in sim_labels:
        count = (sim_dist == label).sum()
        pct = count / len(sim_values) * 100
        logger.info(f"    {label:>10s}: {count:>4} ({pct:5.1f}%)")

    # ══════════════════════════════════════════════════
    # 5. Per-speaker summary
    # ══════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("PER-SPEAKER SUMMARY")
    logger.info("=" * 60)

    # Build speaker stats including SIM
    speaker_stats = df.groupby("speaker").agg(
        count=("wav", "count"),
        wer_mean=("wer", "mean"),
        wer_median=("wer", "median"),
        utmos_mean=("utmos", "mean"),
        utmos_median=("utmos", "median"),
        sim=("speaker_sim", "first"),
    ).sort_values("count", ascending=False)
    logger.info(f"\n{speaker_stats.head(20).to_string()}")

    # ══════════════════════════════════════════════════
    # 6. Save results
    # ══════════════════════════════════════════════════
    output_csv = OUTPUT_DIR / "ground_truth_analysis.csv"
    df.to_csv(output_csv, index=False)
    logger.success(f"Full results saved to {output_csv}")

    speaker_csv = OUTPUT_DIR / "speaker_stats.csv"
    speaker_stats.to_csv(speaker_csv)
    logger.success(f"Speaker stats saved to {speaker_csv}")

    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Samples:        {len(df):,}")
    logger.info(f"  Speakers:       {df['speaker'].nunique()}")
    logger.info(f"  WER mean:       {wer_col.mean()*100:.2f}%")
    logger.info(f"  WER median:     {wer_col.median()*100:.2f}%")
    logger.info(f"  UTMOS mean:     {utmos_col.mean():.4f}")
    logger.info(f"  UTMOS median:   {utmos_col.median():.4f}")
    logger.info(f"  SIM mean:       {np.mean(sim_values):.4f}")
    logger.info(f"  SIM median:     {np.median(sim_values):.4f}")


if __name__ == "__main__":
    main()
