"""Mixed Reward Function for Vi-SparkTTS GSPO Training.

Components:
  - SIM   (30%): Voice similarity — ECAPA-TDNN WavLM cosine similarity
  - UTMOS (40%): Perceived quality — UTMOS22 Strong, normalized [1,5] → [0,1]
  - WER   (20%): Intelligibility  — Whisper Large V3, mapped 1 - min(wer, 1)
  - VietTone (10%): Tone accuracy — Unicode tone-mark comparison (no extra model)

TRL-compatible interface:
    reward_fn(completions, prompts, reference_text, ref_wav, **kwargs) → List[float]

Per-step breakdown appended to a CSV file and logged via loguru.

Usage:
    from scripts.reward_function import build_mixed_reward_fn
    reward_fn = build_mixed_reward_fn(
        sparktts_model, device="cuda:0", log_csv="logs/gspo_rewards.csv"
    )
"""

import csv
import re
import sys
import unicodedata
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger

SAMPLE_RATE = 16_000

# ── Paths ─────────────────────────────────────────────
EVAL_MODELS_DIR = Path("/data1/speech/nhandt23/06_binh/gspo_sparktts/download/tts_eval_models")
UTMOS_CKPT   = EVAL_MODELS_DIR / "mos/utmos22_strong_step7459_v1.pt"
SIM_CKPT     = EVAL_MODELS_DIR / "speaker_similarity/wavlm_large_finetune.pth"
SIM_SSL_DIR  = EVAL_MODELS_DIR / "speaker_similarity/wavlm_large"
WHISPER_DIR  = Path("/data1/speech/nhandt23/06_binh/models/openai--whisper-large-v3")
ZIPVOICE_ROOT = Path("/data1/speech/nhandt23/06_binh/Zipvoice/ZipVoice-flowgrpo")


# ════════════════════════════════════════════════════════
# TOKEN PARSING
# ════════════════════════════════════════════════════════

def parse_semantic_tokens(text: str) -> Optional[list[int]]:
    tokens = [int(t) for t in re.findall(r"bicodec_semantic_(\d+)", text)]
    return tokens if tokens else None


def parse_global_tokens(text: str) -> Optional[list[int]]:
    tokens = [int(t) for t in re.findall(r"bicodec_global_(\d+)", text)]
    return tokens if tokens else None


def extract_text(obj) -> str:
    if isinstance(obj, str):
        return obj
    if isinstance(obj, list):
        return " ".join(m.get("content", "") for m in obj if isinstance(m, dict))
    return str(obj)


# ════════════════════════════════════════════════════════
# AUDIO DECODING
# ════════════════════════════════════════════════════════

def decode_tokens_to_audio(
    sparktts_model,
    semantic_ids: list[int],
    global_ids: list[int],
    device: str,
) -> Optional[np.ndarray]:
    """BiCodec token IDs → float32 numpy waveform at SAMPLE_RATE."""
    try:
        sem_t = torch.tensor(semantic_ids, dtype=torch.long).unsqueeze(0).to(device)
        glo_t = torch.tensor(global_ids, dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            wav_np = sparktts_model.detokenize_audio(glo_t, sem_t)
        if wav_np is None or wav_np.size == 0:
            return None
        return wav_np.astype(np.float32)
    except Exception as e:
        logger.debug(f"BiCodec decode error: {e}")
        return None


# ════════════════════════════════════════════════════════
# WER
# ════════════════════════════════════════════════════════

def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def compute_wer(asr_pipe, audio_np: np.ndarray, ref_text: str) -> float:
    """Return WER ∈ [0, 1], capped at 1.0."""
    from jiwer import wer
    try:
        result = asr_pipe({"raw": audio_np, "sampling_rate": SAMPLE_RATE})
        hyp = _normalize_text(result["text"])
        ref = _normalize_text(ref_text)
        if not ref:
            return 1.0
        return float(min(wer(ref, hyp), 1.0))
    except Exception as e:
        logger.debug(f"ASR/WER error: {e}")
        return 1.0


def wer_to_reward(wer_score: float) -> float:
    """Piecewise WER → reward scalar. Tuned from baseline (mean=5.25%)."""
    if wer_score <= 0.001:   return 1.0
    elif wer_score <= 0.036: return 0.8
    elif wer_score <= 0.072: return 0.5
    elif wer_score <= 0.10:  return 0.2
    elif wer_score <= 0.20:  return -0.5
    else:                    return -2.0


# ════════════════════════════════════════════════════════
# VIET TONE ACCURACY  (Unicode, no extra model)
# ════════════════════════════════════════════════════════

# Precomputed map: NFC vowel char → tone ID (0=flat,1=huyền,2=sắc,3=nặng,4=hỏi,5=ngã)
_TONE_MAP: dict[str, int] = {}
for _tid, _chars in [
    (0, "aăâeêiouươôuy"),
    (1, "àằầèềìòồờùừợởướ"),
    (2, "áắấéếíóốớúứ"),
    (3, "ạặậẹệịọộợụự"),
    (4, "ảẳẩẻểỉỏổởủử"),
    (5, "ãẵẫẽễĩõỗỡũữ"),
]:
    for _ch in _chars:
        _TONE_MAP[unicodedata.normalize("NFC", _ch)] = _tid


def viet_tone_accuracy(hyp: str, ref: str) -> float:
    """Fraction of Vietnamese tone marks in ref that match hyp, positionally."""
    def _tones(t: str) -> list[int]:
        return [_TONE_MAP[ch] for ch in unicodedata.normalize("NFC", t.lower()) if ch in _TONE_MAP]

    ref_tones = _tones(ref)
    if not ref_tones:
        return 1.0  # no tones in ref → not penalized
    hyp_tones = _tones(hyp)
    if not hyp_tones:
        return 0.0
    matches = sum(r == h for r, h in zip(ref_tones, hyp_tones))
    return float(matches / len(ref_tones))


# ════════════════════════════════════════════════════════
# SIM  (ECAPA-TDNN WavLM)
# ════════════════════════════════════════════════════════

def _load_sim_model(device: str):
    if str(ZIPVOICE_ROOT) not in sys.path:
        sys.path.insert(0, str(ZIPVOICE_ROOT))
    from zipvoice_based.eval.models.ecapa_tdnn_wavlm import ECAPA_TDNN_WAVLM
    model = ECAPA_TDNN_WAVLM(
        feat_dim=1024, channels=512, emb_dim=256,
        sr=SAMPLE_RATE, ssl_model_path=str(SIM_SSL_DIR) + "/"
    )
    state = torch.load(str(SIM_CKPT), map_location="cpu")
    model.load_state_dict(state["model"], strict=False)
    return model.to(device).eval()


@torch.no_grad()
def compute_sim(sim_model, gen_wav_np: np.ndarray, ref_wav_np: np.ndarray, device: str) -> float:
    """Cosine similarity in [-1, 1] between generated and reference speaker embeddings."""
    try:
        # ECAPA_TDNN_WAVLM.get_feat expects list of 1D [T] raw waveform tensors
        gen_t = torch.from_numpy(gen_wav_np).float().to(device)  # [T]
        ref_t = torch.from_numpy(ref_wav_np).float().to(device)  # [T]
        gen_emb = sim_model([gen_t])
        ref_emb = sim_model([ref_t])
        return float(F.cosine_similarity(ref_emb, gen_emb, dim=-1).item())
    except Exception as e:
        logger.debug(f"SIM error: {e}")
        return 0.0


# ════════════════════════════════════════════════════════
# UTMOS
# ════════════════════════════════════════════════════════

def _load_utmos(device: str):
    if str(ZIPVOICE_ROOT) not in sys.path:
        sys.path.insert(0, str(ZIPVOICE_ROOT))
    from zipvoice_based.eval.models.utmos import UTMOS22Strong
    model = UTMOS22Strong()
    model.load_state_dict(torch.load(str(UTMOS_CKPT), map_location="cpu"))
    return model.to(device).eval()


@torch.no_grad()
def compute_utmos(utmos_model, gen_wav_np: np.ndarray, device: str) -> float:
    """UTMOS score normalized [1,5] → [0,1]."""
    try:
        wav_t = torch.from_numpy(gen_wav_np).float().unsqueeze(0).to(device)
        score = utmos_model(wav_t, SAMPLE_RATE)
        return float(max(min(score.item(), 5.0), 1.0) - 1.0) / 4.0
    except Exception as e:
        logger.debug(f"UTMOS error: {e}")
        return 0.0


# ════════════════════════════════════════════════════════
# FACTORY
# ════════════════════════════════════════════════════════

def build_mixed_reward_fn(
    sparktts_model,
    device: str = "cuda:0",
    log_csv: Optional[str] = None,
    w_sim: float = 0.30,
    w_utmos: float = 0.40,
    w_wer: float = 0.20,
    w_tone: float = 0.10,
):
    """Factory: create TRL GRPOTrainer-compatible mixed reward function.

    Args:
        sparktts_model: Full SparkTTS model (BiCodec used for audio decoding).
        device: Device for reward models (Whisper, SIM, UTMOS).
        log_csv: File path to append per-step CSV reward breakdown.
        w_*: Weights for each component (should sum to 1.0).

    Returns:
        reward_fn(completions, prompts, reference_text, ref_wav, **kwargs) → List[float]
        Final reward is scaled to [-1, +1] for GRPO stability.
    """
    from transformers import pipeline as hf_pipeline

    logger.info(f"[reward] Loading reward models on {device} ...")
    asr_pipe = hf_pipeline(
        "automatic-speech-recognition",
        model=str(WHISPER_DIR),
        device=device,
        torch_dtype=torch.float16,
    )
    asr_pipe.model.config.forced_decoder_ids = asr_pipe.tokenizer.get_decoder_prompt_ids(
        language="vi", task="transcribe"
    )
    sim_model = _load_sim_model(device)
    utmos_model = _load_utmos(device)
    logger.info("[reward] All models loaded.")

    # CSV setup
    csv_path = Path(log_csv) if log_csv else None
    if csv_path:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not csv_path.exists():
            with open(csv_path, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["step", "idx", "sim", "utmos", "wer_raw", "wer_norm", "tone", "final"]
                )

    _step = [0]

    def reward_fn(
        completions,
        prompts=None,
        reference_text=None,
        ref_wav=None,   # List[np.ndarray] reference waveforms (16kHz mono)
        **kwargs,
    ) -> list[float]:
        """
        Args:
            completions:    List of generated token sequences (str or chat list).
            prompts:        List of prompt strings (for extracting global tokens).
            reference_text: List[str] ground-truth texts (WER + tone).
            ref_wav:        List[np.ndarray] (16kHz mono) for SIM. Optional.
        """
        _step[0] += 1
        step = _step[0]
        rewards: list[float] = []
        csv_rows: list[list] = []

        for idx, completion in enumerate(completions):
            text = extract_text(completion)
            ref_text = reference_text[idx] if reference_text and idx < len(reference_text) else ""
            ref_wav_np = ref_wav[idx] if ref_wav and idx < len(ref_wav) else None

            # 1. Parse tokens
            semantic_ids = parse_semantic_tokens(text)
            if not semantic_ids or len(semantic_ids) < 10:
                rewards.append(-5.0)
                csv_rows.append([step, idx, 0, 0, 1.0, 0.0, 0, -5.0])
                logger.debug(f"[reward] s={step} i={idx} → NO_SEMANTIC_TOKENS")
                continue

            global_ids = parse_global_tokens(text)
            if not global_ids and prompts and idx < len(prompts):
                global_ids = parse_global_tokens(extract_text(prompts[idx]))
            if not global_ids:
                rewards.append(-3.0)
                csv_rows.append([step, idx, 0, 0, 1.0, 0.0, 0, -3.0])
                logger.debug(f"[reward] s={step} i={idx} → NO_GLOBAL_TOKENS")
                continue

            # 2. Decode to audio
            audio_np = decode_tokens_to_audio(sparktts_model, semantic_ids, global_ids, device)
            if audio_np is None or len(audio_np) < SAMPLE_RATE * 0.3:
                rewards.append(-4.0)
                csv_rows.append([step, idx, 0, 0, 1.0, 0.0, 0, -4.0])
                logger.debug(f"[reward] s={step} i={idx} → DECODE_FAILED")
                continue

            # 3. WER
            wer_raw = compute_wer(asr_pipe, audio_np, ref_text)
            wer_norm = 1.0 - wer_raw  # higher = better

            # 4. Tone (reuse ASR result text)
            try:
                hyp_text = _normalize_text(
                    asr_pipe({"raw": audio_np, "sampling_rate": SAMPLE_RATE})["text"]
                )
            except Exception:
                hyp_text = ""
            tone_score = viet_tone_accuracy(hyp_text, ref_text) if ref_text else 0.5

            # 5. SIM
            if ref_wav_np is not None:
                sim_score = compute_sim(sim_model, audio_np, ref_wav_np, device)
                sim_score = (sim_score + 1.0) / 2.0  # [-1,1] → [0,1]
            else:
                sim_score = 0.5  # neutral

            # 6. UTMOS
            utmos_score = compute_utmos(utmos_model, audio_np, device)

            # 7. Weighted sum → scale to [-1, +1]
            raw = w_sim * sim_score + w_utmos * utmos_score + w_wer * wer_norm + w_tone * tone_score
            final_reward = raw * 2.0 - 1.0

            rewards.append(float(final_reward))
            csv_rows.append([step, idx,
                             round(sim_score, 4), round(utmos_score, 4),
                             round(wer_raw, 4), round(wer_norm, 4),
                             round(tone_score, 4), round(final_reward, 4)])

            logger.info(
                f"[reward] s={step} i={idx} | "
                f"SIM={sim_score:.3f} UTMOS={utmos_score:.3f} "
                f"WER={wer_raw:.3f} Tone={tone_score:.3f} → {final_reward:.4f}"
            )

        # Append CSV
        if csv_path and csv_rows:
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerows(csv_rows)

        return rewards

    return reward_fn
