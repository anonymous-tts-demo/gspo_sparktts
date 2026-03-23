"""Stage 2: Train Vi-SparkTTS-0.5B LLM with GSPO.

Loads from SFT checkpoint (LoRA adapter), applies mixed reward:
  SIM 30% | UTMOS 40% | WER 20% | VietTone 10%

Uses TRL GRPOTrainer with importance_sampling_level="sequence" (GSPO mode).
Only the LLM backbone is trained; BiCodec + wav2vec2 stay frozen.

Usage:
    CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
        --config_file configs/accelerate_config.yaml \
        scripts/train_gspo_sparktts.py \
        --sft_checkpoint outputs/sft_run/final \
        --output_dir outputs/gspo_run
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from datasets import Dataset

# ── Local TRL path ────────────────────────────────────
TRL_PATH = Path("/data1/speech/nhandt23/06_binh/trl")
if TRL_PATH.exists():
    sys.path.insert(0, str(TRL_PATH))

# ── Project root (so `scripts.reward_function` is importable) ──
PROJECT_ROOT = Path("/data1/speech/nhandt23/06_binh/gspo_sparktts")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════
SPARKTTS_MODEL_DIR = Path("/data1/speech/nhandt23/06_binh/spark_based/Vi-SparkTTS-0.5B")
LLM_PATH           = SPARKTTS_MODEL_DIR / "LLM"
TRAIN_DATA         = Path("/data1/speech/nhandt23/06_binh/gspo_sparktts/data/sft/train_sft.jsonl")
OUTPUT_DIR         = Path("/data1/speech/nhandt23/06_binh/gspo_sparktts/outputs/gspo_run")
LOG_CSV            = Path("/data1/speech/nhandt23/06_binh/gspo_sparktts/logs/gspo_rewards.csv")
SAMPLE_RATE        = 16_000


# ════════════════════════════════════════════════════════
# DATA LOADING
# ════════════════════════════════════════════════════════

def _extract_content_text(prompt: str) -> str:
    """Extract plain Vietnamese text from between <|start_content|> and <|end_content|>."""
    import re
    m = re.search(r"<\|start_content\|>(.*?)<\|end_content\|>", prompt, re.DOTALL)
    return m.group(1).strip() if m else ""


def load_gspo_dataset(path: Path) -> Dataset:
    """Load JSONL → HuggingFace Dataset.

    Each row must have at minimum:
      - "prompt"  (str):  The LLM prompt (contains text between start/end_content).
      - "wav"     (str):  Path to reference wav for SIM.

    GRPOTrainer expects a "prompt" column at minimum.
    Extra columns (reference_text, wav) are passed as kwargs to reward_fn.
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line.strip())
            prompt = d.get("prompt", d.get("text", ""))
            # Extract plain text for WER/Tone — NOT from completion (which has tokens)
            ref_text = _extract_content_text(prompt)
            rows.append({
                "prompt":         prompt,
                "reference_text": ref_text,
                "wav":            d.get("wav", ""),
            })
    logger.info(f"Loaded {len(rows):,} GSPO samples from {path}")
    if rows:
        logger.info(f"  Example ref_text: '{rows[0]['reference_text'][:100]}...'")
    return Dataset.from_list(rows)


def make_ref_wav_loader(sample_rate: int = SAMPLE_RATE):
    """Build a collate-time ref_wav loader to pass numpy arrays to reward_fn."""
    import torchaudio
    import torchaudio.transforms as T
    import os

    wav_root = "/data1/speech/nhandt23/06_binh/chatter_v2v/data/normalized/phoaudiobook/wavs"

    def load_wav(wav_path: str) -> np.ndarray:
        """Load wav → float32 mono numpy at SAMPLE_RATE."""
        if wav_path and not str(wav_path).startswith("/"):
            wav_path = os.path.join(wav_root, wav_path)

        if not wav_path or not Path(wav_path).exists():
            return np.zeros(sample_rate, dtype=np.float32)
        try:
            audio, sr = torchaudio.load(wav_path)
            if sr != sample_rate:
                audio = T.Resample(sr, sample_rate)(audio)
            audio = audio[0].numpy().astype(np.float32)  # mono
            return audio
        except Exception as e:
            logger.debug(f"ref_wav load failed for {wav_path}: {e}")
            return np.zeros(sample_rate, dtype=np.float32)

    return load_wav


# ════════════════════════════════════════════════════════
# REWARD WRAPPER  (injects ref_wav at call time)
# ════════════════════════════════════════════════════════

def wrap_reward_with_ref_wav(reward_fn, load_wav_fn):
    """Wraps the mixed reward_fn to load ref_wav on the fly from 'wav' column."""

    def wrapped(completions, prompts=None, reference_text=None, wav=None, **kwargs):
        ref_wav = None
        if wav:
            ref_wav = [load_wav_fn(p) for p in wav]
        return reward_fn(
            completions=completions,
            prompts=prompts,
            reference_text=reference_text,
            ref_wav=ref_wav,
            **kwargs,
        )

    return wrapped


# ════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="GSPO training for Vi-SparkTTS")
    parser.add_argument("--sft_checkpoint", type=str, default=None,
                        help="Path to SFT LoRA checkpoint (outputs/sft_run/final). "
                             "If None, trains from base LLM.")
    parser.add_argument("--dataset_path", type=str, default=str(TRAIN_DATA))
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="Override num_train_epochs if > 0.")
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--per_device_batch_size", type=int, default=2)
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--max_completion_length", type=int, default=1024)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--epsilon", type=float, default=3e-4)
    parser.add_argument("--epsilon_high", type=float, default=4e-4)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--reward_device", type=str, default="cuda:0")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Resume from checkpoint, e.g. outputs/gspo_run/checkpoint-1000")
    args = parser.parse_args()

    from trl import GRPOConfig, GRPOTrainer
    from transformers import AutoModel, AutoTokenizer
    from peft import PeftModel

    # ── 1. SparkTTS model  (for BiCodec decoding only) ──
    logger.info("Loading SparkTTS model for BiCodec ...")
    sparktts_model = AutoModel.from_pretrained(
        str(SPARKTTS_MODEL_DIR),
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    sparktts_model.eval()
    sparktts_model.bicodec.to(args.reward_device)
    logger.success("SparkTTS BiCodec ready.")

    # ── 2. Mixed reward function ──
    from scripts.reward_function import build_mixed_reward_fn
    _reward_fn = build_mixed_reward_fn(
        sparktts_model=sparktts_model,
        device=args.reward_device,
        log_csv=str(LOG_CSV),
    )
    load_wav_fn = make_ref_wav_loader()
    reward_fn   = wrap_reward_with_ref_wav(_reward_fn, load_wav_fn)

    # ── 3. Tokenizer ──
    logger.info(f"Loading tokenizer from {LLM_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(str(LLM_PATH))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required for batch generation

    # ── 4. Dataset ──
    dataset = load_gspo_dataset(Path(args.dataset_path))

    # ── 5. Training config ──
    training_args = GRPOConfig(
        output_dir=args.output_dir,

        # GSPO specifics
        importance_sampling_level="sequence",
        epsilon=args.epsilon,
        epsilon_high=args.epsilon_high,
        loss_type="grpo",
        beta=0.01,

        # Training
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,

        # Generation
        temperature=args.temperature,

        # Logging + checkpointing
        logging_steps=1,
        log_completions=True,
        save_steps=100,
        save_total_limit=3,

        # Dtype — must be float32 due to cuFFT dependency in BiCodec
        bf16=False,
        fp16=False,
        gradient_checkpointing=True,
        max_grad_norm=1.0,

        report_to="none",
    )

    # ── 6. Load LLM model ──
    from transformers import AutoModelForCausalLM

    if args.sft_checkpoint and Path(args.sft_checkpoint).exists():
        logger.info(f"Loading LLM from SFT LoRA checkpoint: {args.sft_checkpoint}")
        # Must use AutoModelForCausalLM (not AutoModel) — PEFT CausalLM needs
        # prepare_inputs_for_generation and lm_head from Qwen2ForCausalLM
        base_model = AutoModelForCausalLM.from_pretrained(
            str(LLM_PATH),
            torch_dtype=torch.float32,
        )
        # Load LoRA adapter then merge for clean base → new LoRA in GSPO
        llm_model = PeftModel.from_pretrained(base_model, args.sft_checkpoint)
        llm_model = llm_model.merge_and_unload()
        logger.success("SFT checkpoint merged into LLM.")
    else:
        logger.warning(
            f"SFT checkpoint not found at {args.sft_checkpoint!r}. "
            "Training from base LLM (not recommended for Stage 2)."
        )
        llm_model = str(LLM_PATH)  # GRPOTrainer can take a path string

    # ── 7. LoRA config for GSPO phase ──
    from peft import LoraConfig
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "up_proj", "down_proj", "gate_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )

    # ── 8. Trainer ──
    logger.info("Initializing GRPOTrainer (GSPO mode) with mixed reward ...")
    trainer = GRPOTrainer(
        model=llm_model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=[reward_fn],
        peft_config=peft_config,
    )

    logger.info("Starting GSPO training ...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    logger.info("Saving final GSPO checkpoint ...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.success(f"GSPO model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
