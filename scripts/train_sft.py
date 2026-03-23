"""Stage 1: Supervised Fine-Tuning (SFT) for Vi-SparkTTS-0.5B.

Trains the model on the 10.5k PhoAudiobook train split.
According to the plan:
  - Dataset: TRL JSONL containing Causal LM text+tokens
  - Model: Vi-SparkTTS-0.5B with LoRA
  - Max steps: ~5000 (~2 epochs)
  - Learning rate: 5e-6
  - Effective batch size: 32 (bs=4 * grad_acc=8 per GPU)

Usage (Multi-GPU setup):
    ACCELERATE_LOG_LEVEL=info accelerate launch \
        --config_file configs/accelerate_config.yaml \
        scripts/train_sft.py \
        --output_dir outputs/sft_run
"""

import argparse
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from loguru import logger
from pathlib import Path
from peft import LoraConfig, get_peft_model
from transformers import AutoModel, AutoProcessor, PreTrainedTokenizerBase
from trl import SFTTrainer, SFTConfig

# Constants
SPARKTTS_REPO = "DragonLineageAI/Vi-SparkTTS-0.5B"
LOCAL_MODEL_DIR = "/data1/speech/nhandt23/06_binh/spark_based/Vi-SparkTTS-0.5B"
TRAIN_DATA = "/data1/speech/nhandt23/06_binh/gspo_sparktts/data/sft/train_sft.jsonl"
VAL_DATA = "/data1/speech/nhandt23/06_binh/gspo_sparktts/data/sft/val_sft.jsonl"
MAX_LENGTH = 2048
RESPONSE_TEMPLATE = "<|start_semantic_token|>"


@dataclass
class CompletionOnlyCollator:
    """Pads pre-tokenized input_ids/attention_mask/labels to same length per batch.
    
    Expects each example to already contain:
      - input_ids: List[int]
      - attention_mask: List[int]
      - labels: List[int]  (prompt tokens already masked to -100)
    """
    tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: int | None = None

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        # Determine max_len in this batch
        max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of:
            max_len = ((max_len + self.pad_to_multiple_of - 1)
                       // self.pad_to_multiple_of * self.pad_to_multiple_of)

        pad_id = self.tokenizer.pad_token_id

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for f in features:
            seq_len = len(f["input_ids"])
            pad_len = max_len - seq_len

            batch_input_ids.append(list(f["input_ids"]) + [pad_id] * pad_len)
            # attention_mask may be absent (HF datasets optimization); generate from seq_len
            batch_attention_mask.append([1] * seq_len + [0] * pad_len)
            batch_labels.append(list(f["labels"]) + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


def tokenize_and_mask(example: dict, tokenizer, response_token_ids: list[int]) -> dict:
    """Tokenize a single example's 'text' field and mask prompt tokens in labels."""
    encoded = tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        add_special_tokens=False,
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    # Build labels: copy of input_ids, then mask prompt portion to -100
    labels = list(input_ids)  # copy

    # Find response template position
    tmpl_len = len(response_token_ids)
    template_idx = -1
    for j in range(len(input_ids) - tmpl_len + 1):
        if input_ids[j: j + tmpl_len] == response_token_ids:
            template_idx = j
            break

    if template_idx >= 0:
        # Mask everything before (and including) the response template
        for k in range(template_idx + tmpl_len):
            labels[k] = -100
    else:
        # If template not found, mask everything (safety — no gradient from this sample)
        labels = [-100] * len(labels)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="outputs/sft_run")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--max_steps", type=int, default=5000)
    args = parser.parse_args()

    logger.info("Initializing SFT training setup for Vi-SparkTTS...")
    model_path = LOCAL_MODEL_DIR if Path(LOCAL_MODEL_DIR).exists() else SPARKTTS_REPO

    logger.info("Loading Tokenizer and Processor...")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = processor.tokenizer

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Pre-compute response template token ids
    response_token_ids = tokenizer.encode(RESPONSE_TEMPLATE, add_special_tokens=False)
    logger.info(f"Response template '{RESPONSE_TEMPLATE}' -> token_ids: {response_token_ids}")

    logger.info("Loading Datasets...")
    dataset = load_dataset(
        "json",
        data_files={"train": TRAIN_DATA, "validation": VAL_DATA},
    )

    logger.info("Pre-tokenizing datasets (tokenize + mask prompt)...")
    tokenized_dataset = dataset.map(
        lambda ex: tokenize_and_mask(ex, tokenizer, response_token_ids),
        remove_columns=dataset["train"].column_names,  # drop text, prompt, completion, etc.
        num_proc=4,
        desc="Tokenizing",
    )
    logger.info(
        f"Tokenized: train={len(tokenized_dataset['train'])}, "
        f"val={len(tokenized_dataset['validation'])}"
    )

    logger.info("Loading Model (LoRA)...")
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )

    # Freeze entire model first
    for param in model.parameters():
        param.requires_grad = False

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "up_proj", "down_proj", "gate_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model.llm = get_peft_model(model.llm, lora_config)
    model.llm.print_trainable_parameters()

    # Custom collator that only pads input_ids/attention_mask/labels
    collator = CompletionOnlyCollator(tokenizer=tokenizer)

    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        logging_steps=10,
        save_steps=1000,
        eval_strategy="steps",
        eval_steps=1000,
        warmup_steps=100,
        save_total_limit=3,
        bf16=False,
        fp16=False,
        gradient_checkpointing=False,
        # No dataset_text_field or max_length — data is already tokenized
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    logger.info("Initializing SFTTrainer...")
    # Pass model.llm (Qwen2ForCausalLM + LoRA) directly to the trainer.
    # SparkTTSModel.forward() doesn't accept `labels`, but the LLM does.
    # Only the LLM is trained during SFT — BiCodec and wav2vec2 are frozen/unused.
    trainer = SFTTrainer(
        model=model.llm,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        processing_class=tokenizer,
        data_collator=collator,
    )

    logger.info("Starting SFT training...")
    trainer.train()

    logger.info("Saving final SFT checkpoint...")
    trainer.save_model(os.path.join(args.output_dir, "final"))


if __name__ == "__main__":
    main()
