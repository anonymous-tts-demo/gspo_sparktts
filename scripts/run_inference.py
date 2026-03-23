
import os
import argparse
import torch
import numpy as np
import soundfile as sf
from tqdm import tqdm
from loguru import logger
from transformers import AutoModel, AutoModelForCausalLM, AutoProcessor
from peft import PeftModel
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Inference with trained GSPO model")
    parser.add_argument("--model_dir", type=str, default="/data1/speech/nhandt23/06_binh/spark_based/Vi-SparkTTS-0.5B", help="Path to base model directory")
    parser.add_argument("--lora_path", type=str, default="/data1/speech/nhandt23/06_binh/gspo_sparktts/outputs/gspo_run_2000_steps", help="Path to trained LoRA checkpoint")
    parser.add_argument("--text_file", type=str, default="/data1/speech/nhandt23/06_binh/gspo_sparktts/infer/text.txt", help="Path to text file containing test sentences")
    parser.add_argument("--prompt_wav", type=str, default="/data1/speech/speechData/rawData/Public_Data/phoaudiobook/wavs/Nguyễn_Kim_Ngân_0225494.wav", help="Path to prompt wav file")
    parser.add_argument("--output_dir", type=str, default="/data1/speech/nhandt23/06_binh/gspo_sparktts/infer/visparkRL", help="Directory to save output audios")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on")
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # 1. Load Base SparkTTS Model
    logger.info(f"Loading SparkTTS base from {args.model_dir}")
    sparktts = AutoModel.from_pretrained(
        args.model_dir, 
        trust_remote_code=True, 
        torch_dtype=torch.float32,
    ).eval().to(device)

    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
    processor.link_model(sparktts)
    
    # 2. Load and merge LoRA checkpoint
    if args.lora_path and Path(args.lora_path).exists():
        logger.info(f"Patching LLM with GSPO checkpoint: {args.lora_path}")
        llm_path = os.path.join(args.model_dir, "LLM")
        base_llm = AutoModelForCausalLM.from_pretrained(llm_path, torch_dtype=torch.float32)
        peft_llm = PeftModel.from_pretrained(base_llm, args.lora_path)
        merged   = peft_llm.merge_and_unload()
        # Replace the LLM inside SparkTTS
        sparktts.llm = merged.to(device)
        logger.success("GSPO checkpoint merged into SparkTTS.llm")
    else:
        logger.warning("LoRA checkpoint not found or not provided. Using base model.")
    
    sparktts.eval()
    
    # 3. Read test texts
    logger.info(f"Reading test texts from {args.text_file} ...")
    with open(args.text_file, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    
    logger.info(f"Found {len(texts)} sentences to synthesize.")
    
    # 4. Synthesize
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load prompt audio
    prompt_wav_path = args.prompt_wav
    
    for i, text in enumerate(tqdm(texts, desc="Synthesizing")):
        try:
            logger.debug(f"Processing text {i+1}/{len(texts)}: {text}")
            
            # Use processor to create model inputs
            if prompt_wav_path and Path(prompt_wav_path).exists():
                inputs = processor(text=text, prompt_speech_path=prompt_wav_path, return_tensors="pt")
            else:
                logger.warning(f"Prompt wav not found at {prompt_wav_path}, using default voice")
                inputs = processor(text=text, gender="female", pitch="moderate", speed="moderate", return_tensors="pt")
            
            input_ids   = inputs["input_ids"].to(device)
            attn_mask   = inputs["attention_mask"].to(device)
            input_len   = input_ids.shape[1]
            global_ids  = inputs.get("global_token_ids_prompt", None)
            if global_ids is not None:
                global_ids = global_ids.to(device)
                
            # Generate
            with torch.no_grad():
                generated = sparktts.llm.generate(
                    input_ids=input_ids, attention_mask=attn_mask,
                    max_new_tokens=2048, do_sample=True, temperature=0.8, top_p=0.95,
                )
            
            # Decode to audio
            result = processor.decode(generated, global_token_ids_prompt=global_ids, input_ids_len=input_len)
            wav_np = result.get("audio", None)
            sample_rate = result.get("sample_rate", 16000) # Default SparkTTS SR is 16k
            
            if wav_np is not None:
                # Save audio
                output_path = os.path.join(args.output_dir, f"sample_{i:02d}.wav")
                sf.write(output_path, wav_np, sample_rate)
                logger.debug(f"Saved to {output_path}")
            else:
                logger.error(f"Failed to generate audio for text '{text}'")
                
        except Exception as e:
            logger.error(f"Error synthesizing text '{text}': {e}")
            
    logger.success(f"Inference complete. Audios saved to {args.output_dir}")

if __name__ == "__main__":
    main()
