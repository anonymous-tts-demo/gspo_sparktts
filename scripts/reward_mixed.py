"""Mixed Reward Function for Vi-SparkTTS GSPO.

Components:
  1. SIM (Voice Similarity) - 30% (WavLM + ECAPA-TDNN)
  2. UTMOS (Perceived Quality) - 40%
  3. WER (Intelligibility) - 20% (Whisper Large V3)
  4. VIET_TONE (Thanh điệu accuracy) - 10%
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from loguru import logger
from transformers import pipeline

ZIPVOICE_ROOT = Path("/data1/speech/nhandt23/06_binh/Zipvoice/ZipVoice")
if str(ZIPVOICE_ROOT) not in sys.path:
    sys.path.insert(0, str(ZIPVOICE_ROOT))

class ViSparkTTSReward:
    def __init__(self, ref_audio_path: str, target_text: str, device: str = "cuda"):
        self.device = device
        self.target_text = target_text
        self.ref_audio_path = ref_audio_path
        self.sr = 16000
        
        # Load reference audio for SIM
        self.ref_audio, orig_sr = torchaudio.load(ref_audio_path)
        if orig_sr != self.sr:
            import torchaudio.transforms as T
            self.ref_audio = T.Resample(orig_sr, self.sr)(self.ref_audio)
        if self.ref_audio.shape[0] > 1:
            self.ref_audio = self.ref_audio[0:1, :]
        self.ref_audio = self.ref_audio.to(self.device)

        # Initialize models (lazy load recommended for multi-processing, but keeping simple here)
        self._init_models()

    def _init_models(self):
        logger.info("Loading reward models...")
        # 1. ASR (Whisper for WER)
        from jiwer import wer
        self.compute_wer_fn = wer
        self.asr = pipeline(
            "automatic-speech-recognition", 
            model="/data1/speech/nhandt23/06_binh/models/openai--whisper-large-v3", 
            device=self.device,
            torch_dtype=torch.float16
        )
        self.asr.model.config.forced_decoder_ids = self.asr.tokenizer.get_decoder_prompt_ids(
            language="vi", task="transcribe"
        )

        # 2. UTMOS
        from zipvoice.eval.models.utmos import UTMOS22Strong
        eval_models_dir = Path("/data1/speech/nhandt23/06_binh/gspo_sparktts/download/tts_eval_models")
        utmos_path = eval_models_dir / "mos/utmos22_strong_step7459_v1.pt"
        self.utmos = UTMOS22Strong()
        self.utmos.load_state_dict(torch.load(str(utmos_path), map_location="cpu"))
        self.utmos.to(self.device).eval()

        # 3. SIM (ECAPA-TDNN WavLM)
        from zipvoice.eval.models.ecapa_tdnn_wavlm import ECAPA_TDNN_WAVLM
        sv_ckpt = eval_models_dir / "speaker_similarity/wavlm_large_finetune.pth"
        ssl_dir = eval_models_dir / "speaker_similarity/wavlm_large"
        self.sim_model = ECAPA_TDNN_WAVLM(
            feat_dim=1024, channels=512, emb_dim=256,
            sr=self.sr, ssl_model_path=str(ssl_dir) + "/"
        )
        self.sim_model.load_state_dict(torch.load(str(sv_ckpt), map_location="cpu")["model"], strict=False)
        self.sim_model.to(self.device).eval()
        
        # Precompute Ref Embedding
        with torch.no_grad():
            self.ref_emb = self.sim_model([self.ref_audio])

    def compute_reward(self, generated_audio_path: str):
        # Load generated audio
        gen_audio, orig_sr = torchaudio.load(generated_audio_path)
        if orig_sr != self.sr:
            import torchaudio.transforms as T
            gen_audio = T.Resample(orig_sr, self.sr)(gen_audio)
        if gen_audio.shape[0] > 1:
            gen_audio = gen_audio[0:1, :]
        gen_audio = gen_audio.to(self.device)

        # 1. SIM (Voice Similarity) - 30%
        sim_score = self.voice_similarity(gen_audio)
        
        # 2. UTMOS (Perceived Quality) - 40% 
        utmos_score = self.utmos_score(gen_audio)
        
        # 3. WER (Intelligibility) - 20%
        # asr pipeline expects numpy array
        gen_audio_np = gen_audio.squeeze().cpu().numpy()
        wer_score = self.asr_wer(gen_audio_np)
        
        # 4. VIET_TONE (Thanh điệu accuracy) - 10%
        # Placeholder for tone accuracy (needs specific VIVOS tone logic)
        tone_score = 0.85 # mock for now
        
        # Weighted combination
        final_reward = (
            0.3 * sim_score + 
            0.4 * utmos_score + 
            0.2 * (1 - wer_score) + 
            0.1 * tone_score
        )
        
        return torch.tensor(final_reward, dtype=torch.float32)

    @torch.no_grad()
    def voice_similarity(self, gen_audio):
        gen_emb = self.sim_model([gen_audio])
        sim = F.cosine_similarity(self.ref_emb, gen_emb, dim=-1)
        return sim.item()
    
    @torch.no_grad()
    def utmos_score(self, gen_audio):
        score = self.utmos(gen_audio, self.sr)
        return min(score.item(), 5.0) / 5.0  # Normalize 0-1
    
    def asr_wer(self, gen_audio_np):
        result = self.asr({"raw": gen_audio_np, "sampling_rate": self.sr})
        pred_text = result["text"].lower()
        target = self.target_text.lower()
        wer = self.compute_wer_fn(target, pred_text) if target else 1.0
        return min(wer, 1.0) # Cap at 1.0
