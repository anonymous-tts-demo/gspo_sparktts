# GSPO SparkTTS — Vietnamese TTS Fine-tuning with GSPO

Fine-tune Vi-SparkTTS-0.5B LLM backbone using TRL's GSPO algorithm.

## Project Structure

```
gspo_sparktts/
├── scripts/
│   ├── run_baseline.py          # Phase 1: Run SparkTTS baseline inference
│   ├── prepare_data.py          # Phase 2: Convert data → TRL format
│   ├── reward_function.py       # Phase 3: TTS reward (WER + SIM + UTMOS)
│   └── train_gspo_sparktts.py   # Phase 4: GSPO training
├── configs/
│   └── gspo_config.yaml         # Training hyperparameters
├── data/                        # Processed datasets
├── outputs/                     # Model checkpoints & generated audio
└── logs/                        # Training logs
```

## Resources
- **Model**: `/data1/speech/nhandt23/06_binh/spark_based/Vi-SparkTTS-0.5B/`
- **TRL**: `/data1/speech/nhandt23/06_binh/trl/`
- **Data**: `/data1/speech/nhandt23/06_binh/chatter_v2v/data/normalized/phoaudiobook/filtered_15k.csv`
- **Whisper**: `/data1/speech/nhandt23/06_binh/models/openai--whisper-large-v3`
