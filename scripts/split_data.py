"""Split 15k PhoAudiobook into Train/Val/Test (70/15/15).

15,000 samples:
  - Train: 10,500 (70%)  -> SFT + GSPO
  - Val:    2,250 (15%)  -> Reward tuning, early stopping
  - Test:   2,250 (15%)  -> Final evaluation (unseen speakers if possible)
"""

import argparse
import os
from pathlib import Path

import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

DATA_CSV = Path("/data1/speech/nhandt23/06_binh/chatter_v2v/data/normalized/phoaudiobook/filtered_15k.csv")
OUTPUT_DIR = Path("/data1/speech/nhandt23/06_binh/gspo_sparktts/data/splits")

def main():
    parser = argparse.ArgumentParser(description="Split 15k dataset")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_CSV)
    logger.info(f"Loaded {len(df)} samples from {DATA_CSV}")

    if "speaker" not in df.columns:
        raise ValueError("Column 'speaker' not found in dataset")

    # Group by speaker to ensure speakers aren't split across sets if possible,
    # or just stratified split so train/val/test represent diverse speakers.
    # Since we want unseen speakers in test set ideally:
    speakers = df["speaker"].unique()
    logger.info(f"Total unique speakers: {len(speakers)}")

    # Split speakers: 70% train, 30% temp (val+test)
    train_spks, temp_spks = train_test_split(speakers, test_size=0.3, random_state=args.seed)
    
    # Split temp speakers: 50% val, 50% test
    val_spks, test_spks = train_test_split(temp_spks, test_size=0.5, random_state=args.seed)

    df_train = df[df["speaker"].isin(train_spks)].reset_index(drop=True)
    df_val = df[df["speaker"].isin(val_spks)].reset_index(drop=True)
    df_test = df[df["speaker"].isin(test_spks)].reset_index(drop=True)

    logger.info("Split by unique speakers:")
    logger.info(f"  Train: {len(df_train)} samples ({len(train_spks)} speakers)")
    logger.info(f"  Val:   {len(df_val)} samples ({len(val_spks)} speakers)")
    logger.info(f"  Test:  {len(df_test)} samples ({len(test_spks)} speakers)")

    # Adjust counts if strictly pursuing 10500/2250/2250 size rather than strict unseen speakers
    # The 15k dataset has very few speakers (~13 unique speakers according to previous log), 
    # so splitting by speaker might result in highly unbalanced splits (e.g. 1 speaker = 1100 samples).
    # Let's check how many speakers we actually have in 15k
    num_speakers = len(speakers)
    if num_speakers < 50:
        logger.warning(f"Only {num_speakers} speakers found. Falling back to stratified random split!")
        # Fall back to stratify split to keep speaker ratio consistent across splits
        df_train, df_temp = train_test_split(df, test_size=0.3, random_state=args.seed, stratify=df["speaker"])
        df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=args.seed, stratify=df_temp["speaker"])
        
        logger.info("Stratified split results:")
        logger.info(f"  Train: {len(df_train)} samples")
        logger.info(f"  Val:   {len(df_val)} samples")
        logger.info(f"  Test:  {len(df_test)} samples")

    # Save to CSV
    train_path = OUTPUT_DIR / "train_10k.csv"
    val_path = OUTPUT_DIR / "val_2k.csv"
    test_path = OUTPUT_DIR / "test_2k.csv"

    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)
    df_test.to_csv(test_path, index=False)

    logger.success(f"Saved Train to {train_path}")
    logger.success(f"Saved Val   to {val_path}")
    logger.success(f"Saved Test  to {test_path}")

if __name__ == "__main__":
    main()
