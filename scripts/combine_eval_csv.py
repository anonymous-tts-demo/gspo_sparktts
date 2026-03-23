import pandas as pd
import glob
import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--prefix", type=str, required=True)
    args = parser.parse_args()

    files = sorted(glob.glob(f"{args.output_dir}/{args.prefix}_chunk*.csv"))
    if not files:
        print(f"No chunk files found in {args.output_dir} matching {args.prefix}_chunk*.csv")
        sys.exit(1)

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
            print(f"Loaded {f} ({len(df)} rows)")
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if not dfs:
        print("No valid dataframes loaded.")
        sys.exit(1)

    merged = pd.concat(dfs, ignore_index=True)
    out_path = Path(args.output_dir) / f"{args.prefix}.csv"
    merged.to_csv(out_path, index=False)
    print(f"\nMerged {len(files)} files into {out_path} with {len(merged)} rows.")

    print("\nRESULTS SUMMARY")
    print("="*60)
    print(f"{'Model':<15} {'WER%':>8} {'UTMOS':>8} {'SIM':>8} {'RTF':>8}")
    print("-" * 50)
    
    wer = merged['wer'].mean() * 100
    utmos = merged['utmos'].mean()
    sim = merged['sim'].mean()
    rtf = merged['rtf'].mean()
    
    print(f"{args.prefix:<15} {wer:>8.2f} {utmos:>8.3f} {sim:>8.3f} {rtf:>8.3f}")

if __name__ == "__main__":
    main()
