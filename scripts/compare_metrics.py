import pandas as pd
from pathlib import Path

# Paths
GSPO_CSV = Path("/data1/speech/nhandt23/06_binh/gspo_sparktts/outputs/eval_test_gspo_2800/eval_gspo_2800.csv")
BASELINE_CSV = Path("/data1/speech/nhandt23/06_binh/gspo_sparktts/outputs/eval_test_baseline/eval_base.csv")

def print_metrics(name, df):
    valid = df[df["wer"].notna() & df["utmos"].notna() & df["sim"].notna()]
    wer = valid["wer"].mean() * 100
    utmos = valid["utmos"].mean()
    sim = valid["sim"].mean()
    rtf = valid.get("rtf", pd.Series([0])).mean()
    print(f"| {name:<12} | {wer:>6.2f}% | {utmos:>5.3f} | {sim:>5.3f} | {rtf:>5.3f} |")

print("### Comparison of Metrics (2251 samples from test_2k.csv)\n")
print("| Model        | WER (%) | UTMOS |  SIM  |  RTF  |")
print("|--------------|---------|-------|-------|-------|")

# Ground Truth (from previous analysis)
print("| Ground Truth |   1.12% | 2.336 | 0.811 |   N/A |")

# Baseline
if BASELINE_CSV.exists():
    df_base = pd.read_csv(BASELINE_CSV)
    print_metrics("Baseline", df_base)
else:
    print("| Baseline     | Running | Run.. | Run.. | Run.. |")

# GSPO 2800
if GSPO_CSV.exists():
    df_gspo = pd.read_csv(GSPO_CSV)
    print_metrics("GSPO-2800", df_gspo)
else:
    print("| GSPO-2800    | Missing | Mis.. | Mis.. | Mis.. |")
