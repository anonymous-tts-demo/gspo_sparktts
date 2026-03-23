"""Plot GSPO training progress up to step 2800 with smoothed trend lines.

Combines data from:
  1. logs/gspo_rewards.csv  (steps 1–1836)
  2. logs/train_gspo*.log   (steps 1837+ parsed from [reward] lines)
"""
import re
import os
import warnings

import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── 1. Load CSV data (steps 1–1836) ────────────────────
print("Loading gspo_rewards.csv ...")
df_csv = pd.read_csv(
    "logs/gspo_rewards.csv",
    header=0,  # first row is header: step,idx,sim,utmos,wer_raw,wer_norm,tone,final
)
df_csv = df_csv.rename(columns={"step": "global_step", "final": "reward_total"})
for col in ["global_step", "sim", "utmos", "wer_norm", "tone", "reward_total"]:
    df_csv[col] = pd.to_numeric(df_csv[col], errors="coerce")
df_csv = df_csv.dropna(subset=["global_step"])
# Use wer_norm (normalized WER where 1.0 = perfect) → convert to error ratio
df_csv["wer"] = 1.0 - df_csv["wer_norm"]
df_csv = df_csv[["global_step", "sim", "utmos", "wer", "tone", "reward_total"]]

print(f"  CSV: {len(df_csv)} rows, steps {df_csv['global_step'].min():.0f}–{df_csv['global_step'].max():.0f}")

# ── 2. Parse [reward] lines from log files ─────────────
REWARD_RE = re.compile(
    r"\[reward\] s=(\d+) i=(\d+) \| SIM=([\d.]+) UTMOS=([\d.]+) WER=([\d.]+) Tone=([\d.]+) → ([\-\d.]+)"
)

log_files = sorted(
    f for f in os.listdir("logs")
    if f.startswith("train_gspo") and f.endswith(".log")
)

rows_log = []
for lf in log_files:
    path = os.path.join("logs", lf)
    with open(path, "r", errors="replace") as f:
        for line in f:
            m = REWARD_RE.search(line)
            if m:
                step, idx, sim, utmos, wer, tone, reward = m.groups()
                rows_log.append({
                    "global_step": int(step),
                    "sim": float(sim),
                    "utmos": float(utmos),
                    "wer": float(wer),
                    "tone": float(tone),
                    "reward_total": float(reward),
                })

df_log = pd.DataFrame(rows_log)
# Keep only steps beyond what CSV covers
csv_max = int(df_csv["global_step"].max())
df_log = df_log[df_log["global_step"] > csv_max]
print(f"  Logs: {len(df_log)} rows, steps {df_log['global_step'].min():.0f}–{df_log['global_step'].max():.0f}")

# ── 3. Combine & aggregate ─────────────────────────────
df_all = pd.concat([df_csv, df_log], ignore_index=True)
df_all = df_all[(df_all["global_step"] > 0) & (df_all["global_step"] <= 2800)]

df_agg = (
    df_all
    .groupby("global_step")[["sim", "utmos", "wer", "tone", "reward_total"]]
    .mean()
    .reset_index()
    .sort_values("global_step")
)
print(f"  Combined: {len(df_agg)} unique steps")

# Smooth
WINDOW = 50
for col in ["sim", "utmos", "wer", "tone", "reward_total"]:
    df_agg[f"{col}_smooth"] = df_agg[col].rolling(window=WINDOW, min_periods=1).mean()

# ── 4. Print table ─────────────────────────────────────
print("\n--- Average Metrics (every 400 steps) ---")
display_df = df_agg[df_agg["global_step"] % 400 == 0][
    ["global_step", "sim", "utmos", "wer", "tone", "reward_total"]
].round(4)
# Always include step 2800 if available
if 2800 not in display_df["global_step"].values and 2800 in df_agg["global_step"].values:
    display_df = pd.concat([display_df, df_agg[df_agg["global_step"] == 2800][
        ["global_step", "sim", "utmos", "wer", "tone", "reward_total"]
    ].round(4)])
print(display_df.to_string(index=False))

os.makedirs("plots", exist_ok=True)
with open("plots/training_progress_2800_table.txt", "w") as f:
    f.write(display_df.to_string(index=False))

# ── 5. Plot ────────────────────────────────────────────
print("Plotting ...")
def get_y_lims(series, padding=0.1):
    valid = series.dropna()
    if valid.empty: return None, None
    v_min, v_max = valid.min(), valid.max()
    span = v_max - v_min
    if span <= 0: span = 0.1
    return v_min - span * padding, v_max + span * padding

# ── 5. Plot ────────────────────────────────────────────
print("Plotting ...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Speaker Similarity (SIM)
ax = axes[0, 0]
ax.plot(df_agg["global_step"], df_agg["sim"], alpha=0.1, color="blue")
ax.plot(df_agg["global_step"], df_agg["sim_smooth"], color="blue", lw=2, label="SIM ↑")
ax.set_title(f"Speaker Similarity (SIM) ↑ (w={WINDOW})")
ax.set_xlabel("Step")
ax.set_ylabel("Score")
ax.set_ylim(0.2, 0.6)
ax.legend()
ax.grid(True)

# Audio Quality (UTMOS)
ax = axes[0, 1]
ax.plot(df_agg["global_step"], df_agg["utmos"], alpha=0.1, color="green")
ax.plot(df_agg["global_step"], df_agg["utmos_smooth"], color="green", lw=2, label="UTMOS ↑")
ax.set_title("Audio Quality (UTMOS) ↑")
ax.set_xlabel("Step")
ax.set_ylabel("Score")
ax.set_ylim(0.2, 0.6)
ax.legend()
ax.grid(True)

# WER
ax = axes[1, 0]
ax.plot(df_agg["global_step"], df_agg["wer"], alpha=0.1, color="orange")
ax.plot(df_agg["global_step"], df_agg["wer_smooth"], color="orange", lw=2, label="WER ↓")
ax.set_title("Word Error Rate (WER) ↓")
ax.set_xlabel("Step")
ax.set_ylabel("Ratio")
y1, y2 = get_y_lims(df_agg["wer_smooth"])
if y1 is not None: ax.set_ylim(y1, y2)
ax.legend()
ax.grid(True)

# Tone
ax = axes[1, 1]
ax.plot(df_agg["global_step"], df_agg["tone"], alpha=0.1, color="red")
ax.plot(df_agg["global_step"], df_agg["tone_smooth"], color="red", lw=2, label="VietTone ↑")
ax.set_title("Vietnamese Tone Accuracy ↑")
ax.set_xlabel("Step")
ax.set_ylabel("Ratio")
y1, y2 = get_y_lims(df_agg["tone_smooth"])
if y1 is not None: ax.set_ylim(y1, y2)
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.savefig("plots/training_progress_2800_smoothed.png", dpi=300)
print("✅ Plot saved to plots/training_progress_2800_smoothed.png")
