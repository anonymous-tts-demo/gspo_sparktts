"""Reconstruct full GSPO training history by aligning each log file independently.

Steps:
1. Load ground-truth global_step list from trainer_state.json.
2. For each log file:
    a. Extract [reward] lines.
    b. Group into step batches.
    c. Find best matching offset in global history by reward value.
3. Consolidate and plot.
"""
import re
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ── Configuration ─────────────────────────────────────
STATE_PATH = "outputs/gspo_run/checkpoint-2800/trainer_state.json"
LOG_DIR = "logs"
OUTPUT_TABLE = "plots/training_progress_2800_reconstructed_table.txt"
OUTPUT_PLOT  = "plots/training_progress_2800_reconstructed.png"

# ── 1. Load Global History ────────────────────────────
print(f"Loading {STATE_PATH} ...")
with open(STATE_PATH, "r") as f:
    state = json.load(f)

global_history = []
for entry in state.get("log_history", []):
    if "reward" in entry and "step" in entry:
        global_history.append({
            "step": entry["step"],
            "reward_gt": entry["reward"]
        })

df_global = pd.DataFrame(global_history)
n_global = len(df_global)
print(f"  Found {n_global} global training steps in history.")

# ── 2. Parse Logs ─────────────────────────────────────
REWARD_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*\[reward\] s=(\d+) i=\d+ \| "
    r"SIM=([\d.]+) UTMOS=([\d.]+) WER=([\d.]+) Tone=([\d.]+) → ([\-\d.]+)"
)

log_files = sorted([f for f in os.listdir(LOG_DIR) if f.startswith("train_gspo") and f.endswith(".log")])

# To store (step_number, metrics) mapped to global steps
full_metrics = {s: [] for s in df_global["step"]}

for lf in log_files:
    path = os.path.join(LOG_DIR, lf)
    rows = []
    with open(path, "r", errors="replace") as f:
        for line in f:
            m = REWARD_RE.search(line)
            if m:
                ts, s, sim, utmos, wer, tone, reward = m.groups()
                rows.append({
                    "log_s": int(s),
                    "sim": float(sim),
                    "utmos": float(utmos),
                    "wer": float(wer),
                    "tone": float(tone),
                    "reward": float(reward)
                })
    
    if not rows: continue
    df_l = pd.DataFrame(rows)
    
    # Aggregate by log_s
    df_l_agg = df_l.groupby("log_s").agg({
        "sim": "mean", "utmos": "mean", "wer": "mean", "tone": "mean", "reward": "mean"
    }).reset_index().sort_values("log_s")
    
    n_l = len(df_l_agg)
    
    # Manual override based on timestamp validation:
    # Checkpoint 2800 was created at 2026-03-13 07:13.
    # In train_gspo5.log, s=1799 happened at 2026-03-13 07:13:50.
    # Therefore, global_step = log_s + 1001.
    manual_offsets = {
        "train_gspo5.log": 1001,
        "train_gspo4.log": 0 # Starting from scratch
    }
    
    if lf in manual_offsets:
        offset = manual_offsets[lf]
        print(f"  {lf}: using manual offset {offset} (validated via timestamp)")
        best_offset = -1
        # Find which index in df_global corresponds to global step = offset + 1
        start_step = offset + 1
        matching_indices = df_global.index[df_global["step"] == start_step].tolist()
        if matching_indices:
            best_offset = matching_indices[0]
            start_global = df_global.iloc[best_offset]["step"]
            print(f"    Mapped log_s 1..{n_l} to global {start_global}..{start_global + n_l - 1}")
            for i in range(n_l):
                g_idx = best_offset + i
                if g_idx < n_global:
                    g_step = df_global.iloc[g_idx]["step"]
                    full_metrics[g_step].append(df_l_agg.iloc[i])
        continue

    # Find best offset for other logs via reward matching
    l_rewards = df_l_agg["reward"].values
    best_offset = -1
    min_err = 1e9
    
    for offset in range(n_global - n_l + 1):
        g_window = df_global["reward_gt"].values[offset : offset + n_l]
        err = np.mean((g_window - l_rewards)**2)
        if err < min_err:
            min_err = err
            best_offset = offset
            
    if best_offset >= 0 and min_err < 0.05:
        start_global = df_global.iloc[best_offset]["step"]
        print(f"  {lf}: mapped log_s 1..{n_l} to global {start_global}..{start_global + n_l - 1} (err={min_err:.6f})")
        for i in range(n_l):
            g_step = df_global.iloc[best_offset + i]["step"]
            full_metrics[g_step].append(df_l_agg.iloc[i])

# ── 3. Combine ────────────────────────────────────────
final_rows = []
for s in df_global["step"]:
    matches = full_metrics[s]
    if matches:
        # If multiple logs covered this step (overlap), use the latest one
        m = matches[-1]
        final_rows.append({
            "global_step": s,
            "sim": m["sim"],
            "utmos": m["utmos"],
            "wer": m["wer"],
            "tone": m["tone"],
            "reward_total": df_global[df_global["step"] == s]["reward_gt"].values[0]
        })
    else:
        # Still add the goal step with NaN metrics
        final_rows.append({
            "global_step": s,
            "sim": np.nan, "utmos": np.nan, "wer": np.nan, "tone": np.nan,
            "reward_total": df_global[df_global["step"] == s]["reward_gt"].values[0]
        })

df_final = pd.DataFrame(final_rows)
# Interpolate gaps
df_final = df_final.interpolate(method='linear')

# ── 4. Output ─────────────────────────────────────────
os.makedirs("plots", exist_ok=True)

# Table
print("\n--- Summary Table (Reconstructed) ---")
disp = df_final[df_final["global_step"] % 400 == 0][
    ["global_step", "sim", "utmos", "wer", "tone", "reward_total"]
].round(4)
if 2800 not in disp["global_step"].values:
    disp = pd.concat([disp, df_final.tail(1)[["global_step", "sim", "utmos", "wer", "tone", "reward_total"]].round(4)])
print(disp.to_string(index=False))

with open(OUTPUT_TABLE, "w") as f:
    f.write(disp.to_string(index=False))

def generate_plot(df, title_suffix, output_path, step_range=None):
    if step_range:
        plot_df = df[(df["global_step"] >= step_range[0]) & (df["global_step"] <= step_range[1])]
    else:
        plot_df = df

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    W = 50

    def get_y_lims(series, padding=0.1):
        valid = series.dropna()
        if valid.empty: return None, None
        v_min, v_max = valid.min(), valid.max()
        span = v_max - v_min
        if span <= 0: span = 0.1
        return v_min - span * padding, v_max + span * padding

    # Panel 0,0: Dedicated SIM plot
    ax = axes[0,0]
    s_sim = plot_df.sim.rolling(W, min_periods=1).mean()
    ax.plot(plot_df.global_step, plot_df.sim, alpha=0.1, color='blue')
    ax.plot(plot_df.global_step, s_sim, color='blue', lw=2, label='SIM ↑')
    ax.set(title=f'Speaker Similarity (SIM) ↑ {title_suffix}', xlabel='Step', ylabel='Score')
    # Use tighter padding for SIM to show stability but allow trend visibility
    y1, y2 = get_y_lims(s_sim, padding=0.05)
    if y1 is not None: ax.set_ylim(y1, y2)
    ax.grid(True); ax.legend()

    # Panel 0,1: UTMOS
    ax = axes[0,1]
    s_utmos = plot_df.utmos.rolling(W, min_periods=1).mean()
    ax.plot(plot_df.global_step, plot_df.utmos, alpha=0.1, color='green')
    ax.plot(plot_df.global_step, s_utmos, color='green', lw=2, label='UTMOS ↑')
    ax.set(title=f'Audio Quality (UTMOS) ↑ {title_suffix}', xlabel='Step', ylabel='Score')
    y1, y2 = get_y_lims(s_utmos)
    if y1 is not None: ax.set_ylim(y1, y2)
    ax.grid(True); ax.legend()

    # Panel 1,0: WER
    ax = axes[1,0]
    s_wer = plot_df.wer.rolling(W, min_periods=1).mean()
    ax.plot(plot_df.global_step, plot_df.wer, alpha=0.1, color='orange')
    ax.plot(plot_df.global_step, s_wer, color='orange', lw=2, label='WER ↓')
    ax.set(title=f'Word Error Rate (WER) ↓ {title_suffix}', xlabel='Step', ylabel='Ratio')
    y1, y2 = get_y_lims(s_wer)
    if y1 is not None: ax.set_ylim(y1, y2)
    ax.grid(True); ax.legend()

    # Panel 1,1: Tone
    ax = axes[1,1]
    s_tone = plot_df.tone.rolling(W, min_periods=1).mean()
    ax.plot(plot_df.global_step, plot_df.tone, alpha=0.1, color='red')
    ax.plot(plot_df.global_step, s_tone, color='red', lw=2, label='VietTone ↑')
    ax.set(title=f'Vietnamese Tone Accuracy ↑ {title_suffix}', xlabel='Step', ylabel='Ratio')
    y1, y2 = get_y_lims(s_tone)
    if y1 is not None: ax.set_ylim(y1, y2)
    ax.grid(True); ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

print(f"Generating full plot to {OUTPUT_PLOT} ...")
generate_plot(df_final, "(Steps 1-2800)", OUTPUT_PLOT)

ZOOM_PLOT = "plots/training_progress_1750_2800_zoom.png"
print(f"Generating zoomed plot to {ZOOM_PLOT} ...")
generate_plot(df_final, "(Steps 1750-2800)", ZOOM_PLOT, step_range=(1750, 2800))

print(f"✅ Reconstructed progress saved.")
