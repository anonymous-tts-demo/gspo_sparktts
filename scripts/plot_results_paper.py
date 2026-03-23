import matplotlib.pyplot as plt
import numpy as np

# Styling parameters
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 14,          # Tăng kích thước chữ tổng thể
    "axes.labelsize": 16,     # Tăng kích thước chữ trục y (Scores)
    "axes.titlesize": 16,
    "xtick.labelsize": 16,    # Tăng kích thước chữ trục x (WER, UTMOS...)
    "ytick.labelsize": 14,    # Tăng kích thước chữ số trên trục y
    "legend.fontsize": 16     # Tăng kích thước chữ của Legend (Baseline, ...)
})

# Data (Excluding Ground Truth for grouped comparison, or keeping it as a baseline)
models = ['Baseline', 'Supervised Fine-Tuning', 'Vi-SparkRL (Ours)']
metrics = ['WER (%)', 'UTMOS', 'SIM', 'Tone Acc.']

# Final scores
data = {
    'WER (%)': [6.54, 4.12, 2.85],
    'UTMOS': [3.21, 3.65, 4.10],
    'SIM': [0.45, 0.68, 0.75],
    'Tone Acc.': [0.78, 0.85, 0.96]
}

# Values for Ground Truth (as reference lines)
gt_values = {
    'WER (%)': 1.12,
    'UTMOS': 4.35,
    'SIM': 0.81,
    'Tone Acc.': 1.00
}

# Colors from the image (Grayish, Light Blue, Deep Blue)
colors = ['#D9D9D9', '#7FB3D5', '#5499C7']

fig, ax = plt.subplots(figsize=(14, 6))

x = np.arange(len(metrics))  # 4 metrics
width = 0.25

# Extract values for each model
baseline_vals = [data[m][0] for m in metrics]
sft_vals = [data[m][1] for m in metrics]
ours_vals = [data[m][2] for m in metrics]

# Plot bars
rects1 = ax.bar(x - width, baseline_vals, width, label=models[0], color=colors[0], edgecolor='white')
rects2 = ax.bar(x, sft_vals, width, label=models[1], color=colors[1], edgecolor='white')
rects3 = ax.bar(x + width, ours_vals, width, label=models[2], color=colors[2], edgecolor='white')

ax.set_ylabel('Scores')
# Đã xóa title 'Performance Comparison' theo yêu cầu
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontweight='bold') # In đậm các metrics ở trục X để dễ nhìn hơn

# Grid: only horizontal
ax.yaxis.grid(True, linestyle='-', color='#F0F0F0')
ax.set_axisbelow(True)

# Remove top/right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Label on top
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=14) # Tăng kích thước chữ số trên đầu cột

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

# Add legend
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False)

plt.tight_layout()
plt.savefig('plots/results_bar_chart.png', dpi=600, bbox_inches='tight')
print("Saved F5-style plot to plots/results_bar_chart.png")
