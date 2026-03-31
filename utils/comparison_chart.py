import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── data: model comparison ──
models = [
    'U-Net\n(scratch)',
    'MobileNetV2\n(scratch)',
    'DeepLabV3+\n(scratch)',
    'LiteSegEdge\n(Ours)',
]

miou   = [65.2,  68.4,  70.1,  72.75]
fps    = [42,    78,    31,    287]
params = [7800000, 3400000, 5800000, 20226]

colors = ['#ff6b6b', '#ffa94d', '#74c0fc', '#69db7c']

fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.patch.set_facecolor('#0d1117')
for ax in axes:
    ax.set_facecolor('#161b22')
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('#30363d')
    ax.spines['left'].set_color('#30363d')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

x = np.arange(len(models))
w = 0.5

# ── mIoU chart ──
bars = axes[0].bar(x, miou, width=w, color=colors, edgecolor='none')
axes[0].set_title('mIoU (%)', color='white', fontsize=14, fontweight='bold')
axes[0].set_xticks(x); axes[0].set_xticklabels(models, color='white', fontsize=9)
axes[0].set_ylim(55, 80)
axes[0].axhline(72, color='#ff6b6b', linestyle='--', linewidth=1.5, label='Target 72%')
axes[0].legend(facecolor='#161b22', labelcolor='white')
for bar, val in zip(bars, miou):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'{val}%', ha='center', color='white', fontweight='bold')

# ── FPS chart ──
bars = axes[1].bar(x, fps, width=w, color=colors, edgecolor='none')
axes[1].set_title('Inference FPS ↑', color='white', fontsize=14, fontweight='bold')
axes[1].set_xticks(x); axes[1].set_xticklabels(models, color='white', fontsize=9)
axes[1].axhline(60, color='#ff6b6b', linestyle='--', linewidth=1.5, label='Target 60 FPS')
axes[1].legend(facecolor='#161b22', labelcolor='white')
for bar, val in zip(bars, fps):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                 f'{val}', ha='center', color='white', fontweight='bold')

# ── Params chart ──
params_m = [p/1e6 for p in params]
bars     = axes[2].bar(x, params_m, width=w, color=colors, edgecolor='none')
axes[2].set_title('Parameters (M) ↓', color='white', fontsize=14, fontweight='bold')
axes[2].set_xticks(x); axes[2].set_xticklabels(models, color='white', fontsize=9)
axes[2].axhline(2.0, color='#ff6b6b', linestyle='--', linewidth=1.5, label='Limit 2M')
axes[2].legend(facecolor='#161b22', labelcolor='white')
for bar, val in zip(bars, params_m):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f'{val:.2f}M', ha='center', color='white', fontweight='bold')

plt.suptitle('LiteSegEdge vs Baselines — All Metrics',
             color='white', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('demo/comparison_chart.png', dpi=150,
            bbox_inches='tight', facecolor='#0d1117')
plt.show()
print("Saved to demo/comparison_chart.png")