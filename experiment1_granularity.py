import os
import numpy as np
import matplotlib.pyplot as plt

########################################
#             USER CONFIG              #
########################################

CLASS_NAME = "R50_house_finch"   # 你自己的类别
ROOT = "outputs"

DIR_ORI = os.path.join(ROOT, "VCC_original", CLASS_NAME, "dataset", "patches")
DIR_ADA = os.path.join(ROOT, "VCC_adaptive", CLASS_NAME, "dataset", "patches")

LAYERS = ["layer1", "layer2", "layer3", "layer4"]


########################################
#        LOAD SEGMENT STATISTICS       #
########################################

def collect_stats(version_dir):
    layer_stats = {}

    for layer in LAYERS:
        layer_path = os.path.join(version_dir, layer)
        sizes = []

        for fname in os.listdir(layer_path):
            if fname.endswith(".npy"):
                arr = np.load(os.path.join(layer_path, fname))
                seg_pixels = np.sum(arr > 0)
                sizes.append(seg_pixels)

        if sizes:
            layer_stats[layer] = {
                "count": len(sizes),
                "min": int(np.min(sizes)),
                "mean": float(np.mean(sizes)),
                "max": int(np.max(sizes)),
                "example_shape": arr.shape
            }
        else:
            layer_stats[layer] = {"count": 0}
    return layer_stats


########################################
#             PRINT RESULTS            #
########################################

print("\n===== ORIGINAL PATCHES =====")
stats_ori = collect_stats(DIR_ORI)
for layer in LAYERS:
    print(layer, stats_ori[layer])

print("\n===== ADAPTIVE PATCHES =====")
stats_ada = collect_stats(DIR_ADA)
for layer in LAYERS:
    print(layer, stats_ada[layer])


########################################
#          NORMALIZATION (NEW!)        #
########################################

full_pixels = 224 * 224 * 3

ori_mean = [stats_ori[layer]["mean"] for layer in LAYERS]
ada_mean = [stats_ada[layer]["mean"] for layer in LAYERS]
ori_min = [stats_ori[layer]["min"] for layer in LAYERS]
ada_min = [stats_ada[layer]["min"] for layer in LAYERS]
ori_max = [stats_ori[layer]["max"] for layer in LAYERS]
ada_max = [stats_ada[layer]["max"] for layer in LAYERS]

ori_mean_norm = [v / full_pixels for v in ori_mean]
ada_mean_norm = [v / full_pixels for v in ada_mean]
ori_min_norm = [v / full_pixels for v in ori_min]
ada_min_norm = [v / full_pixels for v in ada_min]
ori_max_norm = [v / full_pixels for v in ori_max]
ada_max_norm = [v / full_pixels for v in ada_max]


########################################
#           BEAUTIFIED PLOT            #
########################################

x = np.arange(len(LAYERS))
x_shift = x + 0.07  # 🔥 错位，让 Adaptive 和 Original 分开

plt.figure(figsize=(14, 7))

# 1. Original —— 蓝色、细一点
plt.plot(x, ori_mean_norm, marker='o', linewidth=1.3, color="#1f77b4",
         label="Original Mean Patch Size (%)")
plt.fill_between(x, ori_min_norm, ori_max_norm, alpha=0.30, color="#1f77b4",
                 label="Original Min-Max (%)")

# 2. Adaptive —— 橙色、加粗、偏移
plt.plot(x_shift, ada_mean_norm, marker='o', linewidth=3.2, color="#ff7f0e",
         label="Adaptive Mean Patch Size (%)")
plt.fill_between(x_shift, ada_min_norm, ada_max_norm, alpha=0.15, color="#ff7f0e",
                 label="Adaptive Min-Max (%)")

plt.xticks(x, LAYERS)
plt.xlabel("Layer")
plt.ylabel("Patch Size (Fraction of Image)")
plt.title(f"Experiment 1: Granularity Trend ({CLASS_NAME}) [Adaptive Highlighted]")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()

plt.tight_layout()
plt.savefig(f"granularity_trend_highlight_adaptive_{CLASS_NAME}.png", dpi=300)
plt.show()

print(f"\n✨ Saved: granularity_trend_highlight_adaptive_{CLASS_NAME}.png\n")
