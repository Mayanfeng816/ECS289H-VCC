import os
import numpy as np
import matplotlib.pyplot as plt

"""
Experiment 1: Layerwise Granularity Trend
-----------------------------------------
Goal:
    Compare the segmentation granularity (patch size) across layers
    between Original VCC vs Adaptive VCC.

Required folder structure:

outputs/
    VCC_original/
        CLASSNAME/
            dataset/images/layer1/*.npy
            dataset/images/layer2/*.npy
            dataset/images/layer3/*.npy
            dataset/images/layer4/*.npy

    VCC_adaptive/
        CLASSNAME/
            dataset/images/layer1/*.npy
            dataset/images/layer2/*.npy
            dataset/images/layer3/*.npy
            dataset/images/layer4/*.npy
"""

############################
#      USER SETTINGS       #
############################

CLASS_NAME = "house_finch"   # 要分析的类别，例如 house_finch、zebra
OUTPUT_ROOT = "outputs"      # 整理好的 output 根目录

ORIGINAL_DIR = os.path.join(OUTPUT_ROOT, "VCC_original", CLASS_NAME)
ADAPTIVE_DIR = os.path.join(OUTPUT_ROOT, "VCC_adaptive", CLASS_NAME)

LAYERS = ["layer1", "layer2", "layer3", "layer4"]   # ResNet50 示例


############################
#   Helper: load .npy     #
############################
def load_patch_sizes(layer_folder):
    """
    Returns a list of pixel counts of all segmentation masks in this layer.
    Each .npy is a segmentation mask of shape (H, W, 3) or (H, W).
    """
    sizes = []
    for file in os.listdir(layer_folder):
        if file.endswith(".npy"):
            path = os.path.join(layer_folder, file)
            try:
                arr = np.load(path)
                sizes.append(arr.shape[0] * arr.shape[1])   # pixel count
            except:
                continue
    return sizes


############################
#  Compute layer curves    #
############################
def compute_stats(base_dir):
    """
    For each layer, compute (min, mean, max) patch size.
    """
    stats_min = []
    stats_mean = []
    stats_max = []

    for layer in LAYERS:
        layer_path = os.path.join(base_dir, "dataset", "images", layer)
        if not os.path.exists(layer_path):
            print(f"[WARNING] Missing layer folder: {layer_path}")
            stats_min.append(0)
            stats_mean.append(0)
            stats_max.append(0)
            continue

        sizes = load_patch_sizes(layer_path)
        if len(sizes) == 0:
            stats_min.append(0)
            stats_mean.append(0)
            stats_max.append(0)
        else:
            stats_min.append(np.min(sizes))
            stats_mean.append(np.mean(sizes))
            stats_max.append(np.max(sizes))

    return stats_min, stats_mean, stats_max


print("🔍 Computing Original VCC stats...")
orig_min, orig_mean, orig_max = compute_stats(ORIGINAL_DIR)

print("🔍 Computing Adaptive VCC stats...")
adapt_min, adapt_mean, adapt_max = compute_stats(ADAPTIVE_DIR)


############################
#      Plot curves         #
############################
plt.figure(figsize=(10, 6))

x = range(1, len(LAYERS)+1)

plt.plot(x, orig_mean, marker='o', label="Original - Mean Patch Size")
plt.plot(x, adapt_mean, marker='o', label="Adaptive - Mean Patch Size")

plt.fill_between(x, orig_min, orig_max, alpha=0.2, label="Original min-max")
plt.fill_between(x, adapt_min, adapt_max, alpha=0.2, label="Adaptive min-max")

plt.xticks(x, LAYERS)
plt.xlabel("Layer")
plt.ylabel("Patch Pixel Count")
plt.title(f"Experiment 1: Layerwise Granularity Trend ({CLASS_NAME})")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(f"granularity_trend_{CLASS_NAME}.png", dpi=300)
plt.show()

print("Experiment 1 completed. Figure saved.")
