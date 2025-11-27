import os
import numpy as np

CLASS_NAME = "R50_house_finch"
ROOT = "./outputs"

DIR_ORI = os.path.join(ROOT, "VCC_original", CLASS_NAME, "dataset", "images")
DIR_ADA = os.path.join(ROOT, "VCC_adaptive", CLASS_NAME, "dataset", "images")

LAYERS = ["layer1", "layer2", "layer3", "layer4"]

def collect_stats(version_dir):
    layer_stats = {}

    for layer in LAYERS:
        layer_path = os.path.join(version_dir, layer)
        sizes = []

        if not os.path.exists(layer_path):
            print(f"[ERROR] Missing: {layer_path}")
            layer_stats[layer] = {"count": 0}
            continue

        for fname in os.listdir(layer_path):
            if fname.endswith(".npy"):
                path = os.path.join(layer_path, fname)
                arr = np.load(path)
                h, w = arr.shape[0], arr.shape[1]
                sizes.append(h * w)

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

print("===== ORIGINAL VCC =====")
ori_stats = collect_stats(DIR_ORI)
for layer, s in ori_stats.items():
    print(layer, s)

print("\n===== ADAPTIVE VCC =====")
ada_stats = collect_stats(DIR_ADA)
for layer, s in ada_stats.items():
    print(layer, s)
