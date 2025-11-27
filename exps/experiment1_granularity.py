import os
import numpy as np
import matplotlib.pyplot as plt

########################################
#             USER CONFIG              #
########################################

ROOT = "./outputs"

# 你的类别列表文件
CLASS_LIST_FILE = "imagenet_names.txt"

# 手动指定前 N 个类别
NUM_CLASSES_TO_USE = 2   # 🔥 你只需要改这里即可

# 如果你的文件夹叫 R50_house_finch / R50_stage，就用这个前缀
CLASS_PREFIX = "R50_"

# 层列表
LAYERS = ["layer1", "layer2", "layer3", "layer4"]

# 图像大小（用于归一化）
FULL_H, FULL_W, FULL_C = 224, 224, 3
FULL_PIXELS = FULL_H * FULL_W * FULL_C


########################################
#         工具函数：读类别列表         #
########################################

def load_class_names(path):
    classes = []
    with open(path, "r") as f:
        for line in f:
            name = line.strip()
            if name:
                classes.append(name)
    return classes


########################################
#       工具函数：单类统计 per-layer    #
########################################

def collect_stats_for_class(class_name):
    stats = {"original": {}, "adaptive": {}}

    for version, vroot in [("original", "VCC_original"), ("adaptive", "VCC_adaptive")]:
        base_dir = os.path.join(ROOT, vroot, CLASS_PREFIX + class_name, "dataset", "patches")

        if not os.path.isdir(base_dir):
            print(f"[WARN] Missing {version} dir: {base_dir}, skip.")
            return None

        for layer in LAYERS:
            layer_path = os.path.join(base_dir, layer)
            if not os.path.isdir(layer_path):
                print(f"[WARN] Missing layer: {layer_path}, skip.")
                return None

            sizes = []
            for fname in os.listdir(layer_path):
                if fname.endswith(".npy"):
                    arr = np.load(os.path.join(layer_path, fname))
                    seg_pixels = np.sum(arr > 0)
                    sizes.append(seg_pixels)

            if not sizes:
                print(f"[WARN] No patches in {layer_path}, skip class.")
                return None

            stats[version][layer] = {
                "count": len(sizes),
                "mean": float(np.mean(sizes)),
                "min": int(np.min(sizes)),
                "max": int(np.max(sizes)),
                "example_shape": arr.shape,
            }

    return stats


########################################
#       主流程：遍历 NUM_CLASSES_TO_USE #
########################################

all_classes = load_class_names(CLASS_LIST_FILE)
print(f"🔎 Total classes in list: {len(all_classes)}")

# 取前 N 个类别
classes_to_use = all_classes[:NUM_CLASSES_TO_USE]
print(f"👉 Using first {NUM_CLASSES_TO_USE} classes.\n")

ori_means_per_layer = {layer: [] for layer in LAYERS}
ada_means_per_layer = {layer: [] for layer in LAYERS}

used_classes = []

for cname in classes_to_use:
    cls_stats = collect_stats_for_class(cname)
    if cls_stats is None:
        continue

    used_classes.append(cname)

    for layer in LAYERS:
        ori_means_per_layer[layer].append(cls_stats["original"][layer]["mean"])
        ada_means_per_layer[layer].append(cls_stats["adaptive"][layer]["mean"])

print(f"✅ Effective classes used: {len(used_classes)}\n")
print(f"Used: {used_classes}\n")


########################################
#       在类别维度求 mean / std         #
########################################

def layerwise_mean_std(means_per_layer_dict):
    layer_mean = []
    layer_std = []
    for layer in LAYERS:
        vals = np.array(means_per_layer_dict[layer], dtype=float)
        layer_mean.append(np.mean(vals))
        layer_std.append(np.std(vals))
    return np.array(layer_mean), np.array(layer_std)


ori_mean_raw, ori_std_raw = layerwise_mean_std(ori_means_per_layer)
ada_mean_raw, ada_std_raw = layerwise_mean_std(ada_means_per_layer)

# Normalize
ori_mean = ori_mean_raw / FULL_PIXELS
ada_mean = ada_mean_raw / FULL_PIXELS
ori_std = ori_std_raw / FULL_PIXELS
ada_std = ada_std_raw / FULL_PIXELS


########################################
#                绘图                   #
########################################

x = np.arange(len(LAYERS))
x_shift = x + 0.05

plt.figure(figsize=(12, 7))

# Original
plt.plot(x, ori_mean, marker="o", linewidth=1.6, color="#1f77b4",
         label="Original (mean over classes)")
plt.fill_between(x, ori_mean - ori_std, ori_mean + ori_std,
                 color="#1f77b4", alpha=0.25, label="Original ±1 std")

# Adaptive
plt.plot(x_shift, ada_mean, marker="o", linewidth=2.8, color="#ff7f0e",
         label="Adaptive (mean over classes)")
plt.fill_between(x_shift, ada_mean - ada_std, ada_mean + ada_std,
                 color="#ff7f0e", alpha=0.18, label="Adaptive ±1 std")

plt.xticks(x, LAYERS)
plt.xlabel("Layer")
plt.ylabel("Normalized Patch Size (fraction of image)")
plt.title(f"Experiment 1 (Multi-class N={len(used_classes)}): Granularity Trend")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()

plt.tight_layout()
out_name = f"exp1_multiclass_trend_top{NUM_CLASSES_TO_USE}.png"
plt.savefig("./exp_outputs/"+out_name, dpi=300)
plt.show()

print(f"✨ Saved figure as {out_name}")
