import os
import numpy as np
import argparse


# -----------------------------
# Config
# -----------------------------
CLASS_LIST_FILE = "./tools/imagenet_names.txt"  # 你已有的类别名单
ORI_ROOT = "./outputs/VCC_original"
ADA_ROOT = "./outputs/VCC_adaptive"
CACHE_DIR = "./cache/exp1"

LAYERS = ["layer1", "layer2", "layer3", "layer4"]


# -----------------------------
# 工具：确保目录存在
# -----------------------------
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# -----------------------------
# 从 patches/ 计算 granularity
# -----------------------------
def calc_layer_stats(patch_dir):
    stats = []

    for layer in LAYERS:
        layer_path = os.path.join(patch_dir, layer)
        sizes = []

        if not os.path.exists(layer_path):
            print(f"[Warning] Layer folder missing: {layer_path}")
            stats.append(np.nan)
            continue

        for fname in os.listdir(layer_path):
            if fname.endswith(".npy"):
                arr = np.load(os.path.join(layer_path, fname))
                seg_pixels = np.sum(arr > 0)
                sizes.append(seg_pixels)

        if len(sizes) == 0:
            stats.append(np.nan)
        else:
            stats.append(np.mean(sizes))

    return stats  # list of 4 layers


# -----------------------------
# 主函数：处理一个 class
# -----------------------------
def process_one_class(classname):
    ori_patch_dir = os.path.join(ORI_ROOT, classname, "dataset", "patches")
    ada_patch_dir = os.path.join(ADA_ROOT, classname, "dataset", "patches")

    if not (os.path.exists(ori_patch_dir) and os.path.exists(ada_patch_dir)):
        print(f"[Skip] Missing patch folder for {classname}")
        return

    print(f"Processing Class: {classname}")

    ori_means = calc_layer_stats(ori_patch_dir)
    ada_means = calc_layer_stats(ada_patch_dir)

    # 存缓存
    ensure_dir(CACHE_DIR)

    save_obj = {
        "class": classname,
        "layers": LAYERS,
        "ori_mean": ori_means,
        "ada_mean": ada_means
    }

    np.save(os.path.join(CACHE_DIR, f"{classname}.npy"),
            save_obj, allow_pickle=True)

    print(f"Saved cache → {CACHE_DIR}/{classname}.npy\n")


# -----------------------------
# 读取 class list
# -----------------------------
def load_class_list(n_limit=None):
    with open(CLASS_LIST_FILE, "r") as f:
        all_classes = [line.strip() for line in f.readlines()]

    if n_limit is not None:
        return all_classes[:n_limit]
    return all_classes


# -----------------------------
# main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_classes", type=int, default=10,
                        help="一次处理多少个类（按前 N 个）")
    args = parser.parse_args()

    classes = load_class_list(args.num_classes)

    for cname in classes:
        process_one_class("R50_"+cname)


if __name__ == "__main__":
    main()
