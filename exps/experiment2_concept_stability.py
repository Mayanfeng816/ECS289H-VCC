import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

########################################
#              配置区                  #
########################################

ROOT = "./outputs"
CLASS_LIST_FILE = "imagenet_names.txt"
CLASS_PREFIX = "R50_"

# 和 Exp1 一样：你可以改这个数字来控制用前多少个类别
NUM_CLASSES_TO_USE = 2

# 如果 layer 名固定也可以手写；也可以从 args 里读
LAYERS = ["layer1", "layer2", "layer3", "layer4"]

PVAL_ALPHA = 0.05  # 判断 TCAV 是否显著的阈值


########################################
#           工具函数：读类别名          #
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
#        工具函数：解析 CAV 文件        #
########################################

def parse_cav_file(path):
    """
    解析 CAV_ace_results.txt，返回：
      layer_to_accs: { "layer1": [acc1, acc2, ...], ... }
    """
    layer_to_accs = defaultdict(list)

    if not os.path.isfile(path):
        return None

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if (not line) or line.startswith("---"):
                continue

            # 例如：layer1:house_finch_concept1:0.90909...
            parts = line.split(":")
            if len(parts) < 3:
                continue
            layer = parts[0].strip()
            try:
                acc = float(parts[-1])
            except ValueError:
                continue

            layer_to_accs[layer].append(acc)

    return layer_to_accs


########################################
#       工具函数：解析 TCAV 文件        #
########################################

def parse_tcav_file(path):
    """
    解析 TCAV_ace_results.txt，返回：
      layer_to_scores: { "layer1": [score1, ...], ... }
      layer_to_pvals:  { "layer1": [pval1,  ...], ... }
    """
    layer_to_scores = defaultdict(list)
    layer_to_pvals = defaultdict(list)

    if not os.path.isfile(path):
        return None, None

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if (not line) or line.startswith("---"):
                continue

            # 例如：layer1:house_finch_concept1:0.576,0.371063...
            parts = line.split(":")
            if len(parts) < 3:
                continue
            layer = parts[0].strip()
            tail = parts[-1]  # "0.576,0.37106..."
            try:
                score_str, pval_str = tail.split(",")
                score = float(score_str)
                pval = float(pval_str)
            except ValueError:
                continue

            layer_to_scores[layer].append(score)
            layer_to_pvals[layer].append(pval)

    return layer_to_scores, layer_to_pvals


########################################
#   按“类别 → 层”计算统计量（单类）     #
########################################

def per_class_stats_cav(layer_to_accs):
    """
    输入单个类的 layer_to_accs（dict），输出：
      mean_acc[layer], std_acc[layer]
    """
    mean_acc = {}
    std_acc = {}

    for layer in LAYERS:
        vals = np.array(layer_to_accs.get(layer, []), dtype=float)
        if vals.size == 0:
            mean_acc[layer] = np.nan
            std_acc[layer] = np.nan
        else:
            mean_acc[layer] = float(vals.mean())
            std_acc[layer] = float(vals.std())
    return mean_acc, std_acc


def per_class_stats_tcav(layer_to_scores, layer_to_pvals):
    """
    输入单个类的 TCAV layer_to_scores/pvals，输出：
      mean_score[layer], std_score[layer], frac_sig[layer]
    其中 frac_sig 为 p < PVAL_ALPHA 的概念比例。
    """
    mean_score = {}
    std_score = {}
    frac_sig = {}

    for layer in LAYERS:
        scores = np.array(layer_to_scores.get(layer, []), dtype=float)
        pvals = np.array(layer_to_pvals.get(layer, []), dtype=float)

        if scores.size == 0:
            mean_score[layer] = np.nan
            std_score[layer] = np.nan
            frac_sig[layer] = np.nan
        else:
            mean_score[layer] = float(scores.mean())
            std_score[layer] = float(scores.std())
            if pvals.size > 0:
                frac_sig[layer] = float((pvals < PVAL_ALPHA).mean())
            else:
                frac_sig[layer] = np.nan

    return mean_score, std_score, frac_sig


########################################
#          主流程：遍历类别             #
########################################

def main():
    all_classes = load_class_names(CLASS_LIST_FILE)
    print(f"Total classes in list: {len(all_classes)}")

    classes_to_use = all_classes[:NUM_CLASSES_TO_USE]
    print(f"Using first {len(classes_to_use)} classes: {classes_to_use}")

    # 聚合：在“类别维度”上做平均
    # 结构：{layer: [per-class-value, per-class-value, ...]}
    agg = {
        "ori_cav_mean": {layer: [] for layer in LAYERS},
        "ada_cav_mean": {layer: [] for layer in LAYERS},
        "ori_tcav_mean": {layer: [] for layer in LAYERS},
        "ada_tcav_mean": {layer: [] for layer in LAYERS},
        "ori_tcav_frac": {layer: [] for layer in LAYERS},
        "ada_tcav_frac": {layer: [] for layer in LAYERS},
    }

    used_classes = []

    for cname in classes_to_use:
        ori_root = os.path.join(ROOT, "VCC_original", CLASS_PREFIX + cname, "results_summaries")
        ada_root = os.path.join(ROOT, "VCC_adaptive", CLASS_PREFIX + cname, "results_summaries")

        cav_ori_path = os.path.join(ori_root, "CAV_ace_results.txt")
        cav_ada_path = os.path.join(ada_root, "CAV_ace_results.txt")
        tcav_ori_path = os.path.join(ori_root, "TCAV_ace_results.txt")
        tcav_ada_path = os.path.join(ada_root, "TCAV_ace_results.txt")

        if not (os.path.isfile(cav_ori_path) and os.path.isfile(cav_ada_path)
                and os.path.isfile(tcav_ori_path) and os.path.isfile(tcav_ada_path)):
            print(f"[WARN] Missing files for class {cname}, skip.")
            continue

        # 解析 CAV
        cav_ori = parse_cav_file(cav_ori_path)
        cav_ada = parse_cav_file(cav_ada_path)
        if cav_ori is None or cav_ada is None:
            print(f"[WARN] Failed to parse CAV for {cname}, skip.")
            continue

        ori_cav_mean, _ = per_class_stats_cav(cav_ori)
        ada_cav_mean, _ = per_class_stats_cav(cav_ada)

        # 解析 TCAV
        tscore_ori, tp_ori = parse_tcav_file(tcav_ori_path)
        tscore_ada, tp_ada = parse_tcav_file(tcav_ada_path)
        if tscore_ori is None or tscore_ada is None:
            print(f"[WARN] Failed to parse TCAV for {cname}, skip.")
            continue

        ori_tcav_mean, _, ori_tcav_frac = per_class_stats_tcav(tscore_ori, tp_ori)
        ada_tcav_mean, _, ada_tcav_frac = per_class_stats_tcav(tscore_ada, tp_ada)

        # 聚合
        used_classes.append(cname)
        for layer in LAYERS:
            agg["ori_cav_mean"][layer].append(ori_cav_mean[layer])
            agg["ada_cav_mean"][layer].append(ada_cav_mean[layer])
            agg["ori_tcav_mean"][layer].append(ori_tcav_mean[layer])
            agg["ada_tcav_mean"][layer].append(ada_tcav_mean[layer])
            agg["ori_tcav_frac"][layer].append(ori_tcav_frac[layer])
            agg["ada_tcav_frac"][layer].append(ada_tcav_frac[layer])

    print(f"\nEffective classes used: {len(used_classes)}")
    print("Used classes:", used_classes)

    if not used_classes:
        print("No valid classes found. Exit.")
        return

    # 把 NaN 去掉
    def clean(arr):
        arr = np.array(arr, dtype=float)
        return arr[~np.isnan(arr)]

    # 在“类别维度”上做 mean / std
    x = np.arange(len(LAYERS))
    ori_cav_mean_layers = []
    ada_cav_mean_layers = []
    ori_tcav_mean_layers = []
    ada_tcav_mean_layers = []
    ori_tcav_frac_layers = []
    ada_tcav_frac_layers = []

    for layer in LAYERS:
        ori_cav_vals = clean(agg["ori_cav_mean"][layer])
        ada_cav_vals = clean(agg["ada_cav_mean"][layer])
        ori_tcav_vals = clean(agg["ori_tcav_mean"][layer])
        ada_tcav_vals = clean(agg["ada_tcav_mean"][layer])
        ori_frac_vals = clean(agg["ori_tcav_frac"][layer])
        ada_frac_vals = clean(agg["ada_tcav_frac"][layer])

        ori_cav_mean_layers.append(ori_cav_vals.mean() if ori_cav_vals.size > 0 else np.nan)
        ada_cav_mean_layers.append(ada_cav_vals.mean() if ada_cav_vals.size > 0 else np.nan)
        ori_tcav_mean_layers.append(ori_tcav_vals.mean() if ori_tcav_vals.size > 0 else np.nan)
        ada_tcav_mean_layers.append(ada_tcav_vals.mean() if ada_tcav_vals.size > 0 else np.nan)
        ori_tcav_frac_layers.append(ori_frac_vals.mean() if ori_frac_vals.size > 0 else np.nan)
        ada_tcav_frac_layers.append(ada_frac_vals.mean() if ada_frac_vals.size > 0 else np.nan)

    ########################################
    #               画图                    #
    ########################################

    x_shift = x + 0.05

    plt.figure(figsize=(12, 8))

    # 子图1：CAV mean accuracy
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(x, ori_cav_mean_layers, marker="o", color="#1f77b4",
             label="Original CAV mean acc")
    ax1.plot(x_shift, ada_cav_mean_layers, marker="o", color="#ff7f0e",
             label="Adaptive CAV mean acc")
    ax1.set_xticks(x)
    ax1.set_xticklabels(LAYERS)
    ax1.set_ylabel("CAV accuracy")
    ax1.set_title(f"Experiment 2: Concept Stability (N={len(used_classes)} classes)")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    # 子图2：TCAV mean score & significant fraction
    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(x, ori_tcav_mean_layers, marker="o", color="#1f77b4",
             label="Original TCAV mean score")
    ax2.plot(x_shift, ada_tcav_mean_layers, marker="o", color="#ff7f0e",
             label="Adaptive TCAV mean score")
    ax2.set_xticks(x)
    ax2.set_xticklabels(LAYERS)
    ax2.set_ylabel("TCAV score")
    ax2.grid(True, linestyle="--", alpha=0.5)

    # 右轴：显著概念比例
    ax3 = ax2.twinx()
    ax3.plot(x, ori_tcav_frac_layers, marker="s", linestyle="--", color="#1f77b4",
             label="Original frac(p<0.05)")
    ax3.plot(x_shift, ada_tcav_frac_layers, marker="s", linestyle="--", color="#ff7f0e",
             label="Adaptive frac(p<0.05)")
    ax3.set_ylabel("Fraction of significant concepts")

    # 合并图例
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax2.legend(lines2 + lines3, labels2 + labels3, loc="lower left")

    plt.tight_layout()
    out_name = f"exp2_concept_stability_top{NUM_CLASSES_TO_USE}.png"
    plt.savefig("./exp_outputs/"+out_name, dpi=300)
    plt.show()

    print(f"\n✨ Figure saved as {out_name}\n")

    ########################################
    #        新增图：CAV Accuracy 斜率图
    ########################################

    # 计算斜率（相邻层 acc 差分）
    def compute_slopes(arr):
        arr = np.array(arr, dtype=float)
        return arr[1:] - arr[:-1]   # length = len(LAYERS)-1

    ori_slopes = compute_slopes(ori_cav_mean_layers)
    ada_slopes = compute_slopes(ada_cav_mean_layers)

    slope_x = np.arange(len(LAYERS) - 1)
    slope_labels = [f"{LAYERS[i]}→{LAYERS[i+1]}" for i in range(len(LAYERS)-1)]

    plt.figure(figsize=(10, 5))
    plt.plot(slope_x, ori_slopes, marker="o", linewidth=2, color="#1f77b4",
             label="Original CAV slope")
    plt.plot(slope_x, ada_slopes, marker="o", linewidth=2, color="#ff7f0e",
             label="Adaptive CAV slope")

    plt.axhline(0, color="gray", linestyle="--", alpha=0.5)

    plt.xticks(slope_x, slope_labels)
    plt.ylabel("Slope of CAV Accuracy (Δ acc)")
    plt.title(f"Experiment 2 (N={len(used_classes)}): CAV Accuracy Slope per Layer")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    out_name_slope = f"exp2_cav_slope_top{NUM_CLASSES_TO_USE}.png"
    plt.tight_layout()
    plt.savefig("./exp_outputs/"+out_name_slope, dpi=300)
    plt.show()

    print(f"✨ CAV slope plot saved as {out_name_slope}")

    # ================================
    #   Experiment 2 (Smoothness): Second Difference Bar Chart
    # ================================

    def compute_second_diff(values):
        """ values: per-layer CAV accuracy mean (list of length L) """
        slopes = np.diff(values)  # first difference
        second_diff = np.diff(slopes)  # second difference
        return np.sum(np.abs(second_diff)), slopes, second_diff

    # 计算 Original 与 Adaptive 的二阶差
    orig_smoothness, orig_slopes, orig_second = compute_second_diff(ori_cav_mean_layers)
    ada_smoothness, ada_slopes, ada_second = compute_second_diff(ada_cav_mean_layers)

    # 绘制二阶差分柱状图
    plt.figure(figsize=(7, 5))
    plt.bar(["Original", "Adaptive"],
            [orig_smoothness, ada_smoothness],
            color=["#1f77b4", "#ff7f0e"])

    plt.ylabel("Second-order Smoothness (Σ |Δ² accuracy|)")
    plt.title(f"Experiment 2: Second-order Smoothness (N={NUM_CLASSES_TO_USE})")

    for i, v in enumerate([orig_smoothness, ada_smoothness]):
        plt.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=12)

    out_name_smoothness = f"exp2_second_order_top{NUM_CLASSES_TO_USE}.png"
    plt.tight_layout()
    plt.savefig("./exp_outputs/"+out_name_smoothness, dpi=300)
    plt.show()




if __name__ == "__main__":
    main()
