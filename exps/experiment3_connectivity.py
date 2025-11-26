import os
import pickle
from typing import Dict, Any, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


########################################
#             用户配置                 #
########################################

ROOT = "./outputs"
CLASS_LIST_FILE = "imagenet_names.txt"
CLASS_PREFIX = "R50_"

# 想用前多少个类（会自动过滤掉没有结果的类）
NUM_CLASSES_TO_USE = 2

# 边权稠密度阈值：|w| > THRESH 视为“有效连接”
EDGE_DENSITY_THRESH = 1e-6


########################################
#       工具函数：读取类别列表         #
########################################

def load_class_names(path: str) -> List[str]:
    classes = []
    with open(path, "r") as f:
        for line in f:
            name = line.strip()
            if name:
                classes.append(name)
    return classes


########################################
#     工具函数：flatten edge weights   #
########################################

def _flatten_numeric(x: Any, buf: List[float]):
    """递归收集任意结构中的数值 / ndarray 到 buf 里。"""
    if x is None:
        return
    # numpy 数组
    if isinstance(x, np.ndarray):
        buf.extend(x.astype(float).ravel().tolist())
    # 标量
    elif isinstance(x, (int, float, np.number)):
        buf.append(float(x))
    # dict
    elif isinstance(x, dict):
        for v in x.values():
            _flatten_numeric(v, buf)
    # list / tuple
    elif isinstance(x, (list, tuple)):
        for v in x:
            _flatten_numeric(v, buf)
    # 其他类型忽略


def flatten_edges_for_layer(edge_obj: Any) -> np.ndarray:
    buf: List[float] = []
    _flatten_numeric(edge_obj, buf)
    return np.array(buf, dtype=float)


########################################
#     针对单个类 & 单个版本的统计       #
########################################

def extract_layer_metrics_from_cd(cd_obj) -> Tuple[List[str], Dict[str, float], Dict[str, float]]:
    """
    从 ConceptDiscovery 对象中提取每层：
      - mean_abs_weight[layer]
      - density[layer]
    返回：
      (layers, mean_abs, density)
    """
    if not hasattr(cd_obj, "edge_weights"):
        raise ValueError("cd object has no attribute 'edge_weights'.")

    edge_weights = cd_obj.edge_weights  # 一般是 dict
    # bottlenecks：layer 顺序
    if hasattr(cd_obj, "bottlenecks"):
        layers = list(cd_obj.bottlenecks)
    else:
        # 兜底：用 edge_weights 的 key 顺序
        if isinstance(edge_weights, dict):
            layers = list(edge_weights.keys())
        else:
            layers = ["layer1"]

    mean_abs: Dict[str, float] = {}
    density: Dict[str, float] = {}

    for layer in layers:
        # 取该层 edge 结构
        if isinstance(edge_weights, dict):
            layer_edges_obj = edge_weights.get(layer, None)
        else:
            # 如果 edge_weights 不是 dict，直接拿整体
            layer_edges_obj = edge_weights

        edges = flatten_edges_for_layer(layer_edges_obj)
        if edges.size == 0:
            mean_abs[layer] = np.nan
            density[layer] = np.nan
        else:
            abs_edges = np.abs(edges)
            mean_abs[layer] = float(abs_edges.mean())
            density[layer] = float((abs_edges > EDGE_DENSITY_THRESH).mean())

    return layers, mean_abs, density


########################################
#            主流程：Exp3              #
########################################

def main():
    all_classes = load_class_names(CLASS_LIST_FILE)
    print(f"Total classes in imagenet_names.txt: {len(all_classes)}")

    # 选前 N 个
    raw_selected = all_classes[:NUM_CLASSES_TO_USE]
    print(f"Requested first {len(raw_selected)} classes: {raw_selected}")

    # 真正有效的类（original & adaptive 都有 cd.pkl）
    valid_classes: List[str] = []
    for cname in raw_selected:
        cname_full = CLASS_PREFIX + cname
        ori_cd_path = os.path.join(ROOT, "VCC_original", cname_full, "cd.pkl")
        ada_cd_path = os.path.join(ROOT, "VCC_adaptive", cname_full, "cd.pkl")
        if os.path.isfile(ori_cd_path) and os.path.isfile(ada_cd_path):
            valid_classes.append(cname)
        else:
            print(f"[WARN] Missing cd.pkl for class {cname_full}, skip.")

    if not valid_classes:
        print("No valid classes with both original & adaptive cd.pkl. Exit.")
        return

    print(f"\nEffective classes used for Exp3: {len(valid_classes)}")
    print("Used classes:", valid_classes, "\n")

    # 聚合结构： {layer: [per-class-value, ...]}
    agg_ori_mean_abs: Dict[str, List[float]] = {}
    agg_ada_mean_abs: Dict[str, List[float]] = {}
    agg_ori_density: Dict[str, List[float]] = {}
    agg_ada_density: Dict[str, List[float]] = {}

    # 决定统一的 layer 顺序：用第一个类的 bottlenecks
    canonical_layers: List[str] = []

    for idx, cname in enumerate(valid_classes):
        cname_full = CLASS_PREFIX + cname
        ori_cd_path = os.path.join(ROOT, "VCC_original", cname_full, "cd.pkl")
        ada_cd_path = os.path.join(ROOT, "VCC_adaptive", cname_full, "cd.pkl")

        # 读取 cd.pkl
        with open(ori_cd_path, "rb") as f:
            cd_ori = pickle.load(f)
        with open(ada_cd_path, "rb") as f:
            cd_ada = pickle.load(f)

        # 取每层 metric
        layers_ori, mean_abs_ori, density_ori = extract_layer_metrics_from_cd(cd_ori)
        layers_ada, mean_abs_ada, density_ada = extract_layer_metrics_from_cd(cd_ada)

        # 统一 layer 顺序：以第一个类的 ori 为 canonical
        if idx == 0:
            canonical_layers = list(layers_ori)
            print("Canonical layer order:", canonical_layers)
        else:
            # 如果后续类的 layer 数不匹配，就对齐 canonical
            pass

        # 初始化 agg 字典
        for layer in canonical_layers:
            agg_ori_mean_abs.setdefault(layer, [])
            agg_ada_mean_abs.setdefault(layer, [])
            agg_ori_density.setdefault(layer, [])
            agg_ada_density.setdefault(layer, [])

        # 填充数据（不存在的层用 NaN）
        for layer in canonical_layers:
            ori_val_ma = mean_abs_ori.get(layer, np.nan)
            ada_val_ma = mean_abs_ada.get(layer, np.nan)
            ori_val_den = density_ori.get(layer, np.nan)
            ada_val_den = density_ada.get(layer, np.nan)

            agg_ori_mean_abs[layer].append(ori_val_ma)
            agg_ada_mean_abs[layer].append(ada_val_ma)
            agg_ori_density[layer].append(ori_val_den)
            agg_ada_density[layer].append(ada_val_den)

    # 在“类别维度”上做平均 / 标准差
    def clean(arr: List[float]) -> np.ndarray:
        a = np.array(arr, dtype=float)
        return a[~np.isnan(a)]

    ori_mean_abs_layers = []
    ada_mean_abs_layers = []
    ori_density_layers = []
    ada_density_layers = []

    ori_mean_abs_std = []
    ada_mean_abs_std = []
    ori_density_std = []
    ada_density_std = []

    for layer in canonical_layers:
        o_ma = clean(agg_ori_mean_abs[layer])
        a_ma = clean(agg_ada_mean_abs[layer])
        o_den = clean(agg_ori_density[layer])
        a_den = clean(agg_ada_density[layer])

        ori_mean_abs_layers.append(o_ma.mean() if o_ma.size > 0 else np.nan)
        ada_mean_abs_layers.append(a_ma.mean() if a_ma.size > 0 else np.nan)
        ori_density_layers.append(o_den.mean() if o_den.size > 0 else np.nan)
        ada_density_layers.append(a_den.mean() if a_den.size > 0 else np.nan)

        ori_mean_abs_std.append(o_ma.std() if o_ma.size > 0 else np.nan)
        ada_mean_abs_std.append(a_ma.std() if a_ma.size > 0 else np.nan)
        ori_density_std.append(o_den.std() if o_den.size > 0 else np.nan)
        ada_density_std.append(a_den.std() if a_den.size > 0 else np.nan)

    ########################################
    #               绘图                    #
    ########################################

    x = np.arange(len(canonical_layers))
    x_shift = x + 0.05

    plt.figure(figsize=(12, 8))

    # 子图1：mean |edge weight|
    ax1 = plt.subplot(2, 1, 1)
    ax1.errorbar(
        x,
        ori_mean_abs_layers,
        yerr=ori_mean_abs_std,
        marker="o",
        linestyle="-",
        color="#1f77b4",
        label="Original mean |edge weight|",
    )
    ax1.errorbar(
        x_shift,
        ada_mean_abs_layers,
        yerr=ada_mean_abs_std,
        marker="o",
        linestyle="-",
        color="#ff7f0e",
        label="Adaptive mean |edge weight|",
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(canonical_layers)
    ax1.set_ylabel("Mean |edge weight|")
    ax1.set_title(f"Experiment 3: Connectivity (N={len(valid_classes)} classes)")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    # 子图2：edge density
    ax2 = plt.subplot(2, 1, 2)
    ax2.errorbar(
        x,
        ori_density_layers,
        yerr=ori_density_std,
        marker="o",
        linestyle="-",
        color="#1f77b4",
        label=f"Original edge density (|w|>{EDGE_DENSITY_THRESH})",
    )
    ax2.errorbar(
        x_shift,
        ada_density_layers,
        yerr=ada_density_std,
        marker="o",
        linestyle="-",
        color="#ff7f0e",
        label=f"Adaptive edge density (|w|>{EDGE_DENSITY_THRESH})",
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(canonical_layers)
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Edge density")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    out_name = f"exp3_connectivity_top{NUM_CLASSES_TO_USE}.png"
    plt.savefig("./exp_outputs/"+out_name, dpi=300)
    plt.show()

    ###############################################################################
    #  Additional Plot 1: Slope of mean |edge weight| across layers
    ###############################################################################

    # 计算斜率（差分）
    ori_slopes = np.diff(ori_mean_abs_layers)
    ada_slopes = np.diff(ada_mean_abs_layers)

    slope_x_labels = [f"{canonical_layers[i]}→{canonical_layers[i+1]}" for i in range(len(canonical_layers)-1)]
    x_s = np.arange(len(slope_x_labels))

    plt.figure(figsize=(12,6))
    plt.plot(x_s, ori_slopes, marker="o", label="Original slope", color="#1f77b4")
    plt.plot(x_s, ada_slopes, marker="o", label="Adaptive slope", color="#ff7f0e")
    plt.axhline(0, linestyle="--", color="gray", alpha=0.6)

    plt.xticks(x_s, slope_x_labels)
    plt.ylabel("Slope Δ mean |edge weight|")
    plt.title(f"Experiment 3: Slope of Mean |edge weight| (N={len(valid_classes)})")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    plt.savefig(f"./exp_outputs/exp3_slope_top{NUM_CLASSES_TO_USE}.png", dpi=300)
    plt.show()

    print("\n✨ Saved slope plot:", f"exp3_slope_top{NUM_CLASSES_TO_USE}.png")


    ###############################################################################
    #  Additional Plot 2: Second-order smoothness (area of |Δ² accuracy|)
    ###############################################################################

    # 二阶差分（平滑度指标，越小越平滑）
    ori_second = np.abs(np.diff(ori_slopes)).sum()
    ada_second = np.abs(np.diff(ada_slopes)).sum()

    plt.figure(figsize=(8,6))
    bars = plt.bar(["Original", "Adaptive"], [ori_second, ada_second],
                color=["#1f77b4", "#ff7f0e"], alpha=0.85)

    # 在柱子上显示数值
    for bar, val in zip(bars, [ori_second, ada_second]):
        plt.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                f"{val:.3f}", ha="center", fontsize=12)

    plt.ylabel("Second-order smoothness (Σ |Δ² mean |edge||)")
    plt.title(f"Experiment 3: Second-order Smoothness (N={len(valid_classes)})")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    plt.savefig(f"./exp_outputs/exp3_second_order_top{NUM_CLASSES_TO_USE}.png", dpi=300)
    plt.show()

    print("✨ Saved second-order smoothness plot:",
        f"exp3_second_order_top{NUM_CLASSES_TO_USE}.png")


    print(f"\n✨ Experiment 3 figure saved as {out_name}")
    print("Canonical layers:", canonical_layers)
    print("Mean |edge| (Original):", ori_mean_abs_layers)
    print("Mean |edge| (Adaptive):", ada_mean_abs_layers)
    print("Density (Original):", ori_density_layers)
    print("Density (Adaptive):", ada_density_layers)


if __name__ == "__main__":
    main()
