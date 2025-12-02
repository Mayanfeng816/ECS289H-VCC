import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import json   # 新增
import csv
import os

def save_csv(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"[CSV Saved] {path}")
############################################################
# 工具：加载某个缓存目录
############################################################
def load_cache(cache_dir):
    data_list = []
    if not os.path.exists(cache_dir):
        print(f"[Error] Cache directory does not exist: {cache_dir}")
        return []

    for fname in os.listdir(cache_dir):
        path = os.path.join(cache_dir, fname)

        # 以前的 npy 缓存（Exp1 / 旧版 Exp2 / Exp3）
        if fname.endswith(".npy"):
            try:
                data = np.load(path, allow_pickle=True).item()
                data_list.append(data)
            except Exception as e:
                print(f"[Warning] Failed to load {path}: {e}")

        # 现在 Exp2_V2 存的 json
        elif fname.endswith(".json"):
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                data_list.append(data)
            except Exception as e:
                print(f"[Warning] Failed to load {path}: {e}")

    print(f"[Info] Loaded {len(data_list)} items from {cache_dir}")
    return data_list



############################################################
# EXP1 — Patch Granularity
############################################################
# def plot_exp1():
    cache_dir = "./cache/exp1"
    data_list = load_cache(cache_dir)
    if len(data_list) == 0:
        print("No Exp1 data found")
        return

    layers = data_list[0]["layers"]

    ori_all = np.array([d["ori_mean"] for d in data_list])
    ada_all = np.array([d["ada_mean"] for d in data_list])

    ori_mean = np.nanmean(ori_all, axis=0)
    ada_mean = np.nanmean(ada_all, axis=0)

    plt.figure(figsize=(8,6))
    plt.plot(layers, ori_mean, "-o", linewidth=2, label="Original")
    plt.plot(layers, ada_mean, "-o", linewidth=2, label="Adaptive")
    plt.title(f"Exp1: Patch Granularity (N={len(data_list)})")
    plt.ylabel("Mean segmentation size")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("exp1_final.png", dpi=300)
    plt.show()
    print("Saved: exp1_final.png")
def plot_exp1():
    cache_dir = "./cache/exp1"
    data_list = load_cache(cache_dir)
    if len(data_list) == 0:
        print("No Exp1 data found")
        return

    layers = data_list[0]["layers"]
    x = np.arange(len(layers))

    # gather per-class means
    ori_all = np.array([d["ori_mean"] for d in data_list])
    ada_all = np.array([d["ada_mean"] for d in data_list])

    # cross-class mean & std
    ori_mean = np.nanmean(ori_all, axis=0)
    ori_std  = np.nanstd(ori_all, axis=0)

    ada_mean = np.nanmean(ada_all, axis=0)
    ada_std  = np.nanstd(ada_all, axis=0)

    plt.figure(figsize=(12,7))

    # original
    plt.plot(x, ori_mean, marker='o', linewidth=1.6, color="#1f77b4",
             label="Original (mean over classes)")
    plt.fill_between(
        x, ori_mean - ori_std, ori_mean + ori_std,
        color="#1f77b4", alpha=0.25, label="Original ±1 std"
    )

    # adaptive
    x_shift = x + 0.05
    plt.plot(x_shift, ada_mean, marker='o', linewidth=2.8, color="#ff7f0e",
             label="Adaptive (mean over classes)")
    plt.fill_between(
        x_shift, ada_mean - ada_std, ada_mean + ada_std,
        color="#ff7f0e", alpha=0.18, label="Adaptive ±1 std"
    )

    plt.xticks(x, layers)
    plt.xlabel("Layer")
    plt.ylabel("Normalized Patch Size (fraction of image)")
    plt.title(f"Exp1: Patch Granularity Trend (N={len(data_list)})")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    plt.tight_layout()
    plt.savefig("./exp_outputs/exp1_final.png", dpi=300)
    plt.show()
    print("Saved: exp1_final.png")

    # ===== CSV Export (Exp1) =====
    csv_path = "./exp_outputs/csv/exp1_values.csv"
    rows = []
    for i, layer in enumerate(layers):
        rows.append([
            layer,
            ori_mean[i], ori_std[i],
            ada_mean[i], ada_std[i]
        ])
    save_csv(csv_path,
             ["Layer", "Ori_Mean", "Ori_Std", "Ada_Mean", "Ada_Std"],
             rows)


############################################################
# EXP2 — Concept Stability  (CAV/TCAV + slope + smoothness)
############################################################
def plot_exp2():
    cache_dir = "./cache/exp2"
    data_list = load_cache(cache_dir)
    if len(data_list) == 0:
        print("No Exp2 data found")
        return

    # 所有类共享的 layer 名
    layers = data_list[0]["layers"]
    L = len(layers)

    def stack(key):
        """把每个类的一维 list 堆成 (N, L) 的数组"""
        return np.array([d[key] for d in data_list], dtype=float)

    # 从缓存里取出 CAV / TCAV / 显著比例
    cav_ori  = stack("cav_ori")
    cav_ada  = stack("cav_ada")
    tcav_ori = stack("tcav_ori")
    tcav_ada = stack("tcav_ada")
    sig_ori  = stack("sig_ori")   # fraction of significant concepts
    sig_ada  = stack("sig_ada")

    # 跨类平均
    cav_ori_mean  = np.nanmean(cav_ori,  axis=0)
    cav_ada_mean  = np.nanmean(cav_ada,  axis=0)
    tcav_ori_mean = np.nanmean(tcav_ori, axis=0)
    tcav_ada_mean = np.nanmean(tcav_ada, axis=0)

    sig_ori_mean = np.nanmean(sig_ori, axis=0)
    sig_ada_mean = np.nanmean(sig_ada, axis=0)

    # ------------------- 主图：CAV + TCAV -------------------
    plt.figure(figsize=(12, 8))

    # 上：CAV ACE 准确率
    plt.subplot(2, 1, 1)
    plt.plot(layers, cav_ori_mean, "-o", linewidth=2, label="Original CAV acc")
    plt.plot(layers, cav_ada_mean, "-o", linewidth=2, label="Adaptive CAV acc")
    plt.ylabel("CAV ACE accuracy")
    plt.title(f"Experiment 2: Concept Stability (N={len(data_list)} classes)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="best")

    # 下：TCAV 分数 + 显著比例 (右轴)
    ax1 = plt.subplot(2, 1, 2)
    ln1 = ax1.plot(layers, tcav_ori_mean, "-o", linewidth=2,
                   label="Original TCAV score")
    ln2 = ax1.plot(layers, tcav_ada_mean, "-o", linewidth=2,
                   label="Adaptive TCAV score")
    ax1.set_ylabel("TCAV score")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    ln3 = ax2.plot(layers, sig_ori_mean, "--s", linewidth=1.8,
                   label="Original frac(p<0.05)")
    ln4 = ax2.plot(layers, sig_ada_mean, "--s", linewidth=1.8,
                   label="Adaptive frac(p<0.05)")
    ax2.set_ylabel("Fraction of significant concepts")

    # 合并双轴图例
    lines = ln1 + ln2 + ln3 + ln4
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left")

    plt.xlabel("Layer")
    plt.tight_layout()
    plt.savefig("./exp_outputs/exp2_final.png", dpi=300)
    plt.show()
    print("Saved: exp2_final.png")
    # ===== CSV Export: Exp2 Main =====
    csv_path = "./exp_outputs/csv/exp2_values_main.csv"
    rows = []
    for i, layer in enumerate(layers):
        rows.append([
            layer,
            cav_ori_mean[i], cav_ada_mean[i],
            tcav_ori_mean[i], tcav_ada_mean[i],
            sig_ori_mean[i], sig_ada_mean[i]
        ])
    save_csv(csv_path,
             ["Layer",
              "CAV_Ori", "CAV_Ada",
              "TCAV_Ori", "TCAV_Ada",
              "Sig_Ori", "Sig_Ada"],
             rows)

    # ------------------- slope 图（ΔCAV per layer） -------------------
    # 对每个类先算一阶差分，再跨类平均
    cav_ori_slope = np.diff(cav_ori, axis=1)  # (N, L-1)
    cav_ada_slope = np.diff(cav_ada, axis=1)

    cav_ori_slope_mean = np.nanmean(cav_ori_slope, axis=0)
    cav_ada_slope_mean = np.nanmean(cav_ada_slope, axis=0)

    slope_x = [f"{layers[i]}→{layers[i+1]}" for i in range(L-1)]

    plt.figure(figsize=(8, 6))
    plt.plot(slope_x, cav_ori_slope_mean, "-o", linewidth=2, label="Original")
    plt.plot(slope_x, cav_ada_slope_mean, "-o", linewidth=2, label="Adaptive")
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.ylabel("Slope of CAV accuracy (Δ per layer)")
    plt.title("Experiment 2: CAV Accuracy Slope per Layer")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("./exp_outputs/exp2_slope_final.png", dpi=300)
    plt.show()
    print("Saved: exp2_slope_final.png")

    csv_path = "./exp_outputs/csv/exp2_slope.csv"
    slope_labels = [f"{layers[i]}->{layers[i+1]}" for i in range(L-1)]
    rows = []
    for i, name in enumerate(slope_labels):
        rows.append([name, cav_ori_slope_mean[i], cav_ada_slope_mean[i]])
    save_csv(csv_path,
             ["Layer_Transition", "Slope_Ori", "Slope_Ada"],
             rows)

    # ------------------- 二阶差分平滑度柱状图 -------------------
    # 对每个类：先求二阶差分，再对绝对值求和，得到「不平滑度」标量
    if L >= 3:
        cav_ori_second = np.diff(cav_ori, n=2, axis=1)   # (N, L-2)
        cav_ada_second = np.diff(cav_ada, n=2, axis=1)

        ori_second_per_class = np.sum(np.abs(cav_ori_second), axis=1)
        ada_second_per_class = np.sum(np.abs(cav_ada_second), axis=1)

        ori_second_mean = float(np.nanmean(ori_second_per_class))
        ada_second_mean = float(np.nanmean(ada_second_per_class))
    else:
        # 防御性：层数太少时给 0
        ori_second_mean = 0.0
        ada_second_mean = 0.0

    plt.figure(figsize=(6, 6))
    plt.bar(["Original", "Adaptive"],
            [ori_second_mean, ada_second_mean],
            color=["#1f77b4", "#ff7f0e"])
    plt.ylabel("Σ|Δ² CAV ACE| (lower = smoother)")
    plt.title(f"Experiment 2: Second-order Smoothness (N={len(data_list)} classes)")
    plt.tight_layout()
    plt.savefig("./exp_outputs/exp2_smoothness_final.png", dpi=300)
    plt.show()
    print("Saved: exp2_smoothness_final.png")

    csv_path = "./exp_outputs/csv/exp2_smoothness.csv"
    rows = [
        ["Original", ori_second_mean],
        ["Adaptive", ada_second_mean]
    ]
    save_csv(csv_path,
             ["Type", "Second_Order_Smoothness"],
             rows)



############################################################
# EXP3 — Connectivity Consistency (edge weight + slope + smoothness)
############################################################
def plot_exp3():
    cache_dir = "./cache/exp3"
    data_list = load_cache(cache_dir)
    if len(data_list) == 0:
        print("No Exp3 data found")
        return

    layers = data_list[0]["layers"]

    ori_edge = np.array([d["ori_mean_edge"] for d in data_list])
    ada_edge = np.array([d["ada_mean_edge"] for d in data_list])

    ori_density = np.array([d["ori_density"] for d in data_list])
    ada_density = np.array([d["ada_density"] for d in data_list])

    ori_edge_mean = np.nanmean(ori_edge, axis=0)
    ada_edge_mean = np.nanmean(ada_edge, axis=0)

    ori_density_mean = np.nanmean(ori_density, axis=0)
    ada_density_mean = np.nanmean(ada_density, axis=0)

    # ------------------- 主图 -------------------
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(layers, ori_edge_mean, "-o", linewidth=2, label="Original")
    plt.plot(layers, ada_edge_mean, "-o", linewidth=2, label="Adaptive")
    plt.title(f"Exp3: Mean |Edge Weight| (N={len(data_list)})")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(layers, ori_density_mean, "-o", linewidth=2, label="Original")
    plt.plot(layers, ada_density_mean, "-o", linewidth=2, label="Adaptive")
    plt.title(f"Exp3: Edge Density (N={len(data_list)})")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    plt.tight_layout()
    plt.savefig("./exp_outputs/exp3_final.png", dpi=300)
    plt.show()
    print("Saved: exp3_final.png")


    csv_path = "./exp_outputs/csv/exp3_values_main.csv"
    rows = []
    for i, layer in enumerate(layers):
        rows.append([
            layer,
            ori_edge_mean[i], ada_edge_mean[i],
            ori_density_mean[i], ada_density_mean[i],
        ])
    save_csv(csv_path,
             ["Layer",
              "Edge_Ori", "Edge_Ada",
              "Density_Ori", "Density_Ada"],
             rows)

    # ------------------- slope 图 -------------------
    ori_slope = np.array([d["ori_slope"] for d in data_list])
    ada_slope = np.array([d["ada_slope"] for d in data_list])

    ori_slope_mean = np.nanmean(ori_slope, axis=0)
    ada_slope_mean = np.nanmean(ada_slope, axis=0)

    slope_x = [f"{layers[i]}→{layers[i+1]}" for i in range(len(layers)-1)]

    plt.figure(figsize=(8,6))
    plt.plot(slope_x, ori_slope_mean, "-o", linewidth=2, label="Original")
    plt.plot(slope_x, ada_slope_mean, "-o", linewidth=2, label="Adaptive")
    plt.title(f"Exp3: Slope of mean |edge weight|")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("./exp_outputs/exp3_slope_final.png", dpi=300)
    plt.show()
    print("Saved: exp3_slope_final.png")

    csv_path = "./exp_outputs/csv/exp3_slope.csv"
    slope_labels = [f"{layers[i]}->{layers[i+1]}" for i in range(len(layers)-1)]

    rows = []
    for i, name in enumerate(slope_labels):
        rows.append([name, ori_slope_mean[i], ada_slope_mean[i]])
    save_csv(csv_path,
             ["Layer_Transition", "Slope_Ori", "Slope_Ada"],
             rows)

    # ------------------- smoothness 图 -------------------
    ori_second = np.array([d["ori_second"] for d in data_list])
    ada_second = np.array([d["ada_second"] for d in data_list])

    ori_second_mean = float(np.nanmean(ori_second))
    ada_second_mean = float(np.nanmean(ada_second))

    plt.figure(figsize=(6,6))
    plt.bar(["Original", "Adaptive"],
            [ori_second_mean, ada_second_mean],
            color=["#1f77b4", "#ff7f0e"])
    plt.ylabel("Σ|Δ² mean |edge|| (lower = smoother)")
    plt.title(f"Exp3: Second-order Smoothness (N={len(data_list)})")
    plt.tight_layout()
    plt.savefig("./exp_outputs/exp3_smoothness_final.png", dpi=300)
    plt.show()
    print("Saved: exp3_smoothness_final.png")

    csv_path = "./exp_outputs/csv/exp3_smoothness.csv"
    rows = [
        ["Original", ori_second_mean],
        ["Adaptive", ada_second_mean]
    ]
    save_csv(csv_path,
             ["Type", "Second_Order_Smoothness"],
             rows)


############################################################
# main
############################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=int, required=True, choices=[1,2,3],
                        help="1 = Exp1, 2 = Exp2, 3 = Exp3")
    args = parser.parse_args()

    if args.exp == 1:
        plot_exp1()
    elif args.exp == 2:
        plot_exp2()
    else:
        plot_exp3()


if __name__ == "__main__":
    main()
