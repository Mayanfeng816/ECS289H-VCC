import os
import pickle
import numpy as np
import plotly.graph_objects as go

# -------------------------
# Config
# -------------------------
CACHE_DIR = "./cache/exp_sankey"
LAYERS = ["layer1", "layer2", "layer3", "layer4"]


# -------------------------
# 工具：加载缓存
# -------------------------
def load_sankey_cache():
    """Load all exp_sankey/*.pkl files and aggregate."""
    if not os.path.exists(CACHE_DIR):
        print(f"[Error] cache dir not found: {CACHE_DIR}")
        return None

    files = [f for f in os.listdir(CACHE_DIR) if f.endswith(".pkl")]
    if len(files) == 0:
        print("[Error] No Sankey cache found in cache/exp_sankey/")
        return None

    flows_ori_list = []
    flows_ada_list = []

    for fname in files:
        path = os.path.join(CACHE_DIR, fname)
        with open(path, "rb") as f:
            data = pickle.load(f)

        flows_ori_list.append(data["flows_ori"])
        flows_ada_list.append(data["flows_ada"])

    # average flow across classes
    avg_ori = {}
    avg_ada = {}

    keys = flows_ori_list[0].keys()
    for k in keys:
        avg_ori[k] = float(np.mean([d[k] for d in flows_ori_list]))
        avg_ada[k] = float(np.mean([d[k] for d in flows_ada_list]))

    return avg_ori, avg_ada


# -------------------------
# 颜色 & 样式辅助
# -------------------------
def normalize(values):
    arr = np.array(values, dtype=float)
    if arr.size == 0:
        return np.zeros_like(arr)
    vmax = np.nanmax(np.abs(arr))
    if vmax <= 0:
        return np.zeros_like(arr)
    return arr / vmax


def build_node_colors(flow_dict, base_hex="#4C78A8"):
    """
    根据每个 layer 的总流量设置 node 颜色透明度（越大越“实”）。
    """
    node_strength = np.zeros(len(LAYERS), dtype=float)
    for (i, j), v in flow_dict.items():
        if v is None or np.isnan(v):
            continue
        node_strength[i] += abs(v)
        node_strength[j] += abs(v)

    node_norm = normalize(node_strength)  # 0~1
    # 把 #RRGGBB 转成 RGB
    r = int(base_hex[1:3], 16)
    g = int(base_hex[3:5], 16)
    b = int(base_hex[5:7], 16)

    node_colors = []
    for a in node_norm:
        alpha = 0.2 + 0.8 * float(a)  # 0.45~0.9
        node_colors.append(f"rgba({r},{g},{b},{alpha})")
    return node_colors


def build_link_style_normal(flow_dict, base_rgb=(76, 114, 176)):
    """
    普通 Original / Adaptive 图：
    - 颜色和透明度都随 flow 大小变化
    """
    keys = list(flow_dict.keys())
    values = [flow_dict[k] for k in keys]
    norm = normalize(values)

    r, g, b = base_rgb
    colors = []
    alphas = []
    for v in norm:
        alpha = 0.05 + 0.95 * float(abs(v))  # 0.25~0.8
        alphas.append(alpha)
        colors.append(f"rgba({r},{g},{b},{alpha})")

    return colors, alphas


def build_link_style_diff(flow_dict):
    """
    Difference 图：
    - 正值：柔和红  (Adaptive > Original)
    - 负值：柔和蓝  (Adaptive < Original)
    - 颜色 & 透明度随 |diff| 变化
    """
    keys = list(flow_dict.keys())
    values = np.array([flow_dict[k] for k in keys], dtype=float)
    norm = normalize(values)

    colors = []
    alphas = []
    for v, n in zip(values, norm):
        alpha = 0.05 + 0.95 * float(abs(n))  # 0.25~0.8
        alphas.append(alpha)
        if v >= 0:
            # 柔和红
            colors.append(f"rgba(250,140,130,{alpha})")
        else:
            # 柔和蓝
            colors.append(f"rgba(130,160,250,{alpha})")
    return colors, alphas


# -------------------------
# 构造单个 Sankey trace
# -------------------------
def make_sankey_trace(flow_dict, title, domain_y, mode="normal"):
    """
    mode: "normal" / "adaptive" / "diff"
    domain_y: (y0, y1) 垂直区间，用来做纵向三联图
    """
    labels = LAYERS
    sources = []
    targets = []
    values = []
    link_labels = []

    for (i, j), v in flow_dict.items():
        sources.append(i)
        targets.append(j)
        values.append(v)
        link_labels.append(f"{LAYERS[i]}→{LAYERS[j]}: {v:.4f}")

    # 节点颜色：用原图的蓝色为基色
    node_colors = build_node_colors(flow_dict, base_hex="#4C78A8")

    # link 颜色
    if mode == "diff":
        link_colors, _ = build_link_style_diff(flow_dict)
    elif mode == "adaptive":
        # 给 Adaptive 换一个偏橘的基色
        link_colors, _ = build_link_style_normal(flow_dict, base_rgb=(242, 142, 43))
    else:
        # Original：偏蓝
        link_colors, _ = build_link_style_normal(flow_dict, base_rgb=(76, 114, 176))

    sankey = go.Sankey(
        domain=dict(
            x=[0.0, 1.0],
            y=list(domain_y),
        ),
        arrangement="snap",
        node=dict(
            pad=25,
            thickness=22,  # 全局 thickness（看起来更“胖”一点）
            line=dict(color="black", width=0.5),
            label=labels,
            color=node_colors
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors,
            label=link_labels,  # (14) 在边上附带数值信息
            hovertemplate="%{label}<extra></extra>",
        )
    )

    return sankey


# -------------------------
# 主流程
# -------------------------
def main():
    result = load_sankey_cache()
    if result is None:
        return

    avg_ori, avg_ada = result
    diff = {k: avg_ada[k] - avg_ori[k] for k in avg_ori.keys()}

    # --------- 单图版本（如果你还想要单独三张 PNG）---------
    # Original
    fig_ori = go.Figure(data=[make_sankey_trace(avg_ori,
                                                "Average Sankey Flow (Original VCC)",
                                                domain_y=(0.0, 1.0),
                                                mode="normal")])
    fig_ori.update_layout(
        title_text="Average Sankey Flow (Original VCC)",
        font_size=16
    )
    fig_ori.write_image("./exp_outputs/sankey_original_final.png")
    print("[Saved] sankey_original_final.png")

    # Adaptive
    fig_ada = go.Figure(data=[make_sankey_trace(avg_ada,
                                                "Average Sankey Flow (Adaptive VCC)",
                                                domain_y=(0.0, 1.0),
                                                mode="adaptive")])
    fig_ada.update_layout(
        title_text="Average Sankey Flow (Adaptive VCC)",
        font_size=16
    )
    fig_ada.write_image("./exp_outputs/sankey_adaptive_final.png")
    print("[Saved] sankey_adaptive_final.png")

    # Diff
    fig_diff = go.Figure(data=[make_sankey_trace(diff,
                                                 "Difference Sankey Flow (Adaptive − Original)",
                                                 domain_y=(0.0, 1.0),
                                                 mode="diff")])
    fig_diff.update_layout(
        title_text="Difference Sankey Flow (Adaptive − Original)",
        font_size=16
    )
    fig_diff.write_image("./exp_outputs/sankey_diff_final.png")
    print("[Saved] sankey_diff_final.png")

    # --------- 三联图（纵向）---------
    traces = [
        make_sankey_trace(avg_ori,
                          "Original VCC",
                          domain_y=(0.70, 1.00),
                          mode="normal"),
        make_sankey_trace(avg_ada,
                          "Adaptive VCC",
                          domain_y=(0.35, 0.65),
                          mode="adaptive"),
        make_sankey_trace(diff,
                          "Adaptive − Original",
                          domain_y=(0.00, 0.30),
                          mode="diff"),
    ]

    fig_all = go.Figure(data=traces)
    fig_all.update_layout(
        height=1000,
        title_text="Sankey Summary: Original vs Adaptive vs Difference",
        font=dict(size=16)
    )

    fig_all.write_image("./exp_outputs/sankey_all_vertical.png")
    print("[Saved] sankey_all_vertical.png")


if __name__ == "__main__":
    main()
