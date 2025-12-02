import os
import numpy as np
import argparse
import pickle
import sys

# 让 pickle 找到 original_vcc / adaptive_vcc
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from typing import Any, List


# Config
CLASS_LIST_FILE = "./tools/imagenet_names.txt"
ORI_ROOT = "./outputs/VCC_original"
ADA_ROOT = "./outputs/VCC_adaptive"
CACHE_DIR = "./cache/exp3"

LAYERS = ["layer1", "layer2", "layer3", "layer4"]
EDGE_DENSITY_THRESH = 1e-6


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def _flatten_numeric(x: Any, buf: List[float]):
    if x is None:
        return
    if isinstance(x, np.ndarray):
        buf.extend(x.astype(float).ravel().tolist())
    elif isinstance(x, (int, float, np.number)):
        buf.append(float(x))
    elif isinstance(x, dict):
        for v in x.values():
            _flatten_numeric(v, buf)
    elif isinstance(x, (list, tuple)):
        for v in x:
            _flatten_numeric(v, buf)


def flatten_edges(edge_obj: Any) -> np.ndarray:
    buf = []
    _flatten_numeric(edge_obj, buf)
    return np.array(buf, dtype=float)



def load_cd_pkl(path):
    if not os.path.exists(path):
        print(f"[Warning] cd.pkl missing: {path}")
        return None

    with open(path, "rb") as f:
        cd = pickle.load(f)

    mean_edge = []
    density = []

    ew = cd.edge_weights  # dict

    for layer in LAYERS:
        edges_raw = ew.get(layer, None)
        flat = flatten_edges(edges_raw)

        if flat.size == 0:
            mean_edge.append(np.nan)
            density.append(np.nan)
        else:
            abs_edges = np.abs(flat)
            mean_edge.append(float(abs_edges.mean()))
            density.append(float((abs_edges > EDGE_DENSITY_THRESH).mean()))

    return mean_edge, density




# slope + second order difference
def compute_slope(arr):
    arr = np.array(arr, dtype=float)
    return np.diff(arr)


def compute_second_order(arr):
    slope = compute_slope(arr)
    return np.sum(np.abs(np.diff(slope)))


def process_one_class(classname):
    print(f"Processing Class: {classname}")

    ori_cd_path = os.path.join(ORI_ROOT, classname, "cd.pkl")
    ada_cd_path = os.path.join(ADA_ROOT, classname, "cd.pkl")

    ori = load_cd_pkl(ori_cd_path)
    ada = load_cd_pkl(ada_cd_path)

    if ori is None or ada is None:
        print(f"[Skip] Missing cd.pkl for {classname}")
        return

    ori_mean_edge, ori_density = ori
    ada_mean_edge, ada_density = ada

    ori_slope = compute_slope(ori_mean_edge)
    ada_slope = compute_slope(ada_mean_edge)

    ori_second = compute_second_order(ori_mean_edge)
    ada_second = compute_second_order(ada_mean_edge)

    ensure_dir(CACHE_DIR)

    save_obj = {
        "class": classname,
        "layers": LAYERS,

        "ori_mean_edge": ori_mean_edge,
        "ada_mean_edge": ada_mean_edge,

        "ori_density": ori_density,
        "ada_density": ada_density,

        "ori_slope": ori_slope.tolist(),
        "ada_slope": ada_slope.tolist(),

        "ori_second": float(ori_second),
        "ada_second": float(ada_second)
    }

    np.save(os.path.join(CACHE_DIR, f"{classname}.npy"),
            save_obj, allow_pickle=True)

    print(f"[OK] Saved cache → {CACHE_DIR}/{classname}.npy\n")


def load_class_list(num_limit):
    with open(CLASS_LIST_FILE, "r") as f:
        all_classes = [line.strip() for line in f.readlines()]
    return all_classes[:num_limit]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_classes", type=int, default=10,
                        help="How many classes are processed at one time")
    args = parser.parse_args()

    class_list = load_class_list(args.num_classes)

    for cname in class_list:
        process_one_class("R50_" + cname)


if __name__ == "__main__":
    main()
