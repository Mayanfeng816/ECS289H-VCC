import os
import pickle
import argparse
import numpy as np
import sys

# let pickle find original_vcc / adaptive_vcc
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

LAYERS = ["layer1","layer2","layer3","layer4"]

def flatten_edges(x, buf):
    """Recursively flatten numeric structures."""
    if x is None:
        return
    if isinstance(x, np.ndarray):
        buf.extend(x.astype(float).ravel().tolist())
    elif isinstance(x, (int, float, np.number)):
        buf.append(float(x))
    elif isinstance(x, dict):
        for v in x.values():
            flatten_edges(v, buf)
    elif isinstance(x, (list, tuple)):
        for v in x:
            flatten_edges(v, buf)

def compute_layer_flow(cd):
    """Compute layer→layer aggregated connectivity for Sankey."""
    flows = { (i, i+1): 0.0 for i in range(len(LAYERS)-1) }

    ew = cd.edge_weights

    for i in range(len(LAYERS)-1):
        layerA = LAYERS[i]
        layerB = LAYERS[i+1]

        flatA, flatB = [], []

        flatten_edges(ew.get(layerA, None), flatA)
        flatten_edges(ew.get(layerB, None), flatB)

        if len(flatA) > 0 and len(flatB) > 0:
            wA = np.abs(np.array(flatA))
            wB = np.abs(np.array(flatB))
            flows[(i, i+1)] = float((wA.mean() + wB.mean())/2)

    return flows

def load_cd(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def ensure_dir(x):
    if not os.path.exists(x):
        os.makedirs(x)

def process_one_class(classname, ori_root, ada_root, cache_dir):
    ori_pkl = os.path.join(ori_root, classname, "cd.pkl")
    ada_pkl = os.path.join(ada_root, classname, "cd.pkl")

    if not (os.path.exists(ori_pkl) and os.path.exists(ada_pkl)):
        print(f"[Skip] Missing cd.pkl for {classname}")
        return

    cd_ori = load_cd(ori_pkl)
    cd_ada = load_cd(ada_pkl)

    flows_ori = compute_layer_flow(cd_ori)
    flows_ada = compute_layer_flow(cd_ada)

    out = {
        "class": classname,
        "flows_ori": flows_ori,
        "flows_ada": flows_ada
    }

    save_path = os.path.join(cache_dir, f"{classname}.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(out, f)

    print(f"[Saved] {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--class_list", default="./tools/imagenet_names.txt", type=str)
    parser.add_argument("--num_classes", default=10, type=int)
    parser.add_argument("--ori_root", default="./outputs/VCC_original")
    parser.add_argument("--ada_root", default="./outputs/VCC_adaptive")
    parser.add_argument("--cache_dir", default="./cache/exp_sankey")
    args = parser.parse_args()

    ensure_dir(args.cache_dir)

    # load class names
    with open(args.class_list, "r") as f:
        all_classes = [x.strip() for x in f.readlines()]

    classes_to_process = all_classes[: args.num_classes]

    print(f"Processing {len(classes_to_process)} classes...")

    for cname in classes_to_process:
        process_one_class("R50_"+cname, args.ori_root, args.ada_root, args.cache_dir)


if __name__ == "__main__":
    main()
