import os
import json
import numpy as np
import argparse


def load_class_list(num_classes):
    """Read the first num_classes classes from imagenet_names.txt"""
    fname = "./tools/imagenet_names.txt"
    classes = []
    with open(fname, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                classes.append(line)
    return classes[:num_classes]


def load_cav_file(path):
    cav_vals = {}   # {layer: [acc_list]}
    if not os.path.exists(path):
        print(f"[WARN] File not found: {path}")
        return cav_vals

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("---"):
                continue

            parts = line.split(":")
            if len(parts) < 3:
                continue

            layer = parts[0].strip()
            acc = float(parts[-1].strip())   # the last is accuracy

            cav_vals.setdefault(layer, []).append(acc)

    return cav_vals


def load_tcav_file(path):
    tcav_score = {}     # {layer: [score_list]}
    tcav_p = {}         # {layer: [p_list]}

    if not os.path.exists(path):
        print(f"[WARN] File not found: {path}")
        return tcav_score, tcav_p

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("---"):
                continue

            parts = line.split(":")
            if len(parts) < 3:
                continue

            layer = parts[0].strip()
            last = parts[-1].strip()

            if "," not in last:
                continue

            score_str, p_str = last.split(",")
            score = float(score_str)
            p = float(p_str)

            tcav_score.setdefault(layer, []).append(score)
            tcav_p.setdefault(layer, []).append(p)

    return tcav_score, tcav_p


def process_one_class(class_name):
    print(f"Processing Class: {class_name}")

    ori_summary = os.path.join("outputs", "VCC_original", class_name, "results_summaries")
    ada_summary = os.path.join("outputs", "VCC_adaptive", class_name, "results_summaries")

    cav_ori = load_cav_file(os.path.join(ori_summary, "CAV_ace_results.txt"))
    cav_ada = load_cav_file(os.path.join(ada_summary, "CAV_ace_results.txt"))

    tcav_ori_score, tcav_ori_p = load_tcav_file(os.path.join(ori_summary, "TCAV_ace_results.txt"))
    tcav_ada_score, tcav_ada_p = load_tcav_file(os.path.join(ada_summary, "TCAV_ace_results.txt"))

    layers = sorted(cav_ori.keys())

    cav_ori_mean = [float(np.mean(cav_ori[l])) for l in layers]
    cav_ada_mean = [float(np.mean(cav_ada[l])) for l in layers]

    tcav_ori_mean = [float(np.mean(tcav_ori_score[l])) for l in layers]
    tcav_ada_mean = [float(np.mean(tcav_ada_score[l])) for l in layers]

    sig_ori = [float(np.mean(np.array(tcav_ori_p[l]) < 0.05)) for l in layers]
    sig_ada = [float(np.mean(np.array(tcav_ada_p[l]) < 0.05)) for l in layers]

    return {
        "class": class_name,
        "layers": layers,
        "cav_ori": cav_ori_mean,
        "cav_ada": cav_ada_mean,
        "tcav_ori": tcav_ori_mean,
        "tcav_ada": tcav_ada_mean,
        "sig_ori": sig_ori,
        "sig_ada": sig_ada,
    }


def save_cache(data, out_dir="./cache/exp2"):
    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(out_dir, f"{data['class']}.json")
    with open(fname, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[Saved] {fname}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_classes", type=int, default=10,
                        help="How many classes are processed at one time")
    args = parser.parse_args()

    # 保留 V2 的类选择方式
    class_list = load_class_list(args.num_classes)

    print(f"Using {args.num_classes} classes: {class_list}")

    for cname in class_list:
        d = process_one_class("R50_"+cname)
        save_cache(d)


if __name__ == "__main__":
    main()
