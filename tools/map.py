import os
import json

index_file = r"imagenet_class_index.json"

with open(index_file, "r") as f:
    idx_map = json.load(f)

synset_to_name = {v[0]: v[1] for v in idx_map.values()}

root = r"./dataset"

output_txt = "./tools/imagenet_names.txt"

names = []


for sub in sorted(os.listdir(root)):
    if sub in synset_to_name:
        names.append(synset_to_name[sub])
    else:
        print(f"warning:{sub} not find in index ")

with open(output_txt, "w", encoding="utf-8") as f:
    for name in names:
        f.write(name + "\n")

print(f"generate:{output_txt}")