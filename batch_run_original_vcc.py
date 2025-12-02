import os


name_file = "./tools/imagenet_names.txt"


with open(name_file, "r") as f:
    # names = [line.strip() for line in f.readlines() if line.strip()]

    names = [line.strip() for line in f.readlines()[:10] if line.strip()]

# Run one by one
for cls in names:
    output_dir = f"outputs/VCC_original/R50_{cls}"
    if os.path.exists(output_dir):
        print(f"⏭  Skip {cls} (already exists)")
        continue
    print(f"\n==============================")
    print(f" Running VCC for class: {cls}")
    print(f"==============================\n")

    cmd = f'python run_original_vcc.py --target_class "{cls}"'
    os.system(cmd)

print("\n All classes processed!")