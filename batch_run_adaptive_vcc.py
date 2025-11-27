import os


name_file = "./tools/imagenet_names.txt"


with open(name_file, "r") as f:
    # names = [line.strip() for line in f.readlines() if line.strip()]

    #现在是1-10，需要11-20改readlines()[10:20]
    names = [line.strip() for line in f.readlines()[:10] if line.strip()]

# 逐个运行
for cls in names:
    output_dir = f"outputs/VCC_original/{cls}"
    if os.path.exists(output_dir):
        print(f"⏭  Skip {cls} (already exists)")
        continue
    print(f"\n==============================")
    print(f" Running VCC for class: {cls}")
    print(f"==============================\n")

    #运行代码改这里
    cmd = f'python run_original_vcc.py --target_class "{cls}"'
    os.system(cmd)

print("\n🎉 All classes processed!")
