import pickle

path = "./outputs/VCC_adaptive/R50_house_finch/cd.pkl"   # 改成你的路径

with open(path, "rb") as f:
    cd = pickle.load(f)

print("Object type:", type(cd))
print("\nAttributes inside ConceptDiscovery:\n")

for attr in dir(cd):
    if not attr.startswith("_"):   # 避免打印私有和系统属性
        try:
            val = getattr(cd, attr)
            print(f"{attr}: {type(val)}")
        except:
            print(f"{attr}: <unreadable>")
