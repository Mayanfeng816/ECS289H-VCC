import pickle

path = "./outputs/VCC_adaptive/R50_house_finch/cd.pkl"   # change your path

with open(path, "rb") as f:
    cd = pickle.load(f)

print("Object type:", type(cd))
print("\nAttributes inside ConceptDiscovery:\n")

for attr in dir(cd):
    if not attr.startswith("_"):   # Avoid printing private and system properties
        try:
            val = getattr(cd, attr)
            print(f"{attr}: {type(val)}")
        except:
            print(f"{attr}: <unreadable>")
