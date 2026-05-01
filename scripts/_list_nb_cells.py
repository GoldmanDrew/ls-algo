"""List notebook cells and headlines."""
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")
nb = json.load(open(r"notebooks/Diamond_Creek_Backtest.ipynb", "r", encoding="utf-8"))
print("cells:", len(nb["cells"]))
for i, c in enumerate(nb["cells"]):
    src = "".join(c["source"])[:240].replace("\n", "\\n")
    print(f"[{i:3d}] {c['cell_type']:8s} | {src}")
