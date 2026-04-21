import json
import sys

sys.stdout.reconfigure(encoding="utf-8")

with open("notebooks/Diamond_Creek_Backtest_v15.ipynb", encoding="utf-8") as f:
    nb = json.load(f)
for i, c in enumerate(nb["cells"]):
    src = "".join(c.get("source", []))[:140].replace("\n", " ")
    ct = c["cell_type"]
    print(f"{i:2d} {ct:8s} | {src}")
