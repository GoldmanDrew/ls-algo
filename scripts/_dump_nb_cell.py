"""Dump full source of one notebook cell."""
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")
nb = json.load(open(r"notebooks/Diamond_Creek_Backtest.ipynb", "r", encoding="utf-8"))
idx = int(sys.argv[1])
print(f"=== cell [{idx}] ({nb['cells'][idx]['cell_type']}) ===")
print("".join(nb["cells"][idx]["source"]))
