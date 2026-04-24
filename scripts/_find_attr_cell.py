import json
from pathlib import Path

nb = json.loads(Path("notebooks/Buckets1-4Backtest.ipynb").read_text(encoding="utf-8"))
for i, c in enumerate(nb["cells"]):
    s = "".join(c.get("source", []))
    if "Txn Costs" in s and "Borrow Costs" in s and "cum_margin_debit" in s:
        print("cell", i)
        Path("_attr_cell.txt").write_text(s, encoding="utf-8")
        break
