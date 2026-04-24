import json
from pathlib import Path

nb = json.loads(Path("notebooks/Buckets1-4Backtest.ipynb").read_text(encoding="utf-8"))
idx = int(Path("_cell_idx.txt").read_text().strip()) if Path("_cell_idx.txt").exists() else 17
Path("_engine_tail.txt").write_text("".join(nb["cells"][idx]["source"]), encoding="utf-8")
print("wrote", idx, "len", len(nb["cells"][idx]["source"]))
