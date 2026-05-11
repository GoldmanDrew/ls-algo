import json
from pathlib import Path

p = Path(__file__).resolve().parent.parent / "notebooks" / "Buckets1-4_v2.ipynb"
nb = json.loads(p.read_text(encoding="utf-8"))
cell = nb["cells"][7]
lines = cell["source"]
insert = (
    " **Charts** (`plot_result`, stacked-dashboard top panel, combined B4 `b4_delta`) use "
    "`b4_trading_nav_series` when `attrs['b4_by_pair_backtests']` is present so growth/drawdown "
    "reflect the **trading path** (pair inception funding steps removed). Raw `nav` rows in `bt` "
    "dataframes remain the accounting sum of sub-account books.\n"
)
new_lines = []
for ln in lines:
    if "Bucket 4 PnL path:" in ln and "b4_trading_nav_series" not in ln:
        if not ln.endswith("\n"):
            ln = ln + "\n"
        ln = ln.rstrip("\n") + insert
    new_lines.append(ln)
cell["source"] = new_lines
p.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("ok")
