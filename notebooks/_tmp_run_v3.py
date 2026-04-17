"""Run only cell 13 of Simple_Pair_Backtest.ipynb as a standalone script.
This executes the v3 mu_fwd side-by-side against current data.
"""
import json, sys, os, io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", line_buffering=True)

nb_path = Path(__file__).parent / "Simple_Pair_Backtest.ipynb"
nb = json.loads(nb_path.read_text(encoding="utf-8"))
src = "".join(nb["cells"][13]["source"])
os.chdir(Path(__file__).parent)
exec(compile(src, "<cell_13>", "exec"), {"__name__": "__main__"})
