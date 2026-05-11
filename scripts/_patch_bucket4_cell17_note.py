"""Prepend Bucket_4_Backtest.ipynb cell 17 with a pointer to scripts.bucket4_weekly_opt2."""

from __future__ import annotations

import json
from pathlib import Path

NOTE = (
    "# Portfolio sim (cell 17) — research reference. For the shared weekly + drift engine and\n"
    "# diagnostics aligned with ``generate_trade_plan`` / pytest, use\n"
    "# ``scripts.bucket4_weekly_opt2.run_bucket4_backtest`` after ``build_bucket4_state`` +\n"
    "# ``compute_bucket4_weights`` (same per-pair economics as ``run_bucket4_backtest_dynamic_h``).\n\n"
)


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    p = repo / "notebooks" / "Bucket_4_Backtest.ipynb"
    nb = json.loads(p.read_text(encoding="utf-8"))
    cell = nb["cells"][17]
    src = "".join(cell["source"])
    if "scripts.bucket4_weekly_opt2.run_bucket4_backtest" in src[:500]:
        print(f"Already noted: {p} cell 17")
        return
    cell["source"] = [ln if ln.endswith("\n") else ln + "\n" for ln in (NOTE + src).splitlines(keepends=True)]
    p.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Patched {p} cell 17 header")


if __name__ == "__main__":
    main()
