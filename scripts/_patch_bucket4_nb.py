"""One-off patcher for notebooks/Bucket_4_Backtest.ipynb cell 14 (weekly rebal)."""

from __future__ import annotations

import json
from pathlib import Path


def _patch_cell2(nb: dict) -> bool:
    cell = nb["cells"][2]
    src = "".join(cell["source"])
    mark = "V6_OPT2_WEEKLY_REBAL_FREQ"
    if mark in src:
        return False
    needle = "END = None\n\n# Direction:"
    if needle not in src:
        raise SystemExit("cell2 anchor missing")
    ins = (
        "END = None\n\n"
        "# Hedge-panel calendar for v6 Opt-2 (cell 14); pandas resample frequency string.\n"
        "if \"V6_OPT2_WEEKLY_REBAL_FREQ\" not in globals():\n"
        "    V6_OPT2_WEEKLY_REBAL_FREQ = \"W-FRI\"\n\n"
        "# Direction:"
    )
    cell["source"] = [ln if ln.endswith("\n") else ln + "\n" for ln in src.replace(needle, ins, 1).splitlines(keepends=True)]
    return True


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    p = repo / "notebooks" / "Bucket_4_Backtest.ipynb"
    nb = json.loads(p.read_text(encoding="utf-8"))
    cell2_dirty = _patch_cell2(nb)
    cell = nb["cells"][14]
    src = "".join(cell["source"])
    if "weekly_rebalance_dates(all_dates" in src and "from scripts.bucket4_weekly_opt2 import weekly_rebalance_dates" in src:
        if cell2_dirty:
            p.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"Updated cell 2 only: {p}")
        else:
            print(f"Already patched: {p}")
        return

    n1 = "import pandas as pd\n\nassert"
    if n1 not in src:
        raise SystemExit("import block missing")
    ins = "import pandas as pd\n\nfrom scripts.bucket4_weekly_opt2 import weekly_rebalance_dates\n\nassert"
    if "weekly_rebalance_dates" not in src:
        src = src.replace(n1, ins, 1)

    n2 = "    rebal = all_dates[WARMUP_BDAYS::V6_BDAY_STEP]\n"
    if n2 not in src:
        raise SystemExit("rebal line missing")
    src = src.replace(
        n2,
        "    _rf = str(globals().get(\"V6_OPT2_WEEKLY_REBAL_FREQ\", \"W-FRI\"))\n"
        "    rebal = weekly_rebalance_dates(all_dates, _rf, warmup_bdays=WARMUP_BDAYS)\n",
        1,
    )

    # Module-level print: use global freq, not _rf (function-local).
    n3 = (
        '    f"rebal dates={len(v6_opt2_rebal_index)} (every {V6_BDAY_STEP} bdays, H_base={V6_OPT2_H_BASE:g}, "'
    )
    n3b = (
        '    f"rebal dates={len(v6_opt2_rebal_index)} (weekly '
        '{str(globals().get(\"V6_OPT2_WEEKLY_REBAL_FREQ\", \"W-FRI\"))}, H_base={V6_OPT2_H_BASE:g}, "'
    )
    if n3 in src:
        src = src.replace(n3, n3b, 1)
    elif "weekly" in src and "V6_OPT2_WEEKLY_REBAL_FREQ" in src:
        pass
    else:
        raise SystemExit("print fragment missing")

    # Header comment still says 10d — update first line of cell.
    src = src.replace(
        "# --- Bucket 4 v6 Option-2: per-pair backtest (locked r_10d + range_expansion, B4 cross-section, 10d rebalance) ---",
        "# --- Bucket 4 v6 Option-2: per-pair backtest (hedge panel on weekly calendar; see scripts.bucket4_weekly_opt2) ---",
        1,
    )

    cell["source"] = [ln if ln.endswith("\n") else ln + "\n" for ln in src.splitlines(keepends=True)]
    p.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Patched {p} (cells 2 + 14)")


if __name__ == "__main__":
    main()
