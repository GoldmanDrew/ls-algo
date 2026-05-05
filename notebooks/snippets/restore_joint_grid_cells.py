#!/usr/bin/env python3
"""Restore the Joint QCQP grid markdown + code cells if they were deleted.

Run from the ls-algo repo root (or anywhere)::

    python notebooks/snippets/restore_joint_grid_cells.py

This reads ``joint_grid_cells_backup.json`` and inserts the two cells
immediately after the cell whose id is ``715cc7dd`` (``perf(nav)``), only when
``43aae8b5`` or ``facc11de`` is missing from ``Buckets1-4Backtest.ipynb``.
"""
from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    here = Path(__file__).resolve()
    snip_dir = here.parent
    nb_path = snip_dir.parent / "Buckets1-4Backtest.ipynb"
    backup_path = snip_dir / "joint_grid_cells_backup.json"

    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    cells = nb["cells"]
    ids = {c.get("id") for c in cells}
    if "43aae8b5" in ids and "facc11de" in ids:
        print("Notebook already has joint grid cells (43aae8b5 + facc11de). Nothing to do.")
        return

    data = json.loads(backup_path.read_text(encoding="utf-8"))
    data.sort(key=lambda c: 0 if c.get("id") == "43aae8b5" else 1)

    insert_at = None
    for i, c in enumerate(cells):
        if c.get("id") == "715cc7dd":
            insert_at = i + 1
            break
    if insert_at is None:
        raise SystemExit("Could not find perf(nav) cell id=715cc7dd")

    if "43aae8b5" not in ids:
        md = next(c for c in data if c.get("id") == "43aae8b5")
        cells.insert(
            insert_at,
            {
                "cell_type": md["cell_type"],
                "metadata": md.get("metadata", {}),
                "source": md["source"],
                "id": md["id"],
            },
        )
        insert_at += 1
        print("Inserted markdown cell 43aae8b5")
    if "facc11de" not in ids:
        code = next(c for c in data if c.get("id") == "facc11de")
        cells.insert(
            insert_at,
            {
                "cell_type": code["cell_type"],
                "metadata": code.get("metadata", {}),
                "source": code["source"],
                "id": code["id"],
            },
        )
        print("Inserted code cell facc11de")

    nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Wrote", nb_path)


if __name__ == "__main__":
    main()
