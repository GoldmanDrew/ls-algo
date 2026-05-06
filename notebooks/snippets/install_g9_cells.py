"""Replace the joint-grid cell (id facc11de) with the named-configs runner and
insert an audit cell after it. Idempotent: re-running updates in place.

Usage:
    python notebooks/snippets/install_g9_cells.py
"""
from __future__ import annotations

import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "Buckets1-4Backtest.ipynb"
SNIPPETS = Path(__file__).resolve().parent
GRID_SRC = (SNIPPETS / "new_grid_cell.py").read_text(encoding="utf-8")
AUDIT_SRC = (SNIPPETS / "new_audit_cell.py").read_text(encoding="utf-8")

GRID_CELL_ID = "facc11de"
AUDIT_CELL_ID = "g9-audit-cell"
AUDIT_MD_ID = "g9-audit-md"


def to_lines(src: str) -> list[str]:
    lines = src.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        # Trailing line without newline is fine for Jupyter; leave as-is.
        pass
    return lines


def _make_cell(cell_type: str, cell_id: str, src: str) -> dict:
    base = {
        "cell_type": cell_type,
        "id": cell_id,
        "metadata": {},
        "source": to_lines(src),
    }
    if cell_type == "code":
        base["execution_count"] = None
        base["outputs"] = []
    return base


def main() -> None:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    cells = nb.get("cells", [])
    grid_idx = next((i for i, c in enumerate(cells) if c.get("id") == GRID_CELL_ID), None)
    if grid_idx is None:
        raise RuntimeError(f"grid cell id={GRID_CELL_ID} not found in {NB}")

    # Replace the grid cell source.
    cells[grid_idx]["source"] = to_lines(GRID_SRC)
    cells[grid_idx]["outputs"] = []
    cells[grid_idx]["execution_count"] = None

    # Insert / update the audit cells immediately after the grid cell.
    md_cell = _make_cell(
        "markdown",
        AUDIT_MD_ID,
        "## Audit a JOINT_CONFIGS run\n\n"
        "Use ``audit_run('<config_name>')`` to inspect the top positions, sleeve\n"
        "totals, and concentration metrics for any saved run. The cell below also\n"
        "auto-loads the highest-Sharpe configuration after the grid finishes.\n",
    )
    code_cell = _make_cell("code", AUDIT_CELL_ID, AUDIT_SRC)

    # Remove any existing g9-audit cells first (idempotent re-install).
    cells = [c for c in cells if c.get("id") not in (AUDIT_CELL_ID, AUDIT_MD_ID)]
    grid_idx = next((i for i, c in enumerate(cells) if c.get("id") == GRID_CELL_ID), None)
    insert_at = grid_idx + 1
    cells.insert(insert_at, md_cell)
    cells.insert(insert_at + 1, code_cell)

    nb["cells"] = cells
    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"installed G9 grid + audit cells -> {NB}")


if __name__ == "__main__":
    main()
