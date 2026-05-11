"""Insert EXP keys for weekly v6 resync into Buckets1-4_v2.ipynb."""

from __future__ import annotations

import json
from pathlib import Path

ANCHOR = '    "combined_union_v6_b4_rebalance": True,\n'
INSERT = (
    '    "combined_union_v6_b4_rebalance": True,\n'
    '    # When True, replace cached ``v6_opt2_rebal_index`` with ``scripts.bucket4_weekly_opt2.weekly_rebalance_dates``.\n'
    '    "b4_resync_v6_rebal_weekly": False,\n'
    '    "b4_weekly_warmup_bdays": 65,\n'
    '    "b4_weekly_rebalance_freq": "W-FRI",\n'
    '    # ``compute_v6_b4_pf_weight_dict(..., use_ibkr_uvix_borrow=...)``; False = screener borrow for UVIX like ``bucket4_weekly_opt2``.\n'
    '    "b4_weights_use_ibkr_uvix_borrow": True,\n'
)


def main() -> None:
    p = Path(__file__).resolve().parent.parent / "notebooks" / "Buckets1-4_v2.ipynb"
    nb = json.loads(p.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        if "b4_resync_v6_rebal_weekly" in src:
            print("EXP already patched")
            return
        if ANCHOR in src and "EXP = {" in src:
            cell["source"] = [
                ln if ln.endswith("\n") else ln + "\n" for ln in src.replace(ANCHOR, INSERT, 1).splitlines(keepends=True)
            ]
            p.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"Patched EXP in {p}")
            return
    raise SystemExit("EXP anchor not found")


if __name__ == "__main__":
    main()
