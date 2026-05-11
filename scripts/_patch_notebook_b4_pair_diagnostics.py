"""Move B4 attribution into scripts/bucket4_pair_diagnostics.py and wire dashboard plot."""

from __future__ import annotations

import json
from pathlib import Path

ANCHOR_IMPORT = "from scripts.bucket4_tail_portfolio import aggregate_tail_risk_weighted_portfolio\n"

EXTRA_IMPORT = (
    "\n"
    "import scripts.bucket4_pair_diagnostics as _b4diag_live\n\n"
    "importlib.reload(_b4diag_live)\n"
    "from scripts.bucket4_pair_diagnostics import (\n"
    "    build_b4_attribution_from_pair_bts as _build_b4_attribution_from_pair_bts,\n"
    "    plot_bucket4_per_pair_equity_and_gross,\n"
    ")\n"
)

OLD_CALL = "        plot_b4_hedge_diagnostics(sized, run_label=best_label, b4_bt=bt)\n"

NEW_CALL = (
    "        plot_b4_hedge_diagnostics(sized, run_label=best_label, b4_bt=bt)\n"
    "        plot_bucket4_per_pair_equity_and_gross(bt, bt_by_pair, run_label=best_label)\n"
)


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    p = repo / "notebooks" / "Buckets1-4_v2.ipynb"
    nb = json.loads(p.read_text(encoding="utf-8"))
    c8 = "".join(nb["cells"][8]["source"])
    c28 = "".join(nb["cells"][28]["source"])

    if "import scripts.bucket4_pair_diagnostics" in c8 and NEW_CALL.strip() in c28:
        print(f"Already patched: {p}")
        return

    a = c8.find("def _build_b4_attribution_from_pair_bts")
    b = c8.find("def verify_b4_portfolio_vs_sum_of_pairs")
    if a < 0 or b < 0 or b <= a:
        raise SystemExit("Could not locate _build_b4_attribution_from_pair_bts block")
    c8 = c8[:a] + c8[b:]

    if ANCHOR_IMPORT not in c8:
        raise SystemExit("aggregate_tail_risk_weighted_portfolio import anchor missing")
    c8 = c8.replace(ANCHOR_IMPORT, ANCHOR_IMPORT + EXTRA_IMPORT, 1)

    if OLD_CALL not in c28:
        raise SystemExit("plot_b4_hedge_diagnostics call not found in cell 28")
    c28 = c28.replace(OLD_CALL, NEW_CALL, 1)

    nb["cells"][8]["source"] = [ln + "\n" for ln in c8.splitlines()]
    nb["cells"][28]["source"] = [ln + "\n" for ln in c28.splitlines()]
    p.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Patched {p}")


if __name__ == "__main__":
    main()
