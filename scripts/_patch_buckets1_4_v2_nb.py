"""Patch notebooks/Buckets1-4_v2.ipynb cell 6: optional weekly v6_opt2_rebal_index resync."""

from __future__ import annotations

import json
from pathlib import Path

MARKER = "_maybe_resync_v6_opt2_rebal_from_weekly_module"


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    p = repo / "notebooks" / "Buckets1-4_v2.ipynb"
    nb = json.loads(p.read_text(encoding="utf-8"))
    cell = nb["cells"][6]
    src = "".join(cell["source"])
    if MARKER in src:
        print(f"Already patched: {p}")
        return

    needle = "_patch_v6_pair_cache_engine_kw()\n\n\ndef compute_b4_v6_override"
    if needle not in src:
        raise SystemExit("anchor not found in cell 6")

    block = '''_patch_v6_pair_cache_engine_kw()


def _maybe_resync_v6_opt2_rebal_from_weekly_module() -> None:
    """When EXP["b4_resync_v6_rebal_weekly"] is True, replace cached ``v6_opt2_rebal_index`` with
    ``scripts.bucket4_weekly_opt2.weekly_rebalance_dates`` (same calendar as ``Bucket_4_Backtest`` cell 14).
    """
    exp = globals().get("EXP") or {}
    if not bool(exp.get("b4_resync_v6_rebal_weekly")):
        return
    try:
        from scripts.bucket4_weekly_opt2 import weekly_rebalance_dates
    except ImportError as e:
        print(f"[v6] weekly resync skipped (import): {e}")
        return
    cb = globals().get("closes_broad")
    if not isinstance(cb, pd.DataFrame) or len(cb.index) == 0:
        u = set()
        for c in (globals().get("_V6_PAIR_CACHE") or {}).values():
            if isinstance(c, dict) and isinstance(c.get("prices"), pd.DataFrame):
                u.update(pd.to_datetime(c["prices"].index, errors="coerce").tolist())
        if not u:
            print("[v6] weekly resync: no trading calendar (need closes_broad or pair prices)")
            return
        ix = pd.DatetimeIndex(sorted(u))
    else:
        ix = pd.DatetimeIndex(cb.index).sort_values()
    wb = int(exp.get("b4_weekly_warmup_bdays", 65))
    freq = str(exp.get("b4_weekly_rebalance_freq", "W-FRI"))
    new_ix = weekly_rebalance_dates(ix, freq, warmup_bdays=wb)
    globals()["v6_opt2_rebal_index"] = new_ix
    print(f"[v6] v6_opt2_rebal_index resynced via bucket4_weekly_opt2: n={len(new_ix)} freq={freq} warmup={wb}")


_maybe_resync_v6_opt2_rebal_from_weekly_module()



def compute_b4_v6_override'''

    src = src.replace(needle, block, 1)

    cell["source"] = [ln if ln.endswith("\n") else ln + "\n" for ln in src.splitlines(keepends=True)]
    p.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Patched {p}")


if __name__ == "__main__":
    main()
