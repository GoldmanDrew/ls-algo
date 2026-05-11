"""Patch Buckets1-4*.ipynb: Bucket 1 dashboard shows all attribution tickers (top_n=None)."""

from __future__ import annotations

import json
from pathlib import Path

OLD_PICK = """def _pick_top_etfs_from_attribution(gross_df: pd.DataFrame, top_n: int) -> list[str]:
    if gross_df.empty:
        return []
    mean_g = gross_df.mean(axis=0).sort_values(ascending=False)
    return [c for c in mean_g.index if float(mean_g[c]) > 0][:top_n]
"""

NEW_PICK = """def _pick_top_etfs_from_attribution(gross_df: pd.DataFrame, top_n: int | None) -> list[str]:
    if gross_df.empty:
        return []
    mean_g = gross_df.mean(axis=0).sort_values(ascending=False)
    pos = [c for c in mean_g.index if float(mean_g[c]) > 0]
    if top_n is None:
        return pos
    return pos[:top_n]
"""

OLD_SIG = '''    top_n: int = 12,
) -> None:
    """Visualize concentration (stacked gross), mark-to-market attribution, and NAV for the best run."""'''

NEW_SIG = '''    top_n: int | None = 12,
) -> None:
    """Visualize concentration (stacked gross), mark-to-market attribution, and NAV for the best run."""'''

OLD_COLS = """    cols = _pick_top_etfs_from_attribution(gross_df, top_n)
    if not cols:
        cols = list(gross_df.columns[:top_n])"""

NEW_COLS = """    cols = _pick_top_etfs_from_attribution(gross_df, top_n)
    if not cols:
        cols = list(gross_df.columns if top_n is None else gross_df.columns[:top_n])"""

OLD_PLOT_CALL = '    plot_stacked_gross_and_cum_pnl_with_nav(bt, attr or {}, title=f"{title} ({sub})", top_n=14)'

NEW_PLOT_CALL = """    plot_stacked_gross_and_cum_pnl_with_nav(
        bt,
        attr or {},
        title=f"{title} ({sub})",
        top_n=(None if bucket == 1 else 14),
    )"""


def _patch(path: Path) -> None:
    nb = json.loads(path.read_text(encoding="utf-8"))
    cell8 = nb["cells"][8]
    s = "".join(cell8["source"])
    if NEW_PICK.splitlines()[0] in s and "top_n=(None if bucket == 1 else 14)" in "".join(nb["cells"][28]["source"]):
        print(f"Already patched: {path}")
        return
    if OLD_PICK not in s:
        raise SystemExit(f"{path}: expected _pick_top_etfs_from_attribution block missing")
    s = s.replace(OLD_PICK, NEW_PICK, 1)
    if OLD_SIG not in s:
        raise SystemExit(f"{path}: expected plot_stacked signature block missing")
    s = s.replace(OLD_SIG, NEW_SIG, 1)
    if OLD_COLS not in s:
        raise SystemExit(f"{path}: expected cols fallback block missing")
    s = s.replace(OLD_COLS, NEW_COLS, 1)
    cell8["source"] = [ln + "\n" for ln in s.splitlines()]

    cell28 = nb["cells"][28]
    s28 = "".join(cell28["source"])
    if OLD_PLOT_CALL not in s28:
        raise SystemExit(f"{path}: expected run_best_bucket_dashboard plot line missing")
    s28 = s28.replace(OLD_PLOT_CALL, NEW_PLOT_CALL, 1)
    cell28["source"] = [ln + "\n" for ln in s28.splitlines()]

    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Patched {path}")


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    for name in ("notebooks/Buckets1-4_v2.ipynb", "notebooks/Buckets1-4.ipynb"):
        _patch(root / name)


if __name__ == "__main__":
    main()
