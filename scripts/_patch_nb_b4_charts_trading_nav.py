"""Patch Buckets1-4_v2.ipynb: B4 charts use b4_trading_nav_series; combined b4_delta too."""

from __future__ import annotations

import json
from pathlib import Path

OLD_IMPORT = (
    "from scripts.bucket4_pair_diagnostics import (\n"
    "    build_b4_attribution_from_pair_bts as _build_b4_attribution_from_pair_bts,\n"
    "    plot_bucket4_per_pair_equity_and_gross,\n"
    ")\n"
)
NEW_IMPORT = (
    "from scripts.bucket4_pair_diagnostics import (\n"
    "    b4_trading_nav_series,\n"
    "    build_b4_attribution_from_pair_bts as _build_b4_attribution_from_pair_bts,\n"
    "    plot_bucket4_per_pair_equity_and_gross,\n"
    ")\n"
)

OLD_PLOT_RESULT = """def plot_result(curves: dict[str, pd.DataFrame], title: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for label, bt in curves.items():
        nav = bt["nav"].astype(float)
        axes[0].plot(nav.index, nav / nav.iloc[0], label=label)
        axes[1].plot(nav.index, nav / nav.cummax() - 1.0, label=label)
    axes[0].set_title(title)
    axes[0].set_ylabel("Growth of $1")
    axes[1].set_ylabel("Drawdown")
    axes[1].axhline(0, color="black", lw=0.8)
    axes[0].legend(loc="best")
    plt.tight_layout()
"""

NEW_PLOT_RESULT = """def plot_result(curves: dict[str, pd.DataFrame], title: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    any_b4_attrs = False
    for label, bt in curves.items():
        nav_raw = bt["nav"].astype(float)
        bp = (getattr(bt, "attrs", None) or {}).get("b4_by_pair_backtests") or {}
        if bp:
            nav = b4_trading_nav_series(nav_raw, bp)
            any_b4_attrs = True
        else:
            nav = nav_raw
        g0 = float(nav.iloc[0])
        if not np.isfinite(g0) or abs(g0) < 1e-12:
            g0 = float(nav.replace(0.0, np.nan).dropna().iloc[0]) if nav.replace(0.0, np.nan).notna().any() else 1.0
        axes[0].plot(nav.index, nav / g0, label=label)
        axes[1].plot(nav.index, nav / nav.cummax() - 1.0, label=label)
    t = title
    if any_b4_attrs:
        t = f"{title}\\n(B4 curves: inception funding stripped — growth/drawdown are trading path.)"
    axes[0].set_title(t)
    axes[0].set_ylabel("Growth of $1")
    axes[1].set_ylabel("Drawdown")
    axes[1].axhline(0, color="black", lw=0.8)
    axes[0].legend(loc="best")
    plt.tight_layout()
"""

OLD_STACK_TOP = """    nav = bt["nav"].astype(float)

    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True, gridspec_kw={"height_ratios": [1.0, 1.25, 1.25]})
    axes[0].plot(nav.index, nav / nav.iloc[0], color="black", lw=2, label="Growth of $1 (after borrow)")
    axes[0].set_ylabel("Growth")
    axes[0].set_title(f"{title} — best grid run: performance vs. sleeve concentration")
"""

NEW_STACK_TOP = """    nav_raw = bt["nav"].astype(float)
    bp_stack = (getattr(bt, "attrs", None) or {}).get("b4_by_pair_backtests") or {}
    nav = b4_trading_nav_series(nav_raw, bp_stack) if bp_stack else nav_raw
    g0s = float(nav.iloc[0])
    if not np.isfinite(g0s) or abs(g0s) < 1e-12:
        g0s = float(nav.replace(0.0, np.nan).dropna().iloc[0]) if nav.replace(0.0, np.nan).notna().any() else 1.0

    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True, gridspec_kw={"height_ratios": [1.0, 1.25, 1.25]})
    axes[0].plot(
        nav.index,
        nav / g0s,
        color="black",
        lw=2,
        label="Growth of $1 (B4: inception stripped)" if bp_stack else "Growth of $1 (after borrow)",
    )
    axes[0].set_ylabel("Growth")
    axes[0].set_title(
        f"{title} — best grid run: performance vs. sleeve concentration"
        + (" (top: trading path)" if bp_stack else "")
    )
"""

OLD_FALLBACK = """        fig, ax = plt.subplots(1, 1, figsize=(11, 4))
        nav = bt["nav"].astype(float)
        ax.plot(nav.index, nav / nav.iloc[0], color="tab:blue", lw=2)
        ax.set_title(f"{title} — growth of $1 (no attribution columns)")
"""

NEW_FALLBACK = """        fig, ax = plt.subplots(1, 1, figsize=(11, 4))
        nav_raw = bt["nav"].astype(float)
        bp_fb = (getattr(bt, "attrs", None) or {}).get("b4_by_pair_backtests") or {}
        nav = b4_trading_nav_series(nav_raw, bp_fb) if bp_fb else nav_raw
        g0f = float(nav.iloc[0])
        if not np.isfinite(g0f) or abs(g0f) < 1e-12:
            g0f = float(nav.replace(0.0, np.nan).dropna().iloc[0]) if nav.replace(0.0, np.nan).notna().any() else 1.0
        ax.plot(nav.index, nav / g0f, color="tab:blue", lw=2)
        ax.set_title(
            f"{title} — growth of $1 (no attribution columns)"
            + ("; B4 inception stripped" if bp_fb else "")
        )
"""

OLD_B4_DELTA = (
    "    b4_nav_full = b4_bt_pf[\"nav\"].reindex(stock_bt.index).ffill().fillna(0.0).astype(float)\n"
    "    b4_nav0 = float(b4_nav_full.iloc[0])\n"
    "    b4_delta = b4_nav_full - b4_nav0\n"
)

NEW_B4_DELTA = (
    "    b4_nav_full = b4_bt_pf[\"nav\"].reindex(stock_bt.index).ffill().fillna(0.0).astype(float)\n"
    "    b4_nav_trading = (\n"
    "        b4_trading_nav_series(b4_nav_full, b4_by_pair)\n"
    "        if b4_by_pair\n"
    "        else b4_nav_full\n"
    "    )\n"
    "    b4_nav0 = float(b4_nav_trading.iloc[0])\n"
    "    b4_delta = b4_nav_trading - b4_nav0\n"
)


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    nb_path = repo / "notebooks" / "Buckets1-4_v2.ipynb"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    c8 = "".join(nb["cells"][8]["source"])
    if "b4_trading_nav_series" in c8 and "b4_nav_trading" in "".join(nb["cells"][22]["source"]):
        print("Already patched")
        return
    if OLD_IMPORT not in c8:
        raise SystemExit("import block not found")
    c8 = c8.replace(OLD_IMPORT, NEW_IMPORT, 1)
    if OLD_PLOT_RESULT not in c8:
        raise SystemExit("plot_result block not found")
    c8 = c8.replace(OLD_PLOT_RESULT, NEW_PLOT_RESULT, 1)
    if OLD_STACK_TOP not in c8:
        raise SystemExit("plot_stacked top block not found")
    c8 = c8.replace(OLD_STACK_TOP, NEW_STACK_TOP, 1)
    if OLD_FALLBACK not in c8:
        raise SystemExit("plot_stacked fallback not found")
    c8 = c8.replace(OLD_FALLBACK, NEW_FALLBACK, 1)
    nb["cells"][8]["source"] = [ln + "\n" for ln in c8.splitlines()]

    c22 = "".join(nb["cells"][22]["source"])
    if OLD_B4_DELTA not in c22:
        raise SystemExit("b4_delta block not found in cell 22")
    c22 = c22.replace(OLD_B4_DELTA, NEW_B4_DELTA, 1)
    nb["cells"][22]["source"] = [ln + "\n" for ln in c22.splitlines()]

    # Cell 22 must see b4_trading_nav_series — import from cell 8 runs first in normal order.
    if "b4_trading_nav_series" not in c22 and "from scripts.bucket4_pair_diagnostics" not in c22:
        pass  # uses globals from cell 8

    nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Patched {nb_path}")


if __name__ == "__main__":
    main()
