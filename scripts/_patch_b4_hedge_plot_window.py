"""Patch Buckets1-4*.ipynb: B4 hedge h plot uses backtest date window when b4_bt is passed."""

from __future__ import annotations

import json
from pathlib import Path

NEW_PLOT_B4 = '''def plot_b4_hedge_diagnostics(
    sized_df: pd.DataFrame,
    *,
    run_label: str,
    b4_bt: pd.DataFrame | None = None,
) -> None:
    if sized_df.empty or "Underlying" not in sized_df.columns:
        print(f"Skipping B4 hedge diagnostics for {run_label}: missing Underlying column.")
        return
    h_map = globals().get("v6_opt2_h_daily_map")
    h_base = globals().get("V6_OPT2_H_BASE")
    if not isinstance(h_map, dict):
        print(f"Skipping B4 hedge diagnostics for {run_label}: v6_opt2_h_daily_map missing.")
        return
    t_win: tuple[pd.Timestamp, pd.Timestamp] | None = None
    if b4_bt is not None and not getattr(b4_bt, "empty", True):
        ix_bt = pd.DatetimeIndex(pd.to_datetime(b4_bt.index, errors="coerce")).dropna()
        if len(ix_bt):
            if getattr(ix_bt, "tz", None) is not None:
                ix_bt = ix_bt.tz_convert("UTC").tz_localize(None)
            ix_bt = ix_bt.normalize()
            lo, hi = ix_bt.min(), ix_bt.max()
            t_win = (pd.Timestamp(lo), pd.Timestamp(hi))
    # Plot every B4 underlying in this sizing table (sorted by gross), not a fixed top-N.
    und_by_gross = sized_df.groupby("Underlying")["gross_target_usd"].sum().sort_values(ascending=False)
    if und_by_gross.empty:
        print(f"No B4 underlyings to plot for {run_label}.")
        return
    _n_und = int(und_by_gross.index.size)
    fig, ax = plt.subplots(1, 1, figsize=(14, 7) if _n_und > 10 else (12, 6))
    plotted = False
    for und in und_by_gross.index:
        ser = h_map.get(norm_sym(und))
        if ser is None or getattr(ser, "empty", True):
            continue
        s_plot = pd.Series(ser).dropna().sort_index()
        ix = pd.to_datetime(s_plot.index, errors="coerce")
        if getattr(ix, "tz", None) is not None:
            ix = ix.tz_convert("UTC").tz_localize(None)
        s_plot.index = pd.DatetimeIndex(ix).normalize()
        if t_win is not None:
            w0, w1 = t_win
            s_plot = s_plot.sort_index().loc[(s_plot.index >= w0) & (s_plot.index <= w1)]
            if len(s_plot) == 0:
                continue
        ax.plot(s_plot.index, s_plot.values, lw=1.2, label=f"{und} (dynamic h)")
        plotted = True
    if not plotted:
        plt.close(fig)
        print(f"No dynamic hedge series matched for {run_label}.")
        return
    if h_base is not None:
        ax.axhline(float(h_base), color="red", linestyle="--", lw=1.5, label=f"Static YAML/base h={float(h_base):.3f}")
    if t_win is not None:
        ax.set_xlim(t_win[0], t_win[1])
    ax.set_title(f"Bucket 4 \u2014 dynamic hedge vs static reference ({run_label})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Hedge ratio")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=(7 if _n_und > 12 else 8))
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
'''

OLD_CALL = "        plot_b4_hedge_diagnostics(sized, run_label=best_label)"
NEW_CALL = "        plot_b4_hedge_diagnostics(sized, run_label=best_label, b4_bt=bt)"


def _replace_plot_b4_block(s: str) -> str:
    a = s.find("def plot_b4_hedge_diagnostics")
    b = s.find("def grid_from_dict")
    if a < 0 or b < 0 or b <= a:
        raise SystemExit("plot_b4_hedge_diagnostics / grid_from_dict anchors missing")
    return s[:a] + NEW_PLOT_B4 + "\n\n" + s[b:]


def _patch_notebook(path: Path) -> None:
    nb = json.loads(path.read_text(encoding="utf-8"))
    c8 = nb["cells"][8]
    s = "".join(c8["source"])
    if "b4_bt: pd.DataFrame | None = None" in s and NEW_CALL.strip() in "".join(nb["cells"][28]["source"]):
        print(f"Already patched: {path}")
        return
    if "def plot_b4_hedge_diagnostics" not in s:
        raise SystemExit(f"{path}: plot_b4_hedge_diagnostics missing")
    s = _replace_plot_b4_block(s)
    c8["source"] = [ln + "\n" for ln in s.splitlines()]

    c28 = nb["cells"][28]
    s28 = "".join(c28["source"])
    if OLD_CALL not in s28:
        raise SystemExit(f"{path}: expected plot_b4_hedge_diagnostics call not found")
    s28 = s28.replace(OLD_CALL, NEW_CALL, 1)
    c28["source"] = [ln + "\n" for ln in s28.splitlines()]

    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Patched {path}")


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    for name in ("notebooks/Buckets1-4_v2.ipynb", "notebooks/Buckets1-4.ipynb"):
        _patch_notebook(root / name)


if __name__ == "__main__":
    main()
