# --- AUDIT: inspect a saved JOINT_CONFIGS run (positions.md helper) ---
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

if "JOINT_GRID_WEIGHTS_ROOT" not in globals():
    JOINT_GRID_WEIGHTS_ROOT = Path("data") / "backtest" / "joint_qcqp_grid_weights"


def _resolve_run_dir(name_or_path: str | Path) -> Path:
    """Find the run dir for a given config ``name`` or path under the grid root."""
    p = Path(name_or_path)
    if p.is_dir():
        return p
    root = Path(JOINT_GRID_WEIGHTS_ROOT)
    if not root.is_dir():
        raise FileNotFoundError(f"weights root not found: {root}")
    matches = sorted([d for d in root.iterdir() if d.is_dir() and (str(name_or_path) in d.name)])
    if not matches:
        raise FileNotFoundError(f"no run dir matching {name_or_path!r} under {root}")
    return matches[-1]


def audit_config(name_or_path: str | Path, *, top_n: int = 30) -> pd.DataFrame:
    """Return the top-N pair_book_weights for a config; useful for further analysis."""
    rd = _resolve_run_dir(name_or_path)
    csv = rd / "pair_book_weights.csv"
    if not csv.is_file():
        raise FileNotFoundError(f"missing {csv}")
    df = pd.read_csv(csv).sort_values("w_book", ascending=False).reset_index(drop=True)
    return df.head(int(top_n))


def audit_run(name_or_path: str | Path, *, top_n: int = 30, plot: bool = True) -> None:
    """Print the run summary + top-N positions and optionally plot weights."""
    rd = _resolve_run_dir(name_or_path)
    meta_path = rd / "joint_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    df = pd.read_csv(rd / "pair_book_weights.csv").sort_values("w_book", ascending=False).reset_index(drop=True)

    name = meta.get("config_name") or rd.name
    print(f"=== audit: {name} ({rd.name}) ===")
    print(
        f"  status={meta.get('status')}  primary={meta.get('primary_status')}  "
        f"fallback={meta.get('joint_fallback_kind')!s}"
    )
    sigma_p = meta.get('sigma_p_book_annual')
    sigma_t = meta.get('sigma_target_annual')
    sigma_p_str = f"{100.0 * float(sigma_p):.1f}%" if sigma_p is not None else "?"
    sigma_t_str = f"{100.0 * float(sigma_t):.1f}%" if sigma_t is not None else "?"
    print(f"  sigma_p_book={sigma_p_str}  target={sigma_t_str}")
    n_active = meta.get("n_pairs_active") or int((df["w_book"] > 1e-12).sum())
    print(
        f"  n_pairs_active={n_active}  eff_n_pair={meta.get('eff_n_pair'):.2f}  "
        f"top1={100.0 * float(meta.get('top1_pair_share', 0.0)):.2f}%  "
        f"top5={100.0 * float(meta.get('top5_pair_share', 0.0)):.2f}%"
    )
    knobs = {
        k: meta.get(k)
        for k in (
            "entropy_lambda", "entropy_reference", "edge_temperature",
            "mv_lambda", "eff_n_min_pairs", "weight_ridge_lambda",
            "turnover_lambda", "confidence_haircut", "mu_shrink_intensity",
        )
        if k in meta
    }
    print(f"  knobs={knobs}")
    sleeve_totals = meta.get("sleeve_totals", {}) or {}
    if sleeve_totals:
        print("  sleeve_totals=", {k: f"{100.0 * float(v):.1f}%" for k, v in sleeve_totals.items()})

    show_cols = [c for c in ("bucket", "etf", "underlying", "w_book", "mu_used",
                             "sigma_eff", "q_prior", "n_obs_decay") if c in df.columns]
    head = df.head(int(top_n))[show_cols].copy()
    if "w_book" in head.columns:
        head["w_book_pct"] = (100.0 * head["w_book"]).round(2)
    try:
        from IPython.display import display
        display(head)
    except Exception:
        print(head.to_string(index=False))

    nz = df[df["w_book"] > 1e-12]
    if len(nz) > top_n:
        tail = nz.tail(min(10, len(nz) - top_n))[show_cols].copy()
        if "w_book" in tail.columns:
            tail["w_book_pct"] = (100.0 * tail["w_book"]).round(3)
        print("  --- smallest active positions ---")
        try:
            from IPython.display import display
            display(tail)
        except Exception:
            print(tail.to_string(index=False))

    if plot:
        try:
            import matplotlib.pyplot as plt
            top = df.head(int(top_n))
            fig, ax = plt.subplots(figsize=(11, max(3.0, 0.30 * len(top))))
            ax.barh(top["etf"][::-1], 100.0 * top["w_book"][::-1], color="#4060c0")
            ax.set_xlabel("book weight (%)")
            ax.set_title(f"{name}: top {len(top)} positions")
            ax.grid(True, axis="x", alpha=0.3)
            plt.tight_layout()
            plt.show()
        except Exception as _e:
            print(f"[audit] plot skipped: {_e}")


def list_runs() -> pd.DataFrame:
    """List all saved runs under the grid weights root with quick metadata."""
    root = Path(JOINT_GRID_WEIGHTS_ROOT)
    rows: list[dict] = []
    if not root.is_dir():
        print(f"[audit] no runs under {root}")
        return pd.DataFrame()
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        mp = d / "joint_meta.json"
        try:
            m = json.loads(mp.read_text(encoding="utf-8")) if mp.is_file() else {}
        except Exception:
            m = {}
        rows.append({
            "run_dir": d.name,
            "name": m.get("config_name") or d.name,
            "primary_status": m.get("primary_status"),
            "fallback": m.get("joint_fallback_kind"),
            "eff_n_pair": m.get("eff_n_pair"),
            "top1_pair_share": m.get("top1_pair_share"),
            "sigma_p_book_annual": m.get("sigma_p_book_annual"),
        })
    return pd.DataFrame(rows)


# Auto-audit the highest-Sharpe configuration from this session.
if "JOINT_GRID_RESULTS" in globals() and not JOINT_GRID_RESULTS.empty:
    _best = JOINT_GRID_RESULTS.sort_values("Sharpe", ascending=False).iloc[0]
    _name = str(_best.get("name", ""))
    _wd = _best.get("weights_dir")
    target = _wd if isinstance(_wd, str) and _wd else _name
    print(f"[audit] auto-loading best Sharpe config: {_name} (Sharpe={_best.get('Sharpe'):.2f})")
    try:
        audit_run(target, top_n=30, plot=True)
    except Exception as _e:
        print(f"[audit] auto-load skipped: {_e}")
else:
    print("[audit] no JOINT_GRID_RESULTS in scope; call audit_run('<config_name>') after running the grid cell.")
