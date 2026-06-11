"""Generate per-pair B4 backtest diagnostic charts (metrics + returns panels).

For each pair in the backtest window, writes two PNGs:
  {ETF}_{UNDERLYING}_metrics.png  — h, expanding CAGR, drawdown, rolling vol
  {ETF}_{UNDERLYING}_returns.png  — pair + leg cumulative returns, rebalance marks

Uses production config knobs (v8 h_mid, cadence, costs) via
``HedgeCadenceKnobs.from_config``.

Usage:
  python -m scripts.bucket4_pair_backtest_plots
  python -m scripts.bucket4_pair_backtest_plots --pair QBTZ/QBTS --max-pairs 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.bucket4_dynamic_bt import run_bucket4_backtest_dynamic_h  # noqa: E402
from scripts.bucket4_hedge_cadence import (  # noqa: E402
    HedgeCadenceKnobs,
    build_h_series,
    load_policy_from_config,
)
from scripts.bucket4_pair_plotting import (  # noqa: E402
    plot_pair_metrics_panel,
    plot_pair_returns_panel,
)
from scripts.bucket4_phase345_backtest import load_pair_data  # noqa: E402
from scripts.bucket4_vol_shape_signals import policy_continuous_interval  # noqa: E402

INITIAL_CAPITAL = 100_000.0
SLIPPAGE_BPS = 20.0


def _load_knobs_and_slip() -> tuple[HedgeCadenceKnobs, float, str]:
    slip = SLIPPAGE_BPS
    try:
        from strategy_config import load_config

        cfg = load_config()
        knobs, _, _ = load_policy_from_config(cfg)
        rules = (
            cfg.get("portfolio", {})
            .get("sleeves", {})
            .get("inverse_decay_bucket4", {})
            .get("rules", {})
        )
        opt2 = rules.get("bucket4_weekly_opt2") or {}
        slip = float(opt2.get("slippage_bps", SLIPPAGE_BPS))
        tag = (
            f"v7 h_mid={knobs.h_mid:.2f} h=[{knobs.h_min:.2f},{knobs.h_max:.2f}] "
            f"cadence base={knobs.base_days:.0f} cap={knobs.max_interval} "
            f"slip={slip:.0f}bps"
        )
        return knobs, slip, tag
    except Exception:
        knobs = HedgeCadenceKnobs()
        return knobs, slip, f"defaults h_mid={knobs.h_mid:.2f}"


def _safe_fname(etf: str, und: str) -> str:
    return f"{etf}_{und}".replace("/", "_").replace(".", "-")


def _write_index_html(outdir: Path, summary: pd.DataFrame) -> None:
    rows = []
    for _, r in summary.sort_values("cagr", ascending=False).iterrows():
        m = Path(str(r["metrics_png"])).name
        ret = Path(str(r["returns_png"])).name
        rows.append(
            f"<tr><td>{r['pair']}</td><td>{r['cagr']:.1%}</td><td>{r['max_dd']:.1%}</td>"
            f"<td>{r['vol']:.1%}</td><td>{int(r['n_rebalances'])}</td>"
            f"<td><a href='pairs/{m}'>metrics</a></td>"
            f"<td><a href='pairs/{ret}'>returns</a></td></tr>"
        )
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>B4 pair backtest charts</title>
<style>body{{font-family:sans-serif;margin:1.5rem}} table{{border-collapse:collapse}}
th,td{{border:1px solid #ccc;padding:4px 8px}} th{{background:#f4f4f4}}</style></head>
<body><h1>Bucket 4 per-pair backtest charts</h1>
<p>{len(summary)} pairs. Sorted by CAGR descending.</p>
<table><tr><th>Pair</th><th>CAGR</th><th>Max DD</th><th>Vol</th><th>Rebals</th>
<th>Metrics</th><th>Returns</th></tr>
{''.join(rows)}
</table></body></html>"""
    (outdir / "index.html").write_text(html, encoding="utf-8")


def run_pair_backtest(pd_: dict, knobs: HedgeCadenceKnobs, slip_bps: float) -> pd.DataFrame:
    h_sig = build_h_series(pd_["sig"], pd_["prices"].index, knobs=knobs)
    rd, _ = policy_continuous_interval(
        pd_["prices"].index, pd_["sig"],
        base_days=knobs.base_days, k_tr=knobs.k_tr, m_vcr=knobs.m_vcr,
        min_interval=knobs.min_interval, max_interval=knobs.max_interval,
    )
    sched = pd.DatetimeIndex(rd).intersection(pd_["prices"].index)
    bt = run_bucket4_backtest_dynamic_h(
        pd_["prices"], h_sig, sched,
        initial_capital=INITIAL_CAPITAL,
        beta_a=-pd_["beta_static"], beta_b=1.0,
        borrow_a_annual=pd_["borrow_a"],
        fee_bps=0.0, slippage_bps=slip_bps,
        opt2_h_base=float(pd_["partial_h"]),
    )
    return bt, h_sig


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Per-pair B4 backtest diagnostic charts.")
    ap.add_argument("--pairs", type=Path, default=None)
    ap.add_argument("--metrics", type=Path,
                    default=REPO.parent / "Levered ETFs/etf-dashboard/data/etf_metrics_daily.csv")
    ap.add_argument("--vol-shape", type=Path,
                    default=REPO.parent / "Levered ETFs/etf-dashboard/data/vol_shape_history.json")
    ap.add_argument("--screened", type=Path, default=REPO / "data/etf_screened_today.csv")
    ap.add_argument("--outdir", type=Path, default=REPO / "notebooks/output/b4_pair_charts")
    ap.add_argument("--start", default="2025-10-07")
    ap.add_argument("--max-pairs", type=int, default=0)
    ap.add_argument("--pair", default=None, help="Single pair filter, e.g. QBTZ/QBTS")
    ap.add_argument("--rolling-vol", type=int, default=21)
    ap.add_argument("--no-html", action="store_true")
    args = ap.parse_args(argv)

    knobs, slip_bps, run_tag = _load_knobs_and_slip()
    pair_data, _ = load_pair_data(args)
    if args.pair:
        want = args.pair.strip().upper().replace(" ", "")
        pair_data = [p for p in pair_data if p["pair"].upper().replace(" ", "") == want]
    print(f"[b4-plots] pairs: {len(pair_data)} | window from {args.start} | {run_tag}")

    pairs_dir = args.outdir / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict] = []

    for i, pd_ in enumerate(pair_data, 1):
        etf, und = pd_["etf"], pd_["underlying"]
        label = pd_["pair"]
        stem = _safe_fname(etf, und)
        try:
            bt, h_sig = run_pair_backtest(pd_, knobs, slip_bps)
        except Exception as exc:
            print(f"  [{i}/{len(pair_data)}] {label} SKIP: {exc}")
            continue
        if bt is None or bt.empty:
            print(f"  [{i}/{len(pair_data)}] {label} SKIP: empty backtest")
            continue

        m_path = pairs_dir / f"{stem}_metrics.png"
        r_path = pairs_dir / f"{stem}_returns.png"
        metrics = plot_pair_metrics_panel(
            bt, pair_label=label, etf=etf, underlying=und,
            h_signal=h_sig, run_label=run_tag, out_path=m_path,
            rolling_vol=args.rolling_vol, initial_capital=INITIAL_CAPITAL,
        )
        max_resid = plot_pair_returns_panel(
            bt, pair_label=label, etf=etf, underlying=und,
            run_label=run_tag, out_path=r_path, initial_capital=INITIAL_CAPITAL,
        )
        summary_rows.append({
            "pair": label, "etf": etf, "underlying": und,
            **metrics,
            "max_leg_residual_usd": max_resid,
            "metrics_png": str(m_path.relative_to(args.outdir)),
            "returns_png": str(r_path.relative_to(args.outdir)),
        })
        print(f"  [{i}/{len(pair_data)}] {label}  CAGR {metrics['cagr']:.1%}  rebals {metrics['n_rebalances']}")

    if not summary_rows:
        print("[b4-plots] no charts produced")
        return 1

    summary = pd.DataFrame(summary_rows).sort_values("cagr", ascending=False)
    csv_path = args.outdir / "b4_pair_backtest_summary.csv"
    summary.to_csv(csv_path, index=False)
    if not args.no_html:
        _write_index_html(args.outdir, summary)
    print(f"[b4-plots] wrote {len(summary)} pairs -> {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
