#!/usr/bin/env python3
"""Build the non-live Appendix-A B5 return-verification artifact."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.bucket5_data import INCEPTION, load_vol_panel
from scripts.bucket5_insurance_bt import production_config
from scripts.bucket5_put_overlay import load_spx_spot
from scripts.bucket5_return_verify import (
    build_event_packets,
    performance_metrics,
    run_verification,
    sensitivity_grid,
    theta_cache_replay,
)


def _fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.2%}" if abs(value) <= 2 else f"{value:,.2f}"
    return str(value)


def _write_report(dest: Path, result, theta: dict, sensitivity: pd.DataFrame, packets: dict[str, pd.DataFrame]) -> Path:
    combined = result.books["combined"]["combined_equity"]
    metrics = performance_metrics(combined)
    lines = [
        "# B5 return verification",
        "",
        "## Scope",
        "This artifact implements Appendix A's non-live checks. Paper/live broker fills, locate archives, recalls, and broker financing records are intentionally out of scope and remain future gating requirements.",
        "",
        "## Hard accounting gates",
    ]
    for name, gate in result.gates.items():
        lines.append(f"- **{name}:** {'PASS' if gate['pass'] else 'FAIL'} ({', '.join(f'{k}={v}' for k, v in gate.items() if k not in ('kind', 'pass'))})")
    lines.extend(["", "## Combined portfolio metrics"])
    lines.extend(f"- **{key}:** {_fmt(value)}" for key, value in metrics.items())
    lines.extend(["", "## Split books and attribution", "- `carry_book.csv`, `put_book.csv`, and `combined_book.csv` all use the same initial capital.", "- `daily_attribution.csv` separates gross/net UVIX and SVIX price P&L, allocated borrow, T-bills, put premium, unrealized put MTM, monetization cash, and redeployment.", "- `daily_reconciliation.csv` reconciles combined NAV from sleeve, T-bill, put MTM/cash, and redeployment components.", "- `harvest_audit.csv` ties banked put cash to monetization contract-sale events where engine event detail exists.", "- **Locate/recall replay:** SKIP. The carry engine cannot yet zero new shorts on no-locate days or force-cover recalls; this is an explicit future gate."])
    lines.extend(["", "## Theta option-mark replay", f"- **Status:** {theta.get('status')}", f"- **Observations:** {theta.get('observations', 0)}"])
    if theta.get("status") == "ok":
        lines.extend([f"- **Mean model-minus-quote error:** {_fmt(theta['mean_error'])}", f"- **Median error:** {_fmt(theta['median_error'])}", f"- **95th percentile absolute error:** {_fmt(theta['tail_abs_error_p95'])}", f"- **Soft flag, Theta tail error large:** {theta['stress_error_large']}"])
    else:
        lines.append(f"- **Skip reason:** {theta.get('reason')}")
    lines.extend(["", "## Stress packets", f"- Wrote {len(packets)} available packets. Fixed packets require data coverage; UVIX spike packets cover available live-era stress dates.", "", "## Sensitivity and soft falsifiers"])
    if sensitivity.empty:
        lines.append("- Sensitivity grid not run (`--skip-sens`).")
    else:
        lines.append("- `sensitivity.csv` covers 5/15/30/50 bp ETP slippage and 1.0/1.5/2.0× borrow.")
        lines.append("- Option execution-side, Theta-only/BS-only forced modes, delayed monetization, and no-locate/recall cannot be injected into the current production engine. They are reported as future soft verification gates, not treated as passes.")
        lines.append(f"- **Soft flag, costs erase edge:** {bool((sensitivity['CAGR'] <= 0).any())}")
    lines.extend(["", "## Gate classification", "- **Hard fail:** combined NAV identity; harvested cash that cannot be traced to a monetization event.", "- **Soft flag/report:** cost-stressed edge erased, weak B_live performance, large Theta stress error, missing cache coverage, and engine capabilities not yet available.", ""])
    report = dest / "REPORT.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=INCEPTION, help="B_live start by default")
    parser.add_argument("--end", default=None)
    parser.add_argument("--output", default=None, help="Defaults to data/runs/<today>/b5_return_verify")
    parser.add_argument("--skip-sens", action="store_true", help="Skip 12-scenario cost grid")
    parser.add_argument("--min-theta-observations", type=int, default=10)
    args = parser.parse_args()
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    dest = Path(args.output) if args.output else Path("data/runs") / today / "b5_return_verify"
    dest.mkdir(parents=True, exist_ok=True)

    panel = load_vol_panel(args.start, args.end, use_synthetic=False)
    spx = load_spx_spot(args.start, args.end)
    common = panel.index.intersection(spx.index)
    panel, spx = panel.loc[common], spx.loc[common]
    result = run_verification(panel, spx, production_config())
    result.books["carry"].to_csv(dest / "carry_book.csv")
    result.books["puts"].to_csv(dest / "put_book.csv")
    result.books["combined"].to_csv(dest / "combined_book.csv")
    result.books["components"].to_csv(dest / "component_book.csv")
    result.reconciliation.to_csv(dest / "daily_reconciliation.csv")
    result.attribution.to_csv(dest / "daily_attribution.csv")
    result.harvest_audit.to_csv(dest / "harvest_audit.csv")

    theta_replay, theta = theta_cache_replay(Path("data/cache/spx_options/theta"), spx, panel["vix"], min_observations=args.min_theta_observations)
    if not theta_replay.empty:
        theta_replay.to_csv(dest / "theta_mark_replay.csv", index=False)
    packets = build_event_packets(result, theta_replay)
    for name, packet in packets.items():
        packet.to_csv(dest / f"event_{name}.csv")
    sensitivity = pd.DataFrame() if args.skip_sens else sensitivity_grid(panel, spx, production_config())
    if not sensitivity.empty:
        sensitivity.to_csv(dest / "sensitivity.csv", index=False)
    report = _write_report(dest, result, theta, sensitivity, packets)
    print(report)
    return 0 if all(gate["pass"] for gate in result.gates.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
