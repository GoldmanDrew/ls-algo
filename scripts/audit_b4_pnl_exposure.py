"""Audit Bucket-4/5 exposures and PnL against the broker's own position file.

For every ``data/runs/<date>`` that has both an IBKR Flex positions snapshot and
an accounting build, this reconciles three things and prints any breaks:

1. EXPOSURE  — accounting ``gross_notional_usd`` (net_exposure_bucket_4_detail)
   vs the broker ``|positionValue|`` (base currency) for each B4 symbol.
2. PNL TRUTH — accounting ``unrealized_pnl`` (pnl_bucket_4_by_symbol) vs the
   broker ``fifoPnlUnrealized`` (base currency). For the inverse-ETF legs (never
   shared with other buckets) this must match to the cent; shared underlyings are
   attributed slices and are reported separately, not failed.
3. DAY-TO-DAY — over each held-flat interval (no share-count change), the change
   in broker unrealized PnL must equal ``shares * (mark_t - mark_{t-1}) * fx``.
   This is the "exposures line up with the day-to-day PnL" check.

Read-only. Usage:  python -m scripts.audit_b4_pnl_exposure [--run-date YYYY-MM-DD]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "data" / "runs"
_ATTR = re.compile(r'([A-Za-z_:][\w:.-]*)="([^"]*)"')

# Tolerances (USD).
EXPOSURE_TOL = 5.0
PNL_TOL = 1.0
DAY_TOL = 5.0          # absolute floor
DAY_TOL_FRAC = 0.02    # 2% of the move


def _norm(x: object) -> str:
    return str(x).strip().upper().replace(".", "-")


def _f(x: object, d: float = 0.0) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else d
    except (TypeError, ValueError):
        return d


def parse_positions(xml: Path) -> pd.DataFrame:
    """Per-symbol broker truth: shares, mark, fx, base position value & unrealized."""
    rows: list[dict] = []
    try:
        with xml.open("r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                if "<OpenPosition " not in line:
                    continue
                a = dict(_ATTR.findall(line))
                if a.get("assetCategory") != "STK":
                    continue
                if str(a.get("levelOfDetail", "")).upper() not in ("", "SUMMARY"):
                    continue
                fx = _f(a.get("fxRateToBase"), 1.0)
                rows.append(
                    {
                        "symbol": _norm(a.get("symbol", "")),
                        "shares": _f(a.get("position")),
                        "mark": _f(a.get("markPrice")),
                        "fx": fx,
                        "posval_base": _f(a.get("positionValue")) * fx,
                        "fifo_unreal_base": _f(a.get("fifoPnlUnrealized")) * fx,
                    }
                )
    except OSError:
        return pd.DataFrame()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # One physical position per symbol per account; aggregate dollar fields, keep a
    # share-weighted mark so single-line US ETFs are exact.
    agg = df.groupby("symbol").agg(
        shares=("shares", "sum"),
        posval_base=("posval_base", "sum"),
        fifo_unreal_base=("fifo_unreal_base", "sum"),
        fx=("fx", "first"),
    )
    wmark = df.assign(_w=df["shares"].abs()).groupby("symbol").apply(
        lambda g: (g["mark"] * g["_w"]).sum() / g["_w"].sum() if g["_w"].sum() else g["mark"].mean()
    )
    agg["mark"] = wmark
    return agg.reset_index()


def run_dates_with_data() -> list[str]:
    out = []
    if not RUNS.is_dir():
        return out
    for d in sorted(RUNS.iterdir()):
        if not d.is_dir():
            continue
        if (d / "ibkr_flex" / "flex_positions.xml").is_file() and (
            d / "accounting" / "pnl_bucket_4_by_symbol.csv"
        ).is_file():
            out.append(d.name)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-date", default=None, help="Latest run date to treat as 'the new email run'.")
    ap.add_argument("--max-breaks", type=int, default=25)
    args = ap.parse_args()

    pairs = pd.read_csv(REPO / "data" / "runs" / (args.run_date or run_dates_with_data()[-1]) / "accounting" / "bucket4_pairs.csv")
    etf_syms = {_norm(s) for s in pairs["etf"]}
    und_syms = {_norm(s) for s in pairs["underlying"]}
    # Accounting B4 exposure is delta/beta-normalised (position * delta * mark * fx),
    # so an inverse ETF's reported gross is ~|delta|x its raw market value.
    etf_delta = {_norm(e): abs(_f(d, 1.0)) for e, d in zip(pairs["etf"], pairs["delta"])}

    dates = run_dates_with_data()
    if args.run_date:
        dates = [d for d in dates if d <= args.run_date]
    latest = dates[-1]
    print(f"audit over {len(dates)} runs; latest = {latest}\n")

    # Cache parsed positions per date.
    pos_by_date: dict[str, pd.DataFrame] = {}
    for ds in dates:
        pos_by_date[ds] = parse_positions(RUNS / ds / "ibkr_flex" / "flex_positions.xml")

    # ── 1) EXPOSURE: accounting gross vs broker |positionValue| (latest run) ──
    print("=" * 78)
    print("1) EXPOSURE vs broker positionValue  (latest run", latest, ")")
    print("   ETF legs: acct delta-adj gross should == broker|posValue| * |delta|.")
    print("   Underlyings: B4 is an attributed slice of a shared spot line (<= broker total).")
    det = pd.read_csv(RUNS / latest / "accounting" / "net_exposure_bucket_4_detail.csv")
    det["symbol"] = det["symbol"].map(_norm)
    det["leg_type"] = det["leg_type"].astype(str).str.lower()
    pos = pos_by_date[latest].set_index("symbol")
    exp_breaks = 0
    shared_slices = 0
    for _, r in det.iterrows():
        sym = r["symbol"]
        acct_gross = _f(r["gross_notional_usd"])
        if acct_gross <= 0:
            continue
        broker_gross = abs(_f(pos["posval_base"].get(sym, np.nan), np.nan))
        if not np.isfinite(broker_gross):
            print(f"   [no broker pos] {sym:6s} acct_gross={acct_gross:,.0f}")
            exp_breaks += 1
            continue
        if r["leg_type"] == "etf":
            expected = broker_gross * etf_delta.get(sym, 1.0)
            tol = max(EXPOSURE_TOL, 0.01 * expected)
            if abs(acct_gross - expected) > tol:
                print(f"   [EXPOSURE ETF] {sym:6s} acct={acct_gross:,.2f} expected(|val|*|delta|)={expected:,.2f} diff={acct_gross-expected:,.2f}")
                exp_breaks += 1
        else:  # underlying: attributed slice of a (possibly shared) spot line
            if acct_gross > broker_gross + EXPOSURE_TOL:
                print(f"   [EXPOSURE UND] {sym:6s} acct_slice={acct_gross:,.2f} > broker_total={broker_gross:,.2f} (slice should not exceed total)")
                exp_breaks += 1
            elif acct_gross < broker_gross - EXPOSURE_TOL:
                shared_slices += 1
    print(f"   -> {exp_breaks} exposure break(s); {shared_slices} underlying(s) are partial slices of larger shared spot lines (expected)\n")

    # ── 2) PNL TRUTH: accounting unrealized vs broker fifo, per run, ETF legs ──
    print("=" * 78)
    print("2) PNL vs broker fifoPnlUnrealized  (ETF legs must match; shared underlyings reported)")
    pnl_breaks = 0
    shared_notes = 0
    for ds in dates:
        sym_csv = RUNS / ds / "accounting" / "pnl_bucket_4_by_symbol.csv"
        try:
            bysym = pd.read_csv(sym_csv)
        except Exception:
            continue
        if bysym.empty or "symbol" not in bysym.columns:
            continue
        bysym["symbol"] = bysym["symbol"].map(_norm)
        acct_unreal = bysym.groupby("symbol")["unrealized_pnl"].sum()
        pos = pos_by_date[ds]
        if pos.empty:
            continue
        bfifo = pos.set_index("symbol")["fifo_unreal_base"]
        for sym in etf_syms:
            a = _f(acct_unreal.get(sym, np.nan), np.nan)
            b = _f(bfifo.get(sym, np.nan), np.nan)
            if not (np.isfinite(a) and np.isfinite(b)):
                continue
            if abs(a) < 1e-6 and abs(b) < 1e-6:
                continue
            if abs(a - b) > PNL_TOL:
                if pnl_breaks < args.max_breaks:
                    print(f"   [PNL {ds}] ETF {sym:6s} acct={a:,.2f} broker={b:,.2f} diff={a-b:,.2f}")
                pnl_breaks += 1
        # shared underlyings: accounting is an attributed slice; just count divergences
        for sym in und_syms:
            a = _f(acct_unreal.get(sym, np.nan), np.nan)
            b = _f(bfifo.get(sym, np.nan), np.nan)
            if np.isfinite(a) and np.isfinite(b) and abs(a - b) > PNL_TOL:
                shared_notes += 1
    print(f"   -> {pnl_breaks} ETF-leg PnL break(s); {shared_notes} shared-underlying attributed-slice divergence(s) (expected)\n")

    # ── 3) DAY-TO-DAY: ΔfifoUnreal == shares*(Δmark)*fx over held-flat intervals ──
    print("=" * 78)
    print("3) DAY-TO-DAY: broker dPnL vs shares*(dMark)  (ETF legs, held-flat intervals)")
    day_breaks = 0
    checks = 0
    for i in range(1, len(dates)):
        d0, d1 = dates[i - 1], dates[i]
        p0, p1 = pos_by_date[d0], pos_by_date[d1]
        if p0.empty or p1.empty:
            continue
        p0i = p0.set_index("symbol")
        p1i = p1.set_index("symbol")
        for sym in etf_syms:
            if sym not in p0i.index or sym not in p1i.index:
                continue
            s0 = _f(p0i["shares"].get(sym)); s1 = _f(p1i["shares"].get(sym))
            if abs(s0) < 1e-9 or abs(s0 - s1) > 1e-6:  # require held flat (no trade)
                continue
            m0 = _f(p0i["mark"].get(sym)); m1 = _f(p1i["mark"].get(sym))
            fx = _f(p1i["fx"].get(sym), 1.0)
            expected = s1 * (m1 - m0) * fx
            actual = _f(p1i["fifo_unreal_base"].get(sym)) - _f(p0i["fifo_unreal_base"].get(sym))
            checks += 1
            tol = max(DAY_TOL, DAY_TOL_FRAC * abs(expected))
            if abs(actual - expected) > tol:
                if day_breaks < args.max_breaks:
                    print(f"   [DAY {d0}->{d1}] {sym:6s} shares={s1:,.0f} dMark={m1-m0:+.4f} "
                          f"expected={expected:,.2f} actual={actual:,.2f} resid={actual-expected:,.2f}")
                day_breaks += 1
    print(f"   -> {day_breaks} day-to-day break(s) out of {checks} held-flat checks\n")

    # ── 4) RATCHET CREEP: held gross vs solved target (convergence monitor) ──
    print("=" * 78)
    print("4) RATCHET CREEP  (held vs target; flag creep>1.5x or |book_h-model_h|>0.25)")
    creep_flags = 0
    ledger_path = REPO / "data" / "ledger" / f"b4_pair_pnl_hedge_summary_{latest}.csv"
    ratchet_path = RUNS / latest / "b4_hedge_cadence" / "b4_ratchet_targets.csv"
    if ledger_path.is_file():
        led = pd.read_csv(ledger_path)
        for _, r in led.iterrows():
            etf = _norm(r.get("etf", ""))
            und = _norm(r.get("underlying", ""))
            gross_held = _f(r.get("current_total_gross"))
            gross_tgt = _f(r.get("gross_target_usd"))
            creep = _f(r.get("ratchet_creep_ratio"), np.nan)
            book_h = _f(r.get("current_book_h"), np.nan)
            model_h = _f(r.get("current_model_h"), np.nan)
            trim_lam = _f(r.get("ratchet_trim_lambda"), np.nan)
            trim_usd = _f(r.get("ratchet_trim_usd"), np.nan)
            gap = _f(r.get("ratchet_gap_usd"), np.nan)
            if not np.isfinite(creep) and gross_tgt > 1e-6 and gross_held > 1e-6:
                creep = gross_held / gross_tgt
            h_gap = abs(book_h - model_h) if np.isfinite(book_h) and np.isfinite(model_h) else np.nan
            flagged = (np.isfinite(creep) and creep > 1.5) or (np.isfinite(h_gap) and h_gap > 0.25)
            if flagged:
                creep_flags += 1
                print(
                    f"   [CREEP] {etf}|{und} held={gross_held:,.0f} tgt={gross_tgt:,.0f} "
                    f"creep={creep:.2f}x book_h={book_h:.3f} model_h={model_h:.3f} "
                    f"λ={trim_lam:.3f} trim_usd={trim_usd:,.0f} gap={gap:,.0f}"
                )
        if creep_flags == 0:
            print("   -> no pairs above creep/h-hedge thresholds")
        else:
            print(f"   -> {creep_flags} pair(s) flagged for convergence review")
    elif ratchet_path.is_file():
        rt = pd.read_csv(ratchet_path)
        for _, r in rt.iterrows():
            etf = _norm(r.get("ETF", ""))
            und = _norm(r.get("Underlying", ""))
            held = _f(r.get("inverse_etf_short_usd"))
            solved = _f(r.get("inverse_short_solved_usd"))
            creep = held / solved if solved > 1e-6 else np.nan
            if np.isfinite(creep) and creep > 1.5:
                creep_flags += 1
                print(f"   [CREEP] {etf}|{und} inv_held={held:,.0f} inv_solved={solved:,.0f} creep={creep:.2f}x")
        print(f"   -> {creep_flags} pair(s) flagged (ratchet targets only; no ledger)")
    else:
        print("   -> skipped (no ledger summary or b4_ratchet_targets for latest run)")
    print()

    print("=" * 78)
    print(
        f"SUMMARY: exposure_breaks={exp_breaks}  etf_pnl_breaks={pnl_breaks}  "
        f"day_breaks={day_breaks}  creep_flags={creep_flags}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
