"""Investigate worst B1 pairs from production actual backtest."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

base = Path(__file__).resolve().parent
ps = pd.read_csv(base / "pair_stats.csv")
pdaily = pd.read_csv(base / "pair_daily_pnl.csv", parse_dates=["date"])
pending = (
    pd.read_csv(base / "pending_target_audit.csv", parse_dates=["date"])
    if (base / "pending_target_audit.csv").exists()
    else pd.DataFrame()
)

b1 = ps[ps["sleeve"] == "core_leveraged"].copy()
worst = b1.nsmallest(12, "pnl_usd").reset_index(drop=True)
print("=== WORST 12 B1 ===")
print(
    worst[
        [
            "ETF",
            "Underlying",
            "pnl_usd",
            "price_pnl_usd",
            "borrow_cost_usd",
            "txn_cost_usd",
            "long_usd",
            "short_usd",
            "n_rebals",
            "hedge_ratio",
            "Delta",
        ]
    ].to_string(index=False)
)

series_out: dict[str, list] = {}
findings: list[dict] = []

for _, r in worst.iterrows():
    etf, und = str(r["ETF"]), str(r["Underlying"])
    key = f"{etf}/{und}"
    d = pdaily[
        (pdaily["ETF"] == etf)
        & (pdaily["Underlying"] == und)
        & (pdaily["sleeve"] == "core_leveraged")
    ].sort_values("date")
    if d.empty:
        print(key, "NO DAILY")
        continue

    etf_abs = d["etf_usd"].abs().replace(0, np.nan)
    ratio = d["underlying_usd"].abs() / etf_abs
    target_h = d["hedge_ratio"].replace(0, np.nan)
    drift = (ratio / target_h - 1).abs()
    g = d["long_usd"].abs() + d["short_usd"].abs()
    gchg = g.diff().abs()
    worst_day = d.loc[d["daily_pnl"].idxmin()]
    best_day = d.loc[d["daily_pnl"].idxmax()]

    # Single-day price pnl as fraction of avg gross
    avg_g = float(g.mean()) if len(g) else 0.0
    day_ret = d["price_pnl"] / max(avg_g, 1e-9)
    big_moves = d.loc[
        day_ret.abs() > 0.05, ["date", "daily_pnl", "price_pnl", "etf_usd", "underlying_usd"]
    ]

    # Sign consistency: B1 should be long und, short etf
    bad_sign_days = int(((d["underlying_usd"] < -1) | (d["etf_usd"] > 1)).sum())

    # Hedge inversion / extreme
    extreme_drift_days = int((drift > 0.5).sum())

    # PnL concentration: top 3 worst days share of total loss
    losses = d["daily_pnl"].clip(upper=0)
    total_loss = float(losses.sum())
    top3 = float(losses.nsmallest(3).sum()) if total_loss < 0 else 0.0
    conc = top3 / total_loss if total_loss < 0 else 0.0

    # Borrow intensity
    borrow_bps = 1e4 * float(d["borrow_cost"].sum()) / max(float(d["short_usd"].abs().mean()) * len(d) / 252 * 252, 1.0)
    # simpler: borrow / |short| / days * 252
    short_mean = float(d["etf_usd"].abs().mean())
    n = len(d)
    borrow_ann = (float(d["borrow_cost"].sum()) / max(short_mean, 1.0)) * (252.0 / max(n, 1))

    print()
    print(f"{key}: days={n} {d['date'].iloc[0].date()} -> {d['date'].iloc[-1].date()}")
    print(
        f"  cum end={d['cum_pnl'].iloc[-1]:.1f} min={d['cum_pnl'].min():.1f} max={d['cum_pnl'].max():.1f}"
    )
    print(f"  avg/max/end gross={g.mean():.0f}/{g.max():.0f}/{g.iloc[-1]:.0f}")
    print(f"  hedge drift p95/max={drift.quantile(0.95):.2%}/{drift.max():.2%} extreme_days(>50%)={extreme_drift_days}")
    print(
        f"  worst day {worst_day['date'].date()} pnl={worst_day['daily_pnl']:.1f} "
        f"price={worst_day['price_pnl']:.1f} borrow={worst_day['borrow_cost']:.1f}"
    )
    print(f"  best day  {best_day['date'].date()} pnl={best_day['daily_pnl']:.1f}")
    print(
        f"  costs borrow={d['borrow_cost'].sum():.1f} txn={d['txn_cost'].sum():.1f} "
        f"credit={d['short_credit'].sum():.1f} borrow_ann~{borrow_ann:.1%}"
    )
    print(f"  rebal flags={int(d['is_rebalance'].sum())} max size jump={gchg.max():.0f} bad_sign_days={bad_sign_days}")
    print(f"  loss concentration top3 days={conc:.1%}")
    if len(big_moves):
        print(f"  |price_pnl|/avg_gross >5% days: {len(big_moves)}")
        print(big_moves.head(8).to_string(index=False))

    flags = []
    if extreme_drift_days >= 3:
        flags.append("persistent_hedge_drift")
    if bad_sign_days > 0:
        flags.append("wrong_leg_signs")
    if conc > 0.6:
        flags.append("loss_concentrated_few_days")
    if borrow_ann > 0.5:
        flags.append("very_high_borrow")
    if float(r["price_pnl_usd"]) > -50 and float(r["pnl_usd"]) < -200:
        flags.append("costs_dominate_price")
    if abs(float(r["hedge_ratio"]) - float(r["Delta"])) / max(abs(float(r["Delta"])), 1e-9) > 0.25:
        flags.append("end_h_vs_delta_mismatch")

    findings.append(
        {
            "pair": key,
            "pnl_usd": float(r["pnl_usd"]),
            "price_pnl_usd": float(r["price_pnl_usd"]),
            "borrow_cost_usd": float(r["borrow_cost_usd"]),
            "txn_cost_usd": float(r["txn_cost_usd"]),
            "avg_gross": float(g.mean()),
            "max_gross": float(g.max()),
            "end_gross": float(g.iloc[-1]),
            "n_days": n,
            "first": str(d["date"].iloc[0].date()),
            "last": str(d["date"].iloc[-1].date()),
            "worst_day": str(worst_day["date"].date()),
            "worst_day_pnl": float(worst_day["daily_pnl"]),
            "hedge_drift_p95": float(drift.quantile(0.95)) if drift.notna().any() else None,
            "hedge_drift_max": float(drift.max()) if drift.notna().any() else None,
            "loss_top3_share": conc,
            "borrow_ann_approx": borrow_ann,
            "bad_sign_days": bad_sign_days,
            "extreme_drift_days": extreme_drift_days,
            "n_rebals": int(r["n_rebals"]),
            "flags": flags,
            "rog": float(r["pnl_usd"]) / max(float(g.mean()), 1.0),
        }
    )

    # compact series for canvas (weekly sample + all rebal days)
    d2 = d.copy()
    d2["gross"] = g
    d2["hedge_drift"] = drift
    keep = d2["is_rebalance"].astype(bool) | (d2.index % 2 == 0)  # every other day + rebals
    # actually keep all — 10 pairs * ~90 days is fine
    series_out[key] = [
        {
            "date": str(row.date.date()),
            "cum_pnl": round(float(row.cum_pnl), 2),
            "daily_pnl": round(float(row.daily_pnl), 2),
            "price_pnl": round(float(row.price_pnl), 2),
            "borrow_cost": round(float(row.borrow_cost), 2),
            "gross": round(float(row.gross), 2),
            "etf_usd": round(float(row.etf_usd), 2),
            "underlying_usd": round(float(row.underlying_usd), 2),
            "hedge_ratio": round(float(row.hedge_ratio), 4) if pd.notna(row.hedge_ratio) else None,
            "Delta": round(float(row.Delta), 4) if pd.notna(row.Delta) else None,
            "hedge_drift": round(float(row.hedge_drift), 4) if pd.notna(row.hedge_drift) else None,
            "is_rebalance": bool(row.is_rebalance),
        }
        for row in d2.itertuples()
    ]

# Cross-pair: same underlying in winners and losers?
print("\n=== SAME UNDERLYING IN WORST vs BEST ===")
best = b1.nlargest(12, "pnl_usd")
wu = set(worst["Underlying"])
bu = set(best["Underlying"])
print("overlap underlyings:", sorted(wu & bu))
for u in sorted(wu & bu):
    sub = b1[b1["Underlying"] == u][["ETF", "Underlying", "pnl_usd", "long_usd", "borrow_cost_usd"]]
    print(sub.to_string(index=False))

# SMU/SMR vs SMUP/SMR specifically
print("\n=== SMR WRAPPER COMPARISON ===")
smr = b1[b1["Underlying"] == "SMR"][
    ["ETF", "pnl_usd", "price_pnl_usd", "borrow_cost_usd", "long_usd", "short_usd", "n_rebals", "hedge_ratio", "Delta"]
]
print(smr.to_string(index=False))

# Check pending audit for worst pairs
if not pending.empty:
    print("\n=== PENDING TARGET FLAGS (sample) ===")
    for _, r in worst.head(5).iterrows():
        etf = r["ETF"]
        p = pending[pending.get("ETF", pd.Series()) == etf] if "ETF" in pending.columns else pd.DataFrame()
        if p.empty and "pair" in pending.columns:
            p = pending[pending["pair"].astype(str).str.contains(etf)]
        if p.empty:
            continue
        print(etf, "pending rows", len(p), "cols", list(p.columns)[:12])

out = {
    "worst_findings": findings,
    "series": series_out,
    "book_b1_pnl": float(b1["pnl_usd"].sum()),
    "worst12_pnl": float(worst["pnl_usd"].sum()),
    "n_b1_pairs": int(len(b1)),
}
(base / "_worst_b1_investigation.json").write_text(json.dumps(out), encoding="utf-8")
print("\nWrote", base / "_worst_b1_investigation.json")
print("worst12 pnl sum", out["worst12_pnl"], "all B1", out["book_b1_pnl"])
