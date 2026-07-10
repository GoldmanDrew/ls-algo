"""B4 conditional-crash budget: per-name sizing cap (tail risk x run-up).

THE RULE (one sentence): if a name's conditional crash hits tomorrow, the pair
may lose at most ``rho`` of the B4 sleeve budget.

This replaces the old unconditional tail penalty inside the opt2 score
(``dd_risk_lambda`` / ``risk_denom_coeff`` set to 0 in config, so the opt2
score is decay/borrow only). Sizing risk now lives in exactly one place.

Per pair, as of the run date (all price inputs = underlying closes):

    runup   = max(0, P / median(close, 252d) - 1)     how stretched vs its anchor
    retrace = theta * runup / (1 + runup)             assume theta of the run-up can retrace
    tail    = worst trailing 20d drop over ~3y        realized crash size
              + 0.45 * annualized downside vol (126d)
    C       = max(tail, retrace)                      conditional crash (return units)
    L       = (1-h) * beta / (1+h*beta) * C * (1+phi*C)   pair loss per gross dollar
    cap_usd = rho * budget / max(L, l_floor)
    gross_i = min(solved_gross_i, cap_usd_i)          TRIM-ONLY

Derivation of L: on an underlying crash of size C the short inverse ETF loses
``inv * beta * C``; the short-underlying hedge recovers ``h * beta * inv * C``.
With ``inv = gross / (1 + h*beta)`` the net loss per gross dollar is
``beta * C * (1-h) / (1 + h*beta)`` (~= (1-h)*C at beta=2). ``(1 + phi*C)``
bumps for daily-rebalance compounding of the inverse on a multi-day crash.

Freed gross is NOT redeployed (validated by the G2/G6 battery,
scripts/b4_crash_sizing_suite.py): redistribution concentrates the book and
neutralizes the overlay. Mechanically that means:

  1. ``cap_pair_weights`` caps the opt2 weights AND returns a shrunken
     effective budget, so ``compute_bucket4_targets``'s internal weight
     renormalization cannot re-deploy the freed dollars;
  2. the caller shrinks the B4 sleeve-rescale target by the freed amount;
  3. ``clamp_sized_to_crash_budget`` re-clamps after the book cap stack,
     whose redistribution could otherwise push rows back above their cap.

See docs/b4_sizing_tail_runup_proposal_2026-07-10.md for research history.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

TRADING_DAYS = 252.0
B4_SLEEVE = "inverse_decay_bucket4"


@dataclass
class CrashBudgetParams:
    rho: float = 0.0075                 # max crash loss per name, fraction of sleeve budget
    theta: float = 0.5                  # fraction of the run-up assumed retraceable
    phi: float = 0.5                    # convexity bump on the conditional crash
    l_floor: float = 0.02               # floor on L in the cap denominator (bounds the cap)
    anchor_window: int = 252            # run-up anchor: rolling-median window
    anchor_min_obs: int = 126
    tail_horizon: int = 20              # crash window (days)
    tail_lookback: int = 756            # search window for the worst crash (~3y)
    tail_min_obs: int = 40
    downside_vol_lookback: int = 126
    downside_vol_blend: float = 0.45
    #: Names with too little history for BOTH signals: "book_quantile" assigns
    #: the book's q-quantile L (conservative; the validated G6 control),
    #: "neutral" leaves them uncapped.
    missing_policy: str = "book_quantile"
    missing_l_quantile: float = 0.75

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any] | None) -> "CrashBudgetParams":
        fields = cls.__dataclass_fields__
        kwargs = {k: v for k, v in dict(cfg or {}).items() if k in fields}
        return cls(**kwargs)


def conditional_crash_stats(close: pd.Series, p: CrashBudgetParams) -> dict[str, float] | None:
    """runup / tail / retrace / C for one underlying, as of the last close.

    Returns None when there is not enough history for either signal.
    """
    c = pd.to_numeric(close, errors="coerce").dropna().astype(float)

    runup = np.nan
    if len(c) >= p.anchor_min_obs:
        anchor = float(c.iloc[-int(p.anchor_window):].median())
        if anchor > 0:
            runup = max(0.0, float(c.iloc[-1]) / anchor - 1.0)

    tail = np.nan
    if len(c) >= max(p.tail_min_obs, p.tail_horizon + 5):
        hret = c.pct_change(p.tail_horizon).dropna().iloc[-int(p.tail_lookback):]
        worst = max(0.0, -float(hret.min())) if len(hret) else 0.0
        dret = c.pct_change().dropna().iloc[-int(p.downside_vol_lookback):]
        down = dret[dret < 0.0]
        dvol = float(down.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(down) >= 5 else 0.0
        tail = worst + p.downside_vol_blend * dvol

    if not (np.isfinite(runup) or np.isfinite(tail)):
        return None
    ru = runup if np.isfinite(runup) else 0.0
    retrace = p.theta * ru / (1.0 + ru)
    crash = max(tail if np.isfinite(tail) else 0.0, retrace)
    return {"runup": runup, "tail": tail, "retrace": retrace, "C": crash}


def pair_loss(crash: float, h: float, beta: float, phi: float) -> float:
    """Pair loss per gross dollar on an underlying crash of size ``crash``."""
    c = max(0.0, float(crash))
    hh = float(np.clip(h, 0.0, 1.0))
    b = max(0.1, abs(float(beta)))
    return (1.0 - hh) * b / (1.0 + hh * b) * c * (1.0 + float(phi) * c)


def compute_crash_caps(
    *,
    pair_cache: Mapping[tuple[str, str], Mapping[str, Any]],
    hedge_by_underlying: Mapping[str, pd.Series],
    closes_broad: pd.DataFrame | None,
    hedge_base: float,
    run_date: str | pd.Timestamp,
    budget_usd: float,
    params: CrashBudgetParams,
    norm_sym: Callable[[str], str],
) -> pd.DataFrame:
    """Per-pair conditional crash table with ``cap_usd = rho * budget / L``.

    Inputs mirror ``compute_bucket4_targets`` (same pair cache, same
    per-underlying hedge series as of the run date), so the cap is consistent
    with the legs the engine will actually propose.
    """
    as_of = pd.Timestamp(run_date)
    rows: list[dict[str, Any]] = []
    for (etf_sym, und_sym), c in pair_cache.items():
        if "skip_reason" in c:
            continue
        etf, und = norm_sym(str(etf_sym)), norm_sym(str(und_sym))

        px = None
        if closes_broad is not None and und in getattr(closes_broad, "columns", []):
            px = pd.to_numeric(closes_broad[und], errors="coerce").dropna()
        if px is None or px.empty:
            px = pd.to_numeric(c["prices"].get("b_px"), errors="coerce").dropna()
        if isinstance(px.index, pd.DatetimeIndex):
            ix = px.index.tz_convert("UTC").tz_localize(None) if px.index.tz is not None else px.index
            px = pd.Series(px.to_numpy(), index=ix)
            px = px.loc[px.index <= as_of]

        h_ser = hedge_by_underlying.get(und_sym)
        if h_ser is not None:
            h_hist = h_ser.loc[pd.DatetimeIndex(h_ser.index) <= as_of].dropna()
            h = float(h_hist.iloc[-1]) if len(h_hist) else float(hedge_base)
        else:
            h = float(hedge_base)
        beta = abs(float(c["kw"].get("beta_a", -2.0)))

        stats = conditional_crash_stats(px, params) if px is not None and len(px) else None
        row: dict[str, Any] = {
            "ETF": etf, "Underlying": und, "hedge_ratio": h, "beta": beta,
            "signal_ok": stats is not None,
        }
        if stats is not None:
            row.update(stats)
            row["L"] = pair_loss(stats["C"], h, beta, params.phi)
            row["crash_l_source"] = "signal"
        else:
            row.update({"runup": np.nan, "tail": np.nan, "retrace": np.nan, "C": np.nan,
                        "L": np.nan, "crash_l_source": "missing"})
        rows.append(row)

    caps = pd.DataFrame(rows)
    if caps.empty:
        return caps

    # Short-history names: conservative default L (the G6 control) or uncapped.
    ok = caps["signal_ok"].astype(bool)
    if str(params.missing_policy) == "book_quantile" and ok.any() and (~ok).any():
        l_default = float(caps.loc[ok, "L"].clip(lower=params.l_floor).quantile(params.missing_l_quantile))
        caps.loc[~ok, "L"] = l_default
        caps.loc[~ok, "crash_l_source"] = "book_quantile"

    l_eff = pd.to_numeric(caps["L"], errors="coerce").clip(lower=params.l_floor)
    caps["cap_usd"] = np.where(l_eff.notna(), params.rho * float(budget_usd) / l_eff, np.inf)
    return caps


def cap_pair_weights(
    pair_weights: Mapping[tuple[str, str], float],
    caps: pd.DataFrame,
    budget_usd: float,
    *,
    norm_sym: Callable[[str], str],
) -> tuple[dict[tuple[str, str], float], float, pd.DataFrame]:
    """Trim-only cap on the opt2 weights: ``w' = min(w, cap_usd / budget)``.

    Returns ``(capped_weights, effective_budget, telemetry_table)``.

    ``effective_budget = budget * sum(w') / sum(w)`` — pass it (not the full
    budget) to ``compute_bucket4_targets`` so its internal renormalization
    yields exactly ``gross_i = min(w_i * budget, cap_usd_i)`` and the freed
    dollars stay in cash instead of being redeployed pro-rata.
    """
    if not pair_weights:
        return dict(pair_weights), float(budget_usd), pd.DataFrame()
    lut: dict[tuple[str, str], pd.Series] = {}
    if caps is not None and not caps.empty:
        for _, r in caps.iterrows():
            lut[(norm_sym(str(r["ETF"])), norm_sym(str(r["Underlying"])))] = r

    b = float(budget_usd)
    # Normalize first so w * budget is the actual solved gross even when the
    # incoming weights don't sum to exactly 1.
    wsum = sum(max(0.0, float(v)) for v in pair_weights.values()) or 1.0
    capped: dict[tuple[str, str], float] = {}
    rows: list[dict[str, Any]] = []
    for key, w in pair_weights.items():
        k = (norm_sym(str(key[0])), norm_sym(str(key[1])))
        w0 = max(0.0, float(w)) / wsum
        r = lut.get(k)
        cap_usd = float(r["cap_usd"]) if r is not None else np.inf
        w1 = min(w0, cap_usd / b) if (np.isfinite(cap_usd) and b > 0) else w0
        capped[key] = w1
        rows.append({
            "ETF": k[0], "Underlying": k[1],
            "weight_solved": w0, "weight_capped": w1,
            "gross_solved_usd": w0 * b, "gross_capped_usd": w1 * b,
            "crash_budget_mult": (w1 / w0) if w0 > 0 else 1.0,
            "cap_usd": cap_usd,
            "L": float(r["L"]) if r is not None and pd.notna(r.get("L")) else np.nan,
            "C": float(r["C"]) if r is not None and pd.notna(r.get("C")) else np.nan,
            "runup": float(r["runup"]) if r is not None and pd.notna(r.get("runup")) else np.nan,
            "tail": float(r["tail"]) if r is not None and pd.notna(r.get("tail")) else np.nan,
            "hedge_ratio": float(r["hedge_ratio"]) if r is not None else np.nan,
            "crash_l_source": str(r["crash_l_source"]) if r is not None else "no_cap",
        })

    # After the per-key normalization above, sum(w0) == 1, so the effective
    # budget is simply budget * sum(capped weights): exactly the solved gross
    # minus the freed dollars. compute_bucket4_targets renormalizes capped
    # weights to sum 1 and multiplies by this budget, reproducing
    # gross_i = min(solved_i, cap_usd_i) without redeploying the freed gross.
    budget_eff = b * sum(capped.values())
    return capped, float(budget_eff), pd.DataFrame(rows)


def clamp_sized_to_crash_budget(frame: pd.DataFrame) -> pd.DataFrame:
    """Final trim-only clamp on the sized book, driven by the
    ``crash_budget_clamp_usd`` column written at plan time.

    Needed because the book cap stack (notional-cap redistribution, covariance
    balance, sleeve-budget rescale) can push a B4 row back above its crash
    cap. Both opt2 legs scale with gross so the hedge ratio is preserved.
    Usually a no-op.
    """
    if (
        frame is None or frame.empty
        or "crash_budget_clamp_usd" not in frame.columns
        or "sleeve" not in frame.columns
    ):
        return frame
    frame = frame.copy()
    b4m = frame["sleeve"].astype(str) == B4_SLEEVE
    for idx in frame.index[b4m]:
        clamp = pd.to_numeric(frame.at[idx, "crash_budget_clamp_usd"], errors="coerce")
        if not np.isfinite(clamp):
            continue
        gross = float(pd.to_numeric(frame.at[idx, "gross_target_usd"], errors="coerce") or 0.0)
        if gross <= float(clamp) * (1.0 + 1e-9) or gross <= 0.0:
            continue
        m = float(clamp) / gross
        frame.at[idx, "gross_target_usd"] = float(clamp)
        for leg in ("b4_opt2_inverse_etf_short_usd", "b4_opt2_underlying_short_usd"):
            if leg in frame.columns:
                v = pd.to_numeric(frame.at[idx, leg], errors="coerce")
                if np.isfinite(v):
                    frame.at[idx, leg] = float(v) * m
        print(
            f"[INFO] crash_budget final clamp {frame.at[idx, 'ETF']}: "
            f"${gross:,.0f} -> ${float(clamp):,.0f} (cap stack had re-inflated the row)"
        )
    return frame
