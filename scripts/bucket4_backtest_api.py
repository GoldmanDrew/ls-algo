"""Bucket 4 production sizing API for dashboard / CI walk-forward backtests.

Exposes the live GTP stack (v6 opt2 decay/borrow/cov + conditional crash budget)
without requiring IBKR, ratchet state files, or the full multi-sleeve book.

Usage::

    from scripts.bucket4_backtest_api import size_b4_book_asof, build_dashboard_sizing

    sized = size_b4_book_asof(
        run_date="2026-07-10",
        pair_cache=...,
        hedge_by_underlying=...,
        closes_broad=...,
        screened_csv="data/etf_screened_today.csv",
        sleeve_budget_usd=100_000.0,
        opt2_cfg={...},
    )
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.b4_crash_budget import (  # noqa: E402
    CrashBudgetParams,
    cap_pair_weights,
    compute_crash_caps,
)
from scripts.v6_b4_pf_weights import (  # noqa: E402
    V6PfParams,
    compute_v6_b4_pf_weight_dict,
)

NormSym = Callable[[str], str]


def _norm_sym(x: object) -> str:
    return str(x).strip().upper().replace(".", "-")


def _stub_ibkr(_symbols: list[str]) -> dict[str, float]:
    return {}


@dataclass
class SizedBook:
    """Opt2 + crash-budget result for one as-of date."""

    run_date: str
    weights_opt2: dict[tuple[str, str], float]
    weights_capped: dict[tuple[str, str], float]
    budget_usd: float
    budget_eff: float
    deployed_fraction: float
    cash_residual: float
    telemetry: list[dict[str, Any]] = field(default_factory=list)
    opt2_meta: dict[str, Any] = field(default_factory=dict)
    sizing_method: str = "v6_opt2_crash_budget"

    def weights_by_etf(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for (etf, _und), w in self.weights_capped.items():
            out[_norm_sym(etf)] = float(w)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_date": self.run_date,
            "sizing_method": self.sizing_method,
            "budget_usd": self.budget_usd,
            "budget_eff": self.budget_eff,
            "deployed_fraction": self.deployed_fraction,
            "cash_residual": self.cash_residual,
            "weights_by_etf": self.weights_by_etf(),
            "weights_opt2_by_etf": {
                _norm_sym(k[0]): float(v) for k, v in self.weights_opt2.items()
            },
            "telemetry": self.telemetry,
            "opt2_meta": self.opt2_meta,
        }


def size_b4_book_asof(
    *,
    run_date: str | pd.Timestamp,
    pair_cache: Mapping[tuple[str, str], Mapping[str, Any]],
    hedge_by_underlying: Mapping[str, pd.Series],
    closes_broad: pd.DataFrame | None,
    screened_csv: str | Path,
    sleeve_budget_usd: float,
    opt2_cfg: Mapping[str, Any] | None = None,
    hedge_base: float | None = None,
    norm_sym: NormSym | None = None,
    get_ibkr_borrow_map: Callable[[list[str]], dict[str, float]] | None = None,
    use_ibkr_uvix_borrow: bool = False,
) -> SizedBook:
    """Production B4 sizing for one as-of date: opt2 weights then crash-budget caps.

    Returns capped weights that generally sum to **less than 1** (freed gross
    stays in cash). ``budget_eff = budget * sum(w_capped)``.
    """
    ns = norm_sym or _norm_sym
    opt2 = dict(opt2_cfg or {})
    as_of = pd.Timestamp(run_date).strftime("%Y-%m-%d")
    budget = float(sleeve_budget_usd)
    if budget <= 0:
        raise ValueError("sleeve_budget_usd must be positive")

    h_policy = opt2.get("hedge_cadence_policy") or {}
    h_base = float(
        hedge_base
        if hedge_base is not None
        else h_policy.get("h_mid", opt2.get("h_base", 0.45))
    )
    pf_min = int(opt2.get("pf_min_pairs", 5))
    params = V6PfParams.from_opt2_dict(opt2, min_pairs=pf_min)

    # Truncate hedge series / prices to as-of for point-in-time sizing.
    h_map: dict[str, pd.Series] = {}
    for und, ser in (hedge_by_underlying or {}).items():
        s = pd.Series(ser).copy()
        if not isinstance(s.index, pd.DatetimeIndex):
            s.index = pd.to_datetime(s.index)
        s = s.loc[s.index <= pd.Timestamp(as_of)].dropna()
        if len(s):
            h_map[und] = s

    cache: dict[tuple[str, str], dict[str, Any]] = {}
    for key, c in (pair_cache or {}).items():
        if "skip_reason" in c:
            continue
        px = c.get("prices")
        if px is None or getattr(px, "empty", True):
            continue
        px2 = px.copy()
        if not isinstance(px2.index, pd.DatetimeIndex):
            px2.index = pd.to_datetime(px2.index)
        px2 = px2.loc[px2.index <= pd.Timestamp(as_of)]
        if px2.empty:
            continue
        cache[(ns(key[0]), ns(key[1]))] = {
            "prices": px2,
            "kw": dict(c.get("kw") or {}),
        }

    cb = closes_broad
    if cb is not None and not cb.empty:
        cb = cb.copy()
        if not isinstance(cb.index, pd.DatetimeIndex):
            cb.index = pd.to_datetime(cb.index)
        cb = cb.loc[cb.index <= pd.Timestamp(as_of)]

    pw, _wdf, meta = compute_v6_b4_pf_weight_dict(
        pair_cache=cache,
        v6_opt2_h_daily_map=h_map,
        screened_csv=str(screened_csv),
        closes_broad=cb,
        norm_sym=ns,
        get_ibkr_borrow_map=get_ibkr_borrow_map or _stub_ibkr,
        opt2_h_base=h_base,
        params=params,
        use_ibkr_uvix_borrow=bool(use_ibkr_uvix_borrow),
    )

    cb_cfg = opt2.get("crash_budget") or {}
    if cb_cfg.get("enabled", True):
        caps = compute_crash_caps(
            pair_cache=cache,
            hedge_by_underlying=h_map,
            closes_broad=cb,
            hedge_base=h_base,
            run_date=as_of,
            budget_usd=budget,
            params=CrashBudgetParams.from_config(cb_cfg),
            norm_sym=ns,
        )
        capped, budget_eff, tel = cap_pair_weights(pw, caps, budget, norm_sym=ns)
        telemetry = tel.to_dict(orient="records") if tel is not None and not tel.empty else []
    else:
        # Normalize opt2 to sum 1; full budget deployed.
        wsum = sum(max(0.0, float(v)) for v in pw.values()) or 1.0
        capped = {k: max(0.0, float(v)) / wsum for k, v in pw.items()}
        budget_eff = budget
        telemetry = []

    deployed = float(sum(capped.values()))
    return SizedBook(
        run_date=as_of,
        weights_opt2={(_norm_sym(k[0]), _norm_sym(k[1])): float(v) for k, v in pw.items()},
        weights_capped={(_norm_sym(k[0]), _norm_sym(k[1])): float(v) for k, v in capped.items()},
        budget_usd=budget,
        budget_eff=float(budget_eff),
        deployed_fraction=deployed,
        cash_residual=max(0.0, 1.0 - deployed),
        telemetry=telemetry,
        opt2_meta={k: v for k, v in (meta or {}).items() if not isinstance(v, (pd.DataFrame, pd.Series))},
    )


def build_pair_cache_from_panel(
    uni: pd.DataFrame,
    panel: Mapping[str, pd.DataFrame],
    *,
    norm_sym: NormSym | None = None,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Build the pair_cache shape expected by opt2 / crash_budget from dashboard panels."""
    ns = norm_sym or _norm_sym
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for _, row in uni.iterrows():
        etf = ns(row.get("ETF"))
        und = ns(row.get("Underlying"))
        px = panel.get(etf)
        if px is None or getattr(px, "empty", True):
            continue
        beta = float(pd.to_numeric(row.get("Delta"), errors="coerce") or -2.0)
        borrow = float(pd.to_numeric(row.get("borrow_current"), errors="coerce") or 0.0)
        if not np.isfinite(borrow) or borrow < 0:
            borrow = 0.0
        out[(etf, und)] = {
            "prices": px[["a_px", "b_px"]].copy() if "a_px" in px.columns else px.copy(),
            "kw": {
                "beta_a": -abs(beta),
                "beta_b": 1.0,
                "borrow_a_annual": borrow,
            },
        }
    return out


def build_closes_broad_from_panel(
    panel: Mapping[str, pd.DataFrame],
    uni: pd.DataFrame,
    *,
    norm_sym: NormSym | None = None,
) -> pd.DataFrame:
    """Wide underlying close panel for cov / crash lookthrough."""
    ns = norm_sym or _norm_sym
    cols: dict[str, pd.Series] = {}
    for _, row in uni.iterrows():
        etf = ns(row.get("ETF"))
        und = ns(row.get("Underlying"))
        px = panel.get(etf)
        if px is None or "b_px" not in getattr(px, "columns", []):
            continue
        if und in cols:
            continue
        cols[und] = pd.to_numeric(px["b_px"], errors="coerce")
    if not cols:
        return pd.DataFrame()
    return pd.DataFrame(cols).sort_index()


def weekly_rebalance_dates(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    *,
    freq: str = "W-FRI",
) -> list[pd.Timestamp]:
    idx = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq=freq)
    return [pd.Timestamp(d) for d in idx]


def build_dashboard_sizing(
    *,
    uni: pd.DataFrame,
    panel: Mapping[str, pd.DataFrame],
    hedge_by_underlying: Mapping[str, pd.Series],
    screened_csv: str | Path,
    opt2_cfg: Mapping[str, Any],
    sleeve_budget_usd: float = 100_000.0,
    run_date: str | pd.Timestamp | None = None,
) -> dict[str, Any]:
    """One-shot sizing payload for the latest (or given) as-of date."""
    cache = build_pair_cache_from_panel(uni, panel)
    closes = build_closes_broad_from_panel(panel, uni)
    as_of = run_date
    if as_of is None:
        # Latest common price date across panel.
        lasts = []
        for px in panel.values():
            if px is not None and len(px):
                lasts.append(pd.Timestamp(px.index.max()))
        as_of = max(lasts) if lasts else pd.Timestamp.today()
    sized = size_b4_book_asof(
        run_date=as_of,
        pair_cache=cache,
        hedge_by_underlying=hedge_by_underlying,
        closes_broad=closes,
        screened_csv=screened_csv,
        sleeve_budget_usd=sleeve_budget_usd,
        opt2_cfg=opt2_cfg,
        use_ibkr_uvix_borrow=False,
    )
    return sized.to_dict()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["dashboard", "self-check"], nargs="?", default="self-check")
    args = ap.parse_args(argv)
    if args.command == "self-check":
        # Smoke: crash budget + empty opt2 path not exercised without data.
        from scripts.b4_crash_budget import pair_loss

        assert pair_loss(0.3, 0.0, 2.0, 0.5) > 0
        print(json.dumps({"ok": True, "module": "bucket4_backtest_api"}))
        return 0
    print("dashboard payload requires etf-dashboard builder inputs; use size_b4_book_asof from Python", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
