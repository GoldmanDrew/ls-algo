"""
Bucket 4 hedge-ratio + rebalance-cadence engine (single, explainable spot).

This is the ONE place that decides, for each Bucket-4 pair on a given day:
  * ``h``            -- the hedge ratio (how much underlying short per unit inverse-ETF beta)
  * ``interval_days``-- how many trading days until the next rebalance

Both are closed-form functions of the pair's own vol-shape signals (TR = trend
ratio, VCR = variance-contribution ratio), so a human (or an AI) can reverse-engineer
any value from the printed inputs:

    hedge ratio (production default: v6 Opt-2 panel when ``hedge_ratio_model: v6``):
        h_star = h_base - opt2_k * z_composite   # cross-section r_10d + range_expansion (+ regime)

    hedge ratio (optional v7 closed form when ``hedge_ratio_model: v7``):
        h_raw = h_mid + k_vcr * (VCR - VCR_med)
        h     = clip(h_raw * tilt.h_mult + tilt.h_shift, h_min, h_max)
        h_ema = (1-alpha)*prev_h + alpha*h        # optional smoothing

    cadence (continuous TR/VCR, always when ``source: tr_vcr``):
        denom    = 1 + k_tr*(TR - 1) + m_vcr*(VCR - VCR_med)
        interval = clip(round(base_days / denom * tilt.interval_mult),
                        min_interval, max_interval)

Design goals: simple, explainable, hard to vary unduly (few hard-clipped knobs),
and one sanctioned place for discretionary per-name tilt (``NameTilt``).

The same functions are used by production (``generate_trade_plan`` /
``bucket4_weekly_opt2``) and by the backtest, so the two can never diverge.

CLI (human-readable + plots):
    python -m scripts.bucket4_hedge_cadence --run-date 2026-06-01 --plots
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

# Reuse the already-tested v7 hedge mapping so prod/backtest share one definition.
try:
    from scripts.bucket4_hedge_v7 import V7_DEFAULT_H_MID, V7_GLOBAL_H_MAX, V7_GLOBAL_H_MIN
except Exception:  # pragma: no cover - allow standalone import
    V7_DEFAULT_H_MID, V7_GLOBAL_H_MIN, V7_GLOBAL_H_MAX = 0.55, 0.30, 0.80


def _norm_sym(x: str) -> str:
    return str(x).strip().upper().replace(".", "-")


# ----------------------------------------------------------------------------
# Knobs + per-name discretionary tilt (config-driven)
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class HedgeCadenceKnobs:
    """All operator-tunable numbers in one place (config: ``hedge_cadence_policy``)."""

    # --- hedge ratio (v7 form) ---
    h_mid: float = V7_DEFAULT_H_MID          # neutral hedge level (set to v6 base to mimic v6)
    k_vcr: float = 1.0                       # VCR sensitivity of h
    h_min: float = V7_GLOBAL_H_MIN           # hard guardrail (lower)
    h_max: float = V7_GLOBAL_H_MAX           # hard guardrail (upper)
    alpha: float = 0.25                      # EMA smoothing on the daily h series (0 = off)
    # --- cadence (continuous TR/VCR) ---
    base_days: float = 4.0
    k_tr: float = 2.25
    m_vcr: float = 2.5                       # mean-best A7 cadence sensitivity
    min_interval: int = 1
    max_interval: int = 10

    @classmethod
    def from_config(cls, block: Mapping[str, Any] | None) -> "HedgeCadenceKnobs":
        b = dict(block or {})
        d = cls()
        return cls(
            h_mid=float(b.get("h_mid", d.h_mid)),
            k_vcr=float(b.get("k_vcr", d.k_vcr)),
            h_min=float(b.get("h_min", d.h_min)),
            h_max=float(b.get("h_max", d.h_max)),
            alpha=float(b.get("alpha", d.alpha)),
            base_days=float(b.get("base_days", d.base_days)),
            k_tr=float(b.get("k_tr", d.k_tr)),
            m_vcr=float(b.get("m_vcr", d.m_vcr)),
            min_interval=int(b.get("min_interval", d.min_interval)),
            max_interval=int(b.get("max_interval", d.max_interval)),
        )


@dataclass(frozen=True)
class NameTilt:
    """Optional discretionary per-name override (the ONLY sanctioned hand-tune)."""

    h_shift: float = 0.0
    h_mult: float = 1.0
    interval_mult: float = 1.0
    note: str = ""

    @classmethod
    def from_config(cls, block: Mapping[str, Any] | None) -> "NameTilt":
        b = dict(block or {})
        return cls(
            h_shift=float(b.get("h_shift", 0.0)),
            h_mult=float(b.get("h_mult", 1.0)),
            interval_mult=float(b.get("interval_mult", 1.0)),
            note=str(b.get("note", "")),
        )

    @property
    def is_identity(self) -> bool:
        return (self.h_shift == 0.0 and self.h_mult == 1.0 and self.interval_mult == 1.0)


def load_name_tilts(block: Mapping[str, Any] | None) -> dict[str, NameTilt]:
    """Parse ``name_tilt: {SYM: {h_mult:..., note:...}}`` into a normalized map."""
    out: dict[str, NameTilt] = {}
    for sym, spec in dict(block or {}).items():
        out[_norm_sym(sym)] = NameTilt.from_config(spec)
    return out


# ----------------------------------------------------------------------------
# The result object: carries the numbers AND a human-readable explanation
# ----------------------------------------------------------------------------
@dataclass
class PairPolicy:
    etf: str
    underlying: str
    # inputs
    tr: float
    vcr: float
    vcr_med: float
    signal_ok: bool
    # hedge ratio
    h: float                 # final (post-tilt, clip, EMA)
    h_raw: float             # pre-clip, pre-EMA
    h_prev: float | None
    # cadence
    interval_days: int
    interval_raw: float
    denom: float
    # provenance
    tilt_note: str = ""
    h_explain: str = ""
    interval_explain: str = ""

    def as_row(self) -> dict[str, Any]:
        return {
            "ETF": self.etf,
            "Underlying": self.underlying,
            "tr": round(self.tr, 4) if np.isfinite(self.tr) else np.nan,
            "vcr": round(self.vcr, 5) if np.isfinite(self.vcr) else np.nan,
            "vcr_med": round(self.vcr_med, 5) if np.isfinite(self.vcr_med) else np.nan,
            "hedge_ratio": round(self.h, 4),
            "hedge_ratio_raw": round(self.h_raw, 4),
            "interval_days": int(self.interval_days),
            "interval_raw": round(self.interval_raw, 3),
            "signal_ok": bool(self.signal_ok),
            "tilt_note": self.tilt_note,
            "h_explain": self.h_explain,
            "interval_explain": self.interval_explain,
        }


def _clip(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


# ----------------------------------------------------------------------------
# THE core function: inputs (TR, VCR, VCR_med) -> outputs (h, interval_days)
# ----------------------------------------------------------------------------
def compute_pair_policy(
    tr: float,
    vcr: float,
    vcr_med: float,
    *,
    knobs: HedgeCadenceKnobs,
    name_tilt: NameTilt | None = None,
    prev_h: float | None = None,
    etf: str = "",
    underlying: str = "",
) -> PairPolicy:
    """Closed-form hedge ratio + rebalance interval for one pair on one day.

    Every returned value is reconstructable from ``h_explain`` / ``interval_explain``.
    Missing/NaN signals fall back to the neutral prior (h_mid, base_days) and say so.
    """
    tilt = name_tilt or NameTilt()
    t = float(tr) if tr is not None and np.isfinite(tr) else np.nan
    v = float(vcr) if vcr is not None and np.isfinite(vcr) else np.nan
    vm = float(vcr_med) if vcr_med is not None and np.isfinite(vcr_med) else np.nan
    signal_ok = np.isfinite(v) and np.isfinite(vm)

    # ---------------- hedge ratio (v7 form) ----------------
    if signal_ok:
        dvcr = v - vm
        h_raw = knobs.h_mid + knobs.k_vcr * dvcr
        h_src = (
            f"h_mid({knobs.h_mid:.3f}) + k_vcr({knobs.k_vcr:.2f})*"
            f"(VCR({v:.5f})-VCR_med({vm:.5f})={dvcr:+.5f}) = {h_raw:.4f}"
        )
    else:
        h_raw = knobs.h_mid
        h_src = f"signal missing -> neutral h_mid({knobs.h_mid:.3f})"

    h_tilted = h_raw * tilt.h_mult + tilt.h_shift
    tilt_part = ""
    if not tilt.is_identity:
        tilt_part = f" -> tilt(x{tilt.h_mult:.2f}{tilt.h_shift:+.3f})={h_tilted:.4f}"
    h_clipped = _clip(h_tilted, knobs.h_min, knobs.h_max)
    clip_part = f" -> clip[{knobs.h_min:.2f},{knobs.h_max:.2f}]={h_clipped:.4f}"

    if prev_h is not None and np.isfinite(prev_h) and 0.0 < knobs.alpha < 1.0:
        h_final = (1.0 - knobs.alpha) * float(prev_h) + knobs.alpha * h_clipped
        h_final = _clip(h_final, knobs.h_min, knobs.h_max)
        ema_part = (
            f" -> EMA(a={knobs.alpha:.2f}; prev={float(prev_h):.4f})={h_final:.4f}"
        )
    else:
        h_final = h_clipped
        ema_part = ""

    h_explain = f"h={h_final:.4f}  |  {h_src}{tilt_part}{clip_part}{ema_part}"

    # ---------------- cadence (continuous TR/VCR) ----------------
    denom = 1.0
    denom_terms = ["1"]
    if np.isfinite(t):
        denom += knobs.k_tr * (t - 1.0)
        denom_terms.append(f"k_tr({knobs.k_tr:.2f})*(TR({t:.3f})-1)={knobs.k_tr*(t-1.0):+.4f}")
    if np.isfinite(v) and np.isfinite(vm):
        denom += knobs.m_vcr * (v - vm)
        denom_terms.append(f"m_vcr({knobs.m_vcr:.2f})*(VCR-VCR_med)={knobs.m_vcr*(v-vm):+.4f}")

    if denom > 1e-9:
        interval_raw = knobs.base_days / denom
    else:
        interval_raw = float(knobs.max_interval)
    interval_tilted = interval_raw * tilt.interval_mult
    interval_days = int(np.clip(round(interval_tilted), knobs.min_interval, knobs.max_interval))

    tilt_cad = ""
    if tilt.interval_mult != 1.0:
        tilt_cad = f" *tilt({tilt.interval_mult:.2f})={interval_tilted:.3f}"
    interval_explain = (
        f"interval={interval_days}d  |  denom=" + " + ".join(denom_terms) + f" = {denom:.4f}"
        f" -> base_days({knobs.base_days:.1f})/denom={interval_raw:.3f}{tilt_cad}"
        f" -> round->clip[{knobs.min_interval},{knobs.max_interval}]={interval_days}"
    )

    return PairPolicy(
        etf=_norm_sym(etf),
        underlying=_norm_sym(underlying),
        tr=t,
        vcr=v,
        vcr_med=vm,
        signal_ok=bool(signal_ok),
        h=float(h_final),
        h_raw=float(h_raw),
        h_prev=(float(prev_h) if prev_h is not None else None),
        interval_days=int(interval_days),
        interval_raw=float(interval_raw),
        denom=float(denom),
        tilt_note=tilt.note,
        h_explain=h_explain,
        interval_explain=interval_explain,
    )


# ----------------------------------------------------------------------------
# Series builders (drop-in replacements for the v6 panel / cadence policy)
# ----------------------------------------------------------------------------
def build_h_series(
    signal: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    *,
    knobs: HedgeCadenceKnobs,
    name_tilt: NameTilt | None = None,
) -> pd.Series:
    """Daily hedge-ratio series on *calendar* with EMA smoothing carried forward."""
    cal = pd.DatetimeIndex(calendar).sort_values()
    if len(cal) == 0:
        return pd.Series(dtype=float)
    tr = signal.get("tr") if signal is not None else None
    vcr = signal.get("vcr") if signal is not None else None
    vm = signal.get("vcr_med") if signal is not None else None
    out = pd.Series(index=cal, dtype=float)
    prev_h: float | None = None
    for d in cal:
        pol = compute_pair_policy(
            float(tr.get(d, np.nan)) if tr is not None else np.nan,
            float(vcr.get(d, np.nan)) if vcr is not None else np.nan,
            float(vm.get(d, np.nan)) if vm is not None else np.nan,
            knobs=knobs,
            name_tilt=name_tilt,
            prev_h=prev_h,
        )
        out.loc[d] = pol.h
        prev_h = pol.h
    return out.astype(float)


def build_rebal_dates(
    signal: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    *,
    knobs: HedgeCadenceKnobs,
    name_tilt: NameTilt | None = None,
    warmup_bdays: int = 0,
) -> tuple[pd.DatetimeIndex, pd.DataFrame]:
    """Rebalance schedule by stepping the per-day interval. Returns (dates, diag).

    ``diag`` has one row per chosen rebalance date with the inputs and the
    human-readable ``interval_explain`` so the schedule is auditable.
    """
    cal = pd.DatetimeIndex(calendar).sort_values().unique()
    if warmup_bdays > 0:
        cal = cal[int(warmup_bdays):]
    if len(cal) == 0:
        return pd.DatetimeIndex([]), pd.DataFrame()
    tr = signal.get("tr") if signal is not None else None
    vcr = signal.get("vcr") if signal is not None else None
    vm = signal.get("vcr_med") if signal is not None else None
    dates: list[pd.Timestamp] = []
    diag: list[dict[str, Any]] = []
    i, n = 0, len(cal)
    while i < n:
        d = pd.Timestamp(cal[i])
        dates.append(d)
        pol = compute_pair_policy(
            float(tr.get(d, np.nan)) if tr is not None else np.nan,
            float(vcr.get(d, np.nan)) if vcr is not None else np.nan,
            float(vm.get(d, np.nan)) if vm is not None else np.nan,
            knobs=knobs,
            name_tilt=name_tilt,
        )
        diag.append({
            "date": d, "tr": pol.tr, "vcr": pol.vcr, "vcr_med": pol.vcr_med,
            "interval_days": pol.interval_days, "interval_explain": pol.interval_explain,
        })
        i += max(1, pol.interval_days)
    return pd.DatetimeIndex(dates), pd.DataFrame(diag)


def build_hedge_by_underlying(
    signals_by_key: Mapping[tuple[str, str], pd.DataFrame],
    calendar_by_key: Mapping[tuple[str, str], pd.DatetimeIndex],
    *,
    knobs: HedgeCadenceKnobs,
    name_tilts: Mapping[str, NameTilt] | None = None,
) -> dict[str, pd.Series]:
    """``{underlying: h_series}`` for all keys (drop-in for ``panel_to_hedge_by_underlying``).

    Per-name tilt is looked up by ETF first, then underlying.
    """
    tilts = dict(name_tilts or {})
    out: dict[str, pd.Series] = {}
    for (etf, und), sig in signals_by_key.items():
        cal = calendar_by_key.get((etf, und))
        if cal is None or len(cal) == 0:
            continue
        tilt = tilts.get(_norm_sym(etf)) or tilts.get(_norm_sym(und))
        out[_norm_sym(und)] = build_h_series(sig, cal, knobs=knobs, name_tilt=tilt)
    return out


# ----------------------------------------------------------------------------
# Human-readable rendering helpers (used by the CLI and by other scripts)
# ----------------------------------------------------------------------------
def render_policy_table(policies: Sequence[PairPolicy]) -> str:
    """One compact, readable line per pair: inputs -> outputs + the 'because Z'."""
    lines = []
    header = f"{'PAIR':<14}{'TR':>7}{'VCR':>9}{'VCRmed':>9}{'h':>7}{'days':>6}   why (days)"
    lines.append(header)
    lines.append("-" * len(header))
    for p in sorted(policies, key=lambda x: x.interval_days):
        pair = f"{p.etf}/{p.underlying}"
        tr = f"{p.tr:.3f}" if np.isfinite(p.tr) else "  n/a"
        vcr = f"{p.vcr:.5f}" if np.isfinite(p.vcr) else "    n/a"
        vm = f"{p.vcr_med:.5f}" if np.isfinite(p.vcr_med) else "    n/a"
        why = _short_reason(p)
        lines.append(f"{pair:<14}{tr:>7}{vcr:>9}{vm:>9}{p.h:>7.3f}{p.interval_days:>6}   {why}")
    return "\n".join(lines)


def _short_reason(p: PairPolicy) -> str:
    if not p.signal_ok:
        return "neutral (signal missing)"
    bits = []
    if np.isfinite(p.tr):
        if p.tr > 1.05:
            bits.append("trending(TR>1)->faster")
        elif p.tr < 0.95:
            bits.append("choppy(TR<1)->slower")
    if np.isfinite(p.vcr) and np.isfinite(p.vcr_med):
        if p.vcr > p.vcr_med:
            bits.append("VCR>med->faster+more hedged")
        elif p.vcr < p.vcr_med:
            bits.append("VCR<med->slower+less hedged")
    return ", ".join(bits) if bits else "near baseline"


# ----------------------------------------------------------------------------
# Config loader
# ----------------------------------------------------------------------------
def load_policy_from_config(cfg: Mapping[str, Any] | None) -> tuple[HedgeCadenceKnobs, dict[str, NameTilt], str]:
    """Read ``portfolio.sleeves.inverse_decay_bucket4.rules.hedge_cadence_policy``.

    Returns (knobs, name_tilts, source) where source is 'tr_vcr' (new engine) or
    'v6_panel' (legacy). Missing config -> sensible defaults + 'tr_vcr'.
    """
    block: Mapping[str, Any] = {}
    try:
        rules = (
            (cfg or {})
            .get("portfolio", {})
            .get("sleeves", {})
            .get("inverse_decay_bucket4", {})
            .get("rules", {})
        )
        # Accept both rules.hedge_cadence_policy and rules.bucket4_weekly_opt2.hedge_cadence_policy
        block = (
            rules.get("hedge_cadence_policy")
            or (rules.get("bucket4_weekly_opt2", {}) or {}).get("hedge_cadence_policy")
            or {}
        )
    except Exception:
        block = {}
    knobs = HedgeCadenceKnobs.from_config(block)
    tilts = load_name_tilts(block.get("name_tilt"))
    source = str(block.get("source", "tr_vcr")).strip().lower()
    return knobs, tilts, source


# ----------------------------------------------------------------------------
# CLI: run the engine for current B4 pairs, print human-readable, optional plots
# ----------------------------------------------------------------------------
def _underlying_prices_from_pair_cache(pair_cache_path: str) -> dict[str, pd.Series]:
    """Load underlying close series from a v6 pair-cache pickle (offline, exact prices).

    Pair cache keys are (etf, und); each value has ``prices`` (DataFrame with a 'b_px'
    underlying-close column). Returns ``{underlying: close_series}``.
    """
    import pickle

    out: dict[str, pd.Series] = {}
    p = Path(pair_cache_path)
    if not p.is_file():
        return out
    try:
        with open(p, "rb") as fh:
            cache = pickle.load(fh)
    except Exception:
        return out
    for key, payload in (cache or {}).items():
        try:
            etf, und = key
        except Exception:
            continue
        if not isinstance(payload, dict) or "skip_reason" in payload:
            continue
        prices = payload.get("prices")
        if not isinstance(prices, pd.DataFrame) or "b_px" not in prices.columns:
            continue
        out[_norm_sym(und)] = pd.to_numeric(prices["b_px"], errors="coerce").dropna()
    return out


def _signal_for_underlying(
    und: str,
    start: str,
    end: str | None,
    *,
    window: int,
    price_series: pd.Series | None = None,
) -> pd.DataFrame:
    """Recompute TR/VCR for an underlying from prices (production path).

    Robust: any failure (delisted ticker, no cache, network) returns an empty frame
    so the caller falls back to a neutral policy instead of crashing the whole run.
    """
    from scripts.bucket4_vol_shape_signals import get_pair_signal

    px = price_series
    if px is None or len(px) == 0:
        try:
            from scripts.bucket4_price_loading import load_single_close

            px = load_single_close(und, start, end)
        except Exception as e:  # delisted / no cache / network
            print(f"[b4-cadence] WARN no prices for {und} ({type(e).__name__}); using neutral policy")
            return pd.DataFrame(columns=["tr", "vcr", "vcr_med"])
    if px is None or len(px) == 0:
        return pd.DataFrame(columns=["tr", "vcr", "vcr_med"])
    try:
        cal = pd.DatetimeIndex(px.index)
        return get_pair_signal(und, und, cal, history={}, underlying_prices=px,
                               window=window, lookahead_shift=1)
    except Exception as e:
        print(f"[b4-cadence] WARN signal recompute failed for {und} ({type(e).__name__}); neutral policy")
        return pd.DataFrame(columns=["tr", "vcr", "vcr_med"])


def _b4_pairs_from_proposed(run_date: str) -> list[tuple[str, str]]:
    """Read inverse_decay_bucket4 (ETF, Underlying) pairs from proposed_trades.csv."""
    candidates = [
        Path("data") / "runs" / run_date / "proposed_trades.csv",
        Path("data") / "proposed_trades.csv",
    ]
    for p in candidates:
        if p.is_file():
            df = pd.read_csv(p)
            if "sleeve" in df.columns:
                df = df[df["sleeve"].astype(str).str.strip().str.lower() == "inverse_decay_bucket4"]
            pairs = []
            for _, r in df.iterrows():
                etf = _norm_sym(r.get("ETF", r.get("etf", "")))
                und = _norm_sym(r.get("Underlying", r.get("underlying", "")))
                if etf and und:
                    pairs.append((etf, und))
            if pairs:
                return sorted(set(pairs))
    return []


def _save_plots(policy_rows: list[PairPolicy], series_by_und: dict[str, pd.DataFrame],
                knobs: HedgeCadenceKnobs, tilts: dict[str, NameTilt], out_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    # 1) Days-to-rebalance bar (current snapshot)
    fig, ax = plt.subplots(figsize=(9, 4.2), constrained_layout=True)
    pol_sorted = sorted(policy_rows, key=lambda x: x.interval_days)
    labels = [f"{p.etf}/{p.underlying}" for p in pol_sorted]
    vals = [p.interval_days for p in pol_sorted]
    colors = ["#1f6f54" if p.signal_ok else "#b0b7c3" for p in pol_sorted]
    ax.bar(labels, vals, color=colors)
    for i, p in enumerate(pol_sorted):
        ax.text(i, p.interval_days + 0.1, f"{p.interval_days}d", ha="center", fontsize=8)
    ax.set_ylabel("days to rebalance")
    ax.set_title(f"Bucket 4 — days to rebalance (base={knobs.base_days:g}, k_tr={knobs.k_tr:g}, m_vcr={knobs.m_vcr:g}, cap={knobs.max_interval})")
    ax.tick_params(axis="x", rotation=45)
    f1 = out_dir / "b4_days_to_rebalance.png"
    fig.savefig(f1, dpi=130)
    plt.close(fig)
    saved.append(f1)

    # 2) Hedge ratio over time, per underlying (history -> + current marker)
    fig, ax = plt.subplots(figsize=(9, 4.6), constrained_layout=True)
    for (und, sig) in series_by_und.items():
        if sig is None or sig.empty:
            continue
        tilt = tilts.get(und)
        cal = pd.DatetimeIndex(sig.index)
        hser = build_h_series(sig, cal, knobs=knobs, name_tilt=tilt).dropna()
        if len(hser):
            ax.plot(hser.index, hser.values, label=und, linewidth=1.3)
    ax.axhline(knobs.h_min, color="#9b2c2c", ls=":", lw=0.8)
    ax.axhline(knobs.h_max, color="#9b2c2c", ls=":", lw=0.8)
    ax.set_ylabel("hedge ratio h")
    ax.set_title(f"Bucket 4 — hedge ratio over time (h_mid={knobs.h_mid:g}, k_vcr={knobs.k_vcr:g}, EMA a={knobs.alpha:g})")
    ax.legend(fontsize=7, ncol=4)
    f2 = out_dir / "b4_hedge_ratio_over_time.png"
    fig.savefig(f2, dpi=130)
    plt.close(fig)
    saved.append(f2)

    return saved


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Bucket 4 hedge-ratio + cadence engine (explainable).")
    ap.add_argument("--run-date", default=date.today().isoformat())
    ap.add_argument("--start", default="2023-01-01", help="history start for TR/VCR recompute")
    ap.add_argument("--window", type=int, default=60, help="vol-shape rolling window (days)")
    ap.add_argument("--pairs", nargs="*", default=None,
                    help="ETF:UND pairs, e.g. CLSZ:CLSK APLZ:APLD (default: read proposed_trades.csv)")
    ap.add_argument("--plots", action="store_true", help="save PNG plots")
    ap.add_argument("--out-dir", default=None, help="output dir (default data/runs/<date>/b4_hedge_cadence)")
    ap.add_argument("--verbose", action="store_true", help="print full arithmetic trace per pair")
    ap.add_argument("--pair-cache", default=None,
                    help="offline price source: v6 pair-cache pickle (e.g. notebooks/v6_b4_gtp_cache.pkl)")
    ap.add_argument("--cache-dirs", nargs="*", default=None,
                    help="CSV price-cache dirs to search before hitting Yahoo")
    args = ap.parse_args(argv)

    if args.cache_dirs:
        try:
            from scripts.bucket4_price_loading import configure_price_cache_dirs
            configure_price_cache_dirs(args.cache_dirs)
        except Exception:
            pass
    prices_by_und: dict[str, pd.Series] = {}
    if args.pair_cache:
        prices_by_und = _underlying_prices_from_pair_cache(args.pair_cache)
        print(f"[b4-cadence] loaded {len(prices_by_und)} underlying price series from {args.pair_cache}")

    # config knobs + tilts
    try:
        from strategy_config import load_config
        cfg = load_config()
    except Exception:
        cfg = {}
    knobs, tilts, source = load_policy_from_config(cfg)

    # pairs
    if args.pairs:
        pairs = [tuple(_norm_sym(x) for x in p.split(":")) for p in args.pairs]
    else:
        pairs = _b4_pairs_from_proposed(args.run_date)
    if not pairs:
        print("[b4-cadence] no Bucket 4 pairs found (pass --pairs ETF:UND ...). Nothing to do.")
        return 1

    print(f"[b4-cadence] run_date={args.run_date} source={source} pairs={len(pairs)}")
    print(f"[b4-cadence] knobs: h_mid={knobs.h_mid} k_vcr={knobs.k_vcr} h_min={knobs.h_min} "
          f"h_max={knobs.h_max} alpha={knobs.alpha} | base_days={knobs.base_days} k_tr={knobs.k_tr} "
          f"m_vcr={knobs.m_vcr} cap={knobs.max_interval}")
    print()

    policies: list[PairPolicy] = []
    series_by_und: dict[str, pd.DataFrame] = {}
    for etf, und in pairs:
        sig = _signal_for_underlying(und, args.start, args.run_date, window=args.window,
                                     price_series=prices_by_und.get(_norm_sym(und)))
        series_by_und[und] = sig
        tilt = tilts.get(etf) or tilts.get(und)
        if sig is not None and not sig.empty:
            row = sig.dropna(subset=["vcr"]).tail(1)
            if len(row):
                d = row.index[-1]
                tr = float(row["tr"].iloc[-1]) if "tr" in row else np.nan
                vcr = float(row["vcr"].iloc[-1])
                vm = float(row["vcr_med"].iloc[-1]) if "vcr_med" in row else np.nan
                # prev_h from one EMA step behind (build full series, take penultimate)
                hser = build_h_series(sig, pd.DatetimeIndex(sig.index), knobs=knobs, name_tilt=tilt).dropna()
                prev_h = float(hser.iloc[-2]) if len(hser) >= 2 else None
                pol = compute_pair_policy(tr, vcr, vm, knobs=knobs, name_tilt=tilt,
                                          prev_h=prev_h, etf=etf, underlying=und)
            else:
                pol = compute_pair_policy(np.nan, np.nan, np.nan, knobs=knobs, name_tilt=tilt,
                                          etf=etf, underlying=und)
        else:
            pol = compute_pair_policy(np.nan, np.nan, np.nan, knobs=knobs, name_tilt=tilt,
                                      etf=etf, underlying=und)
        policies.append(pol)

    print(render_policy_table(policies))
    print()
    if args.verbose:
        for p in sorted(policies, key=lambda x: x.interval_days):
            print(f"== {p.etf}/{p.underlying} ==")
            print(f"   {p.h_explain}")
            print(f"   {p.interval_explain}")
            if p.tilt_note:
                print(f"   tilt: {p.tilt_note}")
        print()

    # outputs
    out_dir = Path(args.out_dir) if args.out_dir else (Path("data") / "runs" / args.run_date / "b4_hedge_cadence")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "b4_hedge_cadence_explain.csv"
    pd.DataFrame([p.as_row() for p in policies]).to_csv(csv_path, index=False)
    print(f"[b4-cadence] wrote {csv_path}")

    if args.plots:
        saved = _save_plots(policies, series_by_und, knobs, tilts, out_dir)
        for s in saved:
            print(f"[b4-cadence] wrote {s}")

    # rebalance-day call-to-action (which to run today)
    today_due = [p for p in policies if p.interval_days <= 1]
    print()
    print("[b4-cadence] REBALANCE GUIDANCE:")
    for p in sorted(policies, key=lambda x: x.interval_days):
        print(f"   {p.etf}/{p.underlying}: rebalance every ~{p.interval_days} trading day(s); "
              f"h={p.h:.3f}  ({_short_reason(p)})")
    return 0


__all__ = [
    "HedgeCadenceKnobs",
    "NameTilt",
    "PairPolicy",
    "compute_pair_policy",
    "build_h_series",
    "build_rebal_dates",
    "build_hedge_by_underlying",
    "render_policy_table",
    "load_policy_from_config",
    "load_name_tilts",
]


if __name__ == "__main__":
    raise SystemExit(main())
