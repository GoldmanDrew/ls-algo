"""Per-pair Bucket 4 rebalance due/defer gate for production ``rebalance_strategy``.

Operator cadence: run ``rebalance_strategy.py`` every ``operator_check_days`` (default 5)
business days. On each run, only Bucket-4 pairs that are *due* participate in Phase 2b
resize; others are deferred until their TR/VCR-derived ``interval_days`` elapses (or
``max_interval`` forces a catch-up).

State is persisted in ``cadence_state_json`` (default ``data/b4_cadence_state.json``):
  { "CLSZ|CLSK": "2026-05-28", ... }  # last successful B4 resize date per ETF|Underlying
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from execute_trade_plan import norm_sym
from scripts.bucket4_hedge_cadence import (
    HedgeCadenceKnobs,
    PairPolicy,
    compute_pair_policy,
    load_name_tilts,
    load_policy_from_config,
)


def _pair_key(etf: str, underlying: str) -> str:
    return f"{norm_sym(etf)}|{norm_sym(underlying)}"


def _parse_date(x: str | date | pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(x).normalize()


def trading_days_since(start: str | date | pd.Timestamp, end: str | date | pd.Timestamp) -> int:
    """Business days elapsed from ``start`` (exclusive) to ``end`` (inclusive)."""
    a = _parse_date(start)
    b = _parse_date(end)
    if b < a:
        return 0
    return int(np.busday_count(a.date(), b.date()))


def load_cadence_state(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    return {str(k): str(v) for k, v in raw.items()}


def save_cadence_state(path: Path, state: Mapping[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(state), indent=2, sort_keys=True) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class CadenceGateConfig:
    enabled: bool = False
    operator_check_days: int = 5
    state_json: Path = Path("data/b4_cadence_state.json")
    force_on_max_interval: bool = True
    # Machine-independent timing: backfill last_rebalance from the broker's Flex
    # trade record (and stagger-seed the rest) so the mutable state JSON never
    # has to be committed/pushed. See scripts/bucket4_cadence_reconstruct.py.
    reconstruct_from_broker: bool = True

    @classmethod
    def from_policy_block(cls, block: Mapping[str, Any] | None, *, opt2_enabled: bool) -> "CadenceGateConfig":
        b = dict(block or {})
        source = str(b.get("source", "tr_vcr")).strip().lower()
        return cls(
            enabled=bool(opt2_enabled) and source == "tr_vcr",
            operator_check_days=max(1, int(b.get("operator_check_days", 5))),
            state_json=Path(str(b.get("cadence_state_json", "data/b4_cadence_state.json"))),
            force_on_max_interval=bool(b.get("force_on_max_interval", True)),
            reconstruct_from_broker=bool(b.get("cadence_reconstruct_from_broker", True)),
        )


@dataclass(frozen=True)
class PairCadenceDecision:
    etf: str
    underlying: str
    due: bool
    reason: str
    interval_days: int
    trading_days_since_last: int | None
    last_rebalance: str | None
    hedge_ratio: float | None = None


def evaluate_pair_due(
    policy: PairPolicy,
    *,
    run_date: str | date | pd.Timestamp,
    last_rebalance: str | None,
    knobs: HedgeCadenceKnobs,
    force_on_max_interval: bool = True,
) -> PairCadenceDecision:
    """Return whether this pair should resize on ``run_date``."""
    rd = _parse_date(run_date)
    interval = int(policy.interval_days)
    if last_rebalance is None:
        return PairCadenceDecision(
            etf=policy.etf,
            underlying=policy.underlying,
            due=True,
            reason="no_prior_rebalance",
            interval_days=interval,
            trading_days_since_last=None,
            last_rebalance=None,
            hedge_ratio=policy.h,
        )
    bdays = trading_days_since(last_rebalance, rd)
    if bdays >= interval:
        reason = f"interval_elapsed({bdays}>={interval})"
        due = True
    elif force_on_max_interval and bdays >= int(knobs.max_interval):
        reason = f"max_interval_force({bdays}>={knobs.max_interval})"
        due = True
    else:
        reason = f"defer({bdays}<{interval})"
        due = False
    return PairCadenceDecision(
        etf=policy.etf,
        underlying=policy.underlying,
        due=due,
        reason=reason,
        interval_days=interval,
        trading_days_since_last=bdays,
        last_rebalance=last_rebalance,
        hedge_ratio=policy.h,
    )


def evaluate_cadence_gate(
    policies: Sequence[PairPolicy],
    *,
    run_date: str,
    state: Mapping[str, str],
    knobs: HedgeCadenceKnobs,
    force_on_max_interval: bool = True,
) -> tuple[set[tuple[str, str]], list[PairCadenceDecision]]:
    due_keys: set[tuple[str, str]] = set()
    decisions: list[PairCadenceDecision] = []
    for pol in policies:
        key = _pair_key(pol.etf, pol.underlying)
        dec = evaluate_pair_due(
            pol,
            run_date=run_date,
            last_rebalance=state.get(key),
            knobs=knobs,
            force_on_max_interval=force_on_max_interval,
        )
        decisions.append(dec)
        if dec.due:
            due_keys.add((pol.etf, pol.underlying))
    return due_keys, decisions


def filter_resize_plan_for_b4_cadence(
    resize_df: pd.DataFrame,
    due_keys: set[tuple[str, str]],
) -> pd.DataFrame:
    """Drop deferred B4 rows; B1/B2 rows unchanged."""
    if resize_df is None or resize_df.empty or "sleeve" not in resize_df.columns:
        return resize_df
    df = resize_df.copy()
    sleeve = df["sleeve"].astype(str).str.strip().str.lower()
    is_b4 = sleeve == "inverse_decay_bucket4"
    if not is_b4.any():
        return df

    def _keep(row: pd.Series) -> bool:
        if not is_b4.loc[row.name]:
            return True
        k = (norm_sym(str(row.get("ETF", ""))), norm_sym(str(row.get("Underlying", ""))))
        return k in due_keys

    out = df[df.apply(_keep, axis=1)].reset_index(drop=True)
    return out


def mark_pairs_rebalanced(
    state: dict[str, str],
    pairs: Sequence[tuple[str, str]],
    run_date: str,
) -> dict[str, str]:
    out = dict(state)
    rd = str(_parse_date(run_date).date())
    for etf, und in pairs:
        out[_pair_key(etf, und)] = rd
    return out


def policies_from_cadence_diag(
    cadence_by_underlying: Mapping[str, Mapping[str, Any]],
    pairs: Sequence[tuple[str, str]],
) -> list[PairPolicy]:
    """Build minimal PairPolicy rows from GTP ``cadence_by_underlying`` diagnostics."""
    out: list[PairPolicy] = []
    for etf, und in pairs:
        c = cadence_by_underlying.get(norm_sym(und), {}) or {}
        out.append(
            PairPolicy(
                etf=norm_sym(etf),
                underlying=norm_sym(und),
                tr=float(c.get("tr", np.nan)),
                vcr=float(c.get("vcr", np.nan)),
                vcr_med=float(c.get("vcr_med", np.nan)),
                signal_ok=bool(c.get("signal_ok", False)),
                h=float(c.get("hedge_ratio", np.nan)),
                h_raw=float(c.get("hedge_ratio", np.nan)),
                h_prev=None,
                interval_days=int(c.get("interval_days", 10)),
                interval_raw=float(c.get("interval_days", 10)),
                denom=float("nan"),
            )
        )
    return out


def _b4_pairs_from_resize_df(resize_df: pd.DataFrame) -> list[tuple[str, str]]:
    if resize_df is None or resize_df.empty or "sleeve" not in resize_df.columns:
        return []
    b4 = resize_df[resize_df["sleeve"].astype(str).str.strip().str.lower() == "inverse_decay_bucket4"]
    pairs = []
    for _, r in b4.iterrows():
        etf = norm_sym(str(r.get("ETF", "")))
        und = norm_sym(str(r.get("Underlying", "")))
        if etf and und:
            pairs.append((etf, und))
    return sorted(set(pairs))


def policies_from_explain_csv(path: Path) -> list[PairPolicy]:
    if not path.is_file():
        return []
    df = pd.read_csv(path)
    out: list[PairPolicy] = []
    for _, r in df.iterrows():
        out.append(
            PairPolicy(
                etf=norm_sym(str(r.get("ETF", ""))),
                underlying=norm_sym(str(r.get("Underlying", ""))),
                tr=float(r.get("tr", np.nan)),
                vcr=float(r.get("vcr", np.nan)),
                vcr_med=float(r.get("vcr_med", np.nan)),
                signal_ok=bool(r.get("signal_ok", False)),
                h=float(r.get("hedge_ratio", np.nan)),
                h_raw=float(r.get("hedge_ratio_raw", r.get("hedge_ratio", np.nan))),
                h_prev=None,
                interval_days=int(r.get("interval_days", 10)),
                interval_raw=float(r.get("interval_raw", r.get("interval_days", 10))),
                denom=float("nan"),
            )
        )
    return out


def resolve_due_pairs_for_rebalance(
    *,
    run_date: str,
    resize_df: pd.DataFrame,
    cfg: Mapping[str, Any] | None,
    run_dir_path: Path | None = None,
    persist_reconstruction: bool = False,
    runs_root: Path | None = None,
) -> tuple[set[tuple[str, str]] | None, list[PairCadenceDecision], CadenceGateConfig]:
    """Return due B4 pair keys, or ``None`` when the gate is disabled.

    Before evaluating, per-pair ``last_rebalance`` is backfilled from the
    broker's Flex trade record (and stagger-seeded when unknown) so cadence
    timing is machine-independent without committing the mutable state JSON.
    Set ``persist_reconstruction`` (only on real, non-dry runs) to write the
    reconciled cache back to disk.
    """
    gate, knobs, tilts = load_gate_config_from_strategy(cfg)
    if not gate.enabled:
        return None, [], gate

    pairs = _b4_pairs_from_resize_df(resize_df)
    if not pairs:
        return set(), [], gate

    policies: list[PairPolicy] = []
    explain = (run_dir_path / "b4_hedge_cadence" / "b4_hedge_cadence_explain.csv") if run_dir_path else None
    if explain is not None and explain.is_file():
        policies = policies_from_explain_csv(explain)
        have = {(p.etf, p.underlying) for p in policies}
        policies = [p for p in policies if (p.etf, p.underlying) in set(pairs)]
        missing = [k for k in pairs if k not in have]
    else:
        missing = list(pairs)

    if missing:
        from scripts.bucket4_hedge_cadence import _signal_for_underlying

        for etf, und in missing:
            sig = _signal_for_underlying(und, "2023-01-01", run_date, window=60)
            tilt = tilts.get(etf) or tilts.get(und)
            if sig is not None and not sig.empty:
                row = sig.dropna(subset=["vcr"]).tail(1)
                if len(row):
                    d = row.index[-1]
                    cadence_col = str(getattr(knobs, "cadence_signal_col", "tr") or "tr")
                    if cadence_col not in row:
                        cadence_col = "tr"
                    tr = float(row[cadence_col].iloc[-1]) if cadence_col in row else np.nan
                    vcr = float(row["vcr"].iloc[-1])
                    vm = float(row["vcr_med"].iloc[-1]) if "vcr_med" in row else np.nan
                    pol = compute_pair_policy(
                        tr,
                        vcr,
                        vm,
                        knobs=knobs,
                        name_tilt=tilt,
                        etf=etf,
                        underlying=und,
                        cadence_signal_col=cadence_col,
                    )
                else:
                    pol = compute_pair_policy(np.nan, np.nan, np.nan, knobs=knobs, name_tilt=tilt, etf=etf, underlying=und)
            else:
                pol = compute_pair_policy(np.nan, np.nan, np.nan, knobs=knobs, name_tilt=tilt, etf=etf, underlying=und)
            policies.append(pol)

    state = load_cadence_state(gate.state_json)
    if gate.reconstruct_from_broker and policies:
        try:
            from scripts.bucket4_cadence_reconstruct import reconcile_cadence_state

            pair_intervals = {
                (p.etf, p.underlying): int(p.interval_days) for p in policies
            }
            state, _prov = reconcile_cadence_state(
                state,
                pair_intervals=pair_intervals,
                run_date=run_date,
                runs_root=runs_root,
                enable_broker=True,
            )
            if persist_reconstruction:
                save_cadence_state(gate.state_json, state)
        except Exception as exc:  # never block a rebalance on reconstruction
            print(f"[B4-CADENCE] broker reconstruction skipped ({exc}); using local state only")

    due_keys, decisions = evaluate_cadence_gate(
        policies,
        run_date=run_date,
        state=state,
        knobs=knobs,
        force_on_max_interval=gate.force_on_max_interval,
    )
    return due_keys, decisions, gate


def load_gate_config_from_strategy(cfg: Mapping[str, Any] | None) -> tuple[CadenceGateConfig, HedgeCadenceKnobs, dict]:
    """Read gate + knobs from full strategy YAML."""
    rules = (
        (cfg or {})
        .get("portfolio", {})
        .get("sleeves", {})
        .get("inverse_decay_bucket4", {})
        .get("rules", {})
    )
    opt2 = rules.get("bucket4_weekly_opt2") or {}
    hcp = opt2.get("hedge_cadence_policy") or {}
    knobs, tilts, _ = load_policy_from_config(cfg)
    gate = CadenceGateConfig.from_policy_block(hcp, opt2_enabled=bool(opt2.get("enabled")))
    return gate, knobs, tilts


__all__ = [
    "CadenceGateConfig",
    "PairCadenceDecision",
    "evaluate_pair_due",
    "evaluate_cadence_gate",
    "filter_resize_plan_for_b4_cadence",
    "resolve_due_pairs_for_rebalance",
    "load_cadence_state",
    "save_cadence_state",
    "mark_pairs_rebalanced",
    "policies_from_cadence_diag",
    "load_gate_config_from_strategy",
    "trading_days_since",
]
