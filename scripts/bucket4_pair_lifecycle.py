"""Bucket 4 pair lifecycle: demotion ladder normal -> half -> freeze -> exit (Phase 2).

Turns the report-only flags from ``scripts.bucket4_pair_monitor`` into sizing
actions consumed by ``generate_trade_plan``:

  normal  full decay-score weight
  half    weight x ``half_weight_mult`` (default 0.5)
  freeze  weight 0 + purgatory=True (keep-open: no new size, no auto-close)
  exit    row dropped from the plan (Phase 1 cleanup closes the pair);
          re-entry blocked for ``reentry_cooldown_days`` business days

Escalation is immediate (a pair can jump straight to exit). Recovery is slow:
a pair must post ``recover_obs_days`` consecutive flag-free monitor runs to be
promoted ONE level. Exit only clears after the cooldown.

State: ``data/b4_pair_lifecycle_state.json``
  { "ETF|UND": {"status": "freeze", "since": "2026-06-01", "reason": "...",
                 "clean_days": 3, "exited_on": null} }

CLI (run after the monitor, e.g. daily):
  python -m scripts.bucket4_pair_lifecycle            # update state from latest monitor csv
  python -m scripts.bucket4_pair_lifecycle --dry-run  # show transitions only
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

STATUS_ORDER = ["normal", "half", "freeze", "exit"]


def norm_sym(x: str) -> str:
    return str(x).strip().upper().replace(".", "-")


def _pair_key(etf: str, und: str) -> str:
    return f"{norm_sym(etf)}|{norm_sym(und)}"


@dataclass(frozen=True)
class LifecycleConfig:
    enabled: bool = False
    state_json: Path = Path("data/b4_pair_lifecycle_state.json")
    min_obs_days: int = 20
    half_capture_lt: float = 0.25
    half_ret_ok: float = 0.10
    half_weight_mult: float = 0.5
    freeze_ret_lt: float = -0.15
    exit_ret_lt: float = -0.30
    recover_obs_days: int = 10
    reentry_cooldown_days: int = 45

    @classmethod
    def from_rules(cls, b4_rules: dict | None) -> "LifecycleConfig":
        lc = (b4_rules or {}).get("pair_lifecycle") or {}
        d = cls()
        return cls(
            enabled=bool(lc.get("enabled", d.enabled)),
            state_json=Path(str(lc.get("state_json", d.state_json))),
            min_obs_days=int(lc.get("min_obs_days", d.min_obs_days)),
            half_capture_lt=float(lc.get("half_capture_lt", d.half_capture_lt)),
            half_ret_ok=float(lc.get("half_ret_ok", d.half_ret_ok)),
            half_weight_mult=float(lc.get("half_weight_mult", d.half_weight_mult)),
            freeze_ret_lt=float(lc.get("freeze_ret_lt", d.freeze_ret_lt)),
            exit_ret_lt=float(lc.get("exit_ret_lt", d.exit_ret_lt)),
            recover_obs_days=int(lc.get("recover_obs_days", d.recover_obs_days)),
            reentry_cooldown_days=int(lc.get("reentry_cooldown_days", d.reentry_cooldown_days)),
        )


def load_state(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def save_state(path: Path, state: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _target_status(row: pd.Series) -> tuple[str, str]:
    """Worst (most severe) status implied by today's monitor flags."""
    if bool(row.get("flag_exit", False)):
        return "exit", str(
            "borrow>decay" if bool(row.get("borrow_exceeds_decay", False)) else "ret<exit_thr"
        )
    if bool(row.get("flag_freeze", False)) or bool(row.get("flag_vol_floor", False)):
        return "freeze", ("vol<keep_floor" if bool(row.get("flag_vol_floor", False)) else "ret<freeze_thr")
    if bool(row.get("flag_half", False)):
        return "half", "edge_capture<thr"
    return "normal", ""


def update_lifecycle(
    monitor: pd.DataFrame,
    state: dict[str, dict],
    cfg: LifecycleConfig,
    run_date: str,
) -> tuple[dict[str, dict], pd.DataFrame]:
    """Advance the per-pair state machine one monitor observation."""
    new_state = {k: dict(v) for k, v in state.items()}
    actions: list[dict] = []

    seen: set[str] = set()
    for _, row in monitor.iterrows():
        key = _pair_key(str(row["etf"]), str(row["underlying"]))
        seen.add(key)
        st = new_state.get(key) or {"status": "normal", "since": run_date, "reason": "",
                                    "clean_days": 0, "exited_on": None}
        old = str(st.get("status", "normal"))

        # Exit cooldown: stay out until cooldown elapses, then drop from state (re-entry allowed)
        if old == "exit":
            exited_on = st.get("exited_on") or st.get("since") or run_date
            bd_out = int(np.busday_count(pd.Timestamp(exited_on).date(), pd.Timestamp(run_date).date()))
            if bd_out >= cfg.reentry_cooldown_days:
                actions.append({"pair": key, "from": "exit", "to": "(cleared)",
                                "reason": f"cooldown {bd_out}bd elapsed"})
                new_state.pop(key, None)
            else:
                new_state[key] = st
            continue

        tgt, why = _target_status(row)
        if STATUS_ORDER.index(tgt) > STATUS_ORDER.index(old):
            # escalate immediately (can jump levels)
            st.update({"status": tgt, "since": run_date, "reason": why, "clean_days": 0})
            if tgt == "exit":
                st["exited_on"] = run_date
            actions.append({"pair": key, "from": old, "to": tgt, "reason": why})
        elif tgt == "normal" and old != "normal":
            # clean day: promote one level only after recover_obs_days consecutive clean days
            st["clean_days"] = int(st.get("clean_days", 0)) + 1
            if st["clean_days"] >= cfg.recover_obs_days:
                promoted = STATUS_ORDER[STATUS_ORDER.index(old) - 1]
                st.update({"status": promoted, "since": run_date,
                           "reason": f"recovered ({cfg.recover_obs_days} clean)", "clean_days": 0})
                actions.append({"pair": key, "from": old, "to": promoted, "reason": "recovery"})
        elif tgt != "normal":
            # same or lower severity while still flagged: hold, reset clean counter
            st["clean_days"] = 0
        new_state[key] = st

    return new_state, pd.DataFrame(actions)


# ---------------------------------------------------------------------------
# generate_trade_plan integration
# ---------------------------------------------------------------------------
def lifecycle_sets(state: dict[str, dict]) -> dict[str, set[str]]:
    """{'half': {ETF,...}, 'freeze': {...}, 'exit': {...}} keyed by ETF symbol."""
    out: dict[str, set[str]] = {"half": set(), "freeze": set(), "exit": set()}
    for key, st in (state or {}).items():
        etf = key.split("|", 1)[0]
        status = str(st.get("status", "normal"))
        if status in out:
            out[status].add(etf)
    return out


def held_etfs(state: dict[str, dict]) -> set[str]:
    """ETFs registered in lifecycle state (proxy for 'currently held' on keep-side gates)."""
    return {key.split("|", 1)[0] for key in (state or {})}


def apply_lifecycle_to_b4(
    b4c: pd.DataFrame,
    w: np.ndarray,
    state: dict[str, dict],
    cfg: LifecycleConfig,
) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Apply exit/freeze/half to the B4 core frame + weights. Renormalizes weights.

    Returns (frame, weights, info). Exit rows are DROPPED (plan omits them ->
    Phase 1 cleanup closes). Freeze rows stay with weight 0 and purgatory=True
    (keep-open). Half rows get weight x mult before renormalization.
    """
    if b4c.empty or not cfg.enabled:
        return b4c, w, {"n_exit": 0, "n_freeze": 0, "n_half": 0}
    sets = lifecycle_sets(state)
    etfs = b4c["ETF"].astype(str).map(norm_sym)

    keep_mask = ~etfs.isin(sets["exit"])
    n_exit = int((~keep_mask).sum())
    b4c = b4c.loc[keep_mask].copy()
    w = np.asarray(w, dtype=float)[keep_mask.to_numpy()]
    etfs = etfs.loc[keep_mask]

    frozen = etfs.isin(sets["freeze"]).to_numpy()
    halved = etfs.isin(sets["half"]).to_numpy() & ~frozen
    w = np.where(frozen, 0.0, w)
    w = np.where(halved, w * float(cfg.half_weight_mult), w)
    tot = float(w.sum())
    if tot > 1e-12:
        w = w / tot
    if frozen.any() and "purgatory" in b4c.columns:
        b4c.loc[frozen, "purgatory"] = True
    info = {"n_exit": n_exit, "n_freeze": int(frozen.sum()), "n_half": int(halved.sum())}
    return b4c, w, info


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _latest_monitor_csv(runs_root: Path) -> Path | None:
    cands = sorted(runs_root.glob("*/b4_monitor/b4_pair_monitor.csv"))
    return cands[-1] if cands else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Bucket 4 pair lifecycle state updater.")
    ap.add_argument("--runs-root", type=Path, default=REPO / "data/runs")
    ap.add_argument("--monitor-csv", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    try:
        from strategy_config import load_config
        cfg_all = load_config()
    except Exception:
        cfg_all = {}
    b4_rules = (
        (cfg_all or {}).get("portfolio", {}).get("sleeves", {})
        .get("inverse_decay_bucket4", {}).get("rules", {})
    )
    cfg = LifecycleConfig.from_rules(b4_rules)

    mon_csv = args.monitor_csv or _latest_monitor_csv(args.runs_root)
    if mon_csv is None or not mon_csv.is_file():
        print("[b4-lifecycle] no monitor csv found; run scripts.bucket4_pair_monitor first")
        return 1
    run_date = mon_csv.parent.parent.name
    monitor = pd.read_csv(mon_csv)

    state_path = cfg.state_json if cfg.state_json.is_absolute() else REPO / cfg.state_json
    state = load_state(state_path)
    new_state, actions = update_lifecycle(monitor, state, cfg, run_date)

    print(f"[b4-lifecycle] enabled={cfg.enabled} run_date={run_date} monitor={mon_csv}")
    print(f"[b4-lifecycle] pairs tracked: {len(new_state)} (was {len(state)})")
    if actions.empty:
        print("[b4-lifecycle] no transitions today")
    else:
        print(actions.to_string(index=False))
    counts = pd.Series([v.get("status") for v in new_state.values()]).value_counts().to_dict()
    print(f"[b4-lifecycle] status counts: {counts}")

    if args.dry_run:
        print("[b4-lifecycle] dry-run: state NOT saved")
        return 0
    save_state(state_path, new_state)
    out_csv = mon_csv.parent / "b4_lifecycle_actions.csv"
    actions.to_csv(out_csv, index=False)
    print(f"[b4-lifecycle] state -> {state_path}")
    print(f"[b4-lifecycle] actions -> {out_csv}")
    return 0


__all__ = [
    "LifecycleConfig",
    "apply_lifecycle_to_b4",
    "held_etfs",
    "lifecycle_sets",
    "load_state",
    "save_state",
    "update_lifecycle",
]


if __name__ == "__main__":
    raise SystemExit(main())
