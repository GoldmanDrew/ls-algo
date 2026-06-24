#!/usr/bin/env python3
"""
strategy_config.py

Shared config loader for the IBKR pair strategy scripts.

The config is split across a few files under ``config/`` for readability:
  * ``strategy_config.yml``   — sizing / execution / sleeves / screener / paths
  * ``accounting_config.yml`` — PnL-attribution knobs (merged into cfg["accounting"])
  * ``pair_overrides.yml``    — operator B4/B5 pair tilts (merged into cfg["pair_overrides"])

``load_config()`` loads the base file then overlays the sibling files when
present, so every consumer sees a single merged dict.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def _norm_pair_sym(x: str) -> str:
    return str(x).strip().upper().replace(".", "-")


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_config(config_path: str | Path = "config/strategy_config.yml") -> Dict[str, Any]:
    path = Path(config_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    config_dir = path.parent

    # Overlay accounting config (file wins over any inline ``accounting`` block).
    acct = _load_yaml(config_dir / "accounting_config.yml").get("accounting")
    if isinstance(acct, dict):
        merged = dict(cfg.get("accounting") or {})
        merged.update(acct)
        cfg["accounting"] = merged

    # Overlay operator pair overrides.
    pair_ovr = _load_yaml(config_dir / "pair_overrides.yml").get("pair_overrides")
    if pair_ovr is not None:
        cfg["pair_overrides"] = pair_ovr

    # Resolve paths relative to repo root (assume config/ is in repo root)
    repo_root = path.parent.parent  # config/.. = repo root

    paths = cfg.get("paths", {})
    resolved = {}
    for k, v in paths.items():
        if v is None:
            continue
        p = Path(v)
        if not p.is_absolute():
            p = (repo_root / p).resolve()
        resolved[k] = str(p)
    cfg["paths"] = resolved
    cfg["_repo_root"] = str(repo_root)

    return cfg


def load_pair_overrides(cfg: Dict[str, Any]) -> Dict[tuple[str, str], Dict[str, Any]]:
    """Normalize ``cfg["pair_overrides"]`` into ``{(ETF, UNDERLYING): {...}}``.

    Keys are ``"ETF/UNDERLYING"`` strings; symbols are upper-cased and ``.`` ->
    ``-`` normalized. Each value may contain ``hedge_ratio_add`` (float),
    ``gross_mult`` (float), and ``note`` (str). Malformed entries are skipped.
    """
    raw = (cfg or {}).get("pair_overrides") or {}
    out: Dict[tuple[str, str], Dict[str, Any]] = {}
    if not isinstance(raw, dict):
        return out
    for key, spec in raw.items():
        if not isinstance(spec, dict):
            continue
        pair = str(key)
        if "/" not in pair:
            continue
        etf, und = pair.split("/", 1)
        etf_n, und_n = _norm_pair_sym(etf), _norm_pair_sym(und)
        if not etf_n or not und_n:
            continue
        entry: Dict[str, Any] = {}
        try:
            entry["hedge_ratio_add"] = float(spec.get("hedge_ratio_add", 0.0) or 0.0)
        except (TypeError, ValueError):
            entry["hedge_ratio_add"] = 0.0
        try:
            gm = float(spec.get("gross_mult", 1.0))
        except (TypeError, ValueError):
            gm = 1.0
        if gm <= 0 or gm != gm:  # reject non-positive / NaN
            gm = 1.0
        entry["gross_mult"] = gm
        entry["note"] = str(spec.get("note", "") or "")
        out[(etf_n, und_n)] = entry
    return out
