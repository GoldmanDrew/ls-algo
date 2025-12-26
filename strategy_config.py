#!/usr/bin/env python3
"""
strategy_config.py

Shared config loader for the IBKR pair strategy scripts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str | Path = "config/strategy_config.yml") -> Dict[str, Any]:
    path = Path(config_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    cfg = yaml.safe_load(path.read_text()) or {}

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
