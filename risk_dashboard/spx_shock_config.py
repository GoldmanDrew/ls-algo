"""SPX slide-risk shock configuration (horizon scaling, stress β, paths)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml

HorizonShockMode = Literal["terminal", "rms", "drift"]

DEFAULT_SPX_SHOCK_CONFIG: dict[str, Any] = {
    "horizon_shock_mode": "rms",
    "show_terminal_12m_row": True,
    "path_steps": 252,
    "stress_beta": {
        "enabled": True,
        "down_delta": 0.15,
        "down_threshold_pct": -0.05,
        "use_variance_decomp_cap": True,
        "use_cumulative_drawdown": True,
        "decomp_cap_factor": 1.0,
    },
    "path_scenarios_enabled": True,
    "path_borrow_stress_enabled": True,
}


def load_spx_shock_config(repo_root: Path | None = None) -> dict[str, Any]:
    """Merge ``spx_shock`` block from strategy_config.yml over defaults."""
    cfg = {
        "horizon_shock_mode": DEFAULT_SPX_SHOCK_CONFIG["horizon_shock_mode"],
        "show_terminal_12m_row": DEFAULT_SPX_SHOCK_CONFIG["show_terminal_12m_row"],
        "stress_beta": dict(DEFAULT_SPX_SHOCK_CONFIG["stress_beta"]),
        "path_scenarios_enabled": DEFAULT_SPX_SHOCK_CONFIG["path_scenarios_enabled"],
    }
    if repo_root is None:
        return cfg
    path = repo_root / "config" / "strategy_config.yml"
    if not path.is_file():
        return cfg
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        block = raw.get("spx_shock") or {}
        if isinstance(block, dict):
            if block.get("horizon_shock_mode") is not None:
                cfg["horizon_shock_mode"] = str(block["horizon_shock_mode"]).lower()
            if block.get("show_terminal_12m_row") is not None:
                cfg["show_terminal_12m_row"] = bool(block["show_terminal_12m_row"])
            if block.get("path_scenarios_enabled") is not None:
                cfg["path_scenarios_enabled"] = bool(block["path_scenarios_enabled"])
            if block.get("path_steps") is not None:
                cfg["path_steps"] = int(block["path_steps"])
            if block.get("path_borrow_stress_enabled") is not None:
                cfg["path_borrow_stress_enabled"] = bool(block["path_borrow_stress_enabled"])
            stress = block.get("stress_beta")
            if isinstance(stress, dict):
                cfg["stress_beta"].update(
                    {k: v for k, v in stress.items() if v is not None}
                )
    except Exception:
        pass
    return cfg
