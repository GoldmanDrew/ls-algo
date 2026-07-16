"""Tests for flow kill / flatten helpers in execute_flow_program."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from execute_flow_program import (
    build_flow_kill_set,
    load_flow_alumni,
    save_flow_alumni,
    scrub_flow_symbols_from_config,
)


def test_build_flow_kill_set_requires_neg_edge_and_high_borrow():
    screened = pd.DataFrame(
        {
            "ETF": ["TQQY", "NVDQ", "SQQQ", "AMZO", "DUST"],
            "net_edge_p50_annual": [-0.19, -0.02, 0.10, -0.05, 0.30],
            "borrow_current": [0.51, 0.66, 0.05, 0.20, float("nan")],
        }
    )
    kill, detail = build_flow_kill_set(
        screened,
        ["TQQY", "NVDQ", "SQQQ", "AMZO", "DUST"],
        borrow_floor=0.40,
        require_neg_edge=True,
    )
    assert kill == {"TQQY", "NVDQ"}
    assert set(detail.loc[detail["kill"], "ETF"]) == {"TQQY", "NVDQ"}


def test_build_flow_kill_set_missing_borrow_not_killed():
    screened = pd.DataFrame(
        {
            "ETF": ["DUST"],
            "net_edge_p50_annual": [-0.50],
            "borrow_current": [float("nan")],
        }
    )
    kill, _ = build_flow_kill_set(screened, ["DUST"], borrow_floor=0.40)
    assert kill == set()


def test_scrub_flow_symbols_from_config(tmp_path: Path):
    cfg = tmp_path / "strategy_config.yml"
    cfg.write_text(
        """
portfolio:
  sleeves:
    flow_program:
      universe:
        shorts:
          - SQQQ
          - TQQY
          - LABD
      weighting:
        method: "fixed"
        normalize: true
        weights:
          SQQQ: 0.5
          TQQY: 0.2
          LABD: 0.3

# SPX slide-risk
spx_shock:
  horizon_shock_mode: rms
""".lstrip(),
        encoding="utf-8",
    )
    removed = scrub_flow_symbols_from_config(cfg, ["TQQY"])
    assert removed == ["TQQY"]
    text = cfg.read_text(encoding="utf-8")
    assert "- TQQY" not in text
    assert "TQQY:" not in text
    assert "- SQQQ" in text
    assert "LABD: 0.3" in text
    assert "spx_shock:" in text


def test_flow_alumni_roundtrip(tmp_path: Path):
    p = tmp_path / "alumni.json"
    save_flow_alumni(p, ["sqqq", "TQQY"])
    got = load_flow_alumni(p, seeds=["NVDQ"])
    assert got == {"SQQQ", "TQQY", "NVDQ"}
