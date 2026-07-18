"""Tests for price integrity, patches, referee, B4 leg split, delist cutoff."""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.pair_price_panel import (
    apply_delist_cutoff,
    apply_price_patches,
    referee_replace_false_prints,
)
from scripts.price_integrity_audit import suspects_mask
from scripts.production_actual_backtest import b4_leg_targets_from_gross


def test_b4_leg_targets_opt2_convention():
    etf, und = b4_leg_targets_from_gross(100_000.0, h=0.75, beta_abs=2.0)
    assert etf < 0 and und < 0
    # G = |etf|+|und|; |und|/|etf| = h*|beta|
    assert abs(abs(etf) + abs(und) - 100_000.0) < 1e-6
    assert abs(abs(und) / abs(etf) - 1.5) < 1e-9


def test_apply_price_patches_overwrites(tmp_path, monkeypatch):
    import scripts.pair_price_panel as ppp

    patch = tmp_path / "price_patches.csv"
    patch.write_text(
        "symbol,date,close,source,note\nRDWU,2026-04-21,11.79,test,x\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(ppp, "REPO", tmp_path)
    # write under data/
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "price_patches.csv").write_text(patch.read_text(encoding="utf-8"), encoding="utf-8")
    idx = pd.DatetimeIndex(["2026-04-20", "2026-04-21", "2026-04-22"])
    s = pd.Series([11.33, 36.04, 36.04], index=idx)
    out = apply_price_patches(s, "RDWU", repo=tmp_path)
    assert abs(float(out.loc[pd.Timestamp("2026-04-21")]) - 11.79) < 1e-9


def test_frames_from_metrics_patches_underlying(tmp_path, monkeypatch):
    import scripts.pair_price_panel as ppp
    from scripts.pair_price_panel import frames_from_metrics

    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "price_patches.csv").write_text(
        "symbol,date,close,source,note\n"
        "CVNA,2026-05-06,77.88,test,scale\n"
        "CVNA,2026-05-07,80.00,test,scale\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(ppp, "REPO", tmp_path)
    idx = pd.bdate_range("2026-04-01", periods=40)
    md = pd.DataFrame(
        {
            "date": idx,
            "ticker": "CVNX",
            "etf_adj_close": 18.0 + 0.01 * np.arange(len(idx)),
            "underlying_adj_close": 380.0,  # fake 5x
        }
    )
    # snap day
    md.loc[md["date"] == pd.Timestamp("2026-05-07"), "underlying_adj_close"] = 80.0
    panel = frames_from_metrics(
        md,
        min_days=20,
        apply_splits=False,
        underlying_by_etf={"CVNX": "CVNA"},
        repo=tmp_path,
    )
    b = panel["CVNX"]["b_px"]
    assert abs(float(b.loc[pd.Timestamp("2026-05-06")]) - 77.88) < 1e-9
    assert abs(float(b.loc[pd.Timestamp("2026-05-07")]) - 80.0) < 1e-9


def test_referee_replaces_phantom_jump(monkeypatch):
    import scripts.pair_price_panel as ppp

    idx = pd.bdate_range("2026-04-15", periods=5)
    metrics = pd.Series([10.0, 10.5, 33.0, 33.0, 12.0], index=idx)  # +214% phantom
    yahoo = pd.Series([10.0, 10.5, 10.8, 11.0, 11.2], index=idx)

    monkeypatch.setattr(ppp, "_yahoo_close", lambda *_a, **_k: yahoo)
    out = referee_replace_false_prints(metrics, "FAKE")
    # Day of phantom should match Yahoo
    assert abs(float(out.iloc[2]) - 10.8) < 1e-9


def test_delist_cutoff_truncates():
    idx = pd.bdate_range("2026-05-01", periods=10)
    df = pd.DataFrame({"a_px": range(10), "b_px": range(10)}, index=idx)
    panel = {"SOLX": df}
    dmap = {"SOLX": pd.Timestamp("2026-05-05")}
    out = apply_delist_cutoff(panel, delist_map=dmap)
    assert out["SOLX"].index.max() <= pd.Timestamp("2026-05-05")


def test_suspects_mask_flags_phantoms():
    df = pd.DataFrame(
        [
            {"n_big50": 0, "yahoo_gt15pct": 0, "max_abs_ratio_m1": 0.0, "phantom_days": 2},
            {"n_big50": 0, "yahoo_gt15pct": 0, "max_abs_ratio_m1": 0.0, "phantom_days": 0},
        ]
    )
    m = suspects_mask(df)
    assert bool(m.iloc[0]) and not bool(m.iloc[1])
