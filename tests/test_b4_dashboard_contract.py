from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.b4_dashboard_contract import export_contract, git_provenance, validate_contract  # noqa: E402


def _fixture(root: Path) -> Path:
    src = root / "prod"
    src.mkdir()
    report = {
        "mode": "prod", "run_date": "2026-06-03", "start": "2026-06-01", "end": "2026-06-03",
        "capital_usd": 1_000_000, "gross_leverage": 4,
        "budgets_usd": {"inverse_decay_bucket4": 100_000},
        "rebalance_knobs": {"b4_execution": "cadence", "execution_lag_sessions": 1},
        "limitations": ["fixture"],
    }
    (src / "report.json").write_text(json.dumps(report), encoding="utf-8")
    pd.DataFrame({"date": ["2026-06-01", "2026-06-02", "2026-06-03"], "BOOK_NAV": [1, 1, 1]}).to_csv(src / "daily_nav.csv", index=False)
    sleeve = pd.DataFrame({
        "date": ["2026-06-01", "2026-06-02", "2026-06-03"],
        "inverse_decay_bucket4": [0.0, 100.0, -20.0],
        "inverse_decay_bucket4__price_pnl": [0.0, 110.0, -18.0],
        "inverse_decay_bucket4__borrow_cost": [0.0, 2.0, 1.0],
        "inverse_decay_bucket4__short_credit": [0.0, 1.0, 0.5],
        "inverse_decay_bucket4__margin_cost": [0.0, 1.0, 0.5],
        "inverse_decay_bucket4__txn_cost": [0.0, 8.0, 1.0],
        "inverse_decay_bucket4__net_cap": [0.0, -10_000.0, -9_000.0],
        "inverse_decay_bucket4__gross_cap": [0.0, 50_000.0, 48_000.0],
        "pnl_recon_residual": [0.0, 0.0, 0.0],
    })
    sleeve.to_csv(src / "sleeve_daily_pnl.csv", index=False)
    pair = pd.DataFrame({
        "date": ["2026-06-02", "2026-06-03"], "ETF": ["NBIZ", "NBIZ"], "Underlying": ["NBIS", "NBIS"],
        "sleeve": ["inverse_decay_bucket4"] * 2, "etf_usd": [-30_000, -29_000], "underlying_usd": [-20_000, -19_000],
        "long_usd": [0, 0], "short_usd": [-50_000, -48_000], "Delta": [-2, -2], "hedge_ratio": [2/3, 19/29],
        "price_pnl": [110, -18], "borrow_cost": [2, 1], "short_credit": [1, .5], "margin_cost": [1, .5],
        "txn_cost": [8, 1], "daily_pnl": [99.5, -20], "is_rebalance": [1, 0],
        "active_plan_date": ["2026-06-01", "2026-06-02"], "cum_pnl": [99.5, 79.5],
    })
    pair.to_csv(src / "pair_daily_pnl.csv", index=False)
    pd.DataFrame({"ETF": ["NBIZ"], "Underlying": ["NBIS"], "sleeve": ["inverse_decay_bucket4"], "n_rebals": [1]}).to_csv(src / "pair_stats.csv", index=False)
    pd.DataFrame({"date": ["2026-06-02"], "turnover_usd": [50_000]}).to_csv(src / "rebalance_audit.csv", index=False)
    pd.DataFrame({"date": ["2026-06-02"], "ETF": ["NBIZ"], "desired_gross_usd": [50_000]}).to_csv(src / "pending_target_audit.csv", index=False)
    pd.DataFrame([
        {"ETF": "NBIZ", "Underlying": "NBIS", "lifecycle_state": "open", "block_reason": ""},
        {"ETF": "CBRZ", "Underlying": "CBRS", "lifecycle_state": "pending_entry", "block_reason": "awaiting_operator_or_execution"},
        {"ETF": "SPCG", "Underlying": "SPCX", "lifecycle_state": "purgatory_not_incumbent", "block_reason": "purgatory_without_prior_fill"},
    ]).to_csv(src / "b4_membership_manifest.csv", index=False)
    pd.DataFrame([{
        "date": "2026-06-02", "replay_end": "2026-06-03", "ETF": "NBIZ",
        "Underlying": "NBIS", "sleeve": "inverse_decay_bucket4",
        "reason": "cadence_resize", "on_model_cadence": True,
        "on_operator_day": False, "turnover_usd": 50_000, "txn_cost": 8,
    }]).to_csv(src / "b4_pair_execution_events.csv", index=False)
    return src


def test_export_is_authoritative_and_reconciles_pairs(tmp_path: Path):
    src = _fixture(tmp_path)
    out = tmp_path / "out"
    contract = export_contract(src, out, REPO)
    manifest = validate_contract(out)
    assert manifest["authoritative"] is True
    assert manifest["reconciliation"]["pair_to_sleeve"]["max_abs_after_usd"] < 0.01
    assert contract["book"]["summary"]["net_pnl_usd"] == pytest.approx(80.0)
    pair = json.loads((out / "pairs" / "NBIZ.json").read_text())
    assert pair["ledger_mode"] == "actual_dollar"
    assert pair["summary"]["actual_pnl_usd"] == pytest.approx(80.0)
    assert pair["daily"]["drawdown"] == pytest.approx([0.0, -20.0 / 50099.5], abs=1e-8)
    assert pair["daily"]["rebalance"] == [1, 0]
    assert manifest["counts"]["membership"] == 3
    members = json.loads((out / "membership.json").read_text())
    assert {m["ETF"] for m in members} == {"NBIZ", "CBRZ", "SPCG"}


def test_hash_tamper_is_rejected(tmp_path: Path):
    src = _fixture(tmp_path)
    out = tmp_path / "out"
    export_contract(src, out, REPO)
    (out / "book.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_contract(out)


def test_generated_replay_outputs_do_not_dirty_source_provenance():
    prov = git_provenance(REPO)
    # This checkout may contain unrelated user work, but generated production
    # replay/output paths must never be the reason it is marked dirty.
    assert prov.get("commit")
    assert prov.get("working_tree_hash") or prov.get("dirty") is False
