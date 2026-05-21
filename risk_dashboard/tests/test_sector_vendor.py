"""Tests for sector vendor cache fetch."""

from __future__ import annotations

from pathlib import Path

from risk_dashboard.sector_vendor import fetch_vendor_info


def test_fetch_vendor_info_uses_cache_without_network(tmp_path: Path):
    sym = "NVDA"
    cache_path = tmp_path / "sector_vendor.json"
    cache_path.write_text(
        '{"symbols": {"NVDA": {"fetched_at_epoch": 9999999999, '
        '"info": {"sector": "Technology", "industry": "Semiconductors", '
        '"longName": "NVIDIA Corporation"}}}}',
        encoding="utf-8",
    )

    def boom(_sym):
        raise AssertionError("network should not be called")

    class FakeYF:
        @staticmethod
        def Ticker(_sym):
            return type("T", (), {"info": boom})()

    out = fetch_vendor_info(
        [sym],
        cache_path=cache_path,
        refresh_max_age_hours=999999.0,
        yf_module=FakeYF,
    )
    assert out["NVDA"]["industry"] == "Semiconductors"
