from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ibkr_accounting import (
    _filter_exposure_df,
    _filter_positions_blacklist,
    expand_blacklist,
)


def _managed_exposure_gross(accounting_dir: Path, blocked_underlyings: set[str]) -> tuple[float, float, float, float]:
    book = _filter_exposure_df(
        pd.read_csv(accounting_dir / "net_exposure_by_underlying.csv"),
        blocked_underlyings,
    )
    bucket_g = 0.0
    bucket_n = 0.0
    for i in (1, 2, 4):
        df = _filter_exposure_df(
            pd.read_csv(accounting_dir / f"net_exposure_bucket_{i}.csv"),
            blocked_underlyings,
        )
        bucket_g += float(df["gross_notional_usd"].sum())
        bucket_n += float(df["net_notional_usd"].sum())
    return (
        float(book["gross_notional_usd"].sum()),
        float(book["net_notional_usd"].sum()),
        bucket_g,
        bucket_n,
    )


def test_expand_blacklist_maps_apld_complex() -> None:
    etf_to_under = {
        "APLD": "APLD",
        "APLX": "APLD",
        "APLZ": "APLD",
        "KEEX": "KEEL",
    }
    blocked_symbols, blocked_underlyings = expand_blacklist({"APLD"}, etf_to_under)
    assert blocked_underlyings == {"APLD"}
    assert blocked_symbols == {"APLD", "APLX", "APLZ"}


def test_filter_exposure_df_drops_blacklisted_underlying() -> None:
    df = pd.DataFrame(
        [
            {"underlying": "APLD", "net_notional_usd": 100.0, "gross_notional_usd": 100.0},
            {"underlying": "KEEL", "net_notional_usd": 50.0, "gross_notional_usd": 50.0},
        ]
    )
    out = _filter_exposure_df(df, {"APLD", "APLX", "APLZ"})
    assert out["underlying"].tolist() == ["KEEL"]


def test_managed_bucket_gross_excludes_blacklisted_underlying(tmp_path: Path) -> None:
    """Blacklisted names in bucket CSVs must not count toward book reconciliation."""
    accounting = tmp_path / "accounting"
    accounting.mkdir()
    pd.DataFrame(
        [
            {"underlying": "KEEL", "gross_notional_usd": 100.0, "net_notional_usd": 50.0},
        ]
    ).to_csv(accounting / "net_exposure_by_underlying.csv", index=False)
    for i, rows in (
        (1, [{"underlying": "KEEL", "gross_notional_usd": 60.0, "net_notional_usd": 30.0}]),
        (2, [{"underlying": "KEEL", "gross_notional_usd": 40.0, "net_notional_usd": 20.0}]),
        (
            4,
            [
                {"underlying": "APLD", "gross_notional_usd": 900.0, "net_notional_usd": 0.0},
                {"underlying": "KEEL", "gross_notional_usd": 10.0, "net_notional_usd": 5.0},
            ],
        ),
    ):
        pd.DataFrame(rows).to_csv(accounting / f"net_exposure_bucket_{i}.csv", index=False)

    book_g, _, bucket_g, _ = _managed_exposure_gross(accounting, {"APLD"})
    assert book_g == pytest.approx(100.0)
    assert bucket_g == pytest.approx(110.0)  # 60+40+10, APLD dropped


def test_filter_positions_blacklist_drops_etf_legs() -> None:
    pos = pd.DataFrame(
        [
            {"symbol": "APLX", "underlyingSymbol": "APLD", "position": 100},
            {"symbol": "KEEX", "underlyingSymbol": "KEEL", "position": 200},
        ]
    )
    etf_to_under = {"APLX": "APLD", "APLZ": "APLD", "KEEX": "KEEL"}
    out = _filter_positions_blacklist(pos, {"APLD", "APLX", "APLZ"}, {"APLD"}, etf_to_under)
    assert out["symbol"].tolist() == ["KEEX"]
