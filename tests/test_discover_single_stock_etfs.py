from pathlib import Path

from scripts.discover_single_stock_etfs import (
    Candidate,
    Source,
    candidates_from_tables,
    candidates_from_text,
    load_screener_symbols,
    patch_daily_screener,
    rejection_summary,
    verify_candidates,
)


ALIASES = {"SPACEX": "SPCX", "POET": "POET"}


def test_candidates_from_table_extracts_long_and_inverse_products() -> None:
    html = """
    <table>
      <thead>
        <tr><th>Ticker</th><th>Fund Name</th><th>Underlying</th><th>Exchange</th></tr>
      </thead>
      <tbody>
        <tr><td>SPCM</td><td>Tradr 2X Long SpaceX Daily ETF</td><td>SpaceX</td><td>Cboe</td></tr>
        <tr><td>SPCG</td><td>Tradr 2X Short SpaceX Daily ETF</td><td>SPCX</td><td>Cboe</td></tr>
      </tbody>
    </table>
    """
    source = Source(name="sample", kind="issuer", url="https://example.test")

    rows = candidates_from_tables(html, source, ALIASES)

    by_ticker = {row.etf: row for row in rows}
    assert by_ticker["SPCM"].direction == "long"
    assert by_ticker["SPCM"].underlying == "SPCX"
    assert by_ticker["SPCG"].direction == "inverse"
    assert by_ticker["SPCG"].underlying == "SPCX"


def test_candidates_from_cboe_detail_table_extracts_underlying_from_name() -> None:
    html = """
    <table>
      <thead><tr><th>Symbol</th><th>Name</th><th>First date of trading</th></tr></thead>
      <tbody>
        <tr><td>ADIU</td><td>Leverage Shares 2X Long ADI Daily ETF</td><td>Tuesday, June 16, 2026</td></tr>
        <tr><td>SNK</td><td>GraniteShares 2x Short SpaceX Daily ETF</td><td>Monday, June 15, 2026</td></tr>
      </tbody>
    </table>
    """
    source = Source(name="cboe-detail", kind="exchange", url="https://example.test")

    rows = candidates_from_tables(html, source, ALIASES)

    by_ticker = {row.etf: row for row in rows}
    assert by_ticker["ADIU"].underlying == "ADI"
    assert by_ticker["ADIU"].direction == "long"
    assert by_ticker["SNK"].underlying == "SPCX"
    assert by_ticker["SNK"].direction == "inverse"


def test_issuer_product_page_prefers_ticker_from_url_slug() -> None:
    html = """
    <main>
      <h1>SPAL GraniteShares 2x Long SpaceX Daily ETF</h1>
      <p>The Fund seeks 200% daily exposure to SpaceX common stock (NASDAQ SPCX).</p>
    </main>
    """
    source = Source(name="graniteshares-spal", kind="issuer", url="https://graniteshares.com/etfs/spal/")

    rows = candidates_from_text(html, source, ALIASES)

    assert len(rows) == 1
    assert rows[0].etf == "SPAL"
    assert rows[0].underlying == "SPCX"
    assert rows[0].direction == "long"


def test_patch_daily_screener_adds_missing_long_and_inverse(tmp_path: Path) -> None:
    repo = tmp_path
    screener = repo / "daily_screener.py"
    screener.write_text(
        """
leverage_pairs = [("TQQQ", "QQQ")]
leverage_pairs_leverageshares = []
new_pairs = [
    ("SPCM", "SPCX"),
]
proshares_pairs_levered = []
graniteshares_pairs_leveraged = []
leverage_pairs_capped_accel = []
YIELDBOOST_BUCKET2_PAIRS = []
covered_call_pairs = []
INVERSE_ETF_UNIVERSE = [
    ("SQQQ", -3, "NDX"),
]
""".lstrip(),
        encoding="utf-8",
    )
    candidates = [
        Candidate("SPCM", "SPCX", "long", 2.0, "tradr"),
        Candidate("SPCL", "SPCX", "long", 2.0, "tradr"),
        Candidate("SPCG", "SPCX", "inverse", -2.0, "tradr"),
    ]

    added = patch_daily_screener(repo, candidates)

    assert [row.etf for row in added] == ["SPCL", "SPCG"]
    symbols = load_screener_symbols(repo)
    assert "SPCL" in symbols["long"]
    assert "SPCG" in symbols["inverse"]
    assert screener.read_text(encoding="utf-8").count('("SPCM", "SPCX")') == 1


def _write_minimal_screener(path: Path, new_pairs: str) -> None:
    path.write_text(
        f"""
leverage_pairs = []
leverage_pairs_leverageshares = []
new_pairs = [{new_pairs}]
proshares_pairs_levered = []
graniteshares_pairs_leveraged = []
leverage_pairs_capped_accel = []
YIELDBOOST_BUCKET2_PAIRS = []
covered_call_pairs = []
INVERSE_ETF_UNIVERSE = []
""".lstrip(),
        encoding="utf-8",
    )


def test_verify_allows_candidate_present_in_only_one_target_repo(tmp_path: Path) -> None:
    repo_a = tmp_path / "a"
    repo_b = tmp_path / "b"
    repo_a.mkdir()
    repo_b.mkdir()
    _write_minimal_screener(repo_a / "daily_screener.py", '("POEL", "POET")')
    _write_minimal_screener(repo_b / "daily_screener.py", "")
    candidate = Candidate(
        "POEL",
        "POET",
        "long",
        2.0,
        "defiance",
        source_kinds=["issuer", "exchange"],
        source_urls=["https://example.test"],
    )

    verified = verify_candidates(
        [candidate],
        repo_roots=[repo_a, repo_b],
        cfg={"denylist": {}},
        skip_market_data=True,
    )

    assert [row.etf for row in verified] == ["POEL"]


def test_rejection_summary_surfaces_non_duplicate_unresolved_candidates() -> None:
    duplicate = Candidate("SPCM", "SPCX", "long", 2.0, "tradr")
    duplicate.status = "rejected"
    duplicate.rejection_reasons = ["already_in_all_universes"]
    missing = Candidate("LOFF", "SPCX", "long", 2.0, "direxion")
    missing.status = "pending_market_data"
    missing.rejection_reasons = ["market_data_empty"]

    summary = rejection_summary([duplicate, missing])

    assert summary["status_counts"]["pending_market_data"] == 1
    assert summary["reason_counts"]["market_data_empty"] == 1
    assert [row["etf"] for row in summary["unresolved_candidates"]] == ["LOFF"]
