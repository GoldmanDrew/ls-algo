"""Local tests for ibkr_flex fetch loop (no live IBKR credentials)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

import ibkr_flex


def test_flex_response_log_hint_includes_code() -> None:
    body = (
        "<FlexStatementResponse>"
        "<Status>Warn</Status>"
        "<ErrorCode>1009</ErrorCode>"
        "<ErrorMessage>heavy</ErrorMessage>"
        "</FlexStatementResponse>"
    )
    s = ibkr_flex._flex_response_log_hint(body)
    assert "1009" in s
    assert "heavy" in s


WARN_1019 = (
    "<FlexStatementResponse>"
    "<Status>Warn</Status>"
    "<ErrorCode>1019</ErrorCode>"
    "<ErrorMessage>Statement generation in progress</ErrorMessage>"
    "</FlexStatementResponse>"
)
OK_XML = (
    "<FlexStatementResponse><Status>Success</Status></FlexStatementResponse>"
    "<FlexQueryResponse><root/></FlexQueryResponse>"
)


class AdvancingClock:
    def __init__(self, start: float = 1_000_000.0) -> None:
        self.t = start

    def time(self) -> float:
        return self.t

    def sleep(self, sec: float) -> None:
        self.t += float(sec)


def test_fetch_and_save_stuck_1019_renews_reference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """After IBKR_FLEX_1019_STUCK_RENEW_AFTER_SEC, flex_trades-style fetch issues a new SendRequest."""
    monkeypatch.setenv("IBKR_FLEX_1019_STUCK_RENEW_AFTER_SEC", "1")
    monkeypatch.setenv("IBKR_FLEX_1019_STUCK_RENEW_INTERVAL_SEC", "1")
    monkeypatch.setenv("IBKR_FLEX_1019_STUCK_MAX_RENEWS", "4")
    out = tmp_path / "ibkr_flex"
    out.mkdir(parents=True)

    q = ibkr_flex.FlexQuery("flex_trades", "999")
    get_statement = Mock(side_effect=[WARN_1019, OK_XML])
    send_request_with_backoff = Mock(side_effect=["REF1", "REF2"])
    clock = AdvancingClock(1_000_000.0)

    with (
        patch.object(ibkr_flex, "send_request_with_backoff", send_request_with_backoff),
        patch.object(ibkr_flex, "get_statement", get_statement),
        patch.object(ibkr_flex.time, "time", clock.time),
        patch.object(ibkr_flex.time, "sleep", clock.sleep),
    ):
        p = ibkr_flex.fetch_and_save(
            base_url=ibkr_flex.DEFAULT_BASE_URL,
            token="tok",
            q=q,
            out_dir=out,
            poll_every_sec=5.0,
            max_wait_sec=3600.0,
            enable_stuck_1019_renew=True,
        )

    assert p.name == "flex_trades.xml"
    assert "<Status>Success</Status>" in p.read_text(encoding="utf-8")
    assert send_request_with_backoff.call_count == 2
    assert get_statement.call_count == 2


def test_fetch_and_save_no_stuck_renew_polls_until_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without stuck renew, prolonged 1019 eventually raises FlexError at max_wait_sec."""
    monkeypatch.delenv("IBKR_FLEX_1019_STUCK_RENEW_AFTER_SEC", raising=False)
    monkeypatch.setenv("IBKR_FLEX_1019_STUCK_MAX_RENEWS", "0")
    out = tmp_path / "ibkr_flex2"
    out.mkdir(parents=True)
    q = ibkr_flex.FlexQuery("flex_cash", "888")

    get_statement = Mock(return_value=WARN_1019)
    send_request_with_backoff = Mock(return_value="REFONLY")
    clock = AdvancingClock(2_000_000.0)

    with (
        patch.object(ibkr_flex, "send_request_with_backoff", send_request_with_backoff),
        patch.object(ibkr_flex, "get_statement", get_statement),
        patch.object(ibkr_flex.time, "time", clock.time),
        patch.object(ibkr_flex.time, "sleep", clock.sleep),
    ):
        with pytest.raises(ibkr_flex.FlexError, match="Timed out fetching"):
            ibkr_flex.fetch_and_save(
                base_url=ibkr_flex.DEFAULT_BASE_URL,
                token="tok",
                q=q,
                out_dir=out,
                poll_every_sec=1.0,
                max_wait_sec=50.0,
                enable_stuck_1019_renew=False,
            )

    assert send_request_with_backoff.call_count == 1
    assert get_statement.call_count >= 2
