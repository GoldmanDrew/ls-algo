from __future__ import annotations

import run_eod_pnl_email as eod


def test_send_email_attaches_pdf_with_pdf_mime(tmp_path, monkeypatch):
    sent = {}

    class FakeSMTP:
        def __init__(self, host, port):
            sent["host"] = host
            sent["port"] = port

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def starttls(self):
            sent["starttls"] = True

        def login(self, user, password):
            sent["login"] = (user, password)

        def send_message(self, msg, to_addrs=None):
            sent["msg"] = msg
            sent["to_addrs"] = to_addrs

    pdf = tmp_path / "b4_pair_pnl_hedge_2026-06-22.pdf"
    pdf.write_bytes(b"%PDF-1.4\n% test\n")

    monkeypatch.setenv("SMTP_HOST", "smtp.example.test")
    monkeypatch.setenv("SMTP_USER", "sender@example.test")
    monkeypatch.setenv("SMTP_PASS", "secret")
    monkeypatch.setattr(eod.smtplib, "SMTP", FakeSMTP)

    eod.send_email(
        subject="EOD",
        body="body",
        attachments=[pdf],
        recipients=["recipient@example.test"],
    )

    attachments = list(sent["msg"].iter_attachments())
    assert sent["to_addrs"] == ["recipient@example.test"]
    assert len(attachments) == 1
    assert attachments[0].get_content_type() == "application/pdf"
    assert attachments[0].get_filename() == pdf.name
