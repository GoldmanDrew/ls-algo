#!/usr/bin/env python3
"""Discover verified single-stock leveraged ETFs missing from daily_screener.py.

The script has two jobs:

1. Crawl configured official issuer/exchange pages and normalize candidates.
2. Optionally patch daily_screener.py in one or more repo roots.

It is intentionally conservative. A candidate is patchable only when it is
seen on at least one issuer page, one exchange page, has a known underlying,
and has live market data unless an explicit operator override is passed.
"""
from __future__ import annotations

import argparse
import ast
import io
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

import pandas as pd
import requests
import yaml
import yfinance as yf
from bs4 import BeautifulSoup


LONG_LIST_NAME = "new_pairs"
INVERSE_LIST_NAME = "INVERSE_ETF_UNIVERSE"
DEFAULT_CONFIG = Path("config/levered_etf_discovery.yml")
USER_AGENT = (
    "Mozilla/5.0 (compatible; ls-algo-universe-discovery/1.0; "
    "+https://github.com/magis-capital-partners/ls-algo)"
)

STOP_UNDERLYING_WORDS = {
    "DAILY",
    "ETF",
    "ETFS",
    "FUND",
    "FUNDS",
    "LONG",
    "SHORT",
    "TARGET",
    "ULTRA",
}


@dataclass
class Source:
    name: str
    kind: str
    url: str


@dataclass
class Candidate:
    etf: str
    underlying: str | None
    direction: str
    leverage: float
    issuer: str
    exchange: str | None = None
    first_trade_date: str | None = None
    source_urls: list[str] = field(default_factory=list)
    source_kinds: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    status: str = "candidate"
    rejection_reasons: list[str] = field(default_factory=list)

    def key(self) -> str:
        return norm_sym(self.etf)


def norm_sym(value: Any) -> str:
    return str(value or "").strip().upper().replace(".", "-")


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg.setdefault("sources", [])
    cfg.setdefault("underlying_aliases", {})
    cfg.setdefault("denylist", {})
    cfg["underlying_aliases"] = {
        str(k).upper().strip(): norm_sym(v)
        for k, v in (cfg.get("underlying_aliases") or {}).items()
    }
    return cfg


def fetch_html(url: str, timeout: int = 25) -> str:
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def compact_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def detect_direction_and_leverage(text: str) -> tuple[str | None, float | None]:
    upper = text.upper()
    if not re.search(r"(?:\b(?:2X|200%|-200%)\b|-2X\b)", upper):
        return None, None
    explicit: list[tuple[int, str]] = []
    for pat in [
        r"\b2X\s+SHORT\b",
        r"\bSHORT\s+[A-Z0-9 .-]{0,40}\s+2X\b",
        r"\bBEAR\s+[A-Z0-9 .-]{0,40}\s+2X\b",
        r"(?:\b-200%\b|-2X\b)",
    ]:
        m = re.search(pat, upper)
        if m:
            explicit.append((m.start(), "inverse"))
    for pat in [
        r"\b2X\s+LONG\b",
        r"\bLONG\s+[A-Z0-9 .-]{0,40}\s+2X\b",
        r"\bBULL\s+[A-Z0-9 .-]{0,40}\s+2X\b",
        r"\b200%\b",
    ]:
        m = re.search(pat, upper)
        if m:
            explicit.append((m.start(), "long"))
    if explicit:
        return ("inverse", -2.0) if min(explicit)[1] == "inverse" else ("long", 2.0)
    if re.search(r"\b(?:SHORT|INVERSE|BEAR)\b", upper):
        return "inverse", -2.0
    if re.search(r"\b(?:LONG|BULL)\b", upper):
        return "long", 2.0
    return None, None


def ticker_from_source_url(url: str) -> str | None:
    parts = [p for p in urlparse(url).path.strip("/").split("/") if p]
    if not parts:
        return None
    token = parts[-1].upper()
    blocked = {"ETF", "ETFS", "US", "PRODUCT", "PRODUCTS", "LEVERAGED-AND-INVERSE"}
    if token in blocked:
        return None
    if re.fullmatch(r"[A-Z][A-Z0-9]{1,5}", token):
        return token
    return None


def ticker_from_labeled_text(text: str) -> str | None:
    patterns = [
        r"\b(?:CBOE|NYSE|NASDAQ|BZX|ARCA)\s*:\s*([A-Z][A-Z0-9]{1,5})\b",
        r"\b(?:TICKER|SYMBOL)\s*[:\-]\s*([A-Z][A-Z0-9]{1,5})\b",
    ]
    for pat in patterns:
        m = re.search(pat, text.upper())
        if m:
            return norm_sym(m.group(1))
    return None


def ticker_from_cells(headers: list[str], values: list[str]) -> str | None:
    preferred = ("TICKER", "SYMBOL", "FUND TICKER", "ETF TICKER")
    for header, value in zip(headers, values):
        if any(key in header.upper() for key in preferred):
            m = re.search(r"\b([A-Z][A-Z0-9]{1,5})\b", value.upper())
            if m:
                return norm_sym(m.group(1))
    return ticker_from_labeled_text(" ".join(values))


def underlying_from_cells(
    headers: list[str],
    values: list[str],
    aliases: dict[str, str],
    row_text: str,
) -> str | None:
    header_keys = ("UNDERLYING", "REFERENCE", "BENCHMARK", "TARGET STOCK")
    for header, value in zip(headers, values):
        if any(key in header.upper() for key in header_keys):
            parsed = parse_underlying(value, aliases)
            if parsed:
                return parsed
    return parse_underlying(row_text, aliases)


def parse_underlying(text: str, aliases: dict[str, str]) -> str | None:
    upper = compact_text(text).upper()
    if not upper:
        return None

    for name, symbol in sorted(aliases.items(), key=lambda kv: len(kv[0]), reverse=True):
        if re.search(rf"\b{re.escape(name)}\b", upper):
            return symbol

    labeled_patterns = [
        r"\bUNDERLYING(?:\s+STOCK|\s+SECURITY)?\s*[:\-]\s*([A-Z][A-Z0-9.-]{1,5})\b",
        r"\bREFERENCE(?:\s+ASSET|\s+STOCK)?\s*[:\-]\s*([A-Z][A-Z0-9.-]{1,5})\b",
        r"\bOF\s+([A-Z][A-Z0-9.-]{1,5})\s+COMMON STOCK\b",
        r"\b2X\s+(?:LONG|SHORT)\s+([A-Z][A-Z0-9.-]{1,5})\s+DAILY ETF\b",
    ]
    for pat in labeled_patterns:
        m = re.search(pat, upper)
        if m:
            token = norm_sym(m.group(1))
            if token not in STOP_UNDERLYING_WORDS:
                return token
    return None


def exchange_from_text(text: str) -> str | None:
    upper = text.upper()
    if "CBOE" in upper or "BZX" in upper:
        return "Cboe"
    if "NASDAQ" in upper:
        return "Nasdaq"
    if "NYSE" in upper or "ARCA" in upper:
        return "NYSE"
    return None


def first_trade_date_from_text(text: str) -> str | None:
    patterns = [
        r"\b(?:FIRST\s+TRADE|LAUNCH(?:ED)?|LIST(?:ED|ING))\D{0,30}"
        r"(\d{4}-\d{2}-\d{2})\b",
        r"\b(\d{4}-\d{2}-\d{2})\b",
        r"\b([A-Z][a-z]+ \d{1,2}, \d{4})\b",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if not m:
            continue
        raw = m.group(1)
        for fmt in ("%Y-%m-%d", "%B %d, %Y"):
            try:
                return datetime.strptime(raw, fmt).date().isoformat()
            except ValueError:
                pass
    return None


def candidates_from_tables(
    html: str,
    source: Source,
    aliases: dict[str, str],
) -> list[Candidate]:
    out: list[Candidate] = []
    try:
        tables = pd.read_html(io.StringIO(html))
    except ValueError:
        return out

    for table in tables:
        headers = [compact_text(c).upper() for c in table.columns]
        for _, row in table.iterrows():
            values = [compact_text(v) for v in row.tolist()]
            row_text = compact_text(" | ".join(values))
            direction, leverage = detect_direction_and_leverage(row_text)
            if direction is None or leverage is None:
                continue
            etf = ticker_from_cells(headers, values)
            if not etf:
                continue
            underlying = underlying_from_cells(headers, values, aliases, row_text)
            out.append(
                Candidate(
                    etf=etf,
                    underlying=underlying,
                    direction=direction,
                    leverage=leverage,
                    issuer=source.name,
                    exchange=exchange_from_text(row_text),
                    first_trade_date=first_trade_date_from_text(row_text),
                    source_urls=[source.url],
                    source_kinds=[source.kind],
                    evidence=[row_text[:500]],
                )
            )
    return out


def nearby_windows(text: str, max_chars: int = 900) -> Iterable[str]:
    upper = text.upper()
    for match in re.finditer(r"\b(?:2X|200%|-200%)\b", upper):
        start = max(0, match.start() - max_chars // 2)
        end = min(len(text), match.end() + max_chars // 2)
        yield compact_text(text[start:end])


def candidates_from_text(
    html: str,
    source: Source,
    aliases: dict[str, str],
) -> list[Candidate]:
    soup = BeautifulSoup(html, "lxml")
    text = compact_text(soup.get_text(" "))
    out: list[Candidate] = []
    url_ticker = ticker_from_source_url(source.url) if source.kind == "issuer" else None
    if url_ticker:
        idx = text.upper().find(url_ticker)
        if idx >= 0:
            window = compact_text(text[max(0, idx - 220): idx + 500])
            direction, leverage = detect_direction_and_leverage(window)
            underlying = parse_underlying(window, aliases)
            if direction is not None and leverage is not None and underlying:
                return [
                    Candidate(
                        etf=url_ticker,
                        underlying=underlying,
                        direction=direction,
                        leverage=leverage,
                        issuer=source.name,
                        exchange=exchange_from_text(window),
                        first_trade_date=first_trade_date_from_text(window),
                        source_urls=[source.url],
                        source_kinds=[source.kind],
                        evidence=[window[:500]],
                    )
                ]
    for window in nearby_windows(text):
        direction, leverage = detect_direction_and_leverage(window)
        if direction is None or leverage is None:
            continue
        etf = url_ticker
        if etf and etf not in window.upper():
            continue
        if not etf:
            etf = ticker_from_labeled_text(window)
        if not etf:
            continue
        underlying = parse_underlying(window, aliases)
        out.append(
            Candidate(
                etf=etf,
                underlying=underlying,
                direction=direction,
                leverage=leverage,
                issuer=source.name,
                exchange=exchange_from_text(window),
                first_trade_date=first_trade_date_from_text(window),
                source_urls=[source.url],
                source_kinds=[source.kind],
                evidence=[window[:500]],
            )
        )
    return out


def discover_candidates(cfg: dict[str, Any]) -> tuple[list[Candidate], list[dict[str, str]]]:
    aliases = cfg.get("underlying_aliases", {})
    candidates: list[Candidate] = []
    errors: list[dict[str, str]] = []
    for item in cfg.get("sources", []):
        source = Source(
            name=str(item["name"]),
            kind=str(item["kind"]),
            url=str(item["url"]),
        )
        try:
            html = fetch_html(source.url)
            candidates.extend(candidates_from_tables(html, source, aliases))
            candidates.extend(candidates_from_text(html, source, aliases))
        except Exception as exc:
            errors.append(
                {
                    "source": source.name,
                    "url": source.url,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    for url in cfg.get("cboe_detail_urls", []) or []:
        source = Source(
            name="cboe-new-listings-detail",
            kind="exchange",
            url=str(url),
        )
        try:
            html = fetch_html(source.url)
            candidates.extend(candidates_from_tables(html, source, aliases))
            candidates.extend(candidates_from_text(html, source, aliases))
        except Exception as exc:
            errors.append(
                {
                    "source": source.name,
                    "url": source.url,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return merge_candidates(candidates), errors


def merge_candidates(candidates: Iterable[Candidate]) -> list[Candidate]:
    by_etf: dict[str, Candidate] = {}
    for cand in candidates:
        key = cand.key()
        if key not in by_etf:
            by_etf[key] = cand
            continue
        current = by_etf[key]
        if not current.underlying and cand.underlying:
            current.underlying = cand.underlying
        if not current.exchange and cand.exchange:
            current.exchange = cand.exchange
        if not current.first_trade_date and cand.first_trade_date:
            current.first_trade_date = cand.first_trade_date
        current.source_urls = sorted(set(current.source_urls) | set(cand.source_urls))
        current.source_kinds = sorted(set(current.source_kinds) | set(cand.source_kinds))
        current.evidence.extend(cand.evidence)
        if current.direction != cand.direction:
            current.rejection_reasons.append("conflicting_direction_sources")
    return sorted(by_etf.values(), key=lambda c: c.etf)


def assignment_value(tree: ast.AST, name: str) -> Any:
    for node in getattr(tree, "body", []):
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(f"assignment not found: {name}")


def load_screener_symbols(repo_root: Path) -> dict[str, set[str]]:
    path = repo_root / "daily_screener.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    pair_lists = [
        "leverage_pairs",
        "leverage_pairs_leverageshares",
        "new_pairs",
        "proshares_pairs_levered",
        "graniteshares_pairs_leveraged",
        "leverage_pairs_capped_accel",
        "YIELDBOOST_BUCKET2_PAIRS",
        "covered_call_pairs",
    ]
    long_etfs: set[str] = set()
    for name in pair_lists:
        try:
            for row in assignment_value(tree, name):
                if row:
                    long_etfs.add(norm_sym(row[0]))
        except Exception:
            continue
    inverse_etfs: set[str] = set()
    try:
        for row in assignment_value(tree, INVERSE_LIST_NAME):
            if row:
                inverse_etfs.add(norm_sym(row[0]))
    except Exception:
        pass
    return {"long": long_etfs, "inverse": inverse_etfs, "all": long_etfs | inverse_etfs}


def market_data_ok(symbol: str) -> tuple[bool, str]:
    try:
        hist = yf.Ticker(symbol).history(period="10d", auto_adjust=False)
    except Exception as exc:
        return False, f"market_data_error:{type(exc).__name__}"
    if hist is None or hist.empty:
        return False, "market_data_empty"
    if "Close" not in hist.columns:
        return False, "market_data_missing_close"
    close = pd.to_numeric(hist["Close"], errors="coerce").dropna()
    if close.empty:
        return False, "market_data_no_close"
    return True, f"market_data_rows:{len(close)}"


def verify_candidates(
    candidates: list[Candidate],
    repo_roots: list[Path],
    cfg: dict[str, Any],
    skip_market_data: bool = False,
    require_exchange_source: bool = True,
    allow_pending_market_data: bool = False,
    allow_future_listings: bool = False,
) -> list[Candidate]:
    known_by_repo = [load_screener_symbols(repo_root)["all"] for repo_root in repo_roots]
    deny_etfs = {norm_sym(x) for x in (cfg.get("denylist", {}) or {}).get("etfs", [])}

    verified: list[Candidate] = []
    today = date.today()
    for cand in candidates:
        cand.status = "verified"
        cand.rejection_reasons = list(dict.fromkeys(cand.rejection_reasons))
        etf = norm_sym(cand.etf)
        cand.etf = etf
        cand.underlying = norm_sym(cand.underlying) if cand.underlying else None

        if known_by_repo and all(etf in symbols for symbols in known_by_repo):
            cand.rejection_reasons.append("already_in_all_universes")
        if etf in deny_etfs:
            cand.rejection_reasons.append("denylisted")
        if cand.direction not in {"long", "inverse"}:
            cand.rejection_reasons.append("unsupported_direction")
        if not cand.underlying:
            cand.rejection_reasons.append("missing_underlying")
        if "issuer" not in cand.source_kinds:
            cand.rejection_reasons.append("missing_issuer_source")
        if require_exchange_source and "exchange" not in cand.source_kinds:
            cand.rejection_reasons.append("missing_exchange_source")
        future_listing = False
        if cand.first_trade_date:
            try:
                if datetime.strptime(cand.first_trade_date, "%Y-%m-%d").date() > today:
                    future_listing = True
                    if not allow_future_listings:
                        cand.rejection_reasons.append("future_first_trade_date")
            except ValueError:
                cand.rejection_reasons.append("invalid_first_trade_date")
        if not skip_market_data:
            ok, reason = market_data_ok(etf)
            cand.evidence.append(reason)
            if not ok and not (allow_pending_market_data and not future_listing):
                cand.rejection_reasons.append(reason)

        cand.rejection_reasons = sorted(set(cand.rejection_reasons))
        if cand.rejection_reasons:
            if "future_first_trade_date" in cand.rejection_reasons:
                cand.status = "pending_first_trade"
            elif any(r.startswith("market_data_") for r in cand.rejection_reasons):
                cand.status = "pending_market_data"
            else:
                cand.status = "rejected"
        else:
            verified.append(cand)
    return verified


def find_list_span(source: str, list_name: str) -> tuple[int, int]:
    marker = f"{list_name} = ["
    start = source.find(marker)
    if start < 0:
        raise ValueError(f"list assignment not found: {list_name}")
    bracket = source.find("[", start)
    depth = 0
    for idx in range(bracket, len(source)):
        ch = source[idx]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return bracket, idx
    raise ValueError(f"list assignment has no closing bracket: {list_name}")


def insert_entries(source: str, list_name: str, lines: list[str]) -> str:
    if not lines:
        return source
    _, close_idx = find_list_span(source, list_name)
    insertion = "\n" + "\n".join(lines) + "\n"
    return source[:close_idx] + insertion + source[close_idx:]


def patch_daily_screener(repo_root: Path, candidates: list[Candidate]) -> list[Candidate]:
    path = repo_root / "daily_screener.py"
    source = path.read_text(encoding="utf-8")
    existing = load_screener_symbols(repo_root)
    to_add: list[Candidate] = []
    for cand in candidates:
        direction_set = existing["inverse"] if cand.direction == "inverse" else existing["long"]
        if cand.etf not in direction_set and cand.etf not in existing["all"]:
            to_add.append(cand)
    if not to_add:
        return []

    today = date.today().isoformat()
    long_lines = [
        f'    ("{cand.etf}", "{cand.underlying}"),'
        for cand in sorted(to_add, key=lambda c: c.etf)
        if cand.direction == "long"
    ]
    inverse_lines = [
        f'    ("{cand.etf}", -2, "{cand.underlying}"),'
        for cand in sorted(to_add, key=lambda c: c.etf)
        if cand.direction == "inverse"
    ]
    if long_lines:
        source = insert_entries(
            source,
            LONG_LIST_NAME,
            [f"    # Auto-discovered single-stock leveraged ETFs ({today})"] + long_lines,
        )
    if inverse_lines:
        source = insert_entries(
            source,
            INVERSE_LIST_NAME,
            [f"    # Auto-discovered inverse single-stock leveraged ETFs ({today})"] + inverse_lines,
        )
    path.write_text(source, encoding="utf-8")
    return to_add


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def rejection_summary(candidates: list[Candidate]) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    for cand in candidates:
        status_counts[cand.status] = status_counts.get(cand.status, 0) + 1
        for reason in cand.rejection_reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    missing = [
        {
            "etf": cand.etf,
            "underlying": cand.underlying,
            "status": cand.status,
            "reasons": cand.rejection_reasons,
            "sources": cand.source_urls,
        }
        for cand in candidates
        if cand.status != "verified"
        and "already_in_all_universes" not in cand.rejection_reasons
    ]
    return {
        "status_counts": dict(sorted(status_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "unresolved_candidates": sorted(missing, key=lambda x: (x["status"], x["etf"])),
    }


def write_summary_markdown(path: Path, candidates: list[Candidate], errors: list[dict[str, str]]) -> None:
    summary = rejection_summary(candidates)
    lines = ["# Levered ETF Discovery Report", ""]
    lines.append("## Status Counts")
    lines.append("")
    for status, count in summary["status_counts"].items():
        lines.append(f"- `{status}`: {count}")
    lines.append("")
    lines.append("## Top Rejection Reasons")
    lines.append("")
    for reason, count in list(summary["reason_counts"].items())[:25]:
        lines.append(f"- `{reason}`: {count}")
    lines.append("")
    unresolved = summary["unresolved_candidates"][:100]
    if unresolved:
        lines.append("## Unresolved Non-Duplicate Candidates")
        lines.append("")
        for row in unresolved:
            reasons = ", ".join(row["reasons"]) if row["reasons"] else "none"
            lines.append(f"- `{row['etf']}` -> `{row['underlying']}`: `{row['status']}` ({reasons})")
    if errors:
        lines.append("")
        lines.append("## Source Fetch Warnings")
        lines.append("")
        for err in errors:
            lines.append(f"- `{err['source']}`: {err['error']}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_pr_body(path: Path, added_by_repo: dict[str, list[Candidate]], errors: list[dict[str, str]]) -> None:
    lines = [
        "## Summary",
        "",
        "Auto-discovered verified single-stock leveraged ETF additions.",
        "",
        "## Added pairs",
        "",
    ]
    total = 0
    for repo, added in added_by_repo.items():
        lines.append(f"### {repo}")
        if not added:
            lines.append("")
            lines.append("No new pairs.")
            lines.append("")
            continue
        for cand in sorted(added, key=lambda c: c.etf):
            total += 1
            lev = "-2x" if cand.direction == "inverse" else "2x"
            lines.append(f"- `{cand.etf}` -> `{cand.underlying}` ({lev}, {cand.issuer})")
            for url in cand.source_urls[:4]:
                lines.append(f"  Source: {url}")
        lines.append("")
    lines.extend(
        [
            "## Verification gates",
            "",
            "- Seen on at least one configured issuer source.",
            "- Seen on at least one configured exchange source.",
            "- Has a resolved underlying/proxy symbol.",
            "- Has live market data from yfinance.",
            "- Is not already present in the target screener universe.",
            "- Is not denylisted in `config/levered_etf_discovery.yml`.",
            "",
        ]
    )
    if errors:
        lines.append("## Source fetch warnings")
        lines.append("")
        for err in errors:
            lines.append(f"- `{err['source']}`: {err['error']}")
        lines.append("")
    if total == 0:
        lines.append("No patchable additions were found in this run.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--repo-root", type=Path, default=Path("."))
    ap.add_argument("--mirror-repo-root", type=Path, action="append", default=[])
    ap.add_argument("--output-dir", type=Path, default=Path("data/discovery"))
    ap.add_argument("--write-patches", action="store_true")
    ap.add_argument("--skip-market-data", action="store_true")
    ap.add_argument("--allow-issuer-only", action="store_true")
    ap.add_argument("--allow-pending-market-data", action="store_true")
    ap.add_argument("--allow-future-listings", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    repo_roots = [args.repo_root.resolve()] + [p.resolve() for p in args.mirror_repo_root]
    candidates, errors = discover_candidates(cfg)
    verified = verify_candidates(
        candidates,
        repo_roots=repo_roots,
        cfg=cfg,
        skip_market_data=args.skip_market_data,
        require_exchange_source=not args.allow_issuer_only,
        allow_pending_market_data=args.allow_pending_market_data,
        allow_future_listings=args.allow_future_listings,
    )

    added_by_repo: dict[str, list[Candidate]] = {}
    if args.write_patches:
        for repo_root in repo_roots:
            added_by_repo[str(repo_root)] = patch_daily_screener(repo_root, verified)
    else:
        for repo_root in repo_roots:
            added_by_repo[str(repo_root)] = []

    output_dir = args.output_dir
    write_json(output_dir / "candidates.json", [asdict(c) for c in candidates])
    write_json(output_dir / "verified.json", [asdict(c) for c in verified])
    write_json(
        output_dir / "added_by_repo.json",
        {repo: [asdict(c) for c in rows] for repo, rows in added_by_repo.items()},
    )
    write_json(output_dir / "source_errors.json", errors)
    write_json(output_dir / "summary.json", rejection_summary(candidates))
    write_summary_markdown(output_dir / "summary.md", candidates, errors)
    write_pr_body(output_dir / "pr_body.md", added_by_repo, errors)

    total_added = sum(len(v) for v in added_by_repo.values())
    print(
        f"discovered={len(candidates)} verified={len(verified)} "
        f"patched={total_added} source_errors={len(errors)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
