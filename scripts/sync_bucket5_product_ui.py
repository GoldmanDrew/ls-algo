#!/usr/bin/env python3
"""Sync Bucket 5 Product JS/CSS from ls-algo site/ to etf-dashboard assets/."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def find_etf() -> Path | None:
    for p in (
        REPO.parent / "etf-dashboard",
        Path.home() / "Projects" / "quant" / "etf-dashboard",
    ):
        if p.is_dir():
            return p
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--etf-root", type=Path, default=None)
    args = ap.parse_args(argv)
    etf = args.etf_root or find_etf()
    if not etf:
        print("etf-dashboard not found", file=sys.stderr)
        return 1
    pairs = [
        (REPO / "site" / "assets" / "js" / "bucket5_product.js", etf / "assets" / "bucket5_product.js"),
        (REPO / "site" / "assets" / "css" / "bucket5_product.css", etf / "assets" / "bucket5_product.css"),
    ]
    for src, dest in pairs:
        if not src.is_file():
            print(f"missing {src}", file=sys.stderr)
            return 1
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        print(f"synced {src.name} -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
