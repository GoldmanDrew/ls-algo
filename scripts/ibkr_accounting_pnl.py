#!/usr/bin/env python3
"""CLI entrypoint for daily IBKR accounting (delegates to ibkr_accounting.py)."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ibkr_accounting import main  # noqa: E402

if __name__ == "__main__":
    _yf_flag = "--yf" in sys.argv
    _args = [a for a in sys.argv[1:] if a != "--yf"]
    _run_date = _args[0] if _args else None
    raise SystemExit(main(_run_date, use_yfinance=_yf_flag if _yf_flag else None))
