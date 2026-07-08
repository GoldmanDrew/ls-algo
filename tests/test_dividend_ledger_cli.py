"""Ensure dividend_ledger.py is invocable the same way CI runs it."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_dividend_ledger_script_imports_from_repo_root() -> None:
    """dashboard_pipeline runs ``python scripts/dividend_ledger.py`` with cwd=repo."""
    proc = subprocess.run(
        [sys.executable, "scripts/dividend_ledger.py"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
