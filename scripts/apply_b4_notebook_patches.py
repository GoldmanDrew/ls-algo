"""Apply Bucket 4 weekly Opt-2 notebook patches (idempotent). Run from repo root."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    here = Path(__file__).resolve().parent
    repo = here.parent
    scripts = [
        "_patch_bucket4_nb.py",
        "_patch_bucket4_cell17_note.py",
        "_patch_buckets1_4_v2_nb.py",
        "_patch_buckets1_4_exp.py",
        "_patch_buckets1_4_v2_pf_call.py",
    ]
    for name in scripts:
        subprocess.check_call([sys.executable, str(here / name)], cwd=str(repo))


if __name__ == "__main__":
    main()
