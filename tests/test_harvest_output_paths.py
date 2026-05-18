from __future__ import annotations

from pathlib import Path

import pandas as pd

from harvest_underexposed_shorts import harvest_execution_dir, write_harvest_csv


def test_write_harvest_csv_creates_parent(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = {"_repo_root": str(tmp_path)}
    outdir = harvest_execution_dir("2099-01-01", cfg)
    target = outdir / "nested" / "harvest_attempted_trades.csv"
    write_harvest_csv(target, pd.DataFrame([{"symbol": "TEST", "status": "ok"}]))
    assert target.exists()
    assert target.parent.exists()
