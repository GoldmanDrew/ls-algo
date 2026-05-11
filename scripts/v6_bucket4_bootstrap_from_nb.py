"""
Execute the minimal Bucket-4 / v6 Option-2 cell chain from ``notebooks/Bucket_4_Backtest.ipynb``
into a caller-provided globals dict (e.g. ``globals()`` from ``Buckets1-4Backtest.ipynb``).

This avoids maintaining a second copy of ~30k lines of notebook code: the source of truth
remains the Bucket 4 notebook cells.

Cells (0-based indices): 1 (imports), 2 (parameters + FTP + borrow helpers), 3 (Yahoo/cache
price helpers + ``load_prices`` + ``REBALANCE_FREQ_MAP``), 5 (``load_bucket4_pairs`` / sweep),
9 (``closes``, ``b4``, ``_robust_z``), 11 (``closes_broad``), 14 (v6 Opt-2 engine + cache).

After executing cell 14, ``run_bucket4_backtest_dynamic_h`` is replaced with the importable
implementation in ``scripts.bucket4_dynamic_bt`` so ``Buckets1-4_v2`` / tests share the same
bytecode as the notebook without relying on duplicated notebook definitions.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import ModuleType
from typing import Any

_BUCKET4_NB = Path(__file__).resolve().parent.parent / "notebooks" / "Bucket_4_Backtest.ipynb"
_CELL_IDX = (1, 2, 3, 5, 9, 11, 14)


def _neutralize_universe_source_assignment(cell2_src: str) -> str:
    """
    Strip every **source** line that assigns ``UNIVERSE_SOURCE_CSV``, then prepend a safe
    empty-string assignment.

    A notebook saved on Windows may contain ``UNIVERSE_SOURCE_CSV = 'C:\\Users\\...'``.
    That is a ``SyntaxError`` (``\\U`` unicode escape) when compiled. The real path is
    set immediately after ``exec`` via :func:`_apply_screened_csv_override`.

    Line-based stripping is more reliable than a single-line regex (``\\r``, BOM,
    multiple assignments, odd spacing).
    """
    hdr = 'UNIVERSE_SOURCE_CSV = ""  # overwritten by v6_bucket4_bootstrap after exec\n'
    out: list[str] = []
    for line in cell2_src.splitlines(keepends=True):
        pre_hash = line.split("#", 1)[0]
        if re.search(r"\bUNIVERSE_SOURCE_CSV\s*=", pre_hash):
            continue
        out.append(line)
    return hdr + "".join(out)


def _apply_screened_csv_override(globs: dict[str, Any], screened_csv: str | Path) -> None:
    """Point Bucket-4 parameters at the same screener CSV as GTP (no ``exec`` string embedding)."""
    p = str(Path(screened_csv).resolve())
    globs["UNIVERSE_SOURCE_CSV"] = p
    inv_fn = globs.get("_inverse_levered_etf_tickers_from_screener")
    if callable(inv_fn):
        globs["INVERSE_LEVERED_ETF_UNIVERSE"] = inv_fn(p)
    globs["BETA_SOURCE_CSV"] = p


def _strip_broken_yfinance_alias(cell1_src: str) -> str:
    # Bucket 4 notebook has a stray ``import yfinance as yfple`` line that raises ImportError.
    return cell1_src.replace("import yfinance as yfple\n", "")


def _apply_cfg_start_end(globs: dict[str, Any]) -> None:
    cfg = globs.get("CFG")
    if not isinstance(cfg, dict):
        return
    sd = cfg.get("start_date")
    if sd is not None and str(sd).strip():
        globs["START"] = str(sd)
    ed = cfg.get("end_date")
    if ed is not None and str(ed).strip():
        globs["END"] = str(ed).strip()
    else:
        globs["END"] = None


def ensure_v6_bucket4_globals_from_notebook(
    globs: dict[str, Any],
    *,
    repo_root: Path | None = None,
    screened_csv: str | Path,
    bucket4_notebook: Path | None = None,
) -> None:
    """
    Populate ``get_ibkr_borrow_map``, ``_V6_PAIR_CACHE``, ``v6_opt2_h_daily_map``,
    ``V6_OPT2_H_BASE``, ``run_bucket4_backtest_dynamic_h``, ``v6_opt2_rebal_index``, etc.

    Parameters
    ----------
    globs :
        Target namespace (pass ``globals()`` from the notebook).
    repo_root :
        Repo root for ``sys.path`` if ``scripts.*`` imports are needed; optional.
    screened_csv :
        Path to ``etf_screened_today.csv`` (same file the GTP mirror will use).
    bucket4_notebook :
        Override path to ``Bucket_4_Backtest.ipynb`` for tests.
    """
    nb_path = Path(bucket4_notebook) if bucket4_notebook is not None else _BUCKET4_NB
    if not nb_path.is_file():
        raise FileNotFoundError(f"Bucket 4 notebook not found: {nb_path}")

    if repo_root is not None:
        rr = Path(repo_root).resolve()
        import sys

        if str(rr) not in sys.path:
            sys.path.insert(0, str(rr))

    raw = nb_path.read_text(encoding="utf-8")
    nb = json.loads(raw)

    for idx in _CELL_IDX:
        cell = nb["cells"][idx]
        if cell.get("cell_type") != "code":
            raise RuntimeError(f"Bucket_4_Backtest.ipynb cell {idx} expected code, got {cell.get('cell_type')}")
        src = "".join(cell.get("source") or [])
        if idx == 1:
            src = _strip_broken_yfinance_alias(src)
        if idx == 2:
            src = _neutralize_universe_source_assignment(src)
        name = f"{nb_path.name}:cell{idx}"
        exec(compile(src, name, "exec"), globs, globs)
        if idx == 2:
            _apply_screened_csv_override(globs, screened_csv)
        if idx == 5:
            _apply_cfg_start_end(globs)
        if idx == 14:
            from scripts.bucket4_dynamic_bt import run_bucket4_backtest_dynamic_h as _run_b4_engine

            globs["run_bucket4_backtest_dynamic_h"] = _run_b4_engine

    need = ("get_ibkr_borrow_map", "_V6_PAIR_CACHE", "v6_opt2_h_daily_map", "V6_OPT2_H_BASE")
    miss = [n for n in need if n not in globs or globs[n] is None]
    if miss:
        raise RuntimeError(f"v6 bootstrap incomplete; missing: {miss}")

    print(
        f"[v6 bootstrap] OK — executed Bucket_4_Backtest.ipynb cells {_CELL_IDX} | "
        f"UNIVERSE_SOURCE_CSV={globs.get('UNIVERSE_SOURCE_CSV')!s}"
    )


def ensure_v6_bucket4_globals_module(
    mod: ModuleType,
    *,
    repo_root: Path | None = None,
    screened_csv: str | Path,
    bucket4_notebook: Path | None = None,
) -> None:
    """Same as :func:`ensure_v6_bucket4_globals_from_notebook` but using a module ``__dict__``."""
    ensure_v6_bucket4_globals_from_notebook(
        mod.__dict__,
        repo_root=repo_root,
        screened_csv=screened_csv,
        bucket4_notebook=bucket4_notebook,
    )
