"""
Restructure notebooks/Buckets1-4Backtest.ipynb:
  - Insert production sizing (GTP + PAIR_WEIGHTS) immediately after UNIVERSE cell.
  - Merge etf-dashboard borrow_avg into borrow cell.
  - Remove legacy equal-weight / v8 PnL-decay sizing cells.
  - Remove duplicate trailing GTP+cov cells (replaced by top section).
  - Patch backtest engine to require PAIR_WEIGHTS from GTP cell.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "Buckets1-4Backtest.ipynb"


NEW_MD = """## Production portfolio sizing (`generate_trade_plan.py` + `execute_flow_program.py`)

After **UNIVERSE** is fixed, we size the backtest using the same sleeve rules as production:

- **Bucket 1 / 2 / 4** — `mirror_generate_trade_plan_sizing` → **PAIR_WEIGHTS** (share of gross by `gross_target_usd` per pair).
- **Bucket 3** — flow sleeve: fixed weights from `strategy_config.yml` (`flow_program.weighting.weights`); tracked separately from pair gross like production.

Borrow on ETF legs for simulation uses **historical average** from `etf-dashboard/data/dashboard_data.json` (`borrow_avg_annual`) merged in the borrow cell below.
"""

NEW_CODE = r'''# --- Production PAIR_WEIGHTS (generate_trade_plan mirror) + flow weights ---
import sys
from pathlib import Path
import yaml
import pandas as pd

_nb_root = Path.cwd().resolve()
for _p in (_nb_root, _nb_root.parent):
    if (_p / "config" / "strategy_config.yml").exists():
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
        break

from scripts.gtp_sizing_mirror import mirror_generate_trade_plan_sizing, pair_weights_from_gtp_mirror
import execute_flow_program as _efp

_cfg_path = next((_p / "config" / "strategy_config.yml" for _p in (_nb_root, _nb_root.parent) if (_p / "config" / "strategy_config.yml").exists()), None)
if _cfg_path is None:
    raise FileNotFoundError("strategy_config.yml not found")
with open(_cfg_path, "r", encoding="utf-8") as _f:
    GTP_CFG = yaml.safe_load(_f)

_screen_paths = [Path("data/etf_screened_today.csv"), Path("../data/etf_screened_today.csv")]
GTP_SCREENED_PATH = next((p for p in _screen_paths if p.exists()), None)
if GTP_SCREENED_PATH is None:
    raise FileNotFoundError("etf_screened_today.csv not found — run daily_screener first")
GTP_SCREENED = pd.read_csv(GTP_SCREENED_PATH)

GTP_MIRROR_RUN_DATE = str(pd.Timestamp.today().date())
HYSTERESIS_TOUCH_DISK = False

GTP_MIRROR_DF, GTP_MIRROR_DIAG = mirror_generate_trade_plan_sizing(
    GTP_SCREENED,
    GTP_CFG,
    run_date=GTP_MIRROR_RUN_DATE,
    hysteresis_touch_disk=HYSTERESIS_TOUCH_DISK,
)

if "UNIVERSE" not in globals():
    raise RuntimeError("UNIVERSE must be defined before this cell.")

PAIR_WEIGHTS = pair_weights_from_gtp_mirror(GTP_MIRROR_DF, UNIVERSE)
print("[GTP]", GTP_MIRROR_DIAG)
print(f"[GTP] PAIR_WEIGHTS -> {len(PAIR_WEIGHTS)} ETFs | sum={sum(PAIR_WEIGHTS.values()):.4f} | max={max(PAIR_WEIGHTS.values()):.2%}")

# Flow program (bucket 3) — fixed weights for reference / downstream cells
_flow_cfg = (GTP_CFG.get("portfolio") or {}).get("sleeves", {}).get("flow_program", {}) or {}
try:
    _tickers, _w = _efp._parse_universe_and_weights(_flow_cfg)
    FLOW_ETFS = list(_tickers)
    FLOW_WEIGHTS_ARR = _w
    FLOW_WEIGHTS = {t: float(_w[i]) for i, t in enumerate(_tickers)}
    print(f"[flow_program] {len(FLOW_ETFS)} tickers | weight sum={float(_w.sum()):.4f}")
except Exception as _fe:
    FLOW_ETFS = []
    FLOW_WEIGHTS = {}
    print(f"[flow_program] skip: {_fe}")
'''


def _cell_should_drop(s: str) -> bool:
    markers = (
        "## Performance Summary (equal weight)",
        "## v8 Diamond Creek Fund Weights",
        "EW_BT = dict(ALL_BT)",
        "Sample-book optimizer: PnL/Gross + NET Decay",
        "Re-run weighted backtest now",
        'if "w_df" not in globals()',
        "## Performance — Diamond Creek Fund weighted",
        "# ---- EW diagnostic: PnL/Gross by pair",
        "## Production sizing mirror (`generate_trade_plan.py`)",
        "## Covariance overlay (research layer)",
        "# Covariance overlay demo (underlying log returns",
    )
    return any(m in s for m in markers)


def _patch_borrow_cell(s: str) -> str:
    if "merge_dashboard_borrow_into_map" in s:
        return s
    needle = "        print(f\"[IBKR] borrow overlay skipped: {_e}\")\n"
    if needle not in s:
        # older notebook without IBKR block — append dashboard at end
        return (
            s.rstrip()
            + "\n\n# --- etf-dashboard historical average borrow (overrides ETF leg where present) ---\n"
            + "try:\n"
            + "    from scripts.gtp_dashboard_borrow import merge_dashboard_borrow_into_map\n"
            + "    BORROW_MAP = merge_dashboard_borrow_into_map(BORROW_MAP, all_etf_syms)\n"
            + "except Exception as _e:\n"
            + "    print(f\"[dashboard] borrow merge skipped: {_e}\")\n"
        )
    insert = (
        "\n"
        "# --- etf-dashboard historical average borrow (overrides ETF leg where present) ---\n"
        "try:\n"
        "    from scripts.gtp_dashboard_borrow import merge_dashboard_borrow_into_map\n"
        "    BORROW_MAP = merge_dashboard_borrow_into_map(BORROW_MAP, all_etf_syms)\n"
        "except Exception as _e:\n"
        "    print(f\"[dashboard] borrow merge skipped: {_e}\")\n"
    )
    return s.replace(needle, needle + insert)


def _patch_engine_cell(s: str) -> str:
    old = '''if "PAIR_WEIGHTS" not in dir():
    PAIR_WEIGHTS = {e: 1.0 / n_pairs for e, _, _ in UNIVERSE}
    print(f"Using equal weights: {1.0/n_pairs:.4%} per pair")
else:
    print(f"Using custom PAIR_WEIGHTS ({len(PAIR_WEIGHTS)} pairs, top weight: {max(PAIR_WEIGHTS.values()):.2%})")'''
    new = '''if "PAIR_WEIGHTS" not in dir() or not isinstance(PAIR_WEIGHTS, dict) or not PAIR_WEIGHTS:
    raise RuntimeError(
        "PAIR_WEIGHTS missing — run the **Production portfolio sizing** cell (right after UNIVERSE) first."
    )
print(
    f"Using production PAIR_WEIGHTS ({len(PAIR_WEIGHTS)} ETFs, top weight: {max(PAIR_WEIGHTS.values()):.2%})"
)'''
    if old not in s:
        if "PAIR_WEIGHTS missing" in s:
            return s
        raise RuntimeError("Engine cell pattern not found; notebook format changed")
    return s.replace(old, new)


def main() -> None:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    cells = nb["cells"]

    # Drop legacy / duplicate sections
    cells = [c for c in cells if not (_cell_should_drop("".join(c.get("source", []))))]

    # Find universe cell index
    uidx = None
    for i, c in enumerate(cells):
        if c.get("cell_type") == "code" and "UNIVERSE = [(e, u, b) for e, u, b in CANDIDATES" in "".join(
            c.get("source", [])
        ):
            uidx = i
            break
    if uidx is None:
        raise RuntimeError("Could not find UNIVERSE cell")

    insert = [
        {"cell_type": "markdown", "metadata": {}, "source": [NEW_MD]},
        {"cell_type": "code", "metadata": {}, "source": [NEW_CODE]},
    ]
    cells = cells[: uidx + 1] + insert + cells[uidx + 1 :]

    # Patch borrow + engine
    for c in cells:
        if c.get("cell_type") != "code":
            continue
        src = "".join(c.get("source", []))
        if "# ---- Borrow rates (historical average first" in src:
            c["source"] = [_patch_borrow_cell(src)]
        if "ALL_BT = {}" in src and "PAIR_WEIGHTS" in src:
            c["source"] = [_patch_engine_cell(src)]

    nb["cells"] = cells
    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", NB, "n_cells=", len(cells))


if __name__ == "__main__":
    main()
