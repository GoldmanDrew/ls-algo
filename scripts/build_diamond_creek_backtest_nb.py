"""
One-off builder: notebooks/Diamond_Creek_Backtest.ipynb from v15 cells + weights + export + fund compare.
Run: python scripts/build_diamond_creek_backtest_nb.py
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "notebooks" / "Diamond_Creek_Backtest_v15.ipynb"
OUT = ROOT / "notebooks" / "Diamond_Creek_Backtest.ipynb"


def _strip_outputs(cell: dict) -> dict:
    c = deepcopy(cell)
    c.pop("execution_count", None)
    if c.get("cell_type") == "code":
        c["outputs"] = []
    return c


def main() -> None:
    nb = json.loads(SRC.read_text(encoding="utf-8"))
    cells: list[dict] = []

    title = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Diamond Creek ETF Arbitrage — Backtest (fixed CSV weights)\n",
            "\n",
            "Minimal slice of `Diamond_Creek_Backtest_v15.ipynb`: same config, universe, borrow, prices, "
            "costs, and **v15 engine**. **`PAIR_WEIGHTS`** are loaded from a CSV (no optimizer / sweeps).\n",
        ],
    }
    cells.append(title)

    # v15 cells 2..15: Setup .. UNIVERSE
    for idx in range(2, 16):
        cells.append(_strip_outputs(nb["cells"][idx]))

    weights_md = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Position weights (`PAIR_WEIGHTS`)\n",
            "\n",
            "- **CSV path:** `notebooks/data/backtest/Diamond_Creek_Backtest_position_weights.csv`\n",
            "- **Column:** `weight` — applied when `(etf, und)` matches a row in `UNIVERSE` (symbols via `norm_sym`).\n",
            "- **Keys:** the v15 engine resolves weights with `PAIR_WEIGHTS.get(etf, 1/n_pairs)` inside "
            "`_frac_rows_for_day`; we set **every ETF in `UNIVERSE` explicitly** (default **0** if missing from the CSV) "
            "so there is **no silent equal-weight fallback**.\n",
            "- **Normalization:** weights are **not** forced to sum to 1. In `_apply_allocs`, `wcomb = wb * rp` and "
            "`tw = sum(wcomb)` rescales allocations each rebalance — **only relative weights** matter.\n",
        ],
    }
    weights_code = {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "source": [
            "WEIGHTS_CSV = Path(\"notebooks/data/backtest/Diamond_Creek_Backtest_position_weights.csv\")\n",
            "_wdf = pd.read_csv(WEIGHTS_CSV)\n",
            "_wdf[\"etf\"] = _wdf[\"etf\"].astype(str).map(norm_sym)\n",
            "_wdf[\"und\"] = _wdf[\"und\"].astype(str).map(norm_sym)\n",
            "_wdf[\"weight\"] = pd.to_numeric(_wdf[\"weight\"], errors=\"coerce\").fillna(0.0)\n",
            "_univ_keys = {(norm_sym(e), norm_sym(u)) for e, u, _ in UNIVERSE}\n",
            "PAIR_WEIGHTS = {norm_sym(e): 0.0 for e, _, _ in UNIVERSE}\n",
            "for _, r in _wdf.iterrows():\n",
            "    key = (r[\"etf\"], r[\"und\"])\n",
            "    if key not in _univ_keys:\n",
            "        continue\n",
            "    PAIR_WEIGHTS[r[\"etf\"]] = float(r[\"weight\"])\n",
            "_pos = sum(1 for v in PAIR_WEIGHTS.values() if v > 0)\n",
            "_raw = float(sum(PAIR_WEIGHTS.values()))\n",
            "print(f\"PAIR_WEIGHTS from CSV: {_pos} positive ETFs, raw sum {_raw:.6f} (relative-only; engine normalizes wcomb).\")\n",
        ],
    }
    cells.append(weights_md)
    cells.append(weights_code)

    # Costs + engine (v15 cells 16–19); replace v15 "equal weight" perf banner with CSV-weight wording.
    for idx in range(16, 20):
        cells.append(_strip_outputs(nb["cells"][idx]))
    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Performance summary (fixed `PAIR_WEIGHTS` from CSV)\n",
            ],
        }
    )

    perf_def = {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "source": [
            "def perf(nav):\n",
            "    r = nav.pct_change().dropna()\n",
            "    ny = len(nav) / TRADING_DAYS\n",
            "    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / ny) - 1\n",
            "    vol = r.std() * (TRADING_DAYS**0.5)\n",
            "    sr = cagr / vol if vol > 0 else 0\n",
            "    neg = r[r < 0]\n",
            "    dv = neg.std() * (TRADING_DAYS**0.5)\n",
            "    sortino = cagr / dv if dv > 0 else 0\n",
            "    dd = (nav - nav.cummax()) / nav.cummax()\n",
            "    calmar = cagr / abs(dd.min()) if dd.min() < 0 else 0\n",
            "    mo = nav.resample(\"ME\").last().pct_change().dropna()\n",
            "    return {\n",
            "        \"CAGR\": cagr,\n",
            "        \"Vol\": vol,\n",
            "        \"Sharpe\": sr,\n",
            "        \"Sortino\": sortino,\n",
            "        \"Max DD\": dd.min(),\n",
            "        \"Calmar\": calmar,\n",
            "        \"Monthly Win%\": (mo > 0).mean(),\n",
            "        \"Final NAV\": nav.iloc[-1],\n",
            "        \"P&L\": nav.iloc[-1] - nav.iloc[0],\n",
            "    }\n",
        ],
    }
    cells.append(perf_def)

    perf_run = {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "source": [
            "_ref = max(LEVERAGE_RUNS)\n",
            "p = perf(ALL_BT[_ref][\"nav\"])\n",
            "bt = ALL_BT[_ref]\n",
            "print(\"Reference gross leverage:\", _ref)\n",
            "for k, v in p.items():\n",
            "    if \"NAV\" in k or \"P&L\" in k:\n",
            "        print(f\"  {k}: ${v:,.0f}\")\n",
            "    elif \"%\" in k:\n",
            "        print(f\"  {k}: {v:.1%}\")\n",
            "    else:\n",
            "        print(f\"  {k}: {v:.2%}\")\n",
            "print(f\"  Txn Costs: ${bt['cum_costs'].iloc[-1]:,.0f}\")\n",
            "print(f\"  Borrow: ${bt['cum_borrow'].iloc[-1]:,.0f}\")\n",
            "print(f\"  Margin debit (cum): ${bt['cum_margin_debit'].iloc[-1]:,.0f}\")\n",
            "print(f\"  Long P&L (cum): ${bt['cum_long_pnl'].iloc[-1]:,.0f}\")\n",
            "print(f\"  Short P&L (cum): ${bt['cum_short_pnl'].iloc[-1]:,.0f}\")\n",
        ],
    }
    cells.append(perf_run)

    export_md = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Ledger export — **January 2023 only** (active pairs)\n",
            "\n",
            "Exports `ALL_PAIRS`, one tab per **active** pair, `portfolio_financing` (book **T-costs** and **margin** as portfolio flows), and `reconciliation_daily`.\n",
            "\n",
            "- **Ex-txn at pair** (`daily_pair_net_ex_txn`); one **book** `book_daily_txn` line. "
            "Margin/borrow in pair rows is all-in financing; book margin lines tie to `ALL_BT` in `portfolio_financing`.\n",
            "- `include_only_active_pairs=True`: drops pair-month rows with no gross in Jan (0 across the month).\n",
        ],
    }
    export_code = {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "source": [
            "import sys\n",
            "from pathlib import Path\n",
            "\n",
            "_ROOT = Path.cwd()\n",
            "if not (_ROOT / \"scripts\" / \"export_diamond_creek_v15_pair_ledger.py\").exists():\n",
            "    _ROOT = _ROOT.parent\n",
            "if str(_ROOT) not in sys.path:\n",
            "    sys.path.insert(0, str(_ROOT))\n",
            "\n",
            "import importlib\n",
            "import scripts.export_diamond_creek_v15_pair_ledger as _exp_ledger\n",
            "importlib.reload(_exp_ledger)\n",
            "\n",
            "from scripts.export_diamond_creek_v15_pair_ledger import export_v15_pair_ledger_to_excel\n",
            "\n",
            "_ref = max(LEVERAGE_RUNS)\n",
            "_base = float(globals().get(\"ATTRIBUTION_BASE_CAPITAL\", CFG.get(\"capital_usd\", 10_000_000.0)))\n",
            "\n",
            "_out = _ROOT / \"notebooks/data/backtest/Diamond_Creek_fixed_weights_daily_pair_ledger.xlsx\"\n",
            "_ledger_artifacts = export_v15_pair_ledger_to_excel(\n",
            "    ALL_PAIR_DAILY[_ref],\n",
            "    ALL_BT[_ref],\n",
            "    _out,\n",
            "    full_date_range=False,\n",
            "    include_only_active_pairs=True,\n",
            "    active_min_abs_gross_sum_usd=0.5,\n",
            "    attribution_base_capital=_base,\n",
            "    ref_leverage=_ref,\n",
            "    include_per_pair_sheets=True,\n",
            "    return_artifacts=True,\n",
            ")\n",
            "JAN_2023_RECONCILIATION = _ledger_artifacts[\"reconciliation_daily\"]\n",
            "JAN_2023_ALL_PAIRS = _ledger_artifacts[\"all_pairs\"]\n",
            "print(f\"Wrote: {_out.resolve()}\")\n",
            "print(_ledger_artifacts[\"metrics\"])\n",
        ],
    }
    cells.append(export_md)
    cells.append(export_code)

    compare_md = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Reconciliation (January 2023) — `DC ETF Arb` + self\n",
            "\n",
            "1. **Fund self-check:** sum of **Daily PnL** in Jan 2023 vs **Monthly Attribution** (pre-fee; net if fees=0 in month).\n",
            "2. **Sim self-check** (`reconciliation_daily` from the export): "
            "attribution = sum(pairs, ex-txn) − one book t-cost matches book / NAV; T-costs and margin at book.\n",
            "3. **Fund vs sim (Jan):** sum sim **attribution_daily_net** vs sum fund **daily pnl**.\n",
        ],
    }
    compare_code = {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "source": [
            "import sys\n",
            "from pathlib import Path\n",
            "\n",
            "_ROOT = Path.cwd()\n",
            "if not (_ROOT / \"scripts\" / \"compare_dc_etf_attribution.py\").exists():\n",
            "    _ROOT = _ROOT.parent\n",
            "if str(_ROOT) not in sys.path:\n",
            "    sys.path.insert(0, str(_ROOT))\n",
            "\n",
            "import importlib\n",
            "import scripts.compare_dc_etf_attribution as _cmp_dc_etf\n",
            "importlib.reload(_cmp_dc_etf)\n",
            "\n",
            "from IPython.display import display\n",
            "\n",
            "from scripts.compare_dc_etf_attribution import (\n",
            "    JAN_2023_PERIOD,\n",
            "    JAN_2023_START,\n",
            "    JAN_2023_END,\n",
            "    aggregate_sim_from_bt,\n",
            "    build_sim_daily_for_compare,\n",
            "    check_fund_daily_rolls_to_monthly_pre_fee,\n",
            "    filter_fund_to_jan_2023,\n",
            "    fund_and_sim_tieout,\n",
            "    fund_excel_tieout,\n",
            "    load_fund_daily,\n",
            "    load_fund_monthly,\n",
            "    reconcile_monthly,\n",
            "    self_check_sim_reconciliation_daily,\n",
            "    sim_recon_window,\n",
            "    reconcile_daily,\n",
            ")\n",
            "\n",
            "FUND_ATTRIBUTION_XLSX = Path(r\"C:/Users/werdn/Downloads/DC ETF Arb Attribution.xlsx\")\n",
            "\n",
            "_ref = max(LEVERAGE_RUNS)\n",
            "fund_m = load_fund_monthly(FUND_ATTRIBUTION_XLSX)\n",
            "fund_d = load_fund_daily(FUND_ATTRIBUTION_XLSX)\n",
            "fund_m_j, fund_d_j = filter_fund_to_jan_2023(fund_m, fund_d)\n",
            "print(\">>> Fund: DC ETF workbook internal — daily Jan vs monthly Jan 2023\")\n",
            "print(fund_excel_tieout(fund_d, fund_m, year_month=\"2023-01\"))\n",
            "print(f\"Roll-up: {check_fund_daily_rolls_to_monthly_pre_fee(fund_m_j, fund_d_j)!r}\")\n",
            "print(\">>> Sim: ledger (Jan 2023) self-check — ex-txn pairs vs book; book t-costs once\")\n",
            "if \"JAN_2023_RECONCILIATION\" in globals():\n",
            "    print(self_check_sim_reconciliation_daily(JAN_2023_RECONCILIATION))\n",
            "else:\n",
            "    print(\"Run the ledger export cell first (JAN_2023_RECONCILIATION).\")\n",
            "_base_cap = float(CFG.get(\"capital_usd\", 10_000_000.0))\n",
            "_, _monthly_sim = aggregate_sim_from_bt(ALL_BT[_ref], attribution_base_capital=_base_cap)\n",
            "import pandas as pd\n",
            "_mjan = _monthly_sim[pd.to_datetime(_monthly_sim[\"date\"]).dt.to_period(\"M\") == JAN_2023_PERIOD]\n",
            "if not _mjan.empty:\n",
            "    print(\">>> Fund vs sim: monthly (Jan 2023)\")\n",
            "    display(reconcile_monthly(fund_m, _mjan))\n",
            "_sim_d = build_sim_daily_for_compare(ALL_BT[_ref], ALL_PAIR_DAILY[_ref], attribution_base_capital=_base_cap)\n",
            "daily_recon = reconcile_daily(fund_d_j, sim_recon_window(_sim_d, JAN_2023_START, JAN_2023_END))\n",
            "print(\">>> Fund vs sim: daily (Jan)\")\n",
            "display(daily_recon)\n",
            "if \"JAN_2023_RECONCILIATION\" in globals():\n",
            "    print(fund_and_sim_tieout(JAN_2023_RECONCILIATION, fund_d_j))\n",
        ],
    }
    cells.append(compare_md)
    cells.append(compare_code)

    out_nb = {
        "nbformat": nb.get("nbformat", 4),
        "nbformat_minor": nb.get("nbformat_minor", 5),
        "metadata": nb.get("metadata", {}),
        "cells": cells,
    }
    OUT.write_text(json.dumps(out_nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
