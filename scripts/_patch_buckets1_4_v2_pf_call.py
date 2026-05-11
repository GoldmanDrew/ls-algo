"""Fix Buckets1-4_v2.ipynb cell 6: reload v6_b4_pf_weights + safe kwargs for compute_v6_b4_pf_weight_dict."""

from __future__ import annotations

import json
from pathlib import Path

OLD_CALL_NO_IBKR = """    weights, diag, meta = compute_v6_b4_pf_weight_dict(
        pair_cache=globals()["_V6_PAIR_CACHE"],
        v6_opt2_h_daily_map=globals()["v6_opt2_h_daily_map"],
        screened_csv=str(SCREENED_PATH),
        closes_broad=closes_broad,
        norm_sym=norm_sym,
        get_ibkr_borrow_map=_v6_backtest_borrow_map,
        opt2_h_base=float(globals()["V6_OPT2_H_BASE"]),
        params=p,
    )
"""

OLD_CALL_WITH_IBKR = """    weights, diag, meta = compute_v6_b4_pf_weight_dict(
        pair_cache=globals()["_V6_PAIR_CACHE"],
        v6_opt2_h_daily_map=globals()["v6_opt2_h_daily_map"],
        screened_csv=str(SCREENED_PATH),
        closes_broad=closes_broad,
        norm_sym=norm_sym,
        get_ibkr_borrow_map=_v6_backtest_borrow_map,
        opt2_h_base=float(globals()["V6_OPT2_H_BASE"]),
        params=p,
        use_ibkr_uvix_borrow=bool((globals().get("EXP") or {}).get("b4_weights_use_ibkr_uvix_borrow", True)),
    )
"""

NEW_CALL = """    import inspect

    _exp = globals().get("EXP") or {}
    _pf_kw = dict(
        pair_cache=globals()["_V6_PAIR_CACHE"],
        v6_opt2_h_daily_map=globals()["v6_opt2_h_daily_map"],
        screened_csv=str(SCREENED_PATH),
        closes_broad=closes_broad,
        norm_sym=norm_sym,
        get_ibkr_borrow_map=_v6_backtest_borrow_map,
        opt2_h_base=float(globals()["V6_OPT2_H_BASE"]),
        params=p,
    )
    if "use_ibkr_uvix_borrow" in inspect.signature(compute_v6_b4_pf_weight_dict).parameters:
        _pf_kw["use_ibkr_uvix_borrow"] = bool(_exp.get("b4_weights_use_ibkr_uvix_borrow", True))
    weights, diag, meta = compute_v6_b4_pf_weight_dict(**_pf_kw)
"""

OLD_HEAD = "from scripts.v6_b4_pf_weights import V6PfParams, compute_v6_b4_pf_weight_dict\n"
NEW_HEAD = (
    "import importlib\n\n"
    "import scripts.v6_b4_pf_weights as _v6_b4_pf_mod\n\n"
    "importlib.reload(_v6_b4_pf_mod)\n"
    "from scripts.v6_b4_pf_weights import V6PfParams, compute_v6_b4_pf_weight_dict\n"
)


def main() -> None:
    p = Path(__file__).resolve().parent.parent / "notebooks" / "Buckets1-4_v2.ipynb"
    nb = json.loads(p.read_text(encoding="utf-8"))
    cell = nb["cells"][6]
    src = "".join(cell["source"])
    if "_pf_kw" in src and "importlib.reload(_v6_b4_pf_mod)" in src:
        print("Already fixed")
        return
    if OLD_CALL_WITH_IBKR in src:
        src = src.replace(OLD_CALL_WITH_IBKR, NEW_CALL, 1)
    elif OLD_CALL_NO_IBKR in src:
        src = src.replace(OLD_CALL_NO_IBKR, NEW_CALL, 1)
    else:
        raise SystemExit("compute_v6_b4_pf_weight_dict call block not found")
    if OLD_HEAD not in src:
        raise SystemExit("import head not found")
    src = src.replace(OLD_HEAD, NEW_HEAD, 1)
    cell["source"] = [ln if ln.endswith("\n") else ln + "\n" for ln in src.splitlines(keepends=True)]
    p.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Patched {p} cell 6")


if __name__ == "__main__":
    main()
