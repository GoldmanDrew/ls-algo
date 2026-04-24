"""One-off: patch v15 notebook pair_day pricing / notionals. Run from repo root."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    ROOT / "notebooks" / "Diamond_Creek_Backtest_v15.ipynb",
    ROOT / "notebooks" / "Diamond_Creek_Backtest.ipynb",
    ROOT / "notebooks" / "Diamond_Creek_Backtest_attribution_aligned.ipynb",
]

OLD_A = (
    "        pair_day = {}\n"
    "        for ek, pos in pair_pos.items():\n"
    "            uk = pos[\"und\"]\n"
    "            lsh = float(pos.get(\"long_sh\", 0))\n"
    "            ssh = float(pos.get(\"short_sh\", 0))\n"
    "            und_long_val = max(lsh, 0.0) * px.get(uk, 0)\n"
    "            und_short_val = max(-lsh, 0.0) * px.get(uk, 0)\n"
    "            etf_short_val = max(-ssh, 0.0) * px.get(ek, 0)\n"
    "            long_val = und_long_val\n"
    "            short_val = und_short_val + etf_short_val\n"
    "            pair_day[ek] = {\n"
    "                \"date\": today,\n"
    "                \"etf\": ek,\n"
    "                \"under\": uk,\n"
    "                \"long_sh\": lsh,\n"
    "                \"short_sh\": ssh,\n"
    "                \"long_notional_usd\": long_val,\n"
    "                \"short_notional_usd\": short_val,\n"
    "                \"gross_notional_usd\": long_val + short_val,\n"
    "                \"net_notional_usd\": lsh * px.get(uk, 0) + ssh * px.get(ek, 0),\n"
)

NEW_A = (
    "        pair_day = {}\n"
    "        for ek, pos in pair_pos.items():\n"
    "            uk = pos[\"und\"]\n"
    "            lsh = float(pos.get(\"long_sh\", 0))\n"
    "            ssh = float(pos.get(\"short_sh\", 0))\n"
    "            p_u = float(px.get(uk, 0.0) or 0.0)\n"
    "            p_e = float(px.get(ek, 0.0) or 0.0)\n"
    "            long_n_sig = lsh * p_u\n"
    "            short_n_sig = ssh * p_e\n"
    "            und_long_val = max(lsh, 0.0) * p_u\n"
    "            und_short_val = max(-lsh, 0.0) * p_u\n"
    "            etf_short_val = max(-ssh, 0.0) * p_e\n"
    "            short_fin_basis = und_short_val + etf_short_val\n"
    "            pair_day[ek] = {\n"
    "                \"date\": today,\n"
    "                \"etf\": ek,\n"
    "                \"under\": uk,\n"
    "                \"long_sh\": lsh,\n"
    "                \"short_sh\": ssh,\n"
    "                \"underlying_price\": p_u if p_u > 0 else float(\"nan\"),\n"
    "                \"etf_price\": p_e if p_e > 0 else float(\"nan\"),\n"
    "                \"long_notional_usd\": long_n_sig,\n"
    "                \"short_notional_usd\": short_n_sig,\n"
    "                \"short_financing_basis_usd\": short_fin_basis,\n"
    "                \"long_margin_basis_usd\": und_long_val,\n"
    "                \"gross_notional_usd\": abs(lsh) * p_u + abs(ssh) * p_e,\n"
    "                \"net_notional_usd\": long_n_sig + short_n_sig,\n"
)

OLD_B = (
    "                if ek not in pair_day and new_pos is not None:\n"
    "                    lsh = float(new_pos.get(\"long_sh\", 0.0))\n"
    "                    ssh = float(new_pos.get(\"short_sh\", 0.0))\n"
    "                    und_long_val = max(lsh, 0.0) * px.get(und, 0)\n"
    "                    und_short_val = max(-lsh, 0.0) * px.get(und, 0)\n"
    "                    etf_short_val = max(-ssh, 0.0) * px.get(ek, 0)\n"
    "                    long_val = und_long_val\n"
    "                    short_val = und_short_val + etf_short_val\n"
    "                    pair_day[ek] = {\n"
    "                        \"date\": today,\n"
    "                        \"etf\": ek,\n"
    "                        \"under\": und,\n"
    "                        \"long_sh\": lsh,\n"
    "                        \"short_sh\": ssh,\n"
    "                        \"long_notional_usd\": long_val,\n"
    "                        \"short_notional_usd\": short_val,\n"
    "                        \"gross_notional_usd\": long_val + short_val,\n"
    "                        \"net_notional_usd\": lsh * px.get(und, 0) + ssh * px.get(ek, 0),\n"
)

NEW_B = (
    "                if ek not in pair_day and new_pos is not None:\n"
    "                    lsh = float(new_pos.get(\"long_sh\", 0.0))\n"
    "                    ssh = float(new_pos.get(\"short_sh\", 0.0))\n"
    "                    p_u = float(px.get(und, 0.0) or 0.0)\n"
    "                    p_e = float(px.get(ek, 0.0) or 0.0)\n"
    "                    long_n_sig = lsh * p_u\n"
    "                    short_n_sig = ssh * p_e\n"
    "                    und_long_val = max(lsh, 0.0) * p_u\n"
    "                    und_short_val = max(-lsh, 0.0) * p_u\n"
    "                    etf_short_val = max(-ssh, 0.0) * p_e\n"
    "                    short_fin_basis = und_short_val + etf_short_val\n"
    "                    pair_day[ek] = {\n"
    "                        \"date\": today,\n"
    "                        \"etf\": ek,\n"
    "                        \"under\": und,\n"
    "                        \"long_sh\": lsh,\n"
    "                        \"short_sh\": ssh,\n"
    "                        \"underlying_price\": p_u if p_u > 0 else float(\"nan\"),\n"
    "                        \"etf_price\": p_e if p_e > 0 else float(\"nan\"),\n"
    "                        \"long_notional_usd\": long_n_sig,\n"
    "                        \"short_notional_usd\": short_n_sig,\n"
    "                        \"short_financing_basis_usd\": short_fin_basis,\n"
    "                        \"long_margin_basis_usd\": und_long_val,\n"
    "                        \"gross_notional_usd\": abs(lsh) * p_u + abs(ssh) * p_e,\n"
    "                        \"net_notional_usd\": long_n_sig + short_n_sig,\n"
)


def _patch_nb(path: Path) -> bool:
    if not path.is_file():
        return False
    nb = json.loads(path.read_text(encoding="utf-8"))
    n = 0
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", [])
        if not isinstance(src, list):
            continue
        text = "".join(src)
        if OLD_A not in text:
            continue  # already patched or different engine snapshot
        text = text.replace(OLD_A, NEW_A, 1)
        text = text.replace(OLD_B, NEW_B, 1)
        if OLD_A in text or OLD_B in text:
            raise RuntimeError("duplicate blocks or missing OLD_B")
        text = text.replace(
            "        l_notional = sum(d[\"long_notional_usd\"] for d in pair_day.values())\n",
            "        l_notional = sum(max(0.0, d[\"long_notional_usd\"]) for d in pair_day.values())\n",
        )
        text = text.replace(
            "            debit_alloc = (\n"
            "                margin_debit_d * (max(0.0, d[\"long_notional_usd\"]) / l_notional)\n"
            "                if l_notional > 0\n"
            "                else 0.0\n"
            "            )\n"
            "            credit = short_credit_interest(\n"
            "                max(0.0, d[\"short_notional_usd\"]), ff, CFG[\"credit_spread\"]\n"
            "            )\n",
            "            debit_alloc = (\n"
            "                margin_debit_d * (max(0.0, d[\"long_notional_usd\"]) / l_notional)\n"
            "                if l_notional > 0\n"
            "                else 0.0\n"
            "            )\n"
            "            credit = short_credit_interest(\n"
            "                float(d[\"short_financing_basis_usd\"]), ff, CFG[\"credit_spread\"]\n"
            "            )\n",
        )
        cell["source"] = text.splitlines(keepends=True)
        n += 1
    if n == 0:
        return False
    if n != 1:
        raise SystemExit(f"{path}: expected exactly 1 patched cell, got {n}")
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("patched", path)
    return True


def main() -> None:
    any_ok = False
    for nb in NOTEBOOKS:
        if _patch_nb(nb):
            any_ok = True
    if not any_ok:
        print("no notebooks needed patching (already updated or missing blocks)")


if __name__ == "__main__":
    main()
