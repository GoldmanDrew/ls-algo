#!/usr/bin/env python3
"""One-shot Beta→Delta rename for hedge-ratio terminology (ls-algo + siblings).

Run from repo root:
  python scripts/migrate_beta_to_delta.py [--repo ROOT] [--data-only]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

# CSV / JSON column and key renames (hedge ratio only)
CSV_COL_MAP = {
    "Beta_product_class": "Delta_product_class",
    "Beta_n_obs": "Delta_n_obs",
    "Beta_source": "Delta_source",
    "Beta_se": "Delta_se",
    "Beta_resid_sigma_annual": "Delta_resid_sigma_annual",
    "Beta_horizon_chosen": "Delta_horizon_chosen",
    "Beta_quality": "Delta_quality",
    "Beta_prior_mu": "Delta_prior_mu",
    "Beta_prior_tau": "Delta_prior_tau",
    "Beta_prior_source": "Delta_prior_source",
    "Beta": "Delta",
}

# Order matters: longer tokens first
TEXT_REPLACEMENTS = [
    ("Beta_product_class", "Delta_product_class"),
    ("Beta_prior_source", "Delta_prior_source"),
    ("Beta_prior_tau", "Delta_prior_tau"),
    ("Beta_prior_mu", "Delta_prior_mu"),
    ("Beta_resid_sigma_annual", "Delta_resid_sigma_annual"),
    ("Beta_horizon_chosen", "Delta_horizon_chosen"),
    ("Beta_n_obs", "Delta_n_obs"),
    ("Beta_source", "Delta_source"),
    ("Beta_quality", "Delta_quality"),
    ("Beta_se", "Delta_se"),
    ("beta_estimator", "delta_estimator"),
    ("BETA_ESTIMATOR", "DELTA_ESTIMATOR"),
    ("compute_beta_for_hedging", "compute_delta_for_hedging"),
    ("compute_beta_adjusted_net_notional", "compute_delta_adjusted_net_notional"),
    ("classify_beta_product_class", "classify_delta_product_class"),
    ("build_yieldboost_family_priors", "build_yieldboost_family_priors"),  # noop anchor
    ("load_etf_beta_map", "load_etf_delta_map"),
    ("validate_beta_estimator", "validate_delta_estimator"),
    ("compare_beta_methods", "compare_delta_methods"),
    ("test_beta_estimator", "test_delta_estimator"),
    ("SIGN_CONFLICT_MIN_BETA", "SIGN_CONFLICT_MIN_DELTA"),
    ("HIGH_BETA_THRESHOLD", "HIGH_DELTA_THRESHOLD"),
    ("MIN_BETA_ABS", "MIN_DELTA_ABS"),
    ("SIZING_V2_MIN_BETA_OBS", "SIZING_V2_MIN_DELTA_OBS"),
    ("use_beta_from_screened", "use_delta_from_screened"),
    ("screened_Beta_csv", "screened_Delta_csv"),
    ("all_pairs_with_betas", "all_pairs_with_deltas"),
    ("add_betas", "add_deltas"),
    ("BetaPrior", "DeltaPrior"),
    ("BetaResult", "DeltaResult"),
    ("beta_from_underlying", "delta_from_underlying"),
    ("beta_adjusted", "delta_adjusted"),
    ("beta-adj", "delta-adj"),
    ("beta_adj", "delta_adj"),
    ("beta_weight", "delta_weight"),
    ("beta_normalized", "delta_normalized"),
    ("beta-normalized", "delta-normalized"),
    ("etf_to_beta", "etf_to_delta"),
    ("peer_betas", "peer_deltas"),
    ("min_beta_days", "min_delta_days"),
    ("min_beta_used", "min_delta_used"),
    ("min_beta_obs", "min_delta_obs"),
    ("core_beta_min", "core_delta_min"),
    ("beta_f", "delta_f"),
    ("betas_out", "deltas_out"),
    ("beta_nobs_out", "delta_nobs_out"),
    ("betas_computed", "deltas_computed"),
    ("beta_col", "delta_col"),
    ("beta_map", "delta_map"),
    ("beta_abs", "delta_abs"),
    ("beta_robust", "delta_robust"),
    ("beta_post", "delta_post"),
    ("beta_raw", "delta_raw"),
    ("beta_h1", "delta_h1"),
    ("beta_h5", "delta_h5"),
    ("beta_n_obs", "delta_n_obs"),
    ("beta_nobs", "delta_nobs"),
    ("beta_source", "delta_source"),
    ("beta_se", "delta_se"),
    ("beta_product_class", "delta_product_class"),
    ("beta_prior", "delta_prior"),
    ("beta_horizon", "delta_horizon"),
    ("beta_quality", "delta_quality"),
    ("screen_beta", "screen_delta"),
    ("missing_beta", "missing_delta"),
    ("etf_to_beta_map", "etf_to_delta_map"),
    ("_bucket_for_etf_beta", "_bucket_for_etf_delta"),
    ("_bucket_from_beta", "_bucket_from_delta"),
    ("test_product_class_high_beta_letf", "test_product_class_high_delta_letf"),
    ("passive_low_beta", "passive_low_delta"),
    ("low-beta", "low-delta"),
    ("low_beta", "low_delta"),
    ("high-beta", "high-delta"),
    ("high_beta", "high_delta"),
    ("short-beta", "short-delta"),
    ("1_over_beta_hedge", "1_over_delta_hedge"),
    ("hedge-beta", "hedge-delta"),
    ("hedge beta", "hedge delta"),
    ("hedge β", "hedge δ"),
    ("hedge-beta", "hedge-delta"),
    ('"Beta"', '"Delta"'),
    ("'Beta'", "'Delta'"),
    ("row.Beta", "row.Delta"),
    ('row["Beta"]', 'row["Delta"]'),
    ('row.get("Beta")', 'row.get("Delta")'),
    ('["Beta"]', '["Delta"]'),
    ('.Beta', '.Delta'),
]

# JSON keys for hedge ratio in dashboard / nav (not factor beta_to_spy)
JSON_HEDGE_KEYS = {
    "beta": "delta",
    "beta_n_obs": "delta_n_obs",
    "beta_source": "delta_source",
    "beta_se": "delta_se",
    "beta_product_class": "delta_product_class",
}

SKIP_PATH_PARTS = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
}

SKIP_FILES = {
    "beta_loader.py",
    "migrate_beta_to_delta.py",
    "factor_map.py",
}

SKIP_SUFFIXES = {".pyc", ".png", ".jpg", ".xml", ".parquet"}

PY_GLOB = ("**/*.py",)
CODE_GLOB = ("**/*.py", "**/*.yml", "**/*.yaml", "**/*.md", "**/*.js", "**/*.html")
DATA_GLOB_CSV = ("**/etf_screened_today.csv", "**/all_pairs_with_betas.csv", "**/all_pairs_with_deltas.csv")
DATA_GLOB_JSON = ("**/dashboard_data.json",)


def _should_skip(path: Path) -> bool:
    parts = set(path.parts)
    if parts & SKIP_PATH_PARTS:
        return True
    if path.name in SKIP_FILES:
        return True
    if path.suffix in SKIP_SUFFIXES:
        return True
    return False


def apply_text_replacements(text: str, *, is_python: bool = False) -> str:
    for old, new in TEXT_REPLACEMENTS:
        if old == new:
            continue
        text = text.replace(old, new)
    # Remaining standalone Beta column references in Python strings
    if is_python:
        text = re.sub(r'\bBeta\b(?=\s*[=:\]])', "Delta", text)
    return text


def migrate_file_text(path: Path) -> bool:
    try:
        raw = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return False
    orig = raw
    is_py = path.suffix == ".py"
    # Skip factor-beta loader module content
    if path.name == "beta_loader.py":
        return False
    new = apply_text_replacements(raw, is_python=is_py)
    if new != orig:
        path.write_text(new, encoding="utf-8")
        return True
    return False


def migrate_csv(path: Path) -> bool:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return False
    if not lines:
        return False
    header = lines[0]
    orig_header = header
    for old, new in sorted(CSV_COL_MAP.items(), key=lambda x: -len(x[0])):
        # Replace whole column names in header
        parts = header.split(",")
        parts = [new if p == old else p for p in parts]
        header = ",".join(parts)
    if header == orig_header and "Beta" not in orig_header:
        return False
    lines[0] = header
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    if path.name == "all_pairs_with_betas.csv":
        new_path = path.with_name("all_pairs_with_deltas.csv")
        if new_path != path:
            shutil.copy2(path, new_path)
    return True


def migrate_dashboard_json(path: Path) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False

    changed = False

    def fix_row(row: dict) -> None:
        nonlocal changed
        for old, new in JSON_HEDGE_KEYS.items():
            if old in row and new not in row:
                row[new] = row.pop(old)
                changed = True
            elif old in row:
                del row[old]
                changed = True

    if isinstance(data.get("symbols"), list):
        for row in data["symbols"]:
            if isinstance(row, dict):
                fix_row(row)

    meta = data.get("meta") or data
    if isinstance(meta, dict):
        dm = meta.get("decay_method") or data.get("decay_method")
        if isinstance(dm, str) and "beta" in dm:
            new_dm = dm.replace("1_over_beta_hedge", "1_over_delta_hedge").replace("beta", "delta")
            if "decay_method" in data:
                data["decay_method"] = new_dm
            else:
                meta["decay_method"] = new_dm
            changed = True

    if changed:
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return changed


def migrate_nav_jsonl(path: Path) -> bool:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return False
    out_lines = []
    changed = False
    for line in lines:
        if not line.strip():
            out_lines.append(line)
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            out_lines.append(line)
            continue
        if "beta" in obj and "delta" not in obj:
            obj["delta"] = obj.pop("beta")
            changed = True
        out_lines.append(json.dumps(obj, separators=(",", ":")))
    if changed:
        path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return changed


def create_delta_estimator(repo: Path) -> None:
    src = repo / "beta_estimator.py"
    dst = repo / "delta_estimator.py"
    if not src.exists():
        return
    text = src.read_text(encoding="utf-8")
    text = apply_text_replacements(text, is_python=True)
    # Module docstring
    text = text.replace("hedge-beta", "hedge-delta").replace("hedge-beta", "hedge-delta")
    text = text.replace("β̂_robust", "δ̂_robust").replace("β_robust", "delta_robust")
    text = text.replace("β̂", "δ̂").replace("β_post", "delta_post")
    # BetaResult fields
    text = re.sub(
        r"(class DeltaResult:.*?)(    beta: float)",
        r"\1    delta: float",
        text,
        flags=re.DOTALL,
        count=1,
    )
    text = text.replace("beta=float(", "delta=float(")
    text = text.replace("beta_se=float(", "delta_se=float(")
    text = text.replace("beta=float(np.round(beta_post", "delta=float(np.round(delta_post")
    text = text.replace("beta_post", "delta_post")
    text = text.replace('return float(beta),', 'return float(delta),')
    text = text.replace('return float(beta)', 'return float(delta)')
    text = text.replace('"beta": float(beta)', '"delta": float(delta)')
    text = text.replace('beta, resid =', 'delta, resid =')
    text = text.replace('y - beta * x', 'y - delta * x')
    text = text.replace('beta = float(np.sum', 'delta = float(np.sum')
    text = text.replace('beta = float(np.sum(ww', 'delta = float(np.sum(ww')
    text = text.replace('var_beta =', 'var_delta =')
    text = text.replace('return float(np.sqrt(var_beta))', 'return float(np.sqrt(var_delta))')
    text = text.replace('if not np.isfinite(beta):', 'if not np.isfinite(delta):')
    text = text.replace('.beta)', '.delta)')
    text = text.replace('result.beta', 'result.delta')
    text = text.replace('compute_delta_for_hedging', 'compute_delta_for_hedging')  # idempotent
    dst.write_text(text, encoding="utf-8")
    if src.exists():
        src.unlink()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--data-only", action="store_true")
    ap.add_argument("--also", type=Path, action="append", default=[])
    args = ap.parse_args()
    repos = [args.repo.resolve()] + [p.resolve() for p in args.also]

    for repo in repos:
        print(f"\n=== {repo} ===")
        if not args.data_only and (repo / "beta_estimator.py").exists():
            create_delta_estimator(repo)
            print("  created delta_estimator.py, removed beta_estimator.py")

        if not args.data_only:
            n = 0
            for pattern in CODE_GLOB:
                for path in repo.glob(pattern):
                    if _should_skip(path):
                        continue
                    if migrate_file_text(path):
                        n += 1
            print(f"  updated {n} code/doc files")

        cn = 0
        for pattern in DATA_GLOB_CSV:
            for path in repo.glob(pattern):
                if _should_skip(path):
                    continue
                if migrate_csv(path):
                    cn += 1
        print(f"  updated {cn} CSV files")

        jn = 0
        for pattern in DATA_GLOB_JSON:
            for path in repo.glob(pattern):
                if migrate_dashboard_json(path):
                    jn += 1
        for path in repo.glob("**/nav_forecasts/**/*.jsonl"):
            if migrate_nav_jsonl(path):
                jn += 1
        for path in repo.glob("**/nav_forecasts/_*.json"):
            if path.name.startswith("_") and migrate_nav_jsonl(path):
                jn += 1
        print(f"  updated {jn} JSON/JSONL artifacts")

        # Rename BETA_ESTIMATOR.md
        for old_name, new_name in [("BETA_ESTIMATOR.md", "DELTA_ESTIMATOR.md")]:
            o, n = repo / old_name, repo / new_name
            if o.exists() and not n.exists():
                o.rename(n)
            elif o.exists():
                migrate_file_text(o)
                o.rename(n)

        # Rename test file
        for old, new in [
            ("tests/test_beta_estimator.py", "tests/test_delta_estimator.py"),
            ("scripts/validate_beta_estimator.py", "scripts/validate_delta_estimator.py"),
            ("scripts/compare_beta_methods.py", "scripts/compare_delta_methods.py"),
        ]:
            o, n = repo / old, repo / new
            if o.exists() and not n.exists():
                o.rename(n)

    return 0


if __name__ == "__main__":
    sys.exit(main())
