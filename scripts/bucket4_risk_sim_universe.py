"""Universe + weight resolution for the B4 risk simulator dataset.

Includes proposed-book pairs, eligible screener names, pair-override templates,
force-includes (SMZ/CBRZ/APLZ), and current-book rows. When ``gross_target_usd``
is zero but short locate exists, uses ``optimal_gross_target_usd``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from trade_plan_targets import maybe_merge_optimal_targets  # noqa: E402

B4_SLEEVES = frozenset({"inverse_decay_bucket4", "volatility_etp_bucket5"})
FORCE_INCLUDE_ETFS = frozenset({"SMZ", "CBRZ", "APLZ"})
PAIR_OVERRIDES_PATH = REPO / "config" / "pair_overrides.yml"


def _norm(x: object) -> str:
    return str(x).strip().upper().replace(".", "-")


def _bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    return s.fillna(False).astype(str).str.lower().isin({"1", "true", "t", "yes", "y"})


def locate_available(row: pd.Series) -> bool:
    sh = pd.to_numeric(row.get("shares_available"), errors="coerce")
    if pd.isna(sh) or float(sh) <= 0:
        return False
    if "purgatory_no_locate" in row.index and _bool_series(pd.Series([row.get("purgatory_no_locate")])).iloc[0]:
        return False
    if "exclude_no_shares" in row.index and _bool_series(pd.Series([row.get("exclude_no_shares")])).iloc[0]:
        return False
    return True


def _edge_signal(row: pd.Series) -> float:
    ne = pd.to_numeric(row.get("net_edge_p50_annual"), errors="coerce")
    if np.isfinite(ne) and ne > 0:
        return float(ne)
    nd = pd.to_numeric(row.get("net_decay_annual"), errors="coerce")
    if np.isfinite(nd):
        return max(float(nd), 1e-6)
    return 0.05


def _median_positive(s: pd.Series) -> float:
    v = pd.to_numeric(s, errors="coerce").replace(0.0, np.nan).dropna()
    v = v[v > 0]
    return float(v.median()) if len(v) else float("nan")


def _load_pair_override_keys() -> list[tuple[str, str, dict]]:
    if not PAIR_OVERRIDES_PATH.is_file():
        return []
    raw = yaml.safe_load(PAIR_OVERRIDES_PATH.read_text(encoding="utf-8")) or {}
    out: list[tuple[str, str, dict]] = []
    for key, spec in (raw.get("pair_overrides") or {}).items():
        if "/" not in str(key):
            continue
        etf, und = str(key).split("/", 1)
        out.append((_norm(etf), _norm(und), spec or {}))
    return out


def _read_proposed(run_date: str, runs_root: Path) -> pd.DataFrame:
    runs_data = REPO / "data" / "runs"
    for path in (
        runs_data / run_date / "proposed_trades.csv",
        runs_root / run_date / "proposed_trades.csv",
        REPO / "data" / "proposed_trades.csv",
    ):
        if path.is_file():
            df = pd.read_csv(path)
            df = maybe_merge_optimal_targets(df, run_date, runs_root=runs_data)
            return df
    return pd.DataFrame()


def _read_screener(run_date: str, runs_root: Path) -> pd.DataFrame:
    for path in (
        runs_root / run_date / "etf_screened_today.csv",
        REPO / "data" / "runs" / run_date / "etf_screened_today.csv",
        REPO / "data" / "etf_screened_today.csv",
    ):
        if path.is_file():
            return pd.read_csv(path)
    return pd.DataFrame()


def _screener_b4_mask(sc: pd.DataFrame) -> pd.Series:
    if sc.empty:
        return pd.Series(dtype=bool)
    bucket = sc.get("bucket", pd.Series("", index=sc.index)).astype(str).str.lower()
    beta = pd.to_numeric(sc.get("Delta"), errors="coerce")
    inv_ok = sc.get("inverse_shortable", pd.Series(True, index=sc.index))
    inv_ok = _bool_series(inv_ok) if inv_ok is not None else pd.Series(True, index=sc.index)
    is_b4 = bucket.str.contains("bucket_4", na=False)
    is_b5 = bucket.str.contains("bucket_5", na=False)
    vol_etp = sc.get("product_class", pd.Series("", index=sc.index)).astype(str).str.lower().eq("volatility_etp")
    neg_beta = beta < 0
    return (is_b4 | is_b5 | vol_etp) & neg_beta & inv_ok


def _safe_mult(row: pd.Series) -> float:
    raw = pd.to_numeric(row.get("pair_override_gross_mult"), errors="coerce")
    return float(raw) if np.isfinite(raw) else 1.0


def _load_strategy_exclusions() -> tuple[frozenset[str], frozenset[str]]:
    cfg_path = REPO / "config" / "strategy_config.yml"
    if not cfg_path.is_file():
        return frozenset(), frozenset()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    b4 = (cfg.get("strategy") or {}).get("sleeves", {}).get("inverse_decay_bucket4", {})
    rules = b4.get("rules") or {}
    excluded = frozenset(_norm(x) for x in (rules.get("excluded_etfs") or []))
    flow = (cfg.get("strategy") or {}).get("flow_program", {}).get("universe", {}).get("shorts") or []
    return excluded, frozenset(_norm(x) for x in flow)


def _gtp_eligible_mask(
    sc: pd.DataFrame,
    *,
    excluded: frozenset[str],
    flow: frozenset[str],
) -> pd.Series:
    """Mirror GTP B4/B5 universe gates (screener rows that could receive structural targets)."""
    if sc.empty:
        return pd.Series(dtype=bool)
    b4m = _screener_b4_mask(sc)
    blk = _bool_series(sc.get("strategy_blacklisted", pd.Series(False, index=sc.index)))
    ne = pd.to_numeric(sc.get("net_edge_p50_annual"), errors="coerce")
    edge_ok = ne.fillna(-1.0) >= 0.0
    not_flow = ~sc["ETF"].map(_norm).isin(flow)
    not_excl = ~sc["ETF"].map(_norm).isin(excluded)
    return b4m & ~blk & edge_ok & not_flow & not_excl


def _finite_float(val, default: float = 0.0) -> float:
    v = float(pd.to_numeric(val, errors="coerce"))
    return default if not np.isfinite(v) else v


def resolve_sim_gross(
    row: pd.Series,
    *,
    median_structural: float,
    force_include: bool = False,
    gtp_eligible: bool = False,
) -> tuple[float, str]:
    """Return (sim_gross_usd, weight_source) for risk-sim weighting."""
    gross = _finite_float(row.get("gross_target_usd"), 0.0)
    optimal = _finite_float(row.get("optimal_gross_target_usd"), 0.0)
    loc_ok = locate_available(row)

    if gross > 0:
        return gross, "proposed"
    if optimal > 0 and loc_ok:
        return optimal, "optimal"
    if optimal > 0 and force_include:
        return optimal, "optimal_forced"
    if force_include and not loc_ok and np.isfinite(median_structural) and median_structural > 0:
        mult = _safe_mult(row)
        return median_structural * mult, "structural_proxy"

    # GTP-eligible / force-include stress names without optimal: edge-scaled proxy when locate exists
    if (force_include or gtp_eligible) and loc_ok and np.isfinite(median_structural) and median_structural > 0:
        mult = _safe_mult(row) if force_include else 1.0
        edge = _edge_signal(row)
        return median_structural * mult * max(edge / 0.5, 0.25), "screener_proxy"

    return 0.0, "excluded"


def _merge_optimal_targets(df: pd.DataFrame, run_date: str, runs_root: Path) -> pd.DataFrame:
    """Attach ``optimal_gross_target_usd`` from ``optimal_targets.csv`` when present."""
    if df.empty:
        return df
    path = runs_root / run_date / "optimal_targets.csv"
    if not path.is_file():
        return df
    opt = pd.read_csv(path)
    if opt.empty or "ETF" not in opt.columns:
        return df
    opt = opt.copy()
    opt["ETF"] = opt["ETF"].map(_norm)
    opt["Underlying"] = opt["Underlying"].map(_norm)
    keep = ["ETF", "Underlying", "optimal_gross_target_usd"]
    keep += [c for c in ("optimal_long_usd", "optimal_short_usd") if c in opt.columns]
    opt = opt[keep].drop_duplicates(subset=["ETF", "Underlying"], keep="first")
    out = df.merge(opt, on=["ETF", "Underlying"], how="left", suffixes=("", "_opt"))
    if "optimal_gross_target_usd_opt" in out.columns:
        cur = pd.to_numeric(out.get("optimal_gross_target_usd"), errors="coerce")
        add = pd.to_numeric(out["optimal_gross_target_usd_opt"], errors="coerce")
        out["optimal_gross_target_usd"] = cur.where(cur.notna() & (cur > 0), add)
        out = out.drop(columns=["optimal_gross_target_usd_opt"])
    return out


def load_risk_sim_universe(run_date: str, *, runs_root: Path | None = None) -> pd.DataFrame:
    """Build de-duplicated B4/B5 universe with resolved sim weights."""
    runs_root = runs_root or (REPO / "data" / "runs")
    proposed = _read_proposed(run_date, runs_root)
    screener = _read_screener(run_date, runs_root)
    override_keys = _load_pair_override_keys()
    excluded, flow = _load_strategy_exclusions()
    override_etfs = frozenset(etf for etf, _, _ in override_keys)

    rows: dict[tuple[str, str], dict] = {}

    def _upsert(row: dict, *, source: str) -> None:
        etf = _norm(row.get("ETF") or row.get("etf"))
        und = _norm(row.get("Underlying") or row.get("underlying"))
        if not etf or not und:
            return
        key = (etf, und)
        cur = rows.get(key, {})
        merged = {**cur, **{k: v for k, v in row.items() if pd.notna(v) and v != ""}}
        merged["ETF"] = etf
        merged["Underlying"] = und
        merged.setdefault("sleeve", "inverse_decay_bucket4")
        merged[f"_src_{source}"] = True
        rows[key] = merged

    if not proposed.empty:
        p = proposed.copy()
        p["ETF"] = p["ETF"].map(_norm)
        p["Underlying"] = p["Underlying"].map(_norm)
        sleeve = p.get("sleeve", pd.Series("", index=p.index)).astype(str)
        mask = sleeve.isin(B4_SLEEVES)
        for _, r in p[mask].iterrows():
            _upsert(r.to_dict(), source="proposed")

    if not screener.empty:
        sc = screener.copy()
        sc["ETF"] = sc["ETF"].map(_norm)
        sc["Underlying"] = sc["Underlying"].map(_norm)
        gtp_m = _gtp_eligible_mask(sc, excluded=excluded, flow=flow)
        add_m = gtp_m | sc["ETF"].isin(FORCE_INCLUDE_ETFS) | sc["ETF"].isin(override_etfs)
        for idx, r in sc[add_m].iterrows():
            d = r.to_dict()
            d["_gtp_eligible"] = bool(gtp_m.loc[idx])
            if "sleeve" not in d or not str(d.get("sleeve", "")).strip():
                bucket = str(r.get("bucket", "")).lower()
                d["sleeve"] = (
                    "volatility_etp_bucket5"
                    if "bucket_5" in bucket or str(r.get("product_class", "")).lower() == "volatility_etp"
                    else "inverse_decay_bucket4"
                )
            _upsert(d, source="screener")

    for etf, und, spec in override_keys:
        _upsert({
            "ETF": etf,
            "Underlying": und,
            "pair_override_gross_mult": spec.get("gross_mult", 1.0),
            "pair_override_hedge_add": spec.get("hedge_ratio_add", 0.0),
            "pair_override_note": spec.get("note", ""),
        }, source="override")

    for etf in FORCE_INCLUDE_ETFS:
        if screener.empty:
            continue
        m = screener[screener["ETF"].astype(str).str.upper().map(_norm) == etf]
        if len(m):
            _upsert({**m.iloc[0].to_dict(), "_force_include": True}, source="force")

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows.values())
    df["ETF"] = df["ETF"].map(_norm)
    df["Underlying"] = df["Underlying"].map(_norm)
    df = _merge_optimal_targets(df, run_date, runs_root)

    # Median structural gross from proposed positive rows
    prop_gross = pd.to_numeric(df.get("gross_target_usd"), errors="coerce").fillna(0.0)
    prop_opt = pd.to_numeric(df.get("optimal_gross_target_usd"), errors="coerce").fillna(0.0)
    med = _median_positive(prop_opt.where(prop_gross > 0, prop_opt))
    if not np.isfinite(med) or med <= 0:
        med = _median_positive(prop_gross)
    if not np.isfinite(med) or med <= 0:
        med = 15_000.0

    sim_gross, sources = [], []
    for _, r in df.iterrows():
        force = _norm(r.get("ETF")) in FORCE_INCLUDE_ETFS or bool(r.get("_force_include"))
        gtp_ok = bool(r.get("_gtp_eligible"))
        g, src = resolve_sim_gross(
            r, median_structural=med, force_include=force, gtp_eligible=gtp_ok,
        )
        sim_gross.append(g)
        sources.append(src)
    df["sim_gross_usd"] = sim_gross
    df["weight_source"] = sources
    df["in_book"] = pd.to_numeric(df.get("gross_target_usd"), errors="coerce").fillna(0.0).gt(0)
    df["locate_ok"] = df.apply(locate_available, axis=1)
    df["blacklisted"] = _bool_series(df.get("strategy_blacklisted", pd.Series(False, index=df.index)))

    # Keep rows with positive sim weight OR force-include (for stress toggles with proxy weight)
    keep = df["sim_gross_usd"].gt(0) | df["ETF"].isin(FORCE_INCLUDE_ETFS)
    df = df[keep].copy()
    df = df.sort_values("sim_gross_usd", ascending=False).drop_duplicates(subset=["ETF"], keep="first")
    return df.reset_index(drop=True)
