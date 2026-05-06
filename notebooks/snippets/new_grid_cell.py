# --- Joint QCQP grid: named-configs runner (G9 diversification + audit) ---
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import sys
from pathlib import Path

import pandas as pd

from generate_trade_plan import _norm_sym as _gtp_norm

# Repo root on path + reload allocate (kernel may cache an older module without
# ``ema_smooth_signal`` after editing ``scripts/buckets_joint_qcqp_allocate.py``).
import importlib as _importlib_bja

_alloc_repo = next(
    (
        _p
        for _p in (Path.cwd().resolve(), *Path.cwd().resolve().parents)
        if (_p / "scripts" / "buckets_joint_qcqp_allocate.py").is_file()
    ),
    None,
)
if _alloc_repo is None:
    raise RuntimeError(
        "scripts/buckets_joint_qcqp_allocate.py not found - cwd should be ls-algo or notebooks/"
    )
if str(_alloc_repo) not in sys.path:
    sys.path.insert(0, str(_alloc_repo))

import scripts.buckets_joint_qcqp_allocate as _buckets_joint_qcqp_allocate_mod

_importlib_bja.reload(_buckets_joint_qcqp_allocate_mod)
from scripts.buckets_joint_qcqp_allocate import (
    build_joint_bundle,
    ema_smooth_signal,
    load_notebook_cell_source,
    run_joint_qcqp_single,
)
try:
    from scripts.buckets_joint_qcqp_allocate import write_positions_md
except ImportError:
    write_positions_md = None  # older allocate; positions.md skipped silently

if "perf" not in globals():
    raise RuntimeError("Run the cell that defines perf(nav) before this grid cell.")

for _g in ("GTP_MIRROR_DF", "GTP_MIRROR_DIAG", "UNIVERSE", "PRICES", "LEVERAGE_RUNS", "TRADING_DAYS"):
    if _g not in globals():
        raise RuntimeError(f"Missing {_g} - run prior setup cells.")

# DCQ on path (same as joint cell)
_repo = next(
    (_p for _p in (Path.cwd().resolve(), *Path.cwd().resolve().parents) if (_p / "scripts" / "gtp_sizing_mirror.py").exists()),
    None,
)
if _repo is None:
    raise RuntimeError("ls-algo repo root not found")
_dcq = Path(os.environ.get("DCQ_ROOT", "")).resolve() if os.environ.get("DCQ_ROOT") else _repo.parent / "Diamond-Creek-Quant"
if not (_dcq / "dcq" / "sizing" / "sizing_v2.py").is_file():
    raise FileNotFoundError(f"Diamond-Creek-Quant not found: {_dcq}")
if str(_dcq) not in sys.path:
    sys.path.insert(0, str(_dcq))

try:
    import cvxpy as _cvxpy_check  # noqa: F401
except ImportError:
    import importlib
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "cvxpy>=1.3,<1.6", "-q"])
    importlib.invalidate_caches()
    import cvxpy as _cvxpy_check  # noqa: F401

from dcq.sizing.sizing_v2 import pair_weights_qcqp_joint  # noqa: E402

import inspect as _inspect_build_bundle  # noqa: E402


def _filter_build_bundle_kw(kwargs: dict) -> dict:
    """Drop kwargs this ``build_joint_bundle`` does not accept (older ls-algo)."""
    try:
        _ok = set(_inspect_build_bundle.signature(build_joint_bundle).parameters)
    except (TypeError, ValueError):
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in _ok}


# Per-run weights + diagnostics (pair tables, ETF weights, joint_meta JSON).
JOINT_GRID_SAVE_WEIGHTS = True
JOINT_GRID_WEIGHTS_ROOT = Path("data") / "backtest" / "joint_qcqp_grid_weights"

# Per-bucket caps applied as a min() on top of the book-wide caps.
JOINT_GRID_SLEEVE_CAP_MAX_PAIR_FRAC = 0.25
JOINT_GRID_SLEEVE_CAP_MAX_UNDER_FRAC = 0.20

# Pulls per-pair mu toward the active cohort mean before QCQP (0 = off).
JOINT_GRID_MU_SHRINK = 0.25

# ---------------------------------------------------------------------------
# Sizing-v2 stability defaults (G1) - mirrored at the top of the standalone
# joint cell ``3f99fcea``; ``globals().get(...)`` keeps user-set values.
# ---------------------------------------------------------------------------
SIZING_V2_MIN_DECAY_OBS = globals().get("SIZING_V2_MIN_DECAY_OBS", 120)
SIZING_V2_MIN_BETA_OBS = globals().get("SIZING_V2_MIN_BETA_OBS", 120)
SIZING_V2_CONFIDENCE_HAIRCUT = globals().get("SIZING_V2_CONFIDENCE_HAIRCUT", True)
SIZING_V2_CONF_FLOOR = globals().get("SIZING_V2_CONF_FLOOR", 0.25)
SIZING_V2_N_OBS_FULL = globals().get("SIZING_V2_N_OBS_FULL", 252)
SIZING_V2_TURNOVER_LAMBDA = globals().get("SIZING_V2_TURNOVER_LAMBDA", 1.0)
SIZING_V2_TURNOVER_L1_MAX = globals().get("SIZING_V2_TURNOVER_L1_MAX", None)
SIZING_V2_EMA_HALFLIFE_WEEKS = globals().get("SIZING_V2_EMA_HALFLIFE_WEEKS", 6.0)
SIZING_V2_SCORE_AWARE_PAIR_CAP = globals().get("SIZING_V2_SCORE_AWARE_PAIR_CAP", True)

# When True, accumulate each run's pair_book_weights into _prev_pw_book and pass
# it into the next run_joint_qcqp_single call so the turnover anchor sees an
# actual sequential history within the sweep.
JOINT_GRID_USE_PREV_AS_ANCHOR = True

_nb = Path.cwd().resolve()
_candidates = [_nb / "Buckets1-4Backtest.ipynb", _nb.parent / "notebooks" / "Buckets1-4Backtest.ipynb"]
NB_PATH = next((p for p in _candidates if p.is_file()), None)
if NB_PATH is None:
    raise FileNotFoundError("Buckets1-4Backtest.ipynb not found from cwd")
ENGINE_SRC = load_notebook_cell_source(NB_PATH, "b9669f21")

_pw_save = dict(PAIR_WEIGHTS) if isinstance(PAIR_WEIGHTS, dict) else {}
_pf_save = dict(PAIR_FRAC_BY_KEY) if isinstance(PAIR_FRAC_BY_KEY, dict) else {}


# ---------------------------------------------------------------------------
# JOINT_CONFIGS: named, hand-curated configurations.
#
# Each entry is one full kwarg set passed to run_joint_qcqp_single (no
# cartesian product). Add or comment out entries to control what runs.
# ---------------------------------------------------------------------------
_DEFAULT_BASE = dict(
    book_sigma_target=0.20,
    book_max_pair=0.10,
    book_max_underlying=0.20,
    weight_ridge_lambda=0.0,
    w_min_floor_frac=0.0,
    turnover_lambda=float(SIZING_V2_TURNOVER_LAMBDA),
    confidence_haircut=bool(SIZING_V2_CONFIDENCE_HAIRCUT),
    min_decay_obs=int(SIZING_V2_MIN_DECAY_OBS),
    min_beta_obs=int(SIZING_V2_MIN_BETA_OBS),
    ema_halflife_weeks=float(SIZING_V2_EMA_HALFLIFE_WEEKS) if SIZING_V2_EMA_HALFLIFE_WEEKS else None,
    # G9 diversification knobs (off by default).
    entropy_lambda=0.0,
    entropy_reference="prior_w_pre",
    edge_temperature=1.0,
    mv_lambda=0.0,
    eff_n_min_pairs=None,
)


def _cfg(name: str, **overrides) -> dict:
    cfg = dict(_DEFAULT_BASE)
    cfg.update(overrides)
    cfg["name"] = str(name)
    return cfg


JOINT_CONFIGS: list[dict] = [
    # Reference: prior behavior (no smoothing).
    _cfg("baseline_v1"),
    # Pure entropy / KL smoothing.
    _cfg("entropy_low", entropy_lambda=0.25, edge_temperature=1.0),
    _cfg("entropy_high_T15", entropy_lambda=1.0, edge_temperature=1.5,
         book_max_pair=0.05, book_max_underlying=0.15, w_min_floor_frac=0.10),
    # Soft mean-variance only (couples names through Sigma).
    _cfg("mv_only", mv_lambda=10.0),
    # Combined: edge-tilted but smooth and risk-aware (recommended baseline).
    _cfg("entropy_plus_mv",
         entropy_lambda=0.5, edge_temperature=1.25, mv_lambda=5.0,
         book_max_pair=0.05, book_max_underlying=0.15, w_min_floor_frac=0.10),
    # Hard concentration cap on top of smoothing.
    _cfg("effN_floor_25",
         entropy_lambda=0.25, edge_temperature=1.25, eff_n_min_pairs=25,
         book_max_pair=0.05, book_max_underlying=0.15, w_min_floor_frac=0.10),
]

JOINT_GRID_MAX_RUNS = int(globals().get("JOINT_GRID_MAX_RUNS", 12))
if len(JOINT_CONFIGS) > JOINT_GRID_MAX_RUNS:
    print(f"[joint_grid] truncating to first {JOINT_GRID_MAX_RUNS} configs (set JOINT_GRID_MAX_RUNS higher)")
    JOINT_CONFIGS = JOINT_CONFIGS[:JOINT_GRID_MAX_RUNS]


def _config_hash(cfg: dict) -> str:
    keys = sorted(k for k in cfg if k != "name")
    payload = json.dumps([(k, cfg[k]) for k in keys], default=str, sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


# Per-session cache so re-running this cell skips already-computed configs.
JOINT_RUN_CACHE = globals().get("JOINT_RUN_CACHE", {})
globals()["JOINT_RUN_CACHE"] = JOINT_RUN_CACHE


def _save_joint_grid_run_artifacts(
    run_id: int,
    *,
    name: str,
    pair_frac_book: dict,
    sleeve_targets: dict,
    pw_j: dict,
    pf_j: dict,
    meta: dict,
    cfg: dict,
    root: Path,
    diag_by_bucket: dict | None = None,
    prev_pw_book: dict | None = None,
) -> Path:
    """Write CSV (+ parquet when available) for one named-config run."""
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / f"run_{int(run_id):03d}_{name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg_row = {"run_id": int(run_id), "name": str(name), **{k: v for k, v in cfg.items() if k != "name"}}
    pd.DataFrame([cfg_row]).to_csv(run_dir / "config.csv", index=False)

    diag_lookup: dict[tuple[str, str], dict] = {}
    if diag_by_bucket:
        for _b, _df in diag_by_bucket.items():
            if _df is None or _df.empty:
                continue
            for _, _r in _df.iterrows():
                k = (str(_r.get("ETF")), str(_r.get("Underlying")))
                diag_lookup[k] = {
                    "n_obs_decay": float(_r.get("n_obs_decay", 0.0) or 0.0),
                    "n_obs_beta": float(_r.get("n_obs_beta", 0.0) or 0.0),
                    "mu_used": float(_r.get("mu_used", 0.0) or 0.0),
                    "sigma_eff": float(_r.get("sigma_eff", 0.0) or 0.0),
                    "w_pre": float(_r.get("w_pre", 0.0) or 0.0),
                    "q_prior": float(_r.get("q_prior", 0.0) or 0.0),
                    "haircut_factor": float(_r.get("confidence_factor", 1.0) or 1.0),
                    "ema_applied": bool(_r.get("ema_applied", False)),
                }

    prev_pw_book = prev_pw_book or {}
    pr_rows: list[dict] = []
    for b, fr in pair_frac_book.items():
        ts = float(sleeve_targets.get(b, 0.0))
        for (e, u), fv in fr.items():
            fv = float(fv)
            if fv <= 0.0:
                continue
            wb = ts * fv
            if wb <= 1e-15:
                continue
            d = diag_lookup.get((str(e), str(u)), {})
            pw_prev = float(prev_pw_book.get((str(e), str(u)), 0.0))
            pr_rows.append({
                "bucket": str(b), "etf": str(e), "underlying": str(u),
                "w_book": float(wb), "frac_sleeve": float(fv),
                "n_obs_decay": d.get("n_obs_decay", 0.0),
                "n_obs_beta": d.get("n_obs_beta", 0.0),
                "mu_used": d.get("mu_used", 0.0),
                "sigma_eff": d.get("sigma_eff", 0.0),
                "w_pre": d.get("w_pre", 0.0),
                "q_prior": d.get("q_prior", 0.0),
                "prev_w_book": pw_prev,
                "delta_w_book": float(wb) - pw_prev,
                "haircut_factor": d.get("haircut_factor", 1.0),
                "ema_applied": d.get("ema_applied", False),
            })
    df_pairs = pd.DataFrame(pr_rows).sort_values("w_book", ascending=False)
    df_pairs.to_csv(run_dir / "pair_book_weights.csv", index=False)
    try:
        df_pairs.to_parquet(run_dir / "pair_book_weights.parquet", index=False)
    except Exception:
        pass

    diag_rows: list[dict] = []
    for (e, u), d in diag_lookup.items():
        diag_rows.append({"etf": str(e), "underlying": str(u), **d,
                          "prev_w_book": float(prev_pw_book.get((str(e), str(u)), 0.0))})
    if diag_rows:
        pd.DataFrame(diag_rows).to_csv(run_dir / "pair_diag.csv", index=False)

    df_etf = pd.DataFrame([{"etf": k, "weight": float(v)} for k, v in sorted(pw_j.items(), key=lambda kv: -kv[1])])
    df_etf.to_csv(run_dir / "etf_weights.csv", index=False)
    df_pf = pd.DataFrame([
        {"etf": a, "underlying": b, "pair_frac_merged": float(v)}
        for (a, b), v in sorted(pf_j.items(), key=lambda kv: -kv[1])
    ])
    df_pf.to_csv(run_dir / "pair_frac_merged.csv", index=False)

    meta_out: dict = {}
    for k, v in dict(meta).items():
        if k in ("rc_top",):
            meta_out[k] = v
        elif isinstance(v, dict):
            meta_out[k] = {str(kk): float(vv) if hasattr(vv, "real") and not isinstance(vv, bool) else vv for kk, vv in v.items()}
        elif hasattr(v, "item"):
            try:
                meta_out[k] = float(v.item())
            except Exception:
                meta_out[k] = str(v)
        else:
            meta_out[k] = v
    meta_out["config_name"] = str(name)
    (run_dir / "joint_meta.json").write_text(json.dumps(meta_out, indent=2, default=str), encoding="utf-8")
    return run_dir


# Bundle cache keyed by (min_decay_obs, min_beta_obs) so different gates don't
# silently share the gated bundle.
_BUNDLE_CACHE: dict[tuple[int, int], dict] = {}


def _bundle_for(min_decay_obs: int, min_beta_obs: int) -> dict:
    key = (int(min_decay_obs), int(min_beta_obs))
    if key not in _BUNDLE_CACHE:
        _BUNDLE_CACHE[key] = build_joint_bundle(
            **_filter_build_bundle_kw({
                "gtp_mirror_df": GTP_MIRROR_DF,
                "gtp_mirror_diag": GTP_MIRROR_DIAG,
                "universe": UNIVERSE,
                "prices": PRICES,
                "norm_sym": _gtp_norm,
                "min_decay_obs": int(min_decay_obs),
                "min_beta_obs": int(min_beta_obs),
            })
        )
    return _BUNDLE_CACHE[key]


# Lazy import of the EMA helper.
from dcq.sizing.sizing_v2 import kelly_pre_weights_from_net_edge  # noqa: E402

rows: list[dict] = []
_weight_paths: list[str] = []
_prev_pw_book: dict[tuple[str, str], float] = {}
_prev_signal_frame: pd.DataFrame | None = None

for _i, cfg in enumerate(JOINT_CONFIGS, start=1):
    name = str(cfg["name"])
    h = _config_hash(cfg)
    if h in JOINT_RUN_CACHE:
        # Skip if exact same kwargs already produced a result row this session.
        cached = JOINT_RUN_CACHE[h]
        rows.extend(cached.get("rows", []))
        if cached.get("weights_dir"):
            _weight_paths.append(cached["weights_dir"])
        print(f"[joint_grid] {_i}/{len(JOINT_CONFIGS)} {name} (cached hash={h}) - skipped re-run")
        continue

    _bndl = _bundle_for(int(cfg["min_decay_obs"]), int(cfg["min_beta_obs"]))
    _st = _bndl["sleeve_targets"]

    ema_hl = cfg.get("ema_halflife_weeks")
    _mu_ov, _s_ov, _cur_sig = ema_smooth_signal(
        bucket_mirror_dfs=_bndl["bucket_mirror_dfs"],
        kelly_pre_weights_from_net_edge=kelly_pre_weights_from_net_edge,
        norm_sym=_gtp_norm,
        prev_signal=_prev_signal_frame,
        halflife_weeks=ema_hl,
        confidence_haircut=bool(cfg["confidence_haircut"]),
        conf_floor=float(SIZING_V2_CONF_FLOOR),
        n_obs_full=int(SIZING_V2_N_OBS_FULL),
    )

    t_lam = float(cfg["turnover_lambda"])
    _prev_for_run = (
        dict(_prev_pw_book) if (JOINT_GRID_USE_PREV_AS_ANCHOR and t_lam > 0 and _prev_pw_book) else None
    )

    _pair_frac_book, _pw_by_b, _diag_by_b, _meta, pw_j, pf_j = run_joint_qcqp_single(
        _bndl,
        pair_weights_qcqp_joint,
        norm_sym=_gtp_norm,
        book_max_pair_weight=float(cfg["book_max_pair"]),
        book_max_underlying_weight=float(cfg["book_max_underlying"]),
        book_sigma_target_annual=float(cfg["book_sigma_target"]),
        weight_ridge_lambda=float(cfg["weight_ridge_lambda"]),
        w_min_floor_frac=float(cfg["w_min_floor_frac"]),
        mu_shrink_intensity=float(JOINT_GRID_MU_SHRINK),
        sleeve_cap_max_pair_frac=float(JOINT_GRID_SLEEVE_CAP_MAX_PAIR_FRAC),
        sleeve_cap_max_underlying_frac=float(JOINT_GRID_SLEEVE_CAP_MAX_UNDER_FRAC),
        confidence_haircut=bool(cfg["confidence_haircut"]),
        conf_floor=float(SIZING_V2_CONF_FLOOR),
        n_obs_full=int(SIZING_V2_N_OBS_FULL),
        turnover_lambda=t_lam if _prev_for_run else 0.0,
        turnover_l1_max=SIZING_V2_TURNOVER_L1_MAX,
        prev_pair_weights=_prev_for_run,
        mu_used_override=_mu_ov if _mu_ov else None,
        sigma_eff_override=_s_ov if _s_ov else None,
        entropy_lambda=float(cfg.get("entropy_lambda") or 0.0),
        entropy_reference=str(cfg.get("entropy_reference") or "prior_w_pre"),
        edge_temperature=float(cfg.get("edge_temperature") or 1.0),
        mv_lambda=float(cfg.get("mv_lambda") or 0.0),
        eff_n_min_pairs=cfg.get("eff_n_min_pairs"),
    )

    _w_arr = pd.Series(pf_j, dtype=float)
    if not _w_arr.empty:
        _w_norm = _w_arr.clip(lower=0.0)
        _hhi = float((_w_norm ** 2).sum())
        _eff_n = float(1.0 / _hhi) if _hhi > 0 else float("nan")
        _top1 = float(_w_norm.max())
        _top5 = float(_w_norm.sort_values(ascending=False).head(5).sum())
    else:
        _hhi = float("nan"); _eff_n = float("nan"); _top1 = 0.0; _top5 = 0.0
    if _prev_pw_book:
        _all_keys = set(_prev_pw_book) | set(pf_j)
        _l1_turn = float(sum(abs(float(pf_j.get(k, 0.0)) - float(_prev_pw_book.get(k, 0.0))) for k in _all_keys))
        _mean_abs_dw = _l1_turn / max(len(_all_keys), 1)
    else:
        _l1_turn = float("nan"); _mean_abs_dw = float("nan")

    _run_dir_str = ""
    _run_dir_path: Path | None = None
    if JOINT_GRID_SAVE_WEIGHTS:
        _rd = _save_joint_grid_run_artifacts(
            _i, name=name, pair_frac_book=_pair_frac_book, sleeve_targets=_st,
            pw_j=pw_j, pf_j=pf_j, meta=_meta, cfg=cfg, root=JOINT_GRID_WEIGHTS_ROOT,
            diag_by_bucket=_diag_by_b, prev_pw_book=_prev_pw_book if _prev_pw_book else None,
        )
        _run_dir_str = str(_rd.resolve())
        _run_dir_path = _rd
        _weight_paths.append(_run_dir_str)

    globals()["PAIR_WEIGHTS"] = dict(pw_j)
    globals()["PAIR_FRAC_BY_KEY"] = dict(pf_j)
    globals()["PAIR_WEIGHTS_PRE_COV"] = dict(pw_j)

    _buf = io.StringIO()
    with contextlib.redirect_stdout(_buf):
        exec(ENGINE_SRC, globals())

    cfg_rows: list[dict] = []
    for lev in LEVERAGE_RUNS:
        bt = ALL_BT[lev]
        nav = bt["nav"]
        p = perf(nav)
        row = {
            "run_id": _i,
            "name": name,
            "config_hash": h,
            **{k: v for k, v in cfg.items() if k != "name"},
            "lev": lev,
            "qcqp_status": str(_meta.get("status")),
            "qcqp_primary_status": str(_meta.get("primary_status", _meta.get("status"))),
            "joint_fallback_kind": str(_meta.get("joint_fallback_kind") or ""),
            "qcqp_fallback": bool(_meta.get("fallback_used")),
            "sigma_p_book_annual": float(_meta.get("sigma_p_book_annual") or 0.0),
            "hhi_pair": float(_meta.get("hhi_pair", _hhi)),
            "eff_n_pair": float(_meta.get("eff_n_pair", _eff_n)),
            "top1_pair_share": float(_meta.get("top1_pair_share", _top1)),
            "top5_pair_share": float(_meta.get("top5_pair_share", _top5)),
            "gini_pair": float(_meta.get("gini_pair", float("nan"))),
            "n_pairs_active": int(_meta.get("n_pairs_active", 0)),
            "l1_turnover_vs_prev": _l1_turn,
            "mean_abs_delta_w": _mean_abs_dw,
            "n_pairs_dropped_by_history": int(_bndl.get("n_pairs_dropped_by_history", 0)),
            "anchor_used": bool(_prev_for_run),
            "ema_override_applied": bool(_mu_ov or _s_ov),
            "CAGR": float(p["CAGR"]),
            "Vol": float(p["Vol"]),
            "Sharpe": float(p["Sharpe"]),
            "Max_DD": float(p["Max DD"]),
            "Sortino": float(p["Sortino"]),
            "Calmar": float(p["Calmar"]),
        }
        if _run_dir_str:
            row["weights_dir"] = _run_dir_str
        rows.append(row)
        cfg_rows.append(row)

    # Write positions.md per run (one canonical recommendation per config).
    if write_positions_md is not None and _run_dir_path is not None:
        try:
            _csv = _run_dir_path / "pair_book_weights.csv"
            _df_pairs = pd.read_csv(_csv) if _csv.is_file() else pd.DataFrame()
            _perf_for_md = cfg_rows[0] if cfg_rows else None
            write_positions_md(
                _run_dir_path,
                config_name=name,
                joint_meta=_meta,
                pair_book_weights=_df_pairs,
                perf=_perf_for_md,
                top_n=30,
            )
        except Exception as _e:
            print(f"[joint_grid] positions.md skipped for {name}: {_e}")

    JOINT_RUN_CACHE[h] = {"rows": cfg_rows, "weights_dir": _run_dir_str, "name": name}

    print(
        f"[joint_grid] {_i}/{len(JOINT_CONFIGS)} {name} | status={_meta.get('status')} "
        f"primary={_meta.get('primary_status', _meta.get('status'))} fb={_meta.get('fallback_used')} "
        f"kind={_meta.get('joint_fallback_kind')!s} | "
        f"top1={_top1:.2%} top5={_top5:.2%} eff_n={_eff_n:.1f} | "
        f"CAGR={p['CAGR']:.2%} Vol={p['Vol']:.2%} Sharpe={p['Sharpe']:.2f} MaxDD={p['Max DD']:.1%}"
        + (f" | -> {_run_dir_str}" if _run_dir_str else "")
    )

    _prev_pw_book = dict(pf_j)
    _prev_signal_frame = _cur_sig if not _cur_sig.empty else _prev_signal_frame

globals()["PAIR_WEIGHTS"] = _pw_save
globals()["PAIR_FRAC_BY_KEY"] = _pf_save

JOINT_GRID_RESULTS = pd.DataFrame(rows)
JOINT_GRID_RESULTS = JOINT_GRID_RESULTS.sort_values(["Sharpe", "CAGR"], ascending=[False, False])

_out = Path("data") / "backtest"
_out.mkdir(parents=True, exist_ok=True)
_csv = _out / "joint_qcqp_grid_last.csv"
JOINT_GRID_RESULTS.to_csv(_csv, index=False)
print(f"\n[joint_grid] wrote {_csv}  rows={len(JOINT_GRID_RESULTS)}")
if JOINT_GRID_SAVE_WEIGHTS and _weight_paths:
    print(f"[joint_grid] per-run weights under: {JOINT_GRID_WEIGHTS_ROOT.resolve()}")
    print(f"[joint_grid] example: {_weight_paths[0]}")

try:
    from IPython.display import display
    _show_cols = [
        "name", "Sharpe", "CAGR", "Vol", "Max_DD", "eff_n_pair",
        "top1_pair_share", "top5_pair_share", "qcqp_primary_status",
        "qcqp_fallback", "weights_dir",
    ]
    _show_cols = [c for c in _show_cols if c in JOINT_GRID_RESULTS.columns]
    display(JOINT_GRID_RESULTS[_show_cols].head(20))
except Exception:
    print(JOINT_GRID_RESULTS.head(20).to_string())
