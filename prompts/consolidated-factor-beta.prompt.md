# Prompt: Consolidated multi-factor beta panel (SPY / QQQ / IWM / BTC)

Use this prompt to merge the **Factor exposure (beta to SPY / QQQ / IWM)** section and the **Factor exposure (beta to BTC.USD)** section into a single dashboard panel with headline betas per accounting bucket and for the entire book.

---

## Goal

Replace the two separate panels (`#factor-section` + `#btc-factor-section`) with **one consolidated factor panel** that answers:

> “What is my net beta to SPY, QQQ, IWM, and BTC — by sleeve (bucket 1–4) and for the whole book?”

The user should not have to cross-reference two sections to compare SPY vs BTC exposure.

---

## Current state (do not break)

| Piece | Location | What exists today |
|-------|----------|-------------------|
| Per-name betas | `risk_dashboard/beta_loader.py` → `compute_betas()` | 252d OLS log-return β to SPY, QQQ (`beta_to_ndx`), IWM (`beta_to_rut`), BTC-USD (`beta_to_btc`) |
| Book factor panel | `compute_factor_panel()` in `metrics.py` | Rows with `beta_to_spy`, `beta_to_qqq`, `beta_to_iwm`, `beta_to_btc`; totals `net_beta_to_spy`, `net_beta_to_btc` |
| Bucket SPY only | `compute_factor_by_bucket()` | `by_bucket[]` with `net_beta_to_spy` per `bucket_1`…`bucket_4` |
| UI — factor | `site/index.html` `#factor-section`, `renderFactor()` | Summary strip, by-sector, by-bucket (SPY only), top long/short SPY tables |
| UI — BTC | `site/index.html` `#btc-factor-section`, `renderBtcFactor()` | Summary strip, top long/short BTC tables |
| Beta source | OLS loader only | Sector uses `OVERRIDE_SECTOR_MAP`; betas must **not** use curated `BETA_TO_SPY` |

---

## Definitions

For each factor `F ∈ {SPY, QQQ, IWM, BTC}`:

```
beta_weighted_net_F_usd  = Σ (net_notional_usd_i × beta_to_F_i)
net_beta_to_F            = beta_weighted_net_F_usd / NAV     # “headline beta”, e.g. 0.35x NAV
```

**Book-level** headline betas: aggregate over `net_exposure_by_underlying.csv` (underlying rollup).

**Bucket-level** headline betas (additive): ratio-scale `bucket_exposure_detail.csv` into B1/B2/B4 + `net_exposure_unbucketed.csv`. B3/B5 are overlays (excluded from the additive sum). `by_bucket_reconciles` compares additive sleeve β-wtd net to book.

Mapping in JSON / loader:

| Display | Field on row | Loader field |
|---------|--------------|--------------|
| β SPY | `beta_to_spy` | `beta_to_spy` |
| β QQQ | `beta_to_qqq` | `beta_to_ndx` |
| β IWM | `beta_to_iwm` | `beta_to_rut` |
| β BTC | `beta_to_btc` | `beta_to_btc` |

BTC beta uses equity-calendar alignment (+1 day shift on Yahoo BTC-USD bars) — do not change that logic.

---

## Target UI (single panel)

### Section title
**Factor exposure (beta to SPY / QQQ / IWM / BTC)**

Remove or collapse `#btc-factor-section`; update subnav to one **Factor** link.

### A. Book headline strip (top)

Four stat cards (or one row in a table), sourced from `factor_panel.totals`:

| Stat | Field | Format |
|------|-------|--------|
| Net β SPY | `net_beta_to_spy` | `+0.28x NAV` |
| Net β QQQ | `net_beta_to_qqq` | `+0.24x NAV` |
| Net β IWM | `net_beta_to_iwm` | `+0.31x NAV` |
| Net β BTC | `net_beta_to_btc` | `+0.11x NAV` |

Optional secondary line per card: β-weighted net $ (e.g. `beta_weighted_net_usd`, `beta_weighted_net_qqq_usd`, …).

### B. Headline betas by accounting bucket (primary new table)

Replace the SPY-only `#factor-by-bucket` table with a **multi-factor bucket matrix**:

| Bucket | Net $ | Net β SPY | Net β QQQ | Net β IWM | Net β BTC |
|--------|------:|----------:|----------:|----------:|----------:|
| Bucket 1 | … | -4.34x | … | … | … |
| Bucket 2 | … | +4.13x | … | … | … |
| Bucket 3 (flow) | … | +0.14x | … | … | … |
| Bucket 4 | … | +0.55x | … | … | … |
| **Book total** | … | **+0.28x** | … | … | **+0.11x** |

- Data: `factor_panel.by_bucket[]` — **extend each row** with `net_beta_to_qqq`, `net_beta_to_iwm`, `net_beta_to_btc` and optional β-wtd net $ columns per factor.
- Footer **Book total** from `factor_panel.totals` (underlying rollup), not sum of bucket rows.
- Keep reconciliation warning when bucket sums ≠ book (extend to all four factors or show SPY-only warning with note).

### C. Top names (optional, below bucket matrix)

Single pair of tables (long / short) ranked by **|β-wtd net SPY|** (keep current ranking), columns:

| Underlying | Sector | β SPY | β QQQ | β IWM | β BTC | Net $ | β-wtd net $ (SPY) |

Remove duplicate BTC-only long/short tables from `renderBtcFactor()`.

### D. Keep unchanged (below consolidated block)
- By-sector table (economic sectors, SPY-weighted is fine for now)
- Sector / beta provenance pills

---

## Backend changes

### 1. Extend `compute_factor_by_bucket()` (`metrics.py`)

For each bucket CSV row, resolve all four betas from `beta_results` (same as `compute_factor_panel`):

```python
beta_net_spy  = net * beta_to_spy
beta_net_qqq  = net * beta_to_qqq   # if not None
beta_net_iwm  = net * beta_to_iwm
beta_net_btc  = net * beta_to_btc   # if not None
```

Per bucket output — add:

```python
{
  "net_beta_to_spy": total_beta_net_spy / nav,
  "net_beta_to_qqq": total_beta_net_qqq / nav,
  "net_beta_to_iwm": total_beta_net_iwm / nav,
  "net_beta_to_btc": total_beta_net_btc / nav,
  "beta_weighted_net_qqq_usd": ...,
  "beta_weighted_net_iwm_usd": ...,
  "beta_weighted_net_btc_usd": ...,
}
```

### 2. Extend `compute_factor_panel()` totals

Add to `factor_panel.totals`:

```python
"net_beta_to_qqq": total_beta_net_qqq / nav,
"net_beta_to_iwm": total_beta_net_iwm / nav,
"beta_weighted_net_qqq_usd": ...,
"beta_weighted_net_iwm_usd": ...,
# net_beta_to_btc already exists
```

Compute book totals by summing per-row `net × beta_to_*` across all underlyings (same pattern as SPY today).

### 3. Helper refactor (recommended)

Extract `_resolve_underlying_betas(underlying, beta_results) -> dict` returning all four betas + sources, used by both `compute_factor_panel` and `compute_factor_by_bucket` to avoid drift.

### 4. Snapshot / build

- `build_snapshot()` already attaches `by_bucket`; no new files required.
- Rebuild: `python -m risk_dashboard.build_site --run-date YYYY-MM-DD`
- Copy `risk_dashboard/data/latest.json` → `site/data/latest.json`

### 5. Tests

Add/update in `risk_dashboard/tests/test_metrics.py`:

- `test_compute_factor_by_bucket_multi_factor` — bucket with NVDA (β_spy=2, β_qqq=1.5, β_iwm=1.2, β_btc=0.5) at $10k net → check all four `net_beta_to_*`.
- `test_factor_panel_totals_include_qqq_iwm_btc` — book totals for all four factors.

---

## Frontend changes

### Files
- `site/index.html` — merge sections; one `#factor-section`; remove `#btc-factor-section` and subnav link
- `site/assets/js/app.js`:
  - Extend `renderFactor()` with multi-factor headline strip + bucket matrix
  - Delete `renderBtcFactor()` (or make it a no-op that delegates to `renderFactor`)
  - Remove `btcFactorSummary`, `btcFactorLong`, `btcFactorShort` element refs if unused

### Formatting
- Headline betas: `{value.toFixed(2)}x` with signed color class (`pos` / `neg`)
- Null beta → `-` (some names may lack BTC history)
- Bucket matrix: use `<table class="tight">`; bucket labels from `bucket_label`

---

## Out of scope (unless explicitly requested)

- Sector-level headline betas (economic `by_sector` — different from accounting buckets)
- Beta to factors beyond SPY/QQQ/IWM/BTC
- Changing OLS window, shrinkage, or curated beta map
- Changing accounting CSV semantics for B3/B4/B5 (dashboard attributes from detail)

---

## Acceptance criteria

1. **One panel** shows net β to SPY, QQQ, IWM, and BTC for the **whole book**.
2. **Same four headline betas** shown per **accounting bucket** in one table (additive sleeves + overlays).
3. Book total row uses underlying rollup; additive sleeves use detail ratio-split; overlays excluded from sum; reconciliation warning when additive ≠ book.
4. No regression: per-name OLS betas unchanged; sector classification unchanged.
5. Tests pass; snapshot rebuilt; site preview shows consolidated layout.

---

## Verify

```bash
python -m pytest risk_dashboard/tests/test_metrics.py -q -k "factor_by_bucket or factor_panel"
python -m risk_dashboard.build_site --run-date 2026-05-19
# Spot-check JSON:
python -c "import json; t=json.load(open('risk_dashboard/data/latest.json'))['factor_panel']['totals']; print({k:t[k] for k in t if 'beta' in k.lower()})"
python -c "import json; fp=json.load(open('risk_dashboard/data/latest.json'))['factor_panel']; print([(r['bucket_label'], r.get('net_beta_to_spy'), r.get('net_beta_to_btc')) for r in fp['by_bucket']])"
```

---

## Reference files

- `risk_dashboard/beta_loader.py` — `compute_betas()`, `BetaResult`
- `risk_dashboard/metrics.py` — `compute_factor_panel`, `compute_factor_by_bucket`, `build_snapshot`
- `site/index.html`, `site/assets/js/app.js` — `renderFactor`, `renderBtcFactor`
- Prior art: `prompts/factor-beta-by-bucket.prompt.md`
