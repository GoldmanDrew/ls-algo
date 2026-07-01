# ls-algo Risk Dashboard

Static site + GitHub Actions pipeline that turns the daily IBKR Flex
output (already pulled by `ibkr_flex.py` + parsed by
`ibkr_accounting.py`) into a PM-grade risk dashboard.

The dashboard is a single-page app deployed to **GitHub Pages by
GitHub Actions**. Sign-in uses a **login id + password** (same static-site
pattern as **etf-dashboard**): PBKDF2 hashes in `site/data/investors.json`,
verified in the browser. The EOD snapshot is **bundled into the Pages
deploy** as `site/data/latest.json` (no GitHub API at runtime).

```
EOD pipeline → data/runs/<date>/accounting/
       ↓ workflow_run
risk_dashboard.yml (build) → risk_dashboard/data/latest.json → commit
       ↓
risk_dashboard.yml (deploy) → copy latest.json + investors.json into site/data/
       ↓
GitHub Pages → SPA + login gate → fetch ./data/latest.json
```

---

## What's in here

```
risk_dashboard/
??? __init__.py
??? README.md             ? this file
??? flex_parser.py        ? parses Flex XML (positions + borrow fees)
??? metrics.py            ? computes the --4 limits table from
?                            Risk_Dashboard_Plan.md
??? build_site.py         ? entrypoint: writes the JSON snapshots
??? tests/test_metrics.py ? smoke tests
??? data/                 ? snapshot output (latest.json + dated)
                            committed by CI on each EOD run
```

```
site/
??? index.html
??? 404.html
??? .nojekyll              (so Pages serves files starting with _)
??? assets/
    ??? css/main.css       (matches Bucket4_Plan_Summary.html style)
    ??? js/
        ??? config.js      (snapshot / investors URLs)
        ??? auth.js        (PBKDF2 investor login + session)
        ??? app.js         (SPA renderer)
??? data/
    ??? latest.json        (copied from risk_dashboard/data on deploy)
    ??? investors.json     (optional; hashed credentials)
    ??? investors.example.json
```

```
.github/workflows/
??? risk_dashboard.yml     (build ? commit ? deploy Pages)
```

---

## One-time setup (under 5 minutes)

### 1. Enable GitHub Pages on the repo

* Go to **Settings ? Pages**.
* **Source**: *GitHub Actions* (not "Deploy from a branch").
* Save.

The workflow's `deploy` job uses `actions/deploy-pages@v4`, which
publishes whatever `site/` contains.

### 2. Confirm the existing Flex secrets

The dashboard does **not** need new secrets. It reuses what
`eod_pnl_email.yml` already requires:

| Secret | Used for |
|---|---|
| `IBKR_FLEX_TOKEN`           | Flex Web Service token |
| `IBKR_FLEX_Q_TRADES`        | trades query id |
| `IBKR_FLEX_Q_CASH`          | cash transactions query id |
| `IBKR_FLEX_Q_POSITIONS`     | positions query id |
| `IBKR_FLEX_Q_BORROW_DETAILS`| borrow fee details query id |

Optional repo **variable** (Settings ? Variables ? Actions):

| Variable | Default | Effect |
|---|---|---|
| `MAGIS_NAV_USD` | *(unset)* | denominator for %-of-NAV metrics. |

**NAV denominator precedence** (highest first): `--nav-usd` flag → env
`MAGIS_NAV_USD` → `strategy.capital_usd` in `config/strategy_config.yml`
(currently **$1,050,000**) → hard default `$800,000`. The resolved value is only
a *fallback*: `build_site` still prefers a real NAV from `totals.json` /
Flex equity when present. The snapshot records where NAV came from in
`nav_source` (e.g. `config:capital_usd`), surfaced in the cockpit.

### Dashboard panels (Phase 0-4)

* **Freshness badge** — header pill comparing the snapshot's `run_date` to the
  latest accounting run + its age in days (`fresh` / `Nd old` / `stale snapshot`).
* **Bucket 5 (volatility ETP)** — a first-class sleeve everywhere B1/B2/B4 are:
  sleeve table, bucket tabs, factor-by-bucket, data-quality scan. It is shown
  but **not** added to the exposure reconciliation set (which mirrors
  `ibkr_accounting`: B1+B2+B4 + unbucketed net).
* **P&L naming** — `pnl_ytd_*` (strategy cumulative) is explicit; `pnl_daily_*`
  is the true day-over-day move vs the prior snapshot. Cockpit shows both.
* **Performance & drawdown** — daily/YTD P&L, current & max drawdown of the
  NAV+cumulative-PnL equity curve, plus top cumulative winners/losers.
* **Borrow shock sensitivity** — annualized carry on held short ETFs under
  ×1.25/×1.5/×2 and +10pp/+25pp APR shocks.
* **Shared underlyings** — names whose exposure spans more than one bucket
  (shared broker spot line that nets across sleeves).
* **EOD sleeve groups** — B1+B2+B4+B5 vs B3 overlay, matching the PnL email,
  plus book margin utilization (Σ margin req / NAV).

### 3. Trigger the workflow once

* **Manual:** Actions ? "Risk Dashboard" ? *Run workflow* ? leave
  inputs blank ? Run.
* This builds `risk_dashboard/data/latest.json` from the most recent
  `data/runs/<date>/`, commits it, and deploys the static site.

### 4. First sign-in

The dashboard URL is shown at the bottom of the deploy job:

```
https://goldmandrew.github.io/ls-algo/
```

Open it. It will ask for a PAT.

#### Generate a fine-grained PAT

* Go to <https://github.com/settings/personal-access-tokens>.
* **Resource owner:** your account.
* **Repository access:** *Only select repositories* ? `ls-algo`.
* **Permissions ? Repository:**
  * `Contents` ? **Read-only**.
  * `Metadata` ? Read-only (always required).
* **Expiration:** 7-30 days is fine.
* Generate, copy.

Paste into the dashboard's login screen. Done.

The PAT is stored in `sessionStorage` only and is wiped when you
close the tab or hit *Sign out*. There is no third-party storage.

---

## How it runs

| Trigger | What happens |
|---|---|
| `workflow_run` after `eod_pnl_email.yml` succeeds | full build + deploy |
| `workflow_dispatch` (manual)                      | full build + deploy (or deploy-only if `skip_build=true`) |
| `push` to `main` touching `site/**` or `risk_dashboard/**` | redeploy site shell only (no rebuild) |

The build step:

```bash
python -m pytest risk_dashboard/tests -q
python -m risk_dashboard.build_site \
    --run-date "$RUN_DATE" \
    --runs-root data/runs \
    --nav-usd "$MAGIS_NAV_USD" \
    --out-dir risk_dashboard/data
git add risk_dashboard/data && git commit ... && git push
```

The deploy step uploads `site/` as a Pages artifact and deploys it.

---

## What the dashboard shows

Each panel maps to a section of the Risk Dashboard Plan
(`Risk_Dashboard_Plan.md`):

| Panel | Sourced from | Plan --|
|---|---|---|
| Book summary strip (NAV, gross, net, P&L) | `data/runs/<date>/accounting/totals.json` | 3.1, 4 |
| Sleeve allocation table | `totals.json` | 3.1, 5.2 |
| Bucket detail tabs (winners / losers / exposures) | `pnl_<bucket>.csv`, `net_exposure_<bucket>.csv` | 3.2, 3.3 |
| Borrow & microstructure | `flex_borrow_fee_details.xml`, `flex_positions.xml` | 3.6, 5.5 |
| Raw totals | `totals.json` | 1.0 |

Future panels (Vol-ETP, Options overlay, NAV dislocation) plug into
`metrics.py` the same way -- add a new compute function, surface its
output on `RiskSnapshot`, and add a panel in `app.js`.

---

## Factor map provenance (sectors + betas)

The factor exposure panel attributes every underlying to a sector and
weights its net / gross by a β-to-SPY (with secondary β to QQQ and
IWM). Both fields are computed live each build with explicit
provenance tags so the UI can show how confident each cell is.

### Sector attribution (tiered)

`risk_dashboard.sector_loader.resolve_sector` walks five tiers and
returns the first hit:

| Tier | Source | Notes |
|------|--------|-------|
| 1. `override` | `OVERRIDE_SECTOR_MAP` in `factor_map.py` | Hand-curated thematic buckets GICS cannot express (quantum, crypto-equity, evtol, drones, space, insurtech). Edit this map when a new theme appears. |
| 2. `screener` | Per-underlying `theme` / `sector` columns in `data/etf_screened_today.csv` | Reuses the screener's existing classification when present. |
| 3. `vendor`   | yfinance `Ticker.info["industry"]` then `["sector"]` mapped through `VENDOR_SECTOR_MAP` | Industry-level keys win over sector-level (more specific). |
| 4. `heuristic`| Regex set in `HEURISTIC_PATTERNS` over `longName` / `industry` / `longBusinessSummary` | Catches new thematics not yet curated. Ordered so multi-word patterns (e.g. *bitcoin miner*) match before single tokens (*bitcoin*). |
| 5. `default`  | `"other"` | Last resort; flagged in UI. |

Each row in the JSON also carries `sector_source` and
`sector_confidence` (1.00 / 0.90 / 0.75 / 0.55 / 0.10 by tier).

### Live betas (two-pass shrinkage)

`risk_dashboard.beta_loader.compute_betas` runs every build:

1. **Fetch closes** with a fail-over chain:
   yfinance (cached in `data/cache/beta_history/<SYM>.csv`) →
   Stooq CSV (`https://stooq.com/q/d/l/?s={sym}.us&i=d`) →
   stale on-disk cache (≤14 days old) → skip.
2. **Pass 1 — OLS** of 252 daily log returns vs SPY / QQQ / IWM. Each
   index regression returns `(β, β_se, n_obs, R²)`. Names with at
   least `MIN_OBS_FOR_TRUST` (60) paired observations are tagged
   `provenance = "computed"` and the raw OLS β is stored as
   `beta_to_spy_raw` for diagnostics.
3. **Build per-sector mean priors** (median of pass-1 `computed`
   betas, requires ≥ 5 reliable names per sector).
4. **Pass 2 — Bayesian shrinkage** toward the prior:

       k = K_BASE · max(1, prior²)               # K_BASE = 60
       n_eff = n · (1 − ρ_AR1) / (1 + ρ_AR1)     # matches daily_screener
       w = n_eff / (n_eff + k)
       β_final = w · β_OLS + (1 − w) · prior

   `prior_source` is `sector_mean` when the sector has ≥ 5 reliable
   computed betas; otherwise the curated `BETA_TO_SPY` value;
   otherwise `DEFAULT_SINGLE_NAME_BETA` (1.20) /
   `DEFAULT_BROAD_INDEX_BETA` (1.00). `shrinkage_applied` is True
   whenever the prior contributed more than 5% (i.e. `w < 0.95`).
5. **Persist** `data/cache/beta_summary.json` with every
   `BetaResult` + the sector means. CI commits the close cache and
   summary back to `main` after each successful build (`[skip ci]`),
   so the next run never starts cold.

### UI badges

Each row in the Factor exposure panel renders a small pill next to the
underlying ticker:

* `computed` (green) — pass-1 OLS, shrinkage didn't materially move
  the estimate.
* `shrunk` (amber)   — Bayesian blend with the sector / curated
  prior pulled the OLS by ≥ 5%.
* `fallback` (amber) — `curated_fallback` (no price data, fell to
  `BETA_TO_SPY` map).
* `fallback` (red)   — `default_fallback` (no price data and not in
  the curated map).

Hovering the pill shows `n_obs`, `R²`, `w`, the prior used and the
raw OLS beta.

---

## Local development

```bash
# 1. Build a snapshot from any date you have under data/runs/
python -m risk_dashboard.build_site \
    --run-date 2026-05-15 \
    --runs-root data/runs \
    --nav-usd 800000 \
    --out-dir risk_dashboard/data

mkdir -p site/data
cp risk_dashboard/data/latest.json site/data/latest.json
# optional: site/data/investors.json from hash_investor_password.py
cd site && python -m http.server 8765
# open http://localhost:8765/
```

---

## Adding a new metric

1. Write the compute function in `metrics.py` (a small `pd.DataFrame`
   in / dict out function with explicit threshold args).
2. Add it to `build_snapshot()` so it lands on the JSON.
3. Render it in `app.js`. Use the existing `tight` table class +
   `pill` status component to match the look of the Bucket 4 PDFs.
4. Add a unit test in `tests/test_metrics.py` (the suite uses
   `pytest`; CI runs it on every build).
5. Update the kill-switch table in `Risk_Dashboard_Plan.md` -- 5 if
   the metric has an associated alert. Every alert must trace to a
   pre-committed memo.

---

## FAQ

**Why username/password instead of GitHub PAT?** Matches etf-dashboard:
no token management for viewers, works on public Pages, same PBKDF2
client-side gate.

**Could the data leak via the deployed site?** Yes — `site/data/latest.json`
is part of the public Pages artifact. Login only hides the UI. Restrict
who receives the URL; use `investors.json` to keep casual visitors out.

**What about CSP / mixed content?** The site loads only same-origin
assets (no CDN scripts) and does not call `api.github.com` after login.

---

## Deletion / rollback

* **Disable the dashboard:** delete the workflow file, the Pages
  source toggle, and (optionally) the `risk_dashboard/data/`
  snapshots. The Python package can stay around -- nothing else
  depends on it.
* **Rotate access:** re-run `hash_investor_password.py` for the user and redeploy.
* **Remove a user:** delete their row from `site/data/investors.json` and redeploy.
