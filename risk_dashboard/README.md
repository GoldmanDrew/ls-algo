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
| `MAGIS_NAV_USD` | `800000` | denominator for %-of-NAV metrics. |

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
