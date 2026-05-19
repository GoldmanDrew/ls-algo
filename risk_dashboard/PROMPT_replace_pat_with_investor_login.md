# Prompt: Replace GitHub PAT login with username/password (etf-dashboard pattern)

## Goal

Update the **ls-algo risk dashboard** (`site/` + `risk_dashboard/`) so operators sign in with a **login id + password**, matching the **etf-dashboard** static-site auth model. Remove the requirement for a GitHub Personal Access Token (PAT) and the runtime fetch of `risk_dashboard/data/latest.json` via the GitHub Contents API.

## Reference implementation (copy behavior from here)

Study and mirror these files in the sibling **etf-dashboard** repo:

| Area | Path |
|------|------|
| Login UI + gate | `etf-dashboard/index.html` — `LoginScreen`, `DashboardRoot`, `verifyInvestorPassword`, `readAuthSession` / `writeAuthSession` |
| Password hashing CLI | `etf-dashboard/scripts/hash_investor_password.py` |
| Example credentials file | `etf-dashboard/data/investors.example.json` |
| Live credentials (hashes only) | `etf-dashboard/data/investors.json` |

Auth mechanics to replicate exactly:

- `data/investors.json` schema: `{ "version": 1, "users": [{ "id", "name", "salt_b64", "hash_b64", "iterations" }] }`
- Browser verifies passwords with **PBKDF2-HMAC-SHA256** via `crypto.subtle` (250k iterations default, 32-byte derived key)
- **Timing-safe** compare of derived bytes vs `hash_b64`
- Session: `sessionStorage` JSON `{ uid, exp }`, **7-day** TTL (`AUTH_SESSION_MS = 7 * 86400000`)
- If `investors.json` is missing or has **zero valid users** → dashboard is **open** (no login gate), same as etf-dashboard
- If users exist → show login until session valid; **Sign out** clears session and re-locks UI

Do **not** commit plaintext passwords. Only commit salted hashes (use the hash script locally).

## Current ls-algo behavior (replace this)

| File | Today |
|------|--------|
| `site/assets/js/auth.js` | Validates PAT via `GET /repos/{owner}/{repo}`; fetches snapshot via Contents API |
| `site/assets/js/app.js` | `loadSnapshot(pat)` → `LSAuth.fetchRepoFile(pat, snapshotPath)` |
| `site/assets/js/config.js` | `repoOwner`, `repoName`, `snapshotPath: "risk_dashboard/data/latest.json"` |
| `site/index.html` | PAT paste form + setup instructions |
| `risk_dashboard/README.md` | Documents PAT-only access |
| `.github/workflows/risk_dashboard.yml` | Deploys `site/` only; snapshot stays in private repo |

## Target architecture

```
EOD / risk_dashboard.yml build job
  → writes risk_dashboard/data/latest.json (unchanged)
  → deploy job copies snapshot into site bundle:
        site/data/latest.json      (from risk_dashboard/data/latest.json)
        site/data/index.json       (optional: date picker manifest)
        site/data/investors.json     (PBKDF2 hashes; committed or CI-injected)

Browser (GitHub Pages, public static host)
  → fetch site/data/investors.json
  → if users.length > 0: Login id + Password gate (client-side PBKDF2)
  → after login: fetch site/data/latest.json (same-origin, no GitHub API)
```

**Important security note (document in README):** Like etf-dashboard, this is **UI gating on a static site**. Anyone who knows the public URL `.../data/latest.json` can download the snapshot without logging in. That is acceptable for our threat model (same as etf-dashboard). Do not claim server-side enforcement.

If stronger protection is needed later, that would require a backend proxy (out of scope).

## Implementation tasks

### 1. Auth module (`site/assets/js/auth.js`)

- Remove PAT validation and GitHub Contents API helpers.
- Port from etf-dashboard:
  - `timingSafeEqualBytes`
  - `verifyInvestorPassword(userId, password, users)`
  - `readAuthSession(validUserIds)` / `writeAuthSession(uid)` / `clearAuthSession()`
- Constants:
  - `INVESTORS_URL = "./data/investors.json"` (or relative path appropriate for Pages base URL)
  - `AUTH_STORAGE_KEY = "ls_risk_dash_session_v1"` (do not reuse etf-dashboard key)
  - `AUTH_SESSION_MS = 7 * 86400000`
- Export a small `window.LSAuth` API, e.g.:
  - `loadInvestors()` → `{ users, authEnabled }`
  - `verifyLogin(userId, password, users)` → boolean
  - session get/set/clear

### 2. App bootstrap (`site/assets/js/app.js`)

- Replace `tryAutoLogin` / PAT form handler with etf-dashboard-style gate:
  - On load: fetch `investors.json`
  - If `authEnabled`: show login panel until session valid
  - If not: load dashboard immediately
- `loadSnapshot()` should `fetch("./data/latest.json", { cache: "no-store" })` and `JSON.parse` — **no PAT parameter**.
- Optional: support `?t=` cache bust on snapshot fetch after deploy.
- Wire **Sign out** to clear session and return to login when auth enabled.
- Show logged-in user id in topbar (match etf-dashboard `topbar-user` pattern).

### 3. Login UI (`site/index.html` + `site/assets/css/main.css`)

- Replace PAT instructions with:
  - **Login id** (text input, `autocomplete="username"`)
  - **Password** (`type="password"`, `autocomplete="current-password"`)
  - Submit → “Sign in” / busy state / error message
- Remove links to GitHub PAT settings.
- Keep existing dashboard markup unchanged after login.

### 4. Config (`site/assets/js/config.js`)

- Remove or deprecate `repoOwner`, `repoName`, `patHelpUrl`, `defaultBranch` if unused.
- Set `snapshotUrl: "./data/latest.json"` and `manifestUrl: "./data/index.json"` if date picker needs manifest.

### 5. Deploy pipeline (`.github/workflows/risk_dashboard.yml`)

In the **deploy** job (after checkout, before `upload-pages-artifact`):

```bash
mkdir -p site/data
cp risk_dashboard/data/latest.json site/data/latest.json
cp risk_dashboard/data/index.json site/data/index.json
# investors.json: copy if present in repo, else skip (open dashboard)
test -f site/data/investors.json || cp site/data/investors.example.json site/data/investors.json || true
```

Ensure `site/data/investors.example.json` is committed; real `site/data/investors.json` may be committed (hashes only) or added manually on first deploy.

### 6. Password tooling

- Add `scripts/hash_investor_password.py` — copy from etf-dashboard (same CLI: `--id`, `--name`, `--password` / `INVESTOR_PASSWORD`, `--merge site/data/investors.json`).
- Add `site/data/investors.example.json` — same schema as etf-dashboard example.

### 7. Documentation

Update `risk_dashboard/README.md`:

- Remove PAT setup section.
- Add “Investor login” section mirroring etf-dashboard:
  - How to create users with `hash_investor_password.py`
  - Open vs locked behavior when `investors.json` is empty
  - Static-site security caveat (JSON URL is public)
- Update architecture diagram (snapshot bundled in Pages deploy, not Contents API).

Update root `README.md` risk dashboard bullet if it mentions PAT.

### 8. Cleanup

- Delete dead code paths referencing `github_pat`, `validatePat`, `fetchRepoFile`.
- Remove `repo-label` element or repurpose for “Risk dashboard” only.

## Acceptance criteria

- [ ] Login screen asks for **login id + password**, not PAT.
- [ ] Valid credentials unlock dashboard; invalid shows generic error (“Invalid login id or password”).
- [ ] Session persists across refresh for 7 days in same tab/storage; logout clears it.
- [ ] With **no** `site/data/investors.json` users (or missing file), dashboard loads without login.
- [ ] Snapshot loads from `./data/latest.json` with no GitHub API calls in Network tab after login.
- [ ] `risk_dashboard.yml` deploy copies fresh `latest.json` into `site/data/` every build.
- [ ] `python scripts/hash_investor_password.py --id … --merge site/data/investors.json` works.
- [ ] `risk_dashboard/tests/` still pass; add a small doc test or comment that frontend auth is manual QA.

## Manual QA checklist

1. Deploy to Pages (or `python -m http.server` from `site/` with `data/` populated).
2. With empty/missing investors → dashboard visible without login.
3. Add one user via hash script → login required; wrong password fails; correct password works.
4. Refresh page → still logged in; Sign out → login again.
5. Confirm `latest.json` is not requested from `api.github.com`.

## Out of scope

- GitHub OAuth Apps or server-side session store
- Encrypting `latest.json` at rest on Pages
- Changing how `build_site.py` computes metrics
- etf-dashboard backend (`backend/main.py`) Basic auth — static front-end only

## Suggested commit message

```
Replace risk dashboard PAT login with investor username/password.

Bundle snapshot JSON into the Pages deploy and verify credentials
client-side with PBKDF2 (same pattern as etf-dashboard).
```
