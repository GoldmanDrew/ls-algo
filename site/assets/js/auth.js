/**
 * Auth + GitHub API helpers for the ls-algo risk dashboard.
 *
 * The PAT is stored only in sessionStorage so it is wiped when the
 * tab closes. There is no fallback to localStorage on purpose - we
 * don't want a stray token to outlive the session.
 *
 * The "Are you allowed to see this dashboard?" check is just:
 *
 *   GET https://api.github.com/repos/{owner}/{repo}
 *
 * with the user's PAT. If GitHub returns 200, the user is a
 * collaborator on the private repo and we proceed. If 401/403/404,
 * the token is invalid or the user lacks access -> reject.
 *
 * The same PAT is then used by app.js to fetch the snapshot via the
 * GitHub Contents API.
 */

(function () {
  const SESSION_KEY = "ls_algo_pat";

  const cfg = window.LS_ALGO_CONFIG;
  const repoLabel = document.getElementById("repo-label");
  if (repoLabel) repoLabel.textContent = `${cfg.repoOwner}/${cfg.repoName}`;

  const ghApi = "https://api.github.com";

  /**
   * Validate a PAT by hitting GET /repos/{owner}/{repo}.
   * Returns the parsed JSON on success; throws on auth failure.
   */
  async function validatePat(pat) {
    const url = `${ghApi}/repos/${cfg.repoOwner}/${cfg.repoName}`;
    const res = await fetch(url, {
      headers: {
        Authorization: `Bearer ${pat}`,
        Accept: "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
      },
    });
    if (res.status === 200) {
      return await res.json();
    }
    if (res.status === 401) {
      throw new Error("Bad credentials. Token is invalid or expired.");
    }
    if (res.status === 403) {
      throw new Error(
        "Forbidden. The token does not have the required scope, " +
          "or the repository is restricted by SAML/SSO."
      );
    }
    if (res.status === 404) {
      throw new Error(
        "Not found. The token cannot see " +
          `${cfg.repoOwner}/${cfg.repoName} - check repo access on the PAT.`
      );
    }
    throw new Error(`Unexpected status ${res.status} from GitHub.`);
  }

  /**
   * Fetch a file from the repo via the Contents API.
   * Uses the raw media type to avoid base64 round-tripping.
   * @param {string} pat
   * @param {string} path - e.g. "risk_dashboard/data/latest.json"
   * @param {string} ref  - branch / tag / sha; defaults to defaultBranch.
   */
  async function fetchRepoFile(pat, path, ref) {
    const url =
      `${ghApi}/repos/${cfg.repoOwner}/${cfg.repoName}/contents/` +
      `${encodeURIComponent(path).replace(/%2F/g, "/")}` +
      `?ref=${encodeURIComponent(ref || cfg.defaultBranch)}`;
    const res = await fetch(url, {
      headers: {
        Authorization: `Bearer ${pat}`,
        Accept: "application/vnd.github.raw",
        "X-GitHub-Api-Version": "2022-11-28",
      },
    });
    if (!res.ok) {
      throw new Error(
        `Could not fetch ${path}: ${res.status} ${res.statusText}`
      );
    }
    const contentType = res.headers.get("content-type") || "";
    if (contentType.includes("application/json")) {
      return await res.json();
    }
    const text = await res.text();
    try {
      return JSON.parse(text);
    } catch (e) {
      return text;
    }
  }

  function getStoredPat() {
    return sessionStorage.getItem(SESSION_KEY);
  }
  function storePat(pat) {
    sessionStorage.setItem(SESSION_KEY, pat);
  }
  function clearPat() {
    sessionStorage.removeItem(SESSION_KEY);
  }

  window.LSAuth = {
    validatePat,
    fetchRepoFile,
    getStoredPat,
    storePat,
    clearPat,
  };
})();
