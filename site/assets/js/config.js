/**
 * Static config for the ls-algo risk dashboard.
 *
 * Edit these constants if you fork the repo or rename it.
 */
window.LS_ALGO_CONFIG = {
  repoOwner: "GoldmanDrew",
  repoName: "ls-algo",
  defaultBranch: "main",

  /**
   * Path inside the repo to the JSON snapshot the workflow writes.
   * The static SPA fetches this file via the GitHub Contents API
   * using the user's PAT, so the file does not need to be deployed
   * with the site itself - it can stay in the private repo.
   */
  snapshotPath: "risk_dashboard/data/latest.json",
  manifestPath: "risk_dashboard/data/index.json",

  /**
   * Required scopes / permissions for the user's PAT. We do not
   * enforce this client-side; it is shown in the login panel as a
   * setup instruction.
   */
  patHelpUrl: "https://github.com/settings/personal-access-tokens",
};
