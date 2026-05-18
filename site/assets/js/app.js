/**
 * SPA entry-point for the ls-algo risk dashboard.
 *
 * Flow:
 *   1) On load, look for a PAT in sessionStorage.
 *   2) If none, show login. On submit, validate the PAT.
 *   3) On success, fetch latest.json from the private repo via the
 *      GitHub Contents API and render every panel.
 */

(function () {
  const cfg = window.LS_ALGO_CONFIG;

  const els = {
    loginPanel: document.getElementById("login-panel"),
    dashboard: document.getElementById("dashboard"),
    loginForm: document.getElementById("login-form"),
    patInput: document.getElementById("pat-input"),
    loginError: document.getElementById("login-error"),
    logoutBtn: document.getElementById("logout-btn"),
    runDateLabel: document.getElementById("run-date-label"),
    generatedAtLabel: document.getElementById("generated-at-label"),
    cockpitStrip: document.getElementById("cockpit-strip"),
    dataQuality: document.getElementById("data-quality"),
    alertRows: document.getElementById("alert-rows"),
    scenarioContent: document.getElementById("scenario-content"),
    contributorContent: document.getElementById("contributor-content"),
    strip: document.getElementById("strip"),
    breaches: document.getElementById("breaches"),
    sleeveBody: document.querySelector("#sleeve-table tbody"),
    bucketTabs: document.getElementById("bucket-tabs"),
    bucketContent: document.getElementById("bucket-content"),
    borrowContent: document.getElementById("borrow-content"),
    rawTotals: document.getElementById("raw-totals"),
  };

  /* ----------------------- Formatters ------------------------- */
  const fmtUsd = (n) =>
    n == null
      ? "-"
      : (n < 0 ? "-$" : "$") +
        Math.abs(n).toLocaleString(undefined, {
          maximumFractionDigits: 0,
        });
  const fmtUsdSigned = (n) =>
    n == null
      ? "-"
      : (n >= 0 ? "+$" : "-$") +
        Math.abs(n).toLocaleString(undefined, {
          maximumFractionDigits: 0,
        });
  const fmtPct = (n, dec) =>
    n == null
      ? "-"
      : (n * 100).toLocaleString(undefined, {
          minimumFractionDigits: dec ?? 1,
          maximumFractionDigits: dec ?? 1,
        }) + "%";
  const fmtPp = (n, dec) =>
    n == null
      ? "-"
      : (n >= 0 ? "+" : "") +
        n.toLocaleString(undefined, {
          minimumFractionDigits: dec ?? 1,
          maximumFractionDigits: dec ?? 1,
        }) +
        " pp";
  const signedClass = (n) => (n == null ? "" : n >= 0 ? "pos" : "neg");
  const rowStatusClass = (status) =>
    status === "hard" ? "row-hard" : status === "warn" ? "row-warn" : "";
  const safeText = (s, fallback) =>
    s == null || String(s).trim() === "" ? fallback || "not available" : String(s);

  /* --------------------- Render helpers ----------------------- */
  function statusPill(status, label) {
    const text = label || status || "unknown";
    return `<span class="pill pill-${status || "unknown"}">${text}</span>`;
  }

  function renderStrip(book) {
    const items = [
      {
        label: "NAV",
        value: fmtUsd(book.nav_usd),
      },
      {
        label: "Gross",
        value: fmtUsd(book.gross_notional_usd),
        sub: fmtPct(book.gross_exposure_pct_nav, 0) + " of NAV",
      },
      {
        label: "Net",
        value: fmtUsd(book.net_notional_usd),
        sub: fmtPct(book.net_exposure_pct_nav, 0) + " of NAV",
        cls: signedClass(book.net_notional_usd),
      },
      {
        label: "Long",
        value: fmtUsd(book.long_notional_usd),
      },
      {
        label: "Short",
        value: fmtUsd(book.short_notional_usd),
      },
      {
        label: "P&L (run)",
        value: fmtUsdSigned(book.pnl_today_usd),
        sub: fmtPct(book.pnl_today_pct_nav, 2) + " of NAV",
        cls: signedClass(book.pnl_today_usd),
      },
    ];
    els.strip.innerHTML = items
      .map(
        (it) => `
        <div class="stat">
          <div class="label">${it.label}</div>
          <div class="value ${it.cls || ""}">${it.value}</div>
          ${it.sub ? `<div class="sub">${it.sub}</div>` : ""}
        </div>`
      )
      .join("");
  }

  function renderDataQuality(dq) {
    if (!dq) {
      els.dataQuality.innerHTML = statusPill("unknown", "data quality unknown");
      return;
    }
    const label =
      dq.status === "ok"
        ? "data quality ok"
        : `${dq.status}: ${dq.missing_source_count || 0} missing sources, ${
            dq.missing_required_column_count || 0
          } missing columns, ${dq.blank_render_field_count || 0} blanks, ${
            (dq.reconciliations || []).filter((r) => r.status === "hard").length
          } reconciliation breaks`;
    els.dataQuality.innerHTML = statusPill(dq.status, label);
  }

  function renderCockpit(snap) {
    const book = snap.book || {};
    const worst = snap.worst_shock || {};
    const top = worst.top_contributor || {};
    const alertRows = snap.alert_rows || [];
    const items = [
      { label: "NAV", value: fmtUsd(book.nav_usd) },
      {
        label: "Gross / NAV",
        value: fmtPct(book.gross_exposure_pct_nav, 0),
        sub: fmtUsd(book.gross_notional_usd),
      },
      {
        label: "Net / NAV",
        value: fmtPct(book.net_exposure_pct_nav, 1),
        sub: fmtUsd(book.net_notional_usd),
        cls: signedClass(book.net_notional_usd),
      },
      {
        label: "Worst shock",
        value: fmtUsdSigned(worst.pnl_usd),
        sub: `${safeText(worst.label, "no scenario")} / ${fmtPct(worst.pnl_pct_nav, 2)}`,
        cls: signedClass(worst.pnl_usd),
      },
      {
        label: "Top offender",
        value: safeText(top.underlying, "none"),
        sub: `${safeText(top.bucket, "-")} ${fmtUsdSigned(top.pnl_usd)}`,
        cls: signedClass(top.pnl_usd),
      },
      {
        label: "Alerts",
        value: String(alertRows.length),
        sub: `${alertRows.filter((r) => r.status === "hard").length} hard`,
        cls: alertRows.some((r) => r.status === "hard") ? "neg" : "",
      },
    ];
    els.cockpitStrip.innerHTML = items
      .map(
        (it) => `
        <div class="stat stat-${it.cls || "neutral"}">
          <div class="label">${it.label}</div>
          <div class="value ${it.cls || ""}">${it.value}</div>
          ${it.sub ? `<div class="sub">${it.sub}</div>` : ""}
        </div>`
      )
      .join("");
  }

  function renderAlerts(rows) {
    if (!rows || !rows.length) {
      els.alertRows.innerHTML = `<span class="pill pill-ok">No risk alerts</span>`;
      return;
    }
    els.alertRows.innerHTML = `
      <table class="tight alert-table"><thead><tr>
        <th>Status</th><th>Metric</th><th>Value</th><th>Limit</th><th>Action</th>
      </tr></thead><tbody>${rows
        .slice(0, 12)
        .map((r) => {
          const limit =
            r.limit && typeof r.limit === "object"
              ? `warn ${r.limit.warn}, hard ${r.limit.hard}`
              : "-";
          const value = typeof r.value === "number" ? r.value.toFixed(3) : safeText(r.value, "-");
          return `<tr class="${rowStatusClass(r.status)}">
            <td>${statusPill(r.status)}</td>
            <td><strong>${safeText(r.label || r.metric)}</strong><div class="dim small">${safeText(
              r.source,
              "source unavailable"
            )}</div></td>
            <td class="num">${value}</td>
            <td>${limit}</td>
            <td>${safeText(r.action, "review")}</td>
          </tr>`;
        })
        .join("")}</tbody></table>`;
  }

  function renderScenarios(panel) {
    const scenarios = panel?.scenarios || [];
    if (!scenarios.length) {
      els.scenarioContent.innerHTML = `<p class="dim">No scenario data available.</p>`;
      els.contributorContent.innerHTML = `<p class="dim">No contributor data available.</p>`;
      return;
    }
    els.scenarioContent.innerHTML = `
      <table class="tight scenario-table"><thead><tr>
        <th>Scenario</th><th>Total P&amp;L</th><th>% NAV</th>
        <th>Book/Other</th><th>B1</th><th>B2</th><th>B3</th><th>B4</th><th>Top offender</th><th>Status</th>
      </tr></thead><tbody>${scenarios
        .map((s) => {
          const b = s.bucket_pnl || {};
          const top = s.top_contributor || {};
          return `<tr class="${rowStatusClass(s.status)}">
            <td><strong>${safeText(s.label)}</strong><div class="dim small">${safeText(
              s.description,
              ""
            )}</div></td>
            <td class="num ${signedClass(s.pnl_usd)}">${fmtUsdSigned(s.pnl_usd)}</td>
            <td class="num ${signedClass(s.pnl_pct_nav)}">${fmtPct(s.pnl_pct_nav, 2)}</td>
            <td class="num ${signedClass(b.book)}">${fmtUsdSigned(b.book || 0)}</td>
            <td class="num ${signedClass(b.bucket_1)}">${fmtUsdSigned(b.bucket_1 || 0)}</td>
            <td class="num ${signedClass(b.bucket_2)}">${fmtUsdSigned(b.bucket_2 || 0)}</td>
            <td class="num ${signedClass(b.bucket_3)}">${fmtUsdSigned(b.bucket_3 || 0)}</td>
            <td class="num ${signedClass(b.bucket_4)}">${fmtUsdSigned(b.bucket_4 || 0)}</td>
            <td>${safeText(top.underlying, "none")} <span class="dim small">${fmtUsdSigned(
            top.pnl_usd
          )}</span></td>
            <td>${statusPill(s.status)}</td>
          </tr>`;
        })
        .join("")}</tbody></table>`;

    const worst = panel.worst_shock || scenarios[0];
    const rows = worst.contributors || [];
    els.contributorContent.innerHTML = `
      <p class="dim small">Showing top contributors for <strong>${safeText(worst.label)}</strong>.</p>
      <table class="tight contributor-table"><thead><tr>
        <th>Bucket</th><th>Underlying</th><th>Symbols</th><th>Driver</th><th>Shock P&amp;L</th><th>Gross $</th>
      </tr></thead><tbody>${rows
        .slice(0, 15)
        .map(
          (r) => `<tr>
            <td>${safeText(r.bucket, "-")}</td>
            <td><strong>${safeText(r.underlying)}</strong></td>
            <td class="dim">${safeText(r.symbols, "-")}</td>
            <td>${safeText(r.driver, "-")}</td>
            <td class="num ${signedClass(r.pnl_usd)}">${fmtUsdSigned(r.pnl_usd)}</td>
            <td class="num">${r.gross_notional_usd == null ? "-" : fmtUsd(r.gross_notional_usd)}</td>
          </tr>`
        )
        .join("")}</tbody></table>`;
  }

  function renderBreaches(breaches) {
    if (!breaches || !breaches.length) {
      els.breaches.innerHTML = `<span class="pill pill-ok">No top-level breaches</span>`;
      return;
    }
    els.breaches.innerHTML = breaches
      .map(
        (b) =>
          `<span class="pill pill-${b.status}">${b.metric}: ${
            typeof b.value === "number" ? b.value.toFixed(2) : b.value
          }${
            b.limit ? ` (limit warn ${b.limit.warn}, hard ${b.limit.hard})` : ""
          }</span>`
      )
      .join("");
  }

  function renderSleeveTable(rows) {
    els.sleeveBody.innerHTML = rows
      .map((r) => {
        const trCls =
          r.drift_status === "hard"
            ? "row-hard"
            : r.drift_status === "warn"
            ? "row-warn"
            : "";
        return `<tr class="${trCls}">
          <td><strong>${r.bucket}</strong></td>
          <td class="num">${fmtUsd(r.gross_usd)}</td>
          <td class="num ${signedClass(r.net_usd)}">${fmtUsd(r.net_usd)}</td>
          <td class="num">${fmtPct(r.actual_weight, 1)}</td>
          <td class="num">${
            r.target_weight == null ? "-" : fmtPct(r.target_weight, 0)
          }</td>
          <td class="num">${fmtPp(r.drift_pp, 1)}</td>
          <td>${statusPill(r.drift_status)}</td>
          <td class="num ${signedClass(r.pnl_usd)}">${fmtUsdSigned(r.pnl_usd)}</td>
        </tr>`;
      })
      .join("");
  }

  function renderBucketContent(bucketKey, bucket) {
    if (!bucket) {
      els.bucketContent.innerHTML = `<p class="dim">No data for ${bucketKey}.</p>`;
      return;
    }

    const winnersTbl = (bucket.winners || [])
      .map(
        (r) => `<tr>
          <td><strong>${safeText(r.display_name || r.symbol)}</strong></td>
          <td class="dim">${safeText(r.description || r.symbols, "not in source")}</td>
          <td class="num pos">${fmtUsdSigned(r.total_pnl)}</td>
        </tr>`
      )
      .join("");
    const losersTbl = (bucket.losers || [])
      .map(
        (r) => `<tr>
          <td><strong>${safeText(r.display_name || r.symbol)}</strong></td>
          <td class="dim">${safeText(r.description || r.symbols, "not in source")}</td>
          <td class="num neg">${fmtUsdSigned(r.total_pnl)}</td>
        </tr>`
      )
      .join("");
    const expoTbl = (bucket.exposure_rows || [])
      .slice(0, 25)
      .map(
        (r) => `<tr>
          <td><strong>${safeText(r.underlying)}</strong></td>
          <td class="dim">${safeText(r.symbols, "not in source")}</td>
          <td class="num">${r.n_legs}</td>
          <td class="num ${signedClass(r.net_notional_usd)}">${fmtUsd(
          r.net_notional_usd
        )}</td>
          <td class="num">${fmtUsd(r.gross_notional_usd)}</td>
        </tr>`
      )
      .join("");
    els.bucketContent.innerHTML = `
      <div class="two-col">
        <div>
          <h3>Top winners (${bucket.winners?.length || 0})</h3>
          <table class="tight"><thead><tr>
            <th>Symbol</th><th>Name</th><th>P&amp;L</th>
          </tr></thead><tbody>${winnersTbl || "<tr><td colspan=3 class=dim>(none)</td></tr>"}</tbody></table>
        </div>
        <div>
          <h3>Top losers (${bucket.losers?.length || 0})</h3>
          <table class="tight"><thead><tr>
            <th>Symbol</th><th>Name</th><th>P&amp;L</th>
          </tr></thead><tbody>${losersTbl || "<tr><td colspan=3 class=dim>(none)</td></tr>"}</tbody></table>
        </div>
      </div>
      <h3>Exposure by underlying (top 25 of ${bucket.n_exposure_rows})</h3>
      <table class="tight"><thead><tr>
        <th>Underlying</th><th>Symbols</th><th>Legs</th><th>Net $</th><th>Gross $</th>
      </tr></thead><tbody>${expoTbl || "<tr><td colspan=5 class=dim>(none)</td></tr>"}</tbody></table>
    `;
  }

  function renderBorrow(borrowPanel) {
    if (!borrowPanel) {
      els.borrowContent.innerHTML = `<p class="dim">No borrow data.</p>`;
      return;
    }
    const b = borrowPanel.borrow || {};
    const p = borrowPanel.positions || {};
    const expensiveRows = (b.names_over_30pct || [])
      .slice(0, 30)
      .map(
        (r) => `<tr class="${r.fee_rate_pct >= 90 ? "row-hard" : r.fee_rate_pct >= 60 ? "row-warn" : ""}">
          <td><strong>${r.symbol}</strong></td>
          <td class="num">${r.fee_rate_pct.toFixed(1)}%</td>
        </tr>`
      )
      .join("");
    els.borrowContent.innerHTML = `
      <div class="strip" style="margin-bottom:8px;">
        <div class="stat"><div class="label">Positions</div><div class="value">${
          p.n_positions ?? 0
        }</div><div class="sub">long ${fmtUsd(p.long_notional_usd)} / short ${fmtUsd(p.short_notional_usd)}</div></div>
        <div class="stat"><div class="label">Borrow rows</div><div class="value">${
          b.n_rows ?? 0
        }</div><div class="sub">${b.n_symbols ?? 0} symbols</div></div>
        <div class="stat"><div class="label">Total borrow interest</div><div class="value neg">${fmtUsdSigned(
          b.total_interest_usd
        )}</div></div>
        <div class="stat"><div class="label">Max APR</div><div class="value">${
          (b.max_fee_rate_pct ?? 0).toFixed(1)
        }%</div></div>
      </div>
      <h3>Names with borrow APR &ge; 30%</h3>
      <table class="tight"><thead><tr>
        <th>Symbol</th><th>APR (max in window)</th>
      </tr></thead><tbody>${expensiveRows || "<tr><td colspan=2 class=dim>(none)</td></tr>"}</tbody></table>
      <p class="dim small">Threshold colours: amber &ge; 60%, red &ge; 90% (Bucket 4 universe entry filter).</p>
    `;
  }

  /* ----------------------- Tabs ------------------------------- */
  function bindTabs(snapshot) {
    els.bucketTabs.querySelectorAll("button.tab").forEach((btn) => {
      btn.onclick = () => {
        els.bucketTabs
          .querySelectorAll("button.tab")
          .forEach((b) => b.classList.toggle("active", b === btn));
        renderBucketContent(btn.dataset.bucket, snapshot.buckets[btn.dataset.bucket]);
      };
    });
    const first = els.bucketTabs.querySelector("button.tab.active");
    if (first) renderBucketContent(first.dataset.bucket, snapshot.buckets[first.dataset.bucket]);
  }

  /* ----------------------- Auth wiring ------------------------ */
  function showDashboard() {
    els.loginPanel.hidden = true;
    els.dashboard.hidden = false;
    els.logoutBtn.hidden = false;
  }
  function showLogin() {
    els.loginPanel.hidden = false;
    els.dashboard.hidden = true;
    els.logoutBtn.hidden = true;
    els.runDateLabel.textContent = "No data loaded";
    els.generatedAtLabel.textContent = "";
  }

  async function loadSnapshot(pat) {
    const snap = await window.LSAuth.fetchRepoFile(pat, cfg.snapshotPath);
    if (typeof snap !== "object" || snap == null) {
      throw new Error("Snapshot is not valid JSON.");
    }
    return snap;
  }

  function renderAll(snap) {
    els.runDateLabel.textContent = `Run: ${snap.run_date}`;
    els.generatedAtLabel.textContent =
      "Generated " + new Date(snap.generated_at_utc).toLocaleString();

    renderDataQuality(snap.data_quality || {});
    renderCockpit(snap);
    renderAlerts(snap.alert_rows || []);
    renderStrip(snap.book || {});
    renderBreaches(snap.book?.breaches || []);
    renderScenarios(snap.scenario_panel || {});
    renderSleeveTable(snap.book?.sleeve_table || []);
    bindTabs(snap);
    renderBorrow(snap.borrow_panel || {});
    els.rawTotals.textContent = JSON.stringify(snap.raw_totals || {}, null, 2);
  }

  async function tryAutoLogin() {
    const pat = window.LSAuth.getStoredPat();
    if (!pat) return false;
    try {
      await window.LSAuth.validatePat(pat);
      const snap = await loadSnapshot(pat);
      renderAll(snap);
      showDashboard();
      return true;
    } catch (e) {
      console.warn("auto-login failed:", e);
      window.LSAuth.clearPat();
      return false;
    }
  }

  function bindLogin() {
    els.loginForm.addEventListener("submit", async (ev) => {
      ev.preventDefault();
      els.loginError.hidden = true;
      const submitBtn = els.loginForm.querySelector("button[type=submit]");
      submitBtn.disabled = true;
      const pat = els.patInput.value.trim();
      try {
        await window.LSAuth.validatePat(pat);
        window.LSAuth.storePat(pat);
        const snap = await loadSnapshot(pat);
        renderAll(snap);
        showDashboard();
        els.patInput.value = "";
      } catch (e) {
        console.error(e);
        els.loginError.textContent = e.message || String(e);
        els.loginError.hidden = false;
      } finally {
        submitBtn.disabled = false;
      }
    });

    els.logoutBtn.addEventListener("click", () => {
      window.LSAuth.clearPat();
      showLogin();
    });
  }

  /* ----------------------- Boot ------------------------------- */
  bindLogin();
  tryAutoLogin();
})();
