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
          } (limit warn ${b.limit.warn}, hard ${b.limit.hard})</span>`
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
          <td><strong>${r.symbol}</strong></td>
          <td class="dim">${r.description || ""}</td>
          <td class="num pos">${fmtUsdSigned(r.total_pnl)}</td>
        </tr>`
      )
      .join("");
    const losersTbl = (bucket.losers || [])
      .map(
        (r) => `<tr>
          <td><strong>${r.symbol}</strong></td>
          <td class="dim">${r.description || ""}</td>
          <td class="num neg">${fmtUsdSigned(r.total_pnl)}</td>
        </tr>`
      )
      .join("");
    const expoTbl = (bucket.exposure_rows || [])
      .slice(0, 25)
      .map(
        (r) => `<tr>
          <td><strong>${r.underlying}</strong></td>
          <td class="dim">${r.symbols}</td>
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

    renderStrip(snap.book || {});
    renderBreaches(snap.book?.breaches || []);
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
