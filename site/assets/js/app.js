/**
 * SPA entry-point for the ls-algo risk dashboard.
 *
 * Flow:
 *   1) Load investors.json; if users exist, require login id + password.
 *   2) On success, fetch ./data/latest.json and render every panel.
 */

(function () {
  const cfg = window.LS_ALGO_CONFIG;
  if (!cfg || !cfg.snapshotUrl) {
    console.error("LS_ALGO_CONFIG missing; check that config.js loaded.");
  }

  const els = {
    loginPanel: document.getElementById("login-panel"),
    dashboard: document.getElementById("dashboard"),
    loginForm: document.getElementById("login-form"),
    loginIdInput: document.getElementById("login-id-input"),
    loginPassInput: document.getElementById("login-pass-input"),
    loginError: document.getElementById("login-error"),
    logoutBtn: document.getElementById("logout-btn"),
    authUserLabel: document.getElementById("auth-user-label"),
    runDateLabel: document.getElementById("run-date-label"),
    generatedAtLabel: document.getElementById("generated-at-label"),
    cockpitStrip: document.getElementById("cockpit-strip"),
    dataQuality: document.getElementById("data-quality"),
    dqDrilldown: document.getElementById("dq-drilldown"),
    alertRows: document.getElementById("alert-rows"),
    scenarioContent: document.getElementById("scenario-content"),
    contributorContent: document.getElementById("contributor-content"),
    factorSummary: document.getElementById("factor-summary"),
    factorSectors: document.getElementById("factor-sectors"),
    factorLong: document.getElementById("factor-long"),
    factorShort: document.getElementById("factor-short"),
    concentrationSummary: document.getElementById("concentration-summary"),
    concentrationNames: document.getElementById("concentration-names"),
    concentrationSectors: document.getElementById("concentration-sectors"),
    squeezeContent: document.getElementById("squeeze-content"),
    actionQueue: document.getElementById("action-queue"),
    slideRiskContent: document.getElementById("slide-risk-content"),
    slideRiskMeta: document.getElementById("slide-risk-meta"),
    borrowShockContent: document.getElementById("borrow-shock-content"),
    borrowShockMeta: document.getElementById("borrow-shock-meta"),
    volShockContent: document.getElementById("vol-shock-content"),
    volShockMeta: document.getElementById("vol-shock-meta"),
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
      if (els.dqDrilldown) {
        els.dqDrilldown.hidden = true;
        els.dqDrilldown.innerHTML = "";
      }
      return;
    }
    const breaks = (dq.reconciliations || []).filter((r) => r.status !== "ok").length;
    const missing = dq.missing_source_count || 0;
    const missCols = dq.missing_required_column_count || 0;
    const blanks = dq.blank_render_field_count || 0;
    const issues = missing + missCols + blanks + breaks;
    const label =
      dq.status === "ok" && issues === 0
        ? `data quality ok (${(dq.sources || []).length} sources)`
        : `${dq.status}: ${missing} missing sources, ${missCols} missing columns, ${blanks} blanks, ${breaks} reconciliation breaks`;
    const expandHint = `<span class="dq-expand-hint">(click for details)</span>`;
    els.dataQuality.innerHTML = `<button type="button" class="dq-pill-button" aria-expanded="false">${statusPill(dq.status, label)}${expandHint}</button>`;
    const btn = els.dataQuality.querySelector(".dq-pill-button");
    if (btn) {
      btn.addEventListener("click", () => {
        if (!els.dqDrilldown) return;
        const open = !els.dqDrilldown.hidden;
        if (open) {
          els.dqDrilldown.hidden = true;
          btn.setAttribute("aria-expanded", "false");
        } else {
          renderDataQualityDetail(dq);
          els.dqDrilldown.hidden = false;
          btn.setAttribute("aria-expanded", "true");
        }
      });
    }
    if (els.dqDrilldown) {
      els.dqDrilldown.hidden = true;
      els.dqDrilldown.innerHTML = "";
    }
  }

  function renderDataQualityDetail(dq) {
    if (!els.dqDrilldown) return;
    const sections = [];

    const recs = dq.reconciliations || [];
    const recBreaks = recs.filter((r) => r.status !== "ok");
    if (recBreaks.length) {
      sections.push(`<div class="dq-group">
        <h4>Reconciliation breaks (${recBreaks.length})</h4>
        <table class="tight">
          <thead><tr>
            <th>Check</th><th>Book</th><th>Sum of components</th>
            <th>Components</th><th>Diff</th><th>Status</th>
          </tr></thead>
          <tbody>
            ${recBreaks
              .map((r) => {
                const diff = r.diff_pct ?? r.diff_pct_of_gross ?? 0;
                const comps = (r.components_included || []).join(", ") || "-";
                return `<tr>
                  <td>${r.name || "-"}</td>
                  <td>${fmtUsd(r.book_value)}</td>
                  <td>${fmtUsd(r.component_sum)}</td>
                  <td>${comps}</td>
                  <td>${fmtPct(diff, 2)}</td>
                  <td>${statusPill(r.status, r.status)}</td>
                </tr>`;
              })
              .join("")}
          </tbody>
        </table>
      </div>`);
    }

    const sources = dq.sources || [];
    const missingSources = sources.filter((s) => !s.exists);
    if (missingSources.length) {
      sections.push(`<div class="dq-group">
        <h4>Missing source files (${missingSources.length})</h4>
        <table class="tight">
          <thead><tr><th>Name</th><th>Expected path</th></tr></thead>
          <tbody>
            ${missingSources
              .map(
                (s) => `<tr>
                <td>${s.name}</td>
                <td><code>${s.path}</code></td>
              </tr>`
              )
              .join("")}
          </tbody>
        </table>
      </div>`);
    }

    const missingCols = sources.filter(
      (s) => (s.missing_required_columns || []).length > 0
    );
    if (missingCols.length) {
      sections.push(`<div class="dq-group">
        <h4>Sources missing required columns (${missingCols.length})</h4>
        <table class="tight">
          <thead><tr>
            <th>Name</th><th>Path</th><th>Missing required columns</th>
          </tr></thead>
          <tbody>
            ${missingCols
              .map(
                (s) => `<tr>
                <td>${s.name}</td>
                <td><code>${s.path}</code></td>
                <td>${(s.missing_required_columns || []).join(", ")}</td>
              </tr>`
              )
              .join("")}
          </tbody>
        </table>
      </div>`);
    }

    const blanks = dq.blank_render_fields || [];
    if (blanks.length) {
      const grouped = {};
      blanks.forEach((b) => {
        const key = `${b.bucket}|${b.section}|${b.field}`;
        grouped[key] = grouped[key] || {
          bucket: b.bucket,
          section: b.section,
          field: b.field,
          rows: [],
        };
        grouped[key].rows.push(b.row);
      });
      const groupRows = Object.values(grouped).sort(
        (a, b) => b.rows.length - a.rows.length
      );
      sections.push(`<div class="dq-group">
        <h4>Blank rendering fields (${blanks.length})</h4>
        <table class="tight">
          <thead><tr>
            <th>Bucket</th><th>Section</th><th>Field</th>
            <th>Row count</th><th>Row indices (first 10)</th>
          </tr></thead>
          <tbody>
            ${groupRows
              .map(
                (g) => `<tr>
                <td>${g.bucket}</td>
                <td>${g.section}</td>
                <td><code>${g.field}</code></td>
                <td>${g.rows.length}</td>
                <td>${g.rows
                  .slice(0, 10)
                  .map((r) => `#${r}`)
                  .join(", ")}${g.rows.length > 10 ? ", ..." : ""}</td>
              </tr>`
              )
              .join("")}
          </tbody>
        </table>
      </div>`);
    }

    const rowCounts = dq.row_counts || {};
    const rcEntries = Object.entries(rowCounts);
    if (rcEntries.length) {
      sections.push(`<div class="dq-group">
        <h4>Per-bucket row counts</h4>
        <table class="tight">
          <thead><tr><th>Bucket</th><th>P&amp;L rows</th><th>Exposure rows</th></tr></thead>
          <tbody>
            ${rcEntries
              .map(
                ([bucket, rc]) => `<tr>
                <td>${bucket}</td>
                <td>${rc.pnl_rows ?? 0}</td>
                <td>${rc.exposure_rows ?? 0}</td>
              </tr>`
              )
              .join("")}
          </tbody>
        </table>
      </div>`);
    }

    // All sources (collapsed, for reference)
    if (sources.length) {
      sections.push(`<div class="dq-group">
        <h4>All sources (${sources.length})</h4>
        <table class="tight sortable">
          <thead><tr>
            <th>Name</th><th>Exists</th><th>Path</th><th># columns</th>
          </tr></thead>
          <tbody>
            ${sources
              .map(
                (s) => `<tr>
                <td>${s.name}</td>
                <td>${s.exists ? "yes" : '<span class="bad">missing</span>'}</td>
                <td><code>${s.path}</code></td>
                <td>${(s.columns || []).length}</td>
              </tr>`
              )
              .join("")}
          </tbody>
        </table>
      </div>`);
    }

    const okRecs = recs.filter((r) => r.status === "ok");
    if (okRecs.length) {
      sections.push(`<div class="dq-group">
        <h4>Reconciliation checks passing (${okRecs.length})</h4>
        <table class="tight">
          <thead><tr>
            <th>Check</th><th>Book</th><th>Sum of components</th>
            <th>Components</th><th>Diff</th>
          </tr></thead>
          <tbody>
            ${okRecs
              .map((r) => {
                const diff = r.diff_pct ?? r.diff_pct_of_gross ?? 0;
                const comps = (r.components_included || []).join(", ") || "-";
                return `<tr>
                  <td>${r.name || "-"}</td>
                  <td>${fmtUsd(r.book_value)}</td>
                  <td>${fmtUsd(r.component_sum)}</td>
                  <td>${comps}</td>
                  <td>${fmtPct(diff, 2)}</td>
                </tr>`;
              })
              .join("")}
          </tbody>
        </table>
      </div>`);
    }

    els.dqDrilldown.innerHTML = sections.length
      ? `<div class="dq-header"><span class="dq-meta">Run date: ${dq.run_date || "-"}</span></div>${sections.join(
          ""
        )}`
      : `<div class="dq-empty">No data quality issues to report.</div>`;
    if (typeof enableSortableTables === "function") {
      enableSortableTables();
    }
  }

  function sparklineSvg(values, opts) {
    const cleaned = (values || []).map((v) => (v == null ? null : Number(v)));
    const real = cleaned.filter((v) => v != null && !Number.isNaN(v));
    if (real.length < 2) return "";
    const width = 80;
    const height = 22;
    const min = Math.min(...real);
    const max = Math.max(...real);
    const span = max - min || 1;
    const n = cleaned.length;
    const pts = cleaned.map((v, i) => {
      const x = (i / (n - 1)) * (width - 2) + 1;
      if (v == null) return null;
      const y = height - 2 - ((v - min) / span) * (height - 4);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    });
    const segments = [];
    let buf = [];
    pts.forEach((p) => {
      if (p == null) {
        if (buf.length) segments.push(buf);
        buf = [];
      } else {
        buf.push(p);
      }
    });
    if (buf.length) segments.push(buf);
    const polylines = segments
      .map(
        (s) =>
          `<polyline fill="none" stroke="${opts?.stroke || "#16537e"}" stroke-width="1.4" points="${s.join(
            " "
          )}"/>`
      )
      .join("");
    const lastVal = real[real.length - 1];
    const lastIdx = cleaned.lastIndexOf(lastVal);
    const lastX = (lastIdx / (n - 1)) * (width - 2) + 1;
    const lastY = height - 2 - ((lastVal - min) / span) * (height - 4);
    return `<svg viewBox="0 0 ${width} ${height}" width="${width}" height="${height}" class="spark">${polylines}<circle cx="${lastX.toFixed(
      1
    )}" cy="${lastY.toFixed(1)}" r="1.6" fill="${opts?.stroke || "#16537e"}"/></svg>`;
  }

  function deltaBadge(value, opts) {
    if (value == null || Number.isNaN(value)) return "";
    const isPct = !!opts?.isPct;
    const isInt = !!opts?.isInt;
    const lowerIsBetter = opts?.lowerIsBetter !== false;
    const sign = value > 0 ? "+" : value < 0 ? "" : "";
    const display = isPct
      ? sign + (value * 100).toFixed(2) + "pp"
      : isInt
      ? sign + Math.round(value)
      : sign + value.toFixed(2);
    const positive = lowerIsBetter ? value < 0 : value > 0;
    const cls = value === 0 ? "delta-flat" : positive ? "delta-pos" : "delta-neg";
    return `<span class="delta ${cls}">${display}</span>`;
  }

  function renderCockpit(snap) {
    const book = snap.book || {};
    const worst = snap.worst_shock || {};
    const top = worst.top_contributor || {};
    const alertRows = snap.alert_rows || [];
    const factor = snap.factor_panel || {};
    const factorTotals = factor.totals || {};
    const history = snap.history || [];
    const deltas = snap.deltas || {};

    const series = (key) => history.map((h) => h?.[key]);

    const items = [
      {
        label: "NAV",
        value: fmtUsd(book.nav_usd),
        spark: sparklineSvg(series("nav_usd")),
        delta: "",
      },
      {
        label: "Gross / NAV",
        value: fmtPct(book.gross_exposure_pct_nav, 0),
        sub: fmtUsd(book.gross_notional_usd),
        spark: sparklineSvg(series("gross_pct_nav")),
        delta: deltaBadge(deltas.delta_gross_pct_nav, { isPct: true }),
      },
      {
        label: "Net / NAV",
        value: fmtPct(book.net_exposure_pct_nav, 1),
        sub: fmtUsd(book.net_notional_usd),
        cls: signedClass(book.net_notional_usd),
        spark: sparklineSvg(series("net_pct_nav")),
        delta: deltaBadge(deltas.delta_net_pct_nav, { isPct: true, lowerIsBetter: false }),
      },
      {
        label: "Net beta",
        value:
          factorTotals.net_beta_to_spy == null
            ? "-"
            : factorTotals.net_beta_to_spy.toFixed(2) + "x",
        sub: factorTotals.beta_coverage_gross_pct == null
          ? "coverage unknown"
          : `${fmtPct(factorTotals.beta_coverage_gross_pct, 0)} curated`,
        cls: signedClass(factorTotals.net_beta_to_spy),
        spark: sparklineSvg(series("net_beta_to_spy")),
        delta: deltaBadge(deltas.delta_net_beta_to_spy, { lowerIsBetter: false }),
      },
      {
        label: "Worst scenario contributor",
        value: safeText(top.underlying, "none"),
        sub: `${safeText(worst.scenario || worst.label, "no scenario")} / ${fmtUsdSigned(top.pnl_usd)}`,
        cls: signedClass(top.pnl_usd),
      },
      {
        label: "Worst slide P&L",
        value: fmtUsdSigned(worst.pnl_usd),
        sub: `${safeText(worst.scenario || worst.label, "no scenario")} / ${fmtPct(
          worst.pnl_pct_nav ?? worst.total_pnl_pct_nav,
          2
        )}`,
        cls: signedClass(worst.pnl_usd ?? worst.pnl_pct_nav),
        spark: sparklineSvg(series("worst_shock_pct_nav"), { stroke: "#c0392b" }),
        delta: deltaBadge(deltas.delta_worst_shock_pct_nav, { isPct: true, lowerIsBetter: false }),
      },
      {
        label: "Alerts",
        value: String(alertRows.length),
        sub: `${alertRows.filter((r) => r.status === "hard").length} hard`,
        cls: alertRows.some((r) => r.status === "hard") ? "neg" : "",
        spark: sparklineSvg(series("n_alerts_hard"), { stroke: "#c0392b" }),
        delta: deltaBadge(deltas.delta_n_alerts_hard, { isInt: true }),
      },
    ];
    els.cockpitStrip.innerHTML = items
      .map(
        (it) => `
        <div class="stat stat-${it.cls || "neutral"}">
          <div class="label">${it.label}</div>
          <div class="value ${it.cls || ""}">${it.value}</div>
          ${it.delta ? `<div class="delta-row">${it.delta}</div>` : ""}
          ${it.sub ? `<div class="sub">${it.sub}</div>` : ""}
          ${it.spark ? `<div class="spark-row">${it.spark}</div>` : ""}
        </div>`
      )
      .join("");
  }

  function formatSqueezeAlertSource(r) {
    if (r.source && !String(r.metric || "").startsWith("borrow_squeeze:")) {
      return safeText(r.source, "source unavailable");
    }
    if (r.source) {
      return safeText(r.source);
    }
    const parts = [];
    if (r.short_qty != null) {
      parts.push(`${Math.round(r.short_qty).toLocaleString()} sh short`);
    }
    if (r.binding_cap_label) {
      parts.push(`Binding: ${r.binding_cap_label}`);
    }
    return parts.length ? parts.join(" · ") : "source unavailable";
  }

  function formatSqueezeAlertValue(r) {
    if (!String(r.metric || "").startsWith("borrow_squeeze:")) {
      return typeof r.value === "number" ? r.value.toFixed(3) : safeText(r.value, "-");
    }
    const util =
      typeof r.value === "number"
        ? `${r.value.toFixed(2)}×`
        : safeText(r.value, "-");
    const bind =
      r.binding_cap === "median_volume"
        ? "median vol cap"
        : r.binding_cap === "shares_outstanding"
        ? "shares-out cap"
        : "liquidity cap";
    const sub = [];
    if (r.short_vs_shares_out_cap != null) {
      sub.push(`shares-out ${fmtPct(r.short_vs_shares_out_cap, 0)}`);
    }
    if (r.short_vs_adv_cap != null) {
      sub.push(`median vol ${fmtPct(r.short_vs_adv_cap, 0)}`);
    }
    return `<div class="num">${util}</div><div class="dim small">${bind}${
      sub.length ? ` · ${sub.join(" · ")}` : ""
    }</div>`;
  }

  function renderAlerts(rows) {
    if (!rows || !rows.length) {
      els.alertRows.innerHTML = `<span class="pill pill-ok">No risk alerts</span>`;
      return;
    }
    els.alertRows.innerHTML = `
      <table class="tight alert-table"><thead><tr>
        <th>Status</th><th>Category</th><th>Metric</th><th>Value</th><th>Limit</th><th>Action</th>
      </tr></thead><tbody>${rows
        .slice(0, 12)
        .map((r) => {
          const limit =
            r.limit && typeof r.limit === "object"
              ? `warn ${r.limit.warn}, hard ${r.limit.hard}`
              : "-";
          const isSqueeze = String(r.metric || "").startsWith("borrow_squeeze:");
          const metricSub = formatSqueezeAlertSource(r);
          const value = isSqueeze
            ? formatSqueezeAlertValue(r)
            : typeof r.value === "number"
            ? r.value.toFixed(3)
            : safeText(r.value, "-");
          return `<tr class="${rowStatusClass(r.status)}">
            <td>${statusPill(r.status)}</td>
            <td class="dim small">${safeText(r.category_label || r.category, "-")}</td>
            <td><strong>${safeText(r.label || r.metric)}</strong><div class="dim small">${metricSub}</div></td>
            <td class="num">${value}</td>
            <td>${limit}</td>
            <td>${safeText(r.action, "review")}</td>
          </tr>`;
        })
        .join("")}</tbody></table>`;
  }

  function scenarioHeatClass(pct) {
    if (pct == null || Number.isNaN(pct)) return "";
    const a = Math.abs(pct);
    const tier = a >= 0.05 ? 3 : a >= 0.03 ? 2 : a >= 0.01 ? 1 : 0;
    if (tier === 0) return "";
    return pct < 0 ? `heat-neg-${tier}` : `heat-pos-${tier}`;
  }

  function formatPnlConcentration(conc, opts = {}) {
    const topN = opts.topN || 5;
    if (!conc || !conc.top_contributors || !conc.top_contributors.length) {
      return `<span class="dim small">No contributor detail</span>`;
    }
    const share = conc.top_n_share_of_scenario;
    const shareTxt =
      share == null ? "" : `Top ${topN} explain <strong>${fmtPct(share, 0)}</strong> of |scenario P&amp;L|`;
    if (conc.diversified && share != null && share < 0.7) {
      return `${shareTxt} <span class="dim small">(diversified)</span>`;
    }
    const chips = conc.top_contributors
      .map((c) => {
        const pct =
          c.pct_of_scenario_abs == null ? "" : ` ${fmtPct(c.pct_of_scenario_abs, 0)}`;
        return `<span class="conc-chip"><strong>${safeText(c.underlying)}</strong> ${fmtUsdSigned(c.pnl_usd)}${pct}</span>`;
      })
      .join("");
    return `<div class="conc-line">${shareTxt}</div><div class="conc-chips">${chips}</div>`;
  }

  function formatDecayConcentration(conc) {
    if (!conc || !conc.top_contributors || !conc.top_contributors.length) {
      return `<span class="dim">—</span>`;
    }
    const share = conc.top_n_share_of_scenario;
    if (conc.diversified && share != null && share < 0.5) {
      return `<span class="dim small">Diversified (max ${fmtPct(
        conc.top_contributors[0].pct_of_scenario_abs,
        0
      )})</span>`;
    }
    const shareLead =
      share == null
        ? ""
        : `<span class="dim small">Top 3 = ${fmtPct(share, 0)} · </span>`;
    const chips = conc.top_contributors
      .map((c) => {
        const pct =
          c.pct_of_scenario_abs == null ? "" : ` (${fmtPct(c.pct_of_scenario_abs, 0)})`;
        return `<strong>${safeText(c.underlying)}</strong> ${fmtUsdSigned(c.pnl_usd)}${pct}`;
      })
      .join(" · ");
    return `${shareLead}<span class="small">${chips}</span>`;
  }

  function formatBorrowVictims(conc) {
    if (!conc || !conc.top_victims || !conc.top_victims.length) {
      return "—";
    }
    if (conc.diversified && conc.top_n_share != null && conc.top_n_share < 0.7) {
      const lead = conc.top_victims[0];
      return `<span class="dim small">Diversified</span> (max ${safeText(lead.symbol)} ${fmtPct(
        lead.pct_of_shock,
        0
      )})`;
    }
    return conc.top_victims
      .map((v) => {
        const pct = v.pct_of_shock == null ? "" : ` (${fmtPct(v.pct_of_shock, 0)})`;
        return `<strong>${safeText(v.symbol)}</strong> ${fmtUsd(v.annual_cost_delta_usd)}${pct}`;
      })
      .join(" · ");
  }

  function borrowCostHeatClass(pctNav) {
    if (pctNav == null || Number.isNaN(Number(pctNav))) return "";
    const a = Math.abs(Number(pctNav));
    const tier = a >= 0.5 ? 3 : a >= 0.15 ? 2 : a >= 0.05 ? 1 : 0;
    if (tier === 0) return "";
    return `borrow-cost-${tier}`;
  }

  function formatSlideShockHeader(row) {
    if (row?.label) return safeText(row.label);
    if (row?.shock_pct != null && !Number.isNaN(Number(row.shock_pct))) {
      const pct = Number(row.shock_pct) * 100;
      const sign = row.shock_pct >= 0 ? "+" : "";
      const body = Math.abs(pct % 1) < 1e-9 ? Math.abs(pct).toFixed(0) : Math.abs(pct).toFixed(1);
      return `${sign}${row.shock_pct < 0 ? "-" : ""}${body}%`;
    }
    if (row?.vix_shock_pts != null && !Number.isNaN(Number(row.vix_shock_pts))) {
      const pts = Number(row.vix_shock_pts);
      return `VIX ${pts >= 0 ? "+" : ""}${pts} pts`;
    }
    return "-";
  }

  function renderScenarios(panel) {
    const scenarios = panel?.scenarios || [];
    const bookOnly = !!panel?.book_only_mode;
    const bookOnlyBadge = bookOnly
      ? `<div class="callout warn" role="status"><strong>Book-only mode.</strong> ${safeText(
          panel.book_only_reason,
          "Bucket attribution unavailable; scenarios computed from book aggregates."
        )}</div>`
      : "";
    if (!scenarios.length) {
      els.scenarioContent.innerHTML = `${bookOnlyBadge}<p class="dim">No scenario data available.</p>`;
      els.contributorContent.innerHTML = `<p class="dim">No contributor data available.</p>`;
      return;
    }
    const sleeveCols = bookOnly
      ? "<th>Book</th>"
      : "<th>B1</th><th>B2</th><th>B3</th><th>B4</th>";
    els.scenarioContent.innerHTML = `
      ${bookOnlyBadge}
      <table class="tight scenario-table"><thead><tr>
        <th>Scenario</th><th>Total P&amp;L</th><th>% NAV</th>
        ${sleeveCols}<th>Top offender</th><th>Status</th>
      </tr></thead><tbody>${scenarios
        .map((s) => {
          const b = s.bucket_pnl || {};
          const top = s.top_contributor || {};
          const sleeveCells = bookOnly
            ? `<td class="num ${signedClass(b.book)}">${fmtUsdSigned(b.book || 0)}</td>`
            : `<td class="num ${signedClass(b.bucket_1)}">${fmtUsdSigned(
                b.bucket_1 || 0
              )}</td>
            <td class="num ${signedClass(b.bucket_2)}">${fmtUsdSigned(
                b.bucket_2 || 0
              )}</td>
            <td class="num ${signedClass(b.bucket_3)}">${fmtUsdSigned(
                b.bucket_3 || 0
              )}</td>
            <td class="num ${signedClass(b.bucket_4)}">${fmtUsdSigned(
                b.bucket_4 || 0
              )}</td>`;
          return `<tr class="${rowStatusClass(s.status)}">
            <td><strong>${safeText(s.label)}</strong><div class="dim small">${safeText(
              s.description,
              ""
            )}</div></td>
            <td class="num ${signedClass(s.pnl_usd)} ${scenarioHeatClass(
            s.pnl_pct_nav
          )}">${fmtUsdSigned(s.pnl_usd)}</td>
            <td class="num ${signedClass(s.pnl_pct_nav)} ${scenarioHeatClass(
            s.pnl_pct_nav
          )}">${fmtPct(s.pnl_pct_nav, 2)}</td>
            ${sleeveCells}
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

  function renderFactor(panel) {
    if (!panel || panel.available === false) {
      els.factorSummary.innerHTML = `<div class="callout warn">${safeText(
        panel?.reason,
        "Factor panel unavailable: missing net_exposure_by_underlying.csv."
      )}</div>`;
      els.factorSectors.innerHTML = "";
      els.factorLong.innerHTML = "";
      els.factorShort.innerHTML = "";
      return;
    }
    const t = panel.totals || {};
    const summary = [
      { label: "Underlyings", value: String(t.n_underlyings ?? 0) },
      {
        label: "Beta-weighted net",
        value: fmtUsdSigned(t.beta_weighted_net_usd),
        sub: t.net_beta_to_spy == null ? "-" : t.net_beta_to_spy.toFixed(2) + "x NAV",
        cls: signedClass(t.beta_weighted_net_usd),
      },
      {
        label: "Beta-weighted gross",
        value: fmtUsd(t.beta_weighted_gross_usd),
        sub: t.gross_beta_to_spy == null ? "-" : t.gross_beta_to_spy.toFixed(2) + "x NAV",
      },
      {
        label: "Beta coverage",
        value: fmtPct(t.beta_coverage_gross_pct, 0),
        sub: "% of gross with curated beta",
        cls: (t.beta_coverage_gross_pct ?? 0) >= 0.7 ? "pos" : "neg",
      },
    ];
    els.factorSummary.innerHTML = summary
      .map(
        (it) => `
        <div class="stat stat-${it.cls || "neutral"}">
          <div class="label">${it.label}</div>
          <div class="value ${it.cls || ""}">${it.value}</div>
          ${it.sub ? `<div class="sub">${it.sub}</div>` : ""}
        </div>`
      )
      .join("");

    const sectors = panel.by_sector || [];
    els.factorSectors.innerHTML = `
      <table class="tight"><thead><tr>
        <th>Sector</th><th>Names</th><th>Book share</th><th>Net $</th><th>Gross $</th>
        <th>Beta-wtd net</th><th>Beta-wtd gross</th>
      </tr></thead><tbody>${sectors
        .map(
          (r) => `<tr>
            <td><strong>${safeText(r.sector)}</strong></td>
            <td class="num">${r.n_names}</td>
            <td class="num">${fmtPct(r.pct_book_gross, 1)}</td>
            <td class="num ${signedClass(r.net_notional_usd)}">${fmtUsdSigned(
            r.net_notional_usd
          )}</td>
            <td class="num">${fmtUsd(r.gross_notional_usd)}</td>
            <td class="num ${signedClass(r.beta_weighted_net_usd)}">${fmtUsdSigned(
            r.beta_weighted_net_usd
          )}</td>
            <td class="num">${fmtUsd(r.beta_weighted_gross_usd)}</td>
          </tr>`
        )
        .join("")}</tbody></table>`;

    function fmtBeta(v, source) {
      if (v == null || Number.isNaN(Number(v))) return "-";
      const dim = source === "default" ? "dim" : "";
      return `<span class="${dim}">${Number(v).toFixed(2)}</span>`;
    }

    function rowTbl(rows) {
      return `<table class="tight"><thead><tr>
        <th>Underlying</th><th>Sector</th><th>Beta SPY</th><th>Beta QQQ</th><th>Beta IWM</th><th>Net $</th><th>Beta net $</th>
      </tr></thead><tbody>${(rows || [])
        .map(
          (r) => `<tr>
            <td><strong>${safeText(r.underlying)}</strong> <span class="dim small">${safeText(
            r.symbols,
            ""
          )}</span></td>
            <td>${safeText(r.sector)}</td>
            <td class="num" title="${safeText(r.beta_source, "")}">${fmtBeta(
            r.beta_to_spy,
            r.beta_source
          )}</td>
            <td class="num">${fmtBeta(r.beta_to_qqq, r.beta_source)}</td>
            <td class="num">${fmtBeta(r.beta_to_iwm, r.beta_source)}</td>
            <td class="num ${signedClass(r.net_notional_usd)}">${fmtUsdSigned(
            r.net_notional_usd
          )}</td>
            <td class="num ${signedClass(r.beta_weighted_net_usd)}">${fmtUsdSigned(
            r.beta_weighted_net_usd
          )}</td>
          </tr>`
        )
        .join("")}</tbody></table>`;
    }
    els.factorLong.innerHTML = rowTbl(panel.top_beta_long);
    els.factorShort.innerHTML = rowTbl(panel.top_beta_short);
  }

  /* ---------------- Slide risk strips (Phase 1) ---------------- */
  function formatSlideHorizonTooltip(h) {
    if (!h) return "";
    const parts = [];
    if (h.beta_pnl_usd != null) parts.push(`Beta ${fmtUsdSigned(h.beta_pnl_usd)}`);
    if (h.decay_pnl_usd != null) parts.push(`Decay ${fmtUsdSigned(h.decay_pnl_usd)}`);
    if (h.borrow_pnl_usd != null) parts.push(`Borrow ${fmtUsdSigned(h.borrow_pnl_usd)}`);
    if (h.distribution_pnl_usd != null && Math.abs(h.distribution_pnl_usd) > 1) {
      parts.push(`Dist ${fmtUsdSigned(h.distribution_pnl_usd)}`);
    }
    if (h.total_pnl_usd != null) parts.push(`Total ${fmtUsdSigned(h.total_pnl_usd)}`);
    return parts.join(" · ");
  }

  function formatDecayReference(ref) {
    if (!ref || typeof ref !== "object") return "";
    const keys = ["1M", "3M", "6M", "12M"].filter((k) => ref[k]);
    if (!keys.length) return "";
    return keys
      .map((k) => {
        const row = ref[k];
        const total = fmtPct(row.total_pnl_pct_nav, 1);
        const decay = fmtPct(row.decay_pnl_pct_nav, 1);
        const borrow = fmtPct(row.borrow_pnl_pct_nav, 1);
        return `<strong>${k}</strong> ${total} (decay ${decay}, borrow ${borrow})`;
      })
      .join(" · ");
  }

  function renderSlideRisk(panel) {
    if (!els.slideRiskContent) return;
    if (!panel || panel.available === false) {
      els.slideRiskContent.innerHTML = `<div class="callout warn">${safeText(
        panel?.reason,
        "Slide risk panel unavailable: factor or NAV missing."
      )}</div>`;
      if (els.slideRiskMeta) els.slideRiskMeta.innerHTML = "";
      return;
    }
    const scenarioHorizons = panel.scenario_horizons || ["1M", "3M", "6M", "12M"];
    if (els.slideRiskMeta) {
      const letf = panel.n_letf_names || 0;
      const total = panel.n_names_total || 0;
      const betaCov =
        panel.beta_coverage_gross_pct == null
          ? ""
          : ` &middot; ${fmtPct(panel.beta_coverage_gross_pct, 0)} beta coverage (gross)`;
      els.slideRiskMeta.innerHTML = `<span class="dim small">T+0 = instantaneous beta; ${scenarioHorizons.join(
        " / "
      )} = etf-dashboard scenario model (forecast vol 1.0&times;, borrow included)${betaCov} &middot; ${letf}/${total} LETF &middot; ${panel.n_names_with_vol || 0} with vol</span>`;
    }
    const worst = panel.worst_shock || null;
    const worstShare =
      worst?.top5_share_of_scenario != null
        ? fmtPct(worst.top5_share_of_scenario, 0)
        : null;
    const worstBanner = worst
      ? `<div class="callout warn" role="status"><strong>Worst instantaneous slide (T+0):</strong> ${safeText(
          worst.label
        )} &rarr; ${fmtPct(worst.pnl_pct_nav ?? worst.total_pnl_pct_nav, 2)} of NAV${
          worstShare
            ? ` &middot; top 5 names explain <strong>${worstShare}</strong> of |scenario P&amp;L|`
            : ""
        }.</div>`
      : "";
    const indices = panel.indices || [];
    const spxIdx = indices.find((idx) => idx.index === "SPX");
    const decayRef = spxIdx?.decay_reference || null;
    const decayBanner = decayRef
      ? `<div class="callout-soft" role="status"><strong>Expected 0&sigma; book carry (forecast vol 1.0&times;):</strong> ${formatDecayReference(
          decayRef
        )}</div>`
      : "";
    const stripsHtml = indices
      .map((idx) => {
        if (idx.strip_type === "vix_pts") {
          const vixRows = idx.shock_rows || [];
          const vixHeader = vixRows
            .map((r) => `<th class="slide-shock">${formatSlideShockHeader(r)}</th>`)
            .join("");
          const vixCells = vixRows
            .map(
              (r) =>
                `<td class="num ${signedClass(r.pnl_pct_nav)} ${scenarioHeatClass(r.pnl_pct_nav)}" title="${fmtUsdSigned(r.pnl_usd)}">${fmtPct(r.pnl_pct_nav, 1)}</td>`
            )
            .join("");
          const volRows = idx.vol_regime_rows || [];
          const volBody = volRows
            .map(
              (r) => `<tr>
              <td><strong>${safeText(r.label)}</strong></td>
              <td class="num ${signedClass(r.pnl_pct_nav)}">${fmtPct(r.pnl_pct_nav, 2)}</td>
              <td class="small">${formatDecayConcentration(r.decay_concentration)}</td>
            </tr>`
            )
            .join("");
          return `<div class="slide-strip">
            <div class="slide-strip-head"><h3>VIX vega shocks (T+0)</h3><span class="dim small">Instantaneous vega P&amp;L</span></div>
            <div class="slide-strip-scroll">
              <table class="tight slide-table"><thead><tr><th class="row-label">Shock</th>${vixHeader}</tr></thead>
              <tbody><tr><th class="row-label">T+0</th>${vixCells}</tr></tbody></table>
            </div>
            <h4 class="dim small">Forecast-vol regime (LETF decay overlay)</h4>
            <table class="tight"><thead><tr><th>Regime</th><th>% NAV</th><th>Decay concentration</th></tr></thead><tbody>${volBody || "<tr><td colspan=3 class=dim>(none)</td></tr>"}</tbody></table>
          </div>`;
        }
        const rows = idx.shock_rows || [];
        const headerCells = rows
          .map((r) => `<th class="slide-shock">${formatSlideShockHeader(r)}</th>`)
          .join("");
        const t0Row = rows
          .map(
            (r) =>
              `<td class="num ${signedClass(r.pnl_pct_nav)} ${scenarioHeatClass(r.pnl_pct_nav)}" title="${safeText(r.label)}: ${fmtUsdSigned(r.pnl_usd)}">${fmtPct(r.pnl_pct_nav, 1)}</td>`
          )
          .join("");
        const horizonRows = scenarioHorizons
          .map((hk) => {
            const cells = rows
              .map((r) => {
                const h = (r.horizons || []).find(
                  (hh) => hh.horizon_key === hk || String(hh.horizon_days) === String(hk)
                );
                if (!h) return `<td class="dim">-</td>`;
                const tip = formatSlideHorizonTooltip(h);
                return `<td class="num ${signedClass(h.total_pnl_pct_nav)} ${scenarioHeatClass(
                  h.total_pnl_pct_nav
                )}" title="${safeText(tip)}">${fmtPct(h.total_pnl_pct_nav, 1)}</td>`;
              })
              .join("");
            return `<tr><th class="row-label">${safeText(hk)} <span class="dim small">modeled</span></th>${cells}</tr>`;
          })
          .join("");
        const binding = idx.binding_shock || null;
        const bindingConc = idx.binding_concentration || binding?.concentration || null;
        const concBlock = binding
          ? `<div class="slide-concentration callout-soft">
              <strong>${safeText(binding.label)} (binding down shock):</strong>
              ${formatPnlConcentration(bindingConc, { topN: 5 })}
            </div>`
          : "";
        return `<div class="slide-strip">
          <div class="slide-strip-head">
            <h3>${safeText(idx.index)} <span class="dim small">(${Math.round((idx.coverage_pct || 0) * 100)}% coverage, ${idx.n_names_covered}/${idx.n_names_total} names)</span></h3>
          </div>
          <div class="slide-strip-scroll">
            <table class="tight slide-table">
              <thead><tr><th class="row-label">Horizon</th>${headerCells}</tr></thead>
              <tbody>
                <tr><th class="row-label">T+0 <span class="dim small">beta only</span></th>${t0Row}</tr>
                ${horizonRows}
              </tbody>
            </table>
          </div>
          ${concBlock}
        </div>`;
      })
      .join("");
    els.slideRiskContent.innerHTML = `${worstBanner}${decayBanner}${stripsHtml || `<p class="dim">No index strips available.</p>`}`;
  }

  /* ---------------- Borrow shock strip (Phase 3) ---------------- */
  function renderBorrowShock(panel) {
    if (!els.borrowShockContent) return;
    if (!panel || panel.available === false) {
      els.borrowShockContent.innerHTML = `<div class="callout warn">${safeText(
        panel?.reason,
        "Borrow shock panel unavailable: positions or borrow data missing."
      )}</div>`;
      if (els.borrowShockMeta) els.borrowShockMeta.innerHTML = "";
      return;
    }
    if (els.borrowShockMeta) {
      els.borrowShockMeta.innerHTML = `<span class="dim small">${panel.n_short_symbols} short symbols &middot; current cost ${fmtUsd(
        panel.current_annual_cost_usd
      )} / yr (${fmtPct(panel.current_annual_cost_pct_nav, 2)} of NAV) &middot; ${panel.persistence_days}-day persistence column</span>`;
    }
    const tiles = panel.summary_tiles || {};
    const focus50 = tiles.focus_abs_50bp;
    const focus2x = tiles.focus_mult_2x;
    const top3Share = panel.current_borrow_concentration?.top_n_share;
    const summaryStrip = `
      <div class="strip borrow-summary-strip">
        <div class="stat stat-neutral">
          <div class="label">Current borrow / yr</div>
          <div class="value">${fmtUsd(panel.current_annual_cost_usd)}</div>
          <div class="sub">${fmtPct(panel.current_annual_cost_pct_nav, 2)} of NAV · ${panel.n_short_symbols} shorts</div>
        </div>
        <div class="stat stat-neutral">
          <div class="label">+50bp stress</div>
          <div class="value">${focus50 ? fmtUsd(focus50.annual_delta_usd) : "—"}</div>
          <div class="sub">${focus50 ? `${fmtPct(focus50.annual_delta_pct_nav, 2)} NAV/yr · ${panel.persistence_days}d ${fmtUsd(focus50.persistence_delta_usd)}` : ""}</div>
        </div>
        <div class="stat stat-neutral">
          <div class="label">2× APR stress</div>
          <div class="value">${focus2x ? fmtUsd(focus2x.annual_delta_usd) : "—"}</div>
          <div class="sub">${focus2x ? `${fmtPct(focus2x.annual_delta_pct_nav, 2)} NAV/yr · ${panel.persistence_days}d ${fmtUsd(focus2x.persistence_delta_usd)}` : ""}</div>
        </div>
        <div class="stat stat-neutral">
          <div class="label">Borrow concentration</div>
          <div class="value">${top3Share == null ? "—" : fmtPct(top3Share, 0)}</div>
          <div class="sub">Top 3 explain share of current cost</div>
        </div>
      </div>`;

    const renderLadder = (ladder, headerLabel) => {
      if (!ladder || !ladder.length) return "";
      const header = `<th>${headerLabel}</th><th>Add'l ann. cost</th><th>% NAV / yr</th><th>${panel.persistence_days}-day persist</th><th>Top 3 add'l cost</th>`;
      const body = ladder
        .map((r) => {
          const heat = borrowCostHeatClass(r.annual_delta_pct_nav);
          const rowCls = r.is_focus ? "row-focus" : "";
          return `<tr class="${rowCls}">
            <td><strong>${safeText(r.label)}</strong>${r.is_focus ? ' <span class="pill pill-warn">focus</span>' : ""}</td>
            <td class="num ${heat}">${fmtUsd(r.annual_delta_usd)}</td>
            <td class="num ${heat}">${fmtPct(r.annual_delta_pct_nav, 2)}</td>
            <td class="num">${fmtUsd(r.persistence_delta_usd)}</td>
            <td class="small">${formatBorrowVictims(r.victim_concentration)}</td>
          </tr>`;
        })
        .join("");
      return `<table class="tight borrow-ladder"><thead><tr>${header}</tr></thead><tbody>${body}</tbody></table>`;
    };

    const namesTable = (panel.names || []).length
      ? `<h3>Top shorts by implied annual borrow cost</h3>
         <table class="tight sortable"><thead><tr>
           <th>Symbol</th><th>Short notional</th><th>Borrow rate</th><th>Implied ann. cost</th>
         </tr></thead><tbody>${(panel.names || [])
           .map(
             (n) => `<tr>
             <td><strong>${safeText(n.symbol)}</strong></td>
             <td class="num">${fmtUsd(n.short_notional_usd)}</td>
             <td class="num">${(n.borrow_rate_pct ?? n.current_apr_pct ?? 0).toFixed(2)}%</td>
             <td class="num">${fmtUsd(n.current_annual_cost_usd)}</td>
           </tr>`
           )
           .join("")}</tbody></table>`
      : "";

    els.borrowShockContent.innerHTML = `
      ${summaryStrip}
      ${namesTable}
      <h3 class="borrow-ladder-title">Borrow rate shocks</h3>
      <p class="dim small">Absolute (+bp) and multiplicative (× APR) ladders. Focus rows highlighted.</p>
      <details open><summary><strong>Absolute shocks (+bp)</strong></summary>
        ${renderLadder(panel.abs_ladder || [], "Shock")}
      </details>
      <details><summary><strong>Multiplicative shocks (× APR)</strong></summary>
        ${renderLadder(panel.mult_ladder || [], "Multiplier")}
      </details>
    `;
  }

  /* ---------------- Vol shock strip (Phase 4) ---------------- */
  function renderVolShock(panel) {
    if (!els.volShockContent) return;
    if (!panel || panel.available === false) {
      els.volShockContent.innerHTML = `<div class="callout warn">${safeText(
        panel?.reason,
        "Vol shock panel unavailable."
      )}</div>`;
      if (els.volShockMeta) els.volShockMeta.innerHTML = "";
      return;
    }
    if (els.volShockMeta) {
      els.volShockMeta.innerHTML = `<span class="dim small">${panel.n_vega_contributors} vega-bearing names &middot; ${panel.n_letf_decay_contributors} LETF decay names &middot; T+${panel.decay_horizon_days}d horizon</span>`;
    }
    const vix = panel.vix_ladder || [];
    const vol = panel.vol_ladder || [];
    const vixTable = vix.length
      ? `<table class="tight"><thead><tr>
          <th>VIX shock</th><th>Vega P&amp;L</th><th>% NAV</th><th>Worst victim</th><th>Best gainer</th>
        </tr></thead><tbody>${vix
          .map((r) => {
            const w = (r.worst_victims || [])[0];
            const g = (r.top_gains || [])[0];
            const heat = scenarioHeatClass(r.pnl_pct_nav);
            return `<tr>
              <td><strong>${safeText(r.label)}</strong></td>
              <td class="num ${signedClass(r.pnl_usd)} ${heat}">${fmtUsdSigned(r.pnl_usd)}</td>
              <td class="num ${signedClass(r.pnl_pct_nav)} ${heat}">${fmtPct(r.pnl_pct_nav, 2)}</td>
              <td class="dim small">${
                w ? `${safeText(w.underlying)} <em>${fmtUsdSigned(w.pnl_usd)}</em><br>${safeText(w.vega_product_class, "")}` : "-"
              }</td>
              <td class="dim small">${
                g ? `${safeText(g.underlying)} <em>${fmtUsdSigned(g.pnl_usd)}</em>` : "-"
              }</td>
            </tr>`;
          })
          .join("")}</tbody></table>`
      : `<p class="dim">No vega-bearing names in book.</p>`;
    const volTable = vol.length
      ? `<table class="tight"><thead><tr>
          <th>Vol regime</th><th>Total decay</th><th>% NAV</th><th>Worst LETF</th><th>Other LETF victims</th>
        </tr></thead><tbody>${vol
          .map((r) => {
            const victims = r.worst_victims || [];
            const v0 = victims[0];
            const others = victims.slice(1, 4);
            const heat = scenarioHeatClass(r.pnl_pct_nav);
            return `<tr>
              <td><strong>${safeText(r.label)}</strong> <span class="dim small">T+${r.horizon_days}d</span></td>
              <td class="num ${signedClass(r.pnl_usd)} ${heat}">${fmtUsdSigned(r.pnl_usd)}</td>
              <td class="num ${signedClass(r.pnl_pct_nav)} ${heat}">${fmtPct(r.pnl_pct_nav, 2)}</td>
              <td>${
                v0
                  ? `<strong>${safeText(v0.underlying)}</strong> <span class="dim small">${fmtUsdSigned(
                      v0.pnl_usd
                    )}<br>k=${v0.leverage.toFixed(1)}, &sigma; ${(v0.current_sigma_pct).toFixed(0)}&rarr;${(v0.stressed_sigma_pct).toFixed(0)}%</span>`
                  : "-"
              }</td>
              <td class="dim small">${others
                .map((v) => `${safeText(v.underlying)} <em>${fmtUsdSigned(v.pnl_usd)}</em>`)
                .join(", ") || "-"}</td>
            </tr>`;
          })
          .join("")}</tbody></table>`
      : `<p class="dim">No LETF positions in book.</p>`;
    els.volShockContent.innerHTML = `
      <h3>VIX absolute shocks (vega P&amp;L on short-vol income / yieldboost positions)</h3>
      ${vixTable}
      <h3>Realized-vol regime multipliers (LETF decay over T+${panel.decay_horizon_days}d)</h3>
      <p class="dim small">Positive decay means the book benefits from higher vol (net short LETF decay).</p>
      ${volTable}
    `;
  }

  function formatSqueezeActionDetail(a) {
    if (!String(a.category || "").startsWith("borrow_squeeze:")) {
      return safeText(a.detail);
    }
    const lines = [];
    if (a.short_qty != null) {
      lines.push(`<strong>${Math.round(a.short_qty).toLocaleString()} sh short</strong>`);
    }
    if (a.binding_cap_label) {
      const capSh =
        a.binding_cap === "median_volume"
          ? a.cap_median_vol_shares
          : a.cap_shares_out_shares;
      const capTxt =
        capSh != null ? ` → max ${Math.round(capSh).toLocaleString()} sh` : "";
      const util =
        a.liquidity_utilization != null
          ? ` · ${fmtPct(a.liquidity_utilization, 0)} of cap`
          : "";
      lines.push(`Binding: ${safeText(a.binding_cap_label)}${capTxt}${util}`);
    }
    if (a.other_cap_label) {
      const otherUtil =
        a.binding_cap === "median_volume"
          ? a.short_vs_shares_out_cap
          : a.short_vs_adv_cap;
      if (otherUtil != null) {
        lines.push(`Other: ${safeText(a.other_cap_label)} · ${fmtPct(otherUtil, 0)} of cap`);
      }
    }
    if (a.trim_qty != null) {
      const over =
        a.over_cap_shares != null
          ? ` (${Math.round(a.over_cap_shares).toLocaleString()} sh over binding cap)`
          : "";
      lines.push(`Cut ~${Math.round(a.trim_qty).toLocaleString()} sh to warn band${over}.`);
    } else if (a.detail) {
      lines.push(safeText(a.detail));
    }
    return lines.join("<br>");
  }

  function renderActionQueue(queuePayload) {
    if (!els.actionQueue) return;
    const actions = Array.isArray(queuePayload) ? queuePayload : queuePayload?.items || [];
    const cap = queuePayload?.cap || 5;
    if (!actions.length) {
      els.actionQueue.innerHTML = `<span class="pill pill-ok">No actions required</span>`;
      return;
    }
    const shown = actions.slice(0, cap);
    const hidden = actions.slice(cap);
    const rowHtml = (a) => `<tr class="${rowStatusClass(a.status)}">
            <td>${statusPill(a.status, a.priority === 0 ? "P0" : "P1")}</td>
            <td class="dim small">${safeText(a.sleeve, "-")}</td>
            <td><strong>${safeText(a.title)}</strong><div class="dim small">${safeText(
            a.breach_category_label || a.category,
            ""
          )}</div></td>
            <td>${formatSqueezeActionDetail(a)}</td>
            <td class="dim">${safeText(a.hedge_hint, "")}</td>
            <td class="dim small">${safeText(a.source, "")}</td>
          </tr>`;
    els.actionQueue.innerHTML = `
      <table class="tight action-table"><thead><tr>
        <th>Priority</th><th>Sleeve</th><th>Action</th><th>Detail</th><th>Hedge hint</th><th>Source</th>
      </tr></thead><tbody>${shown.map(rowHtml).join("")}</tbody></table>
      ${
        hidden.length
          ? `<details class="action-more"><summary class="dim small">Show ${hidden.length} more action(s)</summary>
        <table class="tight action-table"><tbody>${hidden.map(rowHtml).join("")}</tbody></table></details>`
          : ""
      }`;
  }

  function renderConcentration(panel) {
    if (!panel || panel.available === false) {
      els.concentrationSummary.innerHTML = `<div class="callout warn">${safeText(
        panel?.reason,
        "Concentration panel unavailable: factor map missing."
      )}</div>`;
      els.concentrationNames.innerHTML = "";
      els.concentrationSectors.innerHTML = "";
      return;
    }
    const t = panel.totals || {};
    const items = [
      {
        label: "Top 5 names / NAV",
        value: fmtPct(t.top5_pct_nav, 0),
      },
      {
        label: "Top 10 / NAV",
        value: fmtPct(t.top10_pct_nav, 0),
        cls:
          t.top10_status === "hard"
            ? "neg"
            : t.top10_status === "warn"
            ? "neg"
            : "pos",
      },
      {
        label: "HHI underlying",
        value: (t.hhi_underlying || 0).toFixed(0),
        sub:
          t.hhi_underlying_status === "hard"
            ? "concentrated"
            : t.hhi_underlying_status === "warn"
            ? "elevated"
            : "diversified",
        cls:
          t.hhi_underlying_status === "hard"
            ? "neg"
            : t.hhi_underlying_status === "warn"
            ? "neg"
            : "pos",
      },
      {
        label: "HHI sector",
        value: (t.hhi_sector || 0).toFixed(0),
        sub:
          t.hhi_sector_status === "hard"
            ? "concentrated"
            : t.hhi_sector_status === "warn"
            ? "elevated"
            : "diversified",
        cls:
          t.hhi_sector_status === "hard"
            ? "neg"
            : t.hhi_sector_status === "warn"
            ? "neg"
            : "pos",
      },
    ];
    els.concentrationSummary.innerHTML = items
      .map(
        (it) => `
        <div class="stat stat-${it.cls || "neutral"}">
          <div class="label">${it.label}</div>
          <div class="value ${it.cls || ""}">${it.value}</div>
          ${it.sub ? `<div class="sub">${it.sub}</div>` : ""}
        </div>`
      )
      .join("");

    const names = (panel.top_names || []).slice(0, 15);
    els.concentrationNames.innerHTML = `
      <table class="tight"><thead><tr>
        <th>Underlying</th><th>Bucket</th><th>Sector</th><th>Gross $</th><th>Gross / NAV</th><th>Status</th>
      </tr></thead><tbody>${names
        .map(
          (r) => `<tr class="${rowStatusClass(r.status)}">
            <td><strong>${safeText(r.underlying)}</strong></td>
            <td>${safeText(r.bucket)}</td>
            <td>${safeText(r.sector)}</td>
            <td class="num">${fmtUsd(r.gross_notional_usd)}</td>
            <td class="num">${fmtPct(r.pct_nav_gross, 1)}</td>
            <td>${statusPill(r.status)}</td>
          </tr>`
        )
        .join("")}</tbody></table>`;

    const sectors = panel.by_sector || [];
    els.concentrationSectors.innerHTML = `
      <table class="tight"><thead><tr>
        <th>Sector</th><th>Names</th><th>Gross $</th><th>Share of book</th><th>Status</th>
      </tr></thead><tbody>${sectors
        .map(
          (r) => `<tr class="${rowStatusClass(r.status)}">
            <td><strong>${safeText(r.sector)}</strong></td>
            <td class="num">${r.n_names}</td>
            <td class="num">${fmtUsd(r.gross_notional_usd)}</td>
            <td class="num">${fmtPct(r.pct_book_gross, 1)}</td>
            <td>${statusPill(r.status)}</td>
          </tr>`
        )
        .join("")}</tbody></table>`;
  }

  function renderSqueeze(rows) {
    if (!els.squeezeContent) return;
    if (!rows || !rows.length) {
      els.squeezeContent.innerHTML = `<p class="dim small">No short positions found, or screener unavailable.</p>`;
      return;
    }
    const top = rows.slice(0, 100);
    els.squeezeContent.innerHTML = `
      <table class="tight"><thead><tr>
        <th>Symbol</th><th>Bucket</th><th>Short qty</th><th>Binding cap</th>
        <th>vs shares-out cap</th><th>vs median vol cap</th><th>Liquidity util</th>
        <th>Borrow rate</th><th>Status</th>
      </tr></thead><tbody>${top
        .map(
          (r) => `<tr class="${rowStatusClass(r.status)}">
            <td><strong>${safeText(r.symbol)}</strong></td>
            <td>${safeText(r.bucket)}</td>
            <td class="num">${r.short_qty == null ? "-" : Math.round(r.short_qty).toLocaleString()}</td>
            <td class="small">${safeText(r.binding_cap_label, "-")}</td>
            <td class="num">${
              r.short_vs_shares_out_cap == null ? "-" : fmtPct(r.short_vs_shares_out_cap, 0)
            }</td>
            <td class="num">${
              r.short_vs_adv_cap == null ? "-" : fmtPct(r.short_vs_adv_cap, 0)
            }</td>
            <td class="num">${
              r.liquidity_utilization == null ? "-" : fmtPct(r.liquidity_utilization, 0)
            }</td>
            <td class="num">${r.borrow_rate_pct == null ? "-" : r.borrow_rate_pct.toFixed(2) + "%"}</td>
            <td>${statusPill(r.status)}</td>
          </tr>`
        )
        .join("")}</tbody></table>
      <p class="dim small">Liquidity util = max(shares-out cap, median-vol cap). Binding cap is whichever ratio is higher. Warn &ge; 80%, hard &ge; 100% of sizing cap.</p>
    `;
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

  function renderSleeveTable(book) {
    if (!els.sleeveBody) return;
    const rows = book?.sleeve_table || [];
    const available = book?.sleeve_attribution_available !== false;
    const unavailableCell = `<td class="num dim" title="${safeText(
      book?.sleeve_attribution_reason,
      "sleeve attribution unavailable"
    )}">unavailable</td>`;
    els.sleeveBody.innerHTML = rows
      .map((r) => {
        const trCls =
          r.drift_status === "hard"
            ? "row-hard"
            : r.drift_status === "warn"
            ? "row-warn"
            : "";
        const grossCell = available
          ? `<td class="num">${fmtUsd(r.gross_usd)}</td>`
          : unavailableCell;
        const targetGrossCell = available
          ? `<td class="num dim">${r.target_gross_usd == null ? "-" : fmtUsd(r.target_gross_usd)}</td>`
          : unavailableCell;
        const netCell = available
          ? `<td class="num ${signedClass(r.net_usd)}">${fmtUsd(r.net_usd)}</td>`
          : unavailableCell;
        const actualCell = available
          ? `<td class="num">${fmtPct(r.actual_weight, 1)}</td>`
          : unavailableCell;
        const driftCell = available
          ? `<td class="num">${fmtPp(r.drift_pp, 1)}</td>`
          : unavailableCell;
        const statusCell = available
          ? `<td>${statusPill(r.drift_status)}</td>`
          : `<td>${statusPill("unknown", "n/a")}</td>`;
        return `<tr class="${trCls}">
          <td><strong>${safeText(r.bucket_label || r.bucket)}</strong></td>
          ${grossCell}
          ${targetGrossCell}
          ${netCell}
          ${actualCell}
          <td class="num">${
            r.target_weight == null ? "-" : fmtPct(r.target_weight, 0)
          }</td>
          ${driftCell}
          ${statusCell}
          <td class="num ${signedClass(r.pnl_usd)}">${fmtUsdSigned(r.pnl_usd)}</td>
        </tr>`;
      })
      .join("");

    const banner = document.getElementById("sleeve-banner");
    if (banner) {
      if (!available) {
        banner.hidden = false;
        banner.className = "callout hard";
        banner.innerHTML = `<strong>Sleeve attribution unavailable.</strong> ${safeText(
          book?.sleeve_attribution_reason,
          "Bucket totals do not reconcile to book aggregate."
        )} Bucket P&amp;L is still shown because it sums from a separate source.`;
      } else {
        banner.hidden = true;
      }
    }
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
    const shortRows = borrowPanel.short_etf_rows || [];
    const maxBorrowRate = shortRows.reduce(
      (m, r) => Math.max(m, Number(r.borrow_rate_pct) || 0),
      0
    );
    const expensiveRows = (b.names_over_30pct || [])
      .slice(0, 30)
      .map((r) => {
        const br = r.borrow_rate_pct ?? r.fee_rate_pct ?? 0;
        return `<tr class="${br >= 90 ? "row-hard" : br >= 60 ? "row-warn" : ""}">
          <td><strong>${r.symbol}</strong></td>
          <td class="num">${br.toFixed(2)}%</td>
        </tr>`;
      })
      .join("");
    const shortEtfRows = (borrowPanel.short_etf_rows || [])
      .slice(0, 50)
      .map(
        (r) => `<tr>
          <td><strong>${safeText(r.symbol)}</strong></td>
          <td class="num">${fmtUsd(r.short_notional_usd)}</td>
          <td class="num">${r.borrow_rate_pct == null ? "-" : r.borrow_rate_pct.toFixed(2) + "%"}</td>
          <td class="num">${r.implied_annual_cost_usd == null ? "-" : fmtUsd(r.implied_annual_cost_usd)}</td>
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
        <div class="stat"><div class="label">Max borrow rate</div><div class="value">${maxBorrowRate.toFixed(1)}%</div></div>
      </div>
      <h3>High borrow rate (&ge; 30%)</h3>
      <table class="tight"><thead><tr>
        <th>Symbol</th><th>Borrow Rate</th>
      </tr></thead><tbody>${expensiveRows || "<tr><td colspan=2 class=dim>(none)</td></tr>"}</tbody></table>
      <h3>Short ETFs held (${borrowPanel.n_short_etfs ?? 0} of ${borrowPanel.watchlist_n_symbols ?? 0} watchlist)</h3>
      <table class="tight sortable"><thead><tr>
        <th>Symbol</th><th>Short notional</th><th>Borrow Rate</th><th>Implied ann. cost</th>
      </tr></thead><tbody>${shortEtfRows || "<tr><td colspan=4 class=dim>(none)</td></tr>"}</tbody></table>
      <p class="dim small">Borrow Rate = screener <code>borrow_current</code> (else <code>borrow_fee_annual</code>), &times; 100 — same as etf-dashboard.</p>
    `;
  }

  /* ----------------------- Tabs ------------------------------- */
  function bindTabs(snapshot) {
    if (!els.bucketTabs) return;
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
  const subnav = document.getElementById("subnav");
  let authEnabled = false;
  let investorUsers = [];
  let investorsReady = false;

  function showLoginError(message) {
    if (!els.loginError) return;
    els.loginError.textContent = message;
    els.loginError.hidden = false;
    els.loginError.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }

  function clearLoginError() {
    if (!els.loginError) return;
    els.loginError.hidden = true;
    els.loginError.textContent = "";
  }

  function showDashboard(sessionUid) {
    els.loginPanel.hidden = true;
    els.dashboard.hidden = false;
    if (authEnabled) {
      els.logoutBtn.hidden = false;
      if (els.authUserLabel) {
        els.authUserLabel.hidden = false;
        els.authUserLabel.textContent = sessionUid || "";
      }
    } else {
      els.logoutBtn.hidden = true;
      if (els.authUserLabel) els.authUserLabel.hidden = true;
    }
    if (subnav) subnav.hidden = false;
    enableSortableTables();
  }
  function showLogin() {
    els.loginPanel.hidden = false;
    els.dashboard.hidden = true;
    els.logoutBtn.hidden = true;
    if (els.authUserLabel) els.authUserLabel.hidden = true;
    if (subnav) subnav.hidden = true;
    els.runDateLabel.textContent = "No data loaded";
    els.generatedAtLabel.textContent = "";
  }

  /* ----------------------- Theme toggle ----------------------- */
  function applyTheme(theme) {
    document.body.setAttribute("data-theme", theme || "light");
    const btn = document.getElementById("theme-toggle");
    if (btn) btn.textContent = theme === "dark" ? "Light" : "Dark";
  }
  (function initTheme() {
    let stored = null;
    try { stored = localStorage.getItem("ls-algo-theme"); } catch (e) {}
    applyTheme(stored === "dark" ? "dark" : "light");
    const btn = document.getElementById("theme-toggle");
    if (btn) {
      btn.addEventListener("click", () => {
        const next = document.body.getAttribute("data-theme") === "dark" ? "light" : "dark";
        applyTheme(next);
        try { localStorage.setItem("ls-algo-theme", next); } catch (e) {}
      });
    }
  })();

  /* ----------------------- Sortable tables -------------------- */
  function parseSortValue(cell) {
    const raw = cell.textContent.trim();
    const cleaned = raw
      .replace(/[$,+\s]/g, "")
      .replace(/%$/, "")
      .replace(/pp$/, "");
    const n = parseFloat(cleaned);
    return Number.isFinite(n) ? n : raw.toLowerCase();
  }

  function enableSortableTables() {
    document.querySelectorAll("table.tight").forEach((tbl) => {
      const head = tbl.tHead;
      if (!head) return;
      Array.from(head.rows[0].cells).forEach((th, idx) => {
        if (th.classList.contains("sortable")) return;
        th.classList.add("sortable");
        th.addEventListener("click", () => {
          const tbody = tbl.tBodies[0];
          if (!tbody) return;
          const rows = Array.from(tbody.rows);
          const dir = th.classList.contains("sort-asc") ? "desc" : "asc";
          head.rows[0].querySelectorAll("th").forEach((h) => {
            h.classList.remove("sort-asc", "sort-desc");
          });
          th.classList.add(dir === "asc" ? "sort-asc" : "sort-desc");
          rows.sort((a, b) => {
            const av = parseSortValue(a.cells[idx]);
            const bv = parseSortValue(b.cells[idx]);
            if (typeof av === "number" && typeof bv === "number") {
              return dir === "asc" ? av - bv : bv - av;
            }
            return dir === "asc"
              ? String(av).localeCompare(String(bv))
              : String(bv).localeCompare(String(av));
          });
          rows.forEach((r) => tbody.appendChild(r));
        });
      });
    });
  }

  async function loadSnapshot() {
    const url = `${cfg.snapshotUrl}?t=${Date.now()}`;
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) {
      throw new Error(`Could not load snapshot: ${res.status} ${res.statusText}`);
    }
    const snap = await res.json();
    if (typeof snap !== "object" || snap == null) {
      throw new Error("Snapshot is not valid JSON.");
    }
    return snap;
  }

  function renderAll(snap) {
    els.runDateLabel.textContent = `Run: ${snap.run_date}`;
    els.generatedAtLabel.textContent =
      "Generated " + new Date(snap.generated_at_utc).toLocaleString();

    const steps = [
      () => renderDataQuality(snap.data_quality || {}),
      () => renderCockpit(snap),
      () => renderAlerts(snap.alert_rows || []),
      () => renderActionQueue(snap.action_queue || []),
      () => renderSlideRisk(snap.slide_risk_panel || {}),
      () => renderBorrowShock(snap.borrow_shock_panel || {}),
      () => renderConcentration(snap.concentration_panel || {}),
      () => renderFactor(snap.factor_panel || {}),
      () => renderSleeveTable(snap.book || {}),
      () => bindTabs(snap),
      () => renderBorrow(snap.borrow_panel || {}),
      () => renderSqueeze((snap.borrow_panel || {}).squeeze_rows || []),
    ];
    for (const step of steps) {
      try {
        step();
      } catch (e) {
        console.error("render step failed:", e);
        throw new Error(`Dashboard render failed: ${e.message || e}`);
      }
    }
    if (els.rawTotals) {
      els.rawTotals.textContent = JSON.stringify(snap.raw_totals || {}, null, 2);
    }
    enableSortableTables();
  }

  function showDashboardLoading(sessionUid) {
    showDashboard(sessionUid);
    if (els.cockpitStrip) {
      els.cockpitStrip.innerHTML =
        '<div class="callout dim">Loading risk snapshot (~2&nbsp;MB). This can take 10–30 seconds on a slow connection.</div>';
    }
  }

  async function openDashboard(sessionUid) {
    showDashboardLoading(sessionUid);
    const snap = await loadSnapshot();
    renderAll(snap);
  }

  function bindLogin() {
    if (!els.loginForm) {
      console.error("login-form element not found");
      return;
    }
    els.loginForm.addEventListener("submit", async (ev) => {
      ev.preventDefault();
      clearLoginError();
      const submitBtn =
        document.getElementById("login-submit-btn") ||
        els.loginForm.querySelector("button[type=submit]");
      const submitLabel = submitBtn ? submitBtn.textContent : "Sign in";
      if (!investorsReady) {
        showLoginError("Still loading login configuration. Wait a moment and try again.");
        return;
      }
      if (!window.crypto || !window.crypto.subtle) {
        showLoginError(
          "This browser cannot verify passwords (crypto.subtle unavailable). Use HTTPS or a modern browser."
        );
        return;
      }
      if (submitBtn) submitBtn.disabled = true;
      const userId = els.loginIdInput.value.trim();
      const password = els.loginPassInput.value;
      try {
        if (submitBtn) submitBtn.textContent = "Checking password…";
        const ok = await window.LSAuth.verifyLogin(userId, password, investorUsers);
        if (!ok) {
          showLoginError(
            "Invalid login id or password. Use the same id and password as etf-dashboard (e.g. dgoldman)."
          );
          return;
        }
        const uid = userId.toLowerCase();
        window.LSAuth.writeAuthSession(uid);
        els.loginPassInput.value = "";
        if (submitBtn) submitBtn.textContent = "Loading dashboard…";
        await openDashboard(uid);
      } catch (e) {
        console.error(e);
        showLoginError(e.message || String(e));
        showLogin();
      } finally {
        if (submitBtn) {
          submitBtn.disabled = false;
          submitBtn.textContent = submitLabel;
        }
      }
    });

    els.logoutBtn.addEventListener("click", () => {
      window.LSAuth.clearAuthSession();
      showLogin();
    });
  }

  async function boot() {
    bindLogin();
    try {
      const { users, authEnabled: enabled } = await window.LSAuth.loadInvestors();
      investorUsers = users;
      authEnabled = enabled;
      investorsReady = true;
    } catch (e) {
      console.error(e);
      showLoginError(`Could not load ${window.LSAuth?.INVESTORS_URL || "investors.json"}: ${e.message || e}`);
      investorsReady = true;
      showLogin();
      return;
    }

    if (!authEnabled) {
      try {
        await openDashboard(null);
      } catch (e) {
        console.error(e);
        showLoginError(e.message || String(e));
        showLogin();
      }
      return;
    }

    const ids = new Set(investorUsers.map((u) => String(u.id).toLowerCase()));
    const existing = window.LSAuth.getStoredSession(ids);
    if (existing) {
      try {
        await openDashboard(existing);
        return;
      } catch (e) {
        console.warn("session restore failed:", e);
        window.LSAuth.clearAuthSession();
        showLoginError(`Session expired or load failed: ${e.message || e}`);
      }
    }
    showLogin();
  }

  /* ----------------------- Boot ------------------------------- */
  boot();
})();
