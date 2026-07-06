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
    freshnessBadge: document.getElementById("freshness-badge"),
    generatedAtLabel: document.getElementById("generated-at-label"),
    cockpitStrip: document.getElementById("cockpit-strip"),
    pnlStrip: document.getElementById("pnl-strip"),
    drawdownMeta: document.getElementById("drawdown-meta"),
    pnlMeta: document.getElementById("pnl-meta"),
    pnlSummary: document.getElementById("pnl-summary"),
    pnlControls: document.getElementById("pnl-controls"),
    pnlDailyChart: document.getElementById("pnl-daily-chart"),
    pnlDailyTable: document.getElementById("pnl-daily-table"),
    pnlWeeklyTable: document.getElementById("pnl-weekly-table"),
    pnlBucketChart: document.getElementById("pnl-bucket-chart"),
    pnlTableTitle: document.getElementById("pnl-table-title"),
    moversWinners: document.getElementById("movers-winners"),
    moversLosers: document.getElementById("movers-losers"),
    borrowShockContent: document.getElementById("borrow-shock-content"),
    borrowShockMeta: document.getElementById("borrow-shock-meta"),
    sharedUnderlyingContent: document.getElementById("shared-underlying-content"),
    dataQuality: document.getElementById("data-quality"),
    dqDrilldown: document.getElementById("dq-drilldown"),
    alertRows: document.getElementById("alert-rows"),
    scenarioContent: document.getElementById("scenario-content"),
    contributorContent: document.getElementById("contributor-content"),
    factorSummary: document.getElementById("factor-summary"),
    factorSectors: document.getElementById("factor-sectors"),
    factorByBucket: document.getElementById("factor-by-bucket"),
    factorLong: document.getElementById("factor-long"),
    factorShort: document.getElementById("factor-short"),
    concentrationSummary: document.getElementById("concentration-summary"),
    concentrationNames: document.getElementById("concentration-names"),
    concentrationSectors: document.getElementById("concentration-sectors"),
    squeezeContent: document.getElementById("squeeze-content"),
    actionQueue: document.getElementById("action-queue"),
    slideRiskContent: document.getElementById("slide-risk-content"),
    slideRiskMeta: document.getElementById("slide-risk-meta"),
    volShockContent: document.getElementById("vol-shock-content"),
    volShockMeta: document.getElementById("vol-shock-meta"),
    strip: document.getElementById("strip"),
    breaches: document.getElementById("breaches"),
    bucketSleevePanel: document.getElementById("bucket-sleeve-panel"),
    bucketTabs: document.getElementById("bucket-tabs"),
    bucketContent: document.getElementById("bucket-content"),
    borrowContent: document.getElementById("borrow-content"),
    rawTotals: document.getElementById("raw-totals"),
    dashboardTabs: document.getElementById("dashboard-tabs"),
  };

  const DASH_TAB_IDS = ["overview", "pnl", "risk", "book", "data"];
  let _lastSnap = null;
  const _tabRendered = new Set();

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
        label: "P&L YTD",
        value: fmtUsdSigned(book.pnl_today_usd),
        sub: fmtPct(book.pnl_today_pct_nav, 2) + " of NAV (strategy cumulative)",
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
    const lineage = dq.lineage || {};
    if (lineage.manifest_present || lineage.nav_source) {
      sections.unshift(`<div class="dq-group">
        <h4>Lineage</h4>
        <table class="tight"><tbody>
          <tr><td>NAV source</td><td><code>${safeText(lineage.nav_source, "-")}</code></td></tr>
          <tr><td>Manifest</td><td>${lineage.manifest_present ? "present" : "missing"}</td></tr>
          <tr><td>Checksums</td><td>${lineage.checksum_count ?? 0}</td></tr>
          <tr><td>Git SHA</td><td><code>${safeText(lineage.git_sha, "-").slice(0, 12)}</code></td></tr>
        </tbody></table>
      </div>`);
    }
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

  function pnlBarChartSvg(rows, opts) {
    const data = (rows || []).filter((r) => r && r.daily_usd != null);
    if (!data.length) return `<p class="dim">No P&amp;L history to chart.</p>`;
    const width = opts?.width || 720;
    const height = opts?.height || 180;
    const padL = 48;
    const padR = 12;
    const padT = 12;
    const padB = 28;
    const innerW = width - padL - padR;
    const innerH = height - padT - padB;
    const vals = data.map((r) => Number(r.daily_usd));
    const maxAbs = Math.max(...vals.map((v) => Math.abs(v)), 1);
    const barW = Math.max(2, innerW / data.length - 2);
    const zeroY = padT + innerH / 2;
    const bars = data
      .map((r, i) => {
        const v = Number(r.daily_usd);
        const x = padL + i * (innerW / data.length) + 1;
        const h = (Math.abs(v) / maxAbs) * (innerH / 2 - 2);
        const y = v >= 0 ? zeroY - h : zeroY;
        const fill = v >= 0 ? "#1a7f4b" : "#c0392b";
        const tip = `${r.date}: ${fmtUsdSigned(v)}`;
        return `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${barW.toFixed(
          1
        )}" height="${h.toFixed(1)}" fill="${fill}" opacity="0.85"><title>${safeText(tip)}</title></rect>`;
      })
      .join("");
    const labels = data
      .filter((_, i) => i % Math.max(1, Math.floor(data.length / 8)) === 0 || i === data.length - 1)
      .map((r) => {
        const i = data.indexOf(r);
        const x = padL + i * (innerW / data.length) + barW / 2;
        return `<text x="${x.toFixed(1)}" y="${height - 6}" text-anchor="middle" class="pnl-axis-label">${safeText(
          r.date.slice(5)
        )}</text>`;
      })
      .join("");
    return `<svg viewBox="0 0 ${width} ${height}" width="100%" class="pnl-bar-chart" preserveAspectRatio="xMidYMid meet">
      <line x1="${padL}" y1="${zeroY}" x2="${width - padR}" y2="${zeroY}" stroke="#ccc" stroke-width="1"/>
      ${bars}${labels}
    </svg>`;
  }

  function pnlBucketBarsHtml(buckets, labels, opts) {
    const entries = Object.entries(buckets || {}).filter(([, v]) => v != null && Math.abs(v) > 0.5);
    if (!entries.length) return `<p class="dim">No bucket moves in this period.</p>`;
    const maxAbs = Math.max(...entries.map(([, v]) => Math.abs(Number(v))), 1);
    const rows = entries
      .sort((a, b) => Math.abs(Number(b[1])) - Math.abs(Number(a[1])))
      .map(([k, v]) => {
        const pct = (Math.abs(Number(v)) / maxAbs) * 100;
        const cls = Number(v) >= 0 ? "pos" : "neg";
        const lbl = (labels && labels[k]) || k;
        return `<div class="pnl-bucket-row">
          <div class="pnl-bucket-label">${safeText(lbl)}</div>
          <div class="pnl-bucket-track"><div class="pnl-bucket-fill ${cls}" style="width:${pct.toFixed(1)}%"></div></div>
          <div class="pnl-bucket-val num ${cls}">${fmtUsdSigned(v)}</div>
        </div>`;
      })
      .join("");
    return `<div class="pnl-bucket-bars">${rows}</div>
      <p class="dim small">${safeText(opts?.caption || "")}</p>`;
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
    const dailyPnlSeries = series("pnl_daily_usd");
    const hasDailyPnl = dailyPnlSeries.some((v) => v != null && !Number.isNaN(v));

    const items = [
      {
        label: "NAV",
        value: fmtUsd(book.nav_usd),
        sub: snap.nav_source ? `source: ${snap.nav_source}` : "",
        spark: sparklineSvg(series("nav_usd")),
        delta: "",
      },
      {
        label: "P&L today",
        value: fmtUsdSigned(book.pnl_daily_usd),
        sub:
          book.pnl_daily_usd == null
            ? "no prior run"
            : `${fmtPct(book.pnl_daily_pct_nav, 2)} of NAV vs ${safeText(
                book.pnl_daily_prior_run_date,
                "prior"
              )}`,
        cls: signedClass(book.pnl_daily_usd),
        spark: sparklineSvg(hasDailyPnl ? dailyPnlSeries : series("pnl_cum_usd")),
      },
      {
        label: "P&L YTD",
        value: fmtUsdSigned(book.pnl_ytd_usd ?? book.pnl_today_usd),
        sub: `${fmtPct(
          book.pnl_ytd_pct_nav ?? book.pnl_today_pct_nav,
          2
        )} of NAV (cumulative)`,
        cls: signedClass(book.pnl_ytd_usd ?? book.pnl_today_usd),
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
          : `${fmtPct(factorTotals.beta_coverage_gross_pct, 0)} computed`,
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
      els.factorByBucket.innerHTML = "";
      els.factorLong.innerHTML = "";
      els.factorShort.innerHTML = "";
      return;
    }
    const t = panel.totals || {};

    function fmtBetaNav(v) {
      if (v == null || Number.isNaN(Number(v))) return "-";
      const n = Number(v);
      const sign = n >= 0 ? "+" : "";
      return `${sign}${n.toFixed(2)}x NAV`;
    }

    const headline = [
      {
        label: "Net β SPY",
        value: fmtBetaNav(t.net_beta_to_spy),
        sub: fmtUsdSigned(t.beta_weighted_net_usd),
        cls: signedClass(t.net_beta_to_spy),
      },
      {
        label: "Net β QQQ",
        value: fmtBetaNav(t.net_beta_to_qqq),
        sub: fmtUsdSigned(t.beta_weighted_net_qqq_usd),
        cls: signedClass(t.net_beta_to_qqq),
      },
      {
        label: "Net β IWM",
        value: fmtBetaNav(t.net_beta_to_iwm),
        sub: fmtUsdSigned(t.beta_weighted_net_iwm_usd),
        cls: signedClass(t.net_beta_to_iwm),
      },
      {
        label: "Net β BTC",
        value: fmtBetaNav(t.net_beta_to_btc),
        sub: fmtUsdSigned(t.beta_weighted_net_btc_usd),
        cls: signedClass(t.net_beta_to_btc),
      },
    ];
    const meta = [
      { label: "Underlyings", value: String(t.n_underlyings ?? 0) },
      {
        label: "Beta coverage",
        value: fmtPct(t.beta_coverage_gross_pct, 0),
        sub: "% gross w/ OLS β",
        cls: (t.beta_coverage_gross_pct ?? 0) >= 0.7 ? "pos" : "neg",
      },
      {
        label: "BTC β coverage",
        value: fmtPct(t.btc_beta_coverage_gross_pct, 0),
        sub: `${t.n_btc_beta_names ?? 0} names`,
        cls: (t.btc_beta_coverage_gross_pct ?? 0) >= 0.5 ? "pos" : "neg",
      },
    ];
    els.factorSummary.innerHTML =
      headline
        .map(
          (it) => `
        <div class="stat stat-${it.cls || "neutral"}">
          <div class="label">${it.label}</div>
          <div class="value ${it.cls || ""}">${it.value}</div>
          ${it.sub ? `<div class="sub">${it.sub}</div>` : ""}
        </div>`
        )
        .join("") +
      meta
        .map(
          (it) => `
        <div class="stat stat-${it.cls || "neutral"}">
          <div class="label">${it.label}</div>
          <div class="value ${it.cls || ""}">${it.value}</div>
          ${it.sub ? `<div class="sub">${it.sub}</div>` : ""}
        </div>`
        )
        .join("");

    const byBucket = panel.by_bucket || [];
    if (els.factorByBucket) {
      if (!byBucket.length) {
        els.factorByBucket.innerHTML = `<p class="dim">No bucket beta breakdown available.</p>`;
      } else {
        const bucketSum = t.by_bucket_beta_weighted_net_usd;
        const reconciles = t.by_bucket_reconciles;
        const reconNote =
          reconciles === false
            ? `<p class="callout warn small">Bucket β-wtd net sum (${fmtUsdSigned(
                bucketSum
              )}) differs from book total (${fmtUsdSigned(
                t.beta_weighted_net_usd
              )}) — bucket CSVs can double-count legs vs underlying rollup. Applies to all factor columns.</p>`
            : "";
        function bucketBetaCell(v) {
          if (v == null || Number.isNaN(Number(v))) return "-";
          return `<span class="${signedClass(v)}">${Number(v).toFixed(2)}x</span>`;
        }
        els.factorByBucket.innerHTML = `${reconNote}<table class="tight"><thead><tr>
          <th>Bucket</th><th>Net $</th>
          <th>Net β SPY</th><th>Net β QQQ</th><th>Net β IWM</th><th>Net β BTC</th>
        </tr></thead><tbody>${byBucket
          .map(
            (r) => `<tr>
              <td><strong>${safeText(r.bucket_label || r.bucket)}</strong></td>
              <td class="num ${signedClass(r.net_notional_usd)}">${fmtUsdSigned(
              r.net_notional_usd
            )}</td>
              <td class="num">${bucketBetaCell(r.net_beta_to_spy)}</td>
              <td class="num">${bucketBetaCell(r.net_beta_to_qqq)}</td>
              <td class="num">${bucketBetaCell(r.net_beta_to_iwm)}</td>
              <td class="num">${bucketBetaCell(r.net_beta_to_btc)}</td>
            </tr>`
          )
          .join("")}<tr class="dim">
            <td><strong>Book total</strong></td>
            <td class="num ${signedClass(t.net_notional_usd)}">${fmtUsdSigned(
            t.net_notional_usd
          )}</td>
            <td class="num">${bucketBetaCell(t.net_beta_to_spy)}</td>
            <td class="num">${bucketBetaCell(t.net_beta_to_qqq)}</td>
            <td class="num">${bucketBetaCell(t.net_beta_to_iwm)}</td>
            <td class="num">${bucketBetaCell(t.net_beta_to_btc)}</td>
          </tr></tbody></table>`;
      }
    }

    function fmtBeta(v, source) {
      if (v == null || Number.isNaN(Number(v))) return "-";
      const dim = source === "default" || source === "default_fallback" ? "dim" : "";
      return `<span class="${dim}">${Number(v).toFixed(2)}</span>`;
    }

    function betaProvenancePill(row) {
      const src = row.beta_source || "";
      let label = src;
      let cls = "pill-warn";
      if (src === "computed") {
        label = "computed";
        cls = "pill-ok";
      } else if (src === "shrunk") {
        label = "shrunk";
        cls = "pill-warn";
      } else if (src === "curated_fallback" || src === "curated") {
        label = "fallback";
        cls = "pill-warn";
      } else if (src === "default_fallback" || src === "default") {
        label = "fallback";
        cls = "pill-crit";
      } else if (!src) {
        return "";
      }
      const w = row.shrinkage_weight;
      const tipParts = [src];
      if (row.beta_n_obs) tipParts.push(`n=${row.beta_n_obs}`);
      if (row.beta_r2 != null) tipParts.push(`R²=${Number(row.beta_r2).toFixed(2)}`);
      if (w != null) tipParts.push(`w(OLS)=${Number(w).toFixed(2)}`);
      if (row.prior_used_spy != null) {
        tipParts.push(`prior=${Number(row.prior_used_spy).toFixed(2)}`);
      }
      if (row.beta_to_spy_raw != null) {
        tipParts.push(`raw β_SPY=${Number(row.beta_to_spy_raw).toFixed(2)}`);
      }
      const tip = tipParts.join(" · ");
      return ` <span class="pill ${cls} small" title="${tip}">${label}</span>`;
    }

    function sectorProvenancePill(row) {
      const src = row.sector_source || "";
      if (!src) return "";
      let label = src;
      let cls = "pill-warn";
      if (src === "override") {
        cls = "pill-ok";
      } else if (src === "screener" || src === "vendor") {
        cls = "pill-ok";
      } else if (src === "heuristic") {
        cls = "pill-warn";
      } else if (src === "default") {
        cls = "pill-crit";
      }
      const conf = row.sector_confidence;
      const tipParts = [src];
      if (conf != null) tipParts.push(`conf=${Number(conf).toFixed(2)}`);
      if (row.instrument_class) tipParts.push(`inst=${row.instrument_class}`);
      const tip = tipParts.join(" · ");
      return ` <span class="pill ${cls} small" title="${tip}">${label}</span>`;
    }

    function rowTbl(rows) {
      return `<table class="tight"><thead><tr>
        <th>Underlying</th><th>Sector</th><th>β SPY</th><th>β QQQ</th><th>β IWM</th><th>β BTC</th><th>Net $</th><th>β-wtd net $</th>
      </tr></thead><tbody>${(rows || [])
        .map(
          (r) => `<tr>
            <td><strong>${safeText(r.underlying)}</strong>${betaProvenancePill(r)} <span class="dim small">${safeText(
            r.symbols,
            ""
          )}</span></td>
            <td>${safeText(r.sector)}${sectorProvenancePill(r)}${
            r.instrument_class
              ? ` <span class="dim small" title="instrument class">${safeText(r.instrument_class)}</span>`
              : ""
          }</td>
            <td class="num" title="${safeText(r.beta_source, "")}">${fmtBeta(
            r.beta_to_spy,
            r.beta_source
          )}</td>
            <td class="num">${fmtBeta(r.beta_to_qqq, r.beta_source)}</td>
            <td class="num">${fmtBeta(r.beta_to_iwm, r.beta_source)}</td>
            <td class="num">${fmtBeta(r.beta_to_btc, r.beta_source)}</td>
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
  }

  /* ---------------- Slide risk strips (Phase 1) ---------------- */
  function formatSlideHorizonTooltip(h) {
    if (!h) return "";
    const parts = [];
    if (h.horizon_shock_mode) parts.push(`Shock mode ${h.horizon_shock_mode}`);
    if (h.spx_shock_effective_pct != null) {
      parts.push(`Eff. ΔSPX ${fmtPct(h.spx_shock_effective_pct, 2)}`);
    }
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
      const spxMode = panel.horizon_shock_mode || "rms";
      els.slideRiskMeta.innerHTML = `<span class="dim small">T+0 = instantaneous &beta;&times;&Delta;SPX (linear). ${scenarioHorizons.join(
        " / "
      )} = horizon-scaled equity shock (<code>${safeText(spxMode)}</code>) + LETF decay/borrow. VIX panel: same decay/borrow stack at SPX 0% with VIX-stressed vol &amp; borrow.${betaCov} &middot; ${letf}/${total} LETF &middot; ${panel.n_names_with_vol || 0} with vol</span>`;
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
    const carryVal = panel.carry_validation || null;
    const carryValBanner =
      carryVal && carryVal.available && carryVal.warnings && carryVal.warnings.length
        ? `<div class="callout warn" role="status"><strong>Carry model validation:</strong> ${carryVal.warnings
            .map((w) => safeText(w))
            .join(" ")}${
            carryVal.realized_annualized_pct_nav != null
              ? ` Realized (63d ann.): ${fmtPct(carryVal.realized_annualized_pct_nav, 1)}.`
              : ""
          }</div>`
        : "";
    const vixIdx = indices.find((idx) => idx.index === "VIX");
    const vixMatrix = vixIdx?.vix_decay_matrix || null;
    const tailUntrusted =
      vixMatrix && vixMatrix.tail_scenarios_trusted === false
        ? `<div class="callout-soft" role="status"><strong>Tail scenarios:</strong> historical VIX analogs flagged — model carry diverges from recent realized P&amp;L. Use for ordering, not sizing.</div>`
        : "";
    const decayRef = spxIdx?.decay_reference || null;
    const decayBanner = decayRef
      ? `<div class="callout-soft" role="status"><strong>Expected 0&sigma; book carry (forecast vol 1.0&times;):</strong> ${formatDecayReference(
          decayRef
        )}</div>`
      : "";
    const stripsHtml = indices
      .map((idx) => {
        if (idx.strip_type === "vix_pts" || idx.strip_type === "vix_decay") {
          const vixMatrix = idx.vix_decay_matrix || null;
          if (!vixMatrix) {
            return `<div class="slide-strip"><p class="dim small">12M VIX decay projection unavailable.</p></div>`;
          }

          function renderVixMatrixTable(cells, title, subtitle) {
            if (!cells || !cells.length) return "";
            const header = cells
              .map((c) => `<th class="slide-shock">${safeText(c.label)}</th>`)
              .join("");
            const row = cells
              .map((cell) => {
                const tip = [
                  cell.sigma_annual_median != null
                    ? `σ med ${(Number(cell.sigma_annual_median) * 100).toFixed(0)}%`
                    : null,
                  cell.decay_pnl_pct_nav != null
                    ? `Decay ${fmtPct(cell.decay_pnl_pct_nav, 2)}`
                    : null,
                  cell.borrow_pnl_pct_nav != null
                    ? `Borrow ${fmtPct(cell.borrow_pnl_pct_nav, 2)}`
                    : null,
                  cell.delta_vs_current_pct_nav != null
                    ? `Δ vs current ${fmtPct(cell.delta_vs_current_pct_nav, 2)}`
                    : null,
                ]
                  .filter(Boolean)
                  .join(" · ");
                return `<td class="num ${signedClass(cell.total_pnl_pct_nav)} ${scenarioHeatClass(
                  cell.total_pnl_pct_nav
                )}" title="${safeText(tip)}">${fmtPct(cell.total_pnl_pct_nav, 1)}</td>`;
              })
              .join("");
            return `<div class="slide-strip" style="margin-top:12px;">
              <h4>${title}</h4>
              ${subtitle ? `<p class="dim small">${subtitle}</p>` : ""}
              <div class="slide-strip-scroll">
                <table class="tight slide-table"><thead><tr><th class="row-label">Horizon</th>${header}</tr></thead>
                <tbody><tr><th class="row-label">12M <span class="dim small">SPX 0%</span></th>${row}</tr></tbody></table>
              </div>
            </div>`;
          }

          const hl = vixMatrix.headline || {};
          const headlineStrip = `<div class="strip" style="margin-bottom:8px;">
            <div class="stat"><div class="label">VIX spot</div><div class="value">${hl.vix_spot_pts != null ? Number(hl.vix_spot_pts).toFixed(1) : "-"}</div></div>
            <div class="stat"><div class="label">VIX 9D / 3M</div><div class="value">${hl.vix9d_pts != null ? Number(hl.vix9d_pts).toFixed(1) : "-"} / ${hl.vix3m_pts != null ? Number(hl.vix3m_pts).toFixed(1) : "-"}</div><div class="sub">${safeText(hl.term_structure, "")}</div></div>
            <div class="stat"><div class="label">VVIX</div><div class="value">${hl.vvix_pts != null ? Number(hl.vvix_pts).toFixed(1) : "-"}</div></div>
            <div class="stat"><div class="label">σ book (med)</div><div class="value">${hl.sigma_book_median != null ? (Number(hl.sigma_book_median) * 100).toFixed(0) + "%" : "-"}</div></div>
            <div class="stat"><div class="label">12M net carry</div><div class="value ${signedClass(hl.carry_12m_pct_nav)}">${hl.carry_12m_pct_nav != null ? fmtPct(hl.carry_12m_pct_nav, 1) : "-"}</div><div class="sub">${hl.decay_12m_pct_nav != null && hl.borrow_12m_pct_nav != null ? `decay ${fmtPct(hl.decay_12m_pct_nav, 1)}, borrow ${fmtPct(hl.borrow_12m_pct_nav, 1)}` : ""}</div></div>
          </div>`;

          const estVer = vixMatrix.vol_vix_estimator_version || "unknown";
          const decomp = vixMatrix.variance_decomp_summary || {};
          const decayMeta = `<p class="dim small">${safeText(
            vixMatrix.description,
            "Expected 12M book carry at SPX 0% under VIX-shocked vol."
          )} Estimator: <strong>${safeText(estVer)}</strong> · ${vixMatrix.n_vol_betas_computed ?? "?"}/${Object.keys(vixMatrix.vol_vix_betas || {}).length} computed${
            vixMatrix.n_vol_betas_shrunk != null ? `, ${vixMatrix.n_vol_betas_shrunk} shrunk` : ""
          } · variance decomp ${decomp.n_decomp ?? 0}/${decomp.n_total ?? 0} names · VRP k=${decomp.vrp_factor ?? "?"}.</p>`;

          const sustainedTbl = renderVixMatrixTable(
            vixMatrix.cells,
            "Sustained VIX (parallel shifts)",
            "VIX shifts to new level and stays for 12M."
          );
          const spikeTbl = renderVixMatrixTable(
            vixMatrix.cells_spike_revert,
            "Spike & revert (parallel shifts)",
            "VIX jumps then mean-reverts (κ≈5, θ≈18)."
          );

          const hist = vixMatrix.historical_scenarios || [];
          const histTbl = hist.length
            ? `<div class="slide-strip" style="margin-top:12px;">
              <h4>Historical analog scenarios</h4>
              <table class="tight"><thead><tr>
                <th>Scenario</th><th>VIX peak</th><th>12M net</th><th>Δ vs current</th>
              </tr></thead><tbody>${hist
                .map(
                  (h) => `<tr>
                    <td><strong>${safeText(h.label)}</strong></td>
                    <td class="num">${h.vix_peak_pts != null ? Number(h.vix_peak_pts).toFixed(0) : "-"}</td>
                    <td class="num ${signedClass(h.total_pnl_pct_nav)}" title="${safeText(
                      [
                        h.decay_pnl_pct_nav != null ? `Decay ${fmtPct(h.decay_pnl_pct_nav, 2)}` : null,
                        h.borrow_pnl_pct_nav != null ? `Borrow ${fmtPct(h.borrow_pnl_pct_nav, 2)}` : null,
                      ]
                        .filter(Boolean)
                        .join(" · ")
                    )}">${fmtPct(h.total_pnl_pct_nav, 1)}</td>
                    <td class="num ${signedClass(h.delta_vs_current_pct_nav)}">${fmtPct(h.delta_vs_current_pct_nav, 1)}</td>
                  </tr>`
                )
                .join("")}</tbody></table>
            </div>`
            : "";

          const perName = vixMatrix.per_name_contributions || [];
          const perNameTbl = perName.length
            ? `<details style="margin-top:12px;"><summary><strong>Per-name vol sensitivity (top ${Math.min(perName.length, 25)})</strong></summary>
              <table class="tight"><thead><tr>
                <th>Underlying</th><th>β vol</th><th>σ base</th><th>σ @ VIX+20</th><th>12M net $</th><th>Decay $</th><th>Borrow $</th>
              </tr></thead><tbody>${perName
                .slice(0, 25)
                .map(
                  (r) => `<tr>
                    <td><strong>${safeText(r.underlying)}</strong> <span class="dim small">${safeText(r.symbols, "")}</span></td>
                    <td class="num">${r.beta_vol_vix == null ? "-" : Number(r.beta_vol_vix).toFixed(2)}</td>
                    <td class="num">${r.sigma_base == null ? "-" : (Number(r.sigma_base) * 100).toFixed(0) + "%"}</td>
                    <td class="num">${r.sigma_shocked_plus_20 == null ? "-" : (Number(r.sigma_shocked_plus_20) * 100).toFixed(0) + "%"}</td>
                    <td class="num ${signedClass(r.total_pnl_usd)}">${fmtUsdSigned(r.total_pnl_usd)}</td>
                    <td class="num ${signedClass(r.decay_pnl_usd)}">${fmtUsdSigned(r.decay_pnl_usd)}</td>
                    <td class="num ${signedClass(r.borrow_pnl_usd)}">${fmtUsdSigned(r.borrow_pnl_usd)}</td>
                  </tr>`
                )
                .join("")}</tbody></table></details>`
            : "";

          return `<div class="slide-strip">
            <div class="slide-strip-head"><h3>12M expected carry vs VIX (SPX 0%)</h3></div>
            ${headlineStrip}
            ${decayMeta}
            ${sustainedTbl}
            ${spikeTbl}
            ${histTbl}
            ${perNameTbl}
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
        const modeledHorizons = [
          ...scenarioHorizons,
          ...(idx.spx_shock_config?.show_terminal_12m_row ? ["12M-terminal"] : []),
        ];
        const horizonRows = modeledHorizons
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
            const rowLabel =
              hk === "12M-terminal"
                ? "12M terminal"
                : hk;
            const modeHint =
              hk === "12M-terminal"
                ? "full ΔSPX"
                : idx.horizon_shock_mode
                  ? idx.horizon_shock_mode
                  : "modeled";
            return `<tr><th class="row-label">${safeText(rowLabel)} <span class="dim small">${safeText(modeHint)}</span></th>${cells}</tr>`;
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
        const betaHead =
          idx.net_beta_to_spy != null
            ? ` &middot; portfolio &beta; ${Number(idx.net_beta_to_spy).toFixed(2)}`
            : "";
        const histSpx = idx.historical_spx_scenarios || [];
        const histSpxTbl = histSpx.length
          ? `<div class="slide-strip" style="margin-top:12px;">
              <h4>Historical SPX path scenarios (12M)</h4>
              <p class="dim small">${safeText(idx.description, "")}</p>
              <table class="tight"><thead><tr>
                <th>Scenario</th><th>SPX peak</th><th>12M total</th><th>Beta</th><th>Decay</th><th>Borrow</th>
              </tr></thead><tbody>${histSpx
                .map(
                  (h) => `<tr>
                    <td><strong>${safeText(h.label)}</strong></td>
                    <td class="num">${h.spx_peak_pct != null ? fmtPct(h.spx_peak_pct, 1) : "-"}</td>
                    <td class="num ${signedClass(h.total_pnl_pct_nav)}">${fmtPct(h.total_pnl_pct_nav, 1)}</td>
                    <td class="num ${signedClass(h.beta_pnl_pct_nav)}">${h.beta_pnl_pct_nav != null ? fmtPct(h.beta_pnl_pct_nav, 1) : "-"}</td>
                    <td class="num ${signedClass(h.decay_pnl_pct_nav)}">${fmtPct(h.decay_pnl_pct_nav, 1)}</td>
                    <td class="num ${signedClass(h.borrow_pnl_pct_nav)}">${fmtPct(h.borrow_pnl_pct_nav, 1)}</td>
                  </tr>`
                )
                .join("")}</tbody></table>
              <p class="dim small">Daily paths (${histSpx[0]?.path_steps ?? 252} steps), path-realized vol, VIX-linked borrow. Total is compounded; Beta+Decay+Borrow are linear attribution.</p>
            </div>`
          : "";
        return `<div class="slide-strip">
          <div class="slide-strip-head">
            <h3>${safeText(idx.index)} <span class="dim small">(${Math.round((idx.coverage_pct || 0) * 100)}% coverage, ${idx.n_names_covered}/${idx.n_names_total} names${betaHead})</span></h3>
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
          ${histSpxTbl}
        </div>`;
      })
      .join("");
    els.slideRiskContent.innerHTML = `${worstBanner}${decayBanner}${carryValBanner}${tailUntrusted}${stripsHtml || `<p class="dim">No index strips available.</p>`}`;
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

  function renderSleeveGroups(groups, nav) {
    if (!groups || !groups.length) return "";
    const rows = groups
      .map(
        (g) => `<tr>
          <td><strong>${safeText(g.label)}</strong></td>
          <td class="num">${g.gross_usd == null ? "unavailable" : fmtUsd(g.gross_usd)}</td>
          <td class="num ${signedClass(g.net_usd)}">${g.net_usd == null ? "unavailable" : fmtUsd(g.net_usd)}</td>
          <td class="num ${signedClass(g.pnl_usd)}">${fmtUsdSigned(g.pnl_usd)}</td>
          <td class="num">${fmtPct(g.pnl_pct_nav, 2)}</td>
        </tr>
        <tr><td colspan="5" class="dim small">${safeText(g.exposure_note, "")}</td></tr>`
      )
      .join("");
    return `
      <h3>EOD sleeve groups (matches PnL email)</h3>
      <table class="tight"><thead><tr>
        <th>Group</th><th>Gross $</th><th>Net $</th><th>P&amp;L YTD</th><th>% NAV</th>
      </tr></thead><tbody>${rows}</tbody></table>`;
  }

  function renderBucketSleevePanel(panel, book, groups, nav) {
    if (!els.bucketSleevePanel) return;
    const rows = panel?.rows || [];
    const marginSum = rows.reduce(
      (m, r) => m + (Number(r.margin_req_usd) || 0),
      0
    );
    const marginUtil = nav && nav > 0 ? marginSum / nav : null;
    const exposureAvailable = book?.sleeve_attribution_available !== false;
    const capitalAvailable = panel?.capital_available !== false;
    const unavailableCell = (title) =>
      `<td class="num dim" title="${safeText(
        title,
        "unavailable"
      )}">unavailable</td>`;

    els.bucketSleevePanel.innerHTML = `
      <div class="table-scroll">
      <table class="tight">
        <thead><tr>
          <th rowspan="2">Bucket</th>
          <th colspan="4">Exposure (β-normalized)</th>
          <th colspan="3">Deployed capital (snapshot)</th>
          <th colspan="5">Returns (YTD / avg capital)</th>
        </tr>
        <tr>
          <th>Gross $</th><th>Net $</th><th>Target %</th><th>Drift</th>
          <th>Net cap</th><th>Gross cap</th><th>Margin</th>
          <th>P&amp;L YTD</th><th>ROC</th><th>ROG</th><th>ROM</th><th>Status</th>
        </tr></thead>
        <tbody>${rows
          .map((r) => {
            const trCls =
              r.drift_status === "hard"
                ? "row-hard"
                : r.drift_status === "warn"
                ? "row-warn"
                : "";
            const grossCell = exposureAvailable
              ? `<td class="num">${fmtUsd(r.exposure_gross_usd)}</td>`
              : unavailableCell(book?.sleeve_attribution_reason);
            const netCell = exposureAvailable
              ? `<td class="num ${signedClass(r.exposure_net_usd)}">${fmtUsd(r.exposure_net_usd)}</td>`
              : unavailableCell(book?.sleeve_attribution_reason);
            const targetCell = exposureAvailable
              ? `<td class="num">${
                  r.target_weight == null ? "-" : fmtPct(r.target_weight, 0)
                }</td>`
              : unavailableCell(book?.sleeve_attribution_reason);
            const driftCell = exposureAvailable
              ? `<td class="num">${fmtPp(r.drift_pp, 1)}</td>`
              : unavailableCell(book?.sleeve_attribution_reason);
            const netCapCell = capitalAvailable
              ? `<td class="num ${signedClass(r.net_capital_usd)}">${fmtUsd(r.net_capital_usd)}</td>`
              : unavailableCell(panel?.capital_reason);
            const grossCapCell = capitalAvailable
              ? `<td class="num">${fmtUsd(r.gross_capital_usd)}</td>`
              : unavailableCell(panel?.capital_reason);
            const marginCell = capitalAvailable
              ? `<td class="num">${fmtUsd(r.margin_req_usd)}</td>`
              : unavailableCell(panel?.capital_reason);
            const rocCell = capitalAvailable
              ? `<td class="num">${
                  r.roc_on_net_capital == null ? "-" : fmtPct(r.roc_on_net_capital, 2)
                }</td>`
              : unavailableCell(panel?.capital_reason);
            const rogCell = capitalAvailable
              ? `<td class="num">${
                  r.rog_on_gross_capital == null ? "-" : fmtPct(r.rog_on_gross_capital, 2)
                }</td>`
              : unavailableCell(panel?.capital_reason);
            const romCell = capitalAvailable
              ? `<td class="num">${
                  r.rom_on_margin_req == null ? "-" : fmtPct(r.rom_on_margin_req, 2)
                }</td>`
              : unavailableCell(panel?.capital_reason);
            const statusCell = exposureAvailable
              ? `<td>${statusPill(r.drift_status)}</td>`
              : `<td>${statusPill("unknown", "n/a")}</td>`;
            return `<tr class="${trCls}">
          <td><strong>${safeText(r.bucket_label || r.bucket)}</strong></td>
          ${grossCell}
          ${netCell}
          ${targetCell}
          ${driftCell}
          ${netCapCell}
          ${grossCapCell}
          ${marginCell}
          <td class="num ${signedClass(r.pnl_usd)}">${fmtUsdSigned(r.pnl_usd)}</td>
          ${rocCell}
          ${rogCell}
          ${romCell}
          ${statusCell}
        </tr>`;
          })
          .join("")}</tbody>
      </table>
      </div>
      <p class="dim small">${safeText(panel?.exposure_note, "")}</p>
      <p class="dim small">${safeText(panel?.return_denominator_note, "")}</p>
      ${
        capitalAvailable
          ? `<p class="dim small">Source: ${safeText(panel?.source, "totals.json")}. Scoped to etf_screened_today universe (same as EOD email).</p>`
          : `<p class="dim small callout">${safeText(
              panel?.capital_reason,
              "Capital snapshot not available."
            )}</p>`
      }
      ${
        capitalAvailable && marginUtil != null
          ? `<p class="dim small">Margin utilization: ${fmtUsd(
              marginSum
            )} margin req / ${fmtUsd(nav)} NAV = <strong>${fmtPct(
              marginUtil,
              1
            )}</strong> of NAV.</p>`
          : ""
      }
      ${renderSleeveGroups(groups, nav)}`;

    const banner = document.getElementById("sleeve-banner");
    if (banner) {
      if (!exposureAvailable) {
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
    const legRows = bucket.exposure_leg_rows || [];
    const legTbl =
      bucketKey === "bucket_4" && legRows.length
        ? legRows
            .slice(0, 60)
            .map(
              (r) => `<tr>
          <td><strong>${safeText(r.underlying)}</strong></td>
          <td>${safeText(r.symbol)}</td>
          <td class="dim">${safeText(r.leg_type)}</td>
          <td class="num ${signedClass(r.net_notional_usd)}">${fmtUsdSigned(
                r.net_notional_usd
              )}</td>
          <td class="num">${fmtUsd(r.gross_notional_usd)}</td>
        </tr>`
            )
            .join("")
        : "";
    const legSection =
      bucketKey === "bucket_4" && legRows.length
        ? `<h3>Exposure legs (underlying + inverse ETF, top ${Math.min(
            60,
            legRows.length
          )} of ${bucket.n_exposure_leg_rows || legRows.length})</h3>
      <p class="dim">Structural B4 short underlying is plan-implied when IBKR nets spot with B1/B2.</p>
      <table class="tight"><thead><tr>
        <th>Underlying</th><th>Symbol</th><th>Leg</th><th>Net $</th><th>Gross $</th>
      </tr></thead><tbody>${legTbl}</tbody></table>`
        : "";
    const hdr = bucket.exposure_header || {};
    const hdrBlock = hdr.attribution_net_usd != null
      ? `<div class="strip" style="margin-bottom:10px;">
          <div class="stat"><div class="label">Sleeve net (totals.json)</div><div class="value ${signedClass(hdr.attribution_net_usd)}">${fmtUsd(hdr.attribution_net_usd)}</div></div>
          <div class="stat"><div class="label">Sleeve gross (totals.json)</div><div class="value">${fmtUsd(hdr.attribution_gross_usd)}</div></div>
          ${
            bucketKey === "bucket_4" && hdr.pair_view_gross_usd != null
              ? `<div class="stat"><div class="label">Pair CSV gross (detail)</div><div class="value">${fmtUsd(hdr.pair_view_gross_usd)}</div><div class="sub dim">Pair view — not used for sleeve reconciliation</div></div>`
              : ""
          }
        </div>
        <p class="dim small">${safeText(hdr.source, "")}</p>`
      : "";
    els.bucketContent.innerHTML = `
      ${hdrBlock}
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
      ${legSection}
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

  /* ----------------------- Dashboard tabs ----------------------- */
  function dashParseHash() {
    const raw = (location.hash || "").replace(/^#/, "");
    if (!raw) return { tab: "overview" };
    if (raw.startsWith("b4sim=")) return { tab: "risk" };
    try {
      const params = new URLSearchParams(raw);
      const tab = params.get("tab");
      if (tab && DASH_TAB_IDS.includes(tab)) return { tab };
      if (params.has("b4sim")) return { tab: "risk" };
    } catch (_e) { /* ignore */ }
    return { tab: "overview" };
  }

  function dashWriteHash(tabId, { keepB4 = true } = {}) {
    try {
      const params = new URLSearchParams();
      if (tabId && tabId !== "overview") params.set("tab", tabId);
      if (tabId === "risk" && keepB4) {
        const existing = b4ParseHash();
        if (existing) params.set("b4sim", JSON.stringify(existing));
      }
      const next = params.toString();
      location.hash = next || "";
    } catch (_e) { /* ignore */ }
  }

  function renderTabPanel(tabId, snap) {
    if (!snap || _tabRendered.has(tabId)) return;
    _tabRendered.add(tabId);
    const steps = {
      pnl: [
        () => renderPerformance(snap),
        () => renderPnlPanel(snap),
        () => renderMovers(snap.movers_panel || {}),
      ],
      risk: [
        () => renderSlideRisk(snap.slide_risk_panel || {}),
        () => renderBucket4Sim(snap.bucket4_risk_sim || null),
        () => renderBucket5Backtest(snap.bucket5_backtest || null),
        () => renderBorrowShock(snap.borrow_shock_panel || {}, snap.nav_usd),
      ],
      book: [
        () => renderConcentration(snap.concentration_panel || {}),
        () => renderFactor(snap.factor_panel || {}),
        () =>
          renderBucketSleevePanel(
            snap.bucket_sleeve_panel || {},
            snap.book || {},
            snap.display_sleeve_groups || [],
            snap.nav_usd
          ),
        () => bindTabs(snap),
        () => renderSharedUnderlying(snap.shared_underlying_panel || {}),
        () => renderBorrow(snap.borrow_panel || {}),
        () => renderSqueeze((snap.borrow_panel || {}).squeeze_rows || []),
      ],
      data: [
        () => {
          if (els.rawTotals) {
            els.rawTotals.textContent = JSON.stringify(snap.raw_totals || {}, null, 2);
          }
        },
      ],
    };
    for (const step of steps[tabId] || []) {
      try {
        step();
      } catch (e) {
        console.error(`render tab ${tabId} failed:`, e);
        throw new Error(`Dashboard render failed (${tabId}): ${e.message || e}`);
      }
    }
    enableSortableTables();
  }

  function switchDashboardTab(tabId, { updateHash = true, snap = null } = {}) {
    const tab = DASH_TAB_IDS.includes(tabId) ? tabId : "overview";
    document.querySelectorAll(".dash-tab-panel").forEach((panel) => {
      const active = panel.dataset.dashTab === tab;
      panel.hidden = !active;
    });
    if (els.dashboardTabs) {
      els.dashboardTabs.querySelectorAll("button.tab").forEach((btn) => {
        btn.classList.toggle("active", btn.dataset.dashTab === tab);
      });
    }
    const s = snap || _lastSnap;
    if (s) renderTabPanel(tab, s);
    if (updateHash) dashWriteHash(tab);
  }

  function bindDashboardTabs() {
    if (!els.dashboardTabs) return;
    els.dashboardTabs.querySelectorAll("button.tab").forEach((btn) => {
      btn.addEventListener("click", () => {
        switchDashboardTab(btn.dataset.dashTab || "overview");
      });
    });
    document.querySelectorAll(".dash-subnav a").forEach((link) => {
      link.addEventListener("click", (ev) => {
        const href = link.getAttribute("href") || "";
        if (!href.startsWith("#")) return;
        const target = document.querySelector(href);
        if (!target) return;
        ev.preventDefault();
        target.scrollIntoView({ behavior: "smooth", block: "start" });
      });
    });
    window.addEventListener("hashchange", () => {
      if (!_lastSnap) return;
      const { tab } = dashParseHash();
      switchDashboardTab(tab, { updateHash: false, snap: _lastSnap });
    });
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
    enableSortableTables();
  }
  function showLogin() {
    els.loginPanel.hidden = false;
    els.dashboard.hidden = true;
    els.logoutBtn.hidden = true;
    if (els.authUserLabel) els.authUserLabel.hidden = true;
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

  /* ===================== Bucket 4 risk simulator ===================== */
  // Seeded PRNG (mulberry32) so a given control set is reproducible.
  function b4Mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0;
      a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function b4Gauss(rng) {
    let u = 0;
    while (u <= 1e-12) u = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
  }
  // Marsaglia-Tsang gamma sampler (shape > 0) -> chi-square via Gamma(df/2, 2).
  function b4Gamma(rng, shape) {
    if (shape < 1) return b4Gamma(rng, shape + 1) * Math.pow(rng(), 1 / shape);
    const d = shape - 1 / 3;
    const c = 1 / Math.sqrt(9 * d);
    while (true) {
      let x, v;
      do {
        x = b4Gauss(rng);
        v = 1 + c * x;
      } while (v <= 0);
      v = v * v * v;
      const u = rng();
      if (u < 1 - 0.0331 * x * x * x * x) return d * v;
      if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
    }
  }
  function b4Pctl(sorted, p) {
    if (!sorted.length) return NaN;
    const idx = Math.min(sorted.length - 1, Math.max(0, Math.floor((p / 100) * (sorted.length - 1))));
    return sorted[idx];
  }
  function b4RefDd(sim, dist) {
    const ref = sim.reference_mc || {};
    const key = dist === "boot" ? "block_bootstrap" : dist === "t" ? "student_t" : "laplace";
    const r = ref[key] || {};
    return { p95: r.dd_p95, p99: r.dd_p99 };
  }
  const B4_RET_FLOOR = -0.95;
  const B4_RET_CAP = 0.95;
  function b4ClipDailyReturn(r) {
    const x = Number(r) || 0;
    return Math.max(B4_RET_FLOOR, Math.min(B4_RET_CAP, x));
  }
  function b4BuildPortReturns(sim, excludedEtfs) {
    const pairs = (sim.pair_returns || []).filter((p) => p.in_book);
    if (!pairs || !pairs.length) return sim.port_daily_returns || [];
    const ex = new Set(excludedEtfs || []);
    let active = pairs.filter((p) => !ex.has(p.etf));
    if (!active.length) active = pairs.filter((p) => p.in_book) || pairs.slice();
    // Default proposed book: use precomputed series (already per-pair clipped in Python).
    if (
      !ex.size &&
      active.length &&
      active.every((p) => p.in_book) &&
      active.length === (sim.n_pairs_in_book ?? active.length) &&
      sim.port_daily_returns?.length
    ) {
      return sim.port_daily_returns;
    }
    const grossOf = (p) => Number(p.gross_usd ?? p.weight ?? 0);
    let wsum = active.reduce((s, p) => s + grossOf(p), 0);
    if (wsum <= 0) return sim.port_daily_returns || [];
    const n = active[0].returns.length;
    const out = new Float64Array(n);
    for (const p of active) {
      const w = grossOf(p) / wsum;
      for (let i = 0; i < n; i++) out[i] += w * b4ClipDailyReturn(p.returns[i]);
    }
    return Array.from(out);
  }
  // Monte-Carlo: max-drawdown + terminal-return (+ optional equity fan percentiles).
  function b4RunSim(sim, o) {
    const rng = b4Mulberry32(o.seed >>> 0);
    const H = o.horizon | 0;
    const N = o.nSims | 0;
    const k = o.volMult;
    const dist = o.dist;
    const floor = B4_RET_FLOOR;
    const dds = new Float64Array(N);
    const term = new Float64Array(N);
    const data = o.portReturns || sim.port_daily_returns || [];
    const m = data.length;
    const mean = sim.mean_daily || 0;
    const L = o.blockLen || (sim.reference_mc && sim.reference_mc.block_len) || 10;
    const tdf = sim.fit_student_t.df;
    const tloc = sim.fit_student_t.loc;
    const tscale = sim.fit_student_t.scale * k;
    const lloc = sim.fit_laplace.loc;
    const lscale = sim.fit_laplace.scale * k;
    const fanBands = o.fanBands ? Array.from({ length: H + 1 }, () => []) : null;
    for (let i = 0; i < N; i++) {
      let eq = 1, peak = 1, mdd = 0;
      if (fanBands) fanBands[0].push(1);
      if (dist === "boot") {
        let filled = 0;
        while (filled < H) {
          const s = Math.floor(rng() * m);
          for (let b = 0; b < L && filled < H; b++, filled++) {
            let r = data[(s + b) % m];
            r = mean + k * (r - mean);
            if (r < floor) r = floor;
            eq *= 1 + r;
            if (eq > peak) peak = eq;
            const dd = eq / peak - 1;
            if (dd < mdd) mdd = dd;
            if (fanBands) fanBands[filled + 1].push(eq);
          }
        }
      } else {
        for (let s = 0; s < H; s++) {
          let r;
          if (dist === "t") {
            const chi2 = b4Gamma(rng, tdf / 2) * 2;
            r = tloc + tscale * (b4Gauss(rng) / Math.sqrt(chi2 / tdf));
          } else {
            const u = rng() - 0.5;
            r = lloc - lscale * Math.sign(u) * Math.log(1 - 2 * Math.abs(u));
          }
          if (r < floor) r = floor;
          eq *= 1 + r;
          if (eq > peak) peak = eq;
          const dd = eq / peak - 1;
          if (dd < mdd) mdd = dd;
          if (fanBands) fanBands[s + 1].push(eq);
        }
      }
      dds[i] = -mdd;
      term[i] = eq - 1;
    }
    let fan = null;
    if (fanBands) {
      fan = fanBands.map((arr) => {
        const s = Array.from(arr).sort((a, b) => a - b);
        return {
          p5: b4Pctl(s, 5),
          p25: b4Pctl(s, 25),
          p50: b4Pctl(s, 50),
          p75: b4Pctl(s, 75),
          p95: b4Pctl(s, 95),
        };
      });
    }
    return { dds, term, fan };
  }
  function b4Histogram(dds, p95, p99, refP95, refP99) {
    const W = 1000, Hh = 220, pad = 24;
    const nbins = 40, maxX = 1.0;
    const bins = new Array(nbins).fill(0);
    let tail90 = 0;
    for (let i = 0; i < dds.length; i++) {
      if (dds[i] >= 0.9) tail90++;
      let b = Math.floor((dds[i] / maxX) * nbins);
      if (b < 0) b = 0;
      if (b >= nbins) b = nbins - 1;
      bins[b]++;
    }
    const peak = Math.max(1, ...bins);
    const tailPct = dds.length ? (100 * tail90) / dds.length : 0;
    const bw = (W - 2 * pad) / nbins;
    let bars = "";
    for (let b = 0; b < nbins; b++) {
      const h = (bins[b] / peak) * (Hh - 2 * pad);
      const x = pad + b * bw;
      const y = Hh - pad - h;
      bars += `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${(bw - 1).toFixed(1)}" height="${h.toFixed(1)}" fill="var(--accent)" opacity="0.75"></rect>`;
    }
    const mk = (val, color, label, dash) => {
      if (val == null || !isFinite(val)) return "";
      const x = pad + (Math.min(val, maxX) / maxX) * (W - 2 * pad);
      const dashAttr = dash ? ' stroke-dasharray="6 4"' : "";
      return `<line x1="${x.toFixed(1)}" y1="${pad - 6}" x2="${x.toFixed(1)}" y2="${Hh - pad}" stroke="${color}" stroke-width="2"${dashAttr}></line>` +
        (label ? `<text x="${x.toFixed(1)}" y="${pad - 9}" fill="${color}" font-size="13" text-anchor="middle">${label}</text>` : "");
    };
    let axis = "";
    for (let g = 0; g <= 5; g++) {
      const frac = g / 5;
      const x = pad + frac * (W - 2 * pad);
      axis += `<text x="${x.toFixed(1)}" y="${Hh - pad + 16}" fill="var(--text-muted)" font-size="12" text-anchor="middle">${Math.round(frac * 100)}%</text>`;
    }
    const tailNote =
      tail90 > 0
        ? `<text x="${W - pad}" y="${pad + 12}" fill="var(--neg)" font-size="11" text-anchor="end">${tail90} sims (${tailPct.toFixed(1)}%) ≥90% DD</text>`
        : "";
    return `<svg viewBox="0 0 ${W} ${Hh}" preserveAspectRatio="none" style="width:100%;height:220px;display:block">` +
      `<line x1="${pad}" y1="${Hh - pad}" x2="${W - pad}" y2="${Hh - pad}" stroke="var(--border-strong)" stroke-width="1"></line>` +
      bars +
      tailNote +
      mk(refP95, "#64748b", "ref p95", true) +
      mk(refP99, "#94a3b8", "ref p99", true) +
      mk(p95, "#d08b1f", "live p95", false) +
      mk(p99, "var(--neg)", "live p99", false) +
      axis +
      `</svg>`;
  }
  function dashNiceTicks(minV, maxV, n) {
    if (!isFinite(minV) || !isFinite(maxV) || minV === maxV) return [minV];
    const span = maxV - minV;
    const raw = span / Math.max(2, n - 1);
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = Math.ceil(raw / mag) * mag;
    const start = Math.floor(minV / step) * step;
    const out = [];
    for (let v = start; v <= maxV + step * 0.01; v += step) out.push(v);
    return out;
  }
  function dashSvgYAxis(pad, Hh, ticks, yMap, fmt) {
    return ticks.map((v) => {
      const y = yMap(v);
      return `<line x1="${pad - 4}" y1="${y}" x2="${pad}" y2="${y}" stroke="#94a3b8" stroke-width="1"/>` +
        `<text x="${pad - 6}" y="${y + 4}" fill="#64748b" font-size="10" text-anchor="end">${fmt(v)}</text>`;
    }).join("");
  }
  function b4FanChart(fan, horizon) {
    if (!fan || !fan.length) return "";
    const W = 720, Hh = 240, padL = 44, padR = 12, padT = 34, padB = 28;
    const n = fan.length - 1;
    // Zoom y-axis to the central bulk (p25–p95) so a few wipeout paths don't flatten the chart.
    let yMin = Infinity, yMax = -Infinity;
    let p5Min = Infinity;
    for (const pt of fan) {
      yMin = Math.min(yMin, pt.p25, pt.p50);
      yMax = Math.max(yMax, pt.p75, pt.p95);
      p5Min = Math.min(p5Min, pt.p5);
    }
    yMin = Math.min(yMin, 0.88);
    yMax = Math.max(yMax, 1.12);
    const yPad = (yMax - yMin) * 0.1 || 0.05;
    yMin -= yPad;
    yMax += yPad;
    const xAt = (i) => padL + (i / n) * (W - padL - padR);
    const yAt = (v) => Hh - padB - ((v - yMin) / (yMax - yMin)) * (Hh - padT - padB);
    const yClip = (v) => Math.max(padT, Math.min(Hh - padB, yAt(v)));
    let outer = "";
    let inner = "";
    for (let i = 0; i < n; i++) {
      const x0 = xAt(i), x1 = xAt(i + 1);
      outer +=
        `<polygon points="${x0},${yClip(fan[i].p95)} ${x1},${yClip(fan[i + 1].p95)} ${x1},${yClip(fan[i + 1].p5)} ${x0},${yClip(fan[i].p5)}" fill="#0f766e" opacity="0.08"/>`;
      inner +=
        `<polygon points="${x0},${yAt(fan[i].p75)} ${x1},${yAt(fan[i + 1].p75)} ${x1},${yAt(fan[i + 1].p25)} ${x0},${yAt(fan[i].p25)}" fill="#0f766e" opacity="0.22"/>`;
    }
    let med = "";
    for (let i = 0; i < n; i++) med += `${xAt(i)},${yAt(fan[i].p50)} `;
    med += `${xAt(n)},${yAt(fan[n].p50)}`;
    const yTicks = dashNiceTicks(yMin, yMax, 5);
    const xMarks = [0, Math.round(horizon / 4), Math.round(horizon / 2), Math.round(3 * horizon / 4), horizon];
    const end = fan[fan.length - 1];
    const floorNote =
      p5Min < yMin + 0.02
        ? ` · p5 paths floor at ${p5Min.toFixed(2)} (below zoom)`
        : "";
    return `<svg viewBox="0 0 ${W} ${Hh}" width="100%" style="max-width:${W}px;display:block;margin-top:8px">` +
      `<text x="${padL}" y="14" fill="var(--text-muted)" font-size="11">Equity fan · ${horizon}d · dark band = p25–p75, faint = p5–p95, line = p50</text>` +
      `<text x="${padL}" y="28" fill="var(--text-muted)" font-size="10">Y-axis zoomed to central outcomes${floorNote}</text>` +
      dashSvgYAxis(padL, Hh, yTicks, yAt, (v) => v.toFixed(2)) +
      outer +
      inner +
      `<polyline fill="none" stroke="#0f766e" stroke-width="2.25" points="${med}"/>` +
      xMarks.map((d) => `<text x="${xAt(d)}" y="${Hh - 6}" fill="#64748b" font-size="10" text-anchor="middle">${d}d</text>`).join("") +
      `<text x="${W - padR}" y="${padT + 10}" fill="#0f766e" font-size="10" text-anchor="end">end p50=${end.p50.toFixed(2)} · p75=${end.p75.toFixed(2)} · p95=${end.p95.toFixed(2)}</text>` +
      `</svg>`;
  }
  function b4ParseHash() {
    const raw = (location.hash || "").replace(/^#/, "");
    if (!raw) return null;
    const parts = raw.split("&");
    for (const part of parts) {
      if (!part.startsWith("b4sim=")) continue;
      try {
        return JSON.parse(decodeURIComponent(part.slice(6)));
      } catch (_e) {
        return null;
      }
    }
    if (raw.startsWith("b4sim=")) {
      try {
        return JSON.parse(decodeURIComponent(raw.slice(6)));
      } catch (_e) {
        return null;
      }
    }
    try {
      const params = new URLSearchParams(raw);
      const b4 = params.get("b4sim");
      if (b4) return JSON.parse(b4);
    } catch (_e) { /* ignore */ }
    return null;
  }
  function b4WriteHash(o) {
    try {
      const params = new URLSearchParams();
      params.set("tab", "risk");
      params.set("b4sim", JSON.stringify(o));
      location.hash = params.toString();
    } catch (_e) { /* ignore */ }
  }
  function b4PairPickerHtml(sim) {
    const pairs = (Array.isArray(sim.pairs) ? sim.pairs : []).filter((p) => p.in_book);
    if (!pairs.length) return "";
    pairs.sort((a, b) => (Number(b.weight) || 0) - (Number(a.weight) || 0));
    const chips = pairs
      .map((p) => {
        const blkHtml = p.blacklisted ? `<span class="b4sim-pair-tag b4sim-pair-tag-blk">blk</span>` : "";
        return `<label class="b4sim-pair-chip" data-in-book="1">
            <input type="checkbox" class="b4sim-pair" data-etf="${safeText(p.etf)}" checked>
            <span class="b4sim-pair-main">${safeText(p.etf)}/${safeText(p.und)}</span>
            <span class="b4sim-pair-meta">
              <span class="b4sim-pair-wt">${fmtPct(p.weight, 0)}</span>${blkHtml}
            </span>
          </label>`;
      })
      .join("");
    return `<div class="b4sim-pairs-panel">
      <div class="b4sim-pairs-head">
        <div>
          <strong>Book pairs</strong>
          <div class="dim small">Uncheck a name to exclude it; remaining weights renormalize.</div>
        </div>
        <div class="b4sim-pairs-actions">
          <span id="b4sim-pairs-count" class="b4sim-pairs-count dim small"></span>
          <button type="button" class="btn btn-ghost" data-b4sim-pairs="all">All</button>
          <button type="button" class="btn btn-ghost" data-b4sim-pairs="none">None</button>
        </div>
      </div>
      <div class="b4sim-pairs-grid">${chips}</div>
    </div>`;
  }
  function renderBucket4Sim(sim) {
    const content = document.getElementById("b4sim-content");
    const meta = document.getElementById("b4sim-meta");
    if (!content) return;
    if (!sim || !sim.port_daily_returns || !sim.port_daily_returns.length) {
      if (meta) meta.innerHTML = statusPill("unknown", "risk-sim data not available");
      content.innerHTML =
        '<div class="callout dim">No B4 risk-sim dataset in this snapshot. Run <code>python scripts/build_bucket4_risk_sim.py</code> then rebuild the dashboard.</div>';
      return;
    }
    const cad = sim.cadence || {};
    if (meta) {
      const nBook = sim.n_pairs_in_book != null ? sim.n_pairs_in_book : (sim.pairs || []).filter((p) => p.in_book).length;
      meta.innerHTML =
        `<span class="dim">run ${safeText(sim.run_date)} · cadence tr=${safeText(cad.cadence_signal_col)} · base_days=${safeText(cad.base_days)} · ` +
        `proposed book ${nBook} pairs · ${sim.n_obs} obs since ${safeText(sim.window_start)}</span>`;
    }
    const bookPairs = (sim.pairs || []).filter((p) => p.in_book);
    const pairRowHtml = (p) =>
      `<tr><td>${p.etf}${p.blacklisted ? " <span class='dim'>(blk)</span>" : ""}</td><td>${p.und}</td>` +
      `<td class="num">${fmtPct(p.weight, 1)}</td><td class="num">${fmtUsd(p.proposed_gross_usd ?? p.gross_usd)}</td>` +
      `<td class="num">${p.n_days}</td><td class="num">${fmtPct(p.borrow, 1)}</td></tr>`;
    const bookRows = bookPairs.map(pairRowHtml).join("");
    const pairTogglePanel =
      sim.pair_returns && sim.pair_returns.length ? b4PairPickerHtml(sim) : "";

    const rz = sim.realized || {};
    content.innerHTML = `
      <div class="callout dim small">
        Resamples the <strong>proposed B4 book</strong> from <code>proposed_trades.csv</code> for run
        <strong>${safeText(sim.run_date)}</strong> (rows with gross &gt; 0). Uncheck pairs above to see
        concentration effects. <strong>Tune the controls to stress the book.</strong>
      </div>
      <div class="b4sim-controls strip" style="margin:10px 0;gap:14px;flex-wrap:wrap">
        <label class="b4sim-ctl">Distribution
          <select id="b4sim-dist">
            <option value="boot">Block bootstrap (empirical)</option>
            <option value="t">Student-t (fat tail)</option>
            <option value="laplace">Laplace (fat tail)</option>
          </select>
        </label>
        <label class="b4sim-ctl">Horizon (trading days)
          <input id="b4sim-horizon" type="number" min="20" max="756" step="1" value="252">
        </label>
        <label class="b4sim-ctl">Simulations
          <select id="b4sim-nsims">
            <option value="2000">2,000</option>
            <option value="4000" selected>4,000</option>
            <option value="8000">8,000</option>
            <option value="20000">20,000</option>
          </select>
        </label>
        <label class="b4sim-ctl">Vol shock &times;<span id="b4sim-volval">1.0</span>
          <input id="b4sim-vol" type="range" min="0.5" max="3" step="0.1" value="1.0">
        </label>
        <label class="b4sim-ctl">Block length
          <input id="b4sim-block" type="number" min="3" max="30" step="1" value="10">
        </label>
        <label class="b4sim-ctl">Seed
          <input id="b4sim-seed" type="number" min="1" max="99999999" step="1" value="1234567">
        </label>
        <button id="b4sim-run" class="btn btn-primary" type="button">Run</button>
      </div>
      ${pairTogglePanel}
      <div id="b4sim-stats" class="strip"></div>
      <p id="b4sim-takeaway" class="callout dim small" style="margin:6px 0 10px"></p>
      <div class="b4sim-controls strip" style="margin:0 0 10px;gap:14px;flex-wrap:wrap;align-items:flex-end">
        <label class="b4sim-ctl">Max tolerable drawdown (%)
          <input id="b4sim-tol" type="number" min="5" max="80" step="1" value="25" title="Your pain threshold for a bad year">
        </label>
        <div id="b4sim-size" class="callout dim small" style="flex:1;min-width:220px;margin:0;padding:8px 10px"></div>
      </div>
      <h3>1-year max-drawdown distribution</h3>
      <div id="b4sim-chart"></div>
      <div id="b4sim-fan"></div>
      <div class="two-col" style="margin-top:12px">
        <div>
          <h3>Proposed book (${bookPairs.length} pairs)</h3>
          <p class="dim small">From <code>data/runs/${safeText(sim.run_date)}/proposed_trades.csv</code> · gross ${fmtUsd(sim.proposed_book_gross_usd)}</p>
          <table class="tight"><thead><tr><th>ETF</th><th>Und</th><th class="num">Weight</th><th class="num">Gross $</th><th class="num">Days</th><th class="num">Borrow</th></tr></thead>
          <tbody>${bookRows || "<tr><td colspan='6' class='dim'>No proposed B4 rows with gross &gt; 0</td></tr>"}</tbody></table>
          <p class="dim small">Realized (sample): CAGR ${fmtPct(rz.cagr, 1)} · ann vol ${fmtPct(rz.ann_vol, 1)} · Sharpe ${safeText(rz.sharpe)} · maxDD ${fmtPct(rz.hist_maxdd, 1)}.</p>
        </div>
        <div>
          <h3>Reference tail (precomputed, ${(sim.reference_mc || {}).n_sims || "?"} sims)</h3>
          <table class="tight"><thead><tr><th>Method</th><th class="num">p95</th><th class="num">p99</th><th class="num">p99.9</th><th class="num">P(dd&gt;40%)</th></tr></thead>
          <tbody>${["block_bootstrap", "student_t", "laplace"]
            .map((mname) => {
              const r = (sim.reference_mc || {})[mname] || {};
              return `<tr><td>${mname.replace("_", " ")}</td><td class="num">${fmtPct(r.dd_p95, 1)}</td><td class="num">${fmtPct(r.dd_p99, 1)}</td><td class="num">${fmtPct(r.dd_p999, 1)}</td><td class="num">${fmtPct(r["P(dd>40)"], 1)}</td></tr>`;
            })
            .join("")}</tbody></table>
          <p class="dim small">Bootstrap is the headline (most pessimistic / preserves clustering). Even the base case carries a large tail &mdash; B4's drawdown is structural; size accordingly.</p>
        </div>
      </div>`;

    const distEl = document.getElementById("b4sim-dist");
    const horizonEl = document.getElementById("b4sim-horizon");
    const nsimsEl = document.getElementById("b4sim-nsims");
    const volEl = document.getElementById("b4sim-vol");
    const volValEl = document.getElementById("b4sim-volval");
    const blockEl = document.getElementById("b4sim-block");
    const seedEl = document.getElementById("b4sim-seed");
    const tolEl = document.getElementById("b4sim-tol");
    const sizeEl = document.getElementById("b4sim-size");
    const runEl = document.getElementById("b4sim-run");
    const statsEl = document.getElementById("b4sim-stats");
    const chartEl = document.getElementById("b4sim-chart");
    const fanEl = document.getElementById("b4sim-fan");

    const hashInit = b4ParseHash();
    if (hashInit) {
      if (hashInit.dist) distEl.value = hashInit.dist;
      if (hashInit.horizon) horizonEl.value = hashInit.horizon;
      if (hashInit.nSims) nsimsEl.value = String(hashInit.nSims);
      if (hashInit.volMult != null) volEl.value = hashInit.volMult;
      if (hashInit.blockLen) blockEl.value = hashInit.blockLen;
      if (hashInit.seed) seedEl.value = hashInit.seed;
      if (hashInit.tolPct) tolEl.value = hashInit.tolPct;
    } else {
      blockEl.value = (sim.reference_mc && sim.reference_mc.block_len) || 10;
    }

    function readOpts() {
      const excluded = [];
      document.querySelectorAll(".b4sim-pair").forEach((el) => {
        if (!el.checked) excluded.push(el.getAttribute("data-etf"));
      });
      return {
        dist: distEl.value,
        horizon: Math.max(20, Math.min(756, parseInt(horizonEl.value, 10) || 252)),
        nSims: parseInt(nsimsEl.value, 10) || 4000,
        volMult: parseFloat(volEl.value) || 1.0,
        blockLen: Math.max(3, Math.min(30, parseInt(blockEl.value, 10) || 10)),
        seed: parseInt(seedEl.value, 10) || 1234567,
        fanBands: true,
        tolPct: parseFloat(tolEl.value) || 25,
        portReturns: b4BuildPortReturns(sim, excluded),
      };
    }

    function updatePairCount() {
      const countEl = document.getElementById("b4sim-pairs-count");
      if (!countEl) return;
      const boxes = document.querySelectorAll(".b4sim-pair");
      const sel = document.querySelectorAll(".b4sim-pair:checked").length;
      countEl.textContent = `${sel} / ${boxes.length} included`;
    }

    function update() {
      updatePairCount();
      const o = readOpts();
      volValEl.textContent = o.volMult.toFixed(1);
      b4WriteHash(o);
      const t0 = performance.now();
      const ref = b4RefDd(sim, o.dist);
      const { dds, term, fan } = b4RunSim(sim, o);
      const ddSorted = Array.from(dds).sort((a, b) => a - b);
      const termSorted = Array.from(term).sort((a, b) => a - b);
      const p50 = b4Pctl(ddSorted, 50);
      const p95 = b4Pctl(ddSorted, 95);
      const p99 = b4Pctl(ddSorted, 99);
      const p999 = b4Pctl(ddSorted, 99.9);
      const pGt = (thr) => dds.reduce((acc, v) => acc + (v > thr ? 1 : 0), 0) / dds.length;
      // 1y downside: 95% VaR (5th pct of terminal return) and CVaR (mean of worst 5%)
      const var95 = b4Pctl(termSorted, 5);
      const nTail = Math.max(1, Math.floor(0.05 * termSorted.length));
      let cvar = 0;
      for (let i = 0; i < nTail; i++) cvar += termSorted[i];
      cvar /= nTail;
      const pLoss = term.reduce((acc, v) => acc + (v < 0 ? 1 : 0), 0) / term.length;
      const ms = (performance.now() - t0).toFixed(0);

      // Rule-based risk coloring: red at severe thresholds, amber at elevated.
      const ddClass = (v) => (v >= 0.4 ? "stat-neg" : v >= 0.25 ? "stat-warn" : "");
      const probClass = (v) => (v >= 0.25 ? "stat-neg" : v >= 0.1 ? "stat-warn" : "");
      const lossClass = (v) => (v <= -0.2 ? "stat-neg" : v <= -0.1 ? "stat-warn" : "");
      const stats = [
        { label: "Median maxDD", value: fmtPct(p50, 1), tip: "Typical (50th percentile) worst peak-to-trough drop over the horizon." },
        { label: "p95 maxDD", value: fmtPct(p95, 1), cls: ddClass(p95), tip: "A bad year: 5% of simulated paths draw down at least this much." },
        { label: "p99 maxDD", value: fmtPct(p99, 1), cls: ddClass(p99), tip: "A very bad year: only 1% of paths are worse than this." },
        { label: "p99.9 maxDD", value: fmtPct(p999, 1), cls: ddClass(p999), tip: "Tail-of-tail: 1-in-1000 path drawdown." },
        { label: "P(dd > 15%)", value: fmtPct(pGt(0.15), 0), cls: probClass(pGt(0.15)), tip: "Chance the book draws down more than 15% at some point in the horizon." },
        { label: "P(dd > 25%)", value: fmtPct(pGt(0.25), 0), cls: probClass(pGt(0.25)), tip: "Chance of a >25% drawdown over the horizon." },
        { label: "P(dd > 40%)", value: fmtPct(pGt(0.4), 0), cls: probClass(pGt(0.4)), tip: "Chance of a severe >40% drawdown over the horizon." },
        { label: "1y VaR 95%", value: fmtPct(var95, 1), sub: "5th pct annual return", cls: lossClass(var95), tip: "Value at Risk: the 5th-percentile 12-month return. 95% of years beat this." },
        { label: "1y CVaR 95%", value: fmtPct(cvar, 1), sub: "mean of worst 5%", cls: lossClass(cvar), tip: "Expected return in the worst 5% of years (average of the left tail)." },
        { label: "P(annual loss)", value: fmtPct(pLoss, 0), cls: probClass(pLoss), tip: "Chance the 12-month return is negative." },
      ];
      statsEl.innerHTML = stats
        .map((it) => {
          const valCls = it.cls === "stat-neg" ? "neg" : it.cls === "stat-warn" ? "warn" : "";
          return `<div class="stat ${it.cls || ""}" title="${it.tip || ""}"><div class="label">${it.label}</div><div class="value ${valCls}">${it.value}</div>${it.sub ? `<div class="sub">${it.sub}</div>` : ""}</div>`;
        })
        .join("");
      // Plain-language takeaway under the stat strip.
      const takeawayEl = document.getElementById("b4sim-takeaway");
      if (takeawayEl) {
        takeawayEl.innerHTML =
          `<strong>In plain English:</strong> in a typical year this book draws down about ` +
          `<strong>${fmtPct(p50, 0)}</strong>; a bad year (p95) is around <strong>${fmtPct(p95, 0)}</strong>, ` +
          `and there is a <strong>${fmtPct(pLoss, 0)}</strong> chance of losing money over 12 months. ` +
          `Size the sleeve so a p95 drawdown is survivable.`;
      }
      chartEl.innerHTML =
        b4Histogram(dds, p95, p99, ref.p95, ref.p99) +
        `<p class="dim small">${o.nSims.toLocaleString()} sims · ${distEl.options[distEl.selectedIndex].text} · horizon ${o.horizon}d · block ${o.blockLen} · vol &times;${o.volMult.toFixed(1)} · seed ${o.seed} · ${ms} ms. Dashed = precomputed reference tail; solid = this run.</p>`;
      if (fanEl) fanEl.innerHTML = b4FanChart(fan, o.horizon);
      if (sizeEl && p95 > 0) {
        const scale = Math.min(1, (o.tolPct / 100) / p95);
        sizeEl.innerHTML =
          `<strong>Size accordingly:</strong> if your max tolerable 1y drawdown is <strong>${o.tolPct}%</strong>, ` +
          `the live p95 (${fmtPct(p95, 0)}) implies scaling B4 gross to about <strong>${fmtPct(scale, 0)}</strong> of the proposed book ` +
          `(simple linear scaling; tails are nonlinear).`;
      }
    }

    distEl.addEventListener("change", update);
    horizonEl.addEventListener("change", update);
    nsimsEl.addEventListener("change", update);
    blockEl.addEventListener("change", update);
    seedEl.addEventListener("change", update);
    tolEl.addEventListener("change", update);
    volEl.addEventListener("input", () => {
      volValEl.textContent = (parseFloat(volEl.value) || 1).toFixed(1);
    });
    volEl.addEventListener("change", update);
    if (runEl) runEl.addEventListener("click", update);
    document.querySelectorAll(".b4sim-pair").forEach((el) => el.addEventListener("change", update));
    content.querySelectorAll("[data-b4sim-pairs]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const mode = btn.getAttribute("data-b4sim-pairs");
        document.querySelectorAll(".b4sim-pair").forEach((el) => {
          if (mode === "all") el.checked = true;
          else if (mode === "none") el.checked = false;
        });
        update();
      });
    });
    update();
  }

  /* ------------------- Phase 0/1/2 panels --------------------- */
  function renderFreshness(snap) {
    if (!els.freshnessBadge) return;
    const f = snap.freshness || {};
    const badge = els.freshnessBadge;
    badge.hidden = false;
    const age = f.data_age_days;
    if (f.is_latest && (age == null || age <= 1)) {
      badge.className = "pill pill-ok";
      badge.textContent = age === 0 || age == null ? "fresh · EOD ✓" : `${age}d old`;
      badge.title = "Snapshot matches latest accounting run.";
    } else if (!f.is_latest) {
      badge.className = "pill pill-hard";
      badge.textContent = "stale · fix CI";
      badge.title = `Newer accounting run: ${safeText(
        f.latest_accounting_run_date,
        "unknown"
      )}. Dashboard pipeline may have failed.`;
    } else {
      badge.className = "pill pill-warn";
      badge.textContent = `${age}d old`;
      badge.title = "Snapshot is more than a day behind its generation time.";
    }
  }

  function renderPerformance(snap) {
    const dd = snap.drawdown_panel || {};
    if (els.pnlStrip) {
      const items = [
        {
          label: "Current drawdown",
          value: dd.available ? fmtUsd(dd.current_drawdown_usd) : "-",
          sub: dd.available ? `${fmtPct(dd.current_drawdown_pct, 2)} from peak` : "no history",
          cls: signedClass(dd.current_drawdown_usd),
        },
        {
          label: "Max drawdown (YTD)",
          value: dd.available ? fmtUsd(dd.max_drawdown_usd) : "-",
          sub: dd.available
            ? `${fmtPct(dd.max_drawdown_pct, 2)} · ${safeText(dd.max_drawdown_date, "")}`
            : "no history",
          cls: "neg",
        },
        {
          label: "Peak equity",
          value: dd.available ? fmtUsd(dd.peak_equity_usd) : "-",
          sub: dd.available ? `base NAV ${fmtUsd(dd.base_nav_usd)}` : "",
        },
        {
          label: "Current equity",
          value: dd.available ? fmtUsd(dd.current_equity_usd) : "-",
          sub: dd.available ? `cum P&amp;L ${fmtUsdSigned(dd.current_cum_pnl_usd)}` : "",
          cls: signedClass(dd.current_cum_pnl_usd),
        },
      ];
      els.pnlStrip.innerHTML = items
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
    if (els.drawdownMeta) {
      els.drawdownMeta.innerHTML = dd.available
        ? `<span class="dim small">Equity = NAV + cumulative PnL over ${dd.n_points} runs. Day-to-day P&amp;L is on the <a href="#pnl-section">P&amp;L tab</a>.</span>`
        : `<span class="dim small">${safeText(dd.reason, "drawdown unavailable")}</span>`;
    }
  }

  let _pnlPanelState = { view: "daily", lookback: 40, bucketPeriod: "today" };

  function renderPnlPanel(snap) {
    const panel = snap.pnl_panel || {};
    if (!panel.available) {
      if (els.pnlMeta) {
        els.pnlMeta.innerHTML = `<span class="dim small">${safeText(
          panel.reason,
          "P&amp;L history unavailable"
        )}</span>`;
      }
      ["pnlSummary", "pnlControls", "pnlDailyChart", "pnlDailyTable", "pnlWeeklyTable", "pnlBucketChart"].forEach(
        (k) => {
          if (els[k]) els[k].innerHTML = "";
        }
      );
      return;
    }

    const summary = panel.summary || {};
    const bucketLabels = panel.bucket_labels || {};

    if (els.pnlMeta) {
      els.pnlMeta.innerHTML = `<span class="dim small">Source: ${safeText(
        panel.source
      )} · ${panel.n_daily_rows || 0} trading days · vs prior ${safeText(summary.prior_date, "")}</span>`;
    }

    if (els.pnlSummary) {
      const cmpWeek =
        summary.prior_week_usd != null
          ? ` vs prior wk ${fmtUsdSigned(summary.prior_week_usd)}`
          : "";
      const cmpMonth =
        summary.prior_month_usd != null
          ? ` vs prior mo ${fmtUsdSigned(summary.prior_month_usd)}`
          : "";
      els.pnlSummary.innerHTML = [
        {
          label: "Today",
          value: fmtUsdSigned(summary.daily_usd),
          sub: `${fmtPct(summary.daily_pct_nav, 2)} of NAV`,
          cls: signedClass(summary.daily_usd),
        },
        {
          label: "Week to date",
          value: fmtUsdSigned(summary.wtd_usd),
          sub: `${fmtPct(summary.wtd_pct_nav, 2)} of NAV${cmpWeek}`,
          cls: signedClass(summary.wtd_usd),
        },
        {
          label: "Month to date",
          value: fmtUsdSigned(summary.mtd_usd),
          sub: `${fmtPct(summary.mtd_pct_nav, 2)} of NAV${cmpMonth}`,
          cls: signedClass(summary.mtd_usd),
        },
        {
          label: "YTD (cumulative)",
          value: fmtUsdSigned(summary.ytd_usd),
          sub: `${fmtPct(summary.ytd_pct_nav, 2)} of NAV`,
          cls: signedClass(summary.ytd_usd),
        },
      ]
        .map(
          (it) => `<div class="stat">
          <div class="label">${it.label}</div>
          <div class="value ${it.cls || ""}">${it.value}</div>
          ${it.sub ? `<div class="sub">${it.sub}</div>` : ""}
        </div>`
        )
        .join("");
    }

    function bucketRowsForPeriod(period) {
      if (period === "wtd") {
        const agg = {};
        (panel.daily || []).forEach((r) => {
          const d = new Date(r.date + "T12:00:00");
          const run = new Date((panel.run_date || r.date) + "T12:00:00");
          const weekStart = new Date(run);
          weekStart.setDate(run.getDate() - ((run.getDay() + 6) % 7));
          if (d < weekStart) return;
          Object.entries(r.buckets || {}).forEach(([k, v]) => {
            agg[k] = (agg[k] || 0) + Number(v || 0);
          });
        });
        return { buckets: agg, caption: "Week to date bucket moves" };
      }
      if (period === "last_week") {
        const wk = (panel.weekly || []);
        const last = wk.length >= 2 ? wk[wk.length - 2] : null;
        return {
          buckets: (last && last.buckets) || {},
          caption: last
            ? `Prior week ${safeText(last.week_label)} (${safeText(last.week_start)} – ${safeText(last.week_end)})`
            : "No prior week",
        };
      }
      const today = (panel.daily || [])[panel.daily.length - 1];
      return {
        buckets: (today && today.buckets) || {},
        caption: today ? `Today (${safeText(today.date)}) bucket moves` : "Today",
      };
    }

    function paint() {
      const dailyAll = panel.daily || [];
      const weeklyAll = panel.weekly || [];
      const lb = _pnlPanelState.lookback;
      const dailySlice =
        lb === "ytd" ? dailyAll : dailyAll.slice(-Math.min(lb, dailyAll.length));

      if (els.pnlControls) {
        els.pnlControls.innerHTML = `
          <label class="pnl-ctl">View
            <select id="pnl-view">
              <option value="daily" ${_pnlPanelState.view === "daily" ? "selected" : ""}>Daily bars</option>
              <option value="weekly" ${_pnlPanelState.view === "weekly" ? "selected" : ""}>Weekly bars</option>
            </select>
          </label>
          <label class="pnl-ctl">Lookback
            <select id="pnl-lookback">
              <option value="20" ${_pnlPanelState.lookback === 20 ? "selected" : ""}>20 sessions</option>
              <option value="40" ${_pnlPanelState.lookback === 40 ? "selected" : ""}>40 sessions</option>
              <option value="ytd" ${_pnlPanelState.lookback === "ytd" ? "selected" : ""}>All in panel</option>
            </select>
          </label>
          <label class="pnl-ctl">Bucket period
            <select id="pnl-bucket-period">
              <option value="today" ${_pnlPanelState.bucketPeriod === "today" ? "selected" : ""}>Today</option>
              <option value="wtd" ${_pnlPanelState.bucketPeriod === "wtd" ? "selected" : ""}>Week to date</option>
              <option value="last_week" ${_pnlPanelState.bucketPeriod === "last_week" ? "selected" : ""}>Prior week</option>
            </select>
          </label>`;
        const viewEl = document.getElementById("pnl-view");
        const lbEl = document.getElementById("pnl-lookback");
        const bpEl = document.getElementById("pnl-bucket-period");
        if (viewEl) viewEl.addEventListener("change", () => { _pnlPanelState.view = viewEl.value; paint(); });
        if (lbEl) lbEl.addEventListener("change", () => {
          _pnlPanelState.lookback = lbEl.value === "ytd" ? "ytd" : Number(lbEl.value);
          paint();
        });
        if (bpEl) bpEl.addEventListener("change", () => { _pnlPanelState.bucketPeriod = bpEl.value; paint(); });
      }

      const chartRows =
        _pnlPanelState.view === "weekly"
          ? weeklyAll.slice(-16).map((w) => ({
              date: w.week_end || w.week_label,
              daily_usd: w.daily_usd,
            }))
          : dailySlice;

      if (els.pnlDailyChart) {
        els.pnlDailyChart.innerHTML = pnlBarChartSvg(chartRows, { width: 720, height: 200 });
      }
      if (els.pnlTableTitle) {
        els.pnlTableTitle.textContent =
          _pnlPanelState.view === "weekly" ? "Daily detail (reference)" : "Daily P&L";
      }

      const bucketKeys = Object.keys(bucketLabels);
      const dailyHead =
        `<table class="tight"><thead><tr><th>Date</th><th class="num">Daily</th><th class="num">% NAV</th>` +
        bucketKeys.map((k) => `<th class="num">${safeText(bucketLabels[k] || k)}</th>`).join("") +
        `</tr></thead><tbody>`;
      const dailyBody = [...dailySlice]
        .reverse()
        .map((r) => {
          const bks = r.buckets || {};
          return `<tr>
            <td>${safeText(r.date)}</td>
            <td class="num ${signedClass(r.daily_usd)}">${fmtUsdSigned(r.daily_usd)}</td>
            <td class="num">${fmtPct(r.daily_pct_nav, 2)}</td>
            ${bucketKeys
              .map((k) => `<td class="num ${signedClass(bks[k])}">${fmtUsdSigned(bks[k])}</td>`)
              .join("")}
          </tr>`;
        })
        .join("");
      if (els.pnlDailyTable) {
        els.pnlDailyTable.innerHTML =
          dailyBody.length
            ? dailyHead + dailyBody + `</tbody></table>`
            : `<p class="dim">No daily rows.</p>`;
      }

      if (els.pnlWeeklyTable) {
        const wkHead = `<table class="tight"><thead><tr><th>Week</th><th>End</th><th class="num">Days</th><th class="num">P&amp;L</th><th class="num">% NAV</th></tr></thead><tbody>`;
        const wkBody = [...weeklyAll]
          .reverse()
          .map(
            (w) => `<tr>
            <td>${safeText(w.week_label)}</td>
            <td>${safeText(w.week_end)}</td>
            <td class="num">${w.n_days ?? "-"}</td>
            <td class="num ${signedClass(w.daily_usd)}">${fmtUsdSigned(w.daily_usd)}</td>
            <td class="num">${fmtPct(w.daily_pct_nav, 2)}</td>
          </tr>`
          )
          .join("");
        els.pnlWeeklyTable.innerHTML = wkBody
          ? wkHead + wkBody + `</tbody></table>`
          : `<p class="dim">No weekly rows.</p>`;
      }

      const bp = bucketRowsForPeriod(_pnlPanelState.bucketPeriod);
      if (els.pnlBucketChart) {
        els.pnlBucketChart.innerHTML = pnlBucketBarsHtml(bp.buckets, bucketLabels, {
          caption: bp.caption,
        });
      }
    }

    paint();
  }

  function renderMovers(panel) {
    const moverTable = (rows, cls) =>
      `<table class="tight"><thead><tr><th>Underlying</th><th>Symbols</th><th>P&amp;L</th></tr></thead>
       <tbody>${
         (rows || [])
           .map(
             (r) => `<tr>
          <td><strong>${safeText(r.underlying)}</strong></td>
          <td class="dim">${safeText(r.symbols, "-")}</td>
          <td class="num ${cls}">${fmtUsdSigned(r.total_pnl)}</td>
        </tr>`
           )
           .join("") || "<tr><td colspan=3 class=dim>(none)</td></tr>"
       }</tbody></table>`;
    if (els.moversWinners) {
      els.moversWinners.innerHTML = (panel && panel.available)
        ? moverTable(panel.winners, "pos")
        : `<p class="dim">${safeText(panel?.reason, "no mover data")}</p>`;
    }
    if (els.moversLosers) {
      els.moversLosers.innerHTML = (panel && panel.available)
        ? moverTable(panel.losers, "neg")
        : "";
    }
  }

  function renderBorrowShock(panel, nav) {
    if (!els.borrowShockContent) return;
    if (!panel || !panel.available) {
      els.borrowShockContent.innerHTML = `<p class="dim">${safeText(
        panel?.reason,
        "No borrow shock data."
      )}</p>`;
      if (els.borrowShockMeta) els.borrowShockMeta.innerHTML = "";
      return;
    }
    const scen = panel.scenarios || [];
    const rows = scen
      .map((s) => {
        const bite = s.incremental_pct_nav;
        const cls = bite != null && bite >= 0.005 ? "row-warn" : "";
        return `<tr class="${cls}">
          <td><strong>${safeText(s.label)}</strong></td>
          <td class="num">${fmtUsd(s.annual_cost_usd)}</td>
          <td class="num neg">${fmtUsdSigned(-Math.abs(s.incremental_cost_usd))}</td>
          <td class="num">${fmtPct(s.incremental_pct_nav, 3)}</td>
        </tr>`;
      })
      .join("");
    els.borrowShockContent.innerHTML = `
      <div class="strip" style="margin-bottom:8px;">
        <div class="stat"><div class="label">Short ETFs</div><div class="value">${
          panel.n_short_etfs ?? 0
        }</div><div class="sub">${fmtUsd(panel.short_notional_usd)} short</div></div>
        <div class="stat"><div class="label">Current annual carry</div><div class="value neg">${fmtUsdSigned(
          -Math.abs(panel.current_annual_cost_usd)
        )}</div><div class="sub">${fmtPct(panel.current_pct_nav, 3)} of NAV</div></div>
      </div>
      <table class="tight"><thead><tr>
        <th>Shock</th><th>Annual carry</th><th>Incremental drag</th><th>% NAV</th>
      </tr></thead><tbody>${rows}</tbody></table>
      <p class="dim small">${safeText(panel.note, "")}</p>`;
    if (els.borrowShockMeta) {
      els.borrowShockMeta.innerHTML = `<span class="dim small">Annualized; applied to held short-ETF legs.</span>`;
    }
  }

  function renderSharedUnderlying(panel) {
    if (!els.sharedUnderlyingContent) return;
    if (!panel || !panel.available || !(panel.rows || []).length) {
      els.sharedUnderlyingContent.innerHTML = `<p class="dim">${safeText(
        panel?.reason,
        "No underlyings span multiple buckets."
      )}</p>`;
      return;
    }
    const bucketShort = (b) => b.replace("bucket_", "B");
    const rows = panel.rows
      .map((r) => {
        const buckets = (r.buckets || []).map(bucketShort).join(", ");
        return `<tr>
          <td><strong>${safeText(r.underlying)}</strong></td>
          <td>${buckets}</td>
          <td class="num ${signedClass(r.net_usd)}">${fmtUsd(r.net_usd)}</td>
          <td class="num">${fmtUsd(r.gross_usd)}</td>
        </tr>`;
      })
      .join("");
    els.sharedUnderlyingContent.innerHTML = `
      <p class="dim small">${panel.n_shared} of ${panel.n_underlyings} underlyings span more than one bucket. ${safeText(
        panel.note,
        ""
      )}</p>
      <table class="tight sortable"><thead><tr>
        <th>Underlying</th><th>Buckets</th><th>Net $ (summed)</th><th>Gross $ (summed)</th>
      </tr></thead><tbody>${rows}</tbody></table>`;
  }

  function b5SeriesPath(series, width, height, pad, yTransform) {
    if (!series || series.length < 2) return "";
    const xs = series.map((_, i) => pad + (i / (series.length - 1)) * (width - 2 * pad));
    const ys = series.map((pt) => {
      const v = yTransform(Number(pt[1]));
      return v;
    });
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const span = maxY - minY || 1;
    const coords = xs.map((x, i) => {
      const y = height - pad - ((ys[i] - minY) / span) * (height - 2 * pad);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    });
    return coords.join(" ");
  }

  function b5DualChart(equitySeries, ddSeries, title) {
    const w = 720;
    const hEq = 170;
    const hDd = 100;
    const padL = 42;
    const padR = 8;
    const padT = 18;
    const padB = 22;
    const eqVals = (equitySeries || []).map((p) => Number(p[1]) / 1e6);
    const ddVals = (ddSeries || []).map((p) => Number(p[1]) * 100);
    const eqMin = Math.min(...eqVals, 0);
    const eqMax = Math.max(...eqVals, 1);
    const ddMin = Math.min(...ddVals, -5);
    const ddMax = Math.max(...ddVals, 0);
    const yEq = (v) => hEq - padB - ((v - eqMin) / (eqMax - eqMin || 1)) * (hEq - padT - padB);
    const yDd = (v) => hDd - padB - ((v - ddMin) / (ddMax - ddMin || 1)) * (hDd - padT - padB);
    const xAt = (i, n, h) => padL + (i / Math.max(1, n - 1)) * (w - padL - padR);
    const eqPath = eqVals.length >= 2
      ? eqVals.map((v, i) => `${xAt(i, eqVals.length, hEq).toFixed(1)},${yEq(v).toFixed(1)}`).join(" ")
      : "";
    const ddPath = ddVals.length >= 2
      ? ddVals.map((v, i) => `${xAt(i, ddVals.length, hDd).toFixed(1)},${yDd(v).toFixed(1)}`).join(" ")
      : "";
    const eqTicks = dashNiceTicks(eqMin, eqMax, 4);
    const ddTicks = dashNiceTicks(ddMin, ddMax, 4);
    const dates = (equitySeries || []).map((p) => p[0]);
    const xLabels = dates.length >= 2
      ? [0, Math.floor(dates.length / 2), dates.length - 1].map((i) =>
          `<text x="${xAt(i, dates.length, hEq).toFixed(1)}" y="${hEq + hDd + 8}" fill="#64748b" font-size="9" text-anchor="middle">${safeText(String(dates[i]).slice(0, 10))}</text>`
        ).join("")
      : "";
    return `
      <div class="b5bt-chart-block">
        <h4>${title}</h4>
        <svg viewBox="0 0 ${w} ${hEq + hDd + 14}" width="100%" style="max-width:${w}px">
          <text x="${padL}" y="12" class="dim" font-size="11">Equity ($M)</text>
          ${dashSvgYAxis(padL, hEq, eqTicks, yEq, (v) => v.toFixed(1))}
          <polyline fill="none" stroke="#0f766e" stroke-width="2" points="${eqPath}"/>
          <line x1="${padL}" y1="${hEq + 4}" x2="${w - padR}" y2="${hEq + 4}" stroke="#cbd5e1" stroke-width="0.5"/>
          <text x="${padL}" y="${hEq + 18}" class="dim" font-size="11">Drawdown (%)</text>
          ${dashSvgYAxis(padL, hDd, ddTicks, (v) => hEq + 6 + yDd(v), (v) => v.toFixed(0))}
          <polyline fill="none" stroke="#991b1b" stroke-width="1.2" transform="translate(0,${hEq + 6})" points="${ddPath}"/>
          ${xLabels}
        </svg>
      </div>`;
  }

  function renderAssumptionSections(sections) {
    if (!sections || !sections.length) return "";
    return sections.map((sec) => {
      const rows = (sec.rows || []).map(([k, v]) => `<tr><td>${safeText(k)}</td><td>${safeText(v)}</td></tr>`).join("");
      return `<details class="callout dim small" style="margin:6px 0"><summary><strong>${safeText(sec.title)}</strong></summary>` +
        `<table class="tight" style="margin-top:6px"><tbody>${rows}</tbody></table></details>`;
    }).join("");
  }

  function b5NearestGridPoint(grid, sleeve, premium, stress) {
    if (!grid || !grid.points || !grid.points.length) return null;
    let best = grid.points[0];
    let bestD = Infinity;
    for (const p of grid.points) {
      const d =
        Math.abs(p.sleeve_frac - sleeve) +
        Math.abs(p.total_premium - premium) * 10 +
        Math.abs(p.borrow_stress_mult - stress) * 0.05;
      if (d < bestD) {
        bestD = d;
        best = p;
      }
    }
    return best;
  }

  function renderBucket5Backtest(panel) {
    const content = document.getElementById("b5bt-content");
    const meta = document.getElementById("b5bt-meta");
    if (!content) return;
    if (!panel || !panel.variants) {
      if (meta) meta.innerHTML = statusPill("unknown", "B5 backtest data not available");
      content.innerHTML =
        '<div class="callout dim">No Bucket 5 backtest panel. Run <code>python scripts/build_bucket5_backtest_panel.py --with-sensitivity</code> then rebuild the dashboard.</div>';
      return;
    }
    const grid = panel.sensitivity_grid || null;
    const labCmd = safeText(panel.lab_command, "streamlit run scripts/bucket5_lab.py");
    if (meta) {
      meta.innerHTML =
        `<span class="dim">as of ${safeText(panel.run_date)} · generated ${safeText(panel.generated_at_utc)} · ` +
        `full parameter lab: <code>${labCmd}</code></span>`;
    }
    const variantKeys = Object.keys(panel.variants);
    const defaultKey = variantKeys.includes("F_extended") ? "F_extended" : variantKeys[0];
    const options = variantKeys
      .map((k) => {
        const v = panel.variants[k];
        return `<option value="${k}">${safeText(v.label || k)} (${safeText(v.era)})</option>`;
      })
      .join("");
    const axes = (grid && grid.axes) || {};
    const sleeveOpts = (axes.sleeve_frac || [0.15, 0.20, 0.25])
      .map((v) => `<option value="${v}">${Math.round(v * 100)}%</option>`)
      .join("");
    const premOpts = (axes.total_premium || [0.024, 0.03])
      .map((v) => `<option value="${v}">${(v * 100).toFixed(1)}%</option>`)
      .join("");
    const stressOpts = (axes.borrow_stress_mult || [1.0, 1.5, 2.0])
      .map((v) => `<option value="${v}">${v.toFixed(1)}×</option>`)
      .join("");

    content.innerHTML = `
      <div class="callout dim small">
        Short UVIX + short SVIX carry with SPX put hedge, T-bill collateral, and liquidate-and-redeploy monetization.
        <strong>Interactive lab:</strong> run <code>${labCmd}</code> locally for full sweeps, tornado charts, and preset save/load.
      </div>
      <div class="b4sim-controls strip" style="margin:10px 0;gap:14px;flex-wrap:wrap">
        <label class="b4sim-ctl">Variant
          <select id="b5bt-variant">${options}</select>
        </label>
        <label class="b4sim-ctl">Compare to
          <select id="b5bt-compare"><option value="">— none —</option>${options}</select>
        </label>
      </div>
      ${grid ? `
      <div class="callout warn small" style="margin:8px 0">
        <strong>Precomputed sensitivity</strong> (F dynamic deep-skew, ${safeText(grid.era || "extended")}):
        snap sliders to nearest grid point. For continuous exploration use the Streamlit lab.
      </div>
      <div class="b4sim-controls strip" style="margin:0 0 10px;gap:14px;flex-wrap:wrap">
        <label class="b4sim-ctl">Sleeve gross
          <select id="b5bt-sleeve">${sleeveOpts}</select>
        </label>
        <label class="b4sim-ctl">Premium / roll
          <select id="b5bt-prem">${premOpts}</select>
        </label>
        <label class="b4sim-ctl">Borrow stress
          <select id="b5bt-stress">${stressOpts}</select>
        </label>
      </div>
      <div id="b5bt-grid-stats" class="strip"></div>
      <div id="b5bt-grid-charts" class="two-col"></div>
      <p id="b5bt-grid-takeaway" class="callout dim small" style="margin:6px 0 10px"></p>
      ` : `<p class="dim small">Rebuild with <code>--with-sensitivity</code> to enable grid sliders here.</p>`}
      <div id="b5bt-stats" class="strip"></div>
      <p id="b5bt-takeaway" class="callout dim small" style="margin:6px 0 10px"></p>
      <div id="b5bt-livebook"></div>
      <div id="b5bt-regime"></div>
      <div id="b5bt-charts" class="two-col"></div>
      <div id="b5bt-crash"></div>
      <div id="b5bt-assumptions"></div>
      <div id="b5bt-combined"></div>`;

    const variantEl = document.getElementById("b5bt-variant");
    const compareEl = document.getElementById("b5bt-compare");
    variantEl.value = defaultKey;

    function renderVariant(key, compareKey) {
      const v = panel.variants[key];
      if (!v) return;
      const m = v.metrics || {};
      const statsEl = document.getElementById("b5bt-stats");
      const chartsEl = document.getElementById("b5bt-charts");
      const crashEl = document.getElementById("b5bt-crash");
      const assumEl = document.getElementById("b5bt-assumptions");
      const liveEl = document.getElementById("b5bt-livebook");
      const regimeEl = document.getElementById("b5bt-regime");
      const combinedEl = document.getElementById("b5bt-combined");
      const ddCls = (v) => (v <= -0.4 ? "stat-neg" : v <= -0.25 ? "stat-warn" : "");
      const stats = [
        { label: "CAGR", value: fmtPct(m.combined_CAGR, 1), tip: "Compound annual growth rate of combined equity." },
        { label: "Vol", value: fmtPct(m.combined_Vol, 1), tip: "Annualized volatility of daily returns." },
        { label: "Max DD", value: fmtPct(m.combined_MaxDD, 1), cls: ddCls(m.combined_MaxDD), tip: "Worst peak-to-trough drawdown over the backtest." },
        { label: "Sharpe", value: safeText(m.combined_Sharpe), tip: "Return per unit of volatility (higher is better)." },
        { label: "Calmar", value: safeText(m.combined_Calmar), tip: "CAGR divided by max drawdown (return per unit of pain)." },
        { label: "Realized $", value: fmtUsd(m["realized_$"]), tip: "Cash harvested by monetizing the put hedge during vol spikes." },
      ];
      statsEl.innerHTML = stats
        .map((s) => {
          const valCls = s.cls === "stat-neg" ? "neg" : s.cls === "stat-warn" ? "warn" : "";
          return `<div class="summary-card ${s.cls || ""}" title="${s.tip || ""}"><div class="label">${s.label}</div><div class="value ${valCls}">${s.value}</div></div>`;
        })
        .join("");
      const takeawayEl = document.getElementById("b5bt-takeaway");
      if (takeawayEl) {
        const cg = (m.combined_CAGR != null ? m.combined_CAGR * 100 : null);
        const dd = (m.combined_MaxDD != null ? m.combined_MaxDD * 100 : null);
        takeawayEl.innerHTML =
          `<strong>In plain English:</strong> ${safeText(v.label || key)} compounded at ` +
          `<strong>${cg == null ? "—" : cg.toFixed(1) + "%/yr"}</strong> with a worst drawdown of ` +
          `<strong>${dd == null ? "—" : dd.toFixed(1) + "%"}</strong>, harvesting ` +
          `<strong>${fmtUsd(m["realized_$"])}</strong> from the put hedge. ` +
          `The crash payoffs below show what it makes when the market gaps down.`;
      }

      const eq = (v.series || {}).combined_equity || [];
      const dd = (v.series || {}).drawdown || [];
      let chartsHtml = b5DualChart(eq, dd, v.label || key);
      if (compareKey && panel.variants[compareKey]) {
        const c = panel.variants[compareKey];
        chartsHtml += b5DualChart(
          (c.series || {}).combined_equity || [],
          (c.series || {}).drawdown || [],
          (c.label || compareKey) + " (compare)"
        );
      }
      chartsEl.innerHTML = chartsHtml;

      const crash = v.crash || {};
      const crashRows = Object.entries(crash)
        .map(([k, val]) => `<tr><td>${safeText(k)}</td><td class="num">${fmtPct(val, 1)}</td></tr>`)
        .join("");
      crashEl.innerHTML = `<h3>Stylized crash payoffs</h3><table class="tight"><thead><tr><th>Scenario</th><th class="num">Combined P&amp;L</th></tr></thead><tbody>${crashRows}</tbody></table>`;

      const a = v.assumptions || {};
      assumEl.innerHTML =
        `<details open class="callout dim small" style="margin-top:12px"><summary><strong>How this backtest works (all assumptions)</strong></summary>` +
        renderAssumptionSections(v.assumption_sections) +
        `<p class="dim" style="margin-top:8px">Quick summary: UVIX borrow ${fmtPct(a.borrow_uvix_annual, 2)}/yr · SVIX ${fmtPct(a.borrow_svix_annual, 2)}/yr · ` +
        `slippage ${safeText(a.uvix_slip_bps)} bps · T-bills ${fmtPct(a.tbill_rate, 2)}/yr · sleeve ${fmtPct(a.sleeve_frac, 0)} · ` +
        `${safeText(v.meta?.pricing_mode || "")} · ${safeText(v.meta?.start)}→${safeText(v.meta?.end)}` +
        (v.monetize_summary ? ` · ${v.monetize_summary.event_count} monetize events (${fmtUsd(v.monetize_summary.total_usd)} harvested in sim)` : "") +
        `</p></details>`;

      if (liveEl && panel.live_book && panel.live_book.rows && panel.live_book.rows.length) {
        const lb = panel.live_book.rows.map((r) =>
          `<tr><td>${safeText(r.etf)}</td><td>${safeText(r.underlying)}</td>` +
          `<td class="num">${fmtUsd(r.proposed_gross_usd)}</td><td class="num">${fmtUsd(r.optimal_gross_usd)}</td>` +
          `<td class="num">${fmtPct(r.borrow_annual, 1)}</td><td>${r.locate_ok ? "yes" : "no"}</td></tr>`
        ).join("");
        liveEl.innerHTML = `<h3>Live B5 book (proposed trades ${safeText(panel.live_book.run_date)})</h3>` +
          `<table class="tight"><thead><tr><th>ETF</th><th>Und</th><th class="num">Proposed $</th><th class="num">Optimal $</th><th class="num">Borrow</th><th>Locate</th></tr></thead><tbody>${lb}</tbody></table>`;
      } else if (liveEl) liveEl.innerHTML = "";

      if (regimeEl && panel.regime && panel.regime.label) {
        regimeEl.innerHTML = `<p class="callout dim small"><strong>VIX regime (latest in backtest):</strong> ${safeText(panel.regime.label)} · ratio ${safeText(panel.regime.ratio)} as of ${safeText(panel.regime.date)}</p>`;
      } else if (regimeEl) regimeEl.innerHTML = "";

      if (combinedEl) {
        const b4 = (window.__dashboardSnap && window.__dashboardSnap.bucket4_risk_sim) || null;
        const crash30 = crash["crash_severe_-30%"];
        let html = `<details class="callout dim small"><summary><strong>B4 + B5 combined lens</strong></summary>`;
        if (b4 && b4.reference_mc && b4.reference_mc.block_bootstrap) {
          const p95 = b4.reference_mc.block_bootstrap.dd_p95;
          html += `<p>B4 structural tail (ref bootstrap p95 maxDD ~${fmtPct(p95, 0)}) vs B5 crash -30% payoff ${fmtPct(crash30, 0)} on this variant. ` +
            `Insurance sleeve harvests when vol spikes; size B4 for survivable drawdown and B5 for convex crash payoffs.</p>`;
        } else {
          html += `<p>Rebuild B4 risk sim to compare tail drawdown with B5 crash payoffs side by side.</p>`;
        }
        html += `<p class="dim">Full exploration: <code>${labCmd}</code></p></details>`;
        combinedEl.innerHTML = html;
      }
    }

    function renderGridPoint() {
      if (!grid) return;
      const sleeveEl = document.getElementById("b5bt-sleeve");
      const premEl = document.getElementById("b5bt-prem");
      const stressEl = document.getElementById("b5bt-stress");
      const gStats = document.getElementById("b5bt-grid-stats");
      const gCharts = document.getElementById("b5bt-grid-charts");
      const gTake = document.getElementById("b5bt-grid-takeaway");
      if (!sleeveEl || !gStats) return;
      const pt = b5NearestGridPoint(
        grid,
        parseFloat(sleeveEl.value),
        parseFloat(premEl.value),
        parseFloat(stressEl.value)
      );
      if (!pt) return;
      const ddCls = (v) => (v <= -0.4 ? "stat-neg" : v <= -0.25 ? "stat-warn" : "");
      gStats.innerHTML = [
        { label: "Grid CAGR", value: fmtPct(pt.CAGR, 1) },
        { label: "Grid Max DD", value: fmtPct(pt.MaxDD, 1), cls: ddCls(pt.MaxDD) },
        { label: "Grid Sharpe", value: safeText(pt.Sharpe) },
        { label: "Grid Calmar", value: safeText(pt.Calmar) },
        { label: "Realized $", value: fmtUsd(pt.realized_usd) },
        { label: "Crash -30%", value: fmtPct(pt.crash_severe, 1) },
      ]
        .map((s) => `<div class="summary-card ${s.cls || ""}"><div class="label">${s.label}</div><div class="value">${s.value}</div></div>`)
        .join("");
      if (gCharts) {
        const ser = pt.series || {};
        if (ser.combined_equity && ser.combined_equity.length) {
          gCharts.innerHTML = b5DualChart(
            ser.combined_equity,
            ser.drawdown || [],
            `Grid point equity (sleeve ${fmtPct(pt.sleeve_frac, 0)}, prem ${fmtPct(pt.total_premium, 1)}, stress ${pt.borrow_stress_mult.toFixed(1)}×)`
          );
        } else {
          gCharts.innerHTML = `<p class="dim small">Rebuild sensitivity grid to include equity curves per grid point.</p>`;
        }
      }
      if (gTake) {
        gTake.innerHTML =
          `<strong>Grid point:</strong> sleeve ${fmtPct(pt.sleeve_frac, 0)}, premium ${fmtPct(pt.total_premium, 1)}/roll, ` +
          `borrow stress ${pt.borrow_stress_mult.toFixed(1)}×. ` +
          (pt.series && pt.series.combined_equity ? `Equity curve above is from the precomputed grid.` : `Preset variant charts below.`);
      }
    }

    function update() {
      renderVariant(variantEl.value, compareEl.value || null);
      renderGridPoint();
    }
    variantEl.addEventListener("change", update);
    compareEl.addEventListener("change", update);
    if (grid) {
      ["b5bt-sleeve", "b5bt-prem", "b5bt-stress"].forEach((id) => {
        const el = document.getElementById(id);
        if (el) el.addEventListener("change", renderGridPoint);
      });
    }
    update();
  }

  function renderAll(snap) {
    _lastSnap = snap;
    _tabRendered.clear();
    els.runDateLabel.textContent = `Run: ${snap.run_date}`;
    els.generatedAtLabel.textContent =
      "Generated " + new Date(snap.generated_at_utc).toLocaleString();

    const coreSteps = [
      () => renderFreshness(snap),
      () => renderDataQuality(snap.data_quality || {}),
      () => renderCockpit(snap),
      () => renderAlerts(snap.alert_rows || []),
      () => renderActionQueue(snap.action_queue || []),
    ];
    for (const step of coreSteps) {
      try {
        step();
      } catch (e) {
        console.error("render step failed:", e);
        throw new Error(`Dashboard render failed: ${e.message || e}`);
      }
    }
    _tabRendered.add("overview");
    renderTabPanel("overview", snap);

    const { tab } = dashParseHash();
    switchDashboardTab(tab, { updateHash: false, snap });

    fetch("./build_meta.json")
      .then((r) => (r.ok ? r.json() : null))
      .then((meta) => {
        const foot = document.getElementById("deploy-meta");
        if (!foot || !meta) return;
        foot.textContent = `Deployed ${meta.deployed_at_utc || ""} · snapshot ${meta.snapshot_run_date || snap.run_date || ""} · ${String(meta.commit_sha || "").slice(0, 7)}`;
      })
      .catch(() => {});
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
    bindDashboardTabs();
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
