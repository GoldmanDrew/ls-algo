# B5 Product Dashboard UI Polish Plan

**Status:** implemented (Phases 0–3) · 2026-07-15  
**Date:** 2026-07-14 (audit) / 2026-07-15 (implementation)  
**Scope:** visual design, layout, typography, chart scaling, host sync  
**Out of scope:** product/feature logic, backtest methodology, data schema changes (except UI-facing presentation fields)  
**Reference (gold standard):** `C:\Users\drewg\Projects\options-trading\spx-0dte\docs\index.html`  
**Hosts audited:**
- ls-algo Risk Dashboard `#tab=b5` → `site/assets/js/bucket5_product.js` + `site/assets/css/main.css`
- etf-dashboard `#/bucket5` → same JS (byte-identical) + React wrapper `assets/bucket5_insurance_backtest.js` + **almost no product CSS**

Temporary local audit harnesses (may delete after polish):  
`docs/_b5_ui_audit_preview.html`, `docs/_b5_ui_audit_preview_etf.html`, `docs/_spx_layout_reference_mock.html`

---

## 1. Verdict

B5 Product copied SPX-0DTE’s *information architecture* (Guide → KPIs → charts → Regime → Daily) but not its *visual system*. On ls-algo it looks like Magis risk-chrome with stubby sparklines bolted on. On etf-dashboard it is effectively unstyled HTML: subnav collapses to `OverviewRegimeDaily`, KPI strips stack as plain text, panels have no card surface.

**Priority order:** (1) ship shared B5 CSS so etf-dashboard is not broken, (2) fix chart furniture/scale, (3) fix hierarchy/density/guide measure, (4) reach SPX parity on subnav + Daily sticky controls.

---

## 2. Current-state audit

### 2.1 Shared JS (both hosts)

| Fact | Detail |
|------|--------|
| Canonical module | `site/assets/js/bucket5_product.js` (ls-algo) |
| etf copy | `etf-dashboard/assets/bucket5_product.js` — **byte-identical** hash as of audit |
| Pipeline sync | `scripts/build_bucket5_product_dashboard.py --copy-etf-dashboard` copies **JSON only**, not JS/CSS |
| Mount API | `Bucket5Product.mount(container, data)` string-templates HTML |
| Subtabs | Overview / Regime / Daily via `.dash-subnav.b5p-subnav` pill links |
| Charts | Hand-rolled SVG polylines: fixed `viewBox` 720×160–180, `pad=12`, **no axes, grid, ticks, tooltips, or zero line** |
| KPI | Reuses Magis `.strip > .stat` |
| Signed colors | `cls()` emits `pos`/`neg` on table cells; CSS only styles `.value.pos` / `.value.neg` → **crash/event signed colors often do nothing** |

### 2.2 ls-algo host (`#tab=b5`)

**Shell mismatch**
- Tab content lives inside `<section class="panel panel-hero">`, then JS emits more `.panel` children → **3 nested panels** measured at 1440px. Double borders, double padding, cramped inner width.
- Magis chrome defaults to **light** theme (Calibri/Segoe, navy `#0f3460` headers, 4px radius). B5 charts use GitHub-dark accent greens/purples. Dark theme helps, but B5 still inherits Magis density tokens (tight `table.tight` navy theads, 10.5px strip labels, 14px main max-width 1400).

**Guide (Overview)**
- Full-bleed prose: at 1440px guide width **~1314px**, paragraph width **~1277px** (~180+ characters). Unreadable wall of text.
- Guide + “How to read this dashboard” push KPIs/charts **well below the fold**.
- Duplicate metrics: guide `results` strip **and** “Key results” strip both show CAGR / Sharpe / Max DD / Harvested $.

**Charts (measured)**

| Chart | At 1440px | At ~676px |
|-------|-----------|-----------|
| Equity overlay | 720×180 (capped) | 575×144 |
| Drawdown / Put MTM (two-col) | 650×144 each | 575×128 stacked |
| Harvested cash | 720×160 | 575×128 |
| Regime series (5 charts) | ~119–180px tall | ~119px tall |

SPX reference: primary line charts **900×320** with `pad {l:64,r:16,t:16,b:48}`, y-grid, year ticks, formatted y labels. Drawdown **240px**. B5 charts read as decorative sparklines — especially Regime rho/gross (dense square-wave with no scale).

**Subnav**
- Pill chips (`.dash-subnav a`) under Magis panel head — not SPX’s full-width underline tabs.
- Not sticky; scroll Overview far and lose section context.
- Hash is only `#tab=b5`; Overview/Regime/Daily state is in-memory only (refresh loses subtab).

**Daily**
- Run + Day selects in a panel; Day dropdown can have **~4600 options** (every calendar day) — painful UX vs SPX’s trading-day filter.
- Controls are **not sticky**.
- 7 day-KPI cards use the same large Magis strip spacing → sparse for a drill-down.
- Marks/Events tables use navy Magis `table.tight` thead (harsh in dark theme).
- Live sleeve note is good; presentation is another nested panel + strip.

**What looks “messed up” (concrete)**
1. Nested panel-in-panel chrome.
2. Guide line length full bleed.
3. Duplicate KPI blocks.
4. Axis-less 160px charts next to dense 18-year series.
5. Regime two-col collapses fine, but charts stay sparkline-height.
6. Subnav pills feel like secondary Magis anchors, not primary IA.
7. Signed PnL colors missing on `td.pos`/`td.neg`.

### 2.3 etf-dashboard host (`#/bucket5`)

**Critical:** `index.html` has **no** rules for `.panel`, `.strip`, `.dash-subnav`, `.two-col`, `.b5p-*`, `table.tight`, `.pos`/`.neg`, `.callout`, `.dim`, `.row`.  
Wrapper only styles `.backtest-page` + topbar buttons.

**Measured on CSS-stripped mount (1440×900):**
- `.panel` background `rgba(0,0,0,0)`, border `0`
- `.strip` `display: block` (not grid) → KPIs stack vertically as label/value text
- `.b5p-subnav` `display: block` → links read as **`OverviewRegimeDaily`**
- Guide line-height falls back to browser default (~20px), not `1.55`

React host also duplicates chrome: page title “Bucket 5 Product” + JS `panel-head` “Bucket 5 Product — UVIX/SVIX insurance”.

**Net:** etf-dashboard B5 is a product regression waiting to be noticed. Fix CSS sync before any “polish” on ls-algo alone.

### 2.4 SPX-0DTE — what “good” looks like

From `docs/index.html` tokens + structure (live docs are auth-gated; CSS/structure audited in source + layout mock):

| Token / pattern | SPX value |
|-----------------|-----------|
| Page bg / panel / ink / muted / line | `#0b0f17` / `#131a26` / `#e6edf6` / `#8aa0bd` / `#243245` |
| Accent | `#58a6ff` (underline tabs, not pill fill) |
| Radius | cards `10px`, panels `12px`, inputs `8px` |
| Main | `padding: 24px; max-width: 1280px` |
| KPI grid | `.cards` `auto-fill, minmax(170px, 1fr)`, value **22px** |
| Guide results | `.result-grid` `minmax(200px, 1fr)`, value **17px** |
| Guide rhythm | `h3` 15px / margin 20px 0 8px; `p` line-height 1.55 |
| Tabs | underline, 10×16 padding, muted → ink |
| Charts | full panel width, height 240–320, axis furniture |
| Daily | labeled `.row` + select; day KPIs as `.cards`; tables full-bleed in panel |

SPX is a **purpose-built product shell**. B5 is a **widget inside Magis/etf shells**. Parity means adopting SPX patterns *inside* a scoped `.b5p-root`, not painting Magis globally.

---

## 3. Design principles to copy from SPX-0DTE

1. **One product surface** — `.b5p-root` owns background, radius, spacing; do not nest Magis `.panel-hero` around it.
2. **Underline subnav, not pills** — Overview / Regime / Daily are primary IA, equal to SPX Overview / Market factors / Daily.
3. **Readable guide measure** — prose `max-width: 68–72ch`; never full 1280–1400px bleed.
4. **Charts are first-class** — primary equity ≥ **280–320px** tall; always axes + year ticks + y labels; secondary charts ≥ **220px**.
5. **KPI cards, not Magis strip by default** — SPX-style `.b5p-cards` with uppercase muted labels and large tabular values.
6. **One KPI story** — either guide results *or* Overview KPI row, not both identical.
7. **Sticky Daily controls** — run/day pickers stay visible while scrolling marks/events.
8. **Host-agnostic CSS** — same stylesheet on ls-algo and etf-dashboard; map to CSS variables with fallbacks.
9. **Dark-first product chrome** — even if Magis light theme is on, B5 product block should use SPX-like panel contrast (or force dark island). Prefer: respect host theme via tokens, but never rely on host having `.strip` defined.
10. **Density with hierarchy** — SPX is dense but every block has a clear h2, 18–20px panel padding, and breathing room between panels (`margin-bottom: 20px`).

---

## 4. Specific fixes (with file targets)

### 4.1 Shared CSS package (highest leverage)

**Create** `site/assets/css/bucket5_product.css` (canonical) and **copy/symlink** to `etf-dashboard/assets/bucket5_product.css`.

Suggested structure:

```css
/* Scoped under .b5p-root — maps host tokens with SPX fallbacks */
.b5p-root {
  --b5p-bg: var(--bg-panel, var(--bg-card, #131a26));
  --b5p-bg2: var(--bg-soft, var(--bg-secondary, #1a2434));
  --b5p-ink: var(--text-primary, #e6edf6);
  --b5p-muted: var(--text-muted, #8aa0bd);
  --b5p-line: var(--border, #243245);
  --b5p-accent: var(--accent, #58a6ff);
  --b5p-pos: var(--pos, #3fb950);
  --b5p-neg: var(--neg, #f85149);
  --b5p-radius: 12px;
  --b5p-gap: 20px;
  --b5p-measure: 72ch;
  max-width: 1280px;
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  gap: var(--b5p-gap);
  color: var(--b5p-ink);
  font: 14px/1.45 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
```

**Concrete rules to include**

| Class | Spec |
|-------|------|
| `.b5p-tabs` | flex; gap 4px; border-bottom 1px solid line; replace `.dash-subnav` for B5 |
| `.b5p-tabs a` | padding 10px 16px; muted; bottom border 2px transparent |
| `.b5p-tabs a.active` | ink + accent underline (not filled pill) |
| `.b5p-panel` | bg, 1px line, radius 12px, padding 18px, margin 0 (gap from flex) |
| `.b5p-guide p, .b5p-guide li` | max-width `var(--b5p-measure)`; line-height 1.55 |
| `.b5p-guide h3` | 15px; margin 20px 0 8px |
| `.b5p-cards` | grid `repeat(auto-fill, minmax(160px, 1fr))`; gap 12px |
| `.b5p-card .k` | 11px uppercase letter-spacing 0.04em muted |
| `.b5p-card .v` | 20–22px font-weight 650; tabular-nums |
| `.b5p-chart` | width 100%; svg `width:100%; height: auto; min-height` via aspect or fixed height |
| `.b5p-chart--primary svg` | min-height **300px** (or height:300) |
| `.b5p-chart--secondary svg` | min-height **220px** |
| `.b5p-controls` | sticky `top: 0`; z-index 5; backdrop blur/bg; padding 10px 0; border-bottom |
| `.b5p-table-wrap` | overflow-x: auto |
| `.b5p-table` | width 100%; font-size 13px; th muted uppercase 11px; no Magis navy fill |
| `.b5p-root .pos` / `.neg` | color pos/neg |
| `@media (max-width: 900px)` | two-col → 1fr; cards minmax(140px); reduce chart heights to 220/180 |

**Wire-up**
- `site/index.html` — `<link rel="stylesheet" href="./assets/css/bucket5_product.css?v=…">`
- `etf-dashboard/index.html` — same link for `assets/bucket5_product.css`
- Extend `--copy-etf-dashboard` (or a small `scripts/sync_bucket5_product_ui.py`) to copy **JS + CSS**, not only JSON.

### 4.2 HTML shell / mount changes — `bucket5_product.js`

| Change | Detail |
|--------|--------|
| Stop using Magis `.panel` / `.strip` / `.dash-subnav` / `.two-col` / `table.tight` | Emit `.b5p-panel`, `.b5p-cards`, `.b5p-tabs`, `.b5p-grid-2`, `.b5p-table` |
| Outer mount | Single `.b5p-root` with head + tabs + body; no inner Magis panel-head duplication on etf |
| Guide | Collapse “How to read this dashboard” into a `<details>` or move to footer tip |
| KPIs | Prefer guide `results` as the only Overview KPI band **or** Overview strip only — drop duplicate |
| Overview run switcher | Add primary-run `<select>` on Overview (SPX has this when multi-run); today run switcher exists only on Daily |
| Legend | Click-to-toggle series opacity like SPX (optional phase 2) |
| Crash table | Use `.b5p-table`; ensure `.pos`/`.neg` styled |
| Daily day list | Prefer eventful days (rebalance / monetize / live) + “All days” optgroup, or datalist/typeahead — do not dump 4600 naked options as the only UX |

### 4.3 Chart scaling — `bucket5_product.js` `LineChart` / `MultiEquityChart`

Replace sparkline helper with SPX-shaped SVG:

```
viewBox width: 900 (or 100% via preserveAspectRatio)
height: primary 300–320, secondary 220–240, regime 240
pad: { l: 56–64, r: 16, t: 16, b: 40–48 }
y ticks: 4–5 gridlines + formatted labels
x ticks: year labels from series dates (copy SPX yearAxisTicks idea)
zero line when series crosses 0 (drawdown, daily PnL later)
stroke-width: 2 primary / 1.5 secondary
```

**Hard caps**
- Remove `max-width: 720px` on SVG — charts should span the panel (up to `.b5p-root` 1280px).
- Equity overlay: **full width × 300px min**.
- two-col charts: each column full width of half panel, height **≥ 220px** (not 144).
- Regime: stack primary ratio chart full width 260px; rho/gross/VIX in 2-col at 220px.

Optional phase 2: hover tooltip (date + value) — SPX lacks rich tooltips too, so axes alone are the acceptance bar for phase 1.

### 4.4 ls-algo shell — `site/index.html` + light `app.js` if needed

```html
<!-- Prefer flat host, no panel-hero wrapper -->
<div id="dash-tab-b5" class="dash-tab-panel" data-dash-tab="b5" hidden>
  <div id="b5-product-content" class="b5p-host"></div>
</div>
```

Avoid wrapping mount target in `.panel.panel-hero`.

### 4.5 etf-dashboard shell — `assets/bucket5_insurance_backtest.js` + `index.html`

| Change | Detail |
|--------|--------|
| Load CSS | `bucket5_product.css` |
| Reduce duplicate titles | Wrapper: Back + Chart UVIX/SVIX only; let JS own H1 |
| Host width | `.b5-product-page` / `#b5-product-host` max-width 1280, padding align with other backtest pages (24px) |
| Optional | Map etf tokens → b5p vars already covered by fallbacks |

### 4.6 Typography / spacing cheat sheet

| Element | Target |
|---------|--------|
| Root font | 14px / 1.45 system stack (SPX), not Calibri-only |
| Guide lead | 15px muted |
| Guide h3 | 15px / 600 / margin-top 20px |
| Panel h2 | 14px / 600 / margin-bottom 14px |
| Panel padding | 18px |
| Panel gap | 20px |
| Cards gap | 12px |
| Card value | 20–22px (17px inside guide result grid) |
| Table font | 13px; th 11px uppercase muted |
| Main max | 1280px (SPX), even if Magis main is 1400 |

### 4.7 Responsive

| Breakpoint | Behavior |
|------------|----------|
| ≤ 900px | 2-col → 1; sticky controls still work; chart heights 220 / 180 |
| ≤ 640px | cards `minmax(140px, 1fr)`; tabs wrap; truncate long run labels |
| Desktop ≥ 1100px | primary chart 300–320 tall; guide measure still 72ch (do **not** widen prose with viewport) |

### 4.8 Dark / light contrast

- Magis light: B5 panels white/soft with navy accent is OK if tokens map; ensure chart stroke colors have contrast on light (`#0f766e`-family ok; avoid low-contrast `#6366f1` on white without checking).
- Magis dark / etf dark: prefer SPX green/red (`#3fb950` / `#f85149`) for signed values.
- Kill Magis navy-white `table.tight` thead inside `.b5p-root` — use muted header row.

---

## 5. Shared component / CSS sync strategy

```
ls-algo/site/assets/js/bucket5_product.js      ← canonical JS
ls-algo/site/assets/css/bucket5_product.css    ← canonical CSS (NEW)
        │
        │  sync script / CI step
        ▼
etf-dashboard/assets/bucket5_product.js
etf-dashboard/assets/bucket5_product.css
etf-dashboard/assets/bucket5_insurance_backtest.js  ← thin React mount only
```

**Rules**
1. Edit JS/CSS only in ls-algo; etf copies are generated or fail CI if drift.
2. Extend `build_bucket5_product_dashboard.py --copy-etf-dashboard` (or sibling sync) to copy:
   - `bucket5_product.json`
   - `bucket5_product.js`
   - `bucket5_product.css`
3. Cache-bust query params in **both** `index.html` files together (`?v=b5p-YYYYMMDD`).
4. Add a tiny etf/ls test: assert CSS file exists and contains `.b5p-root` and `.b5p-cards` (mirror existing JS schema tests).
5. Do **not** depend on Magis `main.css` or etf global `.summary-card` for B5 layout.

Optional later: extract chart primitives to `b5p_charts.js` shared with future products — not required for polish.

---

## 6. Phased implementation

### Phase 0 — Unbreak etf (½–1 day) — **do first**

- [x] Add `bucket5_product.css` with self-contained `.b5p-*`.
- [x] Link CSS on etf-dashboard + ls-algo.
- [x] Sync script copies CSS+JS (`sync_bucket5_product_ui.py` + `--copy-etf-dashboard`).
- [x] Smoke: etf `#/bucket5` shows spaced tabs, card grid KPIs, bordered panels.

**Exit:** etf no longer looks like raw HTML.

### Phase 1 — Quick visual wins (1–2 days)

- [x] Remove `panel-hero` wrapper on ls-algo B5 tab.
- [x] Cap guide measure at 72ch; tighten guide section spacing to SPX rhythm.
- [x] Deduplicate Overview KPIs (keep one band).
- [x] Collapse “How to read…” into `<details>` (default closed).
- [x] Style `.pos`/`.neg` under `.b5p-root`.
- [x] Replace pill subnav with underline `.b5p-tabs`.
- [x] Chart: remove 720px max-width; bump heights to 300 / 220; add y-grid + year ticks + y labels.
- [x] Sticky Daily controls.

**Exit:** Overview reads like a product page; charts are interpretable; Daily usable while scrolling.

### Phase 2 — Structural polish (2–3 days)

- [x] Rename markup classes away from Magis (`.b5p-panel`, `.b5p-cards`, `.b5p-table`).
- [x] Overview primary-run selector + equity legend toggle.
- [x] Daily day picker: eventful/live subset + search.
- [x] Persist subtab in hash (`#tab=b5&b5=regime` or `#/bucket5/regime`).
- [x] Table overflow wrappers; numeric column alignment.
- [x] Light/dark token QA on both hosts (CSS variables with SPX fallbacks).
- [x] Daily return chart (trailing window from `combined_ret`).

**Exit:** Side-by-side with SPX feels same family.

### Phase 3 — Parity / delight (optional)

- [x] Hover tooltips on charts.
- [x] Crash scenario visual (horizontal bar / bullet chart).
- [x] Print/PDF friendly Overview.
- [x] Motion: tab fade 120ms; no chart animation required.
- [x] Accessibility: tablist roles, focus rings, select labels.

---

## 7. Acceptance criteria / visual checklist

### Both hosts

- [ ] Subnav reads as three distinct tabs with ≥8px gaps (never `OverviewRegimeDaily`).
- [ ] Active tab = underline accent, not only pill fill.
- [ ] Guide prose measure ≤ ~72ch at all desktop widths.
- [ ] Exactly one Overview KPI band for headline metrics.
- [ ] Equity chart height ≥ 280px; has y-axis labels and year ticks.
- [ ] Secondary charts ≥ 220px; Regime charts not sparkline-thin.
- [ ] Charts span panel width (no 720px artificial cap on wide screens).
- [ ] `.pos` / `.neg` visible on signed table cells and KPI values.
- [ ] Daily Run/Day controls remain visible while scrolling marks (sticky).
- [ ] Tables scroll horizontally inside wrapper on narrow viewports; no page-wide overflow.
- [ ] No double-border nested Magis panel around the whole B5 root.
- [ ] 900px and 1440px widths both look intentional (not just “scaled sparklines”).

### etf-dashboard specific

- [ ] Loading `#/bucket5` without Magis CSS still shows cards/panels/tabs correctly.
- [ ] Back / Chart UVIX / Chart SVIX controls don’t fight B5 H1.
- [ ] Visual parity with ls-algo B5 within normal token drift.

### ls-algo specific

- [ ] Works in light and dark Magis themes without unreadable navy-on-navy.
- [ ] B5 tab does not inherit oversized `panel-hero` shadow framing.

### Regression

- [ ] Existing JSON schema `bucket5_product_dashboard.v1` mounts without builder changes.
- [ ] etf tests still pass; add CSS presence assertion.
- [ ] Cache-bust params updated on both sites.

---

## 8. Explicit non-goals

- Rewriting Bucket 5 strategy, monetization rules, or backtest engine.
- Replacing Magis Risk Dashboard global chrome / other tabs.
- Porting SPX Market-factors OLS panels wholesale (B5 Regime is the analogue).
- Building a React rewrite of `bucket5_product.js` (string mount is fine if CSS is scoped).
- Adding Plotly/Chart.js dependency unless axes prove too costly in SVG (prefer pure SVG like SPX).
- Pixel-perfect cloning of Magis light theme onto etf (dark product island is acceptable).
- Editing any existing B5 *product feature* plan doc the user attached earlier.
- Live auth bypass or publishing credentials for screenshot automation.

---

## 9. Suggested implementation order (checklist for the next agent)

1. Author `bucket5_product.css` with legacy class aliases so current JS improves immediately on etf.
2. Link + sync CSS/JS to etf; verify `#/bucket5`.
3. Flatten ls-algo host wrapper; switch tabs to underline.
4. Upgrade chart helper (axes + heights + full width).
5. Guide measure + KPI dedupe + sticky Daily.
6. Markup class rename + hash subtabs + day picker UX.
7. Visual QA checklist above on 1440 / 900 / 390 widths, light+dark.

---

## 10. Appendix — audit measurements (2026-07-14)

| Metric | Value |
|--------|-------|
| JS identity ls ↔ etf | identical SHA-256 |
| Nested `.panel .panel` (ls preview) | 3 |
| Guide width @1440 | ~1314px |
| Paragraph width @1440 | ~1277px |
| Equity SVG @1440 | 720×180 (max-width capped) |
| Two-col chart cell @1440 | 650×144 |
| Regime chart height @~676 | ~119px |
| etf `.panel` bg without CSS | transparent |
| etf `.strip` display without CSS | `block` |
| SPX primary chart | 900×320, pad L64/B48 |
| SPX main max-width | 1280px |

**Bottom line:** treat B5 Product as a **portable SPX-style micro-app**. Give it its own CSS, honest chart scale, and readable guide — then keep ls-algo and etf-dashboard in lockstep. Until Phase 0 ships, etf-dashboard B5 should be considered visually broken.
