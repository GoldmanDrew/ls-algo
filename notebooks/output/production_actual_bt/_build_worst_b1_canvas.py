"""Emit worst-b1-pairs-investigation.canvas.tsx with embedded data."""
from __future__ import annotations

import json
from pathlib import Path

# Rebuild series from full daily so large |PnL| days are never dropped.
base = Path(__file__).resolve().parent
import pandas as pd

compact = json.loads((base / "_worst_b1_compact.json").read_text(encoding="utf-8"))
pdaily = pd.read_csv(base / "pair_daily_pnl.csv", parse_dates=["date"])
keep_pairs = [f["pair"] for f in compact["findings"][:10]] + [
    "SMUP/SMR",
    "RDWU/RDW",
    "CRMX/CRML",
]
keep_pairs = list(dict.fromkeys(keep_pairs))
series: dict[str, list] = {}
for key in keep_pairs:
    etf, und = key.split("/")
    d = pdaily[
        (pdaily["ETF"] == etf)
        & (pdaily["Underlying"] == und)
        & (pdaily["sleeve"] == "core_leveraged")
    ].sort_values("date")
    if d.empty:
        continue
    keep_idx = set()
    for i, r in enumerate(d.itertuples()):
        if (
            i == 0
            or i == len(d) - 1
            or bool(r.is_rebalance)
            or abs(float(r.daily_pnl)) >= 200
            or i % 2 == 0
        ):
            keep_idx.add(i)
    rows = []
    for i, r in enumerate(d.itertuples()):
        if i not in keep_idx:
            continue
        rows.append(
            {
                "d": str(r.date.date())[5:],
                "cum": round(float(r.cum_pnl), 2),
                "dp": round(float(r.daily_pnl), 2),
                "g": round(abs(float(r.long_usd)) + abs(float(r.short_usd)), 2),
                "h": round(float(r.hedge_ratio), 4) if pd.notna(r.hedge_ratio) else None,
                "D": round(float(r.Delta), 4) if pd.notna(r.Delta) else None,
                "b": round(float(r.borrow_cost), 2),
                "rb": 1 if r.is_rebalance else 0,
            }
        )
    series[key] = rows
compact["series"] = series
data_lit = json.dumps(compact, separators=(",", ":"))

canvas = f'''import {{
  Callout,
  Card,
  CardBody,
  CardHeader,
  Divider,
  Grid,
  H1,
  H2,
  LineChart,
  Pill,
  Row,
  Select,
  Stack,
  Stat,
  Table,
  Text,
  useCanvasState,
  useHostTheme,
}} from "cursor/canvas";

const DATA = {data_lit} as const;

type PairKey = keyof typeof DATA.series;

const PAIR_OPTIONS = (Object.keys(DATA.series) as PairKey[]).map((p) => ({{
  value: p,
  label: p,
}}));

const FLAG_LABEL: Record<string, string> = {{
  loss_concentrated_few_days: "Loss in few days",
  end_h_vs_delta_mismatch: "End h vs Delta mismatch",
  persistent_hedge_drift: "Hedge drift",
  wrong_leg_signs: "Wrong leg signs",
  very_high_borrow: "Very high borrow",
  costs_dominate_price: "Costs dominate",
}};

export default function WorstB1PairsInvestigation() {{
  const theme = useHostTheme();
  const [pair, setPair] = useCanvasState<PairKey>("pair", "SMU/SMR");
  const series = DATA.series[pair] ?? [];
  const finding = DATA.findings.find((f) => f.pair === pair);
  const cats = series.map((r) => r.d);
  const cum = series.map((r) => r.cum);
  const gross = series.map((r) => r.g);
  const h = series.map((r) => (r.h == null ? 0 : r.h));
  const delta = series.map((r) => (r.D == null ? 0 : r.D));
  const daily = series.map((r) => r.dp);

  const integrity = (DATA.integrity ?? []).find(
    (r) => r.etf === pair.split("/")[0]
  );

  return (
    <Stack gap={{20}} style={{{{ maxWidth: 1080 }}}}>
      <Stack gap={{6}}>
        <H1>Worst B1 pairs — path investigation</H1>
        <Text tone="secondary" size="small">
          Source: production_actual_bt pair_daily_pnl / pair_stats · 2026-02-27 →
          2026-07-13 · core_leveraged only
        </Text>
      </Stack>

      <Grid columns={{4}} gap={{12}}>
        <Stat
          value={{`$${{DATA.worst12_pnl.toLocaleString(undefined, {{ maximumFractionDigits: 0 }})}}`}}
          label="Worst-12 total PnL"
          tone="danger"
        />
        <Stat
          value={{`$${{DATA.book_b1_pnl.toLocaleString(undefined, {{ maximumFractionDigits: 0 }})}}`}}
          label="All B1 PnL"
          tone="success"
        />
        <Stat value={{String(DATA.n_b1_pairs)}} label="B1 pairs traded" />
        <Stat value="3" label="Artifact / process flags" tone="warning" />
      </Grid>

      <Callout tone="warning" title="What looks off">
        SMCL/SMCI printed a clean +200% ETF notional jump on 2026-06-19 (underlying
        flat) that fully reversed on 2026-06-22 — classic adj-close / phantom spike;
        net PnL almost cancels but the path is broken. RDWU/RDW has a matching
        integrity jump on 2026-05-27 and borrow (~49% ann) ate a +$870 price edge.
        MSTU/MSTR and CRCG/CRCL ended with hedge_ratio far above Delta under the
        calm operator_5d / 50% ratio band (under-hedged shorts drifted). SMU lost
        ~$3.0k on SMR while sister wrapper SMUP made +$1.4k — wrapper selection, not
        a single-name thesis failure.
      </Callout>

      <H2>Worst 12 scoreboard</H2>
      <Table
        headers={{[
          "Pair",
          "PnL $",
          "Price $",
          "Borrow $",
          "Avg gross",
          "ROG",
          "Worst day",
          "Flags",
        ]}}
        columnAlign={{[
          "left",
          "right",
          "right",
          "right",
          "right",
          "right",
          "left",
          "left",
        ]}}
        rows={{DATA.findings.map((f) => [
          f.pair,
          f.pnl_usd.toLocaleString(undefined, {{ maximumFractionDigits: 0 }}),
          f.price_pnl_usd.toLocaleString(undefined, {{ maximumFractionDigits: 0 }}),
          f.borrow_cost_usd.toLocaleString(undefined, {{ maximumFractionDigits: 0 }}),
          f.avg_gross.toLocaleString(undefined, {{ maximumFractionDigits: 0 }}),
          f.rog.toFixed(2),
          `${{f.worst_day}} (${{f.worst_day_pnl.toLocaleString(undefined, {{ maximumFractionDigits: 0 }})}})`,
          f.flags.length ? f.flags.map((x) => FLAG_LABEL[x] ?? x).join("; ") : "—",
        ])}}
        rowTone={{DATA.findings.map((f) =>
          f.pair === "SMCL/SMCI" || f.pair === "RDWU/RDW"
            ? "warning"
            : f.pnl_usd < -800
              ? "danger"
              : undefined
        )}}
      />

      <Divider />

      <Row gap={{12}} style={{{{ alignItems: "center", justifyContent: "space-between" }}}}>
        <H2>Pair time series</H2>
        <Select
          value={{pair}}
          onChange={{(v) => setPair(v as PairKey)}}
          options={{PAIR_OPTIONS}}
        />
      </Row>

      {{finding ? (
        <Grid columns={{4}} gap={{12}}>
          <Stat
            value={{`$${{finding.pnl_usd.toLocaleString()}}`}}
            label="Total PnL"
            tone={{finding.pnl_usd < 0 ? "danger" : "success"}}
          />
          <Stat
            value={{`$${{finding.avg_gross.toLocaleString()}}`}}
            label="Avg pair gross"
          />
          <Stat
            value={{`${{(finding.loss_top3_share * 100).toFixed(0)}}%`}}
            label="Top-3 loss days share"
            tone={{finding.loss_top3_share > 0.5 ? "warning" : "neutral"}}
          />
          <Stat
            value={{`${{(finding.borrow_ann_approx * 100).toFixed(0)}}%`}}
            label="Approx borrow ann. on short"
            tone={{finding.borrow_ann_approx > 0.3 ? "warning" : "neutral"}}
          />
        </Grid>
      ) : null}}

      {{finding?.flags?.length ? (
        <Row gap={{8}}>
          {{finding.flags.map((fl) => (
            <Pill key={{fl}} tone="warning">
              {{FLAG_LABEL[fl] ?? fl}}
            </Pill>
          ))}}
        </Row>
      ) : null}}

      {{integrity &&
      (Number(integrity.phantom_days) > 0 ||
        Number(integrity.scale_bad_days) > 0 ||
        Boolean(integrity.issue)) ? (
        <Callout tone="danger" title={{`Price integrity: ${{pair.split("/")[0]}}`}}>
          worst_abs_ret={{Number(integrity.worst_abs_ret).toFixed(2)}} on{{" "}}
          {{String(integrity.worst_date)}} · phantom_days={{String(integrity.phantom_days)}} ·
          scale_bad_days={{String(integrity.scale_bad_days)}}
          {{integrity.issue ? ` · ${{String(integrity.issue)}}` : ""}}
        </Callout>
      ) : null}}

      <Card>
        <CardHeader>Cumulative pair PnL ($)</CardHeader>
        <CardBody>
          <LineChart
            categories={{cats}}
            series={{[{{ name: "cum_pnl ($)", data: cum, tone: "danger" }}]}}
            height={{220}}
            beginAtZero={{false}}
          />
          <Text tone="secondary" size="small">
            Every other session + rebals + |daily PnL|≥$200 · {{pair}}
          </Text>
        </CardBody>
      </Card>

      <Grid columns={{2}} gap={{16}}>
        <Card>
          <CardHeader>Pair gross deployed ($)</CardHeader>
          <CardBody>
            <LineChart
              categories={{cats}}
              series={{[{{ name: "gross ($)", data: gross, tone: "info" }}]}}
              height={{200}}
            />
          </CardBody>
        </Card>
        <Card>
          <CardHeader>Daily PnL ($)</CardHeader>
          <CardBody>
            <LineChart
              categories={{cats}}
              series={{[{{ name: "daily_pnl ($)", data: daily, tone: "neutral" }}]}}
              height={{200}}
              beginAtZero={{false}}
            />
          </CardBody>
        </Card>
      </Grid>

      <Card>
        <CardHeader>Hedge ratio vs Delta (share-hold drift)</CardHeader>
        <CardBody>
          <LineChart
            categories={{cats}}
            series={{[
              {{ name: "|und|/|etf| hedge_ratio", data: h, tone: "warning" }},
              {{ name: "Delta", data: delta, tone: "info" }},
            ]}}
            height={{200}}
            beginAtZero={{false}}
          />
          <Text tone="secondary" size="small">
            Calm policy only repairs ratio on operator days and only outside a 50%
            hard band — persistent under-hedges (MSTU) are expected under current knobs.
          </Text>
        </CardBody>
      </Card>

      <H2>Flagged cases (detail)</H2>
      <Grid columns={{1}} gap={{12}}>
        <Card>
          <CardHeader trailing={{<Pill tone="danger">price artifact</Pill>}}>
            SMCL / SMCI — exact 3× ETF spike then reverse
          </CardHeader>
          <CardBody>
            <Table
              headers={{["Date", "ETF $", "Und $", "ETF ret", "Price PnL", "h"]}}
              columnAlign={{["left", "right", "right", "right", "right", "right"]}}
              rows={{(DATA.smcl_spike ?? []).map((r: {{
                date: string;
                etf_usd: number;
                underlying_usd: number;
                etf_ret: number | null;
                price_pnl: number;
                hedge_ratio: number;
              }}) => [
                String(r.date).slice(0, 10),
                Number(r.etf_usd).toFixed(0),
                Number(r.underlying_usd).toFixed(0),
                r.etf_ret == null || Number.isNaN(Number(r.etf_ret))
                  ? "—"
                  : `${{(Number(r.etf_ret) * 100).toFixed(0)}}%`,
                Number(r.price_pnl).toFixed(0),
                Number(r.hedge_ratio).toFixed(2),
              ])}}
            />
            <Text tone="secondary" size="small">
              2026-06-19: ETF notional −2.1k → −6.3k (+200%) while SMCI flat; 06-22
              reverses. Net pair loss only −$673 because the phantom mostly cancels —
              still a panel bug, not alpha.
            </Text>
          </CardBody>
        </Card>

        <Card>
          <CardHeader trailing={{<Pill tone="warning">wrapper split</Pill>}}>
            SMR: SMU (−$2,951) vs SMUP (+$1,437)
          </CardHeader>
          <CardBody>
            <Text>
              Same underlying, opposite outcomes. SMU bled steadily Mar→Jun on price;
              SMUP harvested the other side of wrapper tracking / sizing. Select SMUP/SMR
              in the dropdown to overlay the winning path. Points to shared-underlying
              allocator / wrapper switching, not a broken SMR short thesis alone.
            </Text>
          </CardBody>
        </Card>

        <Card>
          <CardHeader trailing={{<Pill tone="warning">borrow + jump</Pill>}}>
            RDWU / RDW — integrity jump + borrow tax
          </CardHeader>
          <CardBody>
            <Text>
              Price PnL +$870 but borrow −$1,017 → net −$162. Integrity file flags
              scale_bad / jump on 2026-05-27 (same day as −$2.3k print). Treat path as
              contaminated; economics are borrow-dominated even after the jump noise.
            </Text>
          </CardBody>
        </Card>

        <Card>
          <CardHeader trailing={{<Pill tone="info">vol path</Pill>}}>
            FIGG / RCAX — real vol, not obvious bugs
          </CardHeader>
          <CardBody>
            <Text>
              FIGG flipped from +$530 (Jun 25) to −$959 by Jul 13 on large daily swings
              (top-3 days = 65% of losses). RCAX saw ±$2–3k days around Jun 3–8 with
              hedge_ratio staying near 2× — looks like volatile pair P&amp;L under
              share-hold, not a sign flip or phantom. Still worth checking whether
              resize bands / ADV caps prevented hedge refresh into the storm.
            </Text>
          </CardBody>
        </Card>
      </Grid>

      <Text tone="secondary" size="small" style={{{{ color: theme.text.secondary }}}}>
        Margin cost is $0 on all pair rows because financing is booked at the netted
        book level when net_shared_underlyings=true — not a per-pair bug. SMCL
        integrity metadata still points at Mar-20; the Jun-19 +200% ETF jump was
        missed by that checker but is visible in pair_daily notionals.
      </Text>
    </Stack>
  );
}}
'''

out = Path(
    r"C:\Users\drewg\.cursor\projects\c-Users-drewg-Projects-quant-ls-algo\canvases\worst-b1-pairs-investigation.canvas.tsx"
)
out.write_text(canvas, encoding="utf-8")
print("wrote", out)
print("bytes", out.stat().st_size)
