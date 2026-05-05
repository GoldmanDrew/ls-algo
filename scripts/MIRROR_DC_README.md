# Diamond-Creek-Quant mirror instructions

This directory ships a ready-to-apply unified diff so the same realized-decay
fix from ls-algo PR #11 can be mirrored into
`GoldmanDrew/Diamond-Creek-Quant` without re-deriving the change. PR #11's
validation report (`scripts/validation_parity_lsalgo_dc_*.txt`) showed that
**260 / 384** intersection rows currently disagree because DC is missing
this fix and ls-algo PR #10's σ-panel pooling.

## Patch contents

`scripts/dc_winsor_only.patch` — winsorization of daily hedge-drag inside
`_compute_gross_decay_daily`. Adaptive percentile cuts (1/99 for n ≥ 100,
2/98 for n ≥ 60, 5/95 below) applied **after** the optional split-day
drop, **before** `mean(drag) × 252`. Logs `[DECAY][winsor]` when clipping
materially changes the mean.

This patch ONLY ships change (A) from ls-algo PR #11. Change (B) — the
`raw_fallback_capped` path inside PASS2 of `enrich_with_decay_and_vol` —
requires PR #10's σ-panel pooling work first; mirror those PR #10 commits
to DC before this patch for full coverage.

## Apply

```bash
git clone https://github.com/GoldmanDrew/Diamond-Creek-Quant.git
cd Diamond-Creek-Quant
git checkout -b cursor/mirror-lsalgo-decay-fix
git apply /path/to/scripts/dc_winsor_only.patch
python3 -c "import ast; ast.parse(open('daily_screener.py').read())"
git add daily_screener.py
git commit -m "screener: winsorize daily hedge-drag (mirror of ls-algo PR #11)"
git push -u origin cursor/mirror-lsalgo-decay-fix
```

Then open a PR titled `screener: mirror ls-algo decay outlier fix`.

## Verification on DC after applying

Re-run DC's screener with `--skip-scrape --skip-ftp --skip-ibkr-check` and
check that:

- DUOG, BAIG, FIGG, BMNG (etc.) realized gross_decay no longer reads
  −500%+ on thin samples.
- `[DECAY][winsor] SYMBOL: ...` lines appear in the log for affected
  rows.
- `data/etf_screened_today.csv` `gross_decay_annual` column for the
  ls-algo-vs-DC parity worst-10 (DUOG, BMNG, FIGG, QBTZ, BAIG, CRWG,
  CRCG, BEZ, IONZ, EOSU) matches the ls-algo post-fix values within
  the parity tolerance (max(0.02, 5%)).

For the BMNR / OPEN / SRPT / SATS / CRML / NVTS / LAC σ_und divergence
(σ_und disagreement, not winsor-only) you still need PR #10 mirrored
separately.
