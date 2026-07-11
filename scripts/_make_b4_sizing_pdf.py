"""Write docs/b4_sizing_method.pdf (plain-English guide + worked example).

Regenerate:
    python scripts/_make_b4_sizing_pdf.py
"""
from __future__ import annotations

from pathlib import Path

from reportlab.lib.colors import HexColor, white
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

OUT = Path(__file__).resolve().parents[1] / "docs" / "b4_sizing_method.pdf"

NAVY = HexColor("#1a365d")
TEAL = HexColor("#0d9488")
SLATE = HexColor("#334155")
LIGHT = HexColor("#f1f5f9")
SOFT = HexColor("#e2e8f0")
ACCENT = HexColor("#0f766e")


def main() -> None:
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "TitleCustom",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=22,
        textColor=NAVY,
        spaceAfter=6,
        leading=26,
        alignment=TA_CENTER,
    )
    subtitle = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        textColor=SLATE,
        spaceAfter=18,
        alignment=TA_CENTER,
        leading=14,
    )
    h1 = ParagraphStyle(
        "H1Custom",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=14,
        textColor=NAVY,
        spaceBefore=16,
        spaceAfter=8,
        leading=18,
    )
    h2 = ParagraphStyle(
        "H2Custom",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11.5,
        textColor=ACCENT,
        spaceBefore=12,
        spaceAfter=6,
        leading=15,
    )
    body = ParagraphStyle(
        "BodyCustom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        textColor=SLATE,
        spaceAfter=8,
        leading=14,
        alignment=TA_JUSTIFY,
    )
    bullet = ParagraphStyle(
        "BulletCustom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        textColor=SLATE,
        leftIndent=14,
        spaceAfter=4,
        leading=13,
    )
    formula = ParagraphStyle(
        "Formula",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=9.5,
        textColor=NAVY,
        backColor=LIGHT,
        leftIndent=8,
        rightIndent=8,
        spaceBefore=4,
        spaceAfter=8,
        leading=13,
    )
    callout = ParagraphStyle(
        "Callout",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=10,
        textColor=ACCENT,
        spaceBefore=6,
        spaceAfter=10,
        leading=13,
        alignment=TA_CENTER,
    )
    small = ParagraphStyle(
        "Small",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        textColor=HexColor("#64748b"),
        spaceAfter=4,
        leading=11,
    )
    step_title = ParagraphStyle(
        "StepTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        textColor=NAVY,
        spaceBefore=8,
        spaceAfter=3,
        leading=13,
    )
    ex_num = ParagraphStyle(
        "ExNum",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=9.5,
        textColor=SLATE,
        leftIndent=10,
        spaceAfter=3,
        leading=12,
    )

    story: list = []
    story.append(Spacer(1, 0.25 * inch))
    story.append(Paragraph("Bucket 4 Sizing Method", title))
    story.append(
        Paragraph(
            "Inverse-decay pairs · How we decide how big each position should be<br/>"
            "Simple guide with a worked example · Production as of 11 July 2026",
            subtitle,
        )
    )
    story.append(HRFlowable(width="100%", thickness=1.5, color=TEAL, spaceAfter=14))

    story.append(
        Paragraph(
            "Bucket 4 shorts inverse ETFs (products that lose value when a stock or sector "
            "rises) and hedges them by also shorting the underlying. The goal is to harvest "
            "structural decay while keeping relative crash risk under control and deploying "
            "the sleeve's target budget.",
            body,
        )
    )
    story.append(
        Paragraph(
            "One rule to remember: ρ sets relative crash risk across names. "
            "With scale_to_budget on, the sleeve still fills its ~3% target — "
            "effective per-name crash loss ≈ ρ × scale_mult.",
            callout,
        )
    )

    story.append(Paragraph("1. The big picture (four steps)", h1))
    story.append(
        Paragraph(
            "Sizing runs in a fixed order. Crash capping sets relative risk; "
            "scale-to-budget then refills the sleeve pro-rata so the YAML target "
            "actually deploys; smoothing damps week-to-week churn without undoing "
            "true own-risk cuts.",
            body,
        )
    )
    for t, d in [
        (
            "Step 1 — Score the book",
            "Rank pairs by expected decay after borrow costs (continuous high-borrow ramp). "
            "Convert scores into relative weights that sum to ~100%.",
        ),
        (
            "Step 2 — Crash-cap, then scale to budget",
            "Trim each name so its conditional crash loss ≤ ρ of the sleeve, then "
            "(with scale_to_budget: true) scale the whole capped book pro-rata up to "
            "the sleeve target. Riskier names stay smaller than quieter ones.",
        ),
        (
            "Step 3 — Post-cap weight smoothing",
            "Trim-only EMA on the final capped weights: raises and new entries fade in; "
            "dilution from new entrants fades out; dropped names soft-exit; true own-risk "
            "cuts still land immediately.",
        ),
        (
            "Step 4 — Leg split + ratchet",
            "Turn each pair's gross into an inverse-ETF short and an underlying short "
            "using hedge ratio h. Grow-only inverse ratchet guards hard-to-relocate inventory.",
        ),
    ]:
        story.append(Paragraph(t, step_title))
        story.append(Paragraph(d, bullet))

    story.append(Paragraph("2. Sleeve budget", h1))
    story.append(
        Paragraph(
            "Bucket 4 gets a fixed share of book target gross (config: "
            "<font face='Courier'>inverse_decay_bucket4.target_weight</font>, typically 3%). "
            "That target is the <b>deploy</b> target — after crash trim + scale_to_budget, "
            "sleeve gross aims at it. Temporary under-deployment (while new names ramp in) "
            "stays in cash and self-corrects.",
            body,
        )
    )
    story.append(
        Paragraph(
            "Example sleeve used below: <b>$115,500</b> (illustrative, shaped like a live July 2026 run).",
            body,
        )
    )

    story.append(Paragraph("3. Step 1 — Decay / borrow weights", h1))
    story.append(
        Paragraph(
            "Crash risk is <b>not</b> in this score. The score only answers: "
            "“How attractive is the decay after expected borrow?”",
            body,
        )
    )
    story.append(
        Paragraph(
            "base_score = decay / (1 + quad·borrow²) / (1 + linear·borrow) / (1 + unc·borrow_var)",
            formula,
        )
    )
    story.append(
        Paragraph(
            "High-borrow names are faded continuously (not a binary cliff): "
            "score × 1.0 below borrow_ramp_lo (80%), linear down to 0 at borrow_ramp_hi (120%), "
            "using the same posterior borrow estimate as the aversion terms.",
            body,
        )
    )
    story.append(
        Paragraph(
            "Then weights = base_score / sum(scores), clipped to a min/max per name "
            "(default 0.5%–35%), then lightly tilted for covariance concentration.",
            body,
        )
    )
    story.append(
        Paragraph(
            "Production knobs: borrow_linear_aversion = 1.5, borrow_uncertainty_penalty = 3.0, "
            "decay_borrow_quad = 0, borrow_ramp_lo/hi = 0.80 / 1.20.",
            small,
        )
    )

    story.append(Paragraph("4. Step 2 — Conditional crash budget + scale", h1))
    story.append(Paragraph("For each underlying, as of the run date:", body))
    story.append(
        Paragraph(
            "runup   = max(0, price / 252-day median − 1)<br/>"
            "retrace = θ · runup / (1 + runup)     with θ = 0.5<br/>"
            "tail    = worst ~3y 20-day drop + 0.45 · downside vol<br/>"
            "C       = max(tail, retrace)          conditional crash size",
            formula,
        )
    )
    story.append(
        Paragraph(
            "Intuition: a name that has already run up a lot can still fall a lot from here, "
            "even if recent realized crashes look quiet. We take the larger of historical tail "
            "and a fraction of the run-up.",
            body,
        )
    )
    story.append(Paragraph("Pair loss per $1 of gross (L):", body))
    story.append(
        Paragraph(
            "L = (1 − h) · β / (1 + h·β) · C · (1 + φ·C)     with φ = 0.5",
            formula,
        )
    )
    story.append(
        Paragraph(
            "L is EMA-smoothed across runs (asymmetric): risk-up binds immediately; "
            "risk-down fades at l_ema_alpha = 0.4 so a single rolling-window print does not "
            "step the whole book.",
            body,
        )
    )
    story.append(Paragraph("Per-name trim, then sleeve refill:", body))
    story.append(
        Paragraph(
            "cap_usd      = ρ · sleeve_budget / max(L, L_floor)<br/>"
            "gross_trim  = min(solved_gross, cap_usd)<br/>"
            "scale_mult  = sleeve_budget / sum(gross_trim)     (if scale_to_budget)<br/>"
            "gross_scaled = gross_trim × scale_mult<br/>"
            "ρ = 0.75%,   L_floor = 2%",
            formula,
        )
    )
    story.append(
        Paragraph(
            "ρ is a <b>relative</b> crash-risk scale. After the pro-rata refill, every "
            "full-size name still contributes roughly the same worst-case dollar loss, "
            "but that loss is ≈ ρ × scale_mult of the sleeve (logged each run as "
            "rho_effective). We do <b>not</b> hand freed dollars preferentially to "
            "uncapped names — the refill preserves the post-cap risk ordering.",
            body,
        )
    )

    story.append(Paragraph("5. Step 3 — Post-cap dilution-aware smoothing", h1))
    story.append(
        Paragraph(
            "Smoothing runs <b>after</b> crash-cap + scale so the refill's cross-coupling "
            "(one name's cap change re-pricing every other name) is damped too. "
            "State lives in <font face='Courier'>data/b4_weight_ema_state.json</font> "
            "(stage-tagged, with pre-scale own-risk for hard-cut detection).",
            body,
        )
    )
    story.append(
        Paragraph(
            "raises          → EMA toward target at alpha (0.50)<br/>"
            "new entries     → ramp at entry_alpha (0.25)<br/>"
            "dilution cuts   → fade at dilution_alpha (0.25) when scale compresses "
            "incumbents but their own pre-scale capacity is unchanged<br/>"
            "soft exits      → fade dropped names at soft_exit_alpha (0.35)<br/>"
            "hard cuts       → immediate if own pre-scale capacity drops &gt; hard_cut_rel (10%)<br/>"
            "no-trade band   → hold if move &lt; 15% rel or 25 bp abs",
            formula,
        )
    )
    story.append(
        Paragraph(
            "Deliberately not renormalized inside the smoother: renormalizing would push "
            "cut names back above their solved weight. Downstream normalizes. Under-deployment "
            "while entries ramp stays in cash.",
            body,
        )
    )

    story.append(Paragraph("6. Step 4 — Split gross into two legs + ratchet", h1))
    story.append(
        Paragraph(
            "gross = smoothed_weight × sleeve_budget<br/>"
            "inv_short  = gross / (1 + h · β)<br/>"
            "und_short  = h · β · inv_short",
            formula,
        )
    )
    story.append(
        Paragraph(
            "h (hedge ratio) comes from the TR/VCR engine (v7 + v9 tilt): typically between "
            "0.30 and 0.80, centered near 0.45. Higher h = more underlying hedge = less crash beta, "
            "but more borrow/margin on the stock leg.",
            body,
        )
    )
    story.append(
        Paragraph(
            "The inverse-ETF short is grow-only by default: we floor it at "
            "max(solved, currently held, persisted floor) so we do not cover inventory that is "
            "hard to re-borrow. When held size has crept above the fresh target, a continuous "
            "trim gradually closes the gap (paired with the underlying so h stays consistent).",
            body,
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("7. Worked example — DAMD / AMD", h1))
    story.append(
        Paragraph(
            "Numbers below are stylized from the 10 July 2026 live shape so the arithmetic is "
            "easy to follow. Sleeve budget = <b>$115,500</b>. Pair = short DAMD (−2× AMD) + short AMD. "
            "Assume the rest of the book is also crash-trimmed so the sleeve-wide "
            "scale_mult ≈ <b>6.47</b>.",
            body,
        )
    )

    story.append(Paragraph("Given inputs", h2))
    given_data = [
        [
            Paragraph("<b>Input</b>", small),
            Paragraph("<b>Value</b>", small),
            Paragraph("<b>Meaning</b>", small),
        ],
        [
            Paragraph("Sleeve budget", small),
            Paragraph("$115,500", small),
            Paragraph("B4 deploy target", small),
        ],
        [
            Paragraph("Opt2 weight (Step 1)", small),
            Paragraph("9.4%", small),
            Paragraph("Decay/borrow share of sleeve", small),
        ],
        [
            Paragraph("AMD conditional crash C", small),
            Paragraph("51%", small),
            Paragraph("max(tail, retrace)", small),
        ],
        [
            Paragraph("Hedge ratio h / beta β", small),
            Paragraph("0.45 / 2.0", small),
            Paragraph("Underlying hedge intensity", small),
        ],
        [
            Paragraph("Pair loss L", small),
            Paragraph("0.377", small),
            Paragraph("$ lost per $1 gross if C hits", small),
        ],
        [
            Paragraph("ρ / scale_to_budget", small),
            Paragraph("0.75% / true", small),
            Paragraph("Relative risk + refill", small),
        ],
        [
            Paragraph("Book scale_mult", small),
            Paragraph("6.47", small),
            Paragraph("budget / sum(post-cap)", small),
        ],
    ]
    t = Table(given_data, colWidths=[1.7 * inch, 1.4 * inch, 3.4 * inch])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), white),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LIGHT, white]),
                ("GRID", (0, 0), (-1, -1), 0.4, SOFT),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(t)
    story.append(Spacer(1, 10))

    story.append(Paragraph("Step 1 — Solved gross (before crash cap)", h2))
    story.append(Paragraph("solved_gross = 9.4% × $115,500 ≈ <b>$10,860</b>", ex_num))

    story.append(Paragraph("Step 2a — Per-name crash trim", h2))
    story.append(
        Paragraph(
            "Allowed crash loss (pre-scale) = ρ · budget = 0.0075 × $115,500 = <b>$866</b>",
            ex_num,
        )
    )
    story.append(Paragraph("cap_usd = $866 / 0.377 ≈ <b>$2,297</b>", ex_num))
    story.append(
        Paragraph(
            "gross_trim = min($10,860, $2,297) = <b>$2,297</b>  → cut to ~21% of the opt2 size",
            ex_num,
        )
    )

    story.append(Paragraph("Step 2b — Scale to budget (sleeve refill)", h2))
    story.append(
        Paragraph(
            "scale_mult ≈ 6.47  (whole capped book was ~15% of the $115,500 target)",
            ex_num,
        )
    )
    story.append(
        Paragraph(
            "gross_scaled = $2,297 × 6.47 ≈ <b>$14,851</b>",
            ex_num,
        )
    )
    story.append(
        Paragraph(
            "rho_effective ≈ ρ × scale_mult = 0.75% × 6.47 ≈ <b>4.85%</b> of the sleeve "
            "if AMD crashes 51%. Relative ordering is preserved: lower-L names still get "
            "more gross than higher-L names.",
            body,
        )
    )
    story.append(
        Paragraph(
            "Weight after scale ≈ $14,851 / $115,500 ≈ <b>12.9%</b> "
            "(up from the 9.4% opt2 weight because DAMD has the lowest L in the book).",
            body,
        )
    )

    story.append(Paragraph("Step 3 — Smoothing (steady-state here)", h2))
    story.append(
        Paragraph(
            "On a quiet week with no new entries / exits, smoothed ≈ capped "
            "(~$14,851). If a new name entered, DAMD's scale compression would fade at "
            "dilution_alpha = 0.25 instead of jumping; a true own-risk cut "
            "(pre-scale capacity down &gt; 10%) would still land immediately.",
            body,
        )
    )

    story.append(Paragraph("Step 4 — Two short legs", h2))
    story.append(
        Paragraph(
            "inv_short (DAMD) = $14,851 / (1 + 0.45·2.0) = $14,851 / 1.90 ≈ <b>$7,816</b>",
            ex_num,
        )
    )
    story.append(
        Paragraph(
            "und_short (AMD)  = 0.45 · 2.0 · $7,816 ≈ <b>$7,034</b>",
            ex_num,
        )
    )
    story.append(
        Paragraph(
            "Check: $7,816 + $7,034 = $14,850 ≈ gross. Check crash loss: "
            "0.377 × $14,851 ≈ $5,600 ≈ 4.85% of the sleeve.",
            body,
        )
    )

    story.append(Paragraph("Example summary", h2))
    sum_data = [
        [
            Paragraph("<b>Quantity</b>", small),
            Paragraph("<b>Opt2 solved</b>", small),
            Paragraph("<b>After trim</b>", small),
            Paragraph("<b>After scale</b>", small),
        ],
        [
            Paragraph("Pair gross", small),
            Paragraph("$10,860", small),
            Paragraph("$2,297", small),
            Paragraph("$14,851", small),
        ],
        [
            Paragraph("Sleeve weight", small),
            Paragraph("9.4%", small),
            Paragraph("2.0%", small),
            Paragraph("12.9%", small),
        ],
        [
            Paragraph("Loss if AMD −51%", small),
            Paragraph("~$4,090", small),
            Paragraph("$866 (0.75%)", small),
            Paragraph("~$5,600 (4.85%)", small),
        ],
    ]
    t2 = Table(sum_data, colWidths=[1.5 * inch, 1.5 * inch, 1.6 * inch, 1.9 * inch])
    t2.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
                ("TEXTCOLOR", (0, 0), (-1, 0), white),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LIGHT, white]),
                ("GRID", (0, 0), (-1, -1), 0.4, SOFT),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(t2)

    story.append(Spacer(1, 12))
    story.append(Paragraph("8. Why this design", h1))
    for line in [
        "• Decay score picks who deserves capital; crash budget sets relative risk; "
        "scale_to_budget fills the deploy target without breaking that ordering.",
        "• Cap is per-name and dollar-interpretable before scale; after scale, "
        "rho_effective ≈ ρ × scale_mult is the number that actually runs.",
        "• Post-cap smoothing stops entry/exit churn and scale-dilution jumps from "
        "looking like overnight risk cuts — without delaying true capacity drops.",
        "• Hedge ratio h enters L, so a tightly hedged pair is allowed more gross "
        "than a loose one with the same crash C.",
    ]:
        story.append(Paragraph(line, bullet))

    story.append(Paragraph("9. Where it lives in code / config", h1))
    story.append(
        Paragraph(
            "• Weights: <font face='Courier'>scripts/v6_b4_pf_weights.py</font><br/>"
            "• Crash cap + scale + L EMA: <font face='Courier'>scripts/b4_crash_budget.py</font><br/>"
            "• Post-cap smooth + legs + ratchet: "
            "<font face='Courier'>scripts/bucket4_weekly_opt2.py</font><br/>"
            "• Walk-forward API: <font face='Courier'>scripts/bucket4_backtest_api.py</font><br/>"
            "• Hedge/cadence: <font face='Courier'>scripts/bucket4_hedge_cadence.py</font><br/>"
            "• Knobs: <font face='Courier'>config/strategy_config.yml</font> → "
            "<font face='Courier'>inverse_decay_bucket4.rules.bucket4_weekly_opt2</font><br/>"
            "  (<font face='Courier'>crash_budget.*</font>, "
            "<font face='Courier'>weight_smoothing.*</font>)<br/>"
            "• Telemetry: <font face='Courier'>data/runs/&lt;date&gt;/b4_crash_budget.csv</font>, "
            "<font face='Courier'>b4_sizing_waterfall.csv</font>",
            body,
        )
    )

    story.append(Spacer(1, 14))
    story.append(HRFlowable(width="100%", thickness=1, color=SOFT, spaceAfter=8))
    story.append(
        Paragraph(
            "Production reference · ls-algo · Bucket 4 sizing · 11 July 2026",
            small,
        )
    )

    def _footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(SOFT)
        canvas.setLineWidth(0.5)
        canvas.line(0.75 * inch, 0.55 * inch, letter[0] - 0.75 * inch, 0.55 * inch)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(HexColor("#94a3b8"))
        canvas.drawString(0.75 * inch, 0.35 * inch, "Bucket 4 sizing method")
        canvas.drawRightString(letter[0] - 0.75 * inch, 0.35 * inch, f"Page {doc.page}")
        canvas.restoreState()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.75 * inch,
        title="Bucket 4 Sizing Method",
        author="ls-algo",
    )
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    print(f"Wrote {OUT} ({OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
