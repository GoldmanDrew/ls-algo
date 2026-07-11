"""One-off: write docs/b4_sizing_method.pdf (simple guide + worked example)."""
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
    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph("Bucket 4 Sizing Method", title))
    story.append(
        Paragraph(
            "Inverse-decay pairs · How we decide how big each position should be<br/>"
            "Simple guide with a worked example · Production as of July 2026",
            subtitle,
        )
    )
    story.append(HRFlowable(width="100%", thickness=1.5, color=TEAL, spaceAfter=14))

    story.append(
        Paragraph(
            "Bucket 4 shorts inverse ETFs (products that lose value when a stock or sector "
            "rises) and hedges them by also shorting the underlying. The goal is to harvest "
            "structural decay while keeping crash risk under a hard dollar budget.",
            body,
        )
    )
    story.append(
        Paragraph(
            "One rule to remember: if a name's conditional crash hits, that pair may lose "
            "at most 0.75% of the Bucket 4 sleeve budget.",
            callout,
        )
    )

    story.append(Paragraph("1. The big picture (four steps)", h1))
    story.append(
        Paragraph(
            "Sizing runs in a fixed order. Later steps can only shrink positions — they never "
            "inflate a risky name by redistributing freed dollars.",
            body,
        )
    )
    for t, d in [
        (
            "Step A — Score the book",
            "Rank pairs by expected decay after borrow costs. Convert scores into weights that sum to 100%.",
        ),
        (
            "Step B — Cap crash risk",
            "Estimate how much each pair would lose in a crash. Cap gross so that loss ≤ 0.75% of the sleeve.",
        ),
        (
            "Step C — Split into two shorts",
            "Turn each pair's gross into an inverse-ETF short and an underlying short using the hedge ratio h.",
        ),
        (
            "Step D — Ratchet (inventory guard)",
            "Never blindly cover hard-to-relocate inverse inventory; trim gradually when gross has crept up.",
        ),
    ]:
        story.append(Paragraph(t, step_title))
        story.append(Paragraph(d, bullet))

    story.append(Paragraph("2. Sleeve budget", h1))
    story.append(
        Paragraph(
            "Bucket 4 gets a fixed share of book target gross (config: "
            "<font face='Courier'>inverse_decay_bucket4.target_weight</font>, typically ~3%). "
            "All pair sizes are fractions of that sleeve budget.",
            body,
        )
    )
    story.append(
        Paragraph(
            "Example sleeve used below: <b>$115,500</b> (illustrative).",
            body,
        )
    )

    story.append(Paragraph("3. Step A — Decay / borrow weights", h1))
    story.append(
        Paragraph(
            "Crash risk is <b>not</b> in this score anymore. The score only answers: "
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
            "Then weights = base_score / sum(scores), clipped to a min/max per name "
            "(default 0.5%–35%), then lightly tilted for covariance concentration so "
            "highly correlated pairs do not all get large weights at once.",
            body,
        )
    )
    story.append(
        Paragraph(
            "Production knobs (live): borrow_linear_aversion = 1.5, "
            "borrow_uncertainty_penalty = 3.0, decay_borrow_quad = 0 "
            "(quadratic off because net edge already nets expected borrow).",
            small,
        )
    )

    story.append(Paragraph("4. Step B — Conditional crash budget (the risk rule)", h1))
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
            "Why this shape: on a crash of size C, the short inverse loses roughly β·C on its "
            "notional; the short-underlying hedge recovers fraction h of that. With "
            "inv = gross / (1 + h·β), net loss per gross dollar is the formula above. "
            "The (1 + φ·C) term bumps for multi-day compounding on daily-rebalanced inverses.",
            body,
        )
    )
    story.append(Paragraph("Cap (trim-only, freed dollars stay in cash):", body))
    story.append(
        Paragraph(
            "cap_usd = ρ · sleeve_budget / max(L, L_floor)<br/>"
            "gross   = min(solved_gross, cap_usd)<br/>"
            "ρ = 0.75%,   L_floor = 2%",
            formula,
        )
    )
    story.append(
        Paragraph(
            "Important: we do <b>not</b> redistribute freed weight into other pairs. "
            "Redeploying would concentrate the book and cancel the safety rule.",
            body,
        )
    )

    story.append(Paragraph("5. Step C — Split gross into two legs", h1))
    story.append(
        Paragraph(
            "gross = pair_weight × sleeve_budget<br/>"
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

    story.append(Paragraph("6. Step D — Inverse ratchet (brief)", h1))
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
            "Numbers below are stylized from a live July 2026 run shape so the arithmetic is easy "
            "to follow. Sleeve budget = <b>$115,500</b>. Pair = short DAMD (inverse AMD) + short AMD.",
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
            Paragraph("B4 target gross", small),
        ],
        [
            Paragraph("Opt2 weight (Step A)", small),
            Paragraph("10.9%", small),
            Paragraph("Decay/borrow share of sleeve", small),
        ],
        [
            Paragraph("AMD price vs 252d median", small),
            Paragraph("+140% run-up", small),
            Paragraph("Stretched vs anchor", small),
        ],
        [
            Paragraph("Historical tail", small),
            Paragraph("51%", small),
            Paragraph("Worst 20d drop + downside vol", small),
        ],
        [
            Paragraph("Hedge ratio h", small),
            Paragraph("0.45", small),
            Paragraph("Underlying hedge intensity", small),
        ],
        [
            Paragraph("Inverse beta β", small),
            Paragraph("2.0", small),
            Paragraph("|beta| of −2× inverse", small),
        ],
        [
            Paragraph("ρ / θ / φ", small),
            Paragraph("0.75% / 0.5 / 0.5", small),
            Paragraph("Live crash-budget knobs", small),
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

    story.append(Paragraph("Step A — Solved gross (before crash cap)", h2))
    story.append(Paragraph("solved_gross = 10.9% × $115,500 = <b>$12,590</b>", ex_num))

    story.append(Paragraph("Step B — Conditional crash C", h2))
    story.append(Paragraph("runup = 1.40", ex_num))
    story.append(
        Paragraph(
            "retrace = 0.5 × 1.40 / (1 + 1.40) = 0.5 × 0.583 = <b>0.292</b> (29%)",
            ex_num,
        )
    )
    story.append(Paragraph("tail = 0.51 (51%)", ex_num))
    story.append(
        Paragraph("C = max(0.51, 0.292) = <b>0.51</b>  → historical tail binds", ex_num)
    )
    story.append(
        Paragraph(
            "(If AMD had a quiet crash history but the same run-up, retrace would bind instead — "
            "that is the “AMD at highs” protection.)",
            small,
        )
    )

    story.append(Paragraph("Step B — Pair loss L and the dollar cap", h2))
    story.append(
        Paragraph(
            "unhedged fraction factor = (1 − 0.45) · 2.0 / (1 + 0.45·2.0)",
            ex_num,
        )
    )
    story.append(
        Paragraph(
            "                     = 0.55 · 2.0 / 1.90 = 1.10 / 1.90 = <b>0.579</b>",
            ex_num,
        )
    )
    story.append(
        Paragraph(
            "L = 0.579 · 0.51 · (1 + 0.5·0.51) = 0.579 · 0.51 · 1.255 ≈ <b>0.370</b>",
            ex_num,
        )
    )
    story.append(
        Paragraph(
            "Meaning: each $1 of pair gross loses about <b>$0.37</b> if AMD crashes 51%.",
            body,
        )
    )
    story.append(
        Paragraph(
            "Allowed crash loss = ρ · budget = 0.0075 × $115,500 = <b>$866</b>",
            ex_num,
        )
    )
    story.append(Paragraph("cap_usd = $866 / 0.370 ≈ <b>$2,340</b>", ex_num))
    story.append(
        Paragraph(
            "gross_final = min($12,590, $2,340) = <b>$2,340</b>  → cut to ~19% of the opt2 size",
            ex_num,
        )
    )
    story.append(
        Paragraph(
            "The other ~$10,250 stays in cash inside the sleeve. It is not handed to quieter names.",
            body,
        )
    )

    story.append(Paragraph("Step C — Two short legs", h2))
    story.append(
        Paragraph(
            "inv_short (DAMD) = $2,340 / (1 + 0.45·2.0) = $2,340 / 1.90 ≈ <b>$1,232</b>",
            ex_num,
        )
    )
    story.append(
        Paragraph(
            "und_short (AMD)  = 0.45 · 2.0 · $1,232 ≈ <b>$1,109</b>",
            ex_num,
        )
    )
    story.append(
        Paragraph(
            "Check: $1,232 + $1,109 = $2,341 ≈ gross. Check crash loss: "
            "0.370 × $2,340 ≈ $866 = exactly the ρ budget.",
            body,
        )
    )

    story.append(Paragraph("Example summary", h2))
    sum_data = [
        [
            Paragraph("<b>Quantity</b>", small),
            Paragraph("<b>Before crash cap</b>", small),
            Paragraph("<b>After crash cap</b>", small),
        ],
        [
            Paragraph("Pair gross", small),
            Paragraph("$12,590", small),
            Paragraph("$2,340", small),
        ],
        [
            Paragraph("DAMD short", small),
            Paragraph("~$6,626", small),
            Paragraph("$1,232", small),
        ],
        [
            Paragraph("AMD short", small),
            Paragraph("~$5,964", small),
            Paragraph("$1,109", small),
        ],
        [
            Paragraph("Loss if AMD −51%", small),
            Paragraph("~$4,660", small),
            Paragraph("$866 (0.75% of sleeve)", small),
        ],
    ]
    t2 = Table(sum_data, colWidths=[1.8 * inch, 2.0 * inch, 2.7 * inch])
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
        "• Decay score picks who deserves capital; crash budget decides how much is safe.",
        "• Cap is per-name and dollar-interpretable: “this name may not lose more than ρ of the sleeve.”",
        "• Trim-only + no redeploy prevents the optimizer from sneaking risk back into the book.",
        "• Hedge ratio h enters L, so a tightly hedged pair is allowed more gross than a loose one "
        "with the same crash C.",
    ]:
        story.append(Paragraph(line, bullet))

    story.append(Paragraph("9. Where it lives in code / config", h1))
    story.append(
        Paragraph(
            "• Weights: <font face='Courier'>scripts/v6_b4_pf_weights.py</font><br/>"
            "• Crash cap: <font face='Courier'>scripts/b4_crash_budget.py</font><br/>"
            "• Legs + ratchet: <font face='Courier'>scripts/bucket4_weekly_opt2.py</font><br/>"
            "• Hedge/cadence: <font face='Courier'>scripts/bucket4_hedge_cadence.py</font><br/>"
            "• Knobs: <font face='Courier'>config/strategy_config.yml</font> → "
            "<font face='Courier'>inverse_decay_bucket4.rules.bucket4_weekly_opt2</font><br/>"
            "• Telemetry: <font face='Courier'>data/runs/&lt;date&gt;/b4_crash_budget.csv</font>",
            body,
        )
    )

    story.append(Spacer(1, 16))
    story.append(HRFlowable(width="100%", thickness=1, color=SOFT, spaceAfter=8))
    story.append(
        Paragraph(
            "Production reference · ls-algo · Bucket 4 sizing · July 2026",
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
