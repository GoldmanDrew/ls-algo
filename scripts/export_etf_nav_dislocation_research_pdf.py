"""One-off report generator: ETF price vs NAV dislocations (historical casebook)."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

OUT_DIR_DROPBOX = Path(r"C:\Users\werdn\Dropbox\Levered ETFs\Levered Research")
OUT_DIR_REPO = Path(__file__).resolve().parents[1] / "Levered Research"
FILENAME = "ETF_Price_vs_NAV_Dislocations_Research_Report.pdf"


def link(url: str, label: str | None = None) -> str:
    lab = label or url
    return f'<a href="{url}" color="blue">{lab}</a>'


def build_story():
    styles = getSampleStyleSheet()
    normal = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontSize=10,
        leading=13,
        alignment=TA_JUSTIFY,
        spaceAfter=6,
    )
    h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=14, spaceAfter=10)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=11, spaceBefore=12, spaceAfter=8)
    small = ParagraphStyle("Small", parent=styles["Normal"], fontSize=8, leading=10, textColor=colors.grey)

    story: list = []
    story.append(Paragraph("ETF Market Price vs. NAV: Historical Dislocations", h1))
    story.append(
        Paragraph(
            "<b>Prepared:</b> April 24, 2026 &nbsp;|&nbsp; <b>Scope:</b> US and select global examples "
            "(ETFs and, where noted, ETNs). Numbers are sourced from public articles, issuer/industry "
            "statistics, and regulator research—verify before trading or citing externally.",
            small,
        )
    )
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("Executive summary", h2))
    story.append(
        Paragraph(
            "Exchange-traded funds usually trade near intraday indicative value because authorized "
            "participant (AP) arbitrage ties secondary-market prices to the basket. In stress, gaps widen. "
            "The dominant explanations differ by asset class: <b>(1)</b> stale or model-based bond NAVs vs. "
            "continuous ETF prices; <b>(2)</b> equity market microstructure (halts, LULD, delayed opens); "
            "<b>(3)</b> constrained or suspended share creation (commodities/ETNs); <b>(4)</b> policy shocks "
            "moving ETF prices before NAV marks catch up.",
            normal,
        )
    )

    story.append(Paragraph("Methodology and double-check notes", h2))
    story.append(
        Paragraph(
            "Prior draft figures were cross-checked against the SEC staff / industry discussion in "
            f'{link("https://www.sec.gov/comments/credit-market-interconnectedness/cll10-2.pdf", "SEC comment letter excerpting ICI COVID ETF statistics")} '
            "(asset-weighted IG ~365 bps below NAV on Mar 12 and Mar 19, 2020; HY ~220 bps; government "
            "bond ETFs &gt;100 bps on Mar 11–12). "
            f'{link("https://www.advisorperspectives.com/articles/2020/03/16/fixed-income-etfs-are-trapped-in-bond-markets-liquidity-crunch", "Advisor Perspectives / Bloomberg")} '
            "reported <b>AGG</b> ~4.4% discount and <b>LQD</b> ~5% at the Mar 12, 2020 close (not identical to ICI averages because "
            "single-ticker vs. asset-weighted universe). "
            "<b>BND</b> end-of-day discount magnitudes differ by source (e.g. ~6% cited in trade press vs. "
            "FactSet analytics emphasizing <b>return gaps vs. AGG</b> rather than halts)—treat single prints as "
            f'noisy; see {link("https://insight.factset.com/bond-etf-price-discovery-is-a-mess-right-now", "FactSet (Kashner, Mar 2020)")}.',
            normal,
        )
    )

    story.append(Paragraph("Episode table (selected)", h2))
    data = [
        ["Period", "Examples", "Direction", "Typical driver"],
        [
            "Mar 2020",
            "LQD, AGG, BND, VCIT, TLT, HYG",
            "Discount to official NAV (IG large)",
            "Bond illiquidity; NAV marking lag; ETF as price discovery",
        ],
        [
            "Apr 2020",
            "LQD, HYG post-Fed",
            "Premium vs NAV / vs bid-ask fair value",
            "Policy bid + flows; lagging marks",
        ],
        ["Aug 24, 2015", "IVV, RSP, QQQ, DVY, many ETFs", "Intraday vs index / peer NAV", "LULD; halts; liquidity withdrawal"],
        ["May 6, 2010", "Broad equity ETFs", "Brief extremes", "Market-wide microstructure; some busted trades"],
        ["Autumn 2008", "AGG, BND, LAG", "Sustained premium", "Frozen bond markets; costly creations"],
        ["Apr 2020", "USO", "Large premium", "Suspended creations; position limits; contango"],
        ["Feb–Mar 2012", "TVIX (ETN)", "~90% premium to IV", "Issuer halted creations (supply cap)"],
        ["Sep 2022", "IGLT (UK gilts)", "~1.1% discount (vs own history large)", "Gilt crash / LDI selling; BoE intervention"],
    ]
    t = Table(data, colWidths=[0.9 * inch, 1.55 * inch, 1.2 * inch, 2.35 * inch])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(t)
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph("1. March–April 2020: US fixed income ETFs", h2))
    story.append(
        Paragraph(
            "<b>Verified aggregates:</b> SEC-sourced recap of ICI figures—IG corporate bond ETFs "
            "~365 bps below NAV on Mar 12 and 19; HY ~220 bps; government bond ETFs exceeded 100 bps "
            "discount on Mar 11–12 (see SEC PDF above). "
            "<b>Single-fund press data:</b> LQD ~5% discount and AGG ~4.4% discount on Mar 12 (Advisor Perspectives). "
            "<b>HYD</b> (high-yield municipal) was reported at an extreme ~19% discount same week—treat as "
            "idiosyncratic/illiquid; confirm in Bloomberg or fund accounting if used formally.",
            normal,
        )
    )
    story.append(
        Paragraph(
            f'<b>Interpretation:</b> Industry (ICI) and SEC staff argued ETFs largely functioned as designed—'
            f'secondary liquidity with price discovery—see {link("https://www.ici.org/publication/a-close-look-at-exchangetraded-funds-and-their-investors-pdf", "ICI (2025 factsheet excerpt)")}.',
            normal,
        )
    )
    story.append(
        Paragraph(
            f'<b>Fed reversal:</b> After corporate bond facilities referenced ETFs, some names flipped to '
            f'premiums; dealer analytics documented wide intraday premia (e.g. {link("https://www.marketaxess.com/pdf/axesspoint_quantifying-corporate-bond-etf-premiums-and-discounts.pdf", "MarketAxess AxessPoint, Apr 2020")}).',
            normal,
        )
    )

    story.append(Paragraph("2. August 24, 2015: US equity ETFs (microstructure)", h2))
    story.append(
        Paragraph(
            "SEC Division of Economic and Risk Analysis documented that 302 of 1,569 ETFs (~19%) hit "
            "LULD-related trading pauses; volume spikes and order-book depth drops explained much of the cross-section. "
            f'{link("https://www.sec.gov/files/determinants-etf-trading-pauses-august-24th-2015", "SEC staff paper")}. '
            "For SPX-linked products, the SEC’s equity volatility review shows IVV trading far below SPY / E-mini / "
            "computed SPY-NAV at the open before reconverging by ~9:43—"
            f'{link("https://www.sec.gov/marketstructure/research/equity_market_volatility.pdf", "SEC equity market volatility study")}. '
            "Comment letters tabulated large drawdowns from prior-day close for liquid S&amp;P 500–linked products (e.g. IVV, RSP)—"
            f'{link("https://www.sec.gov/comments/s7-11-15/s71115-38.pdf", "SEC comment letter tables")}.',
            normal,
        )
    )

    story.append(Paragraph("3. April 2020: USO (commodity)", h2))
    story.append(
        Paragraph(
            "With creations suspended and extraordinary crude futures conditions, USO traded at a large "
            "<b>premium to published NAV</b> (press widely cited ~36% premium on Apr 21, 2020 and smaller premia "
            "adjacent days). This is a primary-market constraint story, not bond-style NAV staleness—"
            f'{link("https://www.cnbc.com/2020/04/21/usos-benchmark-is-the-near-month-crude-oil-futures-contract-traded-on-the-nymex-if-the-near-month-futures-contract-is-within-two-weeks-of-expiration-the-benchmark-will-be-the-next-month-contract-to-ex.html", "CNBC")}.',
            normal,
        )
    )

    story.append(Paragraph("4. TVIX 2012 (ETN—not an ETF)", h2))
    story.append(
        Paragraph(
            "Credit Suisse halted share creation; the note then traded up to roughly <b>90% premium</b> to "
            "indicative intraday value before collapsing when issuance resumed—classic closed-end-style "
            "supply/demand decoupling—"
            f'{link("https://www.etftrends.com/2012/03/what-really-happened-with-tvix/", "ETF Trends")}; '
            f'{link("https://www.investopedia.com/articles/etfs-mutual-funds/050816/tvix-velocityshares-daily-2x-vix-shortterm-etn-who-invested.asp", "Investopedia recap")}.',
            normal,
        )
    )

    story.append(Paragraph("5. February 2018: XIV and “Volmageddon” (ETN)", h2))
    story.append(
        Paragraph(
            "Inverse VIX short-vol ETN experienced an acceleration-linked collapse (~93% one-day move cited "
            "widely) after VIX futures spike—structural termination feature, distinct from AP arbitrage gaps but "
            "relevant to “exchange-traded” vehicles—"
            f'{link("https://www.cnbc.com/2018/02/06/the-obscure-volatility-security-thats-become-the-focus-of-this-sell-off-is-halted-after-an-80-percent-plunge.html", "CNBC")}.',
            normal,
        )
    )

    story.append(Paragraph("6. September 2022: UK gilt ETF (IGLT)", h2))
    story.append(
        Paragraph(
            "During the UK mini-budget gilt crisis, iShares Core UK Gilts UCITS ETF (IGLT) reportedly closed "
            "~1.08% below NAV—small in absolute terms but described as ~180× its typical daily basis—"
            f'{link("https://www.etfstream.com/articles/blackrock-uk-gilt-etf-hits-1-discount-before-bank-of-england-intervention", "ETF Stream")}; '
            f'context: {link("https://www.chicagofed.org/publications/chicago-fed-letter/2023/480", "Chicago Fed letter")}.',
            normal,
        )
    )

    story.append(Paragraph("7. May 6, 2010 flash event", h2))
    story.append(
        Paragraph(
            "Short-lived ETF price dislocations vs. fair value; many ETFs ended near benchmark-aligned NAVs at close; "
            "policy and industry post-mortems emphasized market-wide microstructure.",
            normal,
        )
    )

    story.append(Paragraph("8. Autumn 2008: Aggregate bond ETF premiums", h2))
    story.append(
        Paragraph(
            "Trade press described persistent ~1–3% <b>premiums</b> for broad aggregate bond ETFs when bond markets "
            "seized and creation baskets were hard to assemble—opposite sign from Mar 2020 for many IG products.",
            normal,
        )
    )

    story.append(Paragraph("AMAC / policy follow-up (2020)", h2))
    story.append(
        Paragraph(
            f'The SEC Asset Management Advisory Committee flagged that a large share of taxable bond ETFs '
            f'closed at &gt;1% discount to NAV at stress peaks and recommended further SEC study—'
            f'{link("https://www.sec.gov/file/prelim-recommendations-amac-etps", "AMAC preliminary recommendations, Sep 2020")}.',
            normal,
        )
    )

    story.append(Paragraph("Disclaimer", h2))
    story.append(
        Paragraph(
            "This memorandum is for research organization only. It is not investment advice. Figures are "
            "second-hand from cited public sources and may omit intraday nuance, fair-value adjustments, "
            "and share-class differences.",
            small,
        )
    )

    return story


def main() -> None:
    story = build_story()
    for d in (OUT_DIR_DROPBOX, OUT_DIR_REPO):
        d.mkdir(parents=True, exist_ok=True)
        path = d / FILENAME
        doc = SimpleDocTemplate(
            str(path),
            pagesize=letter,
            rightMargin=0.75 * inch,
            leftMargin=0.75 * inch,
            topMargin=0.75 * inch,
            bottomMargin=0.75 * inch,
        )
        doc.build(story)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
