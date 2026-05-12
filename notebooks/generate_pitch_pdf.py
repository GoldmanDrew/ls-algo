"""
Investor PDF (3 pages): cover + exec summary, diversification, performance dashboard.

Outputs:
  data/backtest/pitch/ETF_Decay_Sleeve_Pitch.pdf
  data/backtest/pitch/figures/fig_pairs_per_underlying_histogram.png

Requires: matplotlib, pandas, openpyxl, reportlab
"""
from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

NOTEBOOK_ROOT = Path(__file__).resolve().parent
PITCH = NOTEBOOK_ROOT / "data" / "backtest" / "pitch"
FIG_DIR = PITCH / "figures"
PORTFOLIO_XLSX = NOTEBOOK_ROOT / "data" / "backtest" / "Institutional_B1_Sample_Portfolio.xlsx"
METRICS_XLSX = NOTEBOOK_ROOT / "data" / "backtest" / "strategy_vs_spy.xlsx"
DASHBOARD_NAME = "performance_dashboard.png"
DASHBOARD_CANDIDATES = [
    PITCH / DASHBOARD_NAME,
    PITCH / "v1_performance_dashboard.png",
    Path(
        r"C:\Users\werdn\.cursor\projects\c-Users-werdn-Documents-Investing-ls-algo-notebooks"
        r"\assets\c__Users_werdn_AppData_Roaming_Cursor_User_workspaceStorage_empty-window_images_image-39062cc4-d3d8-4c23-af15-01d24222447f.png"
    ),
]


def _ensure_dashboard() -> Path:
    PITCH.mkdir(parents=True, exist_ok=True)
    dst = PITCH / DASHBOARD_NAME
    if not dst.exists():
        for src in DASHBOARD_CANDIDATES:
            if src.exists() and src.resolve() != dst.resolve():
                shutil.copy2(src, dst)
                break
    if not dst.exists():
        raise FileNotFoundError(
            f"Performance dashboard PNG missing. Place {DASHBOARD_NAME} in {PITCH}"
        )
    return dst


def load_portfolio_df() -> pd.DataFrame:
    if not PORTFOLIO_XLSX.exists():
        raise FileNotFoundError(f"Portfolio workbook not found: {PORTFOLIO_XLSX}")
    return pd.read_excel(PORTFOLIO_XLSX, sheet_name="positions")


def load_full_sample_metrics() -> pd.DataFrame | None:
    if not METRICS_XLSX.exists():
        return None
    try:
        return pd.read_excel(METRICS_XLSX, sheet_name="FullSample_Summary")
    except Exception:
        return None


def generate_histogram_pairs_per_underlying(df: pd.DataFrame, out_hist: Path) -> tuple[int, int]:
    und_col = "Underlying Ticker"
    n_pairs = len(df)
    n_und = df[und_col].nunique()
    counts = df.groupby(und_col, observed=False).size()

    rcParams.update({"font.size": 11, "axes.titlesize": 12})
    fig1, ax1 = plt.subplots(figsize=(8.2, 5.9), constrained_layout=True)
    ax1.hist(
        counts.values,
        bins=np.arange(0.5, counts.max() + 1.5, 1),
        color="#3949AB",
        edgecolor="white",
        alpha=0.88,
    )
    ax1.set_xlabel("Pairs per underlying")
    ax1.set_ylabel("Number of underlyings")
    ax1.set_title(
        f"Pair concentration across underlyings\n({n_pairs} pairs, {n_und} distinct underlyings)",
        fontsize=12,
        fontweight="semibold",
    )
    ax1.grid(True, axis="y", alpha=0.28)
    fig1.savefig(out_hist, dpi=175, bbox_inches="tight", facecolor="white")
    plt.close(fig1)
    return n_pairs, n_und


def _pct(x: float | None) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"{100.0 * float(x):.2f}%"


def build_full_sample_table(fs: pd.DataFrame | None) -> Table | Paragraph:
    levs = [2.5, 3.0, 3.5, 4.0]
    if fs is None or fs.empty:
        return Paragraph(
            "<i>Full-sample metrics table unavailable. Run the Strategy vs SPY notebook cell to write "
            "<b>strategy_vs_spy.xlsx</b> (sheet FullSample_Summary), then regenerate this PDF.</i>",
            ParagraphStyle(
                name="Note",
                parent=getSampleStyleSheet()["Normal"],
                fontSize=9,
                textColor=colors.HexColor("#546e7a"),
            ),
        )

    spy_row = fs[fs["series"].astype(str).str.upper().eq("SPY")]
    if spy_row.empty:
        return Paragraph("<i>SPY row missing from FullSample_Summary.</i>", getSampleStyleSheet()["Normal"])

    spy = spy_row.iloc[0]
    col_hdr = ["Metric", "SPY"] + [f"{lv:g}x gross" for lv in levs]
    rows_tbl = [col_hdr]

    spy_cagr = spy.get("full_sample_annualized_return")
    spy_vol = spy.get("full_sample_annualized_vol")
    spy_dd = spy.get("full_sample_max_drawdown")

    def strat_row(lev: float):
        m = fs[pd.to_numeric(fs["gross_leverage"], errors="coerce").sub(lev).abs() < 0.01]
        return m.iloc[0] if len(m) else None

    cache = {lv: strat_row(lv) for lv in levs}

    rows_tbl.append(
        ["CAGR (full sample)", _pct(spy_cagr)]
        + [_pct(cache[lv].get("full_sample_annualized_return")) if cache[lv] is not None else "-" for lv in levs]
    )
    rows_tbl.append(
        ["Ann. volatility", _pct(spy_vol)]
        + [_pct(cache[lv].get("full_sample_annualized_vol")) if cache[lv] is not None else "-" for lv in levs]
    )
    rows_tbl.append(
        ["Max drawdown", _pct(spy_dd)]
        + [_pct(cache[lv].get("full_sample_max_drawdown")) if cache[lv] is not None else "-" for lv in levs]
    )

    ncols = len(col_hdr)
    cw = [78] + [52] * (ncols - 1)
    t = Table(rows_tbl, repeatRows=1, colWidths=cw)
    t.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eceff1")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#263238")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cfd8dc")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fafafa")]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    return t


def build_pdf(
    dashboard_path: Path,
    fig_hist: Path,
    portfolio_df: pd.DataFrame,
    fs_df: pd.DataFrame | None,
    n_pairs: int,
    n_und: int,
    out_pdf: Path,
) -> None:
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="TitleDoc",
            parent=styles["Heading1"],
            fontSize=20,
            alignment=TA_CENTER,
            spaceAfter=12,
            textColor=colors.HexColor("#1a237e"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyJustify",
            parent=styles["Normal"],
            fontSize=10,
            leading=13,
            alignment=TA_LEFT,
            spaceAfter=8,
        )
    )

    doc = SimpleDocTemplate(
        str(out_pdf),
        pagesize=LETTER,
        leftMargin=0.65 * inch,
        rightMargin=0.65 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
    )
    story: list = []
    usable_w = LETTER[0] - doc.leftMargin - doc.rightMargin

    # ----- Page 1: cover + executive summary -----
    story.append(Spacer(1, 0.35 * inch))
    story.append(Paragraph("Systematic ETF decay sleeve", styles["TitleDoc"]))
    story.append(Paragraph("(Illustrative investor overview)", styles["Heading2"]))
    story.append(Spacer(1, 0.18 * inch))
    story.append(
        Paragraph(
            "<b>Sleeve objective.</b> Harvest structural headwinds embedded in daily-reset "
            "leveraged and structured ETFs versus hedges, while targeting muted directional beta.",
            styles["BodyJustify"],
        )
    )
    story.append(
        Paragraph(
            "<b>Reporting.</b> Gross leverage scenarios from 2.5x to 4x. Benchmark is SPY "
            "(total return) for context only.",
            styles["BodyJustify"],
        )
    )
    story.append(
        Paragraph(
            "<b>Performance exhibit.</b> Backtest window begins <b>January 2024</b> "
            "(see final page). Open calendar years show return as YTD.",
            styles["BodyJustify"],
        )
    )
    story.append(Spacer(1, 0.14 * inch))
    story.append(Paragraph("<b>Executive summary</b>", styles["Heading2"]))
    story.append(
        Paragraph(
            "Leveraged and inverse ETFs are engineered around <b>daily percentage</b> exposure targets. "
            "Compounding through resets creates <b>path dependence</b>: multi-period returns diverge from "
            "naive beta-times-underlying intuition, especially when volatility clusters.",
            styles["BodyJustify"],
        )
    )
    story.append(
        Paragraph(
            "The sleeve sells decay-heavy ETF legs and hedges with underlying (or tight proxies), aiming to "
            "isolate structural decay while controlling directional drift. Borrow, financing, liquidity, and "
            "gap risk remain material: diversification and sizing discipline matter.",
            styles["BodyJustify"],
        )
    )
    story.append(Spacer(1, 0.12 * inch))
    story.append(
        Paragraph(
            "<i>Confidential and illustrative only. Not an offer or solicitation. "
            "Past or modeled performance is not indicative of future results.</i>",
            styles["BodyJustify"],
        )
    )
    story.append(PageBreak())

    # ----- Page 2: diversification -----
    story.append(Paragraph("Diversification and path dependence", styles["Heading2"]))
    story.append(
        Paragraph(
            f"The modeled sleeve spans <b>{n_pairs} pairs</b> across <b>{n_und} underlyings</b>. "
            "Breadth reduces single-name gap risk but does not remove regime shifts or financing shocks.",
            styles["BodyJustify"],
        )
    )
    story.append(Spacer(1, 0.1 * inch))
    story.append(Image(str(fig_hist), width=usable_w * 0.95, height=3.55 * inch))

    gross_col = "Target Pair Gross (USD)"
    cols_show = [
        "ETF Ticker",
        "Underlying Ticker",
        "Book Gross Weight",
        gross_col,
        "Target Short as % of Average Daily Volume (USD)",
        "Target Short as % of ETF AUM",
    ]
    sub = portfolio_df.copy()
    sub["_gg"] = pd.to_numeric(sub[gross_col], errors="coerce")
    sub = sub.nlargest(15, "_gg")[cols_show]
    story.append(Spacer(1, 0.14 * inch))
    story.append(Paragraph("<b>Top 15 pairs by gross</b> (liquidity snapshot)", styles["Heading3"]))
    tbl_data = [["ETF", "Und", "Wt", "Gross", "%ADV", "%AUM"]]
    for _, row in sub.iterrows():
        tbl_data.append(
            [
                str(row["ETF Ticker"]),
                str(row["Underlying Ticker"]),
                f'{float(row["Book Gross Weight"]):.3f}',
                f'{float(row[gross_col])/1e6:.2f}M',
                f'{float(row["Target Short as % of Average Daily Volume (USD)"])*100:.2f}%',
                f'{float(row["Target Short as % of ETF AUM"])*100:.2f}%',
            ]
        )

    tliq = Table(tbl_data, repeatRows=1, colWidths=[44, 44, 28, 44, 44, 44])
    tliq.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 6),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eceff1")),
                ("GRID", (0, 0), (-1, -1), 0.2, colors.HexColor("#cfd8dc")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fafafa")]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    story.append(Spacer(1, 0.06 * inch))
    story.append(tliq)
    story.append(PageBreak())

    # ----- Page 3: performance -----
    story.append(Paragraph("Performance vs SPY (total return)", styles["Heading2"]))
    story.append(
        Paragraph(
            "<b>Inception through latest backtest date.</b> Full-sample CAGR, annualized volatility, "
            "and max drawdown versus SPY at each reported gross multiple.",
            styles["BodyJustify"],
        )
    )
    story.append(Spacer(1, 0.06 * inch))
    fs_block = build_full_sample_table(fs_df)
    story.append(fs_block)
    story.append(Spacer(1, 0.12 * inch))
    story.append(
        Paragraph(
            "<b>2024 start detail.</b> The chart shows annualized returns for full calendar years and "
            "<b>YTD</b> for the open year across gross scenarios.",
            styles["BodyJustify"],
        )
    )
    story.append(Spacer(1, 0.06 * inch))
    story.append(Image(str(dashboard_path), width=usable_w, height=6.35 * inch))

    doc.build(story)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    dashboard = _ensure_dashboard()
    fig_hist = FIG_DIR / "fig_pairs_per_underlying_histogram.png"

    df = load_portfolio_df()
    n_pairs, n_und = generate_histogram_pairs_per_underlying(df, fig_hist)
    fs_df = load_full_sample_metrics()

    out_pdf = PITCH / "ETF_Decay_Sleeve_Pitch.pdf"
    build_pdf(dashboard, fig_hist, df, fs_df, n_pairs, n_und, out_pdf)

    print(f"Wrote PDF: {out_pdf.resolve()}")
    print(f"Histogram: {fig_hist.resolve()}")
    print(f"Pairs / underlyings: {n_pairs} / {n_und}")
    if fs_df is None:
        print("Note: strategy_vs_spy.xlsx not found; full-sample metrics table omitted.")


if __name__ == "__main__":
    main()
