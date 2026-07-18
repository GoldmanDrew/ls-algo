"""Render the SPY30 Puts 2x fund-summary Markdown as a paginated PDF."""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from fpdf import FPDF, XPos, YPos

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "b5_spy30_puts2x_fund_summary.md"
OUTPUT = ROOT / "docs" / "b5_spy30_puts2x_fund_summary.pdf"


def clean(value: str) -> str:
    """Use a core-font-safe subset while repairing known mojibake."""
    replacements = {
        "�?Ts": "'s", "�?T": "'", "�?o": '"', "�??": '"',
        "–": "-", "—": "-", "−": "-", "’": "'", "‘": "'",
        "“": '"', "”": '"', "…": "...", "•": "-", "×": "x",
        "≤": "<=", "≥": ">=", "%": "%",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    value = unicodedata.normalize("NFKD", value)
    return value.encode("ascii", "ignore").decode("ascii").strip()


def inline(value: str) -> str:
    value = clean(value)
    value = re.sub(r"!\[([^]]*)\]\([^)]+\)", r"\1", value)
    value = re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", value)
    value = value.replace("**", "").replace("__", "").replace("`", "")
    return value


class ReportPDF(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "B", 8)
            self.set_text_color(95, 95, 95)
            self.cell(0, 5, "SPY30 Puts 2x | Research summary", align="R")
            self.ln(4)

    def footer(self):
        self.set_y(-13)
        self.set_draw_color(190, 190, 190)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.set_y(-10)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(100, 100, 100)
        self.cell(0, 5, f"Research document | Page {self.page_no()}", align="C")

    def ensure_space(self, height: float):
        if self.get_y() + height > self.page_break_trigger:
            self.add_page()

    def paragraph(self, text: str, indent: float = 0, font_size: float = 10):
        text = inline(text)
        if not text:
            return
        self.ensure_space(10)
        self.set_x(self.l_margin + indent)
        self.set_font("Helvetica", "", font_size)
        self.set_text_color(35, 35, 35)
        self.multi_cell(self.epw - indent, 5.1, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(1.4)

    def heading(self, text: str, level: int):
        sizes = {1: 21, 2: 15, 3: 12}
        gaps = {1: 5, 2: 5, 3: 3}
        self.ensure_space(18)
        self.ln(gaps.get(level, 2))
        self.set_font("Helvetica", "B", sizes.get(level, 11))
        self.set_text_color(20, 55, 90)
        self.multi_cell(self.epw, sizes.get(level, 11) * 0.42, inline(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(1.5)

    def table(self, lines: list[str]):
        rows = [[inline(cell.strip()) for cell in line.strip().strip("|").split("|")] for line in lines]
        if len(rows) < 2:
            return
        header, data = rows[0], rows[2:]
        cols = len(header)
        widths = [self.epw / cols] * cols
        if cols == 2:
            widths = [self.epw * 0.45, self.epw * 0.55]
        elif cols == 3:
            widths = [self.epw * 0.38, self.epw * 0.31, self.epw * 0.31]

        def draw_row(row: list[str], is_header: bool = False):
            row = (row + [""] * cols)[:cols]
            line_height = 4.1
            counts = [max(1, len(self.multi_cell(widths[i] - 3, line_height, cell, dry_run=True, output="LINES"))) for i, cell in enumerate(row)]
            height = max(counts) * line_height + 3
            if self.get_y() + height > self.page_break_trigger:
                self.add_page()
                draw_row(header, True)
            y = self.get_y()
            x = self.l_margin
            self.set_draw_color(185, 195, 205)
            if is_header:
                self.set_fill_color(31, 78, 121)
                self.set_text_color(255, 255, 255)
                self.set_font("Helvetica", "B", 8.5)
            else:
                self.set_fill_color(247, 249, 251)
                self.set_text_color(35, 35, 35)
                self.set_font("Helvetica", "", 8.3)
            for i, cell in enumerate(row):
                self.rect(x, y, widths[i], height, style="DF" if is_header else "D")
                self.set_xy(x + 1.5, y + 1.5)
                self.multi_cell(widths[i] - 3, line_height, cell, new_x=XPos.RIGHT, new_y=YPos.TOP)
                x += widths[i]
            self.set_y(y + height)

        self.ensure_space(16)
        draw_row(header, True)
        for row in data:
            draw_row(row)
        self.ln(3)


def render(markdown: str) -> ReportPDF:
    pdf = ReportPDF(format="letter")
    pdf.set_auto_page_break(auto=True, margin=17)
    pdf.set_margins(18, 17, 18)
    pdf.set_title("SPY30 Puts 2x")
    pdf.set_author("ls-algo research")
    pdf.add_page()

    lines = markdown.splitlines()
    i = 0
    while i < len(lines):
        raw = lines[i].rstrip()
        stripped = raw.strip()
        if not stripped or stripped == "---":
            i += 1
            continue
        if stripped.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            pdf.table(table_lines)
            continue
        match = re.match(r"^(#{1,3})\s+(.+)$", stripped)
        if match:
            pdf.heading(match.group(2), len(match.group(1)))
        elif re.match(r"^[-*+]\s+", stripped):
            pdf.paragraph("-  " + re.sub(r"^[-*+]\s+", "", stripped), indent=4)
        elif re.match(r"^\d+\.\s+", stripped):
            pdf.paragraph(stripped, indent=4)
        else:
            pdf.paragraph(stripped)
        i += 1
    return pdf


if __name__ == "__main__":
    document = render(SOURCE.read_text(encoding="utf-8"))
    document.output(str(OUTPUT))
    print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size:,} bytes, {document.page_no()} pages)")
