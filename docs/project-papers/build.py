#!/usr/bin/env python3
"""Build the two dated project papers and their standalone vector figures.

Run from any directory with this checkout's Python. See README.md for dependencies.
The authored Markdown is the only prose source; explicit page breaks fix its layout.
"""

from functools import partial
from html import escape
import json
from pathlib import Path
import re

from markdown_it import MarkdownIt
from pypdf import PdfReader
from reportlab.graphics import renderPDF, renderSVG
from reportlab.graphics.shapes import Circle, Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (
    BaseDocTemplate, Frame, KeepTogether, PageBreak, PageTemplate,
    Paragraph, Spacer, Table, TableStyle,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUTPUT = ROOT / "output/pdf"
WIDTH = A4[0] - 108
INK = colors.HexColor("#182c3c")
TEAL = colors.HexColor("#167b79")
MUTED = colors.HexColor("#526572")
PALE = colors.HexColor("#edf5f4")
GRID = colors.HexColor("#d8e3e5")
PARSER = MarkdownIt("commonmark").enable("table")


def label(drawing, x, y, text, size=9, color=INK, font="Helvetica", anchor="start"):
    drawing.add(String(x, y, text, fontName=font, fontSize=size,
                       fillColor=color, textAnchor=anchor))


def arrow(drawing, x1, y1, x2, y2, color=TEAL, dashed=False):
    drawing.add(Line(x1, y1, x2, y2, strokeColor=color, strokeWidth=1,
                     strokeDashArray=[3, 2] if dashed else None))
    if y1 == y2:
        sign = 1 if x2 > x1 else -1
        points = [x2, y2, x2 - sign * 5, y2 + 2.5, x2 - sign * 5, y2 - 2.5]
    else:
        sign = 1 if y2 > y1 else -1
        points = [x2, y2, x2 - 2.5, y2 - sign * 5, x2 + 2.5, y2 - sign * 5]
    drawing.add(Polygon(points, fillColor=color, strokeColor=None))


def box(drawing, x, y, width, title, detail, height=46):
    drawing.add(Rect(x, y, width, height, rx=4, ry=4, fillColor=PALE,
                     strokeColor=GRID, strokeWidth=0.6))
    label(drawing, x + width / 2, y + height - 17, title, 10,
          font="Helvetica-Bold", anchor="middle")
    label(drawing, x + width / 2, y + 12, detail, 8.5, MUTED, anchor="middle")


def learning_loop():
    drawing = Drawing(WIDTH, 94)
    gap = 19
    width = (WIDTH - 3 * gap) / 4
    for i, (title, detail) in enumerate([
        ("Play", "Current + older models"), ("Record", "Choices and outcomes"),
        ("Update", "Adjust the policy"), ("Evaluate", "Separate test games"),
    ]):
        box(drawing, i * (width + gap), 40, width, title, detail)
        if i:
            arrow(drawing, i * (width + gap) - gap + 2, 64,
                  i * (width + gap) - 2, 64, dashed=i == 3)
    start = 2 * (width + gap) + width / 2
    end = width / 2
    drawing.add(Line(start, 40, start, 14, strokeColor=TEAL))
    drawing.add(Line(start, 14, end, 14, strokeColor=TEAL))
    arrow(drawing, end, 14, end, 40)
    label(drawing, (start + end) / 2, 21, "Next policy returns to practice", 8.5,
          MUTED, anchor="middle")
    return drawing


def matrix_chart():
    data = json.loads((HERE / "matrix-data.json").read_text())
    drawing = Drawing(WIDTH, 189)
    x0, x1 = 132, WIDTH - 40
    scale = (x1 - x0) / 35
    for value in range(0, 36, 5):
        x = x0 + value * scale
        drawing.add(Line(x, 34, x, 174, strokeColor=GRID, strokeWidth=0.6))
        label(drawing, x, 20, str(value), 8, MUTED, anchor="middle")
    for i, row in enumerate(data["rows"]):
        y = 161 - i * 23
        label(drawing, 0, y - 3, row["deck"], 9)
        mean, half = row["delta_pp"], row["half_width_pp"]
        lo, hi = x0 + (mean - half) * scale, x0 + (mean + half) * scale
        drawing.add(Line(lo, y, hi, y, strokeColor=TEAL, strokeWidth=1.7))
        for x in (lo, hi):
            drawing.add(Line(x, y - 3, x, y + 3, strokeColor=TEAL, strokeWidth=1))
        drawing.add(Circle(x0 + mean * scale, y, 3.2, fillColor=TEAL, strokeColor=None))
        label(drawing, WIDTH, y - 3, f"+{mean:.1f}", 9,
              font="Helvetica-Bold", anchor="end")
    label(drawing, (x0 + x1) / 2, 3, "Jund score improvement (percentage points)",
          9, MUTED, anchor="middle")
    return drawing


def architecture():
    drawing = Drawing(WIDTH, 163)
    bw, gap = 146, (WIDTH - 438) / 2
    x = [0, bw + gap, 2 * (bw + gap)]
    box(drawing, x[0], 111, bw, "CPU rollout workers", "Rust engine + features")
    box(drawing, x[1], 111, bw, "GPU inference", "Batched policies + memory")
    box(drawing, x[2], 111, bw, "GPU learner", "Recurrent PPO updates")
    arrow(drawing, x[0] + bw, 143, x[1], 143)
    arrow(drawing, x[1], 121, x[0] + bw, 121)
    label(drawing, x[1] + bw / 2, 97, "Shared-memory requests / replies", 8,
          MUTED, anchor="middle")
    box(drawing, x[0], 39, bw, "Recorded trajectories", "Actions, values, rewards", 43)
    box(drawing, x[2], 39, bw, "Saved checkpoints", "Versioned weights + metadata", 43)
    arrow(drawing, bw / 2, 111, bw / 2, 82)
    drawing.add(Line(bw, 59, x[2] - 7, 59, strokeColor=TEAL))
    drawing.add(Line(x[2] - 7, 59, x[2] - 7, 104, strokeColor=TEAL))
    drawing.add(Line(x[2] - 7, 104, x[2] + bw / 2, 104, strokeColor=TEAL))
    arrow(drawing, x[2] + bw / 2, 104, x[2] + bw / 2, 111)
    label(drawing, x[1] + bw / 2, 67, "Training samples", 8.5, MUTED, anchor="middle")
    arrow(drawing, x[2], 134, x[1] + bw, 134)
    label(drawing, x[1] + bw + gap / 2, 161, "Weights", 8, MUTED, anchor="middle")
    arrow(drawing, x[2] + bw - 10, 111, x[2] + bw - 10, 82)
    label(drawing, 0, 15, "Separate CPU play service", 9, font="Helvetica-Bold")
    label(drawing, 0, 1, "Pinned checkpoints > browser games > replays and feedback", 8.5, MUTED)
    arrow(drawing, x[2] + bw / 2, 39, x[2] + bw / 2, 7, dashed=True)
    arrow(drawing, x[2] + bw / 2, 7, 330, 7, dashed=True)
    return drawing


FIGURES = {
    "learning-loop.svg": (learning_loop, "Figure 1. Self-play supplies experience; saved versions provide opponents. Evaluation measures the updated policy in separate games."),
    "jund-matrix.svg": (matrix_chart, "Figure 2. Sampled Jund score gain: r8-jund-pilot v01464 over r7-lr075, against r7-lr075 on each deck. 800 games per score; bars are paired 95% intervals over 200 deal seeds. Preboard, native engine, 10 October 2026. Source: [P1]."),
    "architecture.svg": (architecture, "Figure 1. Training separates simulation, inference, and optimization. Saved checkpoints also serve the independent human-play deployment."),
}

STYLES = {
    "body": ParagraphStyle("Body", fontName="Times-Roman", fontSize=11,
                           leading=14.3, textColor=INK, spaceAfter=7, alignment=TA_LEFT),
    "title": ParagraphStyle("Title", fontName="Helvetica-Bold", fontSize=23,
                            leading=26.5, textColor=INK, spaceAfter=9),
    "subtitle": ParagraphStyle("Subtitle", fontName="Times-Italic", fontSize=15,
                               leading=18, textColor=MUTED, spaceAfter=10),
    "meta": ParagraphStyle("Meta", fontName="Helvetica", fontSize=9,
                           leading=12, textColor=TEAL, spaceAfter=16),
    "h2": ParagraphStyle("Section", fontName="Helvetica-Bold", fontSize=14,
                         leading=17, textColor=TEAL, spaceAfter=10, keepWithNext=True),
    "h3": ParagraphStyle("Subsection", fontName="Helvetica-Bold", fontSize=10.5,
                         leading=13, textColor=TEAL, spaceBefore=3,
                         spaceAfter=6, keepWithNext=True),
    "ref": ParagraphStyle("Reference", fontName="Times-Roman", fontSize=9,
                          leading=11.3, textColor=INK, spaceAfter=4),
    "caption": ParagraphStyle("Caption", fontName="Helvetica", fontSize=8.5,
                              leading=11, textColor=MUTED, spaceAfter=9),
    "cell": ParagraphStyle("Cell", fontName="Helvetica", fontSize=9,
                           leading=11.4, textColor=INK),
}


def inline(token):
    parts = []
    for child in token.children or []:
        kind = child.type
        if kind == "text":
            parts.append(escape(child.content))
        elif kind in ("softbreak", "hardbreak"):
            parts.append(" " if kind == "softbreak" else "<br/>")
        elif kind == "code_inline":
            parts.append(f'<font name="Courier" size="9">{escape(child.content)}</font>')
        elif kind in ("strong_open", "strong_close", "em_open", "em_close"):
            parts.append({"strong_open": "<b>", "strong_close": "</b>",
                          "em_open": "<i>", "em_close": "</i>"}[kind])
        elif kind == "link_open":
            parts.append(f'<a href="{escape(child.attrGet("href"), quote=True)}" color="#167b79">')
        elif kind == "link_close":
            parts.append("</a>")
        else:
            raise ValueError(f"Unsupported inline Markdown: {kind}")
    return "".join(parts)


def story(source):
    tokens = PARSER.parse(source)
    result = []
    references = False
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.type == "html_block" and token.content.strip() == "<!-- pagebreak -->":
            result.append(PageBreak())
        elif token.type == "heading_open":
            text = tokens[i + 1]
            if token.tag == "h3":
                references = True
            result.append(Paragraph(inline(text), STYLES[{"h1": "title", "h2": "h2", "h3": "h3"}[token.tag]]))
            i += 2
        elif token.type == "paragraph_open":
            content = tokens[i + 1]
            images = [c for c in content.children or [] if c.type == "image"]
            if images:
                if len(content.children) != 1:
                    raise ValueError("A figure must occupy a standalone paragraph")
                name = Path(images[0].attrGet("src")).name
                draw, caption = FIGURES[name]
                result.append(KeepTogether([draw(), Spacer(1, 4), Paragraph(caption, STYLES["caption"])]))
            else:
                style = "ref" if references else "body"
                if content.content.startswith("Project paper"):
                    style = "meta"
                elif content.content == "*Inside the mtg-ml Project*":
                    style = "subtitle"
                result.append(Paragraph(inline(content), STYLES[style]))
            i += 2
        elif token.type == "table_open":
            rows = []
            while tokens[i].type != "table_close":
                if tokens[i].type == "tr_open":
                    rows.append([])
                elif tokens[i].type == "inline":
                    rows[-1].append(Paragraph(inline(tokens[i]), STYLES["cell"]))
                i += 1
            table = Table(rows, colWidths=[171, WIDTH - 171], hAlign="LEFT")
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), PALE),
                ("LINEBELOW", (0, 0), (-1, 0), 0.6, TEAL),
                ("LINEBELOW", (0, 1), (-1, -1), 0.4, GRID),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]))
            result.extend([table, Spacer(1, 8)])
        else:
            raise ValueError(f"Unsupported block Markdown: {token.type}")
        i += 1
    return result


def decorate(canvas, doc, edition):
    canvas.saveState()
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 8)
    canvas.drawString(54, A4[1] - 30, "mtg-ml  /  PROJECT PAPERS")
    canvas.drawRightString(A4[0] - 54, A4[1] - 30, "10 OCTOBER 2026")
    canvas.setStrokeColor(GRID)
    canvas.setLineWidth(0.5)
    canvas.line(54, 40, A4[0] - 54, 40)
    canvas.setFont("Helvetica", 8)
    canvas.drawString(54, 27, f"{edition} edition  |  Evidence snapshot: 18a3efa")
    canvas.drawRightString(A4[0] - 54, 27, f"{doc.page} / 5")
    canvas.restoreState()


def build(name):
    source = (HERE / f"{name}.md").read_text()
    target = OUTPUT / f"mtg-ml-{name}.pdf"
    title = next(line[2:] for line in source.splitlines() if line.startswith("# "))
    if name == "general":
        title += ": Inside the mtg-ml Project"
    doc = BaseDocTemplate(str(target), pagesize=A4, title=title, author="mtg-ml",
                          subject=f"{name.title()} project paper, 10 October 2026",
                          leftMargin=54, rightMargin=54, topMargin=54, bottomMargin=52,
                          pageCompression=1, invariant=1)
    frame = Frame(54, 52, WIDTH, A4[1] - 106, leftPadding=0, bottomPadding=0,
                  rightPadding=0, topPadding=0)
    doc.addPageTemplates(PageTemplate(id="paper", frames=[frame],
                                      onPage=partial(decorate, edition=name.title())))
    doc.build(story(source), canvasmaker=partial(Canvas, invariant=1))
    reader = PdfReader(target)
    word_count = len(re.sub(r"\]\([^)]*\)", "]", source).split())
    print(f"{target.relative_to(ROOT)}: {len(reader.pages)} pages; {word_count} source words")
    if len(reader.pages) != 5:
        raise ValueError(f"{name}: expected five pages, got {len(reader.pages)}; revise the layout/prose")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure_dir = HERE / "figures"
    figure_dir.mkdir(exist_ok=True)
    for name, (factory, _) in FIGURES.items():
        drawing = factory()
        renderSVG.drawToFile(drawing, str(figure_dir / name))
        renderPDF.drawToFile(drawing, str(figure_dir / name.replace(".svg", ".pdf")), invariant=1)
    for name in ("general", "technical"):
        build(name)


if __name__ == "__main__":
    main()
