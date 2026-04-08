from __future__ import annotations

import html
import sys
from pathlib import Path

from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, PageBreak


def build_pdf(md_path: Path, pdf_path: Path) -> None:
    styles = getSampleStyleSheet()
    body = ParagraphStyle(
        'BodySmall',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=8,
        leading=10,
        alignment=TA_LEFT,
        spaceAfter=1,
    )
    mono = ParagraphStyle(
        'MonoSmall',
        parent=body,
        fontName='Courier',
        fontSize=7,
        leading=9,
    )
    h1 = ParagraphStyle('H1', parent=styles['Heading1'], fontName='Helvetica-Bold', fontSize=16, leading=19, spaceAfter=6)
    h2 = ParagraphStyle('H2', parent=styles['Heading2'], fontName='Helvetica-Bold', fontSize=13, leading=16, spaceAfter=4)
    h3 = ParagraphStyle('H3', parent=styles['Heading3'], fontName='Helvetica-Bold', fontSize=11, leading=14, spaceAfter=3)
    h4 = ParagraphStyle('H4', parent=styles['Heading4'], fontName='Helvetica-Bold', fontSize=9.5, leading=12, spaceAfter=2)
    bullet = ParagraphStyle('Bullet', parent=body, leftIndent=10, firstLineIndent=-7)

    story = []
    for raw in md_path.read_text(encoding='utf-8').splitlines():
        line = raw.rstrip()
        if not line:
            story.append(Spacer(1, 3))
            continue
        if line.startswith('<a id='):
            continue
        if line.strip() == '---':
            story.append(Spacer(1, 6))
            continue
        if line.startswith('# '):
            story.append(Paragraph(html.escape(line[2:].strip()), h1))
            continue
        if line.startswith('## '):
            text = line[3:].strip()
            if text.startswith('RTLv3-'):
                if story:
                    story.append(PageBreak())
            story.append(Paragraph(html.escape(text), h2))
            continue
        if line.startswith('### '):
            story.append(Paragraph(html.escape(line[4:].strip()), h3))
            continue
        if line.startswith('#### '):
            story.append(Paragraph(html.escape(line[5:].strip()), h4))
            continue
        if line.startswith('<details>') or line.startswith('</details>'):
            continue
        if line.startswith('<summary>') and line.endswith('</summary>'):
            text = line.replace('<summary>', '').replace('</summary>', '')
            text = text.replace('<strong>', '').replace('</strong>', '')
            story.append(Paragraph(html.escape(text), h4))
            continue
        if line.startswith('|'):
            story.append(Paragraph(html.escape(line).replace(' ', '&nbsp;'), mono))
            continue
        if line.startswith('- '):
            story.append(Paragraph('• ' + html.escape(line[2:].strip()), bullet))
            continue
        story.append(Paragraph(html.escape(line), body))

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=12 * mm,
        rightMargin=12 * mm,
        topMargin=12 * mm,
        bottomMargin=12 * mm,
        title=md_path.stem,
    )
    doc.build(story)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit('Usage: markdown_to_pdf_basic.py input.md output.pdf')
    build_pdf(Path(sys.argv[1]), Path(sys.argv[2]))
