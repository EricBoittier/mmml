"""Build a single-file PDF from the MkDocs navigation.

This script intentionally follows ``mkdocs.yml`` instead of globbing Markdown
files, so the PDF order matches the published docs site. It is a lightweight
export path for local review; the canonical documentation build is still
``mkdocs build --strict``.
"""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path
from typing import Any, Iterable

import yaml
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer


ROOT = Path(__file__).resolve().parents[1]
MKDOCS_CONFIG = ROOT / "mkdocs.yml"
DOCS_DIR = ROOT / "docs"
DEFAULT_OUTPUT = ROOT / "site" / "mmml-docs.pdf"
LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
INLINE_CODE_RE = re.compile(r"`([^`]+)`")


def iter_nav_pages(nav_items: Iterable[Any]) -> Iterable[tuple[str, Path]]:
    """Yield ``(title, docs-relative path)`` pairs from MkDocs ``nav`` entries."""
    for item in nav_items:
        if isinstance(item, str):
            yield Path(item).stem.replace("-", " ").title(), Path(item)
            continue
        if not isinstance(item, dict):
            continue
        for title, value in item.items():
            if isinstance(value, str):
                yield str(title), Path(value)
            elif isinstance(value, list):
                yield from iter_nav_pages(value)


def load_nav_pages() -> list[tuple[str, Path]]:
    """Read page order from ``mkdocs.yml``."""
    with MKDOCS_CONFIG.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return list(iter_nav_pages(config.get("nav", [])))


def inline_markdown_to_reportlab(text: str) -> str:
    """Convert simple inline Markdown to ReportLab's small HTML subset."""
    escaped = html.escape(text)
    escaped = LINK_RE.sub(r"\1 (\2)", escaped)
    return INLINE_CODE_RE.sub(r'<font face="Courier">\1</font>', escaped)


def styles() -> dict[str, ParagraphStyle]:
    """Return named styles used by the Markdown-to-PDF renderer."""
    base = getSampleStyleSheet()
    base["Normal"].fontName = "Helvetica"
    base["Normal"].fontSize = 9.5
    base["Normal"].leading = 13
    base["Code"].fontName = "Courier"
    base["Code"].fontSize = 7.5
    base["Code"].leading = 9
    base.add(
        ParagraphStyle(
            name="SourcePath",
            parent=base["Normal"],
            fontSize=7.5,
            textColor="#6b7280",
            spaceAfter=4,
        )
    )
    base.add(
        ParagraphStyle(
            name="BulletBody",
            parent=base["Normal"],
            leftIndent=14,
            firstLineIndent=-8,
        )
    )
    return base


def markdown_page_to_flowables(title: str, path: Path, style_map: dict[str, ParagraphStyle]) -> list[Any]:
    """Render the Markdown subset used by the docs into ReportLab flowables."""
    flowables: list[Any] = [
        Paragraph(html.escape(str(path)), style_map["SourcePath"]),
        Paragraph(html.escape(title), style_map["Title"]),
        Spacer(1, 0.12 * inch),
    ]
    lines = (DOCS_DIR / path).read_text(encoding="utf-8").splitlines()
    in_code_block = False
    code_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            if in_code_block:
                flowables.append(Preformatted("\n".join(code_lines), style_map["Code"]))
                flowables.append(Spacer(1, 0.08 * inch))
                code_lines = []
                in_code_block = False
            else:
                in_code_block = True
            continue
        if in_code_block:
            code_lines.append(line)
            continue
        if not stripped:
            flowables.append(Spacer(1, 0.06 * inch))
            continue
        if stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            heading_text = stripped[level:].strip()
            if heading_text == title:
                continue
            style_name = "Heading2" if level <= 2 else "Heading3"
            flowables.append(Paragraph(inline_markdown_to_reportlab(heading_text), style_map[style_name]))
            continue
        if stripped.startswith("- "):
            bullet_text = stripped[2:].strip()
            flowables.append(Paragraph(f"- {inline_markdown_to_reportlab(bullet_text)}", style_map["BulletBody"]))
            continue
        flowables.append(Paragraph(inline_markdown_to_reportlab(stripped), style_map["Normal"]))

    if code_lines:
        flowables.append(Preformatted("\n".join(code_lines), style_map["Code"]))
    return flowables


def build_pdf(pages: list[tuple[str, Path]], output: Path) -> None:
    """Write a PDF document containing all pages from the MkDocs nav."""
    style_map = styles()
    flowables: list[Any] = []
    for page_index, (title, path) in enumerate(pages):
        if page_index:
            flowables.append(PageBreak())
        flowables.extend(markdown_page_to_flowables(title, path, style_map))

    doc = SimpleDocTemplate(
        str(output),
        pagesize=letter,
        rightMargin=0.6 * inch,
        leftMargin=0.6 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
        title="MMML Documentation",
    )
    doc.build(flowables)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="PDF path to write (default: site/mmml-docs.pdf).",
    )
    args = parser.parse_args()

    pages = load_nav_pages()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    build_pdf(pages, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
