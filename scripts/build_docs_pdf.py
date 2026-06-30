"""Build a single-file PDF from the MkDocs navigation.

This script intentionally follows ``mkdocs.yml`` instead of globbing Markdown
files, so the PDF order matches the published docs site. It is a lightweight
export path for local review; the canonical documentation build is still
``mkdocs build --strict``.
"""

from __future__ import annotations

import argparse
import html
import io
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable

import yaml
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
MKDOCS_CONFIG = ROOT / "mkdocs.yml"
DOCS_DIR = ROOT / "docs"
DEFAULT_OUTPUT = ROOT / "site" / "mmml-docs.pdf"
LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
INLINE_CODE_RE = re.compile(r"`([^`]+)`")
MERMAID_CLI_PACKAGE = "@mermaid-js/mermaid-cli"
TABLE_SEPARATOR_RE = re.compile(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?$")


class MkDocsConfigLoader(yaml.SafeLoader):
    """YAML loader that tolerates MkDocs extension function tags."""


def _ignore_python_name_tag(loader: yaml.Loader, tag_suffix: str, node: yaml.Node) -> str:
    """Keep local ``!!python/name`` MkDocs tags as strings for nav parsing."""
    return f"!!python/name:{tag_suffix}"


MkDocsConfigLoader.add_multi_constructor("tag:yaml.org,2002:python/name:", _ignore_python_name_tag)


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
        config = yaml.load(handle, Loader=MkDocsConfigLoader)
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


def mermaid_render_command() -> list[str] | None:
    """Return a Mermaid CLI command prefix.

    CI installs ``mmdc`` directly. Local ``npx`` fallback is opt-in because
    spawning package resolution for every diagram can make PDF builds appear
    hung on machines without a cached Mermaid CLI.
    """
    if shutil.which("mmdc"):
        return ["mmdc"]
    if os.environ.get("MMML_DOCS_PDF_ALLOW_NPX") == "1" and shutil.which("npx"):
        return ["npx", "-y", MERMAID_CLI_PACKAGE]
    return None


def mermaid_to_image_flowable(source: str, style_map: dict[str, ParagraphStyle]) -> list[Any]:
    """Render Mermaid source to a PNG flowable, or return readable source fallback."""
    command = mermaid_render_command()
    if command is None:
        return [
            Paragraph("Mermaid diagram source (install mmdc/npx to render):", style_map["Normal"]),
            Preformatted(source, style_map["Code"]),
        ]

    with tempfile.TemporaryDirectory(prefix="mmml-docs-mermaid-") as tmp_dir:
        input_path = Path(tmp_dir) / "diagram.mmd"
        output_path = Path(tmp_dir) / "diagram.png"
        input_path.write_text(source, encoding="utf-8")
        puppeteer_config = os.environ.get("MMML_MERMAID_PUPPETEER_CONFIG")
        puppeteer_args = (
            ["--puppeteerConfigFile", puppeteer_config]
            if puppeteer_config
            else []
        )
        try:
            subprocess.run(
                [
                    *command,
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--backgroundColor",
                    "white",
                    "--scale",
                    "2",
                    *puppeteer_args,
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=60,
            )
        except (subprocess.SubprocessError, OSError) as exc:
            return [
                Paragraph(f"Mermaid render failed ({html.escape(type(exc).__name__)}):", style_map["Normal"]),
                Preformatted(source, style_map["Code"]),
            ]

        image = Image(io.BytesIO(output_path.read_bytes()))
        max_width = 7.0 * inch
        max_height = 9.0 * inch
        scale = 1.0
        if image.drawWidth > max_width:
            scale = min(scale, max_width / image.drawWidth)
        if image.drawHeight > max_height:
            scale = min(scale, max_height / image.drawHeight)
        if scale < 1.0:
            image.drawWidth *= scale
            image.drawHeight *= scale
        return [image, Spacer(1, 0.08 * inch)]


def is_table_start(lines: list[str], index: int) -> bool:
    """Return True when ``lines[index:]`` starts a Markdown pipe table."""
    if index + 1 >= len(lines):
        return False
    return "|" in lines[index] and bool(TABLE_SEPARATOR_RE.match(lines[index + 1].strip()))


def split_table_row(line: str) -> list[str]:
    """Split one Markdown table row into cell text."""
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def markdown_table_to_flowables(
    table_lines: list[str],
    style_map: dict[str, ParagraphStyle],
) -> list[Any]:
    """Render a Markdown pipe table as a styled ReportLab table."""
    rows = [split_table_row(table_lines[0])]
    rows.extend(split_table_row(line) for line in table_lines[2:])
    if not rows:
        return []

    max_columns = max(len(row) for row in rows)
    normalized_rows = [row + [""] * (max_columns - len(row)) for row in rows]
    cell_style = ParagraphStyle(
        "TableCell",
        parent=style_map["Normal"],
        fontSize=7.5,
        leading=9,
        wordWrap="CJK",
    )
    header_style = ParagraphStyle(
        "TableHeader",
        parent=cell_style,
        fontName="Helvetica-Bold",
        textColor=colors.HexColor("#111827"),
    )
    data = [
        [
            Paragraph(
                inline_markdown_to_reportlab(cell),
                header_style if row_index == 0 else cell_style,
            )
            for cell in row
        ]
        for row_index, row in enumerate(normalized_rows)
    ]
    available_width = 7.3 * inch
    col_widths = [available_width / max_columns] * max_columns
    table = Table(data, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f3f4f6")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d1d5db")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fafafa")]),
            ]
        )
    )
    return [table, Spacer(1, 0.1 * inch)]


def markdown_page_to_flowables(title: str, path: Path, style_map: dict[str, ParagraphStyle]) -> list[Any]:
    """Render the Markdown subset used by the docs into ReportLab flowables."""
    flowables: list[Any] = [
        Paragraph(html.escape(str(path)), style_map["SourcePath"]),
        Paragraph(html.escape(title), style_map["Title"]),
        Spacer(1, 0.12 * inch),
    ]
    lines = (DOCS_DIR / path).read_text(encoding="utf-8").splitlines()
    in_code_block = False
    code_language = ""
    code_lines: list[str] = []

    line_index = 0
    while line_index < len(lines):
        line = lines[line_index]
        stripped = line.strip()
        if stripped.startswith("```"):
            if in_code_block:
                code_source = "\n".join(code_lines)
                if code_language == "mermaid":
                    flowables.extend(mermaid_to_image_flowable(code_source, style_map))
                else:
                    flowables.append(Preformatted(code_source, style_map["Code"]))
                flowables.append(Spacer(1, 0.08 * inch))
                code_lines = []
                code_language = ""
                in_code_block = False
            else:
                in_code_block = True
                code_language = stripped.removeprefix("```").strip().lower()
            line_index += 1
            continue
        if in_code_block:
            code_lines.append(line)
            line_index += 1
            continue
        if is_table_start(lines, line_index):
            table_lines = [line]
            line_index += 1
            while line_index < len(lines) and "|" in lines[line_index].strip():
                table_lines.append(lines[line_index])
                line_index += 1
            flowables.extend(markdown_table_to_flowables(table_lines, style_map))
            continue
        if not stripped:
            flowables.append(Spacer(1, 0.06 * inch))
            line_index += 1
            continue
        if stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            heading_text = stripped[level:].strip()
            if heading_text == title:
                line_index += 1
                continue
            style_name = "Heading2" if level <= 2 else "Heading3"
            flowables.append(Paragraph(inline_markdown_to_reportlab(heading_text), style_map[style_name]))
            line_index += 1
            continue
        if stripped.startswith("- "):
            bullet_text = stripped[2:].strip()
            flowables.append(Paragraph(f"- {inline_markdown_to_reportlab(bullet_text)}", style_map["BulletBody"]))
            line_index += 1
            continue
        flowables.append(Paragraph(inline_markdown_to_reportlab(stripped), style_map["Normal"]))
        line_index += 1

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
