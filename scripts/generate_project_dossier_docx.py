from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile


STYLE_IDS = {
    "normal": "a",
    "heading1": "1",
    "heading2": "2",
    "heading3": "3",
    "title": "a3",
    "subtitle": "a5",
    "list": "a9",
    "table": "ae",
    "quote": "a7",
}

BULLET_NUM_ID = "10"
DECIMAL_NUM_ID = "11"
HEADER_REL_ID = "rId100"
FOOTER_REL_ID = "rId101"
HEADER_PART = "word/header1.xml"
FOOTER_PART = "word/footer1.xml"
HEADER_CONTENT_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.header+xml"
FOOTER_CONTENT_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"
HEBREW_RE = re.compile(r"[\u0590-\u05FF]")
LATIN_RE = re.compile(r"[A-Za-z0-9]")
TOKEN_RE = re.compile(r"[\u0590-\u05FF]+|[A-Za-z0-9_./:+#%&=@\\-]+|\s+|.")
LRM = "\u200E"
RLM = "\u200F"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a polished DOCX dossier from the Hebrew Markdown source."
    )
    parser.add_argument(
        "--input",
        default="docs/SpamGuard_Project_Dossier_HE.md",
        help="Markdown source path.",
    )
    parser.add_argument(
        "--output",
        default="docs/SpamGuard_Project_Dossier_HE.docx",
        help="Output DOCX path.",
    )
    parser.add_argument(
        "--template",
        default=None,
        help="Template DOCX path. Defaults to the first DOCX found at repo root.",
    )
    return parser.parse_args()


def split_cover_and_body(text: str) -> tuple[list[str], list[str]]:
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if line.strip() == "---":
            return lines[:idx], lines[idx + 1 :]
    return lines, []


def parse_blocks(lines: list[str]) -> list[tuple]:
    blocks: list[tuple] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue

        if stripped == "---":
            blocks.append(("hr",))
            i += 1
            continue

        if stripped.startswith("```"):
            lang = stripped[3:].strip()
            i += 1
            code_lines = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i].rstrip("\n"))
                i += 1
            if i < len(lines):
                i += 1
            blocks.append(("code", lang, code_lines))
            continue

        match = re.match(r"^(#{1,4})\s+(.*)$", line)
        if match:
            blocks.append(("heading", len(match.group(1)), match.group(2).strip()))
            i += 1
            continue

        if stripped.startswith("|") and stripped.endswith("|"):
            table_lines = []
            while i < len(lines):
                current = lines[i].strip()
                if not (current.startswith("|") and current.endswith("|")):
                    break
                table_lines.append(current)
                i += 1
            rows = []
            for raw in table_lines:
                cells = [cell.strip() for cell in raw.strip("|").split("|")]
                if all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in cells):
                    continue
                rows.append(cells)
            if rows:
                blocks.append(("table", rows))
            continue

        if re.match(r"^-\s+", stripped):
            items = []
            while i < len(lines) and re.match(r"^-\s+", lines[i].strip()):
                items.append(lines[i].strip()[2:].strip())
                i += 1
            blocks.append(("ul", items))
            continue

        if re.match(r"^\d+\.\s+", stripped):
            items = []
            while i < len(lines) and re.match(r"^\d+\.\s+", lines[i].strip()):
                raw = lines[i].strip()
                number, content = raw.split(".", 1)
                items.append((number.strip(), content.strip()))
                i += 1
            blocks.append(("ol", items))
            continue

        paragraph_lines = []
        while i < len(lines):
            current = lines[i]
            current_stripped = current.strip()
            if not current_stripped:
                break
            if current_stripped == "---":
                break
            if current_stripped.startswith("```"):
                break
            if re.match(r"^(#{1,4})\s+", current):
                break
            if current_stripped.startswith("|") and current_stripped.endswith("|"):
                break
            if re.match(r"^-\s+", current_stripped):
                break
            if re.match(r"^\d+\.\s+", current_stripped):
                break

            paragraph_lines.append(current.rstrip())
            i += 1

            if current.endswith("  "):
                break

        paragraph = " ".join(part.strip() for part in paragraph_lines if part.strip())
        if paragraph:
            blocks.append(("p", paragraph))

    return blocks


def parse_inlines(text: str) -> list[tuple[str, str]]:
    parts: list[tuple[str, str]] = []
    i = 0
    while i < len(text):
        if text.startswith("**", i):
            end = text.find("**", i + 2)
            if end != -1:
                parts.append(("bold", text[i + 2 : end]))
                i = end + 2
                continue
        if text.startswith("`", i):
            end = text.find("`", i + 1)
            if end != -1:
                parts.append(("code", text[i + 1 : end]))
                i = end + 1
                continue
        j = i
        while j < len(text) and not text.startswith("**", j) and text[j] != "`":
            j += 1
        parts.append(("text", text[i:j]))
        i = j
    return [part for part in parts if part[1]]


def xml_text(text: str) -> str:
    if not text:
        return '<w:t xml:space="preserve"></w:t>'
    preserve = text.startswith(" ") or text.endswith(" ") or "  " in text
    space_attr = ' xml:space="preserve"' if preserve else ""
    return f"<w:t{space_attr}>{escape(text)}</w:t>"


def split_directional_segments(text: str, initial_dir: bool | None = None) -> list[tuple[str, bool | None]]:
    segments: list[tuple[str, bool | None]] = []
    current_text = ""
    current_dir: bool | None = None
    active_dir: bool | None = initial_dir

    for token in TOKEN_RE.findall(text):
        if not token:
            continue

        if token.isspace():
            direction = active_dir
        elif HEBREW_RE.search(token):
            direction = True
            active_dir = True
        elif LATIN_RE.search(token):
            direction = False
            active_dir = False
        else:
            direction = active_dir

        if segments and direction == current_dir:
            current_text += token
            segments[-1] = (current_text, current_dir)
            continue

        current_text = token
        current_dir = direction
        segments.append((current_text, current_dir))

    return segments if segments else [(text, None)]


def anchor_neutral_segment(text: str, direction: bool | None) -> str:
    if direction is None:
        return text

    stripped = text.strip()
    if not stripped:
        return text

    has_strong = HEBREW_RE.search(stripped) if direction else LATIN_RE.search(stripped)
    if has_strong:
        return text

    leading = text[: len(text) - len(text.lstrip())]
    trailing = text[len(text.rstrip()) :]
    core = text[len(leading) : len(text) - len(trailing) if trailing else len(text)]
    mark = RLM if direction else LRM
    return f"{leading}{mark}{core}{mark}{trailing}"


def run_xml(
    text: str,
    *,
    bold: bool = False,
    code: bool = False,
    size: int | None = None,
    rtl: bool | None = None,
) -> str:
    props = []
    if bold:
        props.append("<w:b/>")
    if code:
        props.append('<w:rFonts w:ascii="Consolas" w:hAnsi="Consolas" w:cs="Consolas"/>')
    else:
        hint = ' w:hint="cs"' if rtl is True else ""
        props.append(f'<w:rFonts w:ascii="Arial" w:hAnsi="Arial" w:cs="Arial"{hint}/>')
    if rtl is True:
        props.append("<w:rtl/>")
    if size is not None:
        props.append(f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/>')
    props_xml = f"<w:rPr>{''.join(props)}</w:rPr>" if props else ""
    return f"<w:r>{props_xml}{xml_text(text)}</w:r>"


def runs_xml(text: str, *, default_bold: bool = False, size: int | None = None) -> str:
    runs = []
    previous_dir: bool | None = None
    for kind, value in parse_inlines(text):
        if kind == "code":
            runs.append(
                run_xml(
                    f"{LRM}{value}{LRM}",
                    bold=default_bold,
                    code=True,
                    size=size,
                    rtl=False,
                )
            )
            previous_dir = False
            continue

        for segment_text, segment_rtl in split_directional_segments(value, initial_dir=previous_dir):
            anchored_text = anchor_neutral_segment(segment_text, segment_rtl)
            runs.append(
                run_xml(
                    anchored_text,
                    bold=default_bold or kind == "bold",
                    code=False,
                    size=size,
                    rtl=segment_rtl,
                )
            )
            if segment_rtl is not None and anchored_text.strip():
                previous_dir = segment_rtl
    return "".join(runs) if runs else run_xml("")


def paragraph_xml(
    text: str,
    *,
    style: str | None = None,
    align: str = "right",
    bidi: bool = True,
    spacing_after: int = 160,
    spacing_before: int = 0,
    line: int = 360,
    keep_next: bool = False,
    default_bold: bool = False,
    size: int | None = None,
    right_indent: int | None = None,
    hanging: int | None = None,
    num_id: str | None = None,
    ilvl: int = 0,
) -> str:
    props = []
    if style:
        props.append(f'<w:pStyle w:val="{style}"/>')
    props.append(f'<w:jc w:val="{align}"/>')
    if bidi:
        props.append("<w:bidi/>")
        props.append("<w:mirrorIndents/>")
    if num_id is not None:
        props.append(f'<w:numPr><w:ilvl w:val="{ilvl}"/><w:numId w:val="{num_id}"/></w:numPr>')
    if keep_next:
        props.append("<w:keepNext/>")
    if spacing_before or spacing_after or line:
        props.append(
            f'<w:spacing w:before="{spacing_before}" w:after="{spacing_after}" w:line="{line}" w:lineRule="auto"/>'
        )
    if right_indent is not None or hanging is not None:
        ind_parts = []
        if right_indent is not None:
            ind_parts.append(f'w:right="{right_indent}"')
        if hanging is not None:
            ind_parts.append(f'w:hanging="{hanging}"')
        props.append(f"<w:ind {' '.join(ind_parts)}/>")
    props.append("<w:widowControl/>")
    return f"<w:p><w:pPr>{''.join(props)}</w:pPr>{runs_xml(text, default_bold=default_bold, size=size)}</w:p>"


def blank_paragraph(*, spacing_after: int = 120) -> str:
    return (
        "<w:p>"
        '<w:pPr><w:jc w:val="right"/><w:bidi/>'
        f'<w:spacing w:after="{spacing_after}" w:line="240" w:lineRule="auto"/></w:pPr>'
        "</w:p>"
    )


def page_break_paragraph() -> str:
    return "<w:p><w:r><w:br w:type=\"page\"/></w:r></w:p>"


def heading_xml(level: int, text: str) -> str:
    style = STYLE_IDS["heading1"]
    size = 34
    spacing_before = 120
    spacing_after = 80
    if level == 2:
        style = STYLE_IDS["heading2"]
        size = 30
    elif level == 3:
        style = STYLE_IDS["heading3"]
        size = 26
    elif level >= 4:
        style = "4"
        size = 24

    return paragraph_xml(
        text,
        style=style,
        align="right",
        bidi=True,
        spacing_before=spacing_before,
        spacing_after=spacing_after,
        line=260,
        keep_next=True,
        default_bold=True,
        size=size,
    )


def field_char_run_xml(field_type: str, *, dirty: bool = False) -> str:
    dirty_attr = ' w:dirty="true"' if dirty else ""
    return f'<w:r><w:fldChar w:fldCharType="{field_type}"{dirty_attr}/></w:r>'


def instr_text_run_xml(text: str) -> str:
    return f'<w:r><w:instrText xml:space="preserve">{escape(text)}</w:instrText></w:r>'


def toc_field_paragraph_xml() -> str:
    placeholder = runs_xml(
        "יש לעדכן את השדות בעת פתיחת המסמך ב-Word כדי להציג כאן תוכן עניינים מלא.",
        size=22,
    )
    return (
        "<w:p>"
        '<w:pPr><w:jc w:val="right"/><w:bidi/><w:mirrorIndents/>'
        '<w:spacing w:before="0" w:after="120" w:line="260" w:lineRule="auto"/>'
        "<w:widowControl/></w:pPr>"
        f"{field_char_run_xml('begin', dirty=True)}"
        f'{instr_text_run_xml(r" TOC \\o \"1-3\" \\h \\z \\u ")}'
        f"{field_char_run_xml('separate')}"
        f"{placeholder}"
        f"{field_char_run_xml('end')}"
        "</w:p>"
    )


def toc_page_xml() -> list[str]:
    return [
        paragraph_xml(
            "תוכן עניינים",
            align="right",
            bidi=True,
            spacing_after=120,
            line=260,
            default_bold=True,
            size=32,
        ),
        toc_field_paragraph_xml(),
        blank_paragraph(spacing_after=120),
    ]


def cover_xml(lines: list[str]) -> list[str]:
    out: list[str] = []
    clean_lines = [line.rstrip() for line in lines if line.strip()]
    if not clean_lines:
        return out

    metadata_lines = clean_lines[1:] if clean_lines[0].startswith("# ") else clean_lines

    for _ in range(3):
        out.append(blank_paragraph(spacing_after=0))

    for line in metadata_lines:
        out.append(
            paragraph_xml(
                line.strip(),
                style=None,
                align="right",
                bidi=True,
                spacing_after=0,
                line=260,
                size=24,
            )
        )

    for _ in range(18):
        out.append(blank_paragraph(spacing_after=0))
    return out


def bullet_paragraph_xml(text: str) -> str:
    return paragraph_xml(
        text,
        style=STYLE_IDS["list"],
        align="right",
        bidi=True,
        spacing_after=90,
        line=260,
        right_indent=720,
        hanging=360,
        num_id=BULLET_NUM_ID,
        size=24,
    )


def numbered_paragraph_xml(number: str, text: str) -> str:
    return paragraph_xml(
        text,
        style=STYLE_IDS["list"],
        align="right",
        bidi=True,
        spacing_after=90,
        line=260,
        right_indent=900,
        hanging=540,
        num_id=DECIMAL_NUM_ID,
        size=24,
    )


def code_paragraph_xml(text: str) -> str:
    props = (
        '<w:pPr><w:jc w:val="left"/><w:spacing w:before="0" w:after="40" w:line="260" w:lineRule="auto"/>'
        '<w:ind w:left="240" w:right="240"/>'
        '<w:shd w:val="clear" w:color="auto" w:fill="F3F3F3"/></w:pPr>'
    )
    content = text if text else " "
    return f"<w:p>{props}{run_xml(content, code=True, size=20)}</w:p>"


def table_xml(rows: list[list[str]]) -> str:
    col_count = max(len(row) for row in rows)
    grid_cols = "".join('<w:gridCol w:w="2800"/>' for _ in range(col_count))
    tbl_props = (
        f'<w:tblPr><w:tblStyle w:val="{STYLE_IDS["table"]}"/>'
        '<w:tblW w:w="0" w:type="auto"/><w:bidiVisual/>'
        '<w:tblLook w:val="04A0" w:firstRow="1" w:lastRow="0" w:firstColumn="1" '
        'w:lastColumn="0" w:noHBand="0" w:noVBand="1"/></w:tblPr>'
    )

    row_xml = []
    for row_idx, row in enumerate(rows):
        cells = []
        padded = row + [""] * (col_count - len(row))
        for cell in padded:
            cell_props = '<w:tcPr><w:tcW w:w="2800" w:type="dxa"/>'
            if row_idx == 0:
                cell_props += '<w:shd w:val="clear" w:color="auto" w:fill="E7EDF8"/>'
            cell_props += "</w:tcPr>"
            para = paragraph_xml(
                cell,
                style=STYLE_IDS["normal"],
                align="right",
                bidi=True,
                spacing_after=40,
                line=300,
                default_bold=row_idx == 0,
                size=22 if row_idx == 0 else None,
            )
            cells.append(f"<w:tc>{cell_props}{para}</w:tc>")
        row_xml.append("<w:tr>" + "".join(cells) + "</w:tr>")

    return "<w:tbl>" + tbl_props + "<w:tblGrid>" + grid_cols + "</w:tblGrid>" + "".join(row_xml) + "</w:tbl>"


def quote_paragraph_xml(text: str) -> str:
    return paragraph_xml(
        text,
        style=STYLE_IDS["quote"],
        align="right",
        bidi=True,
        spacing_after=0,
        line=260,
        right_indent=420,
        size=24,
    )


def build_header_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:hdr xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:p>"
        '<w:pPr><w:jc w:val="right"/><w:bidi/><w:mirrorIndents/>'
        '<w:spacing w:before="0" w:after="0" w:line="240" w:lineRule="auto"/>'
        "<w:widowControl/></w:pPr>"
        + runs_xml("SpamGuard | [שם התלמיד]", size=20)
        + "</w:p></w:hdr>"
    )


def build_footer_xml() -> str:
    page_field = (
        field_char_run_xml("begin")
        + instr_text_run_xml(" PAGE ")
        + field_char_run_xml("separate")
        + runs_xml("1", size=20)
        + field_char_run_xml("end")
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:ftr xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:p>"
        '<w:pPr><w:jc w:val="center"/><w:bidi/>'
        '<w:spacing w:before="0" w:after="0" w:line="240" w:lineRule="auto"/>'
        "<w:widowControl/></w:pPr>"
        + runs_xml("עמוד ", size=20)
        + page_field
        + "</w:p></w:ftr>"
    )


def build_document_xml(body_parts: list[str], sect_pr: str) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:wpc="http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas" '
        'xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" '
        'xmlns:o="urn:schemas-microsoft-com:office:office" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math" '
        'xmlns:v="urn:schemas-microsoft-com:vml" '
        'xmlns:wp14="http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing" '
        'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
        'xmlns:w10="urn:schemas-microsoft-com:office:word" '
        'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
        'xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml" '
        'xmlns:w15="http://schemas.microsoft.com/office/word/2012/wordml" '
        'xmlns:w16cex="http://schemas.microsoft.com/office/word/2018/wordml/cex" '
        'xmlns:w16cid="http://schemas.microsoft.com/office/word/2016/wordml/cid" '
        'xmlns:w16="http://schemas.microsoft.com/office/word/2018/wordml" '
        'xmlns:w16du="http://schemas.microsoft.com/office/word/2023/wordml/word16du" '
        'xmlns:w16sdtdh="http://schemas.microsoft.com/office/word/2020/wordml/sdtdatahash" '
        'xmlns:w16sdtfl="http://schemas.microsoft.com/office/word/2024/wordml/sdtformatlock" '
        'xmlns:w16se="http://schemas.microsoft.com/office/word/2015/wordml/symex" '
        'mc:Ignorable="w14 w15 w16se w16cid w16 w16cex w16sdtdh w16sdtfl w16du">'
        "<w:body>"
        + "".join(body_parts)
        + sect_pr
        + "</w:body></w:document>"
    )


def build_numbering_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:numbering xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        '<w:abstractNum w:abstractNumId="10">'
        '<w:nsid w:val="1A2B3C4D"/><w:multiLevelType w:val="singleLevel"/>'
        '<w:tmpl w:val="0409001D"/>'
        '<w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="bullet"/>'
        '<w:lvlText w:val="•"/><w:lvlJc w:val="right"/>'
        '<w:pPr><w:bidi/><w:tabs><w:tab w:val="num" w:pos="720"/></w:tabs>'
        '<w:ind w:right="720" w:hanging="360"/></w:pPr>'
        '<w:rPr><w:rFonts w:ascii="Arial" w:hAnsi="Arial" w:cs="Arial"/></w:rPr>'
        '</w:lvl></w:abstractNum>'
        '<w:abstractNum w:abstractNumId="11">'
        '<w:nsid w:val="5E6F7A8B"/><w:multiLevelType w:val="singleLevel"/>'
        '<w:tmpl w:val="0409001F"/>'
        '<w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="decimal"/>'
        '<w:lvlText w:val="%1."/><w:lvlJc w:val="right"/>'
        '<w:pPr><w:bidi/><w:tabs><w:tab w:val="num" w:pos="900"/></w:tabs>'
        '<w:ind w:right="900" w:hanging="540"/></w:pPr>'
        '<w:rPr><w:rFonts w:ascii="Arial" w:hAnsi="Arial" w:cs="Arial"/></w:rPr>'
        '</w:lvl></w:abstractNum>'
        f'<w:num w:numId="{BULLET_NUM_ID}"><w:abstractNumId w:val="10"/></w:num>'
        f'<w:num w:numId="{DECIMAL_NUM_ID}"><w:abstractNumId w:val="11"/></w:num>'
        "</w:numbering>"
    )


def build_core_props() -> str:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" '
        'xmlns:dcterms="http://purl.org/dc/terms/" '
        'xmlns:dcmitype="http://purl.org/dc/dcmitype/" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
        "<dc:title>SpamGuard Project Dossier</dc:title>"
        "<dc:subject>SpamGuard</dc:subject>"
        "<dc:creator>OpenAI Codex</dc:creator>"
        "<cp:keywords>SpamGuard, spam, transformer, Hebrew</cp:keywords>"
        "<dc:description>Hebrew project dossier for SpamGuard.</dc:description>"
        "<cp:lastModifiedBy>OpenAI Codex</cp:lastModifiedBy>"
        f'<dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>'
        f'<dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>'
        "</cp:coreProperties>"
    )


def build_settings_xml(template_settings_xml: str) -> str:
    settings_xml = re.sub(r"<w:updateFields[^>]*/>", "", template_settings_xml)
    settings_xml = re.sub(r"<w:updateFields[^>]*>[\s\S]*?</w:updateFields>", "", settings_xml)
    return settings_xml.replace("</w:settings>", '<w:updateFields w:val="true"/></w:settings>')


def build_content_types_xml(template_content_types_xml: str) -> str:
    inserts = []
    if f'PartName="/{HEADER_PART}"' not in template_content_types_xml:
        inserts.append(f'<Override PartName="/{HEADER_PART}" ContentType="{HEADER_CONTENT_TYPE}"/>')
    if f'PartName="/{FOOTER_PART}"' not in template_content_types_xml:
        inserts.append(f'<Override PartName="/{FOOTER_PART}" ContentType="{FOOTER_CONTENT_TYPE}"/>')
    if not inserts:
        return template_content_types_xml
    return template_content_types_xml.replace("</Types>", "".join(inserts) + "</Types>")


def build_document_rels_xml(template_rels_xml: str) -> str:
    inserts = []
    if f'Target="{Path(HEADER_PART).name}"' not in template_rels_xml:
        inserts.append(
            f'<Relationship Id="{HEADER_REL_ID}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/header" '
            f'Target="{Path(HEADER_PART).name}"/>'
        )
    if f'Target="{Path(FOOTER_PART).name}"' not in template_rels_xml:
        inserts.append(
            f'<Relationship Id="{FOOTER_REL_ID}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer" '
            f'Target="{Path(FOOTER_PART).name}"/>'
        )
    if not inserts:
        return template_rels_xml
    return template_rels_xml.replace("</Relationships>", "".join(inserts) + "</Relationships>")


def extract_sect_pr(template_doc_xml: str) -> str:
    match = re.search(r"(<w:sectPr[\s\S]*?</w:sectPr>)", template_doc_xml)
    if not match:
        raise RuntimeError("Could not extract section properties from template DOCX.")
    return match.group(1)


def inject_header_footer_refs(sect_pr: str) -> str:
    clean = re.sub(r'<w:headerReference[^>]*/>', "", sect_pr)
    clean = re.sub(r'<w:footerReference[^>]*/>', "", clean)
    refs = (
        f'<w:headerReference w:type="default" r:id="{HEADER_REL_ID}"/>'
        f'<w:footerReference w:type="default" r:id="{FOOTER_REL_ID}"/>'
    )
    if "<w:pgSz" in clean:
        return clean.replace("<w:pgSz", refs + "<w:pgSz", 1)
    return clean.replace("</w:sectPr>", refs + "</w:sectPr>")


def build_body_xml(lines: list[str]) -> list[str]:
    parts: list[str] = []
    for block in parse_blocks(lines):
        kind = block[0]
        if kind == "hr":
            parts.append(blank_paragraph(spacing_after=180))
        elif kind == "heading":
            _, level, text = block
            normalized_level = max(1, level - 1)
            parts.append(heading_xml(normalized_level, text))
        elif kind == "p":
            parts.append(
                paragraph_xml(
                    block[1],
                    style=None,
                    align="right",
                    bidi=True,
                    spacing_after=0,
                    line=260,
                    size=24,
                )
            )
            parts.append(blank_paragraph(spacing_after=0))
        elif kind == "ul":
            for item in block[1]:
                parts.append(bullet_paragraph_xml(item))
            parts.append(blank_paragraph(spacing_after=80))
        elif kind == "ol":
            for number, item in block[1]:
                parts.append(numbered_paragraph_xml(number, item))
            parts.append(blank_paragraph(spacing_after=80))
        elif kind == "code":
            _, _lang, code_lines = block
            for line in code_lines:
                parts.append(code_paragraph_xml(line))
            parts.append(blank_paragraph(spacing_after=120))
        elif kind == "table":
            parts.append(table_xml(block[1]))
            parts.append(blank_paragraph(spacing_after=160))
        elif kind == "quote":
            parts.append(quote_paragraph_xml(block[1]))
    return parts


def render_document(markdown_text: str, template_doc_xml: str) -> str:
    cover_lines, body_lines = split_cover_and_body(markdown_text)
    parts = cover_xml(cover_lines)
    if parts:
        parts.append(page_break_paragraph())
    parts.extend(toc_page_xml())
    parts.append(page_break_paragraph())
    parts.extend(build_body_xml(body_lines))
    return build_document_xml(parts, inject_header_footer_refs(extract_sect_pr(template_doc_xml)))


def resolve_template(path_value: str | None) -> Path:
    if path_value:
        path = Path(path_value)
        if not path.exists():
            raise FileNotFoundError(f"Template DOCX not found: {path}")
        return path

    candidates = sorted(Path(".").glob("*.docx"))
    if not candidates:
        raise FileNotFoundError("No template DOCX found at repo root.")
    return candidates[0]


def write_docx(
    template_path: Path,
    output_path: Path,
    document_xml: str,
    core_xml: str,
    numbering_xml: str,
    settings_xml: str,
    content_types_xml: str,
    document_rels_xml: str,
    header_xml: str,
    footer_xml: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(template_path) as src, ZipFile(output_path, "w", ZIP_DEFLATED) as dst:
        for info in src.infolist():
            if info.filename in {
                "[Content_Types].xml",
                "word/document.xml",
                "word/_rels/document.xml.rels",
                "word/settings.xml",
                "docProps/core.xml",
                "word/numbering.xml",
                HEADER_PART,
                FOOTER_PART,
            }:
                continue
            dst.writestr(info, src.read(info.filename))
        dst.writestr("[Content_Types].xml", content_types_xml)
        dst.writestr("word/document.xml", document_xml)
        dst.writestr("word/_rels/document.xml.rels", document_rels_xml)
        dst.writestr("word/settings.xml", settings_xml)
        dst.writestr("docProps/core.xml", core_xml)
        dst.writestr("word/numbering.xml", numbering_xml)
        dst.writestr(HEADER_PART, header_xml)
        dst.writestr(FOOTER_PART, footer_xml)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    template_path = resolve_template(args.template)

    markdown_text = input_path.read_text(encoding="utf-8")
    with ZipFile(template_path) as zf:
        template_doc_xml = zf.read("word/document.xml").decode("utf-8", errors="ignore")
        template_settings_xml = zf.read("word/settings.xml").decode("utf-8", errors="ignore")
        template_content_types_xml = zf.read("[Content_Types].xml").decode("utf-8", errors="ignore")
        template_document_rels_xml = zf.read("word/_rels/document.xml.rels").decode("utf-8", errors="ignore")

    document_xml = render_document(markdown_text, template_doc_xml)
    core_xml = build_core_props()
    numbering_xml = build_numbering_xml()
    settings_xml = build_settings_xml(template_settings_xml)
    content_types_xml = build_content_types_xml(template_content_types_xml)
    document_rels_xml = build_document_rels_xml(template_document_rels_xml)
    header_xml = build_header_xml()
    footer_xml = build_footer_xml()
    write_docx(
        template_path,
        output_path,
        document_xml,
        core_xml,
        numbering_xml,
        settings_xml,
        content_types_xml,
        document_rels_xml,
        header_xml,
        footer_xml,
    )

    print(output_path)


if __name__ == "__main__":
    main()
