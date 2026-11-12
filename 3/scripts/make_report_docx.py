"""把 pandoc 生成的实验报告 docx 排版成中文论文样式。

pandoc 只负责结构转换，这里补上中文字体、页边距、首行缩进、图片宽度和表格边框。

用法：
    python scripts/make_report_docx.py report/实验报告.docx            # 常规版式
    python scripts/make_report_docx.py report/实验报告.docx --compact   # 紧凑版式（限页数时用）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

BODY_ASCII = "Times New Roman"
BODY_CJK = "宋体"
HEAD_CJK = "黑体"
CODE_FONT = "Consolas"

# 两种版式：常规版适合详细报告，紧凑版用于限页数的提交件
LAYOUTS = {
    "normal": dict(
        body_size=10.5, line_spacing=1.4, indent=21,
        margins=(2.6, 2.4), heads={"Heading 1": 16, "Heading 2": 14, "Heading 3": 12, "Heading 4": 11},
        table_size=9, img_w=15.5, img_h=20.5, para_space=(12, 6),
    ),
    "compact": dict(
        body_size=9.5, line_spacing=1.15, indent=19,
        margins=(1.9, 1.7), heads={"Heading 1": 14, "Heading 2": 11.5, "Heading 3": 10, "Heading 4": 9.5},
        table_size=8, img_w=17.2, img_h=9.5, para_space=(7, 3),
    ),
}


def set_style_font(style, ascii_font: str, cjk_font: str, size: float | None = None,
                   bold: bool | None = None, color: RGBColor | None = None) -> None:
    style.font.name = ascii_font
    if size is not None:
        style.font.size = Pt(size)
    if bold is not None:
        style.font.bold = bold
    if color is not None:
        style.font.color.rgb = color

    rpr = style.element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    for attr in ("w:ascii", "w:hAnsi", "w:cs"):
        rfonts.set(qn(attr), ascii_font)
    rfonts.set(qn("w:eastAsia"), cjk_font)


def add_table_borders(table) -> None:
    tbl_pr = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        element = OxmlElement(f"w:{edge}")
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), "4")
        element.set(qn("w:color"), "999999")
        borders.append(element)
    tbl_pr.append(borders)


def shade_cell(cell, color: str = "F2F2F2") -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), color)
    tc_pr.append(shd)


def repeat_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    header = OxmlElement("w:tblHeader")
    header.set(qn("w:val"), "true")
    tr_pr.append(header)


def add_page_number_footer(section) -> None:
    footer = section.footer
    paragraph = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    for tag, text in (("begin", None), ("instr", "PAGE"), ("end", None)):
        if tag == "instr":
            element = OxmlElement("w:instrText")
            element.set(qn("xml:space"), "preserve")
            element.text = text
        else:
            element = OxmlElement("w:fldChar")
            element.set(qn("w:fldCharType"), tag)
        run._r.append(element)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default="report/实验报告.docx")
    parser.add_argument("--compact", action="store_true", help="紧凑版式，用于限制页数的提交件")
    args = parser.parse_args()

    path = Path(args.path)
    cfg = LAYOUTS["compact" if args.compact else "normal"]
    doc = Document(path)

    section = doc.sections[0]
    section.page_width, section.page_height = Cm(21), Cm(29.7)
    section.left_margin = section.right_margin = Cm(cfg["margins"][0])
    section.top_margin = section.bottom_margin = Cm(cfg["margins"][1])

    # 正文
    for name in ("Normal", "Body Text", "First Paragraph"):
        if name in doc.styles:
            style = doc.styles[name]
            set_style_font(style, BODY_ASCII, BODY_CJK, size=cfg["body_size"])
            style.paragraph_format.line_spacing = cfg["line_spacing"]
    for name in ("Body Text", "First Paragraph"):
        if name in doc.styles:
            doc.styles[name].paragraph_format.first_line_indent = Pt(cfg["indent"])

    # 标题
    heading_specs = {
        "Heading 1": (RGBColor(0, 0, 0)),
        "Heading 2": (RGBColor(0x1F, 0x38, 0x64)),
        "Heading 3": (RGBColor(0x2F, 0x5D, 0x8A)),
        "Heading 4": (RGBColor(0x33, 0x33, 0x33)),
    }
    for name, color in heading_specs.items():
        if name in doc.styles:
            style = doc.styles[name]
            set_style_font(style, BODY_ASCII, HEAD_CJK, size=cfg["heads"][name], bold=True, color=color)
            style.paragraph_format.space_before = Pt(cfg["para_space"][0])
            style.paragraph_format.space_after = Pt(cfg["para_space"][1])
            style.paragraph_format.first_line_indent = Pt(0)

    # 代码块
    for name in ("Source Code", "Verbatim Char", "Compact"):
        if name in doc.styles:
            style = doc.styles[name]
            set_style_font(style, CODE_FONT, BODY_CJK, size=cfg["body_size"] - 0.5)
            if hasattr(style, "paragraph_format"):
                style.paragraph_format.first_line_indent = Pt(0)

    # 列表项交给编号定义控制缩进，清掉首行缩进覆盖，否则悬挂缩进会失效
    for para in doc.paragraphs:
        if para.style.name.startswith("List") or para.style.name == "Compact":
            para.paragraph_format.first_line_indent = None
            para.paragraph_format.left_indent = Cm(0.85)
            para.paragraph_format.first_line_indent = Cm(-0.5)

    # 图片宽度/高度都限制在版心内
    max_w, max_h = Cm(cfg["img_w"]), Cm(cfg["img_h"])
    for shape in doc.inline_shapes:
        scale = 1.0
        if shape.width > max_w:
            scale = min(scale, max_w / shape.width)
        if shape.height > max_h:
            scale = min(scale, max_h / shape.height)
        if scale < 1.0:
            shape.width = int(shape.width * scale)
            shape.height = int(shape.height * scale)

    # 表格样式
    for table in doc.tables:
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        add_table_borders(table)
        repeat_header(table.rows[0])
        # 让整张表尽量留在同一页：除最后一行外都设为"与下段同页"
        for row in table.rows[:-1]:
            for cell in row.cells:
                for para in cell.paragraphs:
                    para.paragraph_format.keep_with_next = True
        for cell in table.rows[0].cells:
            shade_cell(cell)
            for para in cell.paragraphs:
                para.paragraph_format.first_line_indent = Pt(0)
                para.paragraph_format.line_spacing = 1.0
                para.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in para.runs:
                    run.font.bold = True
                    run.font.size = Pt(cfg["table_size"])
        for row in table.rows[1:]:
            for cell in row.cells:
                for para in cell.paragraphs:
                    para.paragraph_format.first_line_indent = Pt(0)
                    para.paragraph_format.line_spacing = 1.0
                    for run in para.runs:
                        run.font.size = Pt(cfg["table_size"])

    # 图片题注居中
    for para in doc.paragraphs:
        if para.style.name in ("Image Caption", "Captioned Figures", "Figure"):
            para.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # 标题不与后文分离，避免标题孤零零留在页底
    for para in doc.paragraphs:
        if para.style.name.startswith("Heading"):
            para.paragraph_format.keep_with_next = True

    # 文档标题居中放大
    if doc.paragraphs and doc.paragraphs[0].style.name == "Heading 1":
        title = doc.paragraphs[0]
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        title.paragraph_format.space_before = Pt(0)
        title.paragraph_format.space_after = Pt(10 if args.compact else 20)
        for run in title.runs:
            run.font.size = Pt(cfg["heads"]["Heading 1"] + 1)

    add_page_number_footer(section)

    doc.save(path)
    print(f"排版完成：{path}")
    print(f"  段落 {len(doc.paragraphs)}，表格 {len(doc.tables)}，图片 {len(doc.inline_shapes)}")


if __name__ == "__main__":
    main()
