"""把 pandoc 生成的实验报告 docx 排版成中文论文样式。

pandoc 只负责结构转换，这里补上中文字体、页边距、首行缩进、图片宽度和表格边框。

用法：python scripts/make_report_docx.py report/实验报告.docx
"""

from __future__ import annotations

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
MAX_IMAGE_WIDTH = Cm(15.5)
# 版心高度约 24.9cm，留出题注空间，太高的图必须按高度缩放否则会溢出到下一页
MAX_IMAGE_HEIGHT = Cm(20.5)


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
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "report/实验报告.docx")
    doc = Document(path)

    section = doc.sections[0]
    section.page_width, section.page_height = Cm(21), Cm(29.7)
    section.left_margin = section.right_margin = Cm(2.6)
    section.top_margin = section.bottom_margin = Cm(2.4)

    # 正文
    for name in ("Normal", "Body Text", "First Paragraph"):
        if name in doc.styles:
            style = doc.styles[name]
            set_style_font(style, BODY_ASCII, BODY_CJK, size=10.5)
            style.paragraph_format.line_spacing = 1.4
    for name in ("Body Text", "First Paragraph"):
        if name in doc.styles:
            doc.styles[name].paragraph_format.first_line_indent = Pt(21)

    # 标题
    heading_specs = {
        "Heading 1": (16, True, RGBColor(0, 0, 0)),
        "Heading 2": (14, True, RGBColor(0x1F, 0x38, 0x64)),
        "Heading 3": (12, True, RGBColor(0x2F, 0x5D, 0x8A)),
        "Heading 4": (11, True, RGBColor(0x33, 0x33, 0x33)),
    }
    for name, (size, bold, color) in heading_specs.items():
        if name in doc.styles:
            style = doc.styles[name]
            set_style_font(style, BODY_ASCII, HEAD_CJK, size=size, bold=bold, color=color)
            style.paragraph_format.space_before = Pt(12)
            style.paragraph_format.space_after = Pt(6)
            style.paragraph_format.first_line_indent = Pt(0)

    # 代码块
    for name in ("Source Code", "Verbatim Char", "Compact"):
        if name in doc.styles:
            style = doc.styles[name]
            set_style_font(style, CODE_FONT, BODY_CJK, size=9)
            if hasattr(style, "paragraph_format"):
                style.paragraph_format.first_line_indent = Pt(0)

    # 列表项交给编号定义控制缩进，清掉首行缩进覆盖，否则悬挂缩进会失效
    for para in doc.paragraphs:
        if para.style.name.startswith("List") or para.style.name == "Compact":
            para.paragraph_format.first_line_indent = None
            para.paragraph_format.left_indent = Cm(1.0)
            para.paragraph_format.first_line_indent = Cm(-0.6)

    # 图片宽度/高度都限制在版心内
    for shape in doc.inline_shapes:
        scale = 1.0
        if shape.width > MAX_IMAGE_WIDTH:
            scale = min(scale, MAX_IMAGE_WIDTH / shape.width)
        if shape.height > MAX_IMAGE_HEIGHT:
            scale = min(scale, MAX_IMAGE_HEIGHT / shape.height)
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
                para.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in para.runs:
                    run.font.bold = True
                    run.font.size = Pt(9)
        for row in table.rows[1:]:
            for cell in row.cells:
                for para in cell.paragraphs:
                    para.paragraph_format.first_line_indent = Pt(0)
                    for run in para.runs:
                        run.font.size = Pt(9)

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
        title.paragraph_format.space_after = Pt(20)
        for run in title.runs:
            run.font.size = Pt(18)

    add_page_number_footer(section)

    doc.save(path)
    print(f"排版完成：{path}")
    print(f"  段落 {len(doc.paragraphs)}，表格 {len(doc.tables)}，图片 {len(doc.inline_shapes)}")


if __name__ == "__main__":
    main()
