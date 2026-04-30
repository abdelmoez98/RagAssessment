import os
import pdfplumber
from docx import Document


def parse_markdown(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def parse_pdf(filepath: str) -> str:
    text_parts = []
    extraction_warnings = []

    with pdfplumber.open(filepath) as pdf:
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
            else:
                extraction_warnings.append(f"Page {i + 1}: no text extracted (may be scanned/image)")

    full_text = "\n".join(text_parts)

    if extraction_warnings:
        filename = os.path.basename(filepath)
        print(f"  {filename}: {len(pdf.pages)} pages, {len(full_text)} chars extracted")
        for w in extraction_warnings:
            print(f"    WARNING: {w}")

    return full_text


def parse_docx(filepath: str) -> str:
    doc = Document(filepath)
    paragraphs = []

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            paragraphs.append("")
            continue

        style_name = para.style.name if para.style else ""

        if "Heading 1" in style_name or "heading 1" in style_name:
            paragraphs.append(f"# {text}")
        elif "Heading 2" in style_name or "heading 2" in style_name:
            paragraphs.append(f"## {text}")
        elif "Heading 3" in style_name or "heading 3" in style_name:
            paragraphs.append(f"### {text}")
        elif para.runs and para.runs[0].bold:
            paragraphs.append(f"**{text}**")
        else:
            paragraphs.append(text)

    return "\n\n".join(paragraphs)


def parse_document(filepath: str) -> tuple[str, str]:
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".md":
        return parse_markdown(filepath), "markdown"
    elif ext == ".pdf":
        return parse_pdf(filepath), "pdf"
    elif ext == ".docx":
        return parse_docx(filepath), "docx"
    else:
        raise ValueError(f"Unsupported format: {ext}")
