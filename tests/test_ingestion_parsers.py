"""Unit tests for PDF, DOCX, and TXT document parsers."""

import io
import docx
from pypdf import PdfWriter
import pytest

from app.core.exceptions import EmptyDocumentError, InvalidDocumentError
from app.ingestion.parsers import (
    DocxDocumentParser,
    PDFDocumentParser,
    PlainTextParser,
    compute_sha256,
    parse_document,
)

SAMPLE_PDF_BYTES = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>
endobj
4 0 obj
<< /Length 55 >>
stream
BT
/F1 12 Tf
72 712 Td
(Section 185 Loans to Directors) Tj
ET
endstream
endobj
5 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000227 00000 n 
0000000334 00000 n 
trailer
<< /Size 6 /Root 1 0 R >>
startxref
405
%%EOF"""


def create_synthetic_docx() -> bytes:
    doc = docx.Document()
    doc.add_heading("Section 135 Corporate Social Responsibility", level=1)
    doc.add_paragraph("Every company having net worth of rupees five hundred crore or more shall constitute a CSR Committee.")
    table = doc.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Condition"
    table.rows[0].cells[1].text = "Threshold"
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def test_compute_sha256_deterministic():
    content = b"Legal research test contract content."
    digest1 = compute_sha256(content)
    digest2 = compute_sha256(content)
    assert digest1 == digest2
    assert len(digest1) == 64


def test_pdf_document_parser_success():
    parsed = PDFDocumentParser.parse(SAMPLE_PDF_BYTES, "companies_act.pdf", domain="corporate_law")
    assert parsed.file_type == "pdf"
    assert parsed.filename == "companies_act.pdf"
    assert "Section 185" in parsed.raw_text
    assert len(parsed.pages) == 1
    assert parsed.pages[0].page_number == 1
    assert parsed.domain == "corporate_law"


def test_docx_document_parser_success():
    docx_bytes = create_synthetic_docx()
    parsed = DocxDocumentParser.parse(docx_bytes, "employment_policy.docx")
    assert parsed.file_type == "docx"
    assert "Section 135" in parsed.raw_text
    assert "CSR Committee" in parsed.raw_text
    assert "Condition | Threshold" in parsed.raw_text


def test_plain_text_parser_utf8():
    text_content = "Article 21: Protection of life and personal liberty under Constitution of India.".encode("utf-8")
    parsed = PlainTextParser.parse(text_content, "constitution.txt")
    assert parsed.file_type == "txt"
    assert "Article 21" in parsed.raw_text
    assert len(parsed.pages) == 1


def test_plain_text_parser_latin1_fallback():
    # Contains a byte valid in latin-1 but invalid in utf-8
    latin1_bytes = b"Section 9 IBC: \xa3100,000 operational debt notice."
    parsed = PlainTextParser.parse(latin1_bytes, "notice.txt")
    assert "Section 9 IBC" in parsed.raw_text


def test_corrupted_pdf_raises_invalid_document():
    corrupted_bytes = b"Not a real PDF file header garbage content"
    with pytest.raises(InvalidDocumentError):
        PDFDocumentParser.parse(corrupted_bytes, "broken.pdf")


def test_empty_scanned_pdf_raises_empty_document():
    # Blank PDF with zero text
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    buf = io.BytesIO()
    writer.write(buf)
    blank_pdf = buf.getvalue()

    with pytest.raises(EmptyDocumentError) as exc_info:
        PDFDocumentParser.parse(blank_pdf, "scanned_image.pdf")

    assert "zero extractable characters" in str(exc_info.value)


def test_empty_txt_file_raises_empty_document():
    with pytest.raises(EmptyDocumentError):
        PlainTextParser.parse(b"   \n\t  ", "empty.txt")


def test_parse_document_dispatcher_rejects_unsupported_format():
    with pytest.raises(InvalidDocumentError) as exc_info:
        parse_document(b"binary data", "malicious_script.exe")

    assert "Unsupported file format" in str(exc_info.value)
