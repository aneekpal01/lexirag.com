"""Unit tests for StructureAwareLegalChunker."""

import pytest

from app.ingestion.chunker import StructureAwareLegalChunker
from app.ingestion.models import DocumentPage, ParsedDocument


@pytest.fixture
def legal_document() -> ParsedDocument:
    raw_text = (
        "Section 185 Loans to Directors\n\n"
        "(1) No company shall, directly or indirectly, advance any loan, including any loan represented "
        "by a book debt, to any of its directors or to any other person in whom the director is interested.\n\n"
        "(2) A company may advance any loan including any loan represented by a book debt, or give any guarantee "
        "or provide any security in connection with any loan taken by any person in whom any of the director of the "
        "company is interested, subject to the condition that a special resolution is passed by the company in general meeting.\n\n"
        "Clause 4.1 Indemnification\n\n"
        "The Company shall indemnify and hold harmless the Director against all losses, liabilities, claims, and expenses.\n\n"
        "Article 21 Protection of Personal Liberty\n\n"
        "No person shall be deprived of his life or personal liberty except according to procedure established by law.\n\n"
        "Schedule VII Corporate Social Responsibility Activities\n\n"
        "Activities which may be included by companies in their Corporate Social Responsibility Policies."
    )
    pages = [
        DocumentPage(page_number=1, text=raw_text[:400]),
        DocumentPage(page_number=2, text=raw_text[400:]),
    ]
    return ParsedDocument(
        document_id="test_sha256_doc_1234567890",
        filename="master_statutory_corpus.pdf",
        file_type="pdf",
        file_size_bytes=len(raw_text.encode("utf-8")),
        pages=pages,
        domain="corporate_law",
        raw_text=raw_text,
    )


def test_chunker_detects_legal_structures(legal_document):
    chunker = StructureAwareLegalChunker(chunk_size_chars=300, chunk_overlap_chars=30)
    chunks = chunker.chunk_document(legal_document)

    assert len(chunks) >= 3

    detected_sections = [c.section for c in chunks]
    assert any("Section 185" in s for s in detected_sections)
    assert any("Clause 4.1" in s for s in detected_sections)
    assert any("Article 21" in s for s in detected_sections)


def test_chunker_detects_subsections(legal_document):
    chunker = StructureAwareLegalChunker(chunk_size_chars=250, chunk_overlap_chars=20)
    chunks = chunker.chunk_document(legal_document)

    sub_sections = [c.sub_section for c in chunks if c.sub_section]
    assert any("(1)" in s or "(2)" in s for s in sub_sections)


def test_chunker_deterministic_uuidv5_ids(legal_document):
    chunker = StructureAwareLegalChunker(chunk_size_chars=400, chunk_overlap_chars=50)
    run1_chunks = chunker.chunk_document(legal_document)
    run2_chunks = chunker.chunk_document(legal_document)

    assert len(run1_chunks) == len(run2_chunks)
    for c1, c2 in zip(run1_chunks, run2_chunks):
        assert c1.chunk_id == c2.chunk_id
        assert c1.document_id == c2.document_id
        assert c1.chunk_index == c2.chunk_index


def test_chunker_adjacency_graph_pointers(legal_document):
    chunker = StructureAwareLegalChunker(chunk_size_chars=200, chunk_overlap_chars=20)
    chunks = chunker.chunk_document(legal_document)

    assert len(chunks) > 2
    # First chunk has no prev
    assert chunks[0].prev_chunk_id is None
    assert chunks[0].next_chunk_id == chunks[1].chunk_id

    # Middle chunks connect to prev and next
    for i in range(1, len(chunks) - 1):
        assert chunks[i].prev_chunk_id == chunks[i - 1].chunk_id
        assert chunks[i].next_chunk_id == chunks[i + 1].chunk_id

    # Last chunk has no next
    assert chunks[-1].prev_chunk_id == chunks[-2].chunk_id
    assert chunks[-1].next_chunk_id is None


def test_chunker_respects_size_limits():
    long_text = "Standard legal clause without break. " * 50
    doc = ParsedDocument(
        document_id="doc_large",
        filename="long.txt",
        file_type="txt",
        file_size_bytes=len(long_text),
        pages=[DocumentPage(page_number=1, text=long_text)],
        raw_text=long_text,
    )
    chunker = StructureAwareLegalChunker(chunk_size_chars=400, chunk_overlap_chars=50)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) > 1
    for chunk in chunks:
        # Each chunk text should be bounded reasonably around chunk_size_chars
        assert len(chunk.text) <= 500
        assert chunk.token_count > 0
