"""Structure-aware chunking engine for legal and compliance documents."""

import re
import uuid
from typing import Optional

from app.ingestion.models import DocumentChunk, ParsedDocument

# Common legal structural patterns in Indian and common-law statutes & contracts
LEGAL_STRUCTURE_REGEX = re.compile(
    r"(?P<structure>"
    r"(?:Section\s+\d+[A-Z]*(?:\([0-9a-zA-Z]+\))*)"
    r"|(?:Sec\.\s+\d+[A-Z]*)"
    r"|(?:Article\s+(?:[0-9]+[A-Z]*|[IVXLCDM]+))"
    r"|(?:Clause\s+\d+(?:\.\d+)*)"
    r"|(?:Schedule\s+(?:[0-9]+|[IVXLCDM]+))"
    r"|(?:Order\s+[IVXLCDM]+)"
    r"|(?:Rule\s+\d+[A-Z]*)"
    r")",
    re.IGNORECASE,
)

SUB_SECTION_REGEX = re.compile(r"^\s*(\([0-9a-zA-Z]+\))\s*")
HEADING_REGEX = re.compile(r"^(?:##\s*|[A-Z\s]{4,}:?\s*$)(?P<heading>.+)")


class StructureAwareLegalChunker:
    """
    Partitions parsed legal documents into structured chunks respecting legal boundaries.
    
    Features:
    1. Structure Detection: Identifies Sections, Clauses, Articles, Sub-sections, and Schedules.
    2. Context Propagation: Carries current active Section and Heading across page/paragraph splits.
    3. Deterministic Identifiers: Generates UUIDv5 for every chunk based on document SHA-256 and index.
    4. Adjacency Graph Linking: Populates prev_chunk_id and next_chunk_id for downstream evidence navigation.
    """

    def __init__(
        self,
        chunk_size_chars: int = 1000,
        chunk_overlap_chars: int = 150,
    ):
        self.chunk_size_chars = max(200, chunk_size_chars)
        self.chunk_overlap_chars = max(0, min(chunk_overlap_chars, self.chunk_size_chars // 2))

    def _estimate_token_count(self, text: str) -> int:
        """Heuristic token estimation (~4 characters per token)."""
        return max(1, len(text) // 4)

    def chunk_document(self, document: ParsedDocument) -> list[DocumentChunk]:
        """Processes a ParsedDocument into a sequence of linked, structure-aware DocumentChunks."""
        raw_chunks: list[dict] = []
        current_section = "General Provision"
        current_sub_section: Optional[str] = None
        current_heading: Optional[str] = None

        for page in document.pages:
            page_number = page.page_number
            # Split page text into natural paragraphs
            paragraphs = [p.strip() for p in page.text.split("\n\n") if p.strip()]

            for para in paragraphs:
                # Detect structure cues
                struct_match = LEGAL_STRUCTURE_REGEX.search(para)
                is_new_section = False
                if struct_match:
                    detected_section = struct_match.group("structure").strip()
                    if detected_section != current_section:
                        is_new_section = True
                        current_section = detected_section

                sub_match = SUB_SECTION_REGEX.search(para)
                if sub_match:
                    current_sub_section = sub_match.group(1).strip()

                heading_match = HEADING_REGEX.search(para)
                if heading_match:
                    current_heading = heading_match.group("heading").strip().strip("#").strip()

                # If paragraph fits comfortably in remaining buffer, accumulate unless starting a new section
                para_len = len(para)
                should_start_new_chunk = (
                    not raw_chunks
                    or is_new_section
                    or (len(raw_chunks[-1]["text"]) + para_len + 2 > self.chunk_size_chars)
                )

                if should_start_new_chunk:
                    # Start a new chunk
                    overlap_prefix = ""
                    if raw_chunks and self.chunk_overlap_chars > 0 and not is_new_section:
                        last_text = raw_chunks[-1]["text"]
                        overlap_prefix = last_text[-self.chunk_overlap_chars :] + " "

                    # If the single paragraph itself exceeds chunk_size_chars, break it cleanly
                    if para_len > self.chunk_size_chars:
                        start_idx = 0
                        while start_idx < para_len:
                            end_idx = min(start_idx + self.chunk_size_chars, para_len)
                            slice_text = para[start_idx:end_idx].strip()
                            raw_chunks.append(
                                {
                                    "text": slice_text,
                                    "page_number": page_number,
                                    "section": current_section,
                                    "sub_section": current_sub_section,
                                    "heading": current_heading,
                                }
                            )
                            start_idx += self.chunk_size_chars - self.chunk_overlap_chars
                    else:
                        raw_chunks.append(
                            {
                                "text": (overlap_prefix + para).strip(),
                                "page_number": page_number,
                                "section": current_section,
                                "sub_section": current_sub_section,
                                "heading": current_heading,
                            }
                        )
                else:
                    # Append paragraph to existing chunk
                    raw_chunks[-1]["text"] += f"\n\n{para}"
                    # Update page number to latest if span crosses boundary
                    raw_chunks[-1]["page_number"] = page_number

        if not raw_chunks:
            # Fallback if document had no paragraph breaks
            raw_chunks.append(
                {
                    "text": document.raw_text[: self.chunk_size_chars],
                    "page_number": 1,
                    "section": current_section,
                    "sub_section": current_sub_section,
                    "heading": current_heading,
                }
            )

        # Build final deterministic DocumentChunk objects with UUIDv5 and graph adjacency
        total_chunks = len(raw_chunks)
        chunk_ids = [
            str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{document.document_id}::chunk_{i}"))
            for i in range(total_chunks)
        ]

        document_chunks: list[DocumentChunk] = []
        for index, item in enumerate(raw_chunks):
            prev_id = chunk_ids[index - 1] if index > 0 else None
            next_id = chunk_ids[index + 1] if index < total_chunks - 1 else None

            chunk = DocumentChunk(
                chunk_id=chunk_ids[index],
                document_id=document.document_id,
                document_name=document.filename,
                source=document.filename,
                file_type=document.file_type,
                page_number=item["page_number"],
                chunk_index=index,
                section=item["section"],
                sub_section=item["sub_section"],
                heading=item["heading"],
                domain=document.domain,
                text=item["text"],
                act_name=document.filename.rsplit(".", 1)[0].replace("_", " ").title(),
                prev_chunk_id=prev_id,
                next_chunk_id=next_id,
                token_count=self._estimate_token_count(item["text"]),
            )
            document_chunks.append(chunk)

        return document_chunks
