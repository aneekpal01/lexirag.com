"""Conservative deterministic legal claim extractor and exhibit reference parser."""

import re
from typing import Any, Optional
from app.core.logging import get_logger
from app.rag.claim_models import LegalClaim, SupportStatus

logger = get_logger(__name__)

# Exhibit citation pattern: matches [EXHIBIT 1], [EXHIBIT 2], [exhibit 3], etc.
EXHIBIT_PATTERN = re.compile(r"\[(?:EXHIBIT|exhibit|Exhibit)\s*(\d+)\]", re.IGNORECASE)

# Legal abbreviations that should not trigger sentence boundaries
ABBREVIATION_PATTERN = re.compile(
    r"\b(?:Sec|Secs|Art|Arts|No|Nos|Co|Corp|Ltd|Pvt|Inc|v|vs|para|paras|cl|cls|e\.g|i\.e|viz|Rs|u/s|w\.e\.f)\.$",
    re.IGNORECASE,
)

# Qualifying legal phrases that make sentence splitting unsafe
UNSAFE_QUALIFIERS = (
    "unless",
    "except",
    "subject to",
    "provided that",
    "provided further that",
    "notwithstanding",
    "where",
    "only if",
    "in the event that",
    "saving that",
    "without prejudice to",
    "save as otherwise",
    "as the case may be",
    "so far as may be",
)


class DeterministicClaimExtractor:
    """
    Extracts verifiable legal claims and parsed exhibit citations from synthesized answers.
    
    Adheres strictly to conservative decomposition rules: sentences containing legal
    qualifications (exceptions, provisos, non-obstante clauses) are kept intact to
    prevent manufacturing distorted atomic claims.
    """

    def __init__(self, available_exhibits: Optional[list[dict[str, Any]]] = None):
        """
        Args:
            available_exhibits: List of included evidence chunks from build_grounded_context_block.
                               Each chunk must contain server-side assigned 'exhibit_id'.
        """
        self.available_exhibits = available_exhibits or []
        self.exhibit_id_to_chunk: dict[str, dict[str, Any]] = {
            chunk["exhibit_id"]: chunk
            for chunk in self.available_exhibits
            if "exhibit_id" in chunk
        }
        self.valid_exhibit_ids: set[str] = set(self.exhibit_id_to_chunk.keys())

    def extract_claims(self, text: str) -> list[LegalClaim]:
        """
        Segments answer text into candidate claims with explicit or tentative exhibit attributions.
        """
        if not text or not text.strip():
            return []

        # If answer is an evidence-first refusal statement, do not manufacture claims
        if (
            "does not contain sufficient statutory evidence" in text.lower()
            or "zero matching statutory provisions" in text.lower()
            or "retrieved corpus yielded zero" in text.lower()
        ):
            logger.info("Detected negative evidence refusal; skipping claim extraction.")
            return []

        raw_sentences = self._segment_sentences(text)
        claims: list[LegalClaim] = []
        claim_counter = 1

        for sent_idx, sentence in enumerate(raw_sentences):
            clean_sent = sentence.strip()
            if not clean_sent or len(clean_sent) < 15:
                # Ignore trivial fragments or structural artifacts
                continue

            # Skip markdown section headers
            if clean_sent.startswith(("#", "###", "##", "**Evidentiary Limitations")):
                continue

            # Extract explicit exhibit tags: [EXHIBIT n]
            exhibit_matches = EXHIBIT_PATTERN.findall(clean_sent)
            cited_exhibit_ids = [f"EXHIBIT_{num}" for num in exhibit_matches]

            # Attempt safe decomposition only if no qualifying conditions exist
            sub_propositions = self._safely_decompose_propositions(clean_sent)

            for prop in sub_propositions:
                claim_id = f"claim-{claim_counter}"
                claim_counter += 1

                # Propagate cited exhibit IDs to proposition if present in parent sentence
                prop_exhibit_matches = EXHIBIT_PATTERN.findall(prop)
                prop_cited_ids = (
                    [f"EXHIBIT_{num}" for num in prop_exhibit_matches]
                    if prop_exhibit_matches
                    else cited_exhibit_ids
                )

                # Attribute chunk IDs
                attributed_chunks: list[str] = []
                for ex_id in prop_cited_ids:
                    if ex_id in self.exhibit_id_to_chunk:
                        chunk = self.exhibit_id_to_chunk[ex_id]
                        if chunk.get("chunk_id"):
                            attributed_chunks.append(chunk["chunk_id"])

                claim = LegalClaim(
                    claim_id=claim_id,
                    claim_text=prop,
                    sentence_index=sent_idx,
                    cited_exhibit_ids=prop_cited_ids,
                    attributed_chunk_ids=attributed_chunks,
                    support_status=SupportStatus.UNSUPPORTED,  # Initial state prior to verification
                    verification_rationale="",
                    is_atomic=len(sub_propositions) > 1,
                )
                claims.append(claim)

        logger.debug("Extracted %d legal claims from text of %d chars", len(claims), len(text))
        return claims

    def _segment_sentences(self, text: str) -> list[str]:
        """Segments prose into candidate legal sentences while respecting abbreviations."""
        # Normalize double newlines and list markers
        lines = text.split("\n")
        filtered_lines: list[str] = []
        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue
            if line_str.startswith(("#", "---", "===")):
                continue
            # Strip markdown list markers if bulleted
            if line_str.startswith(("- ", "* ", "1. ", "2. ", "3. ", "4. ", "5. ")):
                filtered_lines.append(re.sub(r"^[-*\d.]+\s*", "", line_str))
            else:
                filtered_lines.append(line_str)

        combined_text = " ".join(filtered_lines)

        # Split on sentence terminals followed by whitespace and an uppercase letter or bracket
        raw_splits = re.split(r"(?<=[.?!])\s+(?=[A-Z\[])", combined_text)
        sentences: list[str] = []

        buffer = ""
        for token in raw_splits:
            if not buffer:
                buffer = token
            else:
                # Check if buffer ends with an abbreviation like 'Sec.' or 'Co.'
                last_word = buffer.split()[-1] if buffer.split() else ""
                if ABBREVIATION_PATTERN.search(last_word):
                    buffer = f"{buffer} {token}"
                else:
                    sentences.append(buffer.strip())
                    buffer = token

        if buffer.strip():
            sentences.append(buffer.strip())

        return sentences

    def _safely_decompose_propositions(self, sentence: str) -> list[str]:
        """
        Decomposes unambiguous conjunctions or semicolons into distinct propositions.
        If the sentence contains legal qualifiers, exceptions, or negations, preserves it intact.
        """
        lowered = sentence.lower()

        # Rule 1: If sentence contains complex legal qualifiers, DO NOT split
        for qualifier in UNSAFE_QUALIFIERS:
            if qualifier in lowered:
                return [sentence]

        # Rule 2: If sentence contains double negation or complex prohibition, do not split
        if "shall not" in lowered or "neither" in lowered or "nor" in lowered:
            return [sentence]

        # Rule 3: Split on clean semicolons only if both sides form substantial statements (>= 5 words)
        if ";" in sentence:
            parts = [p.strip() for p in sentence.split(";") if p.strip()]
            if all(len(p.split()) >= 5 for p in parts):
                return parts

        # Default conservative rule: keep sentence intact as a single proposition
        return [sentence]
