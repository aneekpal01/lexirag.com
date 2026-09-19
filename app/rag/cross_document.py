"""Cross-document relation detection and provision alignment engine for LexiRAG Phase 5."""

import re
from typing import Any, Optional
from app.core.logging import get_logger
from app.rag.claim_models import SupportStatus
from app.rag.comparison_models import (
    ComparisonAnalysisSummary,
    ComparisonRelation,
    ComparisonRelationType,
    DocumentEvidenceGroup,
)

logger = get_logger(__name__)

# Patterns indicating comparative analysis in claim text
COMPARATIVE_MARKERS = (
    "whereas",
    "while",
    "unlike",
    "in contrast",
    "compared to",
    "longer than",
    "shorter than",
    "higher than",
    "lower than",
    "differs from",
    "difference between",
    "different from",
    "exceeds",
    "both contracts",
    "both agreements",
    "both documents",
    "neither contract",
    "requires",
)

# Numeric difference pattern
NUMBER_PATTERN = re.compile(r"\b(\d+\s*(?:days?|months?|years?|percent|%|lakh|crore)?)\b", re.IGNORECASE)


class CrossDocumentRelationDetector:
    """
    Detects and classifies relational semantics across explicitly requested documents.
    
    Adheres strictly to the following principles:
    - Bounded to verified claims and retrieved corpus exhibits.
    - Employs cautious semantics: uses 'apparent difference' and 'potential conflict',
      strictly avoiding claims of binding legal contradiction or illegal breach.
    - Accounts for asymmetric/missing evidence without hallucinating provisions.
    """

    def __init__(self, target_document_ids: Optional[list[str]] = None):
        self.target_document_ids = [d.strip() for d in target_document_ids if d and d.strip()] if target_document_ids else []

    def detect_relations(
        self,
        claims: Any = None,
        included_chunks: Optional[list[dict[str, Any]]] = None,
        document_groups: Optional[Any] = None,
        missing_documents: Optional[list[str]] = None,
    ) -> ComparisonAnalysisSummary:
        """
        Extracts structured cross-document relations and builds ComparisonAnalysisSummary.
        """
        # Determine whether first parameter is claims or chunks
        actual_claims: list[dict[str, Any]] = []
        if isinstance(claims, list):
            if claims and any("chunk_id" in item or "content" in item for item in claims if isinstance(item, dict)):
                # Called directly with chunks
                included_chunks = claims
                actual_claims = []
            else:
                actual_claims = claims

        included_chunks = included_chunks or []
        missing_documents = missing_documents or []

        # Convert list of groups to dict if list is provided
        if isinstance(document_groups, list):
            doc_groups_dict = {}
            for g in document_groups:
                d_id = g.get("document_id") if isinstance(g, dict) else getattr(g, "document_id", None)
                if d_id:
                    doc_groups_dict[d_id] = g if isinstance(g, dict) else g.model_dump()
            document_groups = doc_groups_dict

        # Auto-construct document_groups if missing or empty
        if not document_groups:
            document_groups = {}
            for c in included_chunks:
                d_id = c.get("document_id") or "unknown_doc"
                if d_id not in document_groups:
                    document_groups[d_id] = {
                        "document_name": c.get("document_name") or d_id,
                        "primary_chunk_count": 0,
                        "supporting_chunk_count": 0,
                        "exhibit_ids": [],
                        "provisions_covered": [],
                        "has_sufficient_evidence": True,
                    }
                grp = document_groups[d_id]
                if c.get("evidence_role") == "primary":
                    grp["primary_chunk_count"] += 1
                else:
                    grp["supporting_chunk_count"] += 1
                ex_id = c.get("exhibit_id")
                if ex_id and ex_id not in grp["exhibit_ids"]:
                    grp["exhibit_ids"].append(ex_id)
                prov = c.get("section") or c.get("heading")
                if prov and prov not in grp["provisions_covered"]:
                    grp["provisions_covered"].append(prov)

        if not self.target_document_ids:
            self.target_document_ids = list(document_groups.keys())

        relations: list[ComparisonRelation] = []
        comparison_notes: list[str] = []

        if missing_documents:
            for m_doc in missing_documents:
                note = (
                    f"Evidence gap: Document '{m_doc}' yielded zero matching provisions in the indexed corpus. "
                    "Cross-document comparison is limited to available document evidence."
                )
                if note not in comparison_notes:
                    comparison_notes.append(note)

        # Build exhibit lookup
        exhibit_by_id: dict[str, dict[str, Any]] = {
            chunk["exhibit_id"]: chunk for chunk in included_chunks if "exhibit_id" in chunk
        }

        # 1. Analyze verified claims that compare documents
        relation_counter = 1
        for claim in actual_claims:
            claim_text = claim.get("claim_text", "")
            cited_exhibits = (
                claim.get("cited_exhibits", [])
                or claim.get("supporting_exhibits", [])
                or claim.get("cited_exhibit_ids", [])
            )
            
            # Identify which documents are touched by the cited exhibits
            exhibit_docs: dict[str, list[dict[str, Any]]] = {}
            for ex_id in cited_exhibits:
                ex = exhibit_by_id.get(ex_id)
                if ex:
                    doc_id = ex.get("document_id") or "unknown"
                    exhibit_docs.setdefault(doc_id, []).append(ex)

            # Check if this claim is comparative across distinct documents
            lowered_claim = claim_text.lower()
            is_comparative = any(m in lowered_claim for m in COMPARATIVE_MARKERS) or len(exhibit_docs) >= 2

            if is_comparative and len(exhibit_docs) >= 2:
                doc_keys = list(exhibit_docs.keys())
                source_doc_id = doc_keys[0]
                target_doc_id = doc_keys[1]

                source_chunks = exhibit_docs[source_doc_id]
                target_chunks = exhibit_docs[target_doc_id]

                source_doc_name = source_chunks[0].get("document_name") or source_doc_id
                target_doc_name = target_chunks[0].get("document_name") or target_doc_id

                source_provision = source_chunks[0].get("section") or source_chunks[0].get("heading") or "General Provision"
                target_provision = target_chunks[0].get("section") or target_chunks[0].get("heading") or "General Provision"

                # Classify relation type
                rel_type, is_conflict, topic, summary_text = self._classify_relation(
                    claim_text=claim_text,
                    source_doc_name=source_doc_name,
                    source_provision=source_provision,
                    target_doc_name=target_doc_name,
                    target_provision=target_provision,
                    source_chunks=source_chunks,
                    target_chunks=target_chunks,
                )

                # Parse support status
                raw_status = claim.get("support_status", "SUPPORTED")
                support_enum = (
                    SupportStatus(raw_status)
                    if raw_status in SupportStatus._value2member_map_
                    else SupportStatus.SUPPORTED
                )

                rel = ComparisonRelation(
                    relation_id=f"rel-{relation_counter}",
                    source_document_id=source_doc_id,
                    source_document_name=source_doc_name,
                    source_provision=source_provision,
                    target_document_id=target_doc_id,
                    target_document_name=target_doc_name,
                    target_provision=target_provision,
                    relation_type=rel_type,
                    provision_topic=topic,
                    claim_id=claim.get("claim_id"),
                    cited_exhibits=cited_exhibits,
                    summary=summary_text,
                    support_status=support_enum,
                    is_potential_conflict=is_conflict,
                )
                relations.append(rel)
                relation_counter += 1

        # 2. Pairwise provision alignment if no claim-based relations were extracted
        if not relations and len(document_groups) >= 2 and not missing_documents:
            fallback_rel = self._attempt_pairwise_alignment(
                included_chunks=included_chunks,
                relation_counter=relation_counter,
            )
            if fallback_rel:
                relations.extend(fallback_rel)

        # Build DocumentEvidenceGroup objects
        doc_group_objs: list[DocumentEvidenceGroup] = []
        for d_id, group_dict in document_groups.items():
            doc_group_objs.append(
                DocumentEvidenceGroup(
                    document_id=d_id,
                    document_name=group_dict.get("document_name", d_id),
                    primary_chunk_count=group_dict.get("primary_chunk_count", 0),
                    supporting_chunk_count=group_dict.get("supporting_chunk_count", 0),
                    exhibit_ids=group_dict.get("exhibit_ids", []),
                    provisions_covered=group_dict.get("provisions_covered", []),
                    has_sufficient_evidence=group_dict.get("has_sufficient_evidence", True),
                )
            )

        if not comparison_notes:
            comparison_notes.append(
                f"Evaluated {len(relations)} cross-document relation(s) across {len(doc_group_objs)} requested document(s)."
            )

        summary = ComparisonAnalysisSummary(
            is_comparison=True,
            is_multi_document=len(self.target_document_ids) > 1,
            document_count=len(doc_group_objs),
            target_document_ids=self.target_document_ids,
            document_groups=doc_group_objs,
            relations=relations,
            missing_documents=missing_documents,
            comparison_notes=comparison_notes,
        )

        logger.info(
            "Cross-document relation detection finished | relations: %d | docs: %d | missing: %d",
            len(relations),
            len(doc_group_objs),
            len(missing_documents),
        )
        return summary

    def _classify_relation(
        self,
        claim_text: str,
        source_doc_name: str,
        source_provision: str,
        target_doc_name: str,
        target_provision: str,
        source_chunks: list[dict[str, Any]],
        target_chunks: list[dict[str, Any]],
    ) -> tuple[ComparisonRelationType, bool, str, str]:
        """Categorizes relation using cautious phrasing."""
        lowered = claim_text.lower()
        source_text = " ".join(c.get("content", "") for c in source_chunks).lower()
        target_text = " ".join(c.get("content", "") for c in target_chunks).lower()
        combined_text = f"{lowered} {source_provision.lower()} {target_provision.lower()} {source_text} {target_text}"

        # Detect topic
        topic = "general provision"
        if any(w in combined_text for w in ("termination", "terminate", "notice period", "days notice")):
            topic = "termination"
        elif any(w in combined_text for w in ("confidential", "confidentiality", "non-disclosure")):
            topic = "confidentiality"
        elif any(w in combined_text for w in ("jurisdiction", "governing law", "courts in", "venue", "dispute")):
            topic = "governing law / jurisdiction"
        elif any(w in combined_text for w in ("indemnity", "indemnify", "hold harmless")):
            topic = "indemnity"
        elif any(w in combined_text for w in ("liability", "damages")):
            topic = "liability"

        # 1. Potential Conflict: exclusive jurisdiction in different venues, or mutually incompatible prohibitions
        venues = ["delhi", "mumbai", "bengaluru", "kolkata", "chennai", "hyderabad"]
        source_venues = [v for v in venues if v in source_text]
        target_venues = [v for v in venues if v in target_text]
        is_venue_conflict = bool(source_venues and target_venues and set(source_venues) != set(target_venues))

        has_prohibition_source = any(p in source_text for p in ("shall not", "prohibited", "exclusive"))
        has_permission_target = any(p in target_text for p in ("shall be permitted", "may", "non-exclusive"))
        is_conflict_keyword = any(k in lowered for k in ("conflict", "inconsistent", "contradiction", "incompatible"))

        if is_venue_conflict or (has_prohibition_source and has_permission_target) or is_conflict_keyword:
            return (
                ComparisonRelationType.POTENTIAL_CONFLICT,
                True,
                topic,
                f"Potential conflict: Provisions between {source_doc_name} ({source_provision}) and "
                f"{target_doc_name} ({target_provision}) suggest divergent or competing obligations.",
            )

        # 2. Differs: quantitative or operational difference
        numbers_in_claim = NUMBER_PATTERN.findall(claim_text)
        numbers_source = NUMBER_PATTERN.findall(source_text)
        numbers_target = NUMBER_PATTERN.findall(target_text)
        is_num_diff = bool(numbers_source and numbers_target and set(numbers_source) != set(numbers_target))
        is_diff_keyword = any(k in lowered for k in ("differ", "different", "whereas", "while", "unlike", "contrast", "longer", "shorter", "higher", "lower"))

        if len(numbers_in_claim) >= 2 or is_num_diff or is_diff_keyword:
            s_val = numbers_source[0] if numbers_source else "specified terms"
            t_val = numbers_target[0] if numbers_target else "specified terms"
            return (
                ComparisonRelationType.DIFFERS,
                False,
                topic,
                f"Apparent difference: {source_doc_name} specifies '{s_val}' whereas "
                f"{target_doc_name} specifies '{t_val}'.",
            )

        # 3. Overlaps: conditional or qualified terms
        is_overlap_keyword = any(k in lowered for k in ("subject to", "provided that", "except", "overlap", "additional"))
        if is_overlap_keyword:
            return (
                ComparisonRelationType.OVERLAPS,
                False,
                topic,
                f"Overlapping provisions: {source_doc_name} ({source_provision}) and "
                f"{target_doc_name} ({target_provision}) share subject matter with conditional variations.",
            )

        # 4. Supports: concordant support / mutual confidentiality
        is_support_keyword = any(k in lowered for k in ("both", "similar", "consistent", "mirrors", "same"))
        is_mutual_confidentiality = "confidential" in source_text and "confidential" in target_text
        if is_support_keyword or is_mutual_confidentiality:
            return (
                ComparisonRelationType.SUPPORTS,
                False,
                topic,
                f"Consistent provisions: {source_doc_name} ({source_provision}) and "
                f"{target_doc_name} ({target_provision}) establish mutually supportive terms.",
            )

        # 5. Default to mentions
        return (
            ComparisonRelationType.MENTIONS,
            False,
            topic,
            f"Provisional comparison: {source_doc_name} ({source_provision}) and "
            f"{target_doc_name} ({target_provision}) address related contractual subject matter.",
        )

    def _attempt_pairwise_alignment(
        self,
        included_chunks: list[dict[str, Any]],
        relation_counter: int,
    ) -> list[ComparisonRelation]:
        """Pairwise provision alignment fallback when claims did not link multiple exhibits directly."""
        chunks_by_doc: dict[str, list[dict[str, Any]]] = {}
        for c in included_chunks:
            d_id = c.get("document_id")
            if d_id:
                chunks_by_doc.setdefault(d_id, []).append(c)

        doc_keys = list(chunks_by_doc.keys())
        if len(doc_keys) < 2:
            return []

        doc_a_id, doc_b_id = doc_keys[0], doc_keys[1]
        chunks_a = chunks_by_doc[doc_a_id]
        chunks_b = chunks_by_doc[doc_b_id]

        relations: list[ComparisonRelation] = []
        for c_a in chunks_a:
            sec_a = c_a.get("section") or c_a.get("heading") or "General Provision"
            for c_b in chunks_b:
                sec_b = c_b.get("section") or c_b.get("heading") or "General Provision"
                
                # Check for same section number, heading match, or shared topical keywords
                text_a = c_a.get("content", "").lower()
                text_b = c_b.get("content", "").lower()
                same_topic = any(
                    (kw in text_a and kw in text_b)
                    for kw in ("termination", "confidential", "jurisdiction", "governing law", "indemnity", "liability", "arbitration")
                )

                if sec_a.lower() == sec_b.lower() or same_topic or (c_a.get("heading") and c_a.get("heading") == c_b.get("heading")):
                    rel_type, is_conflict, topic, summary = self._classify_relation(
                        claim_text="",
                        source_doc_name=c_a.get("document_name", doc_a_id),
                        source_provision=sec_a,
                        target_doc_name=c_b.get("document_name", doc_b_id),
                        target_provision=sec_b,
                        source_chunks=[c_a],
                        target_chunks=[c_b],
                    )
                    exhibits = [e for e in (c_a.get("exhibit_id"), c_b.get("exhibit_id")) if e]
                    relations.append(
                        ComparisonRelation(
                            relation_id=f"rel-{relation_counter}",
                            source_document_id=doc_a_id,
                            source_document_name=c_a.get("document_name", doc_a_id),
                            source_provision=sec_a,
                            target_document_id=doc_b_id,
                            target_document_name=c_b.get("document_name", doc_b_id),
                            target_provision=sec_b,
                            relation_type=rel_type,
                            provision_topic=topic,
                            claim_id=None,
                            cited_exhibits=exhibits,
                            summary=summary,
                            support_status=SupportStatus.SUPPORTED,
                            is_potential_conflict=is_conflict,
                        )
                    )
                    relation_counter += 1
                    break
        return relations
