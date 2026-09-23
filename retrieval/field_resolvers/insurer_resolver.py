"""
Insurer Resolver: Resolves the official issuing insurance company or utility provider from document evidence.
Rejects intermediaries, brokers, and financier banks. Never assumes from filename or cache.
"""

import re
from typing import Dict, Any, List, Optional
from retrieval.field_resolvers.base_resolver import BaseFieldResolver, ResolvedFieldCandidate


class InsurerResolver(BaseFieldResolver):
    field_name: str = "insurance_company"

    KNOWN_INSURERS = [
        "Go Digit General Insurance Ltd.",
        "Go Digit General Insurance",
        "Digit Insurance",
        "Go Digit",
        "IFFCO-TOKIO GENERAL INSURANCE CO.LTD",
        "IFFCO-TOKIO GENERAL INSURANCE CO. LTD.",
        "IFFCO Tokio General Insurance Co. Ltd.",
        "IFFCO-TOKIO",
        "Reliance General Insurance Company Limited",
        "Reliance General Insurance Co. Ltd.",
        "Reliance General Insurance",
        "Tata AIG General Insurance Company Limited",
        "Tata AIG General Insurance",
        "TATA AIG",
        "Bajaj Allianz General Insurance Co. Ltd.",
        "Bajaj Allianz General Insurance Company",
        "Bajaj Allianz General Insurance",
        "HDFC ERGO General Insurance Company Limited",
        "HDFC ERGO Health Insurance Limited",
        "HDFC ERGO General Insurance",
        "HDFC ERGO",
        "Star Health and Allied Insurance Co. Ltd.",
        "Star Health and Allied Insurance",
        "Star Health",
        "ICICI Lombard General Insurance Company Ltd.",
        "ICICI Lombard General Insurance",
        "ICICI Lombard",
        "The New India Assurance Co. Ltd.",
        "New India Assurance",
        "United India Insurance Company Limited",
        "United India Insurance",
        "Oriental Insurance Company Limited",
        "Oriental Insurance",
        "SBI General Insurance Company Limited",
        "SBI General Insurance",
        "Kotak Mahindra General Insurance Co. Ltd.",
        "Kotak Mahindra General Insurance",
        "Niva Bupa Health Insurance Company Limited",
        "Niva Bupa Health Insurance",
        "Care Health Insurance Limited",
        "Cholamandalam MS General Insurance Co. Ltd.",
        "Royal Sundaram General Insurance Co. Limited",
        "Paschim Gujarat Vij Company Limited",
        "Adani Electricity Mumbai Limited",
        "PGVCL",
        "MGVCL",
        "DGVCL",
        "UGVCL",
        "Adani Electricity",
    ]

    DISALLOWED_ORGS = [
        "POLICYBAZAAR", "POLICY BAZAAR", "COVERFOX", "INSURANCE BROKER", "WEB AGGREGATOR",
        "FINANCIER", "BANK", "AUTO LOAN", "HYPOTHECATION"
    ]

    def resolve(
        self,
        full_text: str,
        page_texts: Dict[int, str],
        layout: Optional[Any] = None,
        tables: Optional[List[Dict[str, Any]]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[ResolvedFieldCandidate]:
        first_page_text = page_texts.get(1, full_text[:4000])
        sorted_insurers = sorted(self.KNOWN_INSURERS, key=len, reverse=True)

        # Priority 1: Check table cells on Page 1 (e.g. Header table row 0)
        if tables:
            for t in tables:
                if t.get("page_number", 1) == 1:
                    for row in t.get("raw_matrix", [])[:3]:
                        for cell in row:
                            if not cell:
                                continue
                            c_str = " ".join(str(cell).strip().split())
                            if any(dis in c_str.upper() for dis in self.DISALLOWED_ORGS):
                                continue
                            for insurer in sorted_insurers:
                                words = re.findall(r"[A-Za-z0-9]+", insurer)
                                if not words:
                                    continue
                                pattern = r"[\s\.:\-]*".join(words)
                                if re.search(pattern, c_str, re.IGNORECASE):
                                    return ResolvedFieldCandidate(
                                        field_name=self.field_name,
                                        value=insurer,
                                        exact_label="Table Header / Insurer Block",
                                        raw_evidence=c_str[:120],
                                        page=1,
                                        confidence=0.99,
                                        method="table_cell",
                                        is_verified=True
                                    )

        # Priority 2: Match against registered insurer entities in header of Page 1 (first 3000 chars)
        header_text = first_page_text[:3000]
        for insurer in sorted_insurers:
            words = re.findall(r"[A-Za-z0-9]+", insurer)
            if not words:
                continue
            pattern = r"[\s\.:\-]*".join(words)
            m = re.search(pattern, header_text, re.IGNORECASE)
            if m:
                # Ensure match is not preceded by broker disclaimer
                start_idx = max(0, m.start() - 60)
                context_prefix = header_text[start_idx:m.start()].upper()
                if not any(dis in context_prefix for dis in ["BROKER", "AGENT", "AGGREGATOR"]):
                    return ResolvedFieldCandidate(
                        field_name=self.field_name,
                        value=insurer,
                        exact_label="Header Block",
                        raw_evidence=m.group(0).strip(),
                        page=1,
                        confidence=0.98,
                        method="header_entity_match",
                        is_verified=True
                    )

        # Priority 3: "issued by", "underwritten by", "for <company>", "insurance company <company>"
        m_rel = re.search(
            r"(?i)\b(?:issued\s*by|underwritten\s*by|for\s+and\s+on\s+behalf\s+of|insurance\s*company)[\s]*[:=\-]?\s*([^\n\r,;]{4,70})",
            first_page_text
        )
        if m_rel:
            rel_cand = m_rel.group(1).strip()
            if not any(dis in rel_cand.upper() for dis in self.DISALLOWED_ORGS):
                for insurer in sorted_insurers:
                    words = re.findall(r"[A-Za-z0-9]+", insurer)
                    pattern = r"[\s\.:\-]*".join(words)
                    if re.search(pattern, rel_cand, re.IGNORECASE):
                        return ResolvedFieldCandidate(
                            field_name=self.field_name,
                            value=insurer,
                            exact_label="Issued/Underwritten By",
                            raw_evidence=m_rel.group(0).strip(),
                            page=1,
                            confidence=0.96,
                            method="relationship_phrase",
                            is_verified=True
                        )

        # Priority 4: Explicit Insurer label with colon or separator
        m_label = re.search(
            r"(?i)\b(?:insurer|insurance\s*company|company\s*name)[\s]*[:=\-][\s]*([^\n\r,;]{4,70})",
            first_page_text
        )
        if m_label:
            val = m_label.group(1).strip()
            if not any(dis in val.upper() for dis in self.DISALLOWED_ORGS):
                for insurer in sorted_insurers:
                    words = re.findall(r"[A-Za-z0-9]+", insurer)
                    pattern = r"[\s\.:\-]*".join(words)
                    if re.search(pattern, val, re.IGNORECASE):
                        return ResolvedFieldCandidate(
                            field_name=self.field_name,
                            value=insurer,
                            exact_label="Explicit Insurer Label",
                            raw_evidence=m_label.group(0).strip(),
                            page=1,
                            confidence=0.96,
                            method="explicit_label_match",
                            is_verified=True
                        )
                if len(val) >= 5:
                    return ResolvedFieldCandidate(
                        field_name=self.field_name,
                        value=val,
                        exact_label="Explicit Insurer Label",
                        raw_evidence=m_label.group(0).strip(),
                        page=1,
                        confidence=0.90,
                        method="explicit_label",
                        is_verified=True
                    )

        # Priority 5: Full document search for known insurer entities (e.g. multi-page policies)
        for pno in sorted(page_texts.keys()):
            p_text = page_texts[pno]
            for insurer in sorted_insurers:
                words = re.findall(r"[A-Za-z0-9]+", insurer)
                if not words:
                    continue
                pattern = r"[\s\.:\-]*".join(words)
                m = re.search(pattern, p_text, re.IGNORECASE)
                if m:
                    return ResolvedFieldCandidate(
                        field_name=self.field_name,
                        value=insurer,
                        exact_label="Full Document Entity Match",
                        raw_evidence=m.group(0).strip(),
                        page=pno,
                        confidence=0.88,
                        method="full_document_search",
                        is_verified=True
                    )

        return None
