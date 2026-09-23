"""
Policy Number Resolver: Dedicated resolver for policy number, consumer number, and contract identifier.
Supports multi-token values (e.g. '1-12IXB8DM P400') and cleanly separates multiple labels on the same line.
Rule: Zero hallucination, strict exact-character preservation, ignores PAN/CIN/GSTIN/IRDAI.
"""

import re
from typing import Dict, Any, List, Optional
from retrieval.field_resolvers.base_resolver import BaseFieldResolver, ResolvedFieldCandidate
from ingestion.table_extractor import find_policy_numbers_in_tables


class PolicyNumberResolver(BaseFieldResolver):
    field_name: str = "policy_number"
    aliases: List[str] = [
        "policy certificate no",
        "policy certificate number",
        "policy certificate",
        "policy number",
        "policy no.",
        "policy no",
        "policy #",
        "policy id",
        "certificate of insurance no",
        "certificate no",
        "certificate number",
        "contract number",
        "contract no",
        "policy reference",
        "consumer account number",
        "consumer no",
        "consumer number",
        "consumer id",
        "ca no",
    ]

    NEGATIVE_PATTERNS = [
        r"\b[A-Z]{5}\d{4}[A-Z]\b",  # PAN
        r"\b\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z\d]{1}[Z]{1}[A-Z\d]{1}\b",  # GSTIN
        r"\b(?:CIN|U\d{5}[A-Z]{2}\d{4}[A-Z]{3}\d{6})\b",  # CIN
        r"\b(?:IRDA|IRDAI)[\s\w/:-]*\d+",  # IRDAI registration
        r"\b(?:FORM|SCHEDULE|VERSION)\b",
    ]

    DISALLOWED_WORDS = [
        "PAYABLE", "PREVIOUS", "BENEFIT", "UNLESS", "POLICY", "INSURANCE",
        "SCHEDULE", "CONDITIONS", "LIMITATIONS", "WORDINGS", "VALIDITY",
        "REGISTRATION DETAILS", "RC COPY", "PREMIUM CHEQUE", "CHEQUE",
        "PRIVATE CAR", "TWO WHEELER", "COMMERCIAL VEHICLE", "CERTIFICATE OF INSURANCE"
    ]

    def resolve(
        self,
        full_text: str,
        page_texts: Dict[int, str],
        layout: Optional[Any] = None,
        tables: Optional[List[Dict[str, Any]]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[ResolvedFieldCandidate]:
        candidates: List[ResolvedFieldCandidate] = []

        # -------------------------------------------------------------
        # STEP 1: Table Extraction (Highest Priority)
        # -------------------------------------------------------------
        if tables:
            t_pols = find_policy_numbers_in_tables(tables)
            valid_t_cands = []
            for tp in t_pols:
                val = tp["value"]
                if not self._is_negative_candidate(val):
                    valid_t_cands.append(tp)

            if valid_t_cands:
                primary = valid_t_cands[0]
                all_vals = [c["value"] for c in valid_t_cands]
                sec_val = valid_t_cands[1]["value"] if len(valid_t_cands) > 1 else None

                return ResolvedFieldCandidate(
                    field_name=self.field_name,
                    value=primary["value"],
                    exact_label=primary["exact_label"],
                    raw_evidence=primary["evidence"],
                    page=primary["page"],
                    confidence=0.99,
                    method="table_cell",
                    is_verified=True,
                    metadata={
                        "all_policy_numbers": all_vals,
                        "secondary_policy_number": sec_val,
                        "candidates_count": len(valid_t_cands)
                    }
                )

        # -------------------------------------------------------------
        # STEP 2: Spatial Coordinate Extraction via Layout
        # -------------------------------------------------------------
        if layout and hasattr(layout, "extract_value_for_label"):
            spatial_cands = layout.extract_value_for_label(
                label_aliases=self.aliases,
                value_pattern=r"[A-Za-z0-9][A-Za-z0-9/\- ]{4,35}[A-Za-z0-9]",
                max_dist_right=350.0,
                max_dist_below=40.0
            )
            for sc in spatial_cands:
                raw_lbl = sc.get("raw_text", "").upper()
                if "PREVIOUS" in raw_lbl:
                    continue
                val = sc["value"].strip()
                sub_cands = self._extract_multi_label_candidates(val)
                for cand_val in (sub_cands or [val]):
                    if not self._is_negative_candidate(cand_val):
                        candidates.append(ResolvedFieldCandidate(
                            field_name=self.field_name,
                            value=cand_val,
                            exact_label=sc.get("raw_text", ""),
                            raw_evidence=sc["raw_text"],
                            page=sc["page"],
                            bbox=sc.get("bbox"),
                            confidence=0.96,
                            method=sc.get("method", "spatial"),
                            is_verified=True
                        ))

        # Check for stacked values directly above 'Policy No.' (e.g. Digit Policies: 'D091386640 / 20012023 \n Policy No.')
        for pno in sorted(page_texts.keys())[:2]:
            lines = [l.strip() for l in page_texts[pno].split("\n") if l.strip()]
            for idx, l in enumerate(lines):
                if re.search(r"(?i)^(?:policy\s*(?:#|no\.?|number)|certificate\s*no\.?)$", l):
                    if idx > 0:
                        prev_l = lines[idx - 1]
                        if "PREVIOUS" not in prev_l.upper() and not self._is_negative_candidate(prev_l):
                            if re.search(r"^[A-Za-z0-9][A-Za-z0-9/\- ]{4,35}[A-Za-z0-9]$", prev_l):
                                candidates.append(ResolvedFieldCandidate(
                                    field_name=self.field_name,
                                    value=prev_l,
                                    exact_label=f"{l} [ABOVE]",
                                    raw_evidence=f"{prev_l} [ABOVE] {l}",
                                    page=pno,
                                    confidence=0.97,
                                    method="stacked_above_label",
                                    is_verified=True
                                ))

        # -------------------------------------------------------------
        # STEP 3: Fallback Text / Regex Extraction (Pages 1-2 only)
        # -------------------------------------------------------------
        for pno in sorted(page_texts.keys())[:2]:
            text = page_texts[pno]
            # Pattern matching each policy label without merging adjacent labels
            pol_regex = r"(?i)(?<!previous\s)(policy\s*(?:#|no\.?|number|certificate\s*no\.?)[\s\.:/-]*)\s*([A-Za-z0-9][A-Za-z0-9/\- ]{3,35}[A-Za-z0-9])(?=[ \t]*(?:policy\s*(?:#|no|number)|date|period|unique|\r|\n|$))"
            for m in re.finditer(pol_regex, text):
                lbl = m.group(1).strip()
                cand = m.group(2).strip()
                if not self._is_negative_candidate(cand) and "PREVIOUS" not in lbl.upper():
                    candidates.append(ResolvedFieldCandidate(
                        field_name=self.field_name,
                        value=cand,
                        exact_label=lbl,
                        raw_evidence=m.group(0).strip(),
                        page=pno,
                        confidence=0.92,
                        method="regex_alias",
                        is_verified=True
                    ))

        if candidates:
            # Sort by highest confidence, lowest page
            candidates.sort(key=lambda c: (c.confidence, -c.page), reverse=True)
            primary = candidates[0]
            # Store all unique discovered valid policy numbers in metadata
            valid_vals = [c.value for c in candidates if not self._is_negative_candidate(c.value)]
            unique_vals = list(dict.fromkeys(valid_vals))
            primary.metadata["all_policy_numbers"] = unique_vals
            if len(unique_vals) > 1:
                primary.metadata["secondary_policy_number"] = unique_vals[1]
            return primary

        return None

    def _extract_multi_label_candidates(self, line: str) -> List[str]:
        """Separates multiple policy numbers if present in a single line."""
        pattern = r"(?i)(?:policy\s*(?:#|no\.?|number|certificate\s*no\.?)[\s\.:/-]*)\s*([A-Za-z0-9][A-Za-z0-9/\- ]{3,35}[A-Za-z0-9])(?=[ \t]*(?:policy\s*(?:#|no|number)|date|period|unique|\r|\n|$))"
        matches = [m.group(1).strip() for m in re.finditer(pattern, line)]
        return matches

    def _is_negative_candidate(self, text: str) -> bool:
        """Filters out non-policy numbers like PAN, GSTIN, CIN, pure dates, pure currency, boilerplate, plan names."""
        clean = " ".join(text.strip().split())
        if len(clean) < 4 or len(clean) > 40:
            return True
        # A valid policy identifier MUST contain at least one digit
        if not any(char.isdigit() for char in clean):
            return True
        # Reject checklist items e.g. "b. Registration Details/RC Copy"
        if re.match(r"^[a-zA-Z0-9][\.\)]\s+", clean):
            return True
        upper_c = clean.upper()
        # Reject plan names and generic non-policy labels
        plan_and_label_terms = [
            "SUPPORT PLUS", "GOLD PLAN", "SILVER PLAN", "PLATINUM PLAN",
            "COMPREHENSIVE", "PACKAGE POLICY", "LIABILITY ONLY", "CUSTOMER ID",
            "REGISTRATION NO", "ENGINE NO", "CHASSIS NO", "ROAD", "NAGAR", "STREET", "PLOT"
        ]
        for term in plan_and_label_terms:
            if term in upper_c:
                return True
        for dis in self.DISALLOWED_WORDS:
            if dis in upper_c:
                return True
        for neg in self.NEGATIVE_PATTERNS:
            if re.search(neg, clean, re.IGNORECASE):
                return True
        # Reject pure dates (e.g. 01/04/2024 or 15/03/2019 18:43:24)
        if re.match(r"^\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}(?:\s+\d\d:\d\d:\d\d)?$", clean):
            return True
        return False
