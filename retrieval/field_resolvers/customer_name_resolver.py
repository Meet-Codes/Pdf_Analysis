"""
Customer Name Resolver: Dedicated resolver for insured party, proposer, policyholder, or consumer name.
Uses table header blocks and coordinates first (restricting to Page 1-2); enforces strict negative validation
against policy wording (e.g. 'defence', 'settlement', 'conditions').
"""

import re
from typing import Dict, Any, List, Optional, Tuple
from retrieval.field_resolvers.base_resolver import BaseFieldResolver, ResolvedFieldCandidate
from ingestion.table_extractor import find_proposer_in_header_tables
from schemas.base import DocumentType


class CustomerNameResolver(BaseFieldResolver):
    field_name: str = "customer_name"
    aliases: List[str] = [
        "name of the insured",
        "name of insured",
        "insured name",
        "insured's name",
        "proposer's full name",
        "proposer full name",
        "proposer name",
        "name of proposer",
        "proposer (insured)",
        "policyholder name",
        "policy holder name",
        "policyholder (proposer)",
        "consumer name",
        "name of consumer",
        "customer name",
        "fleet owner",
        "insured employer",
        "employer name",
        "prepared for",
        "billed to",
        "primary party name"
    ]

    DISALLOWED_TERMS = [
        "DEFENCE", "SETTLEMENT", "CONDITIONS", "LIMITATIONS", "POLICY",
        "PREMIUM", "VEHICLE", "CONDUCT", "PROCEEDINGS", "COURT", "LIABILITY",
        "SCHEDULE", "SECTION", "CERTIFICATE", "ENDORSEMENT", "CLAUSE",
        "DISCLAIM", "HEREBY", "STATUTORY", "UNDERWRITTEN", "INSURER",
        "BENEFIT", "RECEIPT", "INVOICE", "OFFICE", "BRANCH", "COMPANY",
        "GENERAL", "INSURANCE", "LIMITED", "LTD", "ADDRESS", "DETAILS",
        "COMMERCIAL", "PACKAGE", "PRIVATE", "PERIOD", "SUM INSURED",
        "POLICYBAZAAR", "POLICY BAZAAR", "COVERFOX", "BROKER", "AGENT",
        "INTERMEDIARY", "ADVISOR", "AGGREGATOR", "BANK", "FINANCIER",
        "DEFENCE OR SETTLEMENT", "THE DEFENCE OR",
        "CUSTOMER ID", "CUSTOMER NO", "POLICY NUMBER", "POLICY NO",
        "PLAN NAME", "CHASSIS", "CHASSIS NO", "ENGINE NO", "ENGINE NUMBER",
        "REGISTRATION NO", "REG NO", "REGN NO", "MAKE", "MODEL"
    ]

    DISALLOWED_EXACT_NAMES = {
        "PERSON", "CUSTOMER", "CUSTOMER ID", "CUSTOMER NO", "CUSTOMER NUMBER",
        "POLICYHOLDER", "POLICYHOLDER NAME", "POLICY HOLDER", "POLICY HOLDER NAME",
        "NAME", "NAME OF INSURED PERSON", "NAME OF INSURED", "NAME OF THE INSURED",
        "NAME OF PROPOSER", "INSURED PERSON", "INSURED PERSON(S)", "INSURED PERSONS",
        "SELF", "SPOUSE", "DAUGHTER", "SON", "FATHER", "MOTHER", "RELATIONSHIP", "RELATION",
        "DATE OF BIRTH", "DOB", "GENDER", "MALE", "FEMALE", "MEMBER", "MEMBER ID",
        "POLICY NUMBER", "POLICY NO", "PLAN NAME", "PRODUCT NAME", "ADDRESS", "PERMANENT ADDRESS",
        "CHASSIS", "CHASSIS NO", "CHASSIS NUMBER", "ENGINE NO", "ENGINE NUMBER",
        "REGISTRATION NO", "REGISTRATION NUMBER", "REG NO", "VEHICLE NO", "MAKE", "MODEL",
        "PERIOD OF INSURANCE", "AGE"
    }

    def resolve(
        self,
        full_text: str,
        page_texts: Dict[int, str],
        layout: Optional[Any] = None,
        tables: Optional[List[Dict[str, Any]]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[ResolvedFieldCandidate]:
        candidates: List[ResolvedFieldCandidate] = []

        doc_type = context.get("doc_type") if context else None
        is_health = (doc_type == DocumentType.HEALTH_INSURANCE)

        # -------------------------------------------------------------
        # For HEALTH_INSURANCE: Priority 1 & 2 (Explicit Labels)
        # -------------------------------------------------------------
        if is_health:
            # 1. Explicit "Policyholder Name"
            ph_cand = self._extract_explicit_labeled_name(
                target_label_keywords=["policyholder name", "policy holder name"],
                page_texts=page_texts,
                tables=tables,
                layout=layout
            )
            if ph_cand:
                return ph_cand

            # 2. Explicit "Customer Name"
            cust_cand = self._extract_explicit_labeled_name(
                target_label_keywords=["customer name", "name of customer"],
                page_texts=page_texts,
                tables=tables,
                layout=layout
            )
            if cust_cand:
                return cust_cand

        # -------------------------------------------------------------
        # STEP 1: Table Header Block Extraction (Highest Priority)
        # -------------------------------------------------------------
        if tables:
            t_prop = find_proposer_in_header_tables(tables)
            if t_prop:
                cleaned_name, cust_id = self._clean_and_split_name(t_prop["value"])
                if cleaned_name and self._is_valid_customer_name(cleaned_name):
                    return ResolvedFieldCandidate(
                        field_name=self.field_name,
                        value=cleaned_name,
                        exact_label=t_prop["exact_label"],
                        raw_evidence=t_prop["evidence"],
                        page=t_prop["page"],
                        confidence=0.99,
                        method=t_prop["method"],
                        is_verified=True,
                        metadata={"customer_id": cust_id} if cust_id else {}
                    )

        # -------------------------------------------------------------
        # STEP 2: Spatial Layout Extraction on Page 1 (Top Half: y < 350)
        # -------------------------------------------------------------
        if layout and hasattr(layout, "extract_value_for_label"):
            spatial_cands = layout.extract_value_for_label(
                label_aliases=self.aliases,
                value_pattern=r"[A-Za-z\.\s]{3,60}",
                max_dist_right=350.0,
                max_dist_below=35.0
            )
            for sc in spatial_cands:
                if sc["page"] > 2:
                    continue
                raw_cand = sc["value"]
                cleaned_name, cust_id = self._clean_and_split_name(raw_cand)
                if cleaned_name and self._is_valid_customer_name(cleaned_name):
                    candidates.append(ResolvedFieldCandidate(
                        field_name=self.field_name,
                        value=cleaned_name,
                        exact_label=sc.get("raw_text", ""),
                        raw_evidence=sc["raw_text"],
                        page=sc["page"],
                        bbox=sc.get("bbox"),
                        confidence=0.96,
                        method=sc.get("method", "spatial"),
                        is_verified=True,
                        metadata={"customer_id": cust_id} if cust_id else {}
                    ))

        # Check Page 1 line boxes directly above 'Address:' or relation prefixes ('S O', 'S/O', 'D/O', etc.) in layout
        if layout and hasattr(layout, "pages_lines"):
            p1_lines = layout.pages_lines.get(1, [])
            for idx, line in enumerate(p1_lines):
                line_lower = line.text.lower().strip()
                is_addr_marker = any(
                    marker in line_lower for marker in [
                        "address", "pincode", "pin code", "s o ", "s/o ", "d/o ", "w/o ", "c/o ",
                        "son of", "daughter of", "wife of", "care of"
                    ]
                ) or bool(re.match(r"^(?:s\s*[/o]|d\s*[/o]|w\s*[/o]|c\s*[/o])\b", line_lower))

                if is_addr_marker and idx > 0:
                    prev_line = p1_lines[idx - 1]
                    prev_text = " ".join(prev_line.text.strip().split())
                    words = prev_text.split()
                    if 2 <= len(words) <= 5 and all(w.replace(".", "").isalpha() for w in words):
                        if self._is_valid_customer_name(prev_text):
                            candidates.append(ResolvedFieldCandidate(
                                field_name=self.field_name,
                                value=prev_text,
                                exact_label="Insured / Proposer (Header Identity Line above Address)",
                                raw_evidence=f"{prev_line.text} [ABOVE] {line.text}",
                                page=1,
                                bbox=prev_line.bbox,
                                confidence=0.96,
                                method="header_identity_block",
                                is_verified=True
                            ))

        # Check block structures on Page 1
        if layout and hasattr(layout, "pages_blocks"):
            p1_blocks = layout.pages_blocks.get(1, [])
            for b in p1_blocks:
                b_lines = [l.strip() for l in b.text.split("\n") if l.strip()]
                if len(b_lines) >= 2:
                    first_line = b_lines[0]
                    first_words = first_line.split()
                    second_line_lower = b_lines[1].lower()
                    has_rel_or_addr = any(
                        m in second_line_lower for m in [
                            "s o ", "s/o ", "d/o ", "w/o ", "c/o ", "address", "pincode", "rajkot", "gujarat", "mumbai", "delhi"
                        ]
                    ) or bool(re.match(r"^(?:s\s*[/o]|d\s*[/o]|w\s*[/o]|c\s*[/o])\b", second_line_lower))

                    if has_rel_or_addr and 2 <= len(first_words) <= 5 and all(w.replace(".", "").isalpha() for w in first_words):
                        if self._is_valid_customer_name(first_line):
                            candidates.append(ResolvedFieldCandidate(
                                field_name=self.field_name,
                                value=first_line,
                                exact_label="Insured Identity Block",
                                raw_evidence=b.text[:120],
                                page=1,
                                bbox=b.bbox,
                                confidence=0.96,
                                method="header_identity_block",
                                is_verified=True
                            ))

        # -------------------------------------------------------------
        # STEP 3: Fallback Text / Regex Extraction (Pages 1-2 only)
        # -------------------------------------------------------------
        for pno in sorted(page_texts.keys())[:2]:
            text = page_texts[pno]
            for alias in self.aliases:
                escaped = re.escape(alias)
                pattern = rf"(?i)\b{escaped}\b[\s\.:/#=\-]*\n?\s*([^\n\r,;]{{3,80}})"
                for m in re.finditer(pattern, text):
                    raw_line = m.group(1).strip()
                    cleaned_name, cust_id = self._clean_and_split_name(raw_line)
                    if cleaned_name and self._is_valid_customer_name(cleaned_name):
                        candidates.append(ResolvedFieldCandidate(
                            field_name=self.field_name,
                            value=cleaned_name,
                            exact_label=alias,
                            raw_evidence=m.group(0).strip(),
                            page=pno,
                            confidence=0.92,
                            method="regex_alias",
                            is_verified=True,
                            metadata={"customer_id": cust_id} if cust_id else {}
                        ))

            # Standalone honorific pattern on first 2500 characters of Page 1
            if pno == 1:
                m_hon = re.search(r"\b(Mr\.?|Mrs\.?|Ms\.?|Dr\.?|M/s\.?)[ \t]+([A-Za-z \t]{3,40})", text[:2500])
                if m_hon:
                    full_hon_name = f"{m_hon.group(1)} {m_hon.group(2)}".strip()
                    if self._is_valid_customer_name(full_hon_name):
                        candidates.append(ResolvedFieldCandidate(
                            field_name=self.field_name,
                            value=full_hon_name,
                            exact_label="Honorific Prefix",
                            raw_evidence=m_hon.group(0).strip(),
                            page=pno,
                            confidence=0.88,
                            method="honorific_pattern",
                            is_verified=True
                        ))

        if candidates:
            # Sort by highest confidence, lowest page
            candidates.sort(key=lambda c: (c.confidence, -c.page), reverse=True)
            return candidates[0]

        return None

    def _clean_and_split_name(self, raw: str) -> Tuple[str, Optional[str]]:
        """
        Splits out adjacent customer IDs or slashes (e.g. 'MR MEET KORAT / 3654789/5' -> 'MR MEET KORAT', '3654789/5').
        """
        text = " ".join(raw.strip().split())
        # Remove trailing label words if line wrapped
        text = re.sub(r"(?i)\s+(?:Address|Mobile|Phone|Period|Policy|Make|Model|Pin\s*Code).*$", "", text)

        cust_id = None
        # Check for slash separator with ID
        if "/" in text:
            parts = [p.strip() for p in text.split("/") if p.strip()]
            if len(parts) >= 2:
                if any(char.isdigit() for char in parts[1]):
                    text = parts[0]
                    cust_id = parts[1]
                elif any(char.isdigit() for char in parts[0]):
                    text = parts[1]
                    cust_id = parts[0]

        # Clean trailing symbols
        text = re.sub(r"^[^\w]+|[^\w\.\)]+$", "", text).strip()
        return text, cust_id

    def _is_valid_customer_name(self, name: str) -> bool:
        """Strict negative validation: rejects legal terms, policy boilerplate, numbers, single letters, or label tokens."""
        if not name or len(name) < 3 or len(name) > 65:
            return False
        if "|" in name or "\n" in name:
            return False
        # Must contain at least two alphabetic characters
        if len(re.sub(r"[^a-zA-Z]", "", name)) < 2:
            return False
        upper_name = " ".join(name.upper().split()).rstrip(" :.-#")
        if upper_name in self.DISALLOWED_EXACT_NAMES:
            return False
        for dis in self.DISALLOWED_EXACT_NAMES:
            if upper_name == dis or upper_name.startswith(dis + ":") or upper_name.startswith(dis + " "):
                if len(upper_name.replace(dis, "").strip(" :.-#")) < 3:
                    return False

        words = set(re.findall(r"[A-Za-z]+", upper_name))
        forbidden_words = {
            "PERSON", "RELATIONSHIP", "RELATION", "DOB", "GENDER",
            "MEMBER", "AGE", "POLICYHOLDER", "PROPOSER", "INSURED"
        }
        if words.issubset(forbidden_words):
            return False
        if "PERSON" in words or "RELATIONSHIP" in words:
            return False

        # Hard rejection of policy and legal wording
        for dis in self.DISALLOWED_TERMS:
            if dis in ("LIMITED", "LTD", "COMPANY", "GENERAL"):
                if any(ins in upper_name for ins in ["INSURANCE", "ASSURANCE", "BROKER", "AGENCY"]):
                    return False
                continue
            if dis == "PRIVATE":
                if any(k in upper_name for k in ["PRIVATE CAR", "PRIVATE VEHICLE", "PRIVATE USE", "PRIVATE COMMERCIAL"]):
                    return False
                continue
            if dis in upper_name:
                return False
        # Reject if contains digits
        if any(char.isdigit() for char in name):
            return False
        return True

    def _extract_explicit_labeled_name(
        self,
        target_label_keywords: List[str],
        page_texts: Dict[int, str],
        tables: Optional[List[Dict[str, Any]]] = None,
        layout: Optional[Any] = None
    ) -> Optional[ResolvedFieldCandidate]:
        """
        Extracts policyholder or customer name associated with an explicit label.
        Checks:
        1. Structured tables: cell to right or cell directly below
        2. Coordinate layout: spatial alignment
        3. Plaintext: regex with optional newline between label and name
        """
        from validation.field_rules import clean_customer_name

        # 1. Check Tables (Pages 1-2)
        if tables:
            for tbl in tables:
                pno = tbl.get("page_number", 1)
                if pno > 2:
                    continue
                matrix = tbl.get("raw_matrix", [])
                for r_idx, row in enumerate(matrix):
                    for c_idx, cell in enumerate(row):
                        if not cell:
                            continue
                        c_str = " ".join(str(cell).strip().split())
                        c_lower = c_str.lower()
                        if any(kw in c_lower for kw in target_label_keywords):
                            # Case A: Value in same cell (e.g. "Policyholder Name: Mr DEVABHAI ...")
                            for kw in target_label_keywords:
                                m_in = re.search(rf"(?i)\b{re.escape(kw)}\b[\s\.:/-]*\n?\s*([A-Za-z\.\t ]{{3,60}})", c_str)
                                if m_in:
                                    cand_val = m_in.group(1).strip()
                                    cleaned_name, cust_id = self._clean_and_split_name(cand_val)
                                    norm_name = clean_customer_name(cleaned_name) or cleaned_name
                                    if norm_name and self._is_valid_customer_name(norm_name):
                                        return ResolvedFieldCandidate(
                                            field_name=self.field_name,
                                            value=norm_name,
                                            exact_label=c_str,
                                            raw_evidence=f"{c_str} -> {norm_name}",
                                            page=pno,
                                            confidence=0.99,
                                            method="table_explicit_inline",
                                            is_verified=True,
                                            metadata={"customer_id": cust_id} if cust_id else {}
                                        )

                            # Case B: Value in cell to the right
                            for c_next in range(c_idx + 1, min(c_idx + 4, len(row))):
                                right_cell = row[c_next]
                                if right_cell and str(right_cell).strip():
                                    cand_val = " ".join(str(right_cell).strip().split())
                                    cleaned_name, cust_id = self._clean_and_split_name(cand_val)
                                    norm_name = clean_customer_name(cleaned_name) or cleaned_name
                                    if norm_name and self._is_valid_customer_name(norm_name):
                                        return ResolvedFieldCandidate(
                                            field_name=self.field_name,
                                            value=norm_name,
                                            exact_label=c_str,
                                            raw_evidence=f"{c_str} -> {norm_name}",
                                            page=pno,
                                            confidence=0.99,
                                            method="table_cell_right",
                                            is_verified=True,
                                            metadata={"customer_id": cust_id} if cust_id else {}
                                        )
                                    break

                            # Case C: Value in cell directly below
                            if r_idx + 1 < len(matrix) and c_idx < len(matrix[r_idx + 1]):
                                down_cell = matrix[r_idx + 1][c_idx]
                                if down_cell and str(down_cell).strip():
                                    cand_val = " ".join(str(down_cell).strip().split())
                                    cleaned_name, cust_id = self._clean_and_split_name(cand_val)
                                    norm_name = clean_customer_name(cleaned_name) or cleaned_name
                                    if norm_name and self._is_valid_customer_name(norm_name):
                                        return ResolvedFieldCandidate(
                                            field_name=self.field_name,
                                            value=norm_name,
                                            exact_label=c_str,
                                            raw_evidence=f"{c_str} [BELOW] -> {norm_name}",
                                            page=pno,
                                            confidence=0.98,
                                            method="table_cell_down",
                                            is_verified=True,
                                            metadata={"customer_id": cust_id} if cust_id else {}
                                        )

        # 2. Check Layout (Pages 1-2)
        if layout and hasattr(layout, "extract_value_for_label"):
            spatial_cands = layout.extract_value_for_label(
                label_aliases=target_label_keywords,
                value_pattern=r"[A-Za-z\.\t ]{3,60}",
                max_dist_right=350.0,
                max_dist_below=40.0
            )
            for sc in spatial_cands:
                if sc.get("page", 1) <= 2:
                    cleaned_name, cust_id = self._clean_and_split_name(sc["value"])
                    norm_name = clean_customer_name(cleaned_name) or cleaned_name
                    if norm_name and self._is_valid_customer_name(norm_name):
                        return ResolvedFieldCandidate(
                            field_name=self.field_name,
                            value=norm_name,
                            exact_label=sc.get("raw_text", ""),
                            raw_evidence=sc.get("raw_text", ""),
                            page=sc.get("page", 1),
                            bbox=sc.get("bbox"),
                            confidence=0.98,
                            method=sc.get("method", "spatial_explicit"),
                            is_verified=True,
                            metadata={"customer_id": cust_id} if cust_id else {}
                        )

        # 3. Check Page Texts (Pages 1-2)
        for pno in sorted(page_texts.keys())[:2]:
            text = page_texts[pno]
            for kw in target_label_keywords:
                pattern = rf"(?i)\b{re.escape(kw)}\b[\s\.:/#=\-]*\n?\s*([A-Za-z\.\t ]{{3,60}})"
                for m in re.finditer(pattern, text):
                    raw_line = m.group(1).strip()
                    cleaned_name, cust_id = self._clean_and_split_name(raw_line)
                    norm_name = clean_customer_name(cleaned_name) or cleaned_name
                    if norm_name and self._is_valid_customer_name(norm_name):
                        return ResolvedFieldCandidate(
                            field_name=self.field_name,
                            value=norm_name,
                            exact_label=kw,
                            raw_evidence=m.group(0).strip(),
                            page=pno,
                            confidence=0.97,
                            method="regex_explicit_label",
                            is_verified=True,
                            metadata={"customer_id": cust_id} if cust_id else {}
                        )
        return None
