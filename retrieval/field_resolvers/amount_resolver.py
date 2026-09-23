"""
Amount Resolver: Precision Semantic Financial Classifier and Arithmetic Validation Engine.
Distinguishes IDV, CNG IDV, NCB, Basic OD Premium, Basic TP Premium, Net Premium, Taxes/GST, and Total Premium.
Ranks candidates across tables, spatial coordinates, layout models, and text windows.
Preserves exact source page and evidence text for every extracted value.
"""

import re
from typing import Dict, Any, List, Optional, Tuple
from retrieval.field_resolvers.base_resolver import BaseFieldResolver, ResolvedFieldCandidate
from validation.financial import parse_currency_amount
from validation.field_rules import clean_ncb
from utils.logger import get_logger

logger = get_logger("amount_resolver")


class AmountResolver(BaseFieldResolver):
    field_name: str = "amount"

    FINANCIAL_ALIASES = {
        "idv": [
            "total idv",
            "insured's declared value (idv)",
            "insured declared value (idv)",
            "insured's declared value",
            "insured declared value",
            "vehicle idv",
            "idv in rs.",
            "idv in rs",
            "idv (in rs)",
            "total value",
            "idv",
        ],
        "cng_idv": [
            "cng/lpg kit idv",
            "cng kit idv",
            "lpg kit idv",
            "bi-fuel kit idv",
            "bi fuel kit idv",
            "cng/lpg kit",
            "cng / lpg kit",
            "bi-fuel kit",
            "bi fuel kit",
            "cng idv",
            "lpg idv",
            "cng value",
            "cng/lpg value",
        ],
        "ncb": [
            "no claim bonus (ncb)",
            "no claim bonus",
            "no claim discount",
            "ncb discount",
            "ncb percentage",
            "ncb %",
            "ncb",
        ],
        "basic_od_premium": [
            "basic own damage premium",
            "basic own damage",
            "own damage premium(a)",
            "own damage premium (a)",
            "own damage premium",
            "basic od premium",
            "basic od",
            "od premium",
            "a. own damage premium(rs.)",
            "a. own damage premium",
            "basic premium",
        ],
        "basic_tp_premium": [
            "basic third-party liability",
            "basic third party liability",
            "basic third-party premium",
            "basic third party premium",
            "third-party liability",
            "third party liability",
            "third-party premium",
            "third party premium",
            "liability premium(b)",
            "liability premium (b)",
            "basic tp premium",
            "basic tp",
            "tp premium",
            "b. third party premium(rs.)",
            "b. third party premium",
            "basic premium",
        ],
        "net_premium": [
            "total package premium (a+b)",
            "total package premium",
            "total net premium",
            "net liability premium (b)",
            "net own damage premium (a)",
            "net premium rs.",
            "net premium",
            "premium before gst",
            "premium before tax",
            "gross premium taxable value",
            "gross premium taxable",
            "taxable value",
        ],
        "tax": [
            "integrated tax 18%",
            "integrated tax",
            "goods & services tax",
            "goods and services tax",
            "goods and service tax",
            "total gst",
            "gst (18%)",
            "gst 18%",
            "igst 18%",
            "cgst + sgst",
            "total tax",
            "taxes (gst 18%)",
            "taxes",
            "cgst",
            "sgst",
            "igst",
            "utgst",
            "gst",
            "tax",
            "electricity duty",
        ],
        "total_premium": [
            "total premium payable",
            "total amount payable",
            "net amount payable",
            "final payable amount",
            "net premium total invoice value",
            "premium paid(total invoice value) rs.",
            "premium paid(total invoice value)",
            "total invoice value",
            "final premium",
            "total premium paid",
            "total premium",
            "gross premium",
            "total bill amount",
            "total amount",
            "amount payable",
            "premium payable",
            "quote estimate amount",
            "total bill",
        ],
        "sum_insured": [
            "total sum insured",
            "basic sum insured",
            "sum insured",
            "sum assured",
            "limit of indemnity",
        ],
        "collection_amount": [
            "premium collection details",
            "collected rs",
            "received rs",
            "amount collected",
        ]
    }

    NEGATIVE_LABELS = {
        "idv": ["premium", "tax", "gst", "cng", "lpg", "bi-fuel", "trailer", "accessories", "date", "policy", "reg"],
        "cng_idv": ["total idv", "vehicle idv", "premium", "tax", "gst", "date", "policy"],
        "ncb": ["date", "policy", "premium payable", "total premium", "invoice", "protection", "add-on", "addon", "cover", "upto", "up to", "limits"],
        "basic_od_premium": ["third party", "tp liability", "liability premium", "net premium", "total premium", "tax", "gst"],
        "basic_tp_premium": ["own damage", "od premium", "net premium", "total premium", "tax", "gst"],
        "net_premium": ["total premium", "total amount payable", "final premium", "final payable", "total invoice value", "grand total", "basic own damage", "basic third party"],
        "tax": ["total premium", "total amount", "grand total", "total payable", "final payable", "amount payable", "final premium", "total invoice", "net premium", "basic premium", "sum insured", "idv", "policy"],
        "total_premium": ["net premium", "basic premium", "basic own damage", "basic tp", "third party", "taxable value", "premium before tax", "gst", "cgst", "sgst", "igst", "tax", "total tax", "total gst"],
    }

    def __init__(self):
        super().__init__()
        self.debug_log: List[Dict[str, Any]] = []

    def resolve(
        self,
        full_text: str,
        page_texts: Dict[int, str],
        layout: Optional[Any] = None,
        tables: Optional[List[Dict[str, Any]]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Optional[ResolvedFieldCandidate]]:
        """
        Extracts and semantically categorizes all financial fields with multi-candidate ranking.
        Supports tables, coordinate layout, multi-line windows, and arithmetic cross-validation.
        """
        self.debug_log = []
        target_categories = [
            "idv", "cng_idv", "ncb", "basic_od_premium", "basic_tp_premium",
            "net_premium", "tax", "total_premium", "sum_insured", "collection_amount"
        ]

        results: Dict[str, Optional[ResolvedFieldCandidate]] = {cat: None for cat in target_categories}
        all_candidates: Dict[str, List[ResolvedFieldCandidate]] = {cat: [] for cat in target_categories}

        # 1. Structured Table Extraction (Tier 4)
        table_cands = self._extract_from_tables(tables)
        for cat, c_list in table_cands.items():
            all_candidates[cat].extend(c_list)

        # 2. Spatial Layout Extraction (Tier 4 / Tier 3)
        if layout:
            layout_cands = self._extract_from_layout(layout)
            for cat, c_list in layout_cands.items():
                all_candidates[cat].extend(c_list)

        # 3. Text Window & Regex Extraction (Tier 2)
        text_cands = self._extract_from_text_windows(page_texts)
        for cat, c_list in text_cands.items():
            all_candidates[cat].extend(c_list)

        # 4. Standalone TOTAL check if total_premium is still unpopulated
        if not all_candidates["total_premium"]:
            standalone_cand = self._extract_standalone_total(page_texts)
            if standalone_cand:
                all_candidates["total_premium"].append(standalone_cand)

        # 5. Score and select winning candidate for each category
        for cat in target_categories:
            cands = all_candidates[cat]
            winning_cand = self._select_winning_candidate(cat, cands)
            results[cat] = winning_cand

            # Record internal diagnostic information (Requirement 13)
            self._log_diagnostics(cat, cands, winning_cand)

        # 6. Arithmetic Cross-Validation and Consistency Reconciliation
        self._cross_validate_financials(results, all_candidates)

        # Check if document is specifically a liability-only / third-party policy
        is_liability_only = False
        all_text_check = full_text.lower()
        if any(pt in all_text_check for pt in ["liability only", "third party only", "act only", "liability policy"]):
            is_liability_only = True
        elif page_texts:
            for pt in list(page_texts.values())[:2]:
                if any(k in pt.lower() for k in ["liability only", "third party only", "act only", "liability policy"]):
                    is_liability_only = True
                    break

        if is_liability_only:
            # Liability-only policies legitimately do not have IDV or Own Damage premium
            results["idv"] = None
            results["total_idv"] = None
            results["basic_od_premium"] = None
            results["own_damage_premium"] = None
            results["od_premium"] = None

        # Populate total_amount alias for backwards compatibility
        if results.get("total_premium") and not results.get("total_amount"):
            results["total_amount"] = results["total_premium"]

        # Populate basic_premium alias for backwards compatibility
        if results.get("basic_od_premium") and not results.get("basic_premium"):
            results["basic_premium"] = results["basic_od_premium"]

        return results

    def _extract_from_tables(self, tables: Optional[List[Dict[str, Any]]]) -> Dict[str, List[ResolvedFieldCandidate]]:
        """Extracts financial candidates from structured table rows, grid cells, and composite cells."""
        table_cands: Dict[str, List[ResolvedFieldCandidate]] = {k: [] for k in self.FINANCIAL_ALIASES}
        if not tables:
            return table_cands

        from ingestion.table_extractor import find_table_field

        # Priority A: Standard downward/rightward grid search
        grid_targets = {
            "total_premium": [
                "premium paid(total invoice value) rs.", "premium paid(total invoice value)",
                "net premium total invoice value", "total amount payable", "total premium",
                "final payable amount", "grand total", "total bill amount"
            ],
            "net_premium": [
                "net premium rs.", "net premium", "total net premium", "gross premium taxable value", "gross premium taxable"
            ],
            "idv": [
                "idv in rs.", "idv in rs", "total idv", "vehicle idv", "insured declared value"
            ],
            "cng_idv": [
                "bi-fuel kit", "bi fuel kit", "cng/lpg kit", "cng kit"
            ],
            "tax": [
                "total gst", "igst", "cgst", "sgst", "gst", "taxes"
            ],
            "basic_od_premium": [
                "basic own damage premium", "basic own damage", "basic od premium", "a. own damage premium"
            ],
            "basic_tp_premium": [
                "basic third party liability", "basic third party premium", "basic tp premium", "b. third party premium"
            ]
        }

        for cat, aliases in grid_targets.items():
            t_res = find_table_field(
                tables,
                aliases,
                value_pattern=r"^\d+(?:\.\d{2})?$",
                max_down=4,
                max_right=4,
                max_page=2
            )
            if t_res:
                num = parse_currency_amount(t_res["value"])
                if num is not None and (num > 0 or (cat == "cng_idv" and num == 0)):
                    table_cands[cat].append(ResolvedFieldCandidate(
                        field_name=cat,
                        value=str(num),
                        exact_label=t_res.get("exact_label", ""),
                        raw_evidence=t_res.get("evidence", ""),
                        page=t_res.get("page", 1),
                        confidence=0.99,
                        method=t_res.get("method", "table_cell_down"),
                        is_verified=True
                    ))

        # Priority B: Detailed row-by-row table inspection
        for tbl in tables:
            pno = tbl.get("page_number", 1)
            t_idx = tbl.get("table_index", 0)
            rows = tbl.get("rows", [])
            num_rows = len(rows)

            # B.1 Consecutive Row Pairing (e.g. Row i = Headers, Row i+1 = Values)
            for i in range(num_rows - 1):
                r_head = rows[i]
                r_val = rows[i + 1]
                if len(r_head) == len(r_val):
                    for c_idx in range(len(r_head)):
                        h_cell = str(r_head[c_idx]).strip().lower()
                        v_cell = str(r_val[c_idx]).strip()
                        if not h_cell or not v_cell:
                            continue

                        # Crucial: v_cell must be purely numeric (no endorsement text or words)
                        if not re.match(r"^[\sRs₹INR\d,\.\-\+]+$", v_cell):
                            continue

                        for cat, aliases in self.FINANCIAL_ALIASES.items():
                            if any(alias in h_cell for alias in aliases):
                                if not any(neg in h_cell for neg in self.NEGATIVE_LABELS.get(cat, [])):
                                    num = parse_currency_amount(v_cell)
                                    if num is not None and (num > 0 or (cat == "cng_idv" and num == 0)):
                                        conf = 0.98
                                        # Boost specific vehicle IDV
                                        if cat == "idv" and any(k in h_cell for k in ["vehicle", "idv in rs", "insured declared"]):
                                            conf = 0.99
                                        elif cat == "idv" and "total value" in h_cell:
                                            conf = 0.95

                                        table_cands[cat].append(ResolvedFieldCandidate(
                                            field_name=cat,
                                            value=str(num),
                                            exact_label=r_head[c_idx].strip(),
                                            raw_evidence=f"Table {t_idx} Col {c_idx}: {r_head[c_idx].strip()} -> {v_cell}",
                                            page=pno,
                                            confidence=conf,
                                            method="table_column_header_row_pair",
                                            is_verified=True,
                                            metadata={"table_index": t_idx, "col_idx": c_idx}
                                        ))

            # B.2 Row-wise label-value pair extraction (e.g. [Label, Value] or [Label1, Val1, Label2, Val2])
            for r_idx, r in enumerate(rows):
                for c_idx, cell_str in enumerate(r):
                    cell_label = str(cell_str).strip().lower()
                    if not cell_label or re.match(r"^[\sRs₹INR\d,\.\-\+]+$", cell_label):
                        continue

                    for cat, aliases in self.FINANCIAL_ALIASES.items():
                        # Determine category applicability based on column position for split OD/TP tables
                        if cat == "basic_tp_premium" and c_idx < 3 and not any(tp in cell_label for tp in ["third party", "tp liability", "liability"]):
                            continue
                        if cat == "basic_od_premium" and (c_idx >= 3 or any(tp in cell_label for tp in ["third party", "tp liability", "liability"])):
                            continue

                        if any(alias in cell_label for alias in aliases):
                            if not any(neg in cell_label for neg in self.NEGATIVE_LABELS.get(cat, [])):
                                # Find nearest numeric value cell to the right in the same row
                                for right_c in range(c_idx + 1, min(c_idx + 6, len(r))):
                                    cell_val = str(r[right_c]).strip()
                                    if cell_val and re.match(r"^[\sRs₹INR\d,\.\-\+]+$", cell_val):
                                        num = parse_currency_amount(cell_val)
                                        if num is not None and (num > 0 or (cat == "cng_idv" and num == 0)):
                                            table_cands[cat].append(ResolvedFieldCandidate(
                                                field_name=cat,
                                                value=str(num),
                                                exact_label=r[c_idx].strip(),
                                                raw_evidence=f"Table {t_idx} row {r_idx}: {r[c_idx].strip()} -> {cell_val}",
                                                page=pno,
                                                confidence=0.98,
                                                method="table_row_pair",
                                                is_verified=True,
                                                metadata={"table_index": t_idx, "row_idx": r_idx, "col_idx": c_idx}
                                            ))
                                            break

            # B.3 Sub-item Space-Separated Matching (e.g. Case C row 8)
            for r_idx, r in enumerate(rows):
                for c_idx, cell_str in enumerate(r):
                    cell_text = str(cell_str).strip()
                    if not cell_text:
                        continue

                    # Look for Basic OD Premium in sub-item row (OD columns on the left, c_idx < 10)
                    if "basic premium" in cell_text.lower() and c_idx < 10 and not any(tp in cell_text.lower() for tp in ["third party", "tp liability", "liability"]):
                        for right_c in range(c_idx + 1, min(c_idx + 8, len(r))):
                            val_str = str(r[right_c]).strip()
                            parts = val_str.split()
                            if parts and all(re.match(r"^[-+]?\d+(?:\.\d+)?$", p) for p in parts):
                                num = parse_currency_amount(parts[0])
                                if num and num > 0:
                                    table_cands["basic_od_premium"].append(ResolvedFieldCandidate(
                                        field_name="basic_od_premium",
                                        value=str(num),
                                        exact_label="Basic Premium (Own Damage)",
                                        raw_evidence=f"Table {t_idx} row {r_idx}: {cell_text} -> {val_str}",
                                        page=pno,
                                        confidence=0.98,
                                        method="table_subitem_first_value",
                                        is_verified=True
                                    ))
                                break

                    # Look for Basic TP Premium in sub-item row (TP columns on the right, c_idx >= 8 or text has third party)
                    if "basic premium" in cell_text.lower() and (c_idx >= 8 or any(tp in cell_text.lower() for tp in ["third party", "tp liability", "liability"])):
                        for right_c in range(c_idx + 1, len(r)):
                            val_str = str(r[right_c]).strip()
                            parts = val_str.split()
                            if parts and all(re.match(r"^[-+]?\d+(?:\.\d+)?$", p) for p in parts):
                                num = parse_currency_amount(parts[0])
                                if num and num > 0:
                                    table_cands["basic_tp_premium"].append(ResolvedFieldCandidate(
                                        field_name="basic_tp_premium",
                                        value=str(num),
                                        exact_label="Basic Premium (Third Party)",
                                        raw_evidence=f"Table {t_idx} row {r_idx}: {cell_text} -> {val_str}",
                                        page=pno,
                                        confidence=0.98,
                                        method="table_subitem_first_value",
                                        is_verified=True
                                    ))
                                break

                    # Look for Tax Amount in Multi-Row Tax Breakdown (e.g. Case C rows 16-18)
                    if any(tx in cell_text.upper() for tx in ["IGST", "CGST", "SGST", "TOTAL GST"]):
                        for down_r in range(r_idx + 1, min(r_idx + 5, num_rows)):
                            row_tag = str(rows[down_r][0]).strip().lower()
                            if "amount" in row_tag and c_idx < len(rows[down_r]):
                                amt_str = str(rows[down_r][c_idx]).strip()
                                num = parse_currency_amount(amt_str)
                                if num and num > 0:
                                    table_cands["tax"].append(ResolvedFieldCandidate(
                                        field_name="tax",
                                        value=str(num),
                                        exact_label=f"{cell_text} Amount",
                                        raw_evidence=f"Table {t_idx} col {c_idx}: {cell_text} (Amount) = {amt_str}",
                                        page=pno,
                                        confidence=0.99,
                                        method="table_tax_amount_row",
                                        is_verified=True
                                    ))

                    # B.3 Embedded Key-Values inside composite cells (e.g. Case A Table 1 row 3)
                    for cat, aliases in self.FINANCIAL_ALIASES.items():
                        for alias in aliases:
                            pattern = rf"(?i)\b{re.escape(alias)}\b[\s\.:/#=\-]*\s*(?:Rs\.?|INR|₹)?\s*([\d,]+(?:\.\d{{2}})?)"
                            m = re.search(pattern, cell_text)
                            if m:
                                raw_val = m.group(1).strip()
                                num = parse_currency_amount(raw_val)
                                if num is not None and (num > 0 or (cat == "cng_idv" and num == 0)):
                                    if not any(neg in alias for neg in self.NEGATIVE_LABELS.get(cat, [])):
                                        table_cands[cat].append(ResolvedFieldCandidate(
                                            field_name=cat,
                                            value=str(num),
                                            exact_label=alias,
                                            raw_evidence=f"Table {t_idx} cell embedded: {m.group(0)}",
                                            page=pno,
                                            confidence=0.96,
                                            method="table_composite_cell_regex",
                                            is_verified=True
                                        ))

                    # B.4 NCB percentage or discount in table
                    if any(term in cell_text.lower() for term in ["no claim bonus", "no claim discount", "ncb"]):
                        if not any(neg in cell_text.lower() for neg in ["protection", "cover", "add-on", "addon"]):
                            # Look inside current cell or row for percentage
                            m_ncb = re.search(r"(\d{1,2}(?:\.\d{1,2})?)\s*%", cell_text)
                            if m_ncb:
                                ncb_val = clean_ncb(m_ncb.group(0).strip())
                                if ncb_val and self._is_valid_ncb(ncb_val):
                                    table_cands["ncb"].append(ResolvedFieldCandidate(
                                        field_name="ncb",
                                        value=ncb_val,
                                        exact_label="No Claim Bonus",
                                        raw_evidence=f"Table {t_idx}: {m_ncb.group(0)}",
                                        page=pno,
                                        confidence=0.97,
                                        method="table_ncb_cell",
                                        is_verified=True
                                    ))
                            # Check rightward cell in same row
                            for right_c in range(c_idx + 1, min(c_idx + 4, len(r))):
                                right_val = str(r[right_c]).strip()
                                m_r_pct = re.search(r"(\d{1,2}(?:\.\d{1,2})?)\s*%", right_val)
                                if m_r_pct:
                                    ncb_val = clean_ncb(m_r_pct.group(0).strip())
                                    if ncb_val and self._is_valid_ncb(ncb_val):
                                        table_cands["ncb"].append(ResolvedFieldCandidate(
                                            field_name="ncb",
                                            value=ncb_val,
                                            exact_label="No Claim Bonus",
                                            raw_evidence=f"Table {t_idx} col {right_c}: {cell_text} -> {right_val}",
                                            page=pno,
                                            confidence=0.96,
                                            method="table_ncb_rightward",
                                            is_verified=True
                                        ))
                                    break

        return table_cands

    def _extract_from_layout(self, layout: Any) -> Dict[str, List[ResolvedFieldCandidate]]:
        """
        Extracts financial candidates using coordinate-aware layout analysis:
        1. Horizontal alignment: Label on left, amount to the right on the same row band (y_mid delta <= 6.5 pt)
        2. Vertical column-alignment: Stacked column header with amount directly below (x_center delta <= 28 pt)
        """
        layout_cands: Dict[str, List[ResolvedFieldCandidate]] = {k: [] for k in self.FINANCIAL_ALIASES}
        if not hasattr(layout, "pages_lines") or not layout.pages_lines:
            return layout_cands

        for pno, lines in layout.pages_lines.items():
            tokens = getattr(layout, "pages_tokens", {}).get(pno, [])

            # --- Pattern 1: Vertical Column-Aligned Header Stacking (as in Case B) ---
            for line in lines:
                l_text_norm = re.sub(r"[-_]", " ", line.text.lower())
                has_financial_headers = any(k in l_text_norm for k in ["net premium", "gross premium", "cgst", "sgst", "igst", "final premium"])
                if has_financial_headers:
                    val_lines = [
                        other_l for other_l in lines
                        if other_l.y0 >= line.y1 - 2.0
                        and other_l.y0 <= line.y1 + 35.0
                        and any(char.isdigit() for char in other_l.text)
                    ]

                    line_toks = line.tokens if line.tokens else [t for t in tokens if t.line_no == line.line_no]
                    for vl in val_lines:
                        vl_toks = vl.tokens if vl.tokens else [t for t in tokens if t.line_no == vl.line_no]

                        # Group header tokens into logical multi-word labels
                        h_groups: List[Tuple[str, float, float]] = []  # (label, x0, x1)
                        curr_words: List[str] = []
                        curr_x0: Optional[float] = None
                        curr_x1: Optional[float] = None

                        for ht in line_toks:
                            ht_text = ht.text.strip()
                            if not ht_text:
                                continue
                            if curr_x0 is None:
                                curr_x0 = ht.x0
                            curr_words.append(ht_text)
                            curr_x1 = ht.x1

                            joined = re.sub(r"[-_]", " ", " ".join(curr_words).lower())
                            if any(joined == alias or joined.startswith(alias) for aliases in self.FINANCIAL_ALIASES.values() for alias in aliases):
                                h_groups.append((joined, curr_x0, curr_x1))
                                curr_words = []
                                curr_x0 = None
                                curr_x1 = None

                        if curr_words and curr_x0 is not None and curr_x1 is not None:
                            h_groups.append((re.sub(r"[-_]", " ", " ".join(curr_words).lower()), curr_x0, curr_x1))

                        # For each header group, find the numeric token directly below with minimum x distance
                        for h_lbl, h_x0, h_x1 in h_groups:
                            h_x_center = (h_x0 + h_x1) / 2.0

                            matched_cat = None
                            for cat, aliases in self.FINANCIAL_ALIASES.items():
                                if any(alias in h_lbl for alias in aliases):
                                    if not any(neg in h_lbl for neg in self.NEGATIVE_LABELS.get(cat, [])):
                                        matched_cat = cat
                                        break

                            if not matched_cat:
                                continue

                            best_tok = None
                            best_x_dist = 999.0
                            for vt in vl_toks:
                                num = parse_currency_amount(vt.text)
                                if num is not None and (num > 0 or (matched_cat in ("cng_idv", "tax") and num == 0)):
                                    vt_x_center = (vt.x0 + vt.x1) / 2.0
                                    x_dist = abs(vt_x_center - h_x_center)
                                    if x_dist < 28.0 and x_dist < best_x_dist:
                                        best_x_dist = x_dist
                                        best_tok = vt

                            if best_tok:
                                num_val = parse_currency_amount(best_tok.text)
                                layout_cands[matched_cat].append(ResolvedFieldCandidate(
                                    field_name=matched_cat,
                                    value=str(num_val),
                                    exact_label=h_lbl,
                                    raw_evidence=f"Layout Stacked: {h_lbl} (x={h_x0:.1f}) -> {best_tok.text} (x={best_tok.x0:.1f}, dist={best_x_dist:.1f})",
                                    page=pno,
                                    confidence=0.96,
                                    method="layout_vertical_column_stack",
                                    is_verified=True
                                ))

            # --- Pattern 2: Horizontal Alignment on Same Row Band ---
            # e.g. Line: "Net Premium" at x=47.2, y=477.7 -> Token: "714.00" at x=386.6, y=477.7
            # e.g. Line: "Final Premium" at x=47.2, y=520.5 -> Token: "842.52" at x=386.6, y=520.5
            # e.g. Line: "Basic Third-Party Liability" at x=47.2, y=443.2 -> Token: "714.00" at x=291.4
            for line in lines:
                l_text_clean = line.text.strip().lower()
                l_text_norm = re.sub(r"[-_]", " ", l_text_clean)

                for cat, aliases in self.FINANCIAL_ALIASES.items():
                    negs = self.NEGATIVE_LABELS.get(cat, [])
                    if any(neg in l_text_norm for neg in negs):
                        continue

                    matched_alias = None
                    for alias in aliases:
                        alias_norm = re.sub(r"[-_]", " ", alias.lower())
                        if re.search(rf"(?i)\b{re.escape(alias_norm)}\b", l_text_norm):
                            matched_alias = alias
                            break

                    if matched_alias:
                        line_y_mid = (line.y0 + line.y1) / 2.0
                        matching_toks = []

                        for tok in tokens:
                            if tok.x0 > line.x1 - 2.0 and tok.x0 < line.x1 + 450.0:
                                tok_y_mid = (tok.y0 + tok.y1) / 2.0
                                if abs(tok_y_mid - line_y_mid) <= 6.5:
                                    num = parse_currency_amount(tok.text)
                                    if num is not None and (num > 0 or (cat == "cng_idv" and num == 0)):
                                        if self._is_valid_monetary(tok.text, cat):
                                            matching_toks.append((tok, num))

                        if matching_toks:
                            matching_toks.sort(key=lambda t: t[0].x0)
                            best_tok, best_num = matching_toks[0]
                            layout_cands[cat].append(ResolvedFieldCandidate(
                                field_name=cat,
                                value=str(best_num),
                                exact_label=matched_alias,
                                raw_evidence=f"Layout Horizontal: {line.text} -> {best_tok.text}",
                                page=pno,
                                confidence=0.95,
                                method="layout_horizontal_row_band",
                                is_verified=True
                            ))

        return layout_cands

    def _extract_from_text_windows(self, page_texts: Dict[int, str]) -> Dict[str, List[ResolvedFieldCandidate]]:
        """Extracts financial candidates from plain-text lines and multi-line windows."""
        text_cands: Dict[str, List[ResolvedFieldCandidate]] = {k: [] for k in self.FINANCIAL_ALIASES}

        for pno, text in page_texts.items():
            lines = text.split("\n")
            num_lines = len(lines)

            for i in range(num_lines):
                line = lines[i].strip()
                line_lower = line.lower()
                line_norm = re.sub(r"[-_]", " ", line_lower)
                if not line:
                    continue

                for cat, aliases in self.FINANCIAL_ALIASES.items():
                    negs = self.NEGATIVE_LABELS.get(cat, [])
                    if any(neg in line_norm for neg in negs):
                        continue

                    for alias in aliases:
                        alias_norm = re.sub(r"[-_]", " ", alias.lower())
                        # Case 1: Inline match on same line
                        pattern = (
                            rf"(?i)\b{re.escape(alias_norm)}\b"
                            rf"(?:\s*(?:\([^)]*\)|@[0-9\.]+%?))?"
                            rf"[\s\.:/#=\-]*\s*(?:Rs\.?|INR|₹)?\s*"
                            rf"([\d,]+(?:\.\d{{2}})?)"
                        )
                        m = re.search(pattern, line_norm)
                        if m:
                            raw_num = m.group(1).strip()
                            if self._is_valid_monetary(raw_num, cat):
                                num_val = parse_currency_amount(raw_num)
                                if num_val is not None and (num_val > 0 or (cat == "cng_idv" and num_val == 0)):
                                    text_cands[cat].append(ResolvedFieldCandidate(
                                        field_name=cat,
                                        value=raw_num,
                                        exact_label=alias,
                                        raw_evidence=line,
                                        page=pno,
                                        confidence=0.92,
                                        method="text_inline_regex",
                                        is_verified=True
                                    ))

                        # Case 2: Multi-line window
                        if re.search(rf"(?i)\b{re.escape(alias_norm)}\b", line_norm):
                            for offset in (1, 2, 3):
                                if i + offset < num_lines:
                                    next_line = lines[i + offset].strip()
                                    m_next = re.search(r"^(?:Rs\.?|INR|₹)?\s*([\d,]+(?:\.\d{2})?)$", next_line)
                                    if m_next:
                                        raw_num = m_next.group(1).strip()
                                        if self._is_valid_monetary(raw_num, cat):
                                            num_val = parse_currency_amount(raw_num)
                                            if num_val is not None and (num_val > 0 or (cat == "cng_idv" and num_val == 0)):
                                                text_cands[cat].append(ResolvedFieldCandidate(
                                                    field_name=cat,
                                                    value=raw_num,
                                                    exact_label=alias,
                                                    raw_evidence=f"{line} \\n {next_line}",
                                                    page=pno,
                                                    confidence=0.90,
                                                    method="text_multiline_window",
                                                    is_verified=True
                                                ))
                                                break

                        # Case 3: NCB Percentage extraction
                        if cat == "ncb" and re.search(rf"(?i)\b{re.escape(alias_norm)}\b", line_norm):
                            if not any(bad in line_norm for bad in ["protection", "cover", "add-on", "addon"]):
                                m_ncb = re.search(r"(\d{1,2}(?:\.\d{1,2})?)\s*%", line)
                                if m_ncb:
                                    ncb_str = f"{m_ncb.group(1)}%"
                                    if self._is_valid_ncb(ncb_str):
                                        text_cands["ncb"].append(ResolvedFieldCandidate(
                                            field_name="ncb",
                                            value=ncb_str,
                                            exact_label=alias,
                                            raw_evidence=line,
                                            page=pno,
                                            confidence=0.93,
                                            method="text_ncb_inline",
                                            is_verified=True
                                        ))

        return text_cands

    def _extract_standalone_total(self, page_texts: Dict[int, str]) -> Optional[ResolvedFieldCandidate]:
        """Extracts standalone Total / Grand Total lines."""
        for pno, text in page_texts.items():
            m_stand = re.search(
                r"(?i)(?:^|\n)\s*(?:GRAND\s*TOTAL|TOTAL\s*AMOUNT\s*PAYABLE|TOTAL\s*PREMIUM|TOTAL)\s*\n\s*(?:Rs\.?|INR|₹)?\s*([\d,]+(?:\.\d{2})?)",
                text
            )
            if m_stand:
                raw_val = m_stand.group(1).strip()
                num_val = parse_currency_amount(raw_val)
                if num_val and num_val > 0 and self._is_valid_monetary(raw_val, "total_premium"):
                    return ResolvedFieldCandidate(
                        field_name="total_premium",
                        value=raw_val,
                        raw_evidence=m_stand.group(0).strip(),
                        page=pno,
                        confidence=0.94,
                        method="table_standalone_total",
                        is_verified=True
                    )
        return None

    def _select_winning_candidate(
        self,
        category: str,
        candidates: List[ResolvedFieldCandidate]
    ) -> Optional[ResolvedFieldCandidate]:
        """Ranks and selects the winning candidate for a category using priority tiers and validation."""
        if not candidates:
            return None

        # Filter candidates through strict negative validation
        valid_cands = [c for c in candidates if self._is_valid_monetary(str(c.value), category)]
        if not valid_cands:
            return None

        # Do NOT assign TOTAL to tax
        if category == "tax":
            valid_cands = [
                c for c in valid_cands
                if not (any(t in c.exact_label.lower() for t in ["total premium", "total amount", "grand total", "total payable", "final payable"])
                        or (c.exact_label.strip().lower() == "total" and "tax" not in c.raw_evidence.lower() and "gst" not in c.raw_evidence.lower()))
            ]
            if not valid_cands:
                return None

        # Do NOT assign CGST/SGST to total_premium
        if category == "total_premium":
            valid_cands = [
                c for c in valid_cands
                if not any(tx == c.exact_label.strip().lower() for tx in ["cgst", "sgst", "igst", "tax", "taxes", "total gst", "total tax"])
                and not any(tx in c.exact_label.lower() for tx in ["cgst", "sgst", "igst"])
            ]
            if not valid_cands:
                return None

        def candidate_rank(c: ResolvedFieldCandidate) -> float:
            score = c.confidence * 100.0
            if "table_column_header" in c.method or "table_tax_amount" in c.method:
                score += 35.0
            elif "table_subitem" in c.method or "table_composite" in c.method:
                score += 30.0
            elif "layout_vertical" in c.method or "layout_horizontal" in c.method:
                score += 25.0
            elif "table_cell" in c.method:
                score += 20.0
            elif "text_multiline" in c.method:
                score += 15.0
            elif "text_inline" in c.method:
                score += 10.0

            # Prefer Page 1 for premium summary
            if c.page == 1:
                score += 10.0
            elif c.page == 2:
                score += 5.0

            # For IDV: prioritize vehicle-specific IDV over generic total value
            if category == "idv":
                lbl_l = c.exact_label.lower()
                if any(k in lbl_l for k in ["vehicle", "idv in rs", "insured declared"]):
                    score += 25.0
                elif "total value" in lbl_l:
                    score -= 10.0

            # For tax: strongly prioritize values > 20 over bare tax rates (like 18.0)
            if category == "tax":
                val_num = parse_currency_amount(str(c.value))
                if val_num is not None:
                    if val_num > 50.0:
                        score += 30.0  # Real tax amount
                    elif val_num in (18.0, 9.0, 12.0, 5.0, 28.0):
                        score -= 25.0  # Likely bare rate percentage

            return score

        valid_cands.sort(key=candidate_rank, reverse=True)
        return valid_cands[0]

    def _cross_validate_financials(
        self,
        results: Dict[str, Optional[ResolvedFieldCandidate]],
        all_candidates: Dict[str, List[ResolvedFieldCandidate]]
    ) -> None:
        """
        Validates arithmetic consistency across premium breakdown:
        Basic OD + Basic TP + Addons - Discounts ≈ Net Premium
        Net Premium + Tax ≈ Total Premium
        """
        net_cand = results.get("net_premium")
        tax_cand = results.get("tax")
        tot_cand = results.get("total_premium")

        # If total_premium is missing, check if Net + Tax matches a candidate
        if not tot_cand and net_cand and tax_cand:
            net_val = parse_currency_amount(str(net_cand.value))
            tax_val = parse_currency_amount(str(tax_cand.value))
            if net_val is not None and tax_val is not None:
                calc_tot = round(net_val + tax_val, 2)
                for c in all_candidates.get("total_premium", []):
                    if abs(parse_currency_amount(str(c.value)) - calc_tot) <= 1.0:
                        results["total_premium"] = c
                        tot_cand = c
                        break

        # If net_premium is missing, check if basic_od + basic_tp is equal to net
        if not net_cand:
            od_cand = results.get("basic_od_premium")
            tp_cand = results.get("basic_tp_premium")
            if od_cand and tp_cand:
                od_val = parse_currency_amount(str(od_cand.value))
                tp_val = parse_currency_amount(str(tp_cand.value))
                if od_val is not None and tp_val is not None:
                    calc_base = round(od_val + tp_val, 2)
                    for c in all_candidates.get("net_premium", []):
                        if abs(parse_currency_amount(str(c.value)) - calc_base) <= 1.0:
                            results["net_premium"] = c
                            break

        # Validate Net + Tax == Total
        if net_cand and tax_cand and tot_cand:
            net_val = parse_currency_amount(str(net_cand.value))
            tax_val = parse_currency_amount(str(tax_cand.value))
            tot_val = parse_currency_amount(str(tot_cand.value))

            if tax_val is not None and tot_val is not None and tot_val > 0:
                if tax_val >= tot_val:
                    logger.warning(f"Rejecting tax candidate ({tax_val}) >= total premium ({tot_val})")
                    results["tax"] = None
                    tax_cand = None

        if net_cand and tax_cand and tot_cand:
            net_val = parse_currency_amount(str(net_cand.value))
            tax_val = parse_currency_amount(str(tax_cand.value))
            tot_val = parse_currency_amount(str(tot_cand.value))

            if net_val is not None and tax_val is not None and tot_val is not None:
                calc_total = round(net_val + tax_val, 2)
                diff = abs(calc_total - tot_val)

                if diff <= 1.0:
                    tot_cand.is_verified = True
                    tot_cand.confidence = 1.0
                    tot_cand.metadata["arithmetic_verified"] = True
                    logger.info(f"Financial arithmetic verified: Net ({net_val}) + Tax ({tax_val}) == Total ({tot_val})")
                else:
                    # Check Case A: Single tax leg was captured (e.g. CGST 64.26 without SGST 64.26)
                    double_tax = round(tax_val * 2, 2)
                    if abs(round(net_val + double_tax, 2) - tot_val) <= 1.0:
                        results["tax"] = ResolvedFieldCandidate(
                            field_name="tax",
                            value=str(double_tax),
                            exact_label="CGST + SGST Total Tax",
                            raw_evidence=f"Reconciled dual tax: 2 * {tax_val} = {double_tax}",
                            page=tax_cand.page,
                            confidence=1.0,
                            method="reconciled_cgst_sgst",
                            is_verified=True
                        )
                        tot_cand.is_verified = True
                        tot_cand.confidence = 1.0
                        tot_cand.metadata["arithmetic_verified"] = True
                        logger.info(f"Financial arithmetic verified with dual tax: Net ({net_val}) + 2*Tax ({double_tax}) == Total ({tot_val})")
                    else:
                        # Check Case B: Find if another tax candidate in all_candidates reconciles
                        needed_tax = round(tot_val - net_val, 2)
                        reconciled = False
                        if needed_tax > 0:
                            for tc in all_candidates.get("tax", []):
                                if abs(parse_currency_amount(str(tc.value)) - needed_tax) <= 1.0:
                                    results["tax"] = tc
                                    tot_cand.is_verified = True
                                    tot_cand.confidence = 1.0
                                    tot_cand.metadata["arithmetic_verified"] = True
                                    logger.info(f"Financial arithmetic reconciled via candidate tax: Net ({net_val}) + Tax ({tc.value}) == Total ({tot_val})")
                                    reconciled = True
                                    break
                        if not reconciled:
                            tot_cand.metadata["arithmetic_discrepancy"] = f"Calculated {calc_total} vs declared {tot_val} (diff: {diff:.2f})"
                            logger.warning(f"Financial arithmetic discrepancy: calculated {calc_total} vs declared {tot_val}")

    def _is_valid_monetary(self, val_str: str, category: str) -> bool:
        """Strict domain-specific validation for monetary amounts."""
        s = val_str.strip()
        if not s:
            return False

        # NCB special handling (percentage or amount)
        if category == "ncb":
            return self._is_valid_ncb(s)

        # Disallow strings with letters or IMT codes
        clean_no_currency = re.sub(r"(?i)rs\.?|inr|₹|/-", "", s).strip()
        if any(c.isalpha() for c in clean_no_currency):
            return False

        # Reject obvious date formats (e.g. 10/01/2023, 2023-01-20, 15/03/2019)
        if re.search(r"\b\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}\b", s):
            return False

        # Reject phone numbers (10 digits starting with 6-9)
        clean_digits = re.sub(r"\D", "", s)
        if len(clean_digits) == 10 and clean_digits[0] in "6789":
            return False

        # Reject pin codes (6 digits)
        if len(clean_digits) == 6 and re.match(r"^\d{6}$", s):
            return False

        # Reject GSTINs (15 alphanumeric characters)
        if len(s) == 15 and re.match(r"^\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z\d]{1}[Z]{1}[A-Z\d]{1}$", s):
            return False

        # Reject Policy numbers / Account numbers (>= 12 digits without decimals)
        if len(clean_digits) >= 12 and "." not in s:
            return False

        # Reject Policy numbers with hyphens/slashes and length > 12
        if len(s) > 12 and any(sep in s for sep in ["/", "-", "#"]) and not re.match(r"^[-+]?\d+(?:\.\d+)?$", s.replace(",", "")):
            return False

        # Convert to float
        num = parse_currency_amount(s)
        if num is None:
            return False

        # Cap maximum plausible motor insurance amount (5 crore rupees)
        if num > 50000000.0:
            return False

        # CNG IDV allows 0.0
        if category == "cng_idv":
            return 0.0 <= num <= 200000.0

        # Total premium, net premium, IDV, and basic premiums MUST be strictly positive
        if category in ("total_premium", "net_premium", "basic_od_premium", "basic_tp_premium", "idv"):
            if num <= 0:
                return False

        # Must be non-negative
        if num < 0:
            return False

        # Discard obvious years (1900-2099) unless IDV or Total >= 10000 or formatted with decimals
        if 1900 <= num <= 2099 and category not in ("total_premium", "idv", "sum_insured"):
            if "." not in s and int(num) == num:
                return False

        return True

    def _is_valid_ncb(self, val_str: str) -> bool:
        """Validates that candidate NCB is a genuine percentage or monetary discount."""
        s = val_str.strip()
        if not s:
            return False
        # Reject names, words, spouses
        clean_letters = re.sub(r"(?i)rs\.?|inr|₹|/-|%", "", s).strip()
        if any(c.isalpha() for c in clean_letters):
            return False
        # Must contain digits
        if not any(c.isdigit() for c in s):
            return False
        if "%" in s:
            m = re.search(r"(\d{1,2}(?:\.\d{1,2})?)\s*%", s)
            if m:
                val = float(m.group(1))
                return 0.0 <= val <= 65.0
        num = parse_currency_amount(s)
        return num is not None and 0.0 <= num <= 50000.0

    def _log_diagnostics(
        self,
        category: str,
        candidates: List[ResolvedFieldCandidate],
        winner: Optional[ResolvedFieldCandidate]
    ):
        """Records diagnostic logging internally for troubleshooting (Requirement 13)."""
        rejection_reasons = []
        for c in candidates:
            if c != winner:
                if not self._is_valid_monetary(str(c.value), category):
                    rejection_reasons.append(f"Candidate {c.value} failed negative monetary validation")
                else:
                    rejection_reasons.append(f"Candidate {c.value} had lower rank score than {winner.value if winner else 'None'}")

        entry = {
            "field": category,
            "matched_label": winner.exact_label if winner else "",
            "nearby_text": winner.raw_evidence if winner else "",
            "candidates_found": [c.value for c in candidates],
            "selected": winner.value if winner else None,
            "rejection_reasons": rejection_reasons
        }
        self.debug_log.append(entry)
        logger.debug(f"AmountResolver diagnostics for {category}: selected={entry['selected']} (found {len(candidates)} candidates)")
