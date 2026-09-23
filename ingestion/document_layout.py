"""
Document Layout Model: Coordinate-aware spatial representation of PDF pages.
Captures tokens, words, lines, blocks, and table cells with bounding boxes [x0, y0, x1, y1].
Enables deterministic spatial queries (e.g. text to the right of label, text below label, table cells).
"""

from typing import List, Dict, Any, Optional, Tuple
import re


class TokenBox:
    def __init__(
        self,
        text: str,
        bbox: Tuple[float, float, float, float],
        page: int,
        line_no: int = 0,
        block_no: int = 0,
        source: str = "native"
    ):
        self.text = text
        self.x0, self.y0, self.x1, self.y1 = bbox
        self.bbox = bbox
        self.page = page
        self.line_no = line_no
        self.block_no = block_no
        self.source = source

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "bbox": [round(c, 2) for c in self.bbox],
            "page": self.page,
            "line_no": self.line_no,
            "block_no": self.block_no,
            "source": self.source
        }


class LineBox:
    def __init__(
        self,
        text: str,
        bbox: Tuple[float, float, float, float],
        page: int,
        line_no: int = 0,
        block_no: int = 0,
        tokens: Optional[List[TokenBox]] = None
    ):
        self.text = text.strip()
        self.x0, self.y0, self.x1, self.y1 = bbox
        self.bbox = bbox
        self.page = page
        self.line_no = line_no
        self.block_no = block_no
        self.tokens = tokens or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "bbox": [round(c, 2) for c in self.bbox],
            "page": self.page,
            "line_no": self.line_no,
            "block_no": self.block_no,
            "token_count": len(self.tokens)
        }


class BlockBox:
    def __init__(
        self,
        text: str,
        bbox: Tuple[float, float, float, float],
        page: int,
        block_no: int = 0,
        lines: Optional[List[LineBox]] = None
    ):
        self.text = text.strip()
        self.x0, self.y0, self.x1, self.y1 = bbox
        self.bbox = bbox
        self.page = page
        self.block_no = block_no
        self.lines = lines or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "bbox": [round(c, 2) for c in self.bbox],
            "page": self.page,
            "block_no": self.block_no,
            "line_count": len(self.lines)
        }


class DocumentLayoutModel:
    """
    In-memory spatial index for an entire document across all its pages.
    Supports high-precision coordinate-aware label-value extraction.
    """
    def __init__(self, document_id: str):
        self.document_id = document_id
        self.pages_tokens: Dict[int, List[TokenBox]] = {}
        self.pages_lines: Dict[int, List[LineBox]] = {}
        self.pages_blocks: Dict[int, List[BlockBox]] = {}
        self.tables: List[Dict[str, Any]] = []

    def add_page_elements(
        self,
        page: int,
        tokens: List[TokenBox],
        lines: List[LineBox],
        blocks: List[BlockBox]
    ):
        self.pages_tokens[page] = tokens
        self.pages_lines[page] = lines
        self.pages_blocks[page] = blocks

    def add_tables(self, tables: List[Dict[str, Any]]):
        self.tables.extend(tables)

    def find_label_lines(self, label_aliases: List[str]) -> List[Tuple[LineBox, str]]:
        """
        Finds all lines or multi-line sequences matching any alias in the alias list.
        Returns list of (LineBox, matched_alias).
        """
        matches = []
        for pno, lines in self.pages_lines.items():
            # 1. Single-line matching
            for line in lines:
                l_text = line.text.lower()
                for alias in label_aliases:
                    alias_clean = alias.lower().strip()
                    pattern = rf"(?i)\b{re.escape(alias_clean)}\b"
                    if re.search(pattern, l_text):
                        # Avoid matching 1-word generic aliases (e.g. 'to', 'from', 'make') against full paragraphs
                        if len(alias_clean.split()) == 1 and len(line.text.split()) > 6:
                            continue
                        matches.append((line, alias))
                        break

            # 2. Multi-line label matching (2, 3, or 4 contiguous lines in same block or proximity)
            num_lines = len(lines)
            for i in range(num_lines - 1):
                for k in (2, 3, 4):
                    if i + k <= num_lines:
                        group = lines[i:i+k]
                        # Must be on same page, close vertical proximity (gap <= 18px) and similar left x (diff <= 40px)
                        y_gaps = [group[j+1].y0 - group[j].y1 for j in range(k-1)]
                        x_diffs = [abs(group[j+1].x0 - group[j].x0) for j in range(k-1)]
                        if all(-8.0 <= yg <= 18.0 for yg in y_gaps) and all(xd <= 50.0 for xd in x_diffs):
                            combined_text = " ".join([l.text.strip() for l in group])
                            norm_comb = re.sub(r"[^\w]+", " ", combined_text).strip().lower()

                            for alias in label_aliases:
                                alias_clean = alias.lower().strip()
                                norm_alias = re.sub(r"[^\w]+", " ", alias_clean).strip().lower()
                                if not norm_alias:
                                    continue
                                if len(norm_alias.split()) == 1:
                                    is_match = (norm_comb == norm_alias)
                                else:
                                    is_match = (norm_alias in norm_comb or norm_comb in norm_alias)
                                if is_match:
                                    merged_bbox = (
                                        min(l.x0 for l in group),
                                        min(l.y0 for l in group),
                                        max(l.x1 for l in group),
                                        max(l.y1 for l in group)
                                    )
                                    merged_line = LineBox(
                                        text=combined_text,
                                        bbox=merged_bbox,
                                        page=pno,
                                        block_no=group[0].block_no,
                                        tokens=[tok for l in group for tok in l.tokens]
                                    )
                                    matches.append((merged_line, alias))
                                    break
        return matches

    def extract_value_for_label(
        self,
        label_aliases: List[str],
        value_pattern: Optional[str] = None,
        max_dist_right: float = 350.0,
        max_dist_below: float = 50.0,
    ) -> List[Dict[str, Any]]:
        """
        Searches for values associated with label aliases:
        1. On the same line immediately following the label separator (':', '-', '=')
        2. To the geometric right on the same horizontal band, strictly respecting column boundaries
        3. Directly below the label within max_dist_below
        """
        LABEL_TERMINATORS = {
            "period", "policy", "date", "issuance", "invoice", "to", "from",
            "registration", "engine", "chassis", "make", "model", "insured",
            "address", "partner", "mobile", "email", "aadhar", "details",
            "vehicle", "endorsement", "receipt", "type", "sub-type", "variant",
            "rto", "location", "fuel", "seating", "capacity", "cubic",
            "odometer", "fastag", "financier", "body", "class", "hypothecation"
        }

        candidates = []
        label_matches = self.find_label_lines(label_aliases)

        for line, matched_alias in label_matches:
            pno = line.page
            line_tokens = self.pages_tokens.get(pno, [])
            page_lines = self.pages_lines.get(pno, [])

            # Case A: Value is on the same line after delimiter
            # e.g., "Policy Number: POL-123456"
            escaped_alias = re.escape(matched_alias)
            match_inline = re.search(
                rf"(?i)\b{escaped_alias}\b\s*[:\.\-/#=\s]*([^\n\r,;]{{2,100}})",
                line.text
            )
            if match_inline:
                val = match_inline.group(1).strip()
                val = re.sub(r"^[:\.\-/#=\s]+", "", val).strip()
                if val:
                    # Strip subsequent column content if another label appears inline
                    for term in LABEL_TERMINATORS:
                        term_match = re.search(rf"(?i)\s+\b{re.escape(term)}\b", val)
                        if term_match:
                            val = val[:term_match.start()].strip()
                    val_clean = val.lower().rstrip("):-/#.").lstrip("(-/#.")
                    if val_clean in LABEL_TERMINATORS:
                        continue
                    if val and (not value_pattern or re.search(value_pattern, val)):
                        candidates.append({
                            "value": val,
                            "raw_text": line.text,
                            "page": line.page,
                            "bbox": line.bbox,
                            "method": "inline_delimiter",
                            "confidence": 0.95
                        })
                        continue

            # Case B: Spatial right on same horizontal band (Column-boundary and Row-alignment aware)
            # Find nearest neighboring column boundary to the right on the same row band
            col_boundary = line.x1 + max_dist_right
            for other_l in page_lines:
                if other_l.page == pno and other_l.x0 > line.x1 + 10.0:
                    vert_overlap = min(line.y1, other_l.y1) - max(line.y0, other_l.y0)
                    if vert_overlap > 1.5:
                        other_clean = other_l.text.lower()
                        if any(term in other_clean for term in LABEL_TERMINATORS):
                            if other_l.x0 < col_boundary:
                                col_boundary = other_l.x0

            # Identify distinct vertical row baselines belonging to this label
            label_tokens = line.tokens if line.tokens else [t for t in line_tokens if t.x0 >= line.x0 - 2.0 and t.x1 <= line.x1 + 2.0 and not (t.y1 < 'y_band_top' or t.y0 > 'y_band_bot')]
            if label_tokens:
                # Group by y_mid to find all valid row centers of this label
                unique_y_mids = []
                for lt in label_tokens:
                    ym = (lt.y0 + lt.y1) / 2.0
                    if not any(abs(ym - existing) <= 4.0 for existing in unique_y_mids):
                        unique_y_mids.append(ym)
            else:
                unique_y_mids = [(line.y0 + line.y1) / 2.0]

            for row_y_mid in unique_y_mids:
                row_tokens_right = [
                    t for t in line_tokens
                    if t.x0 >= line.x1 - 5.0
                    and t.x1 <= col_boundary + 5.0
                    and abs((t.y0 + t.y1) / 2.0 - row_y_mid) <= 5.5
                ]

                if row_tokens_right:
                    row_tokens_right.sort(key=lambda t: t.x0)
                    valid_tokens = []
                    for idx, t in enumerate(row_tokens_right):
                        t_text_clean = t.text.strip().lower()
                        term_token = t_text_clean.rstrip("):-/#.").lstrip("(-/#.")
                        if idx > 0 and term_token in LABEL_TERMINATORS:
                            break
                        if idx > 0 and (t.x0 - row_tokens_right[idx-1].x1) > 28.0:
                            break
                        valid_tokens.append(t)

                    if valid_tokens:
                        cand_str = " ".join([t.text for t in valid_tokens]).strip()
                        cand_str = re.sub(r"^[:\.\-/#=\s]+", "", cand_str).strip()
                        # If a label terminator keyword leaked into the end or inline, strip it
                        for term in LABEL_TERMINATORS:
                            term_m = re.search(rf"(?i)\s+\b{re.escape(term)}\b", cand_str)
                            if term_m:
                                cand_str = cand_str[:term_m.start()].strip()
                        if cand_str and (not value_pattern or re.search(value_pattern, cand_str)):
                            candidates.append({
                                "value": cand_str,
                                "raw_text": f"{line.text} -> {cand_str}",
                                "page": pno,
                                "bbox": (valid_tokens[0].x0, min(t.y0 for t in valid_tokens), valid_tokens[-1].x1, max(t.y1 for t in valid_tokens)),
                                "method": "spatial_right",
                                "confidence": 0.94
                            })

            # Case C: Spatial below (stacked labels above values)
            tokens_below = [
                t for t in line_tokens
                if t.y0 >= line.y1 - 3.0
                and t.y0 <= line.y1 + max_dist_below
                and (t.x0 >= line.x0 - 20.0 and t.x0 <= line.x1 + 60.0)
            ]
            if tokens_below:
                tokens_below.sort(key=lambda t: (t.y0, t.x0))
                first_y = tokens_below[0].y0
                row_tokens = [t for t in tokens_below if abs(t.y0 - first_y) <= 8.0]
                row_tokens.sort(key=lambda t: t.x0)
                first_word = row_tokens[0].text.strip().lower().rstrip("):-/#.")
                if first_word not in LABEL_TERMINATORS:
                    cand_str = " ".join([t.text for t in row_tokens]).strip()
                    cand_str = re.sub(r"^[:\.\-/#=\s]+", "", cand_str).strip()
                    if cand_str and (not value_pattern or re.search(value_pattern, cand_str)):
                        candidates.append({
                            "value": cand_str,
                            "raw_text": f"{line.text} [BELOW] {cand_str}",
                            "page": pno,
                            "bbox": (row_tokens[0].x0, row_tokens[0].y0, row_tokens[-1].x1, row_tokens[-1].y1),
                            "method": "spatial_below",
                            "confidence": 0.88
                        })

        return candidates
