"""
Table Extractor: Preserves structured tabular relationships without flattening.
Combines pdfplumber and PyMuPDF table detection to output rows, columns, and key-value maps.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import re
import pdfplumber
import fitz
from utils.logger import get_logger

logger = get_logger("table_extractor")


def extract_tables_from_page(file_path: Path, page_number: int) -> List[Dict[str, Any]]:
    """
    Extracts structured tables from a specific page.
    Returns:
    [
        {
            "headers": ["Item", "Amount", ...],
            "rows": [["Building", "5000000"], ...],
            "key_value_map": {"Building": "5000000", ...},
            "raw_matrix": [[...]]
        }
    ]
    """
    structured_tables = []

    # Try pdfplumber first for precision layout
    try:
        with pdfplumber.open(file_path) as pdf:
            if 1 <= page_number <= len(pdf.pages):
                plumber_page = pdf.pages[page_number - 1]
                tables = plumber_page.extract_tables()

                for table_idx, raw_table in enumerate(tables):
                    if not raw_table or len(raw_table) < 2:
                        continue

                    # Clean nulls / newlines in cells
                    cleaned_table = []
                    for row in raw_table:
                        cleaned_row = [str(cell).strip().replace("\n", " ") if cell is not None else "" for cell in row]
                        # Discard completely empty rows
                        if any(cleaned_row):
                            cleaned_table.append(cleaned_row)

                    if len(cleaned_table) < 2:
                        continue

                    # Header is typically row 0
                    headers = cleaned_table[0]
                    rows = cleaned_table[1:]

                    # Build key-value map if 2-column or recognized header-value relationship
                    kv_map = {}
                    if len(headers) == 2:
                        for r in rows:
                            if len(r) >= 2 and r[0]:
                                kv_map[r[0]] = r[1]
                    else:
                        # Multi-column table
                        for r in rows:
                            if len(r) >= 2 and r[0]:
                                for c_idx in range(1, len(r)):
                                    col_name = headers[c_idx] if c_idx < len(headers) and headers[c_idx] else f"Col_{c_idx}"
                                    kv_map[f"{r[0]} - {col_name}"] = r[c_idx]

                    structured_tables.append({
                        "table_index": table_idx,
                        "page_number": page_number,
                        "headers": headers,
                        "rows": rows,
                        "key_value_map": kv_map,
                        "raw_matrix": cleaned_table,
                    })

                if structured_tables:
                    return structured_tables
    except Exception as e:
        logger.warning(f"pdfplumber table extraction warning on page {page_number}: {e}")

    # Fallback to PyMuPDF find_tables()
    try:
        doc = fitz.open(file_path)
        if 1 <= page_number <= len(doc):
            page = doc[page_number - 1]
            tabs = page.find_tables()
            for t_idx, tab in enumerate(tabs.tables):
                extracted_data = tab.extract()
                if extracted_data and len(extracted_data) >= 2:
                    headers = [str(c).strip() if c else "" for c in extracted_data[0]]
                    rows = [[str(c).strip() if c else "" for c in r] for r in extracted_data[1:]]
                    kv_map = {}
                    for r in rows:
                        if len(r) >= 2 and r[0]:
                            kv_map[r[0]] = r[1]

                    structured_tables.append({
                        "table_index": t_idx,
                        "page_number": page_number,
                        "headers": headers,
                        "rows": rows,
                        "key_value_map": kv_map,
                        "raw_matrix": extracted_data,
                    })
        doc.close()
    except Exception as e:
        logger.warning(f"PyMuPDF find_tables warning on page {page_number}: {e}")

    return structured_tables


COMMON_TABLE_LABEL_WORDS = [
    "NAME", "ADDRESS", "MOBILE", "EMAIL", "AADHAR", "PAN", "GST", "GSTIN",
    "MAKE", "MODEL", "VARIANT", "ENGINE", "CHASSIS", "REGISTRATION", "REGN",
    "RTO", "BODY TYPE", "FUEL", "SEATING", "CAPACITY", "CUBIC", "ODOMETER",
    "FASTAG", "FINANCIER", "HYPOTHECATION", "INVOICE", "PREMIUM", "POLICY",
    "PERIOD", "CERTIFICATE", "DATE", "NOMINEE", "IDV", "SUM INSURED", "PARTNER",
    "VEHICLE", "TYPE OF", "YEAR OF", "DETAILS", "SCHEDULE", "BRANCH", "AGENT"
]

def is_table_label_cell(text: str) -> bool:
    if not text:
        return False
    clean = " ".join(text.strip().split()).upper().rstrip(" :.-#")
    for term in COMMON_TABLE_LABEL_WORDS:
        if clean == term or clean.startswith(term + " ") or clean.endswith(" " + term) or clean.startswith(term + ":") or clean.startswith(term + "/"):
            return True
    return False


def find_table_field(
    tables: List[Dict[str, Any]],
    label_keywords: List[str],
    value_pattern: Optional[str] = None,
    max_down: int = 4,
    max_right: int = 4,
    max_page: int = 2
) -> Optional[Dict[str, Any]]:
    """
    Searches tables for label-value relationships using grid alignment:
    1. Same-column downwards (Header at row r -> Value at row r+k)
    2. Same-row rightwards (Label at col c -> Value at col c+k)
    3. Inline delimiter within cell (Label: Value)
    Guards strictly against long paragraphs, policy terms, and other label headers.
    """
    if not tables:
        return None

    for table in tables:
        pno = table.get("page_number", 1)
        if pno > max_page:
            continue
        matrix = table.get("raw_matrix", [])
        if not matrix:
            continue

        for r_idx, row in enumerate(matrix):
            for c_idx, cell in enumerate(row):
                if not cell:
                    continue
                c_str = " ".join(str(cell).strip().split())
                # Discard long boilerplate/terms cells
                if len(c_str) > 60 or any(k in c_str.lower() for k in [
                    "payable", "shall", "unless", "hereby", "certify", "endorse", "limitation", "clause"
                ]):
                    continue

                for kw in label_keywords:
                    # Match exact word boundaries
                    pattern = rf"(?i)\b{re.escape(kw)}\b"
                    if re.search(pattern, c_str):
                        # 1. Inline within same cell
                        m_inline = re.search(rf"(?i)\b{re.escape(kw)}\b[\s\.:/#=-]*([^\n\r,;]{{2,60}})", c_str)
                        if m_inline:
                            v = m_inline.group(1).strip()
                            if v and not is_table_label_cell(v) and (not value_pattern or re.search(value_pattern, v)):
                                return {
                                    "label": kw,
                                    "exact_label": c_str,
                                    "value": v,
                                    "page": pno,
                                    "method": "table_cell_inline",
                                    "evidence": f"{c_str}"
                                }

                        # 2. Immediate right neighbor (standard 2-column or 4-column key-value grid)
                        if c_idx + 1 < len(row):
                            v_cand = row[c_idx + 1]
                            if v_cand and str(v_cand).strip():
                                v_str = " ".join(str(v_cand).strip().split())
                                if not is_table_label_cell(v_str) and (not value_pattern or re.search(value_pattern, v_str)):
                                    return {
                                        "label": kw,
                                        "exact_label": c_str,
                                        "value": v_str,
                                        "page": pno,
                                        "method": "table_cell_right",
                                        "evidence": f"{c_str} -> {v_str}"
                                    }

                        # 3. Downward in same column (columnar tables)
                        for r_next in range(r_idx + 1, min(r_idx + 1 + max_down, len(matrix))):
                            v_cand = matrix[r_next][c_idx]
                            if v_cand and str(v_cand).strip():
                                v_str = " ".join(str(v_cand).strip().split())
                                # If the cell directly below is another label header, stop searching down
                                if is_table_label_cell(v_str):
                                    break
                                if not value_pattern or re.search(value_pattern, v_str):
                                    return {
                                        "label": kw,
                                        "exact_label": c_str,
                                        "value": v_str,
                                        "page": pno,
                                        "method": "table_cell_down",
                                        "evidence": f"{c_str} -> {v_str}"
                                    }
                                break

                        # 4. Remaining rightward in same row (if empty gap columns exist)
                        for c_next in range(c_idx + 2, min(c_idx + 1 + max_right, len(row))):
                            v_cand = row[c_next]
                            if v_cand and str(v_cand).strip():
                                v_str = " ".join(str(v_cand).strip().split())
                                if is_table_label_cell(v_str):
                                    break
                                if not value_pattern or re.search(value_pattern, v_str):
                                    return {
                                        "label": kw,
                                        "exact_label": c_str,
                                        "value": v_str,
                                        "page": pno,
                                        "method": "table_cell_right",
                                        "evidence": f"{c_str} -> {v_str}"
                                    }
                                break

    return None


def find_proposer_in_header_tables(tables: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Detects proposer/insured customer name from Page 1-2 header tables.
    Handles:
    1. Explicit 'Name' / 'Insured Name' cells
    2. Proposer box located directly above 'Address:' and adjacent to 'Policy #:'
    """
    if not tables:
        return None

    for table in tables:
        pno = table.get("page_number", 1)
        if pno > 2:
            continue
        matrix = table.get("raw_matrix", [])
        for r_idx, row in enumerate(matrix[:6]):
            for c_idx, cell in enumerate(row):
                if not cell:
                    continue
                c_str = " ".join(str(cell).strip().split())
                if len(c_str) < 3 or len(c_str) > 65:
                    continue
                if any(k in c_str.lower() for k in [
                    "payable", "shall", "policy", "premium", "vehicle", "settlement", "defence",
                    "conditions", "limitations", "insurance", "company", "limited", "office",
                    "branch", "address", "gstin", "service", "schedule", "period"
                ]):
                    continue

                # A. Explicit Name label in same cell
                m_inline = re.search(r"(?i)\b(?:name\s*of\s*(?:the\s*)?insured|insured\s*name|proposer\s*name|customer\s*name)\b[\s\.:/-]*([A-Za-z\s\.]{3,50})", c_str)
                if m_inline:
                    name_cand = m_inline.group(1).strip()
                    if len(name_cand) >= 3 and not any(k in name_cand.lower() for k in ["vehicle", "insurance", "office"]):
                        return {
                            "value": name_cand,
                            "exact_label": c_str,
                            "page": pno,
                            "method": "table_name_label",
                            "evidence": f"{c_str} -> {name_cand}"
                        }

                # B. Cell is 'Name' and value is in next cell to right
                if c_str.lower() in ["name", "insured name", "customer name", "proposer name", "proposer's name"]:
                    for c_next in range(c_idx + 1, len(row)):
                        cand = row[c_next]
                        if cand and str(cand).strip():
                            v = " ".join(str(cand).strip().split())
                            if len(v) >= 3 and not any(k in v.lower() for k in ["vehicle", "insurance"]):
                                v_upper = v.upper().rstrip(" :.-#")
                                if v_upper in [
                                    "PERSON", "CUSTOMER", "SELF", "SPOUSE", "DAUGHTER", "SON",
                                    "FATHER", "MOTHER", "MEMBER", "MEMBER ID", "RELATIONSHIP", "RELATION",
                                    "GENDER", "AGE", "DOB", "DATE OF BIRTH", "INSURED PERSON", "INSURED PERSONS"
                                ] or is_table_label_cell(v):
                                    continue
                                return {
                                    "value": v,
                                    "exact_label": c_str,
                                    "page": pno,
                                    "method": "table_name_cell_right",
                                    "evidence": f"{c_str} -> {v}"
                                }

                # C. Proposer box in header: cell above 'Address:'
                has_address_below = False
                for r_next in range(r_idx + 1, min(r_idx + 3, len(matrix))):
                    cand = matrix[r_next][c_idx]
                    if cand and "address" in str(cand).lower():
                        has_address_below = True
                        break
                if has_address_below:
                    words = c_str.split()
                    if 2 <= len(words) <= 5 and all(w.replace(".", "").isalpha() for w in words):
                        return {
                            "value": c_str,
                            "exact_label": "Insured / Proposer (Header Block)",
                            "page": pno,
                            "method": "table_header_block",
                            "evidence": f"Proposer cell above Address -> {c_str}"
                        }

    return None


def find_policy_numbers_in_tables(tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Finds and cleanly separates all policy number candidates across tables without blind merging.
    Supports multi-token values (e.g. '1-12IXB8DM P400').
    """
    if not tables:
        return []

    candidates: List[Dict[str, Any]] = []
    pol_pattern = r"(?i)(policy\s*(?:#|no\.?|number|certificate\s*no\.?)[\s\.:/-]*)\s*([A-Za-z0-9][A-Za-z0-9/\-\s]{3,35}[A-Za-z0-9])(?=\s+(?:policy\s*(?:#|no|number)|date|period|unique|\Z)|$)"

    for table in tables:
        pno = table.get("page_number", 1)
        matrix = table.get("raw_matrix", [])
        for row in matrix:
            for cell in row:
                if not cell:
                    continue
                c_str = " ".join(str(cell).strip().split())
                # Discard long boilerplate
                if len(c_str) > 120 or any(k in c_str.lower() for k in ["payable", "shall", "benefit under the policy", "unless"]):
                    continue

                for m in re.finditer(pol_pattern, c_str):
                    lbl = m.group(1).strip()
                    val = m.group(2).strip()
                    # Hard negative checks
                    if any(dis in val.upper() for dis in ["PREVIOUS", "PAYABLE", "POLICY", "INSURANCE", "SCHEDULE"]):
                        continue
                    if len(val) >= 5:
                        candidates.append({
                            "exact_label": lbl,
                            "value": val,
                            "page": pno,
                            "method": "table_cell",
                            "evidence": f"{lbl} {val}"
                        })

    return candidates
