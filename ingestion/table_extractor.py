"""
Table Extractor: Preserves structured tabular relationships without flattening.
Combines pdfplumber and PyMuPDF table detection to output rows, columns, and key-value maps.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
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
                        "headers": headers,
                        "rows": rows,
                        "key_value_map": kv_map,
                        "raw_matrix": extracted_data,
                    })
        doc.close()
    except Exception as e:
        logger.warning(f"PyMuPDF find_tables warning on page {page_number}: {e}")

    return structured_tables
