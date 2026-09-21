"""
Form Extractor: Extracts digital AcroForm and interactive widget fields directly.
Avoids OCR when native fillable form fields are present.
"""

from pathlib import Path
from typing import Dict, Any, List
import fitz  # PyMuPDF
from utils.logger import get_logger

logger = get_logger("form_extractor")


def extract_form_fields(file_path: Path) -> Dict[str, Any]:
    """
    Extracts all fillable AcroForm and interactive widget fields across all pages.
    Returns:
    {
        "field_name": "field_value",
        ...
    }
    """
    fields = {}
    try:
        doc = fitz.open(file_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            for widget in page.widgets():
                fname = widget.field_name
                fval = widget.field_value
                ftype = widget.field_type

                if fname:
                    clean_name = fname.strip()
                    # Boolean / checkbox normalization
                    if ftype == fitz.PDF_WIDGET_TYPE_CHECKBOX:
                        fields[clean_name] = bool(fval)
                    elif ftype == fitz.PDF_WIDGET_TYPE_RADIOBUTTON:
                        fields[clean_name] = str(fval).strip() if fval else False
                    else:
                        fields[clean_name] = str(fval).strip() if fval else ""

        doc.close()
    except Exception as e:
        logger.error(f"Error extracting form fields from {file_path}: {e}")

    return fields
