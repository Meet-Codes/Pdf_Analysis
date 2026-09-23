"""
Text Extractor: High-fidelity native text extraction with layout and reading order.
"""

from pathlib import Path
from typing import Dict, List, Any
import fitz  # PyMuPDF
from utils.logger import get_logger

logger = get_logger("text_extractor")


def extract_native_text(file_path: Path, page_numbers: List[int] = None) -> Dict[int, Dict[str, Any]]:
    """
    Extracts native text, blocks, lines, and words with bboxes from PDF pages using PyMuPDF.
    Returns dictionary keyed by 1-based page number:
    {
        page_num: {
            "text": "...",
            "blocks": [...],
            "lines": [...],
            "tokens": [...],
            "word_count": int
        }
    }
    """
    from ingestion.document_layout import TokenBox, LineBox, BlockBox

    results = {}
    try:
        doc = fitz.open(file_path)
        total_pages = len(doc)
        target_pages = page_numbers if page_numbers else list(range(1, total_pages + 1))

        for pno in target_pages:
            if 1 <= pno <= total_pages:
                page = doc[pno - 1]
                text = page.get_text("text")

                # 1. Extract words (tokens) with exact coordinates
                raw_words = page.get_text("words")
                tokens: List[TokenBox] = []
                for w in raw_words:
                    # w: (x0, y0, x1, y1, word_str, block_no, line_no, word_no)
                    if len(w) >= 5 and str(w[4]).strip():
                        tokens.append(TokenBox(
                            text=str(w[4]).strip(),
                            bbox=(float(w[0]), float(w[1]), float(w[2]), float(w[3])),
                            page=pno,
                            line_no=int(w[6]) if len(w) > 6 else 0,
                            block_no=int(w[5]) if len(w) > 5 else 0,
                            source="native"
                        ))

                # 2. Group tokens into LineBoxes
                lines: List[LineBox] = []
                line_map: Dict[Tuple[int, int], List[TokenBox]] = {}
                for t in tokens:
                    key = (t.block_no, t.line_no)
                    line_map.setdefault(key, []).append(t)

                for (b_no, l_no), l_tokens in sorted(line_map.items()):
                    l_tokens.sort(key=lambda tok: tok.x0)
                    line_str = " ".join([tok.text for tok in l_tokens])
                    min_x0 = min(tok.x0 for tok in l_tokens)
                    min_y0 = min(tok.y0 for tok in l_tokens)
                    max_x1 = max(tok.x1 for tok in l_tokens)
                    max_y1 = max(tok.y1 for tok in l_tokens)
                    lines.append(LineBox(
                        text=line_str,
                        bbox=(min_x0, min_y0, max_x1, max_y1),
                        page=pno,
                        line_no=l_no,
                        block_no=b_no,
                        tokens=l_tokens
                    ))

                # 3. Extract raw blocks
                raw_blocks = page.get_text("blocks")
                blocks: List[BlockBox] = []
                for b in raw_blocks:
                    # b: (x0, y0, x1, y1, text, block_no, block_type)
                    if len(b) >= 5 and b[4].strip():
                        b_text = b[4].strip()
                        b_no = b[5] if len(b) > 5 else 0
                        b_lines = [l for l in lines if l.block_no == b_no]
                        blocks.append(BlockBox(
                            text=b_text,
                            bbox=(float(b[0]), float(b[1]), float(b[2]), float(b[3])),
                            page=pno,
                            block_no=b_no,
                            lines=b_lines
                        ))

                results[pno] = {
                    "text": text.strip(),
                    "blocks": blocks,
                    "lines": lines,
                    "tokens": tokens,
                    "word_count": len(text.split()),
                }
        doc.close()
    except Exception as e:
        logger.error(f"Error extracting native text from {file_path}: {e}")

    return results
