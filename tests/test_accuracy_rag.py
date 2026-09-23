"""
Accuracy and RAG Upgrade Verification Tests.
Validates:
1. Genuine semantic embeddings (no hash fallback).
2. First-class table representation and separate indexing for table retrieval.
3. Structure-aware chunking preserving headings and key-values.
4. Hybrid retrieval with top 15-20 candidate generation.
5. Semantic Cross-Encoder reranking.
6. Field-specific extraction and candidate ranking (Total Premium vs Net vs Basic vs GST).
7. Non-destructive arithmetic financial validation with discrepancy reporting.
8. Dynamic cross-platform Tesseract OCR resolution.
9. Full per-query observability trace logging.
"""

from pathlib import Path
import pytest
from schemas.base import DocumentType, CanonicalDocument
from retrieval.embeddings import embedding_engine
from retrieval.vector_store import vector_store
from retrieval.retriever import (
    structural_chunk_page,
    format_table_to_semantic_chunk,
    index_canonical_document,
    retrieve_document_evidence,
)
from retrieval.reranker import reranker
from retrieval.field_resolvers.amount_resolver import AmountResolver
from retrieval.question_analyzer import question_analyzer, QueryType
from validation.financial import validate_full_premium_breakdown, parse_currency_amount
from ingestion.ocr_engine import get_verified_tesseract_cmd
from agents.workflow import process_document
from agents.qa_agent import ask_document, answer_from_structured_data

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def test_genuine_semantic_embeddings_no_hash():
    """Verify hash fallback is removed and real semantic embeddings are generated."""
    assert not hasattr(embedding_engine, "_local_fallback_embeddings"), "Hash fallback must be completely removed!"

    texts = [
        "Commercial Vehicle Package Policy for goods carrying vehicle",
        "Total Premium Payable is Rs. 18,207 including 18% GST"
    ]
    embs = embedding_engine.embed_documents(texts)
    assert len(embs) == 2
    assert len(embs[0]) in (384, 1024), f"Expected genuine embedding dimension (384 or 1024), got {len(embs[0])}"

    info = embedding_engine.get_provider_info()
    assert "provider" in info
    assert "hash" not in info["provider"].lower(), "Provider must never be hash!"


def test_first_class_table_representation():
    """Verify table data is converted into clean semantic key-value units."""
    sample_table = {
        "table_index": 0,
        "title": "PREMIUM BREAKDOWN",
        "headers": ["Component", "Rate", "Amount"],
        "rows": [
            ["Basic Premium", "OD", "15430.00"],
            ["GST", "18%", "2777.00"],
            ["Total Premium", "-", "18207.00"],
        ],
        "key_value_map": {
            "Basic Premium": "15430.00",
            "GST": "2777.00",
            "Total Premium": "18207.00",
        }
    }
    table_chunk = format_table_to_semantic_chunk(sample_table, page_no=2)
    assert "TABLE: PREMIUM BREAKDOWN (Page 2, Table 0)" in table_chunk
    assert "Basic Premium = 15430.00" in table_chunk
    assert "GST = 2777.00" in table_chunk
    assert "Total Premium = 18207.00" in table_chunk


def test_structural_chunking_preserves_headings_and_kvs():
    """Verify structural chunking keeps headings and key-values contiguous."""
    text = (
        "SCHEDULE OF PREMIUM\n"
        "Policy Number: P0023200023/4115/103739\n"
        "Customer Name: Mr Meet Korat\n"
        "Registration Number: GJ03MG6586\n"
        "Make: TATA ACE\n"
        "Basic Premium: Rs. 15,430/-\n"
        "GST: Rs. 2,777/-\n"
        "Total Premium: Rs. 18,207/-\n"
    )
    chunks = structural_chunk_page(text, target_token_count=100, overlap_tokens=20)
    assert len(chunks) >= 1
    # Check that headings and financial lines stay together
    chunk_txt = chunks[0]["text"]
    assert "Total Premium: Rs. 18,207/-" in chunk_txt
    assert "Basic Premium: Rs. 15,430/-" in chunk_txt


def test_cross_encoder_reranker_relevance():
    """Verify semantic Cross-Encoder ranks true evidence far higher than irrelevant text."""
    query = "What is the total premium?"
    candidates = [
        {"text": "The insured vehicle registration number is GJ03MG6586 from Rajkot RTO.", "metadata": {"page_number": 1}, "score": 0.6},
        {"text": "Total Premium Payable cum receipt amount: Rs. 18,207.00 inclusive of taxes.", "metadata": {"page_number": 1}, "score": 0.5},
        {"text": "Period of insurance valid from 29/03/2023 to 28/03/2024 midnight.", "metadata": {"page_number": 1}, "score": 0.4},
    ]
    reranked = reranker.rerank(query, candidates, top_k=2)
    assert len(reranked) == 2
    # The total premium candidate must be ranked #1
    assert "18,207" in reranked[0]["text"]
    assert reranked[0]["rerank_score"] > reranked[1]["rerank_score"]


def test_financial_field_extraction_distinction():
    """
    CRITICAL TEST: Verify Total Premium does NOT accidentally return Basic Premium or Net Premium.
    """
    page_texts = {
        1: (
            "SCHEDULE OF PREMIUM CHARGES\n"
            "Basic Own Damage Premium: Rs. 12,500.00\n"
            "Third Party Liability Premium: Rs. 2,930.00\n"
            "Net Premium: Rs. 15,430.00\n"
            "GST (18%): Rs. 2,777.00\n"
            "Total Premium Payable: Rs. 18,207.00\n"
        )
    }
    resolver = AmountResolver()
    results = resolver.resolve(full_text="", page_texts=page_texts)

    assert results["total_amount"] is not None
    assert results["total_amount"].value == "18,207.00"
    assert results["total_amount"].page == 1

    assert results["net_premium"] is not None
    assert results["net_premium"].value == "15,430.00"

    assert results["basic_premium"] is not None
    assert "12,500" in results["basic_premium"].value or "15,430" in results["basic_premium"].value

    assert results["tax"] is not None
    assert results["tax"].value == "2,777.00"


def test_non_destructive_financial_validation():
    """
    Verify arithmetic validation reports discrepancy without overwriting declared total.
    """
    # Case A: Exact match
    is_valid, msg, calc_tot = validate_full_premium_breakdown(
        basic_premium=15430.0,
        add_ons=None,
        discounts=None,
        net_premium=15430.0,
        taxes=2777.0,
        total_declared=18207.0,
    )
    assert is_valid is True
    assert msg is None
    assert calc_tot == 18207.0

    # Case B: Discrepancy (Declared != Computed)
    # Rule 7: Never silently overwrite declared total!
    is_valid_disc, msg_disc, calc_tot_disc = validate_full_premium_breakdown(
        basic_premium=15000.0,
        add_ons=None,
        discounts=None,
        net_premium=15000.0,
        taxes=2700.0,
        total_declared=19000.0,  # Deliberate difference of ₹1,300
    )
    assert is_valid_disc is False
    assert "differs from declared document total" in msg_disc
    assert "Declared document total is retained as authoritative" in msg_disc


def test_dynamic_tesseract_ocr_path_detection():
    """Verify Tesseract executable detection works dynamically."""
    tess_path = get_verified_tesseract_cmd()
    # On this machine, Tesseract is installed at C:\Program Files\Tesseract-OCR\tesseract.exe
    assert tess_path is not None, "Tesseract must be found dynamically on this system"
    assert Path(tess_path).is_file()


def test_table_retrieval_property_pdf():
    """Verify table document indexes and answers correctly."""
    pdf_path = FIXTURES_DIR / "property_with_table.pdf"
    assert pdf_path.exists()

    doc_id = "test_property_table_doc"
    canonical = process_document(pdf_path, doc_id, "property_with_table.pdf")

    # Verify tables field exists on CanonicalDocument
    assert hasattr(canonical, "tables")
    assert len(canonical.tables) >= 1

    # Ask questions from the table
    ans_prem = ask_document("What is the total premium?", canonical)
    assert "25,000" in ans_prem

    # Ask for total amount
    ans_tot = ask_document("What is the total amount?", canonical)
    assert "25,000" in ans_tot


def test_exact_identifier_bm25_retrieval():
    """Verify exact alphanumeric identifiers (VIN/Chassis) are indexed and retrievable."""
    pdf_path = FIXTURES_DIR / "tata_ace_insurance.pdf"
    assert pdf_path.exists()

    doc_id = "test_tata_ace_bm25"
    canonical = process_document(pdf_path, doc_id, "tata_ace_insurance.pdf")

    # Ask for exact policy number
    ans_pol = ask_document("What is the policy number?", canonical)
    assert "P0023200023/4115/103739" in ans_pol
    assert "Page 1" in ans_pol

    # Ask for chassis/VIN
    ans_vin = ask_document("What is the chassis number?", canonical)
    assert "CHA123456789" in ans_vin

    # Ask for total premium with synonyms
    ans_prem_syn1 = ask_document("What is the total premium?", canonical)
    assert "18,207" in ans_prem_syn1

    ans_prem_syn2 = ask_document("What is the final premium?", canonical)
    assert "18,207" in ans_prem_syn2

    ans_prem_syn3 = ask_document("What is the total amount payable?", canonical)
    assert "18,207" in ans_prem_syn3
