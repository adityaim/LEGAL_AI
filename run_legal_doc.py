#!/usr/bin/env python3
"""
run_legal_doc.py
------------------
Minimal end-to-end script to run the full LegalAI pipeline on a legal document.

Pipeline:
  PDF/Image  ->  OCR  ->  Fusion  ->  RAG  ->  LLM  ->  Formatted Response

Usage:
    # Demo mode (no file needed, no API key needed)
    python run_legal_doc.py --demo

    # With a real document
    python run_legal_doc.py --input contract.pdf --query "What are the payment terms?"

    # With Gemini for real LLM reasoning (requires GEMINI_API_KEY env var)
    python run_legal_doc.py --input contract.pdf --query "Who are the parties?" --gemini

    # Save the formatted Markdown response to a file
    python run_legal_doc.py --demo --output result.md
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent))

# Pipeline module imports
from modules.ocr import OCRDocumentReader
from modules.fusion import TextFusionProcessor
from modules.rag import RAGPipeline
from modules.llm import LegalReasoner
from modules.llm.schema import LegalQuery
from modules.response import ResponseFormatter


def _banner(title):
    print("\n" + "-" * 60)
    print("  " + title)
    print("-" * 60)


def _make_demo_pdf():
    """Create a small synthetic legal PDF using PyMuPDF and return its path."""
    import tempfile
    import io
    import fitz  # PyMuPDF

    doc = fitz.open()
    page = doc.new_page(width=595, height=842)

    paragraphs = [
        ("SERVICE AGREEMENT", 72, 80, 18),
        ("This Service Agreement is entered into on 1 January 2025", 72, 120, 11),
        ("between Alpha Technologies Pvt. Ltd. ('Client') and", 72, 136, 11),
        ("Beta Legal Services LLP ('Service Provider').", 72, 152, 11),
        ("1. Services", 72, 185, 13),
        ("The Service Provider shall render legal advisory services", 72, 205, 11),
        ("as described in Schedule A.", 72, 219, 11),
        ("2. Payment Terms", 72, 250, 13),
        ("The Client shall pay INR 50,000 per month within 15 days of invoice.", 72, 270, 11),
        ("Late payment attracts interest at 18% per annum under Section 34 CPC.", 72, 284, 11),
        ("3. Term", 72, 315, 13),
        ("This Agreement shall remain valid for 12 months from the execution date.", 72, 335, 11),
        ("Either party may terminate with 30 days written notice.", 72, 349, 11),
        ("4. Governing Law", 72, 380, 13),
        ("This Agreement is governed by the laws of India.", 72, 400, 11),
        ("Disputes shall be resolved under the Arbitration and Conciliation Act, 1996.", 72, 414, 11),
        ("Jurisdiction: Courts of Delhi.", 72, 428, 11),
    ]

    for text, x, y, size in paragraphs:
        page.insert_text((x, y), text, fontsize=size)

    buf = io.BytesIO()
    doc.save(buf)
    doc.close()

    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp.write(buf.getvalue())
    tmp.close()
    return Path(tmp.name)


def run_pipeline(
    input_path,
    query,
    use_gemini=False,
    output_file=None,
    demo_mode=False,
):
    """
    Run the full 5-step Legal AI pipeline.

    Steps:
        1. OCR      - extract text from the PDF/image
        2. Fusion   - merge into a FusedDocument
        3. RAG      - ingest + retrieve relevant chunks
        4. LLM      - grounded legal reasoning
        5. Response - format the answer as Markdown
    """

    demo_pdf_path = None

    # ---- Step 1: OCR ---------------------------------------------------------
    _banner("Step 1 -- OCR: Extracting text from document")

    if demo_mode:
        demo_pdf_path = _make_demo_pdf()
        input_path = str(demo_pdf_path)
        print("  [demo] Synthetic legal PDF created: " + input_path)

    print("  Input : " + str(input_path))
    reader = OCRDocumentReader()
    ocr_result = reader.process(input_path)

    print("  Pages      : " + str(ocr_result.pages))
    print("  Blocks     : " + str(len(ocr_result.blocks)))
    print("  Confidence : " + str(round(ocr_result.mean_confidence(), 1)) + "%")
    print("  Doc type   : " + str(ocr_result.metadata.doc_type or "(unknown)"))
    parties = ", ".join(ocr_result.metadata.parties) if ocr_result.metadata.parties else "(none)"
    print("  Parties    : " + parties)

    # ---- Step 2: Fusion ------------------------------------------------------
    _banner("Step 2 -- Fusion: Merging modalities")

    fuser = TextFusionProcessor()
    fused_doc = fuser.fuse(
        ocr=ocr_result,
        # htr=htr_result,   # uncomment if you also have a HandwritingReader result
        # asr=asr_result,   # uncomment if you also have an ASRReader result
        typed_text=query,   # the user's natural-language question
    )
    sources = [m.source for m in fused_doc.modalities]
    print("  Modalities  : " + str(sources))
    print("  Total words : " + str(fused_doc.total_word_count))
    print("  Intent      : " + str(fused_doc.detected_intent))
    print("  Review flags: " + str(len(fused_doc.review_flags)))

    # ---- Step 3: RAG ---------------------------------------------------------
    _banner("Step 3 -- RAG: Ingesting and retrieving relevant chunks")

    rag = RAGPipeline()       # uses MockEmbedder + MockVectorStore by default
    rag.ingest(fused_doc)

    retrieval = rag.query(query, top_k=5)
    print("  Chunks searched : " + str(retrieval.total_chunks_searched))
    print("  Chunks returned : " + str(len(retrieval.chunks)))
    for i, chunk in enumerate(retrieval.chunks[:3], 1):
        preview = chunk.text[:70].replace("\n", " ")
        print("    [" + str(i) + "] score=" + str(round(chunk.score, 3)) + "  " + preview + "...")

    # ---- Step 4: LLM Legal Reasoning -----------------------------------------
    _banner("Step 4 -- LLM: Legal reasoning")

    if use_gemini:
        from modules.llm.engines.gemini_engine import GeminiEngine
        engine = GeminiEngine()
        print("  Engine: Gemini (real API call)")
    else:
        engine = None   # defaults to MockLLMEngine; no API key required
        print("  Engine: Mock (use --gemini for real Gemini responses)")

    reasoner = LegalReasoner(engine=engine)

    legal_query = LegalQuery(
        query=query,
        fused_document_text=fused_doc.for_llm(),
        retrieval_context=retrieval.context_for_llm,
        language=fused_doc.primary_language,
        detected_intent=fused_doc.detected_intent,
    )
    answer = reasoner.reason(legal_query)

    print("  Confidence : " + str(round(answer.confidence, 1)) + "%")
    print("  Grounded   : " + str(answer.is_grounded))
    print("  Citations  : " + str(len(answer.citations)))
    preview = answer.answer_text[:300].replace("\n", " ")
    print("\n  Answer preview:")
    print("  " + preview + "...")

    # ---- Step 5: Response Formatting ------------------------------------------
    _banner("Step 5 -- Response: Formatting output")

    formatter = ResponseFormatter()
    response = formatter.format(answer, fmt="markdown")
    markdown_output = response.to_markdown()

    print("  Format: Markdown (" + str(len(markdown_output)) + " chars)")

    if output_file:
        Path(output_file).write_text(markdown_output, encoding="utf-8")
        print("  Saved to: " + str(output_file))

    # Cleanup temporary demo PDF
    if demo_pdf_path:
        demo_pdf_path.unlink(missing_ok=True)

    return markdown_output


def main():
    parser = argparse.ArgumentParser(
        description="Run the full LegalAI pipeline on a legal document.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", "-i", help="Path to a PDF or image legal document")
    parser.add_argument(
        "--query", "-q",
        default="What are the key terms and obligations in this document?",
        help="Legal question to answer (default: summary question)",
    )
    parser.add_argument(
        "--gemini", action="store_true",
        help="Use Gemini API for LLM reasoning (requires GEMINI_API_KEY env var)",
    )
    parser.add_argument("--output", "-o", help="Save Markdown response to this file")
    parser.add_argument(
        "--demo", action="store_true",
        help="Run with a synthetic demo document (no file needed, no API key needed)",
    )
    args = parser.parse_args()

    if not args.demo and not args.input:
        print("Error: provide --input <file> or use --demo")
        parser.print_help()
        sys.exit(1)

    if args.gemini and not os.environ.get("GEMINI_API_KEY"):
        print("Error: --gemini requires the GEMINI_API_KEY environment variable.")
        sys.exit(1)

    print("\n" + "=" * 52)
    print("  LegalAI -- Multimodal Legal Assistant")
    print("=" * 52)
    print("  Query: " + args.query)

    result_md = run_pipeline(
        input_path=args.input,
        query=args.query,
        use_gemini=args.gemini,
        output_file=args.output,
        demo_mode=args.demo,
    )

    print("\n" + "=" * 60)
    print("  FINAL RESPONSE (Markdown)")
    print("=" * 60)
    print(result_md)
    print("=" * 60)
    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
