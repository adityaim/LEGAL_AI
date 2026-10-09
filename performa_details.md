# Student Contribution Verification Table: Details to Fill

Based on the established group division and avoiding overlapping generic terms, here is exactly what each person should fill in the performa.

### Person 1 (Input Pipeline: OCR + HTR)
*   **Primary Responsibility/tasks undertaken by the student:** 
    Computer Vision Pipeline, Document Layout Analysis Algorithm, ML Model Integration (PaddleOCR, TrOCR), and Image Pre-processing (Deskewing, Denoising).
*   **Evidence Submitted (List):**
    *   Code: `modules/ocr/` and `modules/htr/` (Image preprocessor, heuristic layout detector, and engine adapters).
    *   Research Papers: PaddleOCR, TrOCR, and LayoutParser architectures.
    *   Live Prototype / Demo: `scripts/verify_ocr.py` and `scripts/verify_htr.py` (demonstrating confidence scoring and bounding box extraction on scanned/handwritten documents).

### Person 2 (Audio & Fusion: ASR + Text Fusion)
*   **Primary Responsibility/tasks undertaken by the student:** 
    Speech-to-Text Model Integration (Whisper, faster-whisper), Audio Normalization Pipeline, Cross-Modality Data Deduplication Algorithm (Jaccard similarity), and Text Fusion Logic.
*   **Evidence Submitted (List):**
    *   Code: `modules/asr/` and `modules/fusion/` (Audio resampler, Whisper engine adapter, TextFusionProcessor).
    *   Research Papers: Whisper ASR and Multimodal Fusion techniques.
    *   Live Prototype / Demo: `scripts/verify_asr.py` and `scripts/verify_fusion.py` (demonstrating audio transcription and merging tagged text with `[REVIEW]` markers).

### Person 3 (Retrieval & Reasoning: RAG + LLM)
*   **Primary Responsibility/tasks undertaken by the student:** 
    Information Retrieval Algorithm (Hybrid RAG), Vector Store Integration, Embedding Models (Sentence Transformers), Clause-aware Text Chunking, and LLM Prompt Engineering.
*   **Evidence Submitted (List):**
    *   Code: `modules/rag/` and `modules/llm/` (SentenceAwareChunker, RAGRetriever with MMR reranking, LegalReasoner with citation extraction).
    *   Research Papers: Retrieval-Augmented Generation, Dense Passage Retrieval, and Reciprocal Rank Fusion.
    *   Live Prototype / Demo: `scripts/verify_rag.py` and `scripts/verify_llm.py` (demonstrating context retrieval and grounded legal QA generation).

### Person 4 (Output, API & Infrastructure)
*   **Primary Responsibility/tasks undertaken by the student:** 
    Backend / API Development (FastAPI), System Architecture Integration, Docker Containerization, Response Formatting Engine, and Component Orchestration.
*   **Evidence Submitted (List):**
    *   Code: `api/` (All REST routers and main FastAPI app), `modules/response/` (ResponseFormatter, CitationBuilder), and `Dockerfile.ocr`.
    *   Architecture Diagram/Data flows: System Architecture overview (`WALKTHROUGH.md`).
    *   Live Prototype / Demo: `scripts/verify_response.py` and running the full API server (`uvicorn api.main:app`).
