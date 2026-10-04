# Multimodal Repair Assistant — Codex Instructions

## Project Goal

Build a production-quality multimodal Retrieval-Augmented Generation application.

The first demonstrated domain is bicycle repair, but the architecture must remain domain-agnostic so other technical-document collections can be indexed later.

A user should eventually be able to:
1. Upload an image of a damaged or malfunctioning component.
2. Enter a natural-language question describing the problem.
3. Search a private collection of technical manuals containing text, diagrams, and images.
4. Retrieve the most relevant supporting evidence.
5. Generate a grounded answer using only retrieved evidence.
6. Display document-level and page-level citations.
7. Display relevant diagrams or page images.
8. Return an insufficient-evidence response when the knowledge base cannot support an answer.

## Core Architecture

User → React frontend → FastAPI backend → multimodal retrieval service → Qdrant vector database → grounded generation service → cited response

## Technology Stack

Backend: Python, FastAPI, Pydantic  
Document ingestion: PyMuPDF, Pillow  
Retrieval: Sentence Transformers, CLIP-compatible multimodal embeddings, Qdrant  
Frontend: React  
Testing and quality: pytest, Ruff, GitHub Actions  
Infrastructure later: Docker and cloud deployment

## Engineering Rules

- Keep ingestion, retrieval, generation, API, and shared utilities separate.
- Prefer small reusable functions and use type hints for public functions.
- Use pathlib.Path for filesystem paths.
- Validate API inputs with Pydantic.
- Never silently ignore failures.
- Never hard-code credentials; use environment variables.
- Add tests for important behavior.
- Keep generated files, vector stores, and source manuals out of Git unless redistribution is explicitly allowed.
- Avoid unnecessary dependencies.
- Keep README setup and architecture documentation current.
- Preserve existing functionality when implementing new milestones.

## ML / RAG Rules

Retrieval quality must be validated independently of answer generation.

Do not add an LLM until basic semantic retrieval works.

Preserve metadata for every indexed item:
- source document
- 1-indexed page number
- chunk ID
- content type
- original location

Answers must be grounded in retrieved material. Support an `insufficient evidence` response rather than inventing information. Do not claim model quality without measured evaluation.

## Development Plan

### Phase 1 — Document ingestion
Read PDFs from `data/documents/`, extract page text, render page images, preserve source/page metadata, write structured artifacts to `data/extracted/`, and add tests + CLI execution.

### Phase 2 — Text retrieval
Chunk text, generate embeddings, store vectors/metadata in Qdrant, retrieve top-k chunks, and build a small retrieval benchmark.

### Phase 3 — Multimodal retrieval
Embed page images/diagrams, support image queries, and combine image + text retrieval.

### Phase 4 — RAG generation
Generate answers only from retrieved context, add page-level citations, and add insufficient-evidence behavior.

### Phase 5 — API and frontend
Build FastAPI backend and React UI with image upload, question input, source previews, and error handling.

### Phase 6 — Evaluation
Measure Recall@K, MRR, nDCG, citation accuracy, and latency with a labeled evaluation set.

### Phase 7 — Production engineering
Add Docker, CI/CD, structured logging, monitoring, and cloud deployment.

## Workflow for Codex

Before implementing a task:
1. Read this file.
2. Inspect the repository.
3. Briefly explain the planned change.
4. Implement only the requested milestone.
5. Preserve existing functionality.
6. Run relevant tests and quality checks.
7. Update documentation when setup or architecture changes.
8. Report files changed, commands run, test results, and remaining limitations.

Do not implement future phases unless explicitly requested.

## Current Milestone

The next implementation milestone is **Phase 1: PDF document ingestion**.

Do not add embeddings, Qdrant indexing, an LLM, FastAPI routes, React functionality, or deployment infrastructure as part of Phase 1 unless explicitly requested.
