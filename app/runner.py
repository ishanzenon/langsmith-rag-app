"""Application entry point that coordinates settings, services, RAG bot, and evaluators.

Will mirror the behavior of `langsmith-rag.py` by composing the modular pieces and
triggering LangSmith evaluations with the appropriate metadata.
"""

from __future__ import annotations

from typing import Any, Dict

from langchain_core.documents import Document
from langchain_text_splitters.base import TextSplitter

from . import ingestion, sources
from .services import Services, build_services


def run_ingestion() -> Dict[str, Any]:
    """Replicate the ingestion pipeline from the original script."""

    raw_documents: tuple[Document, ...] = ingestion.load_documents(
        collection=sources.COLLECTION_BLOG_POSTS,
        tags=("blog",),
    )
    splitter: TextSplitter = ingestion.build_text_splitter()
    document_chunks: tuple[Document, ...] = ingestion.chunk_documents(
        raw_documents,
        splitter=splitter,
    )
    return {
        "raw_documents": raw_documents,
        "splitter": splitter,
        "document_chunks": document_chunks,
    }


def main() -> Dict[str, Any]:
    """Bootstrap shared services ready for downstream orchestration."""

    services: Services = build_services()
    ingestion_state = run_ingestion()
    ingestion_state["services"] = services
    return ingestion_state


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
