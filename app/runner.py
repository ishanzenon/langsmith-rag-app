"""Application entry point that coordinates settings, services, RAG bot, and evaluators.

Will mirror the behavior of `langsmith-rag.py` by composing the modular pieces and
triggering LangSmith evaluations with the appropriate metadata.
"""

from __future__ import annotations

from typing import Any, Dict

from langchain_core.documents import Document
from langchain_text_splitters.base import TextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.vectorstores.base import VectorStoreRetriever

from . import ingestion, sources, vectorstore
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


def build_vectorstore_state(
    *,
    documents: tuple[Document, ...],
    services: Services,
    retriever_k: int | None = None,
) -> Dict[str, Any]:
    """Create the vector store and retriever mirroring ``langsmith-rag.py``."""

    vector_store: InMemoryVectorStore = vectorstore.build_vectorstore(
        documents,
        embeddings=services.embeddings,
    )
    retriever: VectorStoreRetriever = vectorstore.get_retriever(
        vector_store,
        k=retriever_k,
    )
    return {
        "vector_store": vector_store,
        "retriever": retriever,
    }


def main() -> Dict[str, Any]:
    """Bootstrap shared services ready for downstream orchestration."""

    services: Services = build_services()
    ingestion_state = run_ingestion()
    ingestion_state["services"] = services
    vector_state = build_vectorstore_state(
        documents=ingestion_state["document_chunks"],
        services=services,
    )
    ingestion_state.update(vector_state)
    return ingestion_state


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
