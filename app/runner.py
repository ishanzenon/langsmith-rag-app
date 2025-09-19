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

from . import datasets, ingestion, sources, vectorstore
from .rag import build_rag_bot
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


def build_rag_state(*, retriever: VectorStoreRetriever, services: Services) -> Dict[str, Any]:
    """Instantiate the RAG bot using the provided retriever and chat model."""

    rag_bot = build_rag_bot(retriever=retriever, chat_model=services.chat_llm)
    return {"rag_bot": rag_bot}


def main() -> Dict[str, Any]:
    """Bootstrap shared services ready for downstream orchestration."""

    services: Services = build_services()
    dataset_state = datasets.ensure_default_dataset(services)
    pipeline_state = run_ingestion()
    pipeline_state["services"] = services
    pipeline_state["dataset_state"] = dataset_state
    vector_state = build_vectorstore_state(
        documents=pipeline_state["document_chunks"],
        services=services,
    )
    rag_state = build_rag_state(
        retriever=vector_state["retriever"],
        services=services,
    )
    pipeline_state.update(vector_state)
    pipeline_state.update(rag_state)
    return pipeline_state


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
