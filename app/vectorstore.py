"""Vector store construction and retrieval helpers.

Mirrors the vector store logic from ``langsmith-rag.py`` by taking pre-split
documents, indexing them with ``InMemoryVectorStore``, and exposing a retriever
configured with the default ``k`` from settings unless overridden.
"""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.vectorstores.base import VectorStoreRetriever

from app import settings

__all__ = ["build_vectorstore", "get_retriever"]


def build_vectorstore(
    documents: Sequence[Document],
    *,
    embeddings: Embeddings,
) -> InMemoryVectorStore:
    """Create the in-memory vector store from prepared document chunks."""

    # ``from_documents`` matches the construction used in ``langsmith-rag.py``.
    return InMemoryVectorStore.from_documents(
        documents=list(documents),
        embedding=embeddings,
    )


def get_retriever(
    vectorstore: InMemoryVectorStore,
    *,
    k: int | None = None,
) -> VectorStoreRetriever:
    """Return a retriever backed by the vector store with the configured ``k``."""

    top_k = k if k is not None else settings.RETRIEVER_TOP_K
    return vectorstore.as_retriever(k=top_k)
