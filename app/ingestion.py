"""Document loading and splitting utilities for preparing retrieval inputs.

This module coordinates fetching raw sources, flattening loader outputs, and
chunking documents so that the vector store receives ready-to-index data.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters.base import TextSplitter

from app import settings, sources
from app.sources import DocumentSource

__all__ = [
    "build_text_splitter",
    "load_documents",
    "chunk_documents",
    "ingest_documents",
]


def build_text_splitter(
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> RecursiveCharacterTextSplitter:
    """Create the default text splitter used by the ingestion pipeline."""

    resolved_chunk_size = (
        chunk_size if chunk_size is not None else settings.TEXT_SPLITTER_CHUNK_SIZE
    )
    resolved_chunk_overlap = (
        chunk_overlap
        if chunk_overlap is not None
        else settings.TEXT_SPLITTER_CHUNK_OVERLAP
    )
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=resolved_chunk_size,
        chunk_overlap=resolved_chunk_overlap,
    )


def load_documents(
    *,
    collection: str | None = None,
    tags: Iterable[str] | None = None,
    enrich_metadata: bool = True,
    sources_override: Iterable[DocumentSource] | None = None,
) -> tuple[Document, ...]:
    """Load documents from the configured sources.

    Parameters
    ----------
    collection:
        Optional collection name used to filter the source registry.
    tags:
        Optional iterable of tags; loaded sources must contain *all* requested
        tags.
    enrich_metadata:
        When ``True`` (default) merge provenance metadata onto each document.
    sources_override:
        Explicit list of sources to load instead of pulling from the registry.
    """

    selected_sources: tuple[DocumentSource, ...]
    if sources_override is not None:
        selected_sources = tuple(sources_override)
    else:
        selected_sources = tuple(sources.iter_sources(collection=collection, tags=tags))

    documents: list[Document] = []
    for source in selected_sources:
        # Delegate loader specifics to the DocumentSource definition.
        documents.extend(source.load(enrich_metadata=enrich_metadata))
    return tuple(documents)


def chunk_documents(
    documents: Sequence[Document],
    *,
    splitter: TextSplitter | None = None,
) -> tuple[Document, ...]:
    """Split documents into retrieval-friendly chunks."""

    if not documents:
        return tuple()

    active_splitter = splitter or build_text_splitter()
    return tuple(active_splitter.split_documents(list(documents)))


def ingest_documents(
    *,
    splitter: TextSplitter | None = None,
    collection: str | None = None,
    tags: Iterable[str] | None = None,
    enrich_metadata: bool = True,
    sources_override: Iterable[DocumentSource] | None = None,
) -> tuple[Document, ...]:
    """End-to-end ingestion: load sources then split into chunks."""

    documents = load_documents(
        collection=collection,
        tags=tags,
        enrich_metadata=enrich_metadata,
        sources_override=sources_override,
    )
    return chunk_documents(documents, splitter=splitter)
