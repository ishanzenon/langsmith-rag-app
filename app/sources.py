"""Structured source catalogue for documents consumed by the RAG pipeline.

Usage pattern for ingestion or any downstream module:

```python
from app import sources

all_docs = sources.load_sources(
    sources.iter_sources(collection="blog_posts")
)
```

Ingestion stays loader-agnostic: iterate, load, and receive LangChain
``Document`` instances that already carry provenance metadata. This module keeps
loader construction and source registration centralized so the rest of the
application only deals with the standard LangChain interfaces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Iterator, Mapping, Sequence, Tuple

from langchain_core.document_loaders.base import BaseLoader
from langchain_core.documents import Document

from langchain_community.document_loaders import WebBaseLoader

from . import settings


LoaderFactory = Callable[[], BaseLoader]


@dataclass(frozen=True, slots=True)
class DocumentSource:
    """Domain-agnostic description of a document source.

    Attributes
    ----------
    id:
        Stable identifier used by downstream systems for provenance and tracing.
    title:
        Human-friendly label for logging and experimentation dashboards.
    loader_factory:
        Callable returning a LangChain loader capable of fetching this source's
        documents.
    collection:
        Logical grouping (e.g., ``"blog_posts"`` or ``"product_docs"``) useful
        for bulk filtering.
    tags:
        Arbitrary descriptors that allow building subsets (e.g., ``("blog",
        "mechanistic-interpretability")``).
    metadata:
        Key/value pairs merged into each ``Document``'s metadata when loading,
        ensuring retrievers, evaluators, and tracing runs have consistent
        context.
    """

    id: str
    title: str
    loader_factory: LoaderFactory
    collection: str = "default"
    tags: Tuple[str, ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def create_loader(self) -> BaseLoader:
        """Instantiate the underlying LangChain loader."""

        return self.loader_factory()

    def load(self, *, enrich_metadata: bool = True) -> Sequence[Document]:
        """Load documents using the configured loader.

        Parameters
        ----------
        enrich_metadata:
            When ``True`` (default), merge provenance details onto each
            ``Document`` returned by the loader.
        """

        loader = self.create_loader()
        documents = list(loader.load())
        if not enrich_metadata:
            return tuple(documents)

        base_metadata: Dict[str, Any] = {
            "source_id": self.id,
            "source_title": self.title,
            "source_collection": self.collection,
        }
        if self.tags:
            base_metadata["source_tags"] = list(self.tags)
        if self.metadata:
            base_metadata.update(dict(self.metadata))

        enriched: list[Document] = []
        for doc in documents:
            merged_metadata: Dict[str, Any] = {
                **doc.metadata,
                **base_metadata,
            }
            enriched.append(
                Document(page_content=doc.page_content, metadata=merged_metadata)
            )
        return tuple(enriched)


# ----- Internal source registry construction ---------------------------------

COLLECTION_BLOG_POSTS = "blog_posts"


def _build_registry() -> Dict[str, DocumentSource]:
    """Create the default source registry from settings-provided catalogs."""

    registry: Dict[str, DocumentSource] = {}
    for post in settings.LESSWRONG_POSTS:
        source = DocumentSource(
            id=str(post["id"]),
            title=post["title"],
            collection=COLLECTION_BLOG_POSTS,
            tags=("blog",),
            metadata={
                "source_url": post["url"],
            },
            loader_factory=lambda url=post["url"]: WebBaseLoader(url),
        )
        registry[source.id] = source
    return registry


_SOURCE_REGISTRY: Dict[str, DocumentSource] = _build_registry()


# ----- Public API -------------------------------------------------------------

def get_source(source_id: str) -> DocumentSource:
    """Return a source by identifier.

    Raises
    ------
    KeyError
        If the source identifier is unknown.
    """

    return _SOURCE_REGISTRY[source_id]


def iter_sources(
    *,
    collection: str | None = None,
    tags: Iterable[str] | None = None,
) -> Iterator[DocumentSource]:
    """Iterate over sources, optionally filtering by collection and tags."""

    required_tags = tuple(tags or ())

    for source in _SOURCE_REGISTRY.values():
        if collection and source.collection != collection:
            continue
        if required_tags and not set(required_tags).issubset(set(source.tags)):
            continue
        yield source


def all_sources() -> Sequence[DocumentSource]:
    """Return an immutable snapshot of all known document sources."""

    return tuple(_SOURCE_REGISTRY.values())


def load_sources(
    sources: Iterable[DocumentSource], *, enrich_metadata: bool = True
) -> Sequence[Document]:
    """Helper to load and concatenate documents from multiple sources."""

    documents: list[Document] = []
    for source in sources:
        documents.extend(source.load(enrich_metadata=enrich_metadata))
    return tuple(documents)
