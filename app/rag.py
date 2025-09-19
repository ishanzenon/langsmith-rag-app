"""RAG bot orchestration and LangSmith tracing utilities.

Encapsulates prompt formatting, retriever interaction, and chat model invocation
so the rest of the application gets consistent answers and traced executions.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol, TypedDict

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.vectorstores.base import VectorStoreRetriever
from langsmith import traceable

from . import settings

__all__ = ["RAGResponse", "RAGCallable", "build_rag_bot"]


class RAGResponse(TypedDict):
    """Response payload returned by the RAG bot."""

    answer: str | list[str | dict] | list[Document]
    documents: Sequence[Document]


class RAGCallable(Protocol):
    """Callable signature for the RAG bot entrypoint."""

    def __call__(self, question: str) -> RAGResponse:  # pragma: no cover - protocol
        ...


def _build_instructions(docs: Sequence[Document]) -> str:
    """Mirror the instruction prompt used by the original script."""

    docs_string = "".join(doc.page_content for doc in docs)
    return (
        "You are a helpful assistant who is good at analyzing source information and "
        "answering questions.\n"
        "Use the following source documents to answer the user's questions.\n"
        "If you don't know the answer, just say that you don't know.\n"
        "Use three sentences maximum and keep the answer concise.\n\n"
        "Documents:\n"
        f"{docs_string}"
    )


def build_rag_bot(
    *,
    retriever: VectorStoreRetriever,
    chat_model: BaseChatModel,
) -> RAGCallable:
    """Create a traceable RAG bot mirroring ``langsmith-rag.py`` behavior."""

    @traceable(project_name=settings.TRACE_PROJECT_NAME, name=settings.TRACE_RUN_NAME)
    def rag_bot(question: str) -> RAGResponse:
        docs = retriever.invoke(question)
        instructions = _build_instructions(docs)
        ai_msg = chat_model.invoke(
            [
                {"role": "system", "content": instructions},
                {"role": "user", "content": question},
            ]
        )
        return {"answer": ai_msg.content, "documents": docs}

    return rag_bot
