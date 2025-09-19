"""Factories for shared LangSmith and LangChain service clients.

Holds construction logic for objects such as the LangSmith `Client`, `ChatOpenAI`,
and `OpenAIEmbeddings` so the rest of the application can pull from a unified
service layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypeVar

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langsmith import Client
from langchain_core.runnables.base import Runnable

from . import settings

__all__ = ["Services", "build_services"]

SchemaT = TypeVar("SchemaT")


@dataclass(frozen=True)
class StructuredChatFactory:
    """Callable wrapper around ``ChatOpenAI.with_structured_output`` defaults."""

    _chat_model: ChatOpenAI
    _default_method: Literal["function_calling", "json_mode", "json_schema"] = (
        "json_schema"
    )
    _default_strict: bool = True

    def __call__(
        self,
        schema: type[SchemaT],
        /,
        *,
        method: Literal["function_calling", "json_mode", "json_schema"] = "json_schema",
        strict: bool | None = None,
        **kwargs: Any,
    ) -> Runnable[Any, SchemaT | dict[str, Any]]:
        """Return a structured-output variant of the base chat model."""

        return self._chat_model.with_structured_output(
            schema,
            method=method or self._default_method,
            strict=self._default_strict if strict is None else strict,
            **kwargs,
        )


@dataclass(frozen=True)
class Services:
    """Container for cross-module service singletons."""

    client: Client
    chat_llm: ChatOpenAI
    structured_chat_llm: StructuredChatFactory
    embeddings: OpenAIEmbeddings


def _build_langsmith_client() -> Client:
    """Create the LangSmith client using environment-backed configuration."""

    return Client()


def _build_chat_model() -> ChatOpenAI:
    """Provision the chat model used by the core RAG bot."""

    return ChatOpenAI(
        model=settings.CHAT_MODEL_NAME,
        temperature=settings.CHAT_TEMPERATURE,
    )


def _build_structured_chat_model(chat_model: ChatOpenAI) -> StructuredChatFactory:
    """Expose a default-parameter structured output factory."""

    return StructuredChatFactory(chat_model)


def _build_embeddings() -> OpenAIEmbeddings:
    """Provision the embeddings model consumed by the vector store."""

    return OpenAIEmbeddings()


def build_services() -> Services:
    """Instantiate the shared service clients for application use."""

    chat_model = _build_chat_model()
    return Services(
        client=_build_langsmith_client(),
        chat_llm=chat_model,
        structured_chat_llm=_build_structured_chat_model(chat_model),
        embeddings=_build_embeddings(),
    )
