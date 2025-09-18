"""Factories for shared LangSmith and LangChain service clients.

Holds construction logic for objects such as the LangSmith `Client`, `ChatOpenAI`,
and `OpenAIEmbeddings` so the rest of the application can pull from a unified
service layer.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langsmith import Client

from . import settings

__all__ = ["Services", "build_services"]


@dataclass(frozen=True)
class Services:
    """Container for cross-module service singletons."""

    client: Client
    chat_llm: ChatOpenAI
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


def _build_embeddings() -> OpenAIEmbeddings:
    """Provision the embeddings model consumed by the vector store."""

    return OpenAIEmbeddings()


def build_services() -> Services:
    """Instantiate the shared service clients for application use."""

    return Services(
        client=_build_langsmith_client(),
        chat_llm=_build_chat_model(),
        embeddings=_build_embeddings(),
    )
