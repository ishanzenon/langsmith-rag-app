"""Configuration utilities for loading environment variables and shared constants.

This module centralizes `.env` parsing, LangSmith/OpenAI keys, dataset names,
post catalogs, and future settings so that other modules can rely on a single
source of truth in place of the original monolithic script.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final, Optional, TypedDict, Any

from dotenv import load_dotenv


class BlogPost(TypedDict):
    """Typed representation of a LessWrong blog post configuration."""

    id: Optional[Any]
    title: str
    url: str


class DatasetExample(TypedDict):
    """Structure for LangSmith dataset entries."""

    inputs: dict[str, str]
    outputs: dict[str, str]


REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
DOTENV_PATH: Final[Path] = REPO_ROOT / ".env"

# Environment keys managed by the application.
ENVIRONMENT_KEYS: Final[tuple[str, ...]] = (
    "LANGSMITH_TRACING",
    "LANGSMITH_API_KEY",
    "OPENAI_API_KEY",
    "LANGSMITH_PROJECT",
    "OPENAI_API_BASE_URL",
)


def load_environment(dotenv_path: Path | None = None) -> None:
    """Load environment variables from `.env` and promote known keys to os.environ."""

    load_dotenv(dotenv_path=dotenv_path or DOTENV_PATH)
    for key in ENVIRONMENT_KEYS:
        value = os.getenv(key)
        if value:
            os.environ[key] = value


# Ensure configuration is available as soon as the module is imported.
load_environment()

# ----- Application constants -----

# LangSmith tracing configuration
TRACE_PROJECT_NAME: Final[str] = "genai-labs-tracing-project"
TRACE_RUN_NAME: Final[str] = "RAG Bot"

# Model defaults for chat and evaluators
CHAT_MODEL_NAME: Final[str] = "gpt-4o"
CHAT_TEMPERATURE: Final[float] = 0.001
CHAT_BASE_URL: Final[Optional[str]] = os.getenv("OPENAI_API_BASE_URL", None)

# Document splitting defaults
TEXT_SPLITTER_CHUNK_SIZE: Final[int] = 250
TEXT_SPLITTER_CHUNK_OVERLAP: Final[int] = 0

# Retriever configuration
RETRIEVER_TOP_K: Final[int] = 6

# Dataset and experiment metadata
LESSWRONG_DATASET_NAME: Final[str] = "LessWrong Mech Interp Blogs Q&A"
EXPERIMENT_PREFIX: Final[str] = "genai-labs-experiment"
EXPERIMENT_METADATA_VERSION: Final[str] = "LCEL context, gpt-4-0125-preview"

# Catalog of LessWrong posts consumed by the ingestion pipeline.
LESSWRONG_POSTS: Final[tuple[BlogPost, ...]] = (
    {
        "id": 1,
        "title": "Mechanistic Interpretability Quickstart Guide — Neel Nanda",
        "url": "https://www.lesswrong.com/posts/jLAvJt8wuSFySN975/mechanistic-interpretability-quickstart-guide",
    },
    {
        "id": 2,
        "title": "A Barebones Guide to Mechanistic Interpretability Prerequisites — Neel Nanda",
        "url": "https://www.lesswrong.com/posts/AaABQpuoNC8gpHf2n/a-barebones-guide-to-mechanistic-interpretability",
    },
    {
        "id": 3,
        "title": "Toy Models of Superposition — Anthropic (crosspost)",
        "url": "https://www.lesswrong.com/posts/CTh74TaWgvRiXnkS6/toy-models-of-superposition",
    },
    {
        "id": 4,
        "title": "Some Lessons Learned from Studying Indirect Object Identification in GPT-2 small — Redwood Research",
        "url": "https://www.lesswrong.com/posts/3ecs6duLmTfyra3Gp/some-lessons-learned-from-studying-indirect-object",
    },
    {
        "id": 5,
        "title": "Explaining the Transformer Circuits Framework by Example",
        "url": "https://www.lesswrong.com/posts/CJsxd8ofLjGFxkmAP/explaining-the-transformer-circuits-framework-by-example",
    },
)

# Convenience accessor mirroring the helper list in ``langsmith-rag.py``.
LESSWRONG_URLS: Final[tuple[str, ...]] = tuple(post["url"] for post in LESSWRONG_POSTS)


def get_env_var(name: str, default: str | None = None) -> str | None:
    """Fetch a managed environment variable with an optional default."""

    if name not in ENVIRONMENT_KEYS:
        return os.getenv(name, default)
    return os.environ.get(name, default)


# ----- LessWrong LangSmith dataset examples (mirrors langsmith-rag.py) --------
LESSWRONG_DATASET_EXAMPLES: Final[tuple[DatasetExample, ...]] = (
    {
        "inputs": {
            "question": (
                "In Neel Nanda’s Quickstart Guide, what is the goal of mechanistic "
                "interpretability?"
            )
        },
        "outputs": {
            "answer": (
                "To reverse-engineer trained networks—like reversing a program from "
                "its binary—to understand the internal algorithms and cognition."
            )
        },
    },
    {
        "inputs": {
            "question": (
                "According to the Quickstart Guide, what minimal setup is recommended "
                "to start practical transformer MI work?"
            )
        },
        "outputs": {
            "answer": (
                "Copy the TransformerLens demo into a Google Colab with a free GPU "
                "and experiment on a small model."
            )
        },
    },
    {
        "inputs": {
            "question": (
                "Which architecture does the Barebones Guide say you must deeply "
                "understand for MI, and which variant is most relevant?"
            )
        },
        "outputs": {
            "answer": (
                "Transformers, especially decoder-only GPT-style models like GPT-2."
            )
        },
    },
    {
        "inputs": {
            "question": (
                "Which two tensor tools does the Barebones Guide strongly recommend "
                "to avoid common PyTorch pitfalls?"
            )
        },
        "outputs": {
            "answer": "einops for reshaping and einsum for tensor multiplication."
        },
    },
    {
        "inputs": {
            "question": (
                "In ‘Toy Models of Superposition’, what is superposition and why is "
                "it useful?"
            )
        },
        "outputs": {
            "answer": (
                "Representing more features than dimensions; with sparse features this "
                "compresses information, though it introduces interference requiring "
                "nonlinear filtering."
            )
        },
    },
    {
        "inputs": {
            "question": (
                "In the toy example from ‘Toy Models of Superposition’, what changes "
                "when features become sparse?"
            )
        },
        "outputs": {
            "answer": (
                "The model stores additional features in superposition instead of just "
                "learning an orthogonal basis for the top features."
            )
        },
    },
    {
        "inputs": {
            "question": (
                "What is the IOI task Redwood studied, and how large was the circuit "
                "they found?"
            )
        },
        "outputs": {
            "answer": (
                "Choosing the correct recipient in sentences like “... gave a drink to "
                "...”; they found a 26-head attention circuit grouped into seven "
                "classes."
            )
        },
    },
    {
        "inputs": {
            "question": (
                "Name one interaction phenomenon between attention heads observed in "
                "the IOI work."
            )
        },
        "outputs": {
            "answer": (
                "Heads communicate with pointers—passing positions rather than copying "
                "content."
            )
        },
    },
    {
        "inputs": {
            "question": (
                "What does the transformer circuits framework help you do when "
                "understanding models?"
            )
        },
        "outputs": {
            "answer": (
                "Decompose a transformer into identifiable parts (circuits/effective "
                "weights) so the overall model is more tractable to analyze."
            )
        },
    },
    {
        "inputs": {
            "question": (
                "Which large multi-head circuit example is cited in the ‘Transformer "
                "Circuits Framework’ post?"
            )
        },
        "outputs": {
            "answer": (
                "The 26-head mechanism for detecting indirect objects (IOI) in GPT-2 "
                "small."
            )
        },
    },
)
# -----------------------------------------------------------------------------
