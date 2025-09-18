"""Configuration utilities for loading environment variables and shared constants.

This module centralizes `.env` parsing, LangSmith/OpenAI keys, dataset names,
post catalogs, and future settings so that other modules can rely on a single
source of truth in place of the original monolithic script.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final, TypedDict

from dotenv import load_dotenv


class BlogPost(TypedDict):
    """Typed representation of a LessWrong blog post configuration."""

    title: str
    url: str


REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
DOTENV_PATH: Final[Path] = REPO_ROOT / ".env"

# Environment keys managed by the application. These mirror the variables the
# original script surfaced via python-dotenv.
ENVIRONMENT_KEYS: Final[tuple[str, ...]] = (
    "LANGSMITH_TRACING",
    "LANGSMITH_API_KEY",
    "OPENAI_API_KEY",
    "LANGSMITH_PROJECT",
)


def load_environment(dotenv_path: Path | None = None) -> None:
    """Load environment variables from `.env` and promote known keys to os.environ."""

    load_dotenv(dotenv_path=dotenv_path or DOTENV_PATH)
    for key in ENVIRONMENT_KEYS:
        value = os.getenv(key)
        if value:
            os.environ[key] = value


# Ensure configuration is available as soon as the module is imported, matching
# the eager loading behavior of ``langsmith-rag.py``.
load_environment()

# ----- Application constants derived from the original script -----

# LangSmith tracing configuration
TRACE_PROJECT_NAME: Final[str] = "genai-labs-tracing-project"
TRACE_RUN_NAME: Final[str] = "RAG Bot"

# Model defaults for chat and evaluators
CHAT_MODEL_NAME: Final[str] = "gpt-4o"
CHAT_TEMPERATURE: Final[float] = 0.001

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
        "title": "Mechanistic Interpretability Quickstart Guide — Neel Nanda",
        "url": "https://www.lesswrong.com/posts/jLAvJt8wuSFySN975/mechanistic-interpretability-quickstart-guide",
    },
    {
        "title": "A Barebones Guide to Mechanistic Interpretability Prerequisites — Neel Nanda",
        "url": "https://www.lesswrong.com/posts/AaABQpuoNC8gpHf2n/a-barebones-guide-to-mechanistic-interpretability",
    },
    {
        "title": "Toy Models of Superposition — Anthropic (crosspost)",
        "url": "https://www.lesswrong.com/posts/CTh74TaWgvRiXnkS6/toy-models-of-superposition",
    },
    {
        "title": "Some Lessons Learned from Studying Indirect Object Identification in GPT-2 small — Redwood Research",
        "url": "https://www.lesswrong.com/posts/3ecs6duLmTfyra3Gp/some-lessons-learned-from-studying-indirect-object",
    },
    {
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
