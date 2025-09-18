"""Application entry point that coordinates settings, services, RAG bot, and evaluators.

Will mirror the behavior of `langsmith-rag.py` by composing the modular pieces and
triggering LangSmith evaluations with the appropriate metadata.
"""

from __future__ import annotations

from . import settings
from .services import Services, build_services


def main():
    """Bootstrap shared services ready for downstream orchestration."""
    
    services = build_services()


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
