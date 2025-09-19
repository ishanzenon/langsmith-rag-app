"""Dataset lifecycle management for LangSmith evaluations.

Mirrors the dataset setup logic from ``langsmith-rag.py`` by ensuring the
LessWrong evaluation dataset exists inside LangSmith and registering the
reference examples when missing. Future orchestration code (e.g.,
``app.runner``) can call the public helpers here to guarantee the dataset is in
place before running evaluations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from langsmith import Client

from app import settings
from app.services import Services

DatasetExample = settings.DatasetExample

__all__ = ["DatasetExample", "DatasetState", "ensure_dataset", "ensure_default_dataset"]




@dataclass(frozen=True)
class DatasetState:
    """Return payload describing the ensured dataset."""

    dataset: Any
    created: bool
    example_count: int


def _materialize_examples(examples: Sequence[DatasetExample]) -> list[dict[str, Any]]:
    """Convert typed examples into plain dictionaries for the LangSmith client."""

    payload: list[dict[str, Any]] = []
    for example in examples:
        payload.append(
            {
                "inputs": dict(example["inputs"]),
                "outputs": dict(example["outputs"]),
            }
        )
    return payload


def ensure_dataset(
    *,
    client: Client,
    dataset_name: str,
    examples: Sequence[DatasetExample],
) -> DatasetState:
    """Ensure a LangSmith dataset exists, creating it and its examples if needed."""

    if client.has_dataset(dataset_name=dataset_name):
        dataset = client.read_dataset(dataset_name=dataset_name)
        return DatasetState(dataset=dataset, created=False, example_count=0)

    dataset = client.create_dataset(dataset_name=dataset_name)
    created_examples = client.create_examples(
        dataset_id=dataset.id,
        examples=_materialize_examples(examples),
    )
    example_count = len(created_examples) if created_examples is not None else 0
    return DatasetState(
        dataset=dataset,
        created=True,
        example_count=example_count,
    )


def ensure_default_dataset(services: Services) -> DatasetState:
    """Ensure the default evaluation dataset configured in settings exists."""

    return ensure_dataset(
        client=services.client,
        dataset_name=settings.LESSWRONG_DATASET_NAME,
        examples=settings.LESSWRONG_DATASET_EXAMPLES,
    )
