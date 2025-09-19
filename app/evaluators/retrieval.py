"""Retrieval relevance evaluator built on the shared boolean scaffolding."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from langchain_core.documents import Document
from typing_extensions import Annotated, TypedDict

from app.services import Services
from app.evaluators.base import (
    BooleanEvaluator,
    BooleanEvaluatorSpec,
    build_boolean_evaluator,
)

__all__ = [
    "build_retrieval_relevance_evaluator",
    "RetrievalRelevanceGrade",
    "retrieval_relevance_instructions",
]


class RetrievalRelevanceGrade(TypedDict):
    """Schema emitted by the retrieval relevance grader LLM."""

    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    relevant: Annotated[
        bool,
        ...,
        "True if the retrieved documents are relevant to the question, False otherwise",
    ]


retrieval_relevance_instructions = """You are a teacher grading a quiz. You will be given a QUESTION and a set of FACTS provided by the student. Here is the grade criteria to follow:
(1) You goal is to identify FACTS that are completely unrelated to the QUESTION
(2) If the facts contain ANY keywords or semantic meaning related to the question, consider them relevant
(3) It is OK if the facts have SOME information that is unrelated to the question as long as (2) is met

Relevance:
A relevance value of True means that the FACTS contain ANY keywords or semantic meaning related to the QUESTION and are therefore relevant.
A relevance value of False means that the FACTS are completely unrelated to the QUESTION.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""


def _extract_documents(outputs: Mapping[str, Any]) -> Sequence[Document]:
    documents = outputs.get("documents")
    if not isinstance(documents, Sequence):
        raise ValueError("Expected 'documents' sequence in evaluator outputs")
    return documents  # type: ignore[return-value]


def _build_retrieval_prompt(
    inputs: Mapping[str, Any],
    outputs: Mapping[str, Any],
    reference_outputs: Optional[Mapping[str, Any]] = None,
) -> str:
    documents = _extract_documents(outputs)
    doc_string = "\n\n".join(doc.page_content for doc in documents)
    return "FACTS: " f"{doc_string}\n" "QUESTION: " f"{inputs['question']}"


def build_retrieval_relevance_evaluator(*, services: Services) -> BooleanEvaluator:
    """Create the retrieval relevance evaluator bound to shared services."""

    spec = BooleanEvaluatorSpec(
        schema=RetrievalRelevanceGrade,
        instructions=retrieval_relevance_instructions,
        result_key="relevant",
        build_user_message=_build_retrieval_prompt,
    )
    return build_boolean_evaluator(
        structured_chat_factory=services.structured_chat_llm,
        spec=spec,
    )
