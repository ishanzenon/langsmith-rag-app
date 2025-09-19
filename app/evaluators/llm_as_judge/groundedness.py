"""Groundedness evaluator built on the shared LLM judge scaffolding."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from langchain_core.documents import Document
from typing_extensions import Annotated, TypedDict

from app.services import Services
from app.evaluators.llm_as_judge.base import (
    LlmJudgeBooleanEvaluator,
    LlmJudgeBooleanEvaluatorSpec,
    build_llm_judge_boolean_evaluator,
)

__all__ = [
    "build_groundedness_evaluator",
    "GroundedGrade",
    "grounded_instructions",
]


class GroundedGrade(TypedDict):
    """Schema emitted by the groundedness grader LLM."""

    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    grounded: Annotated[
        bool, ..., "Provide the score on if the answer hallucinates from the documents"
    ]


grounded_instructions = """You are a teacher grading a quiz. You will be given FACTS and a STUDENT ANSWER. Here is the grade criteria to follow:
(1) Ensure the STUDENT ANSWER is grounded in the FACTS. (2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.

Grounded:
A grounded value of True means that the student's answer meets all of the criteria.
A grounded value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""


def _extract_documents(outputs: Mapping[str, Any]) -> Sequence[Document]:
    documents = outputs.get("documents")
    if not isinstance(documents, Sequence):
        raise ValueError("Expected 'documents' sequence in evaluator outputs")
    return documents  # type: ignore[return-value]


def _build_groundedness_prompt(
    inputs: Mapping[str, Any],
    outputs: Mapping[str, Any],
    reference_outputs: Optional[Mapping[str, Any]] = None,
) -> str:
    documents = _extract_documents(outputs)
    doc_string = "\n\n".join(doc.page_content for doc in documents)
    return "FACTS: " f"{doc_string}\n" "STUDENT ANSWER: " f"{outputs['answer']}"


def build_groundedness_evaluator(*, services: Services) -> LlmJudgeBooleanEvaluator:
    """Create the groundedness evaluator bound to shared services."""

    spec = LlmJudgeBooleanEvaluatorSpec(
        schema=GroundedGrade,
        instructions=grounded_instructions,
        result_key="grounded",
        build_user_message=_build_groundedness_prompt,
    )
    return build_llm_judge_boolean_evaluator(
        structured_chat_factory=services.structured_chat_llm,
        spec=spec,
    )
