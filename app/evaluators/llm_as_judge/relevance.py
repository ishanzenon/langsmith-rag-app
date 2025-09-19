"""Relevance evaluator built on the shared LLM judge scaffolding."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from typing_extensions import Annotated, TypedDict

from app.services import Services
from app.evaluators.llm_as_judge.base import (
    LlmJudgeBooleanEvaluator,
    LlmJudgeBooleanEvaluatorSpec,
    build_llm_judge_boolean_evaluator,
)

__all__ = ["build_relevance_evaluator", "RelevanceGrade", "relevance_instructions"]


class RelevanceGrade(TypedDict):
    """Schema emitted by the relevance grader LLM."""

    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    relevant: Annotated[
        bool, ..., "Provide the score on whether the answer addresses the question"
    ]


relevance_instructions = """You are a teacher grading a quiz. You will be given a QUESTION and a STUDENT ANSWER. Here is the grade criteria to follow:
(1) Ensure the STUDENT ANSWER is concise and relevant to the QUESTION
(2) Ensure the STUDENT ANSWER helps to answer the QUESTION

Relevance:
A relevance value of True means that the student's answer meets all of the criteria.
A relevance value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""


def _build_relevance_prompt(
    inputs: Mapping[str, Any],
    outputs: Mapping[str, Any],
    reference_outputs: Optional[Mapping[str, Any]] = None,
) -> str:
    return (
        "QUESTION: " f"{inputs['question']}\n" "STUDENT ANSWER: " f"{outputs['answer']}"
    )


def build_relevance_evaluator(*, services: Services) -> LlmJudgeBooleanEvaluator:
    """Create the relevance evaluator bound to shared services."""

    spec = LlmJudgeBooleanEvaluatorSpec(
        schema=RelevanceGrade,
        instructions=relevance_instructions,
        result_key="relevant",
        build_user_message=_build_relevance_prompt,
    )
    return build_llm_judge_boolean_evaluator(
        structured_chat_factory=services.structured_chat_llm,
        spec=spec,
    )
