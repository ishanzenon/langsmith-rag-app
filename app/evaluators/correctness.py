"""Correctness evaluator built on top of the shared boolean scaffolding."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from typing_extensions import Annotated, TypedDict

from ..services import Services
from .base import BooleanEvaluator, BooleanEvaluatorSpec, build_boolean_evaluator

__all__ = ["build_correctness_evaluator", "CorrectnessGrade", "correctness_instructions"]


class CorrectnessGrade(TypedDict):
    """Schema emitted by the correctness grader LLM."""

    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    correct: Annotated[bool, ..., "True if the answer is correct, False otherwise."]


correctness_instructions = """You are a teacher grading a quiz. You will be given a QUESTION, the GROUND TRUTH (correct) ANSWER, and the STUDENT ANSWER. Here is the grade criteria to follow:
(1) Grade the student answers based ONLY on their factual accuracy relative to the ground truth answer. (2) Ensure that the student answer does not contain any conflicting statements.
(3) It is OK if the student answer contains more information than the ground truth answer, as long as it is factually accurate relative to the  ground truth answer.

Correctness:
A correctness value of True means that the student's answer meets all of the criteria.
A correctness value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""


def _build_correctness_prompt(
    inputs: Mapping[str, Any],
    outputs: Mapping[str, Any],
    reference_outputs: Optional[Mapping[str, Any]],
) -> str:
    assert reference_outputs is not None
    return (
        "QUESTION: "
        f"{inputs['question']}\n"
        "GROUND TRUTH ANSWER: "
        f"{reference_outputs['answer']}\n"
        "STUDENT ANSWER: "
        f"{outputs['answer']}"
    )


def build_correctness_evaluator(*, services: Services) -> BooleanEvaluator:
    """Create the correctness evaluator bound to shared services."""

    spec = BooleanEvaluatorSpec(
        schema=CorrectnessGrade,
        instructions=correctness_instructions,
        result_key="correct",
        build_user_message=_build_correctness_prompt,
        require_reference_outputs=True,
    )
    return build_boolean_evaluator(
        structured_chat_factory=services.structured_chat_llm,
        spec=spec,
    )
