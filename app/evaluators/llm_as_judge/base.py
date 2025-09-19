"""Shared evaluator scaffolding for LangSmith LLM-judge boolean graders.

Provides a reusable abstraction so individual evaluator modules only define the
schema, instructions, and message formatting logic. Invocation and structured
LLM plumbing live in this helper to keep behaviour uniform across evaluators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, Mapping, Optional, Protocol, TypeVar

from langchain_core.runnables.base import Runnable

from app.services import StructuredChatFactory

__all__ = [
    "LlmJudgeBooleanEvaluatorSpec",
    "build_llm_judge_boolean_evaluator",
    "LlmJudgeBooleanEvaluator",
]


SchemaT = TypeVar("SchemaT", bound=Mapping[str, Any])
Inputs = Mapping[str, Any]
Outputs = Mapping[str, Any]
ReferenceOutputs = Mapping[str, Any]


class LlmJudgeBooleanEvaluator(Protocol):
    """Callable protocol for LangSmith-style LLM judge evaluators."""

    def __call__(
        self,
        inputs: Mapping[str, Any],
        outputs: Mapping[str, Any],
        reference_outputs: Optional[Mapping[str, Any]] = None,
    ) -> bool:
        ...


MessageBuilder = Callable[[Inputs, Outputs, Optional[ReferenceOutputs]], str]


@dataclass(frozen=True)
class LlmJudgeBooleanEvaluatorSpec(Generic[SchemaT]):
    """Configuration required to assemble an LLM-judge boolean evaluator."""

    schema: type[SchemaT]
    instructions: str
    result_key: str
    build_user_message: MessageBuilder
    require_reference_outputs: bool = False
    structured_kwargs: Mapping[str, Any] | None = None


def build_llm_judge_boolean_evaluator(
    *,
    structured_chat_factory: StructuredChatFactory,
    spec: LlmJudgeBooleanEvaluatorSpec[SchemaT],
) -> LlmJudgeBooleanEvaluator:
    """Create an LLM-judge boolean evaluator bound to the structured chat factory."""

    grader: Runnable[Any, SchemaT | Dict[str, Any]] = structured_chat_factory(
        spec.schema,
        **(dict(spec.structured_kwargs) if spec.structured_kwargs else {}),
    )

    def _ensure_reference(reference: Optional[ReferenceOutputs]) -> ReferenceOutputs:
        if reference is None:
            raise ValueError(
                "Evaluator requires reference outputs but none were provided."
            )
        return reference

    def evaluator(
        inputs: Mapping[str, Any],
        outputs: Mapping[str, Any],
        reference_outputs: Optional[Mapping[str, Any]] = None,
    ) -> bool:
        if spec.require_reference_outputs:
            reference_outputs = _ensure_reference(reference_outputs)

        user_content = spec.build_user_message(inputs, outputs, reference_outputs)
        messages = [
            {"role": "system", "content": spec.instructions},
            {"role": "user", "content": user_content},
        ]

        grade: Mapping[str, Any] = grader.invoke(messages)
        flag = grade.get(spec.result_key)
        if flag is None:
            raise KeyError(
                f"Structured grader response missing '{spec.result_key}' field: {grade}"
            )
        return bool(flag)

    return evaluator
