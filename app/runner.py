"""Application entry point that coordinates settings, services, RAG bot, and evaluators.

Will mirror the behavior of `langsmith-rag.py` by composing the modular pieces and
triggering LangSmith evaluations with the appropriate metadata.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, cast

from langchain_core.documents import Document
from langchain_text_splitters.base import TextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.vectorstores.base import VectorStoreRetriever

from app import datasets, evaluators, ingestion, settings, sources, vectorstore
from app.rag import RAGCallable, RAGResponse, build_rag_bot
from app.services import Services, build_services
from app.evaluators.base import BooleanEvaluator


def run_ingestion() -> Dict[str, Any]:
    """Replicate the ingestion pipeline from the original script."""

    raw_documents: tuple[Document, ...] = ingestion.load_documents(
        collection=sources.COLLECTION_BLOG_POSTS,
        tags=("blog",),
    )
    splitter: TextSplitter = ingestion.build_text_splitter()
    document_chunks: tuple[Document, ...] = ingestion.chunk_documents(
        raw_documents,
        splitter=splitter,
    )
    return {
        "raw_documents": raw_documents,
        "splitter": splitter,
        "document_chunks": document_chunks,
    }


def build_vectorstore_state(
    *,
    documents: tuple[Document, ...],
    services: Services,
    retriever_k: int | None = None,
) -> Dict[str, Any]:
    """Create the vector store and retriever mirroring ``langsmith-rag.py``."""

    vector_store: InMemoryVectorStore = vectorstore.build_vectorstore(
        documents,
        embeddings=services.embeddings,
    )
    retriever: VectorStoreRetriever = vectorstore.get_retriever(
        vector_store,
        k=retriever_k,
    )
    return {
        "vector_store": vector_store,
        "retriever": retriever,
    }


def build_rag_state(
    *, retriever: VectorStoreRetriever, services: Services
) -> Dict[str, Any]:
    """Instantiate the RAG bot using the provided retriever and chat model."""

    rag_bot = build_rag_bot(retriever=retriever, chat_model=services.chat_llm)
    return {"rag_bot": rag_bot}


def build_evaluators_state(*, services: Services) -> Dict[str, Any]:
    """Construct all boolean evaluators backed by shared services."""

    correctness = evaluators.build_correctness_evaluator(services=services)
    relevance = evaluators.build_relevance_evaluator(services=services)
    groundedness = evaluators.build_groundedness_evaluator(services=services)
    retrieval_relevance = evaluators.build_retrieval_relevance_evaluator(
        services=services
    )

    evaluator_map: Dict[str, BooleanEvaluator] = {
        "correctness": correctness,
        "relevance": relevance,
        "groundedness": groundedness,
        "retrieval_relevance": retrieval_relevance,
    }
    return {"evaluators": evaluator_map}


def _build_target_callable(
    rag_bot: RAGCallable,
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    def target(inputs: Dict[str, Any]) -> Dict[str, Any]:
        response: RAGResponse = rag_bot(inputs["question"])
        return cast(Dict[str, Any], response)

    return target


def main() -> Dict[str, Any]:
    """Bootstrap shared services ready for downstream orchestration."""

    services: Services = build_services()
    dataset_state = datasets.ensure_default_dataset(services)
    pipeline_state = run_ingestion()
    pipeline_state["services"] = services
    pipeline_state["dataset_state"] = dataset_state
    vector_state = build_vectorstore_state(
        documents=pipeline_state["document_chunks"],
        services=services,
    )
    rag_state = build_rag_state(
        retriever=vector_state["retriever"],
        services=services,
    )
    evaluator_state = build_evaluators_state(services=services)
    target = _build_target_callable(rag_state["rag_bot"])
    evaluator_sequence = [
        evaluator_state["evaluators"]["correctness"],
        evaluator_state["evaluators"]["relevance"],
        evaluator_state["evaluators"]["groundedness"],
        evaluator_state["evaluators"]["retrieval_relevance"],
    ]
    experiment_results = services.client.evaluate(
        target,
        data=settings.LESSWRONG_DATASET_NAME,
        evaluators=evaluator_sequence,
        experiment_prefix=settings.EXPERIMENT_PREFIX,
        metadata={"version": settings.EXPERIMENT_METADATA_VERSION},
    )
    pipeline_state.update(vector_state)
    pipeline_state.update(rag_state)
    pipeline_state.update(evaluator_state)
    pipeline_state["experiment_results"] = experiment_results
    return pipeline_state


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
