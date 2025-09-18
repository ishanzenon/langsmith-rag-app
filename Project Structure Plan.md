# Project Structure Plan

## Overview
This document captures the agreed modular architecture for the `app` package that will replace the monolithic `langsmith-rag.py` script. It describes each module's responsibilities, justifications for its existence, and key implementation notes so that any future session has the full context needed to build the codebase.

## Package Layout
```
app/
  __init__.py
  settings.py
  services.py
  sources.py
  ingestion.py
  vectorstore.py
  rag.py
  datasets.py
  evaluators/
    __init__.py
    correctness.py
    relevance.py
    groundedness.py
    retrieval.py
  runner.py
```

## Module Details

### `app/settings.py`
- **Purpose:** Centralize configuration, environment loading, and constant definitions.
- **Justification:** Avoids scattering `.env` parsing and hard-coded constants across modules; promotes a single source of truth for configuration.
- **Key Notes:**
  - Load `.env` located at repo root.
  - Provide helpers for LangSmith project name, dataset name, LessWrong post catalog, and other constants needed elsewhere.
  - Expose typed accessors (functions or dataclasses) so downstream modules consume configuration cleanly.

### `app/services.py`
- **Purpose:** Build shared service clients (LangSmith `Client`, `ChatOpenAI`, `OpenAIEmbeddings`, etc.).
- **Justification:** Keeps instantiation logic in one place, making it easy to swap models or clients without touching multiple modules.
- **Key Notes:**
  - Use settings exports for API keys and model names.
  - Expose the instantiated services or factory functions.

### `app/sources.py`
- **Purpose:** Store metadata about document sources (e.g., LessWrong posts) and any future source registries.
- **Justification:** Separates static source definitions from ingestion logic; simplifies maintenance when adding or modifying sources.
- **Key Notes:** Provide structured data (list/dict) consumed by ingestion.

### `app/ingestion.py`
- **Purpose:** Handle document loading, flattening, and text splitting.
- **Justification:** Encapsulates the lifecycle of raw documents into vector-ready chunks; keeps data prep concerns isolated.
- **Key Notes:**
  - Use `WebBaseLoader` (or future loaders) to fetch docs.
  - Apply `RecursiveCharacterTextSplitter` or configurable splitter.
  - Return processed documents for the vector store module.

### `app/vectorstore.py`
- **Purpose:** Build and manage the vector store, exposing retriever interfaces.
- **Justification:** Decouples vector store creation from ingestion and model logic, enabling easy experimentation with alternate stores.
- **Key Notes:**
  - Accept documents from `ingestion` and embeddings from `services`.
  - Provide a helper like `get_retriever(k: int)`.

### `app/rag.py`
- **Purpose:** Define the RAG bot orchestration, including prompt formatting and LangSmith tracing.
- **Justification:** Isolates core RAG interaction logic; ensures tracing decorators and prompt instructions live with the bot implementation.
- **Key Notes:**
  - Implement as a class or factory function that takes a retriever and chat model.
  - Return answers plus document references consistent with the original script.
  - Apply `@traceable` with the existing project/name parameters.

### `app/datasets.py`
- **Purpose:** Manage dataset existence checks, creation, and example registration with LangSmith.
- **Justification:** Keeps evaluation data lifecycle separate from runtime bot logic; simplifies reuse for other experiments.
- **Key Notes:**
  - Use services’ LangSmith client.
  - Expose a function that ensures the dataset exists and returns identifiers as needed.

### `app/evaluators/`
- **Purpose:** House individual evaluation metrics and schemas.
- **Justification:** Modularizes each evaluator so they can be extended or reused independently.
- **Files:**
  - `correctness.py`: TypedDict schema, instructions, evaluator callable for answer correctness.
  - `relevance.py`: Evaluator for answer relevance.
  - `groundedness.py`: Evaluator ensuring answers stay grounded in retrieved docs.
  - `retrieval.py`: Evaluator for document relevance to the question.
- **Key Notes:**
  - Each file defines its schema and callable and instantiates its dedicated `ChatOpenAI` with structured output.
  - `app/evaluators/__init__.py` should expose a registry/list for easy import by the runner.

### `app/runner.py`
- **Purpose:** Act as the driver/orchestrator that wires all components and triggers evaluation.
- **Justification:** Provides a single entry point mirroring the behavior of `langsmith-rag.py` but leveraging modular imports.
- **Key Notes:**
  - Load settings and services.
  - Run dataset setup, build retriever, instantiate RAG bot, gather evaluators, and call `client.evaluate` with metadata.
  - Preserve the experiment prefix and metadata currently used.

## Implementation Reminders
- This plan assumes `langsmith-rag.py` remains available for reference; code should follow the behavior defined there.
- Environment handling (`dotenv`) must continue to occur only within the new settings module.
- All new modules should rely on existing configuration utilities rather than re-reading env variables.
- Keep module interfaces minimal but sufficient to plug into the runner exactly as the script currently does.

