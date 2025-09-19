# LangSmith RAG App

Structured reference implementation of the LangSmith Retrieval-Augmented Generation tutorial. The repository contains both the original monolithic walkthrough (`langsmith-rag.py`) and a modular `app` package that mirrors the same behaviour while being easier to extend and maintain.

## Requirements
- Python 3.10+
- pip (or an equivalent package manager)
- LangSmith account and API key
- OpenAI / Groq / Fireworks API key (for both chat completions and embeddings)

> Note: When opting for Groq / Fireworks as your LLM provider please also populate the environment variable ``OPENAI_API_BASE_URL`` accordingly for your LLM provider. Refer to [Groq OpenAI Compatibility](https://console.groq.com/docs/openai) and [Fireworks OpenAI Compatibility](https://docs.fireworks.ai/tools-sdks/openai-compatibility) for more details.

Install the Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Environment Configuration
1. Copy `.env.example` to `.env` at the repository root.
2. Populate the placeholder values with your LangSmith and OpenAI credentials.

The application automatically loads this file on start-up and promotes the relevant variables into the environment.

## Running the Modular App
The modular implementation is packaged under `app/`. Because intra-package imports are absolute, run it as a module:

```bash
python -m app.runner
```

The command performs the following steps:
- Ensures the LangSmith evaluation dataset exists (creating it on first run).
- Fetches and chunks the LessWrong blog posts.
- Builds an in-memory vector store backed by OpenAI embeddings.
- Instantiates the RAG bot and LLM-as-judge evaluators.
- Executes `langsmith.Client.evaluate` with the configured evaluators and metadata.

## Running the Original Script
`langsmith-rag.py` remains in the repository as a standalone version of the tutorial. It can be executed directly and does not depend on the modular package layout:

```bash
python langsmith-rag.py
```

Keep this script unchanged if you need a one-file reference or want to compare behaviours with the modular variant.

## Primer: `app/` Package Structure

| Module | Responsibility | Key Interfaces |
| --- | --- | --- |
| `app/settings.py` | Loads `.env`, centralises constants, and exposes dataset examples and post catalogues. | `load_environment`, `LESSWRONG_POSTS`, `LESSWRONG_DATASET_EXAMPLES` |
| `app/services.py` | Builds shared LangSmith and LangChain services (client, chat model, embeddings). | `build_services()`, `Services` dataclass |
| `app/sources.py` | Maintains the document source registry for LessWrong posts and enriches metadata. | `DocumentSource`, `iter_sources()`, `load_sources()` |
| `app/ingestion.py` | Orchestrates document loading and text splitting into retrievable chunks. | `ingest_documents()`, `build_text_splitter()` |
| `app/vectorstore.py` | Wraps `InMemoryVectorStore` setup and retriever creation. | `build_vectorstore()`, `get_retriever()` |
| `app/rag.py` | Defines the LangSmith-traced RAG bot, including prompt assembly. | `build_rag_bot()` |
| `app/datasets.py` | Ensures the LangSmith dataset and seed examples exist. | `ensure_default_dataset()` |
| `app/evaluators/` | Aggregates evaluator builders; `llm_as_judge/` houses the LLM-as-judge boolean graders for correctness, relevance, groundedness, and retrieval relevance. | `build_*_evaluator()` helpers |
| `app/runner.py` | Entry point that wires all modules together and calls `Client.evaluate`. | `main()` |

## Troubleshooting
- **Import errors when running the app:** Always invoke the runner as a module (`python -m app.runner`) or set `PYTHONPATH=.` when executing scripts directly.
- **Missing API keys:** Confirm `.env` exists and contains valid `OPENAI_API_KEY`, `LANGSMITH_API_KEY`, and (optionally) `LANGSMITH_PROJECT`/`LANGSMITH_TRACING`.
- **Evaluator cost/latency differences:** The modular runner wires all four LLM-as-judge evaluators by default. Adjust `app/runner.py` if you want parity with the original script, which only runs correctness and retrieval evaluators.

## Additional Resources
- Project structure rationale: `Project Structure Plan.md`
- Original tutorial reference: [Evaluate a RAG application | LangSmith](https://docs.langchain.com/langsmith/evaluate-rag-tutorial)
