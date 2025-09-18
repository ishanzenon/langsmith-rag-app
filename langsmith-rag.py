import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import Client, traceable
from typing_extensions import Annotated, TypedDict

# Load environment variables from .env at repo root (do not commit real secrets)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

# Ensure tracing and API keys are read from environment rather than hardcoded.
# LANGSMITH_TRACING should be set in the .env file if tracing is desired.
if os.getenv("LANGSMITH_TRACING"):
    os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING")

# LANGSMITH_API_KEY and OPENAI_API_KEY are loaded from the environment by python-dotenv.
# Do not set actual secret values in code.
if os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")

if os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

if os.getenv("LANGSMITH_PROJECT"):
    os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT")

# List of URLs to load documents from
lesswrong_posts = [
    {
        "title": "Mechanistic Interpretability Quickstart Guide — Neel Nanda",
        "url": "https://www.lesswrong.com/posts/jLAvJt8wuSFySN975/mechanistic-interpretability-quickstart-guide",
    },
    {
        "title": "A Barebones Guide to Mechanistic Interpretability Prerequisites — Neel Nanda",
        "url": "https://www.lesswrong.com/posts/AaABQpuoNC8gpHf2n/a-barebones-guide-to-mechanistic-interpretability",
    },
    {
        "title": "Toy Models of Superposition — Anthropic (crosspost)",
        "url": "https://www.lesswrong.com/posts/CTh74TaWgvRiXnkS6/toy-models-of-superposition",
    },
    {
        "title": "Some Lessons Learned from Studying Indirect Object Identification in GPT-2 small — Redwood Research",
        "url": "https://www.lesswrong.com/posts/3ecs6duLmTfyra3Gp/some-lessons-learned-from-studying-indirect-object",
    },
    {
        "title": "Explaining the Transformer Circuits Framework by Example",
        "url": "https://www.lesswrong.com/posts/CJsxd8ofLjGFxkmAP/explaining-the-transformer-circuits-framework-by-example",
    },
]

urls = [post["url"] for post in lesswrong_posts]



# Load documents from the URLs
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

# Initialize a text splitter with specified chunk size and overlap
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)

# Split the documents into chunks
doc_splits = text_splitter.split_documents(docs_list)

# Add the document chunks to the "vector store" using OpenAIEmbeddings
vectorstore = InMemoryVectorStore.from_documents(
    documents=doc_splits,
    embedding=OpenAIEmbeddings(),
)

# With langchain we can easily turn any vector store into a retrieval component:
retriever = vectorstore.as_retriever(k=6)

llm = ChatOpenAI(model="gpt-4o", temperature=0.001)

# Add decorator so this function is traced in LangSmith
@traceable(project_name="genai-labs-tracing-project", name="RAG Bot")
def rag_bot(question: str) -> dict:
    # langchain Retriever will be automatically traced
    docs = retriever.invoke(question)
    docs_string = "".join(doc.page_content for doc in docs)
    instructions = f"""You are a helpful assistant who is good at analyzing source information and answering questions.
       Use the following source documents to answer the user's questions.
       If you don't know the answer, just say that you don't know.
       Use three sentences maximum and keep the answer concise.

Documents:
{docs_string}"""
    # langchain ChatModel will be automatically traced
    ai_msg = llm.invoke([
            {"role": "system", "content": instructions},
            {"role": "user", "content": question},
        ],
    )
    return {"answer": ai_msg.content, "documents": docs}



client = Client()

# Define the examples for the dataset
examples = [
    {
        "inputs": {"question": "In Neel Nanda’s Quickstart Guide, what is the goal of mechanistic interpretability?"},
        "outputs": {"answer": "To reverse-engineer trained networks—like reversing a program from its binary—to understand the internal algorithms and cognition."},
    },
    {
        "inputs": {"question": "According to the Quickstart Guide, what minimal setup is recommended to start practical transformer MI work?"},
        "outputs": {"answer": "Copy the TransformerLens demo into a Google Colab with a free GPU and experiment on a small model."},
    },
    {
        "inputs": {"question": "Which architecture does the Barebones Guide say you must deeply understand for MI, and which variant is most relevant?"},
        "outputs": {"answer": "Transformers, especially decoder-only GPT-style models like GPT-2."},
    },
    {
        "inputs": {"question": "Which two tensor tools does the Barebones Guide strongly recommend to avoid common PyTorch pitfalls?"},
        "outputs": {"answer": "einops for reshaping and einsum for tensor multiplication."},
    },
    {
        "inputs": {"question": "In ‘Toy Models of Superposition’, what is superposition and why is it useful?"},
        "outputs": {"answer": "Representing more features than dimensions; with sparse features this compresses information, though it introduces interference requiring nonlinear filtering."},
    },
    {
        "inputs": {"question": "In the toy example from ‘Toy Models of Superposition’, what changes when features become sparse?"},
        "outputs": {"answer": "The model stores additional features in superposition instead of just learning an orthogonal basis for the top features."},
    },
    {
        "inputs": {"question": "What is the IOI task Redwood studied, and how large was the circuit they found?"},
        "outputs": {"answer": "Choosing the correct recipient in sentences like “... gave a drink to ...”; they found a 26-head attention circuit grouped into seven classes."},
    },
    {
        "inputs": {"question": "Name one interaction phenomenon between attention heads observed in the IOI work."},
        "outputs": {"answer": "Heads communicate with pointers—passing positions rather than copying content."},
    },
    {
        "inputs": {"question": "What does the transformer circuits framework help you do when understanding models?"},
        "outputs": {"answer": "Decompose a transformer into identifiable parts (circuits/effective weights) so the overall model is more tractable to analyze."},
    },
    {
        "inputs": {"question": "Which large multi-head circuit example is cited in the ‘Transformer Circuits Framework’ post?"},
        "outputs": {"answer": "The 26-head mechanism for detecting indirect objects (IOI) in GPT-2 small."},
    },
]

# Create the dataset and examples in LangSmith
dataset_name = "LessWrong Mech Interp Blogs Q&A"
if not client.has_dataset(dataset_name=dataset_name):
    dataset = client.create_dataset(dataset_name=dataset_name)
    client.create_examples(
        dataset_id=dataset.id,
        examples=examples
    )

# Grade output schema
class CorrectnessGrade(TypedDict):
    # Note that the order in the fields are defined is the order in which the model will generate them.
    # It is useful to put explanations before responses because it forces the model to think through
    # its final response before generating it:
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    correct: Annotated[bool, ..., "True if the answer is correct, False otherwise."]

# Grade prompt
correctness_instructions = """You are a teacher grading a quiz. You will be given a QUESTION, the GROUND TRUTH (correct) ANSWER, and the STUDENT ANSWER. Here is the grade criteria to follow:
(1) Grade the student answers based ONLY on their factual accuracy relative to the ground truth answer. (2) Ensure that the student answer does not contain any conflicting statements.
(3) It is OK if the student answer contains more information than the ground truth answer, as long as it is factually accurate relative to the  ground truth answer.

Correctness:
A correctness value of True means that the student's answer meets all of the criteria.
A correctness value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""

# Grader LLM
grader_llm = ChatOpenAI(model="gpt-4o", temperature=0.001).with_structured_output(
    CorrectnessGrade, method="json_schema", strict=True
)

def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    """An evaluator for RAG answer accuracy"""
    answers = f"""\
QUESTION: {inputs['question']}
GROUND TRUTH ANSWER: {reference_outputs['answer']}
STUDENT ANSWER: {outputs['answer']}"""
    # Run evaluator
    grade = grader_llm.invoke([
            {"role": "system", "content": correctness_instructions},
            {"role": "user", "content": answers},
        ]
    )
    return grade["correct"]

# Grade output schema
class RelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    relevant: Annotated[
        bool, ..., "Provide the score on whether the answer addresses the question"
    ]

# Grade prompt
relevance_instructions = """You are a teacher grading a quiz. You will be given a QUESTION and a STUDENT ANSWER. Here is the grade criteria to follow:
(1) Ensure the STUDENT ANSWER is concise and relevant to the QUESTION
(2) Ensure the STUDENT ANSWER helps to answer the QUESTION

Relevance:
A relevance value of True means that the student's answer meets all of the criteria.
A relevance value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""

# Grader LLM
relevance_llm = ChatOpenAI(model="gpt-4o", temperature=0.001).with_structured_output(
    RelevanceGrade, method="json_schema", strict=True
)

# Evaluator
def relevance(inputs: dict, outputs: dict) -> bool:
    """A simple evaluator for RAG answer helpfulness."""
    answer = f"QUESTION: {inputs['question']}\nSTUDENT ANSWER: {outputs['answer']}"
    grade = relevance_llm.invoke([
            {"role": "system", "content": relevance_instructions},
            {"role": "user", "content": answer},
        ]
    )
    return grade["relevant"]

# Grade output schema
class GroundedGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    grounded: Annotated[
        bool, ..., "Provide the score on if the answer hallucinates from the documents"
    ]

# Grade prompt
grounded_instructions = """You are a teacher grading a quiz. You will be given FACTS and a STUDENT ANSWER. Here is the grade criteria to follow:
(1) Ensure the STUDENT ANSWER is grounded in the FACTS. (2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.

Grounded:
A grounded value of True means that the student's answer meets all of the criteria.
A grounded value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""

# Grader LLM
grounded_llm = ChatOpenAI(model="gpt-4o", temperature=0.001).with_structured_output(
    GroundedGrade, method="json_schema", strict=True
)

# Evaluator
def groundedness(inputs: dict, outputs: dict) -> bool:
    """A simple evaluator for RAG answer groundedness."""
    doc_string = "\n\n".join(doc.page_content for doc in outputs["documents"])
    answer = f"FACTS: {doc_string}\nSTUDENT ANSWER: {outputs['answer']}"
    grade = grounded_llm.invoke([
            {"role": "system", "content": grounded_instructions},
            {"role": "user", "content": answer},
        ]
    )
    return grade["grounded"]

# Grade output schema
class RetrievalRelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    relevant: Annotated[
        bool,
        ...,
        "True if the retrieved documents are relevant to the question, False otherwise",
    ]

# Grade prompt
retrieval_relevance_instructions = """You are a teacher grading a quiz. You will be given a QUESTION and a set of FACTS provided by the student. Here is the grade criteria to follow:
(1) You goal is to identify FACTS that are completely unrelated to the QUESTION
(2) If the facts contain ANY keywords or semantic meaning related to the question, consider them relevant
(3) It is OK if the facts have SOME information that is unrelated to the question as long as (2) is met

Relevance:
A relevance value of True means that the FACTS contain ANY keywords or semantic meaning related to the QUESTION and are therefore relevant.
A relevance value of False means that the FACTS are completely unrelated to the QUESTION.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""

# Grader LLM
retrieval_relevance_llm = ChatOpenAI(
    model="gpt-4o", temperature=0.001
).with_structured_output(RetrievalRelevanceGrade, method="json_schema", strict=True)

def retrieval_relevance(inputs: dict, outputs: dict) -> bool:
    """An evaluator for document relevance"""
    doc_string = "\n\n".join(doc.page_content for doc in outputs["documents"])
    answer = f"FACTS: {doc_string}\nQUESTION: {inputs['question']}"
    # Run evaluator
    grade = retrieval_relevance_llm.invoke([
            {"role": "system", "content": retrieval_relevance_instructions},
            {"role": "user", "content": answer},
        ]
    )
    return grade["relevant"]

def target(inputs: dict) -> dict:
    return rag_bot(inputs["question"])

experiment_results = client.evaluate(
    target,
    data=dataset_name,
    evaluators=[correctness, retrieval_relevance],
    experiment_prefix="genai-labs-experiment",
    metadata={"version": "LCEL context, gpt-4-0125-preview"},
)

# Explore results locally as a dataframe if you have pandas installed
# experiment_results.to_pandas()