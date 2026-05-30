import logging
import json

from crewai.tools import tool
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.core import Settings
import chromadb

from src.agents_src.config.agent_settings import AgentSettings

# Get a logger for this module
logger = logging.getLogger(__name__)

# Load embedding model once
logger.info("Loading HuggingFace embedding model...")
embed_model = HuggingFaceEmbedding()


def _normalize_query(query: str | dict) -> str:
    """Extract the actual user query from CrewAI tool input."""

    if isinstance(query, dict):
        return str(query.get("query") or query.get("user_query") or "")

    if isinstance(query, str):
        stripped_query = query.strip()
        if not stripped_query:
            return ""

        try:
            parsed_query = json.loads(stripped_query)
            if isinstance(parsed_query, dict):
                return str(parsed_query.get("query") or parsed_query.get("user_query") or stripped_query)
        except json.JSONDecodeError:
            pass

        return stripped_query

    return str(query)


@tool
def rag_query_tool(query: str) -> dict:
    """
    Answers a query by retrieving relevant documents and generating a response.
    Returns the generated answer, source file names, and a short rationale for transparency.
    """

    settings = AgentSettings()
    normalized_query = _normalize_query(query)

    # Configure LLM
    Settings.llm = Groq(
        model=settings.MODEL_NAME,
        temperature=settings.MODEL_TEMPERATURE,
        api_key=settings.GROQ_API_KEY,
    )

    # Load Chroma collection
    db = chromadb.PersistentClient(path=settings.VECTOR_STORE_DIR)

    chroma_collection = db.get_or_create_collection(
        settings.COLLECTION_NAME
    )

    # Connect vector store
    vector_store = ChromaVectorStore(
        chroma_collection=chroma_collection
    )

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )

    # Load index
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
        embed_model=embed_model,
    )

    # Create query engine
    query_engine = index.as_query_engine(
        similarity_top_k=3
    )

    # Execute query
    response = query_engine.query(normalized_query)

    # Debug logs
    logger.info(f"Response type: {type(response)}")
    logger.info(f"Metadata: {getattr(response, 'metadata', None)}")

    source_file_names = set()

    # Preferred approach for newer LlamaIndex versions
    if hasattr(response, "source_nodes") and response.source_nodes:
        for node in response.source_nodes:
            metadata = getattr(node.node, "metadata", {}) or {}

            file_name = metadata.get("file_name")
            if file_name:
                source_file_names.add(file_name)

    # Fallback for older versions
    else:
        metadata = getattr(response, "metadata", None) or {}

        for value in metadata.values():
            if isinstance(value, dict):
                file_name = value.get("file_name")
                if file_name:
                    source_file_names.add(file_name)

    source_files = sorted(source_file_names)
    rationale = (
        "No relevant source nodes were retrieved from the vector store. "
        "The response is therefore based on the knowledge source's inability to answer the query."
        if not source_files
        else f"Retrieved supporting context from: {', '.join(source_files)}."
    )

    answer_text = response.response or ""
    if not answer_text or answer_text.strip().lower() == "empty response":
        answer_text = "The knowledge source does not contain the required information."

    return {
        "answer": answer_text,
        "sources": source_files,
        "source_files": source_files,
        "tool_used": "rag_query_tool",
        "rationale": rationale,
    }


