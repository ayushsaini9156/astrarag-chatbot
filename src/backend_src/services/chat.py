import logging
import json

from src.agents_src.crew import qa_crew

logger = logging.getLogger(__name__)

FALLBACK_ANSWER = "The knowledge source does not contain the required information."


def _normalize_answer_payload(result_dict: dict) -> dict:
    """Coerce CrewAI output into the response shape expected by the API and UI."""

    normalized = dict(result_dict or {})

    raw_payload = normalized.get("raw")
    if isinstance(raw_payload, str) and raw_payload.strip():
        try:
            raw_json = json.loads(raw_payload)
            if isinstance(raw_json, dict):
                normalized.update(raw_json)
        except json.JSONDecodeError:
            pass

    answer = normalized.get("answer") or normalized.get("output") or FALLBACK_ANSWER

    sources = normalized.get("sources") or normalized.get("source_files") or []
    if isinstance(sources, str):
        sources = [sources]
    elif not isinstance(sources, list):
        sources = list(sources) if sources else []

    tool_used = normalized.get("tool_used") or "rag_query_tool"

    rationale = normalized.get("rationale")
    if not rationale:
        rationale = (
            "No relevant documents were retrieved from the knowledge source."
            if answer == FALLBACK_ANSWER
            else "Answer synthesized from retrieved context."
        )

    normalized.update(
        {
            "answer": answer,
            "sources": sources,
            "source_files": sources,
            "tool_used": tool_used,
            "rationale": rationale,
        }
    )

    return normalized


def get_answer(chat_history: list) -> dict:
    logger.info(f"Received chat_history: {chat_history}")
    # get the last message in the chat_history as user_query
    last_user_message = chat_history[-1]
    user_query = last_user_message["content"]
    logger.info(f"Extracted user_query: {user_query}")
    # Remove the last user message from chat_history
    history_without_last = chat_history[:-1]
    input_data = {
        "user_query": user_query,
        "chat_history": history_without_last,
    }
    logger.debug(f"Input data for qa_crew: {input_data}")
    result = qa_crew.kickoff(input_data)
    result_dict = result.to_dict()
    result_dict = _normalize_answer_payload(result_dict)
    logger.info(f"Result from qa_crew: {result_dict}")
    return result_dict