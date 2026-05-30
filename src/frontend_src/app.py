import sys
import os
import subprocess
from pathlib import Path
from urllib.parse import urljoin, urlparse
import time
# Add project root to sys.path BEFORE any imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import streamlit as st
import requests
from src.frontend_src.config.frontend_settings import Settings

settings = Settings()


def _backend_base_url() -> str:
    parsed_url = urlparse(settings.CHAT_ENDPOINT_URL)
    return f"{parsed_url.scheme}://{parsed_url.netloc}"


def _backend_health_url() -> str:
    return urljoin(_backend_base_url(), "/health")


def _backend_is_healthy() -> bool:
    try:
        response = requests.get(_backend_health_url(), timeout=1.5)
        return response.ok
    except requests.RequestException:
        return False


def _start_backend_process() -> subprocess.Popen:
    project_root = Path(__file__).resolve().parents[2]
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

    return subprocess.Popen(
        [sys.executable, "-m", "src.backend_src.main"],
        cwd=str(project_root),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
    )


def ensure_backend_ready(timeout_seconds: int = 20) -> bool:
    if _backend_is_healthy():
        return True

    if not st.session_state.get("backend_starting"):
        st.session_state.backend_starting = True
        st.session_state.backend_process = _start_backend_process()

    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if _backend_is_healthy():
            st.session_state.backend_starting = False
            return True
        time.sleep(0.5)

    return False
st.set_page_config(
    page_title="AstraRAG",
    page_icon="🤖",
    layout="centered",
)
st.title("💬 AstraRAG - Agentic RAG Chatbot")

if not ensure_backend_ready():
    st.error("Backend is starting up or unavailable. Please wait a moment and retry.")
    st.stop()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("role") == "assistant":
            sources = message.get("sources", [])
            tool_used = message.get("tool_used")
            rationale = message.get("rationale")
            if sources:
                st.markdown(f"**Sources:** {', '.join(sources)}")
            if tool_used or rationale:
                with st.expander("Show details (tool & rationale)"):
                    st.markdown(f"**Tool Used:** {tool_used if tool_used else 'N/A'}")
                    st.markdown(f"**Rationale:** {rationale if rationale else 'N/A'}")

user_prompt = st.chat_input("Ask Chatbot...")

if user_prompt:
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    # Prepare payload for API
    payload = {"chat_history": st.session_state.chat_history}
    try:
        response = requests.post(settings.CHAT_ENDPOINT_URL, json=payload)
        response.raise_for_status()
        response_json = response.json()
        assistant_response = response_json.get("answer", "(No response)")
        tool_used = response_json.get("tool_used", "N/A")
        rationale = response_json.get("rationale", "N/A")
        sources = response_json.get("sources", [])
    except Exception as e:
        assistant_response = f"Error: {e}"
        tool_used = "N/A"
        rationale = "N/A"
        sources = []

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": assistant_response,
        "tool_used": tool_used,
        "rationale": rationale,
        "sources": sources
    })
    with st.chat_message("assistant"):
        st.markdown(assistant_response)
        if sources:
            st.markdown(f"**Sources:** {', '.join(sources)}")
        with st.expander("Show details (tool & rationale)"):
            st.markdown(f"**Tool Used:** {tool_used}")
            st.markdown(f"**Rationale:** {rationale}")