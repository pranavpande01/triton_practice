import os
from typing import List, Dict

import streamlit as st
from google import genai
from google.genai import types

DEFAULT_MODEL = "gemini-2.5-flash"


def call_gemini(api_key: str, model: str, messages: List[Dict[str, str]]) -> str:
    client = genai.Client(api_key=api_key)
    history = [
        types.Content(
            role="user" if m["role"] == "user" else "model",
            parts=[types.Part(text=m["content"])],
        )
        for m in messages[:-1]
    ]

    if history:
        chat = client.chats.create(model=model, history=history)
    else:
        chat = client.chats.create(model=model)

    response = chat.send_message(messages[-1]["content"])
    text = getattr(response, "text", None)
    return text.strip() if text else "No text response returned by model."


def main() -> None:
    st.set_page_config(page_title="Gemini Chat", page_icon=":speech_balloon:", layout="centered")
    st.title("Gemini Chatbot")

    with st.sidebar:
        st.subheader("Settings")
        api_key = st.text_input(
            "Gemini API Key",
            value=os.getenv("GEMINI_API_KEY", ""),
            type="password",
            help="Set GEMINI_API_KEY in env or paste it here.",
        )
        model = st.text_input("Model", value=DEFAULT_MODEL)
        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Type your message...")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if not api_key:
        err = "Missing API key. Set GEMINI_API_KEY or enter it in the sidebar."
        st.session_state.messages.append({"role": "assistant", "content": err})
        with st.chat_message("assistant"):
            st.error(err)
        return

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer = call_gemini(api_key=api_key, model=model, messages=st.session_state.messages)
            except Exception as exc:
                answer = f"Request failed: {exc}"
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()

