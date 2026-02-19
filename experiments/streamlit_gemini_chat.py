import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple

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


def run_model(api_key: str, model: str, prompt: str) -> Tuple[str, str]:
    messages = [{"role": "user", "content": prompt}]
    try:
        return model, call_gemini(api_key=api_key, model=model, messages=messages)
    except Exception as exc:
        return model, f"Request failed: {exc}"


def run_models_parallel(model_entries: List[Dict[str, str]], prompt: str) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    workers = min(max(len(model_entries), 1), 8)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(run_model, entry["api_key"], entry["model"], prompt): idx
            for idx, entry in enumerate(model_entries)
        }
        ordered: Dict[int, Dict[str, str]] = {}

        for future in as_completed(futures):
            idx = futures[future]
            model, output = future.result()
            ordered[idx] = {"model": model, "output": output}

    for idx in sorted(ordered):
        results.append(ordered[idx])
    return results


def main() -> None:
    st.set_page_config(page_title="Gemini Chat", page_icon=":speech_balloon:", layout="centered")
    st.title("Gemini Multi-Model Chat")

    if "model_entries" not in st.session_state:
        st.session_state.model_entries = [
            {"model": DEFAULT_MODEL, "api_key": os.getenv("GEMINI_API_KEY", "")}
        ]
    if "outputs" not in st.session_state:
        st.session_state.outputs = []
    if "last_prompt" not in st.session_state:
        st.session_state.last_prompt = ""

    with st.sidebar:
        st.subheader("Settings")
        st.markdown("### Models + Keys")
        delete_index = None
        for idx, entry in enumerate(st.session_state.model_entries):
            st.session_state.model_entries[idx]["model"] = st.text_input(
                f"Model {idx + 1}",
                value=entry["model"],
                key=f"model_{idx}"
            )
            col_key, col_delete = st.columns([5, 1])
            st.session_state.model_entries[idx]["api_key"] = col_key.text_input(
                f"API Key {idx + 1}",
                value=entry["api_key"],
                type="password",
                key=f"api_key_{idx}",
                help="Use a separate Gemini key for each model.",
            )
            if col_delete.button("X", key=f"delete_{idx}"):
                delete_index = idx

        if st.button("+ Add model"):
            st.session_state.model_entries.append({"model": DEFAULT_MODEL, "api_key": ""})
            st.rerun()

        if delete_index is not None:
            if len(st.session_state.model_entries) > 1:
                st.session_state.model_entries.pop(delete_index)
                st.rerun()
            st.warning("At least one model is required.")

        if st.button("Clear outputs"):
            st.session_state.outputs = []
            st.session_state.last_prompt = ""
            st.rerun()

    with st.form("prompt_form"):
        prompt = st.text_area("Prompt", value=st.session_state.last_prompt, placeholder="Type your prompt...")
        submitted = st.form_submit_button("Run all models")

    if submitted:
        cleaned_entries = []
        for entry in st.session_state.model_entries:
            model_name = entry["model"].strip()
            key_value = entry["api_key"].strip()
            if model_name and key_value:
                cleaned_entries.append({"model": model_name, "api_key": key_value})

        if not cleaned_entries:
            st.error("Add at least one valid model name.")
            return
        if len(cleaned_entries) != len(st.session_state.model_entries):
            st.error("Every model must have both a model name and an API key.")
            return
        unique_keys = {entry["api_key"] for entry in cleaned_entries}
        if len(unique_keys) != len(cleaned_entries):
            st.error("Each model must use a different API key.")
            return
        if not prompt.strip():
            st.error("Prompt cannot be empty.")
            return

        st.session_state.model_entries = cleaned_entries
        st.session_state.last_prompt = prompt
        with st.spinner("Thinking..."):
            st.session_state.outputs = run_models_parallel(
                model_entries=cleaned_entries,
                prompt=prompt,
            )

    if st.session_state.outputs:
        st.markdown(f"### Prompt\n{st.session_state.last_prompt}")
        columns = st.columns(len(st.session_state.outputs))
        for col, result in zip(columns, st.session_state.outputs):
            with col:
                st.markdown(f"#### `{result['model']}`")
                st.write(result["output"])


if __name__ == "__main__":
    main()

