import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Any

import streamlit as st
from google import genai
from google.genai import types

DEFAULT_MODEL = "gemini-2.5-flash"


def message_to_parts(message: Dict[str, Any]) -> List[types.Part]:
    parts: List[types.Part] = []
    text = message.get("content", "")
    if text:
        parts.append(types.Part(text=text))
    image_bytes = message.get("image_bytes")
    image_mime_type = message.get("image_mime_type")
    if image_bytes and image_mime_type:
        parts.append(types.Part.from_bytes(data=image_bytes, mime_type=image_mime_type))
    return parts


def call_gemini(api_key: str, model: str, messages: List[Dict[str, Any]]) -> str:
    client = genai.Client(api_key=api_key)
    history = [
        types.Content(
            role="user" if m["role"] == "user" else "model",
            parts=message_to_parts(m),
        )
        for m in messages[:-1]
    ]

    if history:
        chat = client.chats.create(model=model, history=history)
    else:
        chat = client.chats.create(model=model)

    response = chat.send_message(message_to_parts(messages[-1]))
    text = getattr(response, "text", None)
    return text.strip() if text else "No text response returned by model."


def build_messages_for_model(
    turns: List[Dict[str, Any]],
    model: str,
    prompt: str,
    image_bytes: bytes | None,
    image_mime_type: str | None,
) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    for turn in turns:
        user_message: Dict[str, Any] = {"role": "user", "content": turn["prompt"]}
        if turn.get("image_bytes") and turn.get("image_mime_type"):
            user_message["image_bytes"] = turn["image_bytes"]
            user_message["image_mime_type"] = turn["image_mime_type"]
        messages.append(user_message)
        model_response = turn["responses"].get(model)
        if model_response:
            messages.append({"role": "assistant", "content": model_response})

    latest_user_message: Dict[str, Any] = {"role": "user", "content": prompt}
    if image_bytes and image_mime_type:
        latest_user_message["image_bytes"] = image_bytes
        latest_user_message["image_mime_type"] = image_mime_type
    messages.append(latest_user_message)
    return messages


def run_model(
    api_key: str,
    model: str,
    turns: List[Dict[str, Any]],
    prompt: str,
    image_bytes: bytes | None,
    image_mime_type: str | None,
) -> Tuple[str, str]:
    messages = build_messages_for_model(
        turns=turns,
        model=model,
        prompt=prompt,
        image_bytes=image_bytes,
        image_mime_type=image_mime_type,
    )
    try:
        return model, call_gemini(api_key=api_key, model=model, messages=messages)
    except Exception as exc:
        return model, f"Request failed: {exc}"


def run_models_parallel(
    model_entries: List[Dict[str, str]],
    turns: List[Dict[str, Any]],
    prompt: str,
    image_bytes: bytes | None,
    image_mime_type: str | None,
) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    workers = min(max(len(model_entries), 1), 8)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                run_model,
                entry["api_key"],
                entry["model"],
                turns,
                prompt,
                image_bytes,
                image_mime_type,
            ): idx
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


def validate_model_entries(model_entries: List[Dict[str, str]]) -> Tuple[bool, str, List[Dict[str, str]]]:
    cleaned_entries = []
    for entry in model_entries:
        model_name = entry["model"].strip()
        key_value = entry["api_key"].strip()
        if model_name and key_value:
            cleaned_entries.append({"model": model_name, "api_key": key_value})

    if not cleaned_entries:
        return False, "Add at least one valid model name.", []
    if len(cleaned_entries) != len(model_entries):
        return False, "Every model must have both a model name and an API key.", []
    unique_keys = {entry["api_key"] for entry in cleaned_entries}
    if len(unique_keys) != len(cleaned_entries):
        return False, "Each model must use a different API key.", []
    return True, "", cleaned_entries


def render_chat(turns: List[Dict[str, Any]], current_models: List[str]) -> None:
    for turn in turns:
        with st.chat_message("user"):
            st.markdown(turn["prompt"])
            if turn.get("image_bytes") and turn.get("image_mime_type"):
                st.image(turn["image_bytes"], caption=turn.get("image_name", "Uploaded image"))
        with st.chat_message("assistant"):
            columns = st.columns(len(current_models))
            for col, model in zip(columns, current_models):
                with col:
                    st.markdown(f"#### `{model}`")
                    output = turn["responses"].get(model)
                    if output:
                        st.write(output)
                    else:
                        st.caption("No response for this turn.")


def main() -> None:
    st.set_page_config(page_title="Gemini Chat", page_icon=":speech_balloon:", layout="centered")
    st.title("Gemini Multi-Model Chat")

    if "model_entries" not in st.session_state:
        st.session_state.model_entries = [
            {"model": DEFAULT_MODEL, "api_key": os.getenv("GEMINI_API_KEY", "")}
        ]
    if "turns" not in st.session_state:
        st.session_state.turns = []
    if "image_uploader_nonce" not in st.session_state:
        st.session_state.image_uploader_nonce = 0

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

        if st.button("Clear chat"):
            st.session_state.turns = []
            st.rerun()

    is_valid, err_msg, cleaned_entries = validate_model_entries(st.session_state.model_entries)
    if not is_valid:
        st.warning(err_msg)
    else:
        st.session_state.model_entries = cleaned_entries

    current_models = [entry["model"] for entry in st.session_state.model_entries]
    if current_models:
        render_chat(turns=st.session_state.turns, current_models=current_models)

    uploaded_image = st.file_uploader(
        "Optional image for next message",
        type=["png", "jpg", "jpeg", "webp"],
        key=f"pending_image_{st.session_state.image_uploader_nonce}",
    )

    prompt = st.chat_input("Type your message...")
    if not prompt:
        return

    if not is_valid:
        st.error(err_msg)
        return
    if not prompt.strip():
        st.error("Prompt cannot be empty.")
        return

    with st.chat_message("user"):
        st.markdown(prompt)
        current_image_bytes = None
        current_image_mime_type = None
        current_image_name = None
        if uploaded_image is not None:
            current_image_bytes = uploaded_image.getvalue()
            current_image_mime_type = uploaded_image.type
            current_image_name = uploaded_image.name
            st.image(current_image_bytes, caption=current_image_name)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            outputs = run_models_parallel(
                model_entries=cleaned_entries,
                turns=st.session_state.turns,
                prompt=prompt,
                image_bytes=current_image_bytes,
                image_mime_type=current_image_mime_type,
            )
        response_map = {item["model"]: item["output"] for item in outputs}
        columns = st.columns(len(current_models))
        for col, model in zip(columns, current_models):
            with col:
                st.markdown(f"#### `{model}`")
                st.write(response_map.get(model, "No response returned for this model."))

    st.session_state.turns.append(
        {
            "prompt": prompt,
            "responses": response_map,
            "image_bytes": current_image_bytes,
            "image_mime_type": current_image_mime_type,
            "image_name": current_image_name,
        }
    )
    st.session_state.image_uploader_nonce += 1


if __name__ == "__main__":
    main()

