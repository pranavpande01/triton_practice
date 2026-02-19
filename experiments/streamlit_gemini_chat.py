import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Any
import mimetypes
from datetime import datetime
import ctypes
import ctypes.wintypes as wintypes

import streamlit as st
from google import genai
from google.genai import types
from PIL import ImageGrab

DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_IMAGE_FOLDER = Path(r"C:\Users\Pranav\Desktop\proj\github_projects\triton_practice\images")


def get_message_images(message: Dict[str, Any]) -> List[Dict[str, Any]]:
    images = message.get("images")
    if images:
        return images
    image_bytes = message.get("image_bytes")
    image_mime_type = message.get("image_mime_type")
    image_name = message.get("image_name", "Uploaded image")
    if image_bytes and image_mime_type:
        return [{"bytes": image_bytes, "mime_type": image_mime_type, "name": image_name}]
    return []


def load_images_from_folder(folder: Path) -> List[Dict[str, Any]]:
    if not folder.exists() or not folder.is_dir():
        return []
    supported_suffixes = {".png", ".jpg", ".jpeg", ".webp"}
    images: List[Dict[str, Any]] = []
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.suffix.lower() not in supported_suffixes:
            continue
        mime_type = mimetypes.guess_type(path.name)[0]
        if not mime_type:
            continue
        images.append({"bytes": path.read_bytes(), "mime_type": mime_type, "name": path.name})
    return images


def clear_images_in_folder(folder: Path) -> int:
    if not folder.exists() or not folder.is_dir():
        return 0
    supported_suffixes = {".png", ".jpg", ".jpeg", ".webp"}
    deleted = 0
    for path in folder.iterdir():
        if not path.is_file() or path.suffix.lower() not in supported_suffixes:
            continue
        path.unlink(missing_ok=True)
        deleted += 1
    return deleted


def find_matching_window(window_title: str) -> Tuple[int, str] | None:
    query = window_title.strip().lower()
    if not query:
        return None

    user32 = ctypes.windll.user32
    candidates: List[Tuple[int, str]] = []
    foreground = user32.GetForegroundWindow()

    @ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
    def enum_windows_proc(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        if user32.IsIconic(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buffer, length + 1)
        title_raw = buffer.value.strip()
        title = title_raw.lower()
        if query in title:
            candidates.append((hwnd, title_raw))
        return True

    user32.EnumWindows(enum_windows_proc, 0)
    if not candidates:
        return None

    for hwnd, title in candidates:
        if hwnd == foreground:
            return hwnd, title

    return candidates[0]


def get_client_bbox(hwnd: int) -> Tuple[int, int, int, int] | None:
    rect = wintypes.RECT()
    ok = ctypes.windll.user32.GetClientRect(hwnd, ctypes.byref(rect))
    if not ok:
        return None
    top_left = wintypes.POINT(rect.left, rect.top)
    bottom_right = wintypes.POINT(rect.right, rect.bottom)
    if not ctypes.windll.user32.ClientToScreen(hwnd, ctypes.byref(top_left)):
        return None
    if not ctypes.windll.user32.ClientToScreen(hwnd, ctypes.byref(bottom_right)):
        return None
    if bottom_right.x <= top_left.x or bottom_right.y <= top_left.y:
        return None
    return top_left.x, top_left.y, bottom_right.x, bottom_right.y


def capture_window_screenshot_to_folder(window_title: str, folder: Path) -> Tuple[bool, str]:
    match = find_matching_window(window_title)
    if match is None:
        return False, f"Window not found: {window_title}"
    hwnd, matched_title = match
    bbox = get_client_bbox(hwnd)
    if bbox is None:
        return False, f"Could not read client area for window: {matched_title}"
    folder.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = folder / f"window_capture_{timestamp}.png"
    image = ImageGrab.grab(bbox=bbox, all_screens=True)
    if image.getbbox() is None:
        return False, f"Captured blank image for window: {matched_title}"
    image.save(output_path)
    return True, f"{output_path} (matched: {matched_title})"


def message_to_parts(message: Dict[str, Any]) -> List[types.Part]:
    parts: List[types.Part] = []
    text = message.get("content", "")
    if text:
        parts.append(types.Part(text=text))
    for image in get_message_images(message):
        parts.append(types.Part.from_bytes(data=image["bytes"], mime_type=image["mime_type"]))
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
    current_images: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    for turn in turns:
        user_message: Dict[str, Any] = {"role": "user", "content": turn["prompt"]}
        turn_images = get_message_images(turn)
        if turn_images:
            user_message["images"] = turn_images
        messages.append(user_message)
        model_response = turn["responses"].get(model)
        if model_response:
            messages.append({"role": "assistant", "content": model_response})

    latest_user_message: Dict[str, Any] = {"role": "user", "content": prompt}
    if current_images:
        latest_user_message["images"] = current_images
    messages.append(latest_user_message)
    return messages


def run_model(
    api_key: str,
    model: str,
    turns: List[Dict[str, Any]],
    prompt: str,
    current_images: List[Dict[str, Any]],
) -> Tuple[str, str]:
    messages = build_messages_for_model(
        turns=turns,
        model=model,
        prompt=prompt,
        current_images=current_images,
    )
    try:
        return model, call_gemini(api_key=api_key, model=model, messages=messages)
    except Exception as exc:
        return model, f"Request failed: {exc}"


def run_models_parallel(
    model_entries: List[Dict[str, str]],
    turns: List[Dict[str, Any]],
    prompt: str,
    current_images: List[Dict[str, Any]],
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
                current_images,
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


def run_models_parallel_progress(
    model_entries: List[Dict[str, str]],
    turns: List[Dict[str, Any]],
    prompt: str,
    current_images: List[Dict[str, Any]],
):
    workers = min(max(len(model_entries), 1), 8)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                run_model,
                entry["api_key"],
                entry["model"],
                turns,
                prompt,
                current_images,
            ): entry["model"]
            for entry in model_entries
        }
        for future in as_completed(futures):
            model = futures[future]
            _, output = future.result()
            yield model, output


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
            turn_images = get_message_images(turn)
            for image in turn_images:
                st.image(image["bytes"], caption=image.get("name", "Uploaded image"))
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
    st.markdown(
        """
        <style>
        [data-testid="column"] {
          min-width: 0;
          overflow: hidden;
        }
        [data-testid="column"] .stMarkdown,
        [data-testid="column"] .stCodeBlock,
        [data-testid="column"] pre,
        [data-testid="column"] code {
          white-space: pre-wrap !important;
          overflow-wrap: anywhere !important;
          word-break: break-word !important;
        }
        [data-testid="column"] .stMarkdown table {
          display: block !important;
          max-width: 100% !important;
          overflow-x: auto !important;
          white-space: nowrap !important;
          border-collapse: collapse;
        }
        [data-testid="column"] .stMarkdown thead,
        [data-testid="column"] .stMarkdown tbody,
        [data-testid="column"] .stMarkdown tr {
          display: table;
          width: max-content;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if "model_entries" not in st.session_state:
        st.session_state.model_entries = [
            {"model": DEFAULT_MODEL, "api_key": os.getenv("GEMINI_API_KEY", "")}
        ]
    if "turns" not in st.session_state:
        st.session_state.turns = []
    if "use_folder_images_next" not in st.session_state:
        st.session_state.use_folder_images_next = False
    if "capture_window_query" not in st.session_state:
        st.session_state.capture_window_query = ""
    if "capture_feedback" not in st.session_state:
        st.session_state.capture_feedback = ""
    if "capture_feedback_kind" not in st.session_state:
        st.session_state.capture_feedback_kind = "info"

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

        st.markdown("### Window Capture")
        st.session_state.capture_window_query = st.text_input(
            "Target window title",
            value=st.session_state.capture_window_query,
            help="Partial title match is supported (example: Chrome, Notepad, Visual Studio Code).",
        )

    is_valid, err_msg, cleaned_entries = validate_model_entries(st.session_state.model_entries)
    if not is_valid:
        st.warning(err_msg)
    else:
        st.session_state.model_entries = cleaned_entries

    current_models = [entry["model"] for entry in st.session_state.model_entries]
    if current_models:
        render_chat(turns=st.session_state.turns, current_models=current_models)

    controls_left, controls_mid, controls_right = st.columns([4, 1, 1])
    with controls_mid:
        if st.button("Img+", help=f"Attach all images from {DEFAULT_IMAGE_FOLDER} to next sent message."):
            st.session_state.use_folder_images_next = True
    with controls_right:
        if st.button("Win+", help="Capture screenshot from target window into images folder."):
            ok, result = capture_window_screenshot_to_folder(
                window_title=st.session_state.capture_window_query,
                folder=DEFAULT_IMAGE_FOLDER,
            )
            if ok:
                st.session_state.capture_feedback = f"Saved screenshot: {result}"
                st.session_state.capture_feedback_kind = "success"
            else:
                st.session_state.capture_feedback = result
                st.session_state.capture_feedback_kind = "error"
    with controls_left:
        if st.session_state.use_folder_images_next:
            st.caption(f"Folder images will be attached on next send: `{DEFAULT_IMAGE_FOLDER}`")
        if st.session_state.capture_feedback:
            if st.session_state.capture_feedback_kind == "success":
                st.success(st.session_state.capture_feedback)
            elif st.session_state.capture_feedback_kind == "error":
                st.error(st.session_state.capture_feedback)
            else:
                st.info(st.session_state.capture_feedback)

    chat_input_value = st.chat_input(
        "Type your message...",
        accept_file="multiple",
        file_type=["png", "jpg", "jpeg", "webp"],
    )
    if not chat_input_value:
        return

    prompt = ""
    current_images: List[Dict[str, Any]] = []
    if isinstance(chat_input_value, str):
        prompt = chat_input_value
    else:
        prompt = chat_input_value.text
        if chat_input_value.files:
            for file in chat_input_value.files:
                if file.type:
                    current_images.append(
                        {"bytes": file.getvalue(), "mime_type": file.type, "name": file.name}
                    )

    if st.session_state.use_folder_images_next:
        folder_images = load_images_from_folder(DEFAULT_IMAGE_FOLDER)
        if folder_images:
            current_images.extend(folder_images)
            deleted_count = clear_images_in_folder(DEFAULT_IMAGE_FOLDER)
            st.caption(f"Attached {len(folder_images)} folder image(s) and cleared {deleted_count} file(s).")
            st.session_state.use_folder_images_next = False
        else:
            st.session_state.use_folder_images_next = False
            st.warning(f"No supported images found in `{DEFAULT_IMAGE_FOLDER}`.")

    if not is_valid:
        st.error(err_msg)
        return
    if not prompt.strip() and not current_images:
        st.error("Message cannot be empty. Enter text or attach an image.")
        return
    if not prompt.strip() and current_images:
        prompt = "Please analyze this image."

    with st.chat_message("user"):
        st.markdown(prompt)
        for image in current_images:
            st.image(image["bytes"], caption=image.get("name", "Uploaded image"))

    with st.chat_message("assistant"):
        response_map: Dict[str, str] = {}
        columns = st.columns(len(current_models))
        placeholders: Dict[str, Any] = {}
        for col, model in zip(columns, current_models):
            with col:
                st.markdown(f"#### `{model}`")
                placeholders[model] = st.empty()
                placeholders[model].caption("Waiting...")

        with st.spinner("Thinking..."):
            for model, output in run_models_parallel_progress(
                model_entries=cleaned_entries,
                turns=st.session_state.turns,
                prompt=prompt,
                current_images=current_images,
            ):
                response_map[model] = output
                placeholders[model].write(output)

        for model in current_models:
            if model not in response_map:
                response_map[model] = "No response returned for this model."
                placeholders[model].write(response_map[model])

    st.session_state.turns.append(
        {
            "prompt": prompt,
            "responses": response_map,
            "images": current_images,
        }
    )


if __name__ == "__main__":
    main()

