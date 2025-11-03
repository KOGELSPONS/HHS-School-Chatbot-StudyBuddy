import streamlit as st
import uuid
import requests
import os
import json
from dotenv import load_dotenv

load_dotenv()

BACKEND_BASE = os.getenv("BACKEND_BASE", "http://localhost:5000")
STREAM_PATH = os.getenv("STREAM_PATH", "/studybot/stream")

st.set_page_config(page_title="StudyBot Chat", page_icon="💬", layout="centered")
GREETING = "Hi, I'm your study chatbot. Ask me anything related to choosing your major."

# --- State ---
if "sid" not in st.session_state:
    st.session_state.sid = str(uuid.uuid4())
if "chat" not in st.session_state:
    st.session_state.chat = []
if "chats_meta" not in st.session_state:
    st.session_state.chats_meta = [{"sid": st.session_state.sid, "title": "New chat"}]

# --- Small API helpers ---
def api_history(session_id: str):
    r = requests.get(f"{BACKEND_BASE}/history", params={"session_id": session_id}, timeout=20)
    r.raise_for_status()
    return r.json().get("messages", [])

def api_reset(session_id: str):
    r = requests.post(f"{BACKEND_BASE}/reset", params={"session_id": session_id}, timeout=20)
    r.raise_for_status()
    return r.json()

# --- util: inject the greeting if there is no assistant yet ---
def ensure_greeting():
    chat = st.session_state.chat or []
    has_assistant = any(m.get("role") == "assistant" for m in chat)
    if not has_assistant:
        st.session_state.chat = [{"role": "assistant", "content": GREETING}] + chat

# --- load chat history---
def load_current_history():
    try:
        st.session_state.chat = api_history(st.session_state.sid)
    except Exception as e:
        st.session_state.chat = []
        st.error(f"Failed to load history: {e}")

# on the first run, load the history of the current sid
if not st.session_state.chat:
    load_current_history()

# inject the greeting if no assistant exists yet
ensure_greeting()

# --- Sidebar: chat list + actions ---
with st.sidebar:
    st.markdown("### Chats")

    # New chat = new session_id + named
    if st.button("New chat", icon=":material/add:", use_container_width=True, key="new_chat_btn"):
        new_sid = str(uuid.uuid4())

        from datetime import datetime
        new_title = f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')} • {new_sid[:4]}"
        st.session_state.sid = new_sid
        st.session_state.chats_meta.insert(0, {"sid": new_sid, "title": new_title})
        try:
            requests.post(
                f"{BACKEND_BASE}/reset",
                params={"session_id": new_sid},
                timeout=10
            )
            # load fresh history
            r = requests.get(f"{BACKEND_BASE}/history", params={"session_id": new_sid}, timeout=10)
            st.session_state.chat = r.json().get("messages", [])
        except Exception:
            st.session_state.chat = []
        ensure_greeting()
        st.rerun()

    # List of chats
    for meta in st.session_state.chats_meta:
        is_active = (meta["sid"] == st.session_state.sid)

        if st.button(
            meta.get("title") or "Chat",
            key=f"chat-{meta['sid']}",
            type=("primary" if is_active else "secondary"),
            use_container_width=True,
            icon=":material/chat:"
        ):
            if not is_active:
                st.session_state.sid = meta["sid"]
                try:
                    r = requests.get(f"{BACKEND_BASE}/history", params={"session_id": meta['sid']}, timeout=10)
                    st.session_state.chat = r.json().get("messages", [])
                except Exception:
                    st.session_state.chat = []
                ensure_greeting()
                st.rerun()

    # reset current chat
    if st.button("Reset current chat", icon=":material/cleaning_services:", use_container_width=True):
        try:
            requests.post(f"{BACKEND_BASE}/reset", params={"session_id": st.session_state.sid}, timeout=10)
            r = requests.get(f"{BACKEND_BASE}/history", params={"session_id": st.session_state.sid}, timeout=10)
            st.session_state.chat = r.json().get("messages", [])
            ensure_greeting()
            st.rerun()
        except Exception as e:
            st.error(f"Reset failed: {e}")

st.title("StudyBot Chat")

# --- History display (from /history?session_id=SID) ---
for msg in st.session_state.chat:
    # ignore “system” messages coming from the backend
    if msg.get("role") == "system":
        continue
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Main stream, pass the current SID ---
def stream_reply(prompt: str):
    with requests.get(
        f"{BACKEND_BASE}{STREAM_PATH}",
        params={"prompt": prompt, "session_id": st.session_state.sid},
        stream=True,
        timeout=60,
    ) as r:
        r.raise_for_status()
        for chunk in r.iter_lines(decode_unicode=True):
            if not chunk:
                continue
            line = chunk.strip()
            if line.startswith("event:"):
                if line.split(":", 1)[1].strip() == "done":
                    break
                continue
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            tok = obj.get("token")
            if tok is not None:
                yield tok

# --- Input & stream ---
prompt = st.chat_input("Type your message…")
if prompt:
    # 1) the user message is displayed
    st.session_state.chat.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2) we're streaming the answer
    with st.chat_message("assistant"):
        placeholder = st.empty()
        acc = ""
        try:
            for tok in stream_reply(prompt):
                acc += tok
                placeholder.markdown(acc)
        except Exception as e:
            acc += f"\n\n_Streaming error: {e}_"
            placeholder.markdown(acc)

    # 3) we save on the UI side
    st.session_state.chat.append({"role": "assistant", "content": acc})

    # 4) title the chat if necessary (front-only)
    cur = next((c for c in st.session_state.chats_meta if c["sid"] == st.session_state.sid), None)
    if cur and (not cur.get("title") or cur["title"] == "New chat"):
        cur["title"] = (prompt[:40] + "…") if len(prompt) > 40 else prompt

    st.rerun()
