# -*- coding: utf-8 -*-

# === SLOT 1 — Setup for TinyLlama SSE Streaming ===

# ---------- Imports ----------
import os, json, asyncio, threading
from pathlib import Path
from uuid import uuid4
import nest_asyncio
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from pyngrok import ngrok, conf
import uvicorn
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
import commentjson
from pathlib import Path

from retriever import CorpusIndex, load_corpus_from_schema, format_hits_as_context

# ---------- ngrok auth token ----------
# Load environment variables from .env
load_dotenv()

# Get the token safely
auth_token = os.getenv("AUTH_TOKEN")

# Use it in your config
conf.get_default().auth_token = auth_token

# ---------- FastAPI app ----------
app = FastAPI()

# ---------- Loading .json and setting and loading the model ----------
CONFIG_FILE = "config.json"  # path to your JSON

def load_config(path: str = CONFIG_FILE) -> dict:
    """Load config.json (supports // and /* */ comments)."""
    with open(path, "r", encoding="utf-8") as f:
        return commentjson.load(f)

def select_default_model(cfg: dict) -> dict:
    """Return the model dict referenced by default_model."""
    models = {m["key"]: m for m in cfg.get("models", [])}
    key = cfg.get("default_model")
    if not key or key not in models:
        raise RuntimeError(f"default_model '{key}' not found in config.json")
    return models[key]

def prune_none(d: dict) -> dict:
    """Remove keys with value None so llama_cpp uses its defaults."""
    return {k: v for k, v in d.items() if v is not None}

# ---------- Setting the model ----------
cfg = load_config()
model_cfg = select_default_model(cfg)

MODEL_URL  = model_cfg["url"]
MODEL_PATH = model_cfg["path"]
LLAMA_KW   = prune_none(model_cfg.get("llama_kw", {}))

# Auto thread fallback if not set
if "n_threads" not in LLAMA_KW:
    LLAMA_KW["n_threads"] = max(1, (os.cpu_count() or 4) - 2)

# Ensure model directory exists
Path(os.path.dirname(MODEL_PATH)).mkdir(parents=True, exist_ok=True)

# Download if missing
if not os.path.exists(MODEL_PATH):
    import urllib.request
    print(f"⏳ Downloading model:\n{MODEL_URL}\n-> {MODEL_PATH}")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("✅ Downloaded.")

# ---------- Initialize Llama ----------
_llm = None
def _get_llm():
    """Lazy-load the Llama model."""
    global _llm
    if _llm is None:
        from llama_cpp import Llama
        assert os.path.exists(MODEL_PATH), f"Model not found at {MODEL_PATH}"
        _llm = Llama(model_path=MODEL_PATH, **LLAMA_KW)
    return _llm

# Optional confirmation print
print(f"🔧 Using model '{model_cfg['key']}'")

# ---------- Server lifecycle globals ----------
SERVER_THREAD = None
UVICORN_SERVER = None
PUBLIC_TUNNEL = None
PORT = 8000

# ---------- Constants ----------
STREAM_PATH = "/studybot/stream"

# ---------- Conversation store (per-session) ----------
CONV_DIR = Path("./conversations")
CONV_DIR.mkdir(exist_ok=True)
SYSTEM_PROMPT = """
You are StudyBot, a friendly and knowledgeable study advisor for
The Hague University of Applied Sciences (THUAS).

Your goal is to help prospective students find suitable study programs
based on their interests, preferences, and questions.

---

### General Rules
1. Base every answer ONLY on the data provided in the "Relevant programs" context.
   - If information is not in the data, say you don’t know.
   - Never invent or guess program names, descriptions, details, or URLs.
   - Copy all factual fields (program name, URL, level, variant) exactly as shown.
2. Keep your responses concise, natural, and conversational — respond like a real advisor, not a database.
3. Use clear bullet points or short paragraphs to organize information.
4. When a user’s question is too broad or general, ask friendly follow-up questions to clarify their preferences before giving results. Examples:
   - "Are you looking for a Bachelor or a Master program?"
   - "Do you prefer full-time or part-time studies?"
   - "Which field interests you most: ICT, Business, or Design?"
   - "Would you like me to focus on programs with strong career prospects or student satisfaction?"
5. When the search is specific enough, present up to the top 3 matching programs using the structured format below.
6. Maintain a natural conversation with the user. Acknowledge their answers and guide them smoothly toward relevant programs.
7. If the user changes their mind (e.g., “Actually, I want a Master instead”), update filters and refine the next search.
8. Always reply in English with a helpful and positive tone.
9. When unsure or when information is missing, say so politely and ask for clarification.

---

### Response Format
StudyBot Recommendation:
- Program: <program_name> (<program_name_en>)
  • Degree type: <program_type>
  • Variant: <program_variant>
  • Cluster: <cluster>
  • Summary: <short 1–3 line description or key point>
  • Conformity: <short 1–3 line explanation of why this program fits the user's interests or criteria>
  • URL: <url>
  • Program ID: <program_id>
  • Source: [<program_id>]

If there are multiple programs, repeat the bullet section for each.

When answering factual questions (for example, about career prospects or details),
use only the text from the "Relevant programs" context and cite the relevant program_id(s) as sources.

When the user's query is too broad, respond naturally and ask one or two
clarifying questions before suggesting any programs.

---

### Remember
You are an official representative of THUAS.
Be accurate, polite, and conversational.
Engage the user naturally, but stay fully grounded in the provided data.
"""

# In-memory: { session_id: [ {role, content}, ... ] }
CONV_STORE: dict[str, list[dict]] = {}

# Locks per session to avoid interleaved writes
SESS_LOCKS: dict[str, asyncio.Lock] = {}
def _get_lock(session_id: str) -> asyncio.Lock:
    lock = SESS_LOCKS.get(session_id)
    if lock is None:
        lock = asyncio.Lock()
        SESS_LOCKS[session_id] = lock
    return lock

def _conv_path(session_id: str) -> Path:
    return CONV_DIR / f"{session_id}.json"

def load_conv(session_id: str) -> list[dict]:
    p = _conv_path(session_id)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return [{"role":"system","content": SYSTEM_PROMPT}]

def save_conv(session_id: str, messages: list[dict]):
    _conv_path(session_id).write_text(json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8")

def get_messages(session_id: str) -> list[dict]:
    msgs = CONV_STORE.get(session_id)
    if msgs is None:
        msgs = load_conv(session_id)
        CONV_STORE[session_id] = msgs
    return msgs

def prune_messages_for_context(messages, max_chars=None, ctx_tokens=16384, chars_per_tok=4.0, reserve_frac=0.2):
    # keep ~80% of context for prompt; leave 20% for the model’s reply & system
    if max_chars is None:
        max_prompt_toks = int(ctx_tokens * (1 - reserve_frac))
        max_chars = int(max_prompt_toks * chars_per_tok)

    sys = [m for m in messages if m["role"] == "system"][:1]
    convo = [m for m in messages if m["role"] != "system"]
    total = sum(len(m["content"]) for m in sys)
    acc = []
    for m in reversed(convo):
        c = len(m["content"])
        if total + c > max_chars and acc:
            break
        acc.append(m)
        total += c
    return sys + list(reversed(acc))


_INDEX = None
_EMBED_CAPABLE = None

def _resolve_schema_path() -> str:
    candidates = [
        os.getenv("SCHEMA_PATH"),
        str(Path(__file__).parent / "schema.json"),
        "/mnt/data/processed/schema.json",
    ]
    for p in candidates:
        if p and os.path.exists(p):
            return p
    raise RuntimeError(
        "schema.json not found. Set SCHEMA_PATH or place schema.json next to main.py."
    )

def _ensure_index():
    """
    Build a tiny hybrid index (keyword + optional vector) once and cache it.
    Uses llama.cpp embeddings if available; falls back to keyword-only.
    """
    global _INDEX, _EMBED_CAPABLE
    if _INDEX is not None:
        return _INDEX

    schema_path = _resolve_schema_path()
    docs = load_corpus_from_schema(schema_path)

    # Try to expose an embed() backed by llama.cpp
    llm = _get_llm()
    embed_fn = None
    try:
        # llama.cpp-python: create_embedding(input=[...]) -> {"data":[{"embedding":[...]}]}
        def _embed(batch_texts):
            out = llm.create_embedding(input=batch_texts)
            return [e["embedding"] for e in out["data"]]

        # quick sanity check
        _ = _embed(["ok"])
        _EMBED_CAPABLE = True
        embed_fn = _embed
    except Exception:
        _EMBED_CAPABLE = False
        embed_fn = None  # keyword-only fallback

    _INDEX = CorpusIndex(docs, embed_fn=embed_fn)
    return _INDEX


def build_messages_with_context(history: list[dict], user_prompt: str) -> list[dict]:
    """
    Runs retrieval for the latest user prompt, injects a 'Relevant programs' context
    into the system message (prepended so your prune keeps it), then returns messages.
    """
    index = _ensure_index()

    # (Optional) parse filters from your own state; keep empty for now
    filters = {}  # e.g. {"program_type": "Bachelor", "cluster": "Engineering"}

    hits = index.search(user_prompt, top_k=6, filters=filters)
    context_block = format_hits_as_context(hits, max_chars=2800)

    system_with_context = {
        "role": "system",
        "content": (
            SYSTEM_PROMPT.strip()
            + "\n\n----\nRelevant programs (top matches; ONLY use these as your source):\n"
            + context_block
        ),
    }

    # Rebuild: our new system message FIRST, then prior (non-system) history, then this user turn
    non_system = [m for m in history if m["role"] != "system"]
    run_messages = [system_with_context] + non_system + [{"role": "user", "content": user_prompt}]
    return run_messages


# Clear any previous routes and (re)register
app.router.routes.clear()

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

# ---- HTML page: simple multi-turn chat with per-session memory ----
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "stream_path": STREAM_PATH}
    )

# ---- SSE endpoint: stream tokens and persist assistant turn ----
@app.get(STREAM_PATH)
async def tinyllama_stream(prompt: str = Query(...), session_id: str = Query(...)):
    async def gen():
        llm = _get_llm()
        lock = _get_lock(session_id)

        async with lock:
            history = get_messages(session_id)

            # >>> NEW: Build retrieval-augmented messages
            run_messages = build_messages_with_context(history, prompt)

            try:
                stream = llm.create_chat_completion(
                    messages=prune_messages_for_context(run_messages),
                    stream=True,
                    temperature=0.2,     # lower temp for grounded answers
                    max_tokens=512,
                )
                chunks = []
                for chunk in stream:
                    delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
                    tok = delta.get("content", "")
                    if tok:
                        chunks.append(tok)
                        yield f"data: {json.dumps({'token': tok})}\n\n"
                    await asyncio.sleep(0)

                assistant_text = "".join(chunks)

                # Save *only* non-system turns in your history
                history.append({"role": "user", "content": prompt})
                history.append({"role": "assistant", "content": assistant_text})
                save_conv(session_id, history)

                yield "event: done\ndata: ok\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                yield "event: done\ndata: error\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )

# ---- Helpers: view/reset/export conversation ----
@app.get("/history", response_class=JSONResponse)
def get_history(session_id: str = Query(...)):
    return {"session_id": session_id, "messages": get_messages(session_id)}

@app.post("/reset", response_class=JSONResponse)
def reset_history(session_id: str = Query(...)):
    CONV_STORE[session_id] = [{"role":"system","content": SYSTEM_PROMPT}]
    save_conv(session_id, CONV_STORE[session_id])
    return {"ok": True}

@app.get("/export", response_class=HTMLResponse)
def export_history(session_id: str = Query(...)):
    p = _conv_path(session_id)
    if not p.exists():
        save_conv(session_id, get_messages(session_id))
    return HTMLResponse(
        content=f"<pre>{p.read_text(encoding='utf-8')}</pre>",
        headers={"Content-Disposition": f'attachment; filename=\"{session_id}.json\"'}
    )

# === SLOT 3 — Start/stop Uvicorn + ngrok safely ===

SERVER_THREAD = None
UVICORN_SERVER = None
PUBLIC_TUNNEL = None

# ---------- Main entry point ----------
if __name__ == "__main__":
    # Allow nested event loops (needed if already running async / notebooks)
    nest_asyncio.apply()

    # Optional: ensure ngrok is authed up-front (if needed)
    # ngrok.set_auth_token(os.environ["NGROK_TOKEN"])

    # 1) Close old tunnels (useful in notebooks / reused processes)
    try:
        ngrok.kill()
    except Exception:
        pass

    # 2) Start a fresh public tunnel for this run
    PUBLIC_TUNNEL = ngrok.connect(PORT, "http")  # or ngrok.connect(addr=PORT, proto="http")
    print("🔗 To chat with the model open the URL below")
    print("🔗 Public URL:", PUBLIC_TUNNEL.public_url)

    try:
        # 3) Run the Uvicorn server in the MAIN thread so Ctrl+C works correctly
        config = uvicorn.Config(app, host="0.0.0.0", port=PORT, log_level="info")
        uvicorn.Server(config).run()
    except KeyboardInterrupt:
        # 4) Graceful shutdown on Ctrl+C
        print("\n🛑 Server interrupted by user, shutting down...")
    finally:
        # 5) Always clean up ngrok tunnels on exit
        try:
            ngrok.kill()
        except Exception:
            pass