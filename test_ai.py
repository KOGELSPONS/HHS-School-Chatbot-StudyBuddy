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
import subprocess, signal, gc
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

# ---------- ngrok auth token ----------
# Get one at: https://dashboard.ngrok.com/get-started/your-authtoken
conf.get_default().auth_token = "34RT26Lqxaa1azQj9tuezgdRPyz_72syymZdPnwFoAXcmuPqA"

# ---------- FastAPI app ----------
app = FastAPI()

# ---------- TinyLlama model (small & fast) ----------
MODEL_URL  = "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q6_K_L.gguf" # Test model
MODEL_PATH = "models\Llama-3.2-3B-Instruct-Q6_K_L.gguf"

if not os.path.exists(MODEL_PATH):
    import urllib.request
    print("⏳ Downloading model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("✅ Downloaded.")

LLAMA_KW = dict(
    # model="Llama-3.2-3B-Instruct-Q6_K_L.gguf",  # wherever you load this
    n_ctx=8192,                # safe default for GGUF quant; raise only if your build/quant supports it
    n_threads=os.cpu_count(),  # OK to keep; can cap to physical cores if you see contention
    n_batch=1024,               # throughput; 256–1024 works well for 3B on modern CPUs
    n_gpu_layers=0,            # CPU-only
    f16_kv=True,               # faster KV cache
    use_mmap=True,             # quick-load when available
    use_mlock=True,           # True only if you have RAM to pin (reduces paging)
    verbose=False,
)

_llm = None
def _get_llm():
    global _llm
    if _llm is None:
        from llama_cpp import Llama
        assert os.path.exists(MODEL_PATH), f"Model not found at {MODEL_PATH}"
        _llm = Llama(model_path=MODEL_PATH, **LLAMA_KW)
    return _llm

# ---------- Server lifecycle globals ----------
SERVER_THREAD = None
UVICORN_SERVER = None
PUBLIC_TUNNEL = None
PORT = 8000

# ---------- Constants ----------
STREAM_PATH = "/tinyllama/stream"

# ---------- Conversation store (per-session) ----------
CONV_DIR = Path("./conversations")
CONV_DIR.mkdir(exist_ok=True)
SYSTEM_PROMPT = "You are a helpful assistant."

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

def prune_messages_for_context(messages, max_chars=None, ctx_tokens=8192, chars_per_tok=4.0, reserve_frac=0.2):
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

# Clear any previous routes and (re)register
app.router.routes.clear()

templates = Jinja2Templates(directory="/templates")

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
            # Append user turn
            history.append({"role": "user", "content": prompt})
            # Prune to fit context
            run_messages = prune_messages_for_context(history)

            try:
                stream = llm.create_chat_completion(
                    messages=run_messages,
                    stream=True,
                    temperature=0.7,
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
                # Save assistant turn when complete
                assistant_text = "".join(chunks)
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

if __name__ == "__main__":
    nest_asyncio.apply()

    # Optional: ensure ngrok is authed up-front (if needed)
    # ngrok.set_auth_token(os.environ["NGROK_TOKEN"])

    # 2) Close old tunnels (useful in notebooks / reused processes)
    try:
        ngrok.kill()
    except Exception:
        pass

    # 3) Start fresh tunnel
    PUBLIC_TUNNEL = ngrok.connect(PORT, "http")  # or ngrok.connect(addr=PORT, proto="http")
    print("🔗 To chat with the model open the URL below")
    print("🔗 Public URL:", PUBLIC_TUNNEL.public_url)

    # 4) Launch Uvicorn in a background thread
    config = uvicorn.Config(app, host="0.0.0.0", port=PORT, log_level="info")
    UVICORN_SERVER = uvicorn.Server(config)

    def run_server():
        UVICORN_SERVER.run()

    SERVER_THREAD = threading.Thread(target=run_server, daemon=True)
    SERVER_THREAD.start()

    try:
        SERVER_THREAD.join()
    except KeyboardInterrupt:
        pass
    finally:
        # graceful shutdown if the process stays alive
        try:
            if UVICORN_SERVER is not None:
                UVICORN_SERVER.should_exit = True
        except Exception:
            pass
        try:
            ngrok.kill()
        except Exception:
            pass
