"""ElectraGuard chatbot backend — FastAPI port of the Node/Express gemini-backend."""
import os
import time
import asyncio
import logging
from collections import OrderedDict, defaultdict, deque
from contextlib import asynccontextmanager
from urllib.parse import parse_qs

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

import gemini_client as gemini
from rag_engine import KnowledgeBase

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("server")

GENERATION_MODELS = [
    m.strip()
    for m in os.getenv(
        "GENERATION_MODELS",
        "gemini-2.5-flash,gemini-2.0-flash,gemini-3.5-flash",
    ).split(",")
    if m.strip()
]
DEGRADED_FALLBACK = os.getenv("DEGRADED_FALLBACK", "true").lower() == "true"
KEEPALIVE_URL = os.getenv("KEEPALIVE_URL", "https://electraguard-backend-1.onrender.com")
RATE_LIMIT, RATE_WINDOW = 100, 60  # 100 requests / minute / IP
CACHE_MAX = 100

SYSTEM_PREAMBLE = """You are ElectraGuard's assistant. ElectraGuard is an electricity theft
detection and consumer-support app for Pakistan. Answer the user's question in a clear,
friendly, and concise way.

If context from Pakistani electricity utility sources (DISCOs, NEPRA, WAPDA, K-Electric) is
provided below, base your answer on it and mention which utility/source the info is from when
relevant. If no context is provided, or the context doesn't cover the question, answer using
your general knowledge but make clear you're not quoting an official source, and suggest the
user verify with their local DISCO or NEPRA for anything bill/tariff specific."""

kb = KnowledgeBase()
reply_cache: "OrderedDict[str, dict]" = OrderedDict()
embed_cache: "OrderedDict[str, list[float]]" = OrderedDict()
hits: dict[str, deque] = defaultdict(deque)


def lru_set(cache: OrderedDict, key, value):
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > CACHE_MAX:
        cache.popitem(last=False)


async def keepalive_loop(client: httpx.AsyncClient):
    while True:
        await asyncio.sleep(10 * 60)
        try:
            r = await client.get(KEEPALIVE_URL)
            log.info("Keep-alive ping: %s", r.status_code)
        except Exception as e:
            log.error("Keep-alive error: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    kb.load()
    app.state.http = httpx.AsyncClient(timeout=30)
    task = asyncio.create_task(keepalive_loop(app.state.http)) if KEEPALIVE_URL else None
    yield
    if task:
        task.cancel()
    await app.state.http.aclose()


app = FastAPI(title="Gemini + RAG backend", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# Keep the same {"error": "..."} shape the frontend already expects (FastAPI default is {"detail": ...})
@app.exception_handler(HTTPException)
async def http_exc(_: Request, exc: HTTPException):
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code, headers=exc.headers)


def client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")  # behind Render's proxy
    return fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "unknown")


def check_rate_limit(request: Request):
    ip, now = client_ip(request), time.time()
    q = hits[ip]
    while q and now - q[0] > RATE_WINDOW:
        q.popleft()
    if len(q) >= RATE_LIMIT:
        raise HTTPException(429, "Too many requests. Please wait.", headers={"Retry-After": str(RATE_WINDOW)})
    q.append(now)


async def read_message(request: Request):
    """Accepts JSON or urlencoded bodies, like express.json() + express.urlencoded()."""
    ctype = request.headers.get("content-type", "")
    try:
        if "application/x-www-form-urlencoded" in ctype:
            body = {k: v[0] for k, v in parse_qs((await request.body()).decode()).items()}
        else:
            body = await request.json()
    except Exception:
        body = {}
    return body.get("message") if isinstance(body, dict) else None


@app.api_route("/", methods=["GET", "HEAD"])
async def root():
    return PlainTextResponse("Gemini + RAG backend is running 🚀")

    
@app.post("/chat")
async def chat(request: Request):
    check_rate_limit(request)

    message = await read_message(request)
    if not message or not isinstance(message, str):
        raise HTTPException(400, "Message required")
    if len(message) > 2000:
        raise HTTPException(400, "Message too long.")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        log.error("GEMINI_API_KEY is not set")
        raise HTTPException(500, "Server error")

    # Normalised cache key: "Hello?" and " hello? " cost one Gemini call, not two
    key = " ".join(message.lower().split())
    if key in reply_cache:
        reply_cache.move_to_end(key)
        return reply_cache[key]

    http: httpx.AsyncClient = request.app.state.http

    # 1) Retrieve context (embedding failures must not kill the request)
    chunks, sources = [], []
    if kb.ready:
        try:
            vec = embed_cache.get(key)
            if vec is None:
                vec = await gemini.embed_query(http, message, api_key)
                if vec:
                    lru_set(embed_cache, key, vec)
            if vec:
                chunks, sources = kb.search(vec, 4)
        except gemini.GeminiError as e:
            log.warning("Embedding failed (%s): %s", e.status, e.message[:200])
        except httpx.HTTPError as e:
            log.warning("Embedding network error: %s", e)

    if chunks:
        ctx = "\n\nRELEVANT CONTEXT:\n" + "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(chunks))
    else:
        ctx = "\n\n(No matching context was found in the knowledge base for this question.)"
    prompt = f"{SYSTEM_PREAMBLE}{ctx}\n\nUSER QUESTION: {message}"

    # 2) Generate (auto-fallback across GENERATION_MODELS on quota errors)
    try:
        reply = await gemini.generate(http, prompt, GENERATION_MODELS, api_key)
    except gemini.GeminiError as e:
        log.error("Gemini error %s: %s", e.status, e.message[:300])
        if e.status == 429:
            if DEGRADED_FALLBACK and chunks:
                # Quota gone: still be useful by returning the best official excerpt. Not cached.
                excerpt = chunks[0][:900].rsplit(" ", 1)[0]
                return {
                    "reply": "The AI service is temporarily at its usage limit, so here is the most "
                             f"relevant information from official sources:\n\n{excerpt}…",
                    "sources": sources,
                    "degraded": True,
                }
            headers = {"Retry-After": str(int(e.retry_after))} if e.retry_after else None
            raise HTTPException(429, "Gemini quota exceeded. Please try again later.", headers=headers)
        raise HTTPException(500, "Server error")
    except httpx.HTTPError as e:
        log.error("Gemini network error: %s", e)
        raise HTTPException(500, "Server error")

    payload = {"reply": reply, "sources": sources}
    lru_set(reply_cache, key, payload)
    return payload


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
