"""Thin async Gemini REST client with quota-aware error handling."""
import os
import re
import time
import asyncio
import logging
import httpx

log = logging.getLogger("gemini")
BASE = "https://generativelanguage.googleapis.com/v1beta/models"

EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")
EMBED_DIMENSIONS = 768  # must match build_index.py
GENERATE_TIMEOUT = float(os.getenv("GENERATE_TIMEOUT", "60"))  # seconds per model attempt


# model -> unix time until which we skip it (set when its quota is exhausted)
_cooldown: dict[str, float] = {}


class GeminiError(Exception):
    def __init__(self, status: int, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.status = status
        self.message = message
        self.retry_after = retry_after


def parse_duration(text: str) -> float | None:
    """'19h15m44.23s' / '34s' / '1m2s' -> seconds."""
    m = re.fullmatch(r"\s*(?:(\d+)h)?(?:(\d+)m)?(?:([\d.]+)s)?\s*", text or "")
    if not m or not any(m.groups()):
        return None
    h, mi, s = m.groups()
    return int(h or 0) * 3600 + int(mi or 0) * 60 + float(s or 0)


def _retry_after_from(resp: httpx.Response) -> float | None:
    try:
        err = resp.json().get("error", {})
    except Exception:
        return None
    for d in err.get("details", []) or []:
        if isinstance(d, dict) and "retryDelay" in d:
            v = parse_duration(str(d["retryDelay"]))
            if v is not None:
                return v
    m = re.search(r"retry in ([0-9hms.]+?)\.?\s*(?:\n|$)", err.get("message", ""))
    return parse_duration(m.group(1)) if m else None


async def _post(client: httpx.AsyncClient, model: str, method: str, body: dict, api_key: str,
                timeout: float | None = None):
    kwargs = {"timeout": timeout} if timeout is not None else {}
    resp = await client.post(
        f"{BASE}/{model}:{method}",
        json=body,
        headers={"x-goog-api-key": api_key},  # header, so the key never lands in URLs/logs
        **kwargs,
    )
    if resp.status_code != 200:
        try:
            msg = resp.json().get("error", {}).get("message", resp.text)
        except Exception:
            msg = resp.text
        raise GeminiError(resp.status_code, msg, _retry_after_from(resp))
    return resp.json()


async def embed_query(client: httpx.AsyncClient, text: str, api_key: str) -> list[float] | None:
    data = await _post(
        client,
        EMBED_MODEL,
        "embedContent",
        {
            "content": {"parts": [{"text": text}]},
            "taskType": "RETRIEVAL_QUERY",
            "outputDimensionality": EMBED_DIMENSIONS,
        },
        api_key,
    )
    return (data.get("embedding") or {}).get("values")


async def generate(client: httpx.AsyncClient, prompt: str, models: list[str], api_key: str) -> str:
    """Try each model in order; on quota/unknown-model errors move to the next one."""
    last_quota: GeminiError | None = None
    last_other: GeminiError | None = None
    last_net: httpx.HTTPError | None = None

    now = time.time()
    available = [m for m in models if _cooldown.get(m, 0) <= now]
    # If every model is cooling down, still try them (quota may have been reset early).
    for model in available or models:
        for attempt in (1, 2):
            try:
                data = await _post(
                    client, model, "generateContent",
                    {"contents": [{"parts": [{"text": prompt}]}]}, api_key,
                    timeout=GENERATE_TIMEOUT,
                )
                parts = (((data.get("candidates") or [{}])[0]).get("content") or {}).get("parts") or [{}]
                return parts[0].get("text") or "No response received."
            except (httpx.TimeoutException, httpx.TransportError) as e:
                # Slow/unreachable model -> try the next one instead of failing the request
                log.warning("Model %s network error (%s) -> trying next model", model, type(e).__name__)
                last_net = e
                break
            except GeminiError as e:
                if e.status == 429:
                    # Short per-minute limit: wait once and retry the same model.
                    if attempt == 1 and e.retry_after is not None and e.retry_after <= 5:
                        await asyncio.sleep(e.retry_after + 0.5)
                        continue
                    log.warning("Quota exhausted on %s (retry in %s s)", model, e.retry_after)
                    _cooldown[model] = time.time() + min(e.retry_after or 60, 3600)
                    last_quota = e
                    break
                if e.status in (400, 404):  # bad / unavailable model name -> try next
                    log.warning("Model %s rejected: %s", model, e.message[:200])
                    last_other = e
                    break
                raise

    if last_quota is not None:
        raise last_quota
    if last_net is not None and last_other is None:
        raise last_net
    raise last_other or GeminiError(500, "No Gemini model available")
