from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import httpx

THINK_RE = re.compile(r"<think>.*?</think>\s*", re.S)
LLM_TYPES = ("llm", "vlm")
RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
NOT_LOADED_MARKERS = ("not loaded", "no models loaded", "model_not_found", "model not found", "failed to load")


class LMStudioError(RuntimeError):
    pass


class LMStudioDown(LMStudioError):
    """The server does not answer at all (not started, wrong port). Never retried."""


CONNECT_ERRORS = (httpx.ConnectError, httpx.ConnectTimeout)
START_HINT = ("LM Studio server is not reachable at {url}. Start it: open LM Studio -> Developer -> "
              "Start Server, or run `lms server start` (lms lives in ~/.lmstudio/bin). "
              "Check `server.base_url` in the config if it runs on another port.")


@dataclass
class ChatResult:
    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    elapsed: float = 0.0
    raw: dict = field(default_factory=dict)


class LMStudioClient:
    """LM Studio over its REST API only (no lms CLI: the CLI blocks when run inside an MCP server).

    Uses /api/v1/models for discovery, /api/v1/models/load|unload for lifecycle,
    /v1/chat/completions and /v1/embeddings for inference.
    """

    def __init__(self, base_url: str = "http://localhost:1234", timeout: float = 900,
                 retries: int = 2, retry_delay: float = 2.0):
        self.base_url = base_url.rstrip("/")
        self.http = httpx.Client(base_url=self.base_url, timeout=httpx.Timeout(timeout, connect=5))
        self.retries = retries
        self.retry_delay = retry_delay
        self._supports_template_kwargs = True
        # Called with the model id when a request fails because the model is no longer loaded (e.g. TTL).
        self.on_not_loaded = None
        self.last_load_seconds: float | None = None

    def _down(self, e: Exception) -> LMStudioDown:
        return LMStudioDown(START_HINT.format(url=self.base_url) + f" [{type(e).__name__}]")

    # --- discovery -------------------------------------------------------
    def is_up(self) -> bool:
        try:
            self.http.get("/v1/models")
            return True
        except httpx.HTTPError:
            return False

    def models(self) -> list[dict]:
        """Normalized model list: id, type (llm|embeddings), state, loaded_context_length, parallel, instance_id."""
        try:
            r = self.http.get("/api/v1/models")
        except CONNECT_ERRORS as e:
            raise self._down(e) from e
        except httpx.HTTPError as e:
            raise LMStudioError(f"GET /api/v1/models failed: {e}") from e
        if r.status_code >= 400:
            raise LMStudioError(f"GET /api/v1/models -> HTTP {r.status_code}: {r.text[:300]}")
        out = []
        for m in r.json().get("models", []):
            inst = m.get("loaded_instances") or []
            cfg = (inst[0].get("config") or {}) if inst else {}
            mtype = m.get("type")
            out.append({
                "id": m.get("key"),
                "type": "embeddings" if mtype == "embedding" else mtype,
                "state": "loaded" if inst else "not-loaded",
                "loaded_context_length": cfg.get("context_length"),
                "parallel": cfg.get("parallel"),
                "max_context_length": m.get("max_context_length"),
                "instance_id": inst[0].get("id") if inst else None,
                "reasoning": (m.get("capabilities") or {}).get("reasoning"),
            })
        return out

    def model_info(self, model: str) -> dict | None:
        for m in self.models():
            if m.get("id") == model:
                return m
        return None

    def loaded_llms(self) -> list[dict]:
        return [m for m in self.models() if m.get("state") == "loaded" and m.get("type") in LLM_TYPES]

    def is_loaded(self, model: str) -> bool:
        info = self.model_info(model)
        return bool(info and info.get("state") == "loaded")

    def loaded_context(self, model: str, default: int) -> int:
        info = self.model_info(model) or {}
        return int(info.get("loaded_context_length") or default)

    def loaded_parallel(self, model: str, default: int) -> int:
        info = self.model_info(model) or {}
        return int(info.get("parallel") or default)

    # --- load / unload ---------------------------------------------------
    def _load_request(self, body: dict, wait: float) -> dict:
        try:
            r = self.http.post("/api/v1/models/load", json=body, timeout=httpx.Timeout(wait, connect=5))
        except CONNECT_ERRORS as e:
            raise self._down(e) from e
        except httpx.HTTPError as e:
            raise LMStudioError(f"load request failed: {e}") from e
        if r.status_code >= 400:
            raise LMStudioError(f"load failed: HTTP {r.status_code}: {r.text[:400]}")
        data = r.json()
        self.last_load_seconds = float(data.get("load_time_seconds") or 0.0)
        return data

    def load(self, model: str, context_length: int, ttl: int, parallel: int,
             unload_others: bool = True, wait: float = 300) -> bool:
        """Load an LLM. Returns True if a load happened, False if already loaded."""
        info = self.model_info(model)
        if info is None:
            raise LMStudioError(f"Model '{model}' is not downloaded in LM Studio.")
        if info.get("state") == "loaded":
            return False
        if unload_others:
            for m in self.loaded_llms():
                self.unload(m["id"])
        body = {"model": model, "context_length": int(context_length), "ttl_seconds": int(ttl),
                "parallel": int(parallel)}
        self._load_request(body, wait)
        return True

    def load_embedding(self, model: str, wait: float = 120) -> bool:
        """Load an embedding model next to whatever LLM is loaded (they are small)."""
        info = self.model_info(model)
        if info is None:
            raise LMStudioError(f"Embedding model '{model}' is not downloaded in LM Studio.")
        if info.get("state") == "loaded":
            return False
        self._load_request({"model": model}, wait)
        return True

    def unload(self, model: str) -> None:
        info = self.model_info(model) or {}
        instance = info.get("instance_id") or model
        try:
            self.http.post("/api/v1/models/unload", json={"instance_id": instance},
                           timeout=httpx.Timeout(120, connect=5))
        except httpx.HTTPError as e:
            raise LMStudioError(f"unload request failed: {e}") from e

    # --- inference -------------------------------------------------------
    def chat(self, model: str, messages: list[dict], temperature: float = 0.2,
             max_tokens: int = 4096, json_schema: dict | None = None,
             thinking: bool = False) -> ChatResult:
        body: dict = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if json_schema:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "result", "strict": True, "schema": json_schema},
            }
        if not thinking and self._supports_template_kwargs:
            # Measured on LM Studio: chat_template_kwargs/enable_thinking is ignored, reasoning_effort works.
            body["reasoning_effort"] = "none"

        t0 = time.time()
        r = self._post_with_retry(body, model)
        data = r.json()
        msg = data["choices"][0]["message"]
        content = THINK_RE.sub("", msg.get("content") or "").strip()
        usage = data.get("usage") or {}
        return ChatResult(
            content=content,
            model=data.get("model", model),
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            elapsed=time.time() - t0,
            raw=data,
        )

    def _post_with_retry(self, body: dict, model: str) -> httpx.Response:
        attempt = 0
        delay = self.retry_delay
        while True:
            try:
                r = self.http.post("/v1/chat/completions", json=body)
            except CONNECT_ERRORS as e:
                raise self._down(e) from e
            except httpx.HTTPError as e:
                if attempt >= self.retries:
                    raise LMStudioError(f"request failed after {attempt + 1} attempt(s): {e}") from e
                attempt += 1
                time.sleep(delay)
                delay *= 2
                continue
            if r.status_code < 400:
                return r
            text_l = r.text.lower()
            if r.status_code == 400 and "reasoning_effort" in body and "reasoning_effort" in text_l:
                self._supports_template_kwargs = False
                body.pop("reasoning_effort")
                continue
            if attempt >= self.retries:
                raise LMStudioError(f"HTTP {r.status_code}: {r.text[:800]}")
            if any(m in text_l for m in NOT_LOADED_MARKERS) and self.on_not_loaded is not None:
                self.on_not_loaded(model)
            elif r.status_code not in RETRYABLE_STATUS:
                raise LMStudioError(f"HTTP {r.status_code}: {r.text[:800]}")
            else:
                time.sleep(delay)
                delay *= 2
            attempt += 1

    def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        try:
            r = self.http.post("/v1/embeddings", json={"model": model, "input": texts})
        except CONNECT_ERRORS as e:
            raise self._down(e) from e
        except httpx.HTTPError as e:
            raise LMStudioError(f"embeddings request failed: {e}") from e
        if r.status_code >= 400:
            raise LMStudioError(f"HTTP {r.status_code}: {r.text[:800]}")
        return [d["embedding"] for d in r.json()["data"]]
