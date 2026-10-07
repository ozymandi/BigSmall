from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

THINK_RE = re.compile(r"<think>.*?</think>\s*", re.S)
LLM_TYPES = ("llm", "vlm")
RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
NOT_LOADED_MARKERS = ("not loaded", "no models loaded", "model_not_found", "model not found", "failed to load")


class LMStudioError(RuntimeError):
    pass


@dataclass
class ChatResult:
    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    elapsed: float = 0.0
    raw: dict = field(default_factory=dict)


def _lms_binary() -> str:
    found = shutil.which("lms")
    if found:
        return found
    candidate = Path.home() / ".lmstudio" / "bin" / "lms.exe"
    if candidate.is_file():
        return str(candidate)
    raise LMStudioError("lms CLI not found. Install LM Studio and run: lms bootstrap")


class LMStudioClient:
    """Thin wrapper over the LM Studio REST API plus the lms CLI for load/unload."""

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

    # --- discovery -------------------------------------------------------
    def is_up(self) -> bool:
        try:
            self.http.get("/v1/models")
            return True
        except httpx.HTTPError:
            return False

    def models(self) -> list[dict]:
        try:
            r = self.http.get("/api/v0/models")
        except httpx.HTTPError as e:
            raise LMStudioError(f"LM Studio server not reachable at {self.base_url}: {e}") from e
        r.raise_for_status()
        return r.json().get("data", [])

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
        """Number of parallel prediction slots the model was loaded with (from lms ps)."""
        try:
            proc = subprocess.run([_lms_binary(), "ps", "--json"], capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", timeout=30)
            for m in json.loads(proc.stdout or "[]"):
                if m.get("identifier") == model or m.get("modelKey") == model:
                    return int(m.get("parallel") or default)
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
        return default

    # --- load / unload ---------------------------------------------------
    def load(self, model: str, context_length: int, ttl: int, parallel: int,
             unload_others: bool = True, wait: float = 180) -> bool:
        """Load a model via lms load. Returns True if a load happened, False if already loaded."""
        if self.is_loaded(model):
            return False
        if self.model_info(model) is None:
            raise LMStudioError(f"Model '{model}' is not downloaded in LM Studio.")
        if unload_others:
            for m in self.loaded_llms():
                self.unload(m["id"])
        cmd = [_lms_binary(), "load", model, "-y",
               "--context-length", str(context_length),
               "--ttl", str(ttl),
               "--parallel", str(parallel)]
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=wait)
        if proc.returncode != 0:
            raise LMStudioError(f"lms load failed: {proc.stderr.strip() or proc.stdout.strip()}")
        deadline = time.time() + wait
        while time.time() < deadline:
            if self.is_loaded(model):
                self.last_load_seconds = time.time() - t0
                return True
            time.sleep(1)
        raise LMStudioError(f"Model '{model}' did not report loaded state within {wait}s.")

    def unload(self, model: str) -> None:
        subprocess.run([_lms_binary(), "unload", model], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=60)

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
        r = self.http.post("/v1/embeddings", json={"model": model, "input": texts})
        if r.status_code >= 400:
            raise LMStudioError(f"HTTP {r.status_code}: {r.text[:800]}")
        return [d["embedding"] for d in r.json()["data"]]
