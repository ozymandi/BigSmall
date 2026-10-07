from __future__ import annotations

import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

THINK_RE = re.compile(r"<think>.*?</think>\s*", re.S)
LLM_TYPES = ("llm", "vlm")


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

    def __init__(self, base_url: str = "http://localhost:1234", timeout: float = 900):
        self.base_url = base_url.rstrip("/")
        self.http = httpx.Client(base_url=self.base_url, timeout=httpx.Timeout(timeout, connect=5))
        self._supports_template_kwargs = True

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
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=wait)
        if proc.returncode != 0:
            raise LMStudioError(f"lms load failed: {proc.stderr.strip() or proc.stdout.strip()}")
        deadline = time.time() + wait
        while time.time() < deadline:
            if self.is_loaded(model):
                return True
            time.sleep(1)
        raise LMStudioError(f"Model '{model}' did not report loaded state within {wait}s.")

    def unload(self, model: str) -> None:
        subprocess.run([_lms_binary(), "unload", model], capture_output=True, text=True, timeout=60)

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
            body["chat_template_kwargs"] = {"enable_thinking": False}

        t0 = time.time()
        try:
            r = self.http.post("/v1/chat/completions", json=body)
        except httpx.HTTPError as e:
            raise LMStudioError(f"request failed: {e}") from e
        if r.status_code == 400 and "chat_template_kwargs" in body and "chat_template_kwargs" in r.text:
            self._supports_template_kwargs = False
            body.pop("chat_template_kwargs")
            r = self.http.post("/v1/chat/completions", json=body)
        if r.status_code >= 400:
            raise LMStudioError(f"HTTP {r.status_code}: {r.text[:800]}")
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

    def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        r = self.http.post("/v1/embeddings", json={"model": model, "input": texts})
        if r.status_code >= 400:
            raise LMStudioError(f"HTTP {r.status_code}: {r.text[:800]}")
        return [d["embedding"] for d in r.json()["data"]]
