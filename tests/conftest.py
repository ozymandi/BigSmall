from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pytest

from lmagent.client import ChatResult
from lmagent.config import load_config

QWEN = "qwen/qwen3.8-27b"
EMBED = "text-embedding-nomic-embed-text-v1.5"
DIM = 16


def toy_vector(text: str) -> list[float]:
    """Deterministic bag-of-characters embedding: enough for ranking tests, no server needed."""
    v = np.zeros(DIM, dtype=np.float32)
    for ch in text.lower():
        if ch.isalnum():
            v[ord(ch) % DIM] += 1.0
    return v.tolist()


class FakeClient:
    """Stands in for LMStudioClient: scripted model list, no loading, scripted chat replies."""

    def __init__(self, loaded: str = QWEN, context: int = 89344, parallel: int = 4,
                 reply: Callable[[list[dict]], str] | None = None):
        self.loaded_model = loaded
        self.context = context
        self.parallel = parallel
        self.reply = reply or (lambda messages: "ok")
        self.calls: list[list[dict]] = []
        self.embedded: list[str] = []
        self.on_not_loaded = None
        self.last_load_seconds = None

    # discovery
    def models(self) -> list[dict]:
        out = [{"id": QWEN, "type": "llm", "state": "not-loaded"},
               {"id": "google/gemma-4-31b", "type": "llm", "state": "not-loaded"},
               {"id": EMBED, "type": "embeddings", "state": "loaded", "loaded_context_length": 2048}]
        for m in out:
            if m["id"] == self.loaded_model:
                m.update(state="loaded", loaded_context_length=self.context, parallel=self.parallel)
        return out

    def model_info(self, model: str):
        return next((m for m in self.models() if m["id"] == model), None)

    def loaded_llms(self) -> list[dict]:
        return [m for m in self.models() if m["state"] == "loaded" and m["type"] == "llm"]

    def loaded_context(self, model: str, default: int) -> int:
        info = self.model_info(model) or {}
        return int(info.get("loaded_context_length") or default)

    def loaded_parallel(self, model: str, default: int) -> int:
        info = self.model_info(model) or {}
        return int(info.get("parallel") or default)

    # lifecycle: never loads anything
    def load(self, model, context_length, ttl, parallel, unload_others=True, wait=300) -> bool:
        return False

    def load_embedding(self, model: str, wait: float = 120) -> bool:
        return False

    # inference
    def chat(self, model, messages, temperature=0.2, max_tokens=4096, json_schema=None, thinking=False):
        self.calls.append(messages)
        content = self.reply(messages)
        return ChatResult(content=content, model=model, prompt_tokens=len(messages[-1]["content"]) // 4,
                          completion_tokens=len(content) // 4)

    def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        self.embedded.extend(texts)
        return [toy_vector(t) for t in texts]


@pytest.fixture
def project(tmp_path: Path, monkeypatch) -> Path:
    """An empty project directory with the default config and a log file inside tmp."""
    monkeypatch.delenv("LMAGENT_CONFIG", raising=False)
    return tmp_path


@pytest.fixture
def cfg(project: Path, monkeypatch):
    from lmagent import config as config_mod
    monkeypatch.setattr(config_mod, "USER_PATH", project / "no-user-config.yaml")
    c = load_config(cwd=project)
    c["output"]["log"] = str(project / "log.jsonl")
    return c
