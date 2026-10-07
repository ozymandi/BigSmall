from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml

DEFAULT_PATH = Path(__file__).with_name("default_config.yaml")
USER_PATH = Path.home() / ".lmagent" / "config.yaml"


def _deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(explicit: str | None = None, cwd: str | Path | None = None) -> dict:
    """Layered config: package default < ~/.lmagent/config.yaml < ./lmagent.yaml < explicit/env path."""
    cfg = yaml.safe_load(DEFAULT_PATH.read_text(encoding="utf-8")) or {}
    cwd = Path(cwd or os.getcwd())
    candidates = [USER_PATH, cwd / "lmagent.yaml"]
    env_path = explicit or os.environ.get("LMAGENT_CONFIG")
    if env_path:
        candidates.append(Path(env_path))
    for p in candidates:
        if p.is_file():
            cfg = _deep_merge(cfg, yaml.safe_load(p.read_text(encoding="utf-8")) or {})
    env_key = os.environ.get("LMSTUDIO_API_KEY")
    if env_key:
        cfg.setdefault("server", {})["api_key"] = env_key
    cfg["_cwd"] = str(cwd)
    return cfg
