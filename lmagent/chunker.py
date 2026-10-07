from __future__ import annotations

import glob
import os
from pathlib import Path

import tiktoken

_enc = tiktoken.get_encoding("cl100k_base")

SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".lmagent", "dist", "build", ".next"}


def count_tokens(text: str) -> int:
    return len(_enc.encode(text, disallowed_special=()))


def _hard_split(text: str, chunk_tokens: int) -> list[str]:
    ids = _enc.encode(text, disallowed_special=())
    return [_enc.decode(ids[i:i + chunk_tokens]) for i in range(0, len(ids), chunk_tokens)]


def split_text(text: str, chunk_tokens: int, overlap_tokens: int = 0) -> list[str]:
    """Split on line boundaries into chunks of at most chunk_tokens, with a small line-based overlap."""
    if count_tokens(text) <= chunk_tokens:
        return [text]
    chunks: list[str] = []
    cur: list[str] = []
    cur_tok = 0
    for line in text.splitlines(keepends=True):
        t = count_tokens(line)
        if t > chunk_tokens:
            if cur:
                chunks.append("".join(cur))
                cur, cur_tok = [], 0
            chunks.extend(_hard_split(line, chunk_tokens))
            continue
        if cur_tok + t > chunk_tokens and cur:
            chunks.append("".join(cur))
            tail: list[str] = []
            tail_tok = 0
            for prev in reversed(cur):
                pt = count_tokens(prev)
                if tail_tok + pt > overlap_tokens:
                    break
                tail.insert(0, prev)
                tail_tok += pt
            cur, cur_tok = tail, tail_tok
        cur.append(line)
        cur_tok += t
    if cur:
        chunks.append("".join(cur))
    return chunks


def _is_binary(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return b"\x00" in f.read(8192)
    except OSError:
        return True


def expand_paths(patterns: list[str], cwd: str | Path) -> list[Path]:
    """Expand files, directories (recursively) and glob patterns into a sorted unique file list."""
    cwd = Path(cwd)
    out: list[Path] = []
    seen: set[Path] = set()
    for pat in patterns:
        p = Path(pat)
        if not p.is_absolute():
            p = cwd / p
        matches: list[Path]
        if p.is_file():
            matches = [p]
        elif p.is_dir():
            matches = [f for f in p.rglob("*")
                       if f.is_file() and not (set(f.relative_to(p).parts) & SKIP_DIRS)]
        else:
            matches = [Path(m) for m in glob.glob(str(p), recursive=True) if Path(m).is_file()]
        for m in sorted(matches):
            rm = m.resolve()
            if rm not in seen:
                seen.add(rm)
                out.append(m)
    return out


def read_inputs(patterns: list[str], cwd: str | Path, max_bytes: int) -> tuple[list[tuple[str, str]], list[str]]:
    """Return [(display_path, text)] and a list of skip notes."""
    items: list[tuple[str, str]] = []
    notes: list[str] = []
    cwd = Path(cwd)
    for f in expand_paths(patterns, cwd):
        try:
            size = f.stat().st_size
        except OSError:
            notes.append(f"skip (unreadable): {f}")
            continue
        if size > max_bytes:
            notes.append(f"skip (too large, {size} bytes): {f}")
            continue
        if _is_binary(f):
            notes.append(f"skip (binary): {f}")
            continue
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            notes.append(f"skip ({e}): {f}")
            continue
        try:
            display = str(f.relative_to(cwd))
        except ValueError:
            display = str(f)
        items.append((display.replace(os.sep, "/"), text))
    return items, notes
