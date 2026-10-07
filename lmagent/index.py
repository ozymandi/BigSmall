from __future__ import annotations

import fnmatch
import json
from pathlib import Path

import numpy as np

from .chunker import SKIP_DIRS, count_tokens, expand_paths
from .client import LMStudioClient, LMStudioError

DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


def split_lines(text: str, chunk_tokens: int, overlap_tokens: int) -> list[tuple[int, int, str]]:
    """Split text on line boundaries; returns (start_line, end_line, chunk_text) with 1-based lines."""
    lines = text.splitlines(keepends=True)
    out: list[tuple[int, int, str]] = []
    cur: list[tuple[int, str, int]] = []  # (line_no, text, tokens)
    cur_tok = 0
    for i, line in enumerate(lines, 1):
        t = count_tokens(line)
        if cur and cur_tok + t > chunk_tokens:
            out.append((cur[0][0], cur[-1][0], "".join(x[1] for x in cur)))
            tail: list[tuple[int, str, int]] = []
            tt = 0
            for item in reversed(cur):
                if tt + item[2] > overlap_tokens:
                    break
                tail.insert(0, item)
                tt += item[2]
            cur, cur_tok = tail, tt
        cur.append((i, line, t))
        cur_tok += t
    if cur:
        out.append((cur[0][0], cur[-1][0], "".join(x[1] for x in cur)))
    return out


class Index:
    """Incremental embedding index of a directory tree, stored in <root>/<index.dir>/."""

    def __init__(self, cfg: dict, client: LMStudioClient, root: str | Path):
        self.cfg = cfg
        self.client = client
        self.root = Path(root)
        icfg = cfg.get("index", {})
        self.dir = self.root / icfg.get("dir", ".lmagent/index")
        self.model = cfg["models"]["embed"]
        self.chunk_tokens = int(icfg.get("chunk_tokens", 400))
        self.overlap = int(icfg.get("overlap_tokens", 40))
        self.batch = int(icfg.get("batch", 32))
        self.exclude = list(icfg.get("exclude", []))
        self.max_bytes = int(cfg["chunking"]["max_file_bytes"])
        self.files: dict[str, dict] = {}   # rel path -> {"mtime", "size", "chunks": [{"start","end","text"}], "rows": [a, b)}
        self.vectors = np.zeros((0, 0), dtype=np.float32)

    # --- persistence ---------------------------------------------------
    def load(self) -> bool:
        meta = self.dir / "meta.json"
        vec = self.dir / "vectors.npy"
        if not (meta.is_file() and vec.is_file()):
            return False
        data = json.loads(meta.read_text(encoding="utf-8"))
        if data.get("model") != self.model or data.get("chunk_tokens") != self.chunk_tokens:
            return False
        self.files = data["files"]
        self.vectors = np.load(vec)
        return True

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "meta.json").write_text(
            json.dumps({"model": self.model, "chunk_tokens": self.chunk_tokens, "files": self.files},
                       ensure_ascii=False), encoding="utf-8")
        np.save(self.dir / "vectors.npy", self.vectors)

    # --- embedding -----------------------------------------------------
    def _ensure_embed_model(self) -> None:
        # Embedding models are small and coexist with the LLM: load without unloading anything.
        self.client.load_embedding(self.model)

    def _embed(self, texts: list[str], prefix: str) -> np.ndarray:
        self._ensure_embed_model()
        rows: list[list[float]] = []
        for i in range(0, len(texts), self.batch):
            rows.extend(self.client.embed(self.model, [prefix + t for t in texts[i:i + self.batch]]))
        arr = np.asarray(rows, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms

    # --- update --------------------------------------------------------
    def _candidate_files(self, paths: list[str] | None) -> dict[str, Path]:
        out: dict[str, Path] = {}
        for f in expand_paths(paths or ["."], self.root):
            try:
                rel = f.relative_to(self.root).as_posix()
            except ValueError:
                rel = f.as_posix()
            parts = Path(rel).parts
            if set(parts) & SKIP_DIRS or any(p.endswith(".egg-info") for p in parts):
                continue
            if any(fnmatch.fnmatch(f.name, pat) for pat in self.exclude):
                continue
            out[rel] = f
        return out

    def update(self, paths: list[str] | None = None) -> dict:
        """Index new/changed files, drop deleted ones. Returns counts."""
        self.load()
        candidates = self._candidate_files(paths)
        stats = {"indexed": 0, "unchanged": 0, "removed": 0, "skipped": 0, "chunks": 0}

        keep: dict[str, dict] = {}
        keep_rows: list[np.ndarray] = []
        to_embed: list[tuple[str, dict, list[tuple[int, int, str]]]] = []

        # Files already indexed but outside this update's candidates: with a full update they are gone
        # (deleted or now excluded); with a partial update they are kept if they still exist.
        for rel, info in self.files.items():
            if rel in candidates:
                continue
            if paths is not None and (self.root / rel).is_file():
                keep[rel] = info
                keep_rows.append(self.vectors[info["rows"][0]:info["rows"][1]])

        for rel, f in candidates.items():
            try:
                st = f.stat()
            except OSError:
                stats["skipped"] += 1
                continue
            old = self.files.get(rel)
            if old and old["mtime"] == st.st_mtime and old["size"] == st.st_size and self.vectors.size:
                keep[rel] = old
                keep_rows.append(self.vectors[old["rows"][0]:old["rows"][1]])
                stats["unchanged"] += 1
                continue
            if st.st_size > self.max_bytes:
                stats["skipped"] += 1
                continue
            try:
                raw = f.read_bytes()
                if b"\x00" in raw[:8192]:
                    stats["skipped"] += 1
                    continue
                text = raw.decode("utf-8", errors="replace")
            except OSError:
                stats["skipped"] += 1
                continue
            chunks = split_lines(text, self.chunk_tokens, self.overlap)
            if not chunks:
                stats["skipped"] += 1
                continue
            to_embed.append((rel, {"mtime": st.st_mtime, "size": st.st_size}, chunks))

        stats["removed"] = len([p for p in self.files if p not in keep and p not in {t[0] for t in to_embed}])

        if to_embed:
            texts = [c[2] for _, _, chunks in to_embed for c in chunks]
            vecs = self._embed(texts, DOC_PREFIX)
            pos = 0
            for rel, info, chunks in to_embed:
                n = len(chunks)
                keep_rows.append(vecs[pos:pos + n])
                info["chunks"] = [{"start": s, "end": e, "text": t} for s, e, t in chunks]
                keep[rel] = info
                pos += n
                stats["indexed"] += 1
                stats["chunks"] += n

        # rebuild row ranges in the order of `keep`
        rows_in_order: list[np.ndarray] = []
        offset = 0
        rebuilt: dict[str, dict] = {}
        # keep_rows was appended in the same order as keep insertions
        for (rel, info), block in zip(keep.items(), keep_rows, strict=True):
            n = block.shape[0]
            info["rows"] = [offset, offset + n]
            rebuilt[rel] = info
            rows_in_order.append(block)
            offset += n
        self.files = rebuilt
        self.vectors = np.concatenate(rows_in_order) if rows_in_order else np.zeros((0, 0), dtype=np.float32)
        self.save()
        stats["total_files"] = len(self.files)
        stats["total_chunks"] = int(self.vectors.shape[0]) if self.vectors.size else 0
        return stats

    # --- search --------------------------------------------------------
    def search(self, query: str, k: int = 8, files_only: bool = False) -> list[dict]:
        if not self.files and not self.load():
            raise LMStudioError(f"No index at {self.dir}. Run `lmagent index` first.")
        if not self.vectors.size:
            return []
        q = self._embed([query], QUERY_PREFIX)[0]
        scores = self.vectors @ q
        order = np.argsort(-scores)
        hits: list[dict] = []
        lookup: list[tuple[str, int]] = []
        for rel, info in self.files.items():
            for ci in range(len(info["chunks"])):
                lookup.append((rel, ci))
        if files_only:
            best: dict[str, dict] = {}
            for idx in order:
                rel, ci = lookup[int(idx)]
                if rel not in best:
                    c = self.files[rel]["chunks"][ci]
                    best[rel] = {"file": rel, "score": round(float(scores[idx]), 4),
                                 "start_line": c["start"], "end_line": c["end"]}
                if len(best) >= k:
                    break
            return list(best.values())
        for idx in order[:k]:
            rel, ci = lookup[int(idx)]
            c = self.files[rel]["chunks"][ci]
            hits.append({"file": rel, "score": round(float(scores[idx]), 4),
                         "start_line": c["start"], "end_line": c["end"], "text": c["text"]})
        return hits
