"""Attachments from Worksection into the project folder, without the model: lmagent reads the
`worksection://file/<id>` MCP resource straight from the Worksection MCP server and writes the bytes.
Archives are saved as they are, never unpacked."""
from __future__ import annotations

import asyncio
import base64
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

DOC_EXTS = {".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".txt", ".md", ".rtf", ".csv",
            ".html", ".htm", ".json", ".zip", ".rar", ".7z"}

ResourceReader = Callable[[str], bytes]  # uri -> bytes


@dataclass
class Attachment:
    id: str
    name: str
    size: int
    origin: str = ""   # "task" | "comment" | "project"


@dataclass
class DownloadReport:
    saved: list[dict] = field(default_factory=list)     # {id, name, size, path}
    skipped: list[dict] = field(default_factory=list)   # {id, name, size, reason}
    errors: list[dict] = field(default_factory=list)    # {id, name, error}

    def section(self, files_dir: str) -> str:
        lines = ["## Файли на диску", f"Папка: `{files_dir}`", ""]
        for s in self.saved:
            lines.append(f"- {s['name']} ({_mb(s['size'])})")
        for s in self.skipped:
            lines.append(f"- пропущено: {s['name']} ({_mb(s['size'])}, {s['reason']})")
        for e in self.errors:
            lines.append(f"- помилка: {e['name']}: {e['error']}")
        if not (self.saved or self.skipped or self.errors):
            lines.append("- вкладень немає")
        return "\n".join(lines)


def _mb(n: int) -> str:
    return f"{n / 1_000_000:.1f} MB" if n >= 100_000 else f"{n / 1000:.0f} KB"


def attachments_from_raw(raw_dir: Path) -> list[Attachment]:
    """Attachments listed in the saved tool outputs (get_all_task_attachments, get_task_files,
    get_project_files), deduplicated by id."""
    found: dict[str, Attachment] = {}
    for p in sorted(raw_dir.glob("*.json")):
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if rec.get("tool") not in ("get_all_task_attachments", "get_task_files", "get_project_files"):
            continue
        for origin, item in _file_records(rec.get("output")):
            fid = str(item.get("id") or "")
            name = str(item.get("name") or "").strip()
            if not fid or not name or fid in found:
                continue
            try:
                size = int(str(item.get("size") or 0))
            except ValueError:
                size = 0
            found[fid] = Attachment(id=fid, name=name, size=size, origin=origin)
    return list(found.values())


def _file_records(output) -> Iterable[tuple[str, dict]]:
    """Walk the tool output (MCP content blocks wrapping the server's JSON) and yield file dicts."""
    data = _unwrap(output)
    if isinstance(data, dict):
        for key, origin in (("task_files", "task"), ("comment_files", "comment"), ("files", "project"),
                            ("data", "project")):
            items = data.get(key)
            if isinstance(items, list):
                for it in items:
                    if isinstance(it, dict) and "id" in it and "name" in it:
                        yield origin, it
    elif isinstance(data, list):
        for it in data:
            if isinstance(it, dict) and "id" in it and "name" in it:
                yield "project", it


def _unwrap(output):
    """Tool outputs are stored as the server returned them: either a JSON string of MCP content blocks
    ([{"type": "text", "text": "<json>"}]) or the JSON itself."""
    if isinstance(output, str):
        try:
            output = json.loads(output)
        except ValueError:
            return None
    if isinstance(output, list) and output and isinstance(output[0], dict) and output[0].get("type") == "text":
        try:
            return json.loads(output[0].get("text") or "")
        except ValueError:
            return None
    return output


def select(attachments: list[Attachment], mode: str, max_mb: float) -> tuple[list[Attachment], list[dict]]:
    """Which attachments to fetch: mode docs (documents and archives) | all | none; size cap in MB."""
    keep, skipped = [], []
    if mode == "none":
        return keep, [{"id": a.id, "name": a.name, "size": a.size, "reason": "download=none"} for a in attachments]
    limit = int(max_mb * 1_000_000)
    for a in attachments:
        ext = Path(a.name).suffix.lower()
        if mode == "docs" and ext not in DOC_EXTS:
            skipped.append({"id": a.id, "name": a.name, "size": a.size, "reason": "not a document"})
        elif a.size > limit:
            skipped.append({"id": a.id, "name": a.name, "size": a.size, "reason": f"over {max_mb:g} MB"})
        else:
            keep.append(a)
    return keep, skipped


def safe_name(name: str) -> str:
    name = re.sub(r'[\\/:*?"<>|\x00-\x1f]+', "_", name).strip(" .")
    return name or "file"


def download(attachments: list[Attachment], dest: Path, mode: str, max_mb: float,
             reader: ResourceReader, on_progress: Callable[[int, int, str], None] | None = None) -> DownloadReport:
    rep = DownloadReport()
    keep, rep.skipped = select(attachments, mode, max_mb)
    if not keep:
        return rep
    dest.mkdir(parents=True, exist_ok=True)
    taken: set[str] = set()
    for i, a in enumerate(keep, 1):
        if on_progress:
            on_progress(i, len(keep), f"file {a.name}")
        name = safe_name(a.name)
        if name.lower() in taken:
            p = Path(name)
            name = f"{p.stem}_{a.id}{p.suffix}"
        taken.add(name.lower())
        try:
            data = reader(f"worksection://file/{a.id}")
            (dest / name).write_bytes(data)
            rep.saved.append({"id": a.id, "name": name, "size": len(data), "path": str(dest / name)})
        except Exception as e:  # one bad file must not stop the rest
            rep.errors.append({"id": a.id, "name": a.name, "error": f"{type(e).__name__}: {str(e)[:160]}"})
    return rep


def mcp_resource_reader(server_url: str) -> ResourceReader:
    """Reader that opens a short MCP session over streamable HTTP per file and reads one resource."""
    def read(uri: str) -> bytes:
        return asyncio.run(_read_resource(server_url, uri))
    return read


async def _read_resource(server_url: str, uri: str) -> bytes:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client
    from pydantic import AnyUrl

    async with streamablehttp_client(server_url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            res = await session.read_resource(AnyUrl(uri))
    return bytes_from_contents([(getattr(c, "blob", None), getattr(c, "text", None)) for c in res.contents])


def bytes_from_contents(contents: list[tuple[str | None, str | None]]) -> bytes:
    """File bytes from MCP resource contents given as (blob, text) pairs. A native blob is base64.
    FastMCP serializes a resource function that returns a dict as JSON *text*, so the Worksection server's
    {"uri", "mimeType", "blob"|"text"} arrives inside the text and has to be unwrapped."""
    for blob, text in contents:
        if blob:
            return base64.b64decode(blob)
        if text is None:
            continue
        try:
            inner = json.loads(text)
        except ValueError:
            return text.encode("utf-8")
        if isinstance(inner, dict) and ("blob" in inner or "text" in inner):
            if inner.get("blob"):
                return base64.b64decode(inner["blob"])
            return str(inner.get("text") or "").encode("utf-8")
        return text.encode("utf-8")
    raise RuntimeError("empty resource")
