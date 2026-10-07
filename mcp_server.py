"""MCP server exposing lmagent tasks to Claude Code (stdio transport).

Register once, globally:
    claude mcp add --scope user lmagent -- python "E:/Projects/lm agent/mcp_server.py"
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from typing import Any

from mcp.server.fastmcp import Context, FastMCP

from lmagent.client import LMStudioClient, LMStudioError
from lmagent.config import load_config
# Import at startup on purpose: importing numpy lazily inside a tool call (event-loop thread) deadlocks
# on Windows while loading its C extension.
from lmagent.index import Index
from lmagent.intake import Intake
from lmagent.answer import answer
from lmagent.runner import Runner
from lmagent.tasks import TASKS

logging.getLogger("httpx").setLevel(logging.WARNING)
mcp = FastMCP("lmagent")


def _runner(cwd: str | None) -> Runner:
    cfg = load_config(cwd=cwd or os.environ.get("LMAGENT_CWD") or os.getcwd())
    return Runner(cfg)


def _compact(r, cfg_limit: int) -> dict[str, Any]:
    """Return only what the caller needs: short text inline, long text as a file path."""
    text = r.text
    truncated = False
    if len(text) > cfg_limit:
        text = text[:cfg_limit] + f"\n...[truncated, full result in {r.output_path}]"
        truncated = True
    return {
        "result": text,
        "truncated": truncated,
        "output_path": r.output_path,
        "files_written": r.files_written,
        "model": r.model,
        "offloaded_tokens": {"in": r.prompt_tokens, "out": r.completion_tokens},
        "calls": r.calls,
        "elapsed_s": round(r.elapsed, 1),
        "notes": r.notes,
    }


def _run(task: str, cwd: str | None, **kw) -> dict[str, Any]:
    try:
        runner = _runner(cwd)
        r = runner.run(task, **kw)
        return _compact(r, runner.cfg["output"]["inline_limit"])
    except (LMStudioError, ValueError, KeyError, OSError) as e:
        return {"error": str(e)}


def _progress_sink(ctx: Context | None, loop: asyncio.AbstractEventLoop):
    """Progress callback usable from any worker thread: schedules ctx.report_progress on the loop.
    FastMCP drops the notification silently when the client sent no progressToken."""
    if ctx is None:
        return None

    def sink(done: int, total: int, message: str) -> None:
        try:
            asyncio.run_coroutine_threadsafe(ctx.report_progress(done, total, message), loop)
        except Exception:
            pass

    return sink


async def _run_async(ctx: Context | None, task: str, cwd: str | None, **kw) -> dict[str, Any]:
    """Run a job in a worker thread so the server keeps answering (and progress gets through)."""
    sink = _progress_sink(ctx, asyncio.get_running_loop())
    return await asyncio.to_thread(_run, task, cwd, on_progress=sink, **kw)


@mcp.tool()
async def lm_delegate(instruction: str, ctx: Context, files: list[str] | None = None, text: str = "",
                      task: str = "ask", model: str = "", params: dict[str, str] | None = None,
                      output_file: str = "", in_place: bool = False, cwd: str = "") -> dict[str, Any]:
    """Delegate a simple but token-heavy job to a local LM Studio model. The model reads the files;
    only the compact result comes back. Use for: summarizing big files/logs, translating, extracting
    structured data, classifying files, bulk mechanical rewrites, explaining diffs, generating boilerplate.

    Args:
        instruction: what to do (any language).
        files: file paths, directories or globs, relative to cwd. The local model reads them, not you.
        text: inline text input instead of (or in addition to) files.
        task: ask | summarize | translate | extract | classify | rewrite | explain_diff | generate.
        model: optional LM Studio model id to force. Default: the model already loaded.
        params: template params, e.g. {"to": "English"} for translate, {"focus": "errors"} for summarize.
        output_file: write the full result to this path (relative to cwd).
        in_place: for task=rewrite, overwrite the source files instead of writing to .lmagent/out/.
        cwd: project directory to resolve relative paths against. Default: server working directory.
    """
    return await _run_async(ctx, task, cwd or None, instruction=instruction, files=files or [], text=text,
                            model=model or None, params=params or {}, output=output_file or None,
                            in_place=in_place)


@mcp.tool()
async def lm_summarize_files(files: list[str], ctx: Context, focus: str = "everything important",
                             instruction: str = "", cwd: str = "") -> dict[str, Any]:
    """Summarize large files, directories or logs with a local model and return only the key facts.
    Cheap way to learn what is in big inputs without reading them yourself."""
    return await _run_async(ctx, "summarize", cwd or None, instruction=instruction, files=files,
                            params={"focus": focus})


@mcp.tool()
async def lm_intake(link: str, ctx: Context, cwd: str = "", lang: str = "",
                    output_file: str = "", files: str = "") -> dict[str, Any]:
    """Project intake on the designer's command ("читай агентом <link>"): the local model reads a
    Worksection project or task itself through the Worksection MCP server (tasks, comments, PDF/docx
    attachments) and writes a digest. Only the digest comes back; raw tool outputs are saved under
    intake/raw/ for later lm_delegate jobs. Do not read the raw files yourself.

    Args:
        link: https://<acct>.worksection.com/project/<pid>/[<task_id>/] (a #com... anchor is fine).
        cwd: the new project folder, absolute. The digest goes to <cwd>/intake/digest.md.
        lang: digest language (default: intake.lang in the config, Ukrainian).
        output_file: digest path relative to cwd (default intake/digest.md).
        files: attachments copied into <cwd>/intake/files after the run: docs (default) | all | none.
            Archives are saved as they are. Bigger than intake.download_max_mb are only listed.
    """
    def job(on_progress):
        cfg = load_config(cwd=cwd or os.environ.get("LMAGENT_CWD") or os.getcwd())
        res = Intake(cfg).run(link, lang=lang or None, out=output_file or None, on_progress=on_progress,
                              download=files or None)
        d = res.to_dict()
        d["digest"] = res.digest
        return d

    sink = _progress_sink(ctx, asyncio.get_running_loop())
    try:
        return await asyncio.to_thread(job, sink)
    except (LMStudioError, ValueError, OSError) as e:
        return {"error": str(e)}


@mcp.tool()
async def lm_batch(items: list[dict[str, Any]], ctx: Context, task: str = "ask", model: str = "",
                   cwd: str = "") -> list[dict[str, Any]]:
    """Run several independent jobs in parallel on the local model. Each item is
    {"instruction": str, "files": [..], "text": str, "params": {..}, "output_file": str}.
    Returns one compact result per item, in order."""
    from concurrent.futures import ThreadPoolExecutor
    import threading

    sink = _progress_sink(ctx, asyncio.get_running_loop())
    done = 0
    lock = threading.Lock()

    def one(item: dict[str, Any]) -> dict[str, Any]:
        nonlocal done
        res = _run(item.get("task", task), cwd or None,
                   instruction=item.get("instruction", ""), files=item.get("files") or [],
                   text=item.get("text", ""), model=model or item.get("model") or None,
                   params=item.get("params") or {}, output=item.get("output_file") or None,
                   in_place=bool(item.get("in_place", False)))
        with lock:
            done += 1
            n = done
        if sink:
            sink(n, len(items), f"batch item {n}/{len(items)}: {item.get('instruction', '')[:50]}")
        return res

    def run_all() -> list[dict[str, Any]]:
        with ThreadPoolExecutor(max_workers=4) as ex:
            return list(ex.map(one, items))

    return await asyncio.to_thread(run_all)


def _index(cwd: str | None) -> Index:
    cfg = load_config(cwd=cwd or os.environ.get("LMAGENT_CWD") or os.getcwd())
    client = LMStudioClient.from_config(cfg)
    return Index(cfg, client, cfg["_cwd"])


@mcp.tool()
async def lm_index(paths: list[str] | None = None, cwd: str = "") -> dict[str, Any]:
    """Build or incrementally update the local embedding index of a project (only changed files are
    re-embedded). Stored in .lmagent/index/. Usually not needed: lm_search refreshes the index itself."""
    try:
        return await asyncio.to_thread(_index(cwd or None).update, paths)
    except (LMStudioError, OSError, ValueError) as e:
        return {"error": str(e)}


@mcp.tool()
async def lm_search(query: str, k: int = 8, files_only: bool = False, kind: str = "all", ask: str = "",
                    cwd: str = "") -> dict[str, Any]:
    """Semantic search over the project's files using a local embedding model. Finds where something is
    handled by meaning, not by exact words. Returns file, line range, score and snippet per hit.
    The index is refreshed incrementally before searching. Use the hits as `files` for lm_delegate.

    Args:
        kind: all | code | docs. Use `code` for "where is X handled" so README/markdown does not outrank code.
        ask: a question. The top hits are sent to the local model and only its answer with `file:start-end`
            citations comes back (no snippets). Cheapest way to learn how something works.
    """
    def work() -> dict[str, Any]:
        idx = _index(cwd or None)
        stats = idx.update(None)
        index_info = {"files": stats["total_files"], "chunks": stats["total_chunks"], "reindexed": stats["indexed"]}
        if ask:
            runner = _runner(cwd or None)
            res = answer(runner, idx, query, ask, k=max(k, int(runner.cfg["index"].get("answer_k", 12))), kind=kind)
            res["index"] = index_info
            return res
        hits = idx.search(query, k=k, files_only=files_only, kind=kind)
        return {"hits": hits, "index": index_info}

    try:
        return await asyncio.to_thread(work)
    except (LMStudioError, OSError, ValueError) as e:
        return {"error": str(e)}


@mcp.tool()
def lm_models(cwd: str = "") -> dict[str, Any]:
    """List LM Studio models, which one is loaded, and the configured role mapping (code/text/bulk)."""
    try:
        cfg = load_config(cwd=cwd or None)
        client = LMStudioClient.from_config(cfg)
        return {
            "roles": cfg["models"],
            "models": [{"id": m["id"], "state": m.get("state"), "type": m.get("type"),
                        "loaded_context": m.get("loaded_context_length")} for m in client.models()],
        }
    except LMStudioError as e:
        return {"error": str(e)}


@mcp.tool()
def lm_tasks() -> list[dict[str, str]]:
    """List available task templates with their default model role and processing mode."""
    return [{"name": s.name, "role": s.role, "mode": s.mode, "description": s.description,
             "params": ", ".join(f"{k}={v}" for k, v in s.defaults.items())} for s in TASKS.values()]


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    mcp.run(transport="stdio")
