"""Project intake: the local model reads a Worksection project or task through the Worksection MCP
server (driven by LM Studio's own agent loop) and writes a digest. The caller only reads the digest.

Link forms: https://<acct>.worksection.com/project/<project_id>/            -> whole project
            https://<acct>.worksection.com/project/<project_id>/<task_id>/   -> one task (+ #com<id> ignored)
            https://<acct>.worksection.com/project/<pid>/<tid>/<subtask_id>/ -> the subtask
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .client import LMStudioClient, AgentResult
from .runner import Runner, RunResult
from .tasks import TASKS

ProgressCb = Callable[[int, int, str], None]

_LINK_RE = re.compile(r"/project/(\d+)(?:/(\d+))?(?:/(\d+))?/?(?:[?#].*)?$")


@dataclass
class Link:
    project_id: str
    task_id: str | None = None
    comment_id: str | None = None


def parse_link(url: str) -> Link:
    """Project id, deepest task id and comment id from a Worksection URL."""
    url = url.strip()
    m = _LINK_RE.search(url)
    if not m:
        raise ValueError(f"not a Worksection project/task link: {url}")
    pid, tid, sub = m.groups()
    cm = re.search(r"#com(\d+)", url)
    return Link(project_id=pid, task_id=sub or tid, comment_id=cm.group(1) if cm else None)


SYSTEM = """You are an intake clerk for a design/development studio. You read a client's project in
Worksection through the tools and write a factual digest that a senior designer will use to estimate
the work. You never invent facts. If something is missing or unclear, say so under "Open questions".
Work step by step, call the tools in the order given, then write the digest. Do not ask the user
anything; there is no user to answer."""

TASK_STEPS = """Project id: {pid}. Task id: {tid}.

Steps (at most {max_calls} tool calls in total):
1. get_task for task {tid}: title, description, status, dates, assignees, priority, tags.
2. get_task_discussion for task {tid}: every comment with author and date, and the files they mention.
3. get_all_task_attachments for task {tid}.
4. For every attachment that is a document (pdf, docx, xlsx, pptx, txt, md): get_file_content with its file id.
   Skip images and archives; just list them.
5. If a tool answer says it was offloaded, call read_offloaded_response_text until you have read it all.
6. get_task_subtasks for task {tid}; if there are subtasks, get_task for each (no deeper).
Then write the digest.
"""

PROJECT_STEPS = """Project id: {pid}.

Steps (at most {max_calls} tool calls in total):
1. get_project for project {pid}: name, description, dates, status, manager, client.
2. get_tasks for project {pid}: list every task with id, title, status, dates.
3. get_project_files for project {pid}; for every document (pdf, docx, xlsx, pptx, txt, md) attached at
   project level: get_file_content. Skip images and archives; just list them.
4. If a tool answer says it was offloaded, call read_offloaded_response_text until you have read it all.
Then write the digest.
"""

DIGEST_FORMAT = """Write the digest in {lang}, Markdown, under 1500 words, exactly these sections:

## Клієнт і контекст
Who the client is, what the project is about, where this task sits in it.
## Суть задачі
What has to be made, in plain words. Quote the client's own key phrases.
## Обсяг і deliverables
A bullet per deliverable: what, how many, format. Counts matter (pages, screens, models, variants, languages).
## Терміни
Dates and deadlines with their source (task field, comment of <author> on <date>).
## Технічні вимоги і обмеження
Tools, platforms, sizes, formats, brand rules, references, anything that constrains the work.
## Вкладення
One bullet per file: name, type, size if known, 2-4 lines on what is inside (for documents you read).
Images and archives: name and type only.
## Хід обговорення
Chronological bullets: date, author, the point they made or the decision taken. Skip chatter.
## Відкриті питання
What is unclear, contradictory or missing for an estimate.
## Факти для оцінки
Short bullets with the numbers and facts an estimator needs, nothing else.

Cite the source of each fact in parentheses: (task), (comment: author, date), (file: name).
No preamble, no closing remarks, start with the first heading."""

MERGE_SYSTEM = "You merge several intake digests of tasks from one project into one project digest."
MERGE_USER = """Project: {name}

Below are digests of {n} tasks of this project, each under a '# Task' heading. Write ONE merged digest in
{lang}, Markdown, under 2500 words, with the same section headings as the task digests (Клієнт і контекст,
Суть задачі, Обсяг і deliverables, Терміни, Технічні вимоги і обмеження, Вкладення, Хід обговорення,
Відкриті питання, Факти для оцінки). Keep every fact once, keep the task titles as sub-bullets where the
tasks differ, keep the source citations. Start with the first heading.

{body}"""


def final_digest(r: AgentResult) -> str:
    """The digest proper: the model narrates between tool calls ("Let me read the discussion..."), so take
    the last message and drop anything before its first Markdown heading."""
    text = r.text or ""
    m = re.search(r"(?m)^#{1,3} ", text)
    return text[m.start():].strip() if m else text.strip()


@dataclass
class IntakeResult:
    digest: str
    digest_path: str
    raw_dir: str
    tool_calls: list[dict] = field(default_factory=list)
    invalid: list[dict] = field(default_factory=list)
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    elapsed: float = 0.0
    calls: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "tool_calls"}
        d["tool_calls"] = [{"tool": c.get("tool"), "arguments": c.get("arguments"),
                            "output_chars": len(c.get("output") or "")} for c in self.tool_calls]
        return d


class Intake:
    def __init__(self, cfg: dict, client: LMStudioClient | None = None):
        self.cfg = cfg
        self.icfg = cfg["intake"]
        self.cwd = Path(cfg["_cwd"])
        self.runner = Runner(cfg, client=client)
        self.client = self.runner.client

    # --- pieces -----------------------------------------------------------
    def _integrations(self) -> list[dict]:
        return [{"type": "plugin", "id": f"mcp/{self.icfg['mcp']}",
                 "allowed_tools": list(self.icfg.get("allowed_tools") or [])}]

    def _agent(self, model: str, user: str) -> AgentResult:
        return self.client.chat_agent(
            model, user, self._integrations(), system_prompt=SYSTEM, reasoning="off",
            max_output_tokens=int(self.icfg.get("max_output_tokens", 6000)),
        )

    def _save_raw(self, raw_dir: Path, calls: list[dict], start: int) -> int:
        raw_dir.mkdir(parents=True, exist_ok=True)
        n = start
        for c in calls:
            n += 1
            name = re.sub(r"[^A-Za-z0-9_]+", "_", str(c.get("tool") or "tool"))
            (raw_dir / f"{n:02d}_{name}.json").write_text(
                json.dumps({"tool": c.get("tool"), "arguments": c.get("arguments"), "output": c.get("output")},
                           ensure_ascii=False, indent=1), encoding="utf-8")
        return n

    def _task_digest(self, model: str, link: Link, lang: str) -> AgentResult:
        user = (TASK_STEPS.format(pid=link.project_id, tid=link.task_id,
                                  max_calls=self.icfg.get("max_tool_calls", 40))
                + "\n" + DIGEST_FORMAT.format(lang=lang))
        return self._agent(model, user)

    def _project_overview(self, model: str, link: Link, lang: str) -> AgentResult:
        user = (PROJECT_STEPS.format(pid=link.project_id, max_calls=self.icfg.get("max_tool_calls", 40))
                + "\n" + DIGEST_FORMAT.format(lang=lang))
        return self._agent(model, user)

    @staticmethod
    def _task_ids_from(calls: list[dict]) -> list[tuple[str, str]]:
        """(id, title) of tasks found in a get_tasks output, best effort over the server's JSON."""
        out: list[tuple[str, str]] = []
        for c in calls:
            if c.get("tool") != "get_tasks":
                continue
            text = c.get("output") or ""
            for m in re.finditer(r'"id"\s*:\s*"?(\d+)"?[^{}]*?"(?:title|name)"\s*:\s*"([^"]*)"', text):
                out.append((m.group(1), m.group(2)))
        seen, uniq = set(), []
        for tid, title in out:
            if tid not in seen:
                seen.add(tid)
                uniq.append((tid, title))
        return uniq

    # --- entry point ------------------------------------------------------
    def run(self, url: str, lang: str | None = None, out: str | None = None,
            on_progress: ProgressCb | None = None) -> IntakeResult:
        link = parse_link(url)
        lang = lang or self.icfg.get("lang", "Ukrainian")
        out_path = self.cwd / (out or self.icfg.get("out", "intake/digest.md"))
        raw_dir = self.cwd / self.icfg.get("raw_dir", "intake/raw")
        notes: list[str] = []
        t0 = time.time()
        role = self.icfg.get("role", "text")
        spec = next((s for s in TASKS.values() if s.role == role), TASKS["summarize"])
        model = self.runner.resolve_model(spec, notes=notes, strict=True)

        def tick(done: int, total: int, msg: str) -> None:
            if on_progress:
                on_progress(done, total, msg)

        results: list[AgentResult] = []
        raw_n = 0
        if link.task_id:
            tick(0, 1, f"task {link.task_id}")
            r = self._task_digest(model, link, lang)
            results.append(r)
            raw_n = self._save_raw(raw_dir, r.tool_calls, raw_n)
            digest = final_digest(r)
            tick(1, 1, "digest")
        else:
            tick(0, 2, f"project {link.project_id}")
            ov = self._project_overview(model, link, lang)
            results.append(ov)
            raw_n = self._save_raw(raw_dir, ov.tool_calls, raw_n)
            tasks = self._task_ids_from(ov.tool_calls)[: int(self.icfg.get("max_tasks", 20))]
            parts = [f"# Project overview\n\n{final_digest(ov)}"]
            total = 2 + len(tasks)
            for i, (tid, title) in enumerate(tasks, 1):
                tick(i, total, f"task {tid}: {title[:40]}")
                r = self._task_digest(model, Link(link.project_id, tid), lang)
                results.append(r)
                raw_n = self._save_raw(raw_dir, r.tool_calls, raw_n)
                parts.append(f"# Task {tid}: {title}\n\n{final_digest(r)}")
            if tasks:
                tick(total - 1, total, "merge")
                merged = self.client.chat(model, [
                    {"role": "system", "content": MERGE_SYSTEM},
                    {"role": "user", "content": MERGE_USER.format(name=link.project_id, n=len(tasks), lang=lang,
                                                                  body="\n\n".join(parts))},
                ], max_tokens=int(self.icfg.get("max_output_tokens", 6000)))
                digest = merged.content
                results_tokens = (merged.prompt_tokens, merged.completion_tokens)
            else:
                digest = final_digest(ov)
                results_tokens = (0, 0)
            tick(total, total, "digest")

        calls = [c for r in results for c in r.tool_calls]
        invalid = [c for r in results for c in r.invalid]
        ptok = sum(r.prompt_tokens for r in results)
        ctok = sum(r.completion_tokens for r in results)
        if not link.task_id:
            ptok += results_tokens[0]
            ctok += results_tokens[1]
        if not digest.strip():
            notes.append("the model returned no digest text; see intake/raw and run.json")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        header = (f"<!-- lmagent intake | {url} | {model} | {len(calls)} tool calls | "
                  f"in {ptok} / out {ctok} tok | {time.time() - t0:.0f}s -->\n\n")
        out_path.write_text(header + digest.rstrip() + "\n", encoding="utf-8", newline="\n")
        res = IntakeResult(digest=digest, digest_path=str(out_path), raw_dir=str(raw_dir),
                           tool_calls=calls, invalid=invalid, model=model, prompt_tokens=ptok,
                           completion_tokens=ctok, elapsed=time.time() - t0, calls=len(results),
                           notes=notes)
        (out_path.parent / "run.json").write_text(json.dumps(res.to_dict(), ensure_ascii=False, indent=1),
                                                  encoding="utf-8", newline="\n")
        self.runner._log(RunResult(task="intake", model=model, text=digest, output_path=str(out_path),
                                   prompt_tokens=ptok, completion_tokens=ctok, elapsed=res.elapsed,
                                   calls=len(results) + len(calls), load_seconds=self.runner._load_seconds),
                         url)
        return res
