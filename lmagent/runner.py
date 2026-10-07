from __future__ import annotations

import datetime as _dt
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .chunker import count_tokens, read_inputs, split_text
from .client import ChatResult, LMStudioClient
from .tasks import MERGE_SYSTEM, TASKS, TaskSpec, get_task

PROMPT_OVERHEAD = 600  # tokens reserved for system prompt + template text
FENCE_RE = re.compile(r"^```[\w+.-]*\n(.*?)\n?```\s*$", re.S)


class _SafeDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


@dataclass
class RunResult:
    task: str
    model: str
    text: str
    output_path: str | None = None
    files_written: list[str] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    elapsed: float = 0.0
    calls: int = 0
    notes: list[str] = field(default_factory=list)

    def stats_line(self) -> str:
        return (f"[{self.task} | {self.model} | {self.calls} call(s) | "
                f"in {self.prompt_tokens} / out {self.completion_tokens} tok | {self.elapsed:.1f}s]")

    def to_dict(self) -> dict:
        return asdict(self)


def strip_fences(text: str) -> str:
    m = FENCE_RE.match(text.strip())
    return m.group(1) if m else text


def _parse_json(text: str):
    text = strip_fences(text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


class Runner:
    def __init__(self, cfg: dict, client: LMStudioClient | None = None):
        self.cfg = cfg
        self.cwd = Path(cfg["_cwd"])
        self.client = client or LMStudioClient(cfg["server"]["base_url"], cfg["server"]["timeout"])
        self._lock = threading.Lock()

    # --- model handling --------------------------------------------------
    def resolve_model(self, spec: TaskSpec, override: str | None = None,
                      strict: bool = False, notes: list[str] | None = None) -> str:
        role_model = self.cfg["models"][spec.role]
        if override:
            model = override
        elif not strict and self.cfg["load"]["policy"] == "prefer_loaded":
            loaded = self.client.loaded_llms()
            if any(m["id"] == role_model for m in loaded):
                model = role_model
            elif loaded:
                model = loaded[0]["id"]
                if notes is not None:
                    notes.append(f"using already loaded {model} instead of role model {role_model}")
            else:
                model = role_model
        else:
            model = role_model
        ld = self.cfg["load"]
        if self.client.load(model, ld["context_length"], ld["ttl"], ld["parallel"], ld["unload_others"]):
            if notes is not None:
                notes.append(f"loaded {model}")
        return model

    def _budget(self, model: str) -> int:
        ctx = self.client.loaded_context(model, self.cfg["load"]["context_length"])
        return max(1000, min(self.cfg["chunking"]["chunk_tokens"],
                             ctx - self.cfg["generation"]["max_tokens"] - PROMPT_OVERHEAD))

    # --- low level call --------------------------------------------------
    def _call(self, model: str, system: str, user: str, acc: dict,
              json_schema: dict | None = None) -> str:
        gen = self.cfg["generation"]
        res: ChatResult = self.client.chat(
            model,
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=gen["temperature"], max_tokens=gen["max_tokens"],
            json_schema=json_schema, thinking=gen["thinking"],
        )
        with self._lock:
            acc["prompt_tokens"] += res.prompt_tokens
            acc["completion_tokens"] += res.completion_tokens
            acc["calls"] += 1
        return res.content

    def _fill(self, template: str, spec: TaskSpec, instruction: str, content: str, params: dict) -> str:
        d = _SafeDict(spec.defaults)
        d.update(params)
        d["instruction"] = instruction
        d["content"] = content
        return template.format_map(d)

    def _pmap(self, fn, items: list) -> list:
        n = max(1, min(self.cfg["chunking"]["max_parallel"], len(items)))
        if n == 1:
            return [fn(x) for x in items]
        with ThreadPoolExecutor(max_workers=n) as ex:
            return list(ex.map(fn, items))

    # --- text processing per mode ---------------------------------------
    def _process_text(self, model: str, spec: TaskSpec, instruction: str, text: str,
                      params: dict, mode: str, budget: int, acc: dict, notes: list[str]) -> str:
        chunks = split_text(text, budget, self.cfg["chunking"]["overlap_tokens"])

        def run_chunk(chunk: str) -> str:
            return self._call(model, spec.system, self._fill(spec.user, spec, instruction, chunk, params),
                              acc, spec.json_schema)

        if len(chunks) == 1 and mode != "per_chunk":
            return run_chunk(chunks[0])
        if mode == "single":
            notes.append(f"input exceeds budget ({budget} tok); truncated to the first chunk")
            return run_chunk(chunks[0])
        if mode == "per_chunk":
            return "\n".join(self._pmap(run_chunk, chunks))

        # map_reduce (and auto that did not fit)
        notes.append(f"map_reduce over {len(chunks)} chunks")
        partials = self._pmap(run_chunk, chunks)
        reduce_tpl = spec.reduce or TASKS["ask"].reduce
        while len(partials) > 1:
            batches: list[list[str]] = []
            cur: list[str] = []
            cur_tok = 0
            for p in partials:
                t = count_tokens(p) + 10
                if cur and cur_tok + t > budget:
                    batches.append(cur)
                    cur, cur_tok = [], 0
                cur.append(p)
                cur_tok += t
            if cur:
                batches.append(cur)

            def reduce_batch(batch: list[str]) -> str:
                joined = "\n\n---\n\n".join(f"[part {i + 1}]\n{p}" for i, p in enumerate(batch))
                return self._call(model, MERGE_SYSTEM,
                                  self._fill(reduce_tpl, spec, instruction, joined, params), acc)

            if len(batches) == 1:
                return reduce_batch(batches[0])
            partials = self._pmap(reduce_batch, batches)
        return partials[0]

    # --- entry point -----------------------------------------------------
    def run(self, task: str, instruction: str = "", files: list[str] | tuple = (), text: str = "",
            model: str | None = None, strict: bool = False, params: dict | None = None,
            output: str | None = None, in_place: bool = False) -> RunResult:
        spec = get_task(task)
        params = params or {}
        notes: list[str] = []
        acc = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
        t0 = time.time()

        items: list[tuple[str, str]] = []
        if files:
            items, skip_notes = read_inputs(list(files), self.cwd, self.cfg["chunking"]["max_file_bytes"])
            notes += skip_notes
        if text:
            items.append(("<inline>", text))
        if not items and not instruction:
            raise ValueError("Nothing to do: give an instruction, files or text.")

        model = self.resolve_model(spec, model, strict, notes)
        budget = self._budget(model)
        files_written: list[str] = []
        out_dir = self.cwd / self.cfg["output"]["dir"]

        if spec.mode == "per_file":
            if not items:
                raise ValueError(f"Task '{task}' needs files or text.")

            def one(item: tuple[str, str]):
                path, content = item
                out = self._process_text(model, spec, instruction, content, params,
                                         spec.sub_mode, budget, acc, notes)
                return path, out

            results = self._pmap(one, items)
            if spec.json_schema:
                rows = []
                for path, out in results:
                    data = _parse_json(out) or {"raw": out}
                    rows.append({"file": path, **data} if isinstance(data, dict) else {"file": path, "raw": out})
                result_text = json.dumps(rows, ensure_ascii=False, indent=2)
            elif spec.output_ext == "":
                lines = []
                for path, out in results:
                    new_content = strip_fences(out)
                    if path == "<inline>":
                        lines.append(new_content)
                        continue
                    target = (self.cwd / path) if in_place else (out_dir / path)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(new_content, encoding="utf-8")
                    files_written.append(str(target))
                    lines.append(f"- {path} -> {target}")
                result_text = "\n".join(lines) if lines else ""
            else:
                result_text = "\n\n".join(f"### {path}\n{out}" for path, out in results)
        else:
            if len(items) == 1:
                content = items[0][1]
            else:
                content = "\n\n".join(f"### FILE: {path}\n{body}" for path, body in items)
            result_text = self._process_text(model, spec, instruction, content, params,
                                             spec.mode, budget, acc, notes)
            if spec.output_ext == "":
                result_text = strip_fences(result_text)

        output_path: str | None = None
        if output:
            target = Path(output)
            if not target.is_absolute():
                target = self.cwd / target
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(result_text, encoding="utf-8")
            output_path = str(target)
        elif len(result_text) > self.cfg["output"]["inline_limit"] and not files_written:
            ext = spec.output_ext or "txt"
            stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            target = out_dir / f"{stamp}_{task}.{ext}"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(result_text, encoding="utf-8")
            output_path = str(target)

        result = RunResult(
            task=task, model=model, text=result_text, output_path=output_path,
            files_written=files_written, prompt_tokens=acc["prompt_tokens"],
            completion_tokens=acc["completion_tokens"], elapsed=time.time() - t0,
            calls=acc["calls"], notes=notes,
        )
        self._log(result, instruction)
        return result

    # --- logging ---------------------------------------------------------
    def _log_path(self) -> Path:
        return Path(self.cfg["output"]["log"]).expanduser()

    def _log(self, r: RunResult, instruction: str) -> None:
        try:
            p = self._log_path()
            p.parent.mkdir(parents=True, exist_ok=True)
            entry = {
                "ts": _dt.datetime.now().isoformat(timespec="seconds"),
                "cwd": str(self.cwd), "task": r.task, "model": r.model,
                "instruction": instruction[:200], "calls": r.calls,
                "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens,
                "elapsed": round(r.elapsed, 1), "output_path": r.output_path,
                "files_written": len(r.files_written),
            }
            with open(p, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def stats(self) -> dict:
        p = self._log_path()
        totals = {"runs": 0, "calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "elapsed": 0.0,
                  "by_task": {}, "by_model": {}}
        if not p.is_file():
            return totals
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            totals["runs"] += 1
            totals["calls"] += e.get("calls", 0)
            totals["prompt_tokens"] += e.get("prompt_tokens", 0)
            totals["completion_tokens"] += e.get("completion_tokens", 0)
            totals["elapsed"] += e.get("elapsed", 0)
            for key, bucket in (("task", "by_task"), ("model", "by_model")):
                b = totals[bucket].setdefault(e.get(key, "?"), {"runs": 0, "prompt_tokens": 0, "completion_tokens": 0})
                b["runs"] += 1
                b["prompt_tokens"] += e.get("prompt_tokens", 0)
                b["completion_tokens"] += e.get("completion_tokens", 0)
        return totals
