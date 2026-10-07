from __future__ import annotations

import datetime as _dt
import difflib
import json
import re
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from .chunker import count_tokens, read_inputs, split_text
from .client import ChatResult, LMStudioClient, LMStudioError
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
    load_seconds: float = 0.0
    notes: list[str] = field(default_factory=list)

    def stats_line(self) -> str:
        return (f"[{self.task} | {self.model} | {self.calls} call(s) | "
                f"in {self.prompt_tokens} / out {self.completion_tokens} tok | {self.elapsed:.1f}s]")

    def to_dict(self) -> dict:
        return asdict(self)


def strip_fences(text: str) -> str:
    m = FENCE_RE.match(text.strip())
    return m.group(1) if m else text


def clean_output(text: str) -> str:
    """Model output that should be a whole file: drop code fences and an echoed <content> wrapper."""
    text = strip_fences(text)
    s = text.strip()
    if s.startswith("<content>"):
        s = s[len("<content>"):]
        if s.rstrip().endswith("</content>"):
            s = s.rstrip()[: -len("</content>")]
        text = s.strip("\r\n")
    return text


def check_syntax(path: str, text: str) -> str | None:
    """Parse the new content with the parser its extension implies. Returns an error string or None."""
    ext = Path(path).suffix.lower()
    try:
        if ext == ".py":
            compile(text, path, "exec")
        elif ext == ".json":
            json.loads(text)
        elif ext in (".yaml", ".yml"):
            yaml.safe_load(text)
    except (SyntaxError, ValueError, yaml.YAMLError) as e:
        msg = str(e).splitlines()[0] if str(e) else ""
        return f"{type(e).__name__}: {msg[:160]}"
    return None


def diff_stat(old: str, new: str) -> tuple[int, int]:
    """Lines added and removed between two texts (line endings ignored)."""
    added = removed = 0
    for line in difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm="", n=0):
        if line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    return added, removed


def write_like(target: Path, text: str, source: Path | None) -> None:
    """Write text with the line endings and trailing-newline convention of `source` (default LF).
    Path.write_text would turn LF into CRLF on Windows and change every line of the file."""
    eol, trailing = "\n", True
    if source is not None and source.is_file():
        raw = source.read_bytes()
        if b"\r\n" in raw:
            eol = "\r\n"
        trailing = raw.endswith(b"\n")
    text = text.replace("\r\n", "\n")
    if trailing and not text.endswith("\n"):
        text += "\n"
    elif not trailing:
        text = text.rstrip("\n")
    with open(target, "w", encoding="utf-8", newline="") as f:
        f.write(text.replace("\n", eol))


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
        self.client = client or LMStudioClient.from_config(cfg)
        self.client.on_not_loaded = self._reload
        self._lock = threading.Lock()
        self._load_seconds = 0.0
        self._workers = int(cfg["chunking"]["max_parallel"])
        self._progress = None
        self._prog_done = 0
        self._prog_total = 0

    # --- progress --------------------------------------------------------
    def _add_total(self, n: int) -> None:
        with self._lock:
            self._prog_total += n

    def _tick(self, message: str) -> None:
        """One model call finished. Calls on_progress(done, total, message); total is a running estimate."""
        with self._lock:
            self._prog_done += 1
            done, total = self._prog_done, max(self._prog_total, self._prog_done)
        cb = self._progress
        if cb is None:
            return
        try:
            cb(done, total, message)
        except Exception:  # a broken progress sink must not fail the job
            pass

    def _reload(self, model: str) -> None:
        """Re-load a model that LM Studio dropped (TTL expiry, manual unload) mid-run."""
        ld = self.cfg["load"]
        with self._lock:
            if self.client.load(model, ld["context_length"], ld["ttl"], ld["parallel"], ld["unload_others"]):
                self._load_seconds += self.client.last_load_seconds or 0.0

    # --- model handling --------------------------------------------------
    def resolve_model(self, spec: TaskSpec, override: str | None = None,
                      strict: bool = False, notes: list[str] | None = None,
                      input_tokens: int = 0) -> str:
        """Pick a model. Policies: strict (always role model), prefer_loaded (never switch),
        smart (switch to the role model only when the input is big enough to justify a reload)."""
        notes = notes if notes is not None else []
        ld = self.cfg["load"]
        role_model = self.cfg["models"][spec.role]
        policy = "strict" if strict else ld.get("policy", "smart")
        if override:
            model = override
        elif policy == "strict":
            model = role_model
        else:
            loaded = self.client.loaded_llms()
            if not loaded or any(m["id"] == role_model for m in loaded):
                model = role_model
            else:
                threshold = int(ld.get("switch_min_tokens", 40000))
                if policy == "smart" and input_tokens >= threshold:
                    model = role_model
                    notes.append(f"switching to role model {role_model}: input {input_tokens} tok >= {threshold}")
                else:
                    model = loaded[0]["id"]
                    notes.append(f"using already loaded {model} instead of role model {role_model}")
        if self.client.load(model, ld["context_length"], ld["ttl"], ld["parallel"], ld["unload_others"]):
            secs = self.client.last_load_seconds or 0.0
            self._load_seconds += secs
            notes.append(f"loaded {model} in {secs:.0f}s")
        return model

    def _plan(self, model: str, spec: TaskSpec, notes: list[str]) -> int:
        """Decide workers and chunk budget for this model and task.

        Measured behaviour of LM Studio: the loaded context is one shared budget for all concurrent
        sequences (prompt + generated tokens), max_tokens is only a cap. So per worker we need
        chunk*safety + output + overhead <= ctx / workers, where output is the larger of
        generation.reserve_tokens and chunk*safety*spec.output_ratio, and output must also fit max_tokens.
        Returns the chunk budget in tiktoken tokens; sets self._workers.
        """
        ch = self.cfg["chunking"]
        gen = self.cfg["generation"]
        ctx = self.client.loaded_context(model, self.cfg["load"]["context_length"])
        slots = self.client.loaded_parallel(model, self.cfg["load"]["parallel"])
        safety = float(ch.get("token_safety", 1.5))
        ratio = float(spec.output_ratio)
        reserve = int(gen.get("reserve_tokens", 2048))
        max_tokens = int(gen["max_tokens"])
        min_chunk = int(ch.get("min_chunk_tokens", 6000))

        def fits_for(workers: int) -> int:
            room = ctx // workers - PROMPT_OVERHEAD
            c = min((room - reserve) / safety, room / (safety * (1 + ratio)))
            if ratio > 0:
                c = min(c, max_tokens / (safety * ratio))
            return int(c)

        workers = max(1, min(int(ch["max_parallel"]), slots))
        while workers > 1 and fits_for(workers) < min_chunk and fits_for(workers - 1) > fits_for(workers):
            workers -= 1
        fits = fits_for(workers)
        self._workers = workers
        budget = max(500, min(int(ch["chunk_tokens"]), fits))
        if workers < slots or budget < int(ch["chunk_tokens"]):
            notes.append(f"plan: ctx {ctx}, {workers} worker(s), chunk {budget} tok (output ratio {ratio})")
        return budget

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
        n = max(1, min(self._workers, len(items)))
        if n == 1:
            return [fn(x) for x in items]
        with ThreadPoolExecutor(max_workers=n) as ex:
            return list(ex.map(fn, items))

    # --- text processing per mode ---------------------------------------
    def _process_text(self, model: str, spec: TaskSpec, instruction: str, text: str,
                      params: dict, mode: str, budget: int, acc: dict, notes: list[str],
                      label: str = "") -> str:
        chunks = split_text(text, budget, self.cfg["chunking"]["overlap_tokens"])
        # per-file callers already counted one unit for the file
        self._add_total(len(chunks) - (1 if label else 0))

        def run_chunk(chunk: str) -> str:
            try:
                out = self._call(model, spec.system, self._fill(spec.user, spec, instruction, chunk, params),
                                 acc, spec.json_schema)
                self._tick(label or f"chunk ({len(chunks)} total)")
                return out
            except LMStudioError as e:
                if "context size" not in str(e).lower() or count_tokens(chunk) < 500:
                    raise
            # The model's tokenizer counted more than tiktoken did: split the chunk in half and retry.
            half = max(250, count_tokens(chunk) // 2)
            parts = split_text(chunk, half, 0)
            with self._lock:
                notes.append(f"context exceeded, re-split a chunk into {len(parts)} parts")
            return "\n".join(run_chunk(p) for p in parts)

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
            self._add_total(len(batches))

            def reduce_batch(batch: list[str]) -> str:
                joined = "\n\n---\n\n".join(f"[part {i + 1}]\n{p}" for i, p in enumerate(batch))
                out = self._call(model, MERGE_SYSTEM,
                                 self._fill(reduce_tpl, spec, instruction, joined, params), acc)
                self._tick(f"reduce {len(batch)} parts" + (f" ({label})" if label else ""))
                return out

            if len(batches) == 1:
                return reduce_batch(batches[0])
            partials = self._pmap(reduce_batch, batches)
        return partials[0]

    # --- entry point -----------------------------------------------------
    def run(self, task: str, instruction: str = "", files: list[str] | tuple = (), text: str = "",
            model: str | None = None, strict: bool = False, params: dict | None = None,
            output: str | None = None, in_place: bool = False,
            on_progress=None) -> RunResult:
        """on_progress(done, total, message) is called after every model call; total is an estimate
        that grows as chunks and reduce rounds become known."""
        spec = get_task(task)
        params = params or {}
        notes: list[str] = []
        self._progress = on_progress
        self._prog_done = 0
        self._prog_total = 0
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

        self._load_seconds = 0.0
        input_tokens = sum(count_tokens(body) for _, body in items)
        model = self.resolve_model(spec, model, strict, notes, input_tokens)
        budget = self._plan(model, spec, notes)
        files_written: list[str] = []
        out_dir = self.cwd / self.cfg["output"]["dir"]

        if spec.mode == "per_file":
            if not items:
                raise ValueError(f"Task '{task}' needs files or text.")
            self._add_total(len(items))

            def one(item: tuple[str, str]):
                path, content = item
                out = self._process_text(model, spec, instruction, content, params,
                                         spec.sub_mode, budget, acc, notes, label=path)
                return path, out

            results = self._pmap(one, items)
            if spec.json_schema:
                rows = []
                for path, out in results:
                    data = _parse_json(out) or {"raw": out}
                    rows.append({"file": path, **data} if isinstance(data, dict) else {"file": path, "raw": out})
                result_text = json.dumps(rows, ensure_ascii=False, indent=2)
            elif spec.output_ext == "":
                originals = dict(items)
                backup_root = self.cwd / self.cfg["output"].get("backup_dir", ".lmagent/backup")
                stamp: str | None = None
                lines = []
                for path, out in results:
                    new_content = clean_output(out)
                    if path == "<inline>":
                        lines.append(new_content)
                        continue
                    rel = Path(path)
                    if rel.is_absolute():
                        rel = Path(*rel.parts[1:])
                    src = self.cwd / path
                    added, removed = diff_stat(originals.get(path, ""), new_content)
                    err = "empty output" if not new_content.strip() else check_syntax(path, new_content)
                    if in_place:
                        if err:
                            lines.append(f"- {path}: NOT written ({err}); original kept")
                            continue
                        if added == 0 and removed == 0:
                            lines.append(f"- {path}: unchanged")
                            continue
                        stamp = stamp or _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
                        bak = backup_root / stamp / rel
                        bak.parent.mkdir(parents=True, exist_ok=True)
                        if src.is_file():
                            shutil.copy2(src, bak)
                        target = src
                    else:
                        target = out_dir / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    write_like(target, new_content, src if src.is_file() else None)
                    files_written.append(str(target))
                    line = f"- {path} -> {target} (+{added} -{removed})"
                    if err:
                        line += f"; CHECK FAILED: {err}"
                    lines.append(line)
                if stamp:
                    notes.append(f"originals backed up in {backup_root / stamp}")
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
                result_text = clean_output(result_text)

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
            calls=acc["calls"], load_seconds=round(self._load_seconds, 1), notes=notes,
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
                "elapsed": round(r.elapsed, 1), "load_s": r.load_seconds, "output_path": r.output_path,
                "files_written": len(r.files_written),
            }
            with open(p, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def log_entries(self, days: int | None = None) -> list[dict]:
        p = self._log_path()
        if not p.is_file():
            return []
        cutoff = None
        if days:
            cutoff = (_dt.datetime.now() - _dt.timedelta(days=days)).isoformat(timespec="seconds")
        out = []
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if cutoff and e.get("ts", "") < cutoff:
                continue
            out.append(e)
        return out

    def stats(self, days: int | None = None, by: str = "day") -> dict:
        """Aggregate the run log. by: day | task | model | cwd."""
        groups: dict[str, dict] = {}
        total = {"runs": 0, "calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "elapsed": 0.0, "load_s": 0.0}
        for e in self.log_entries(days):
            key = e.get("ts", "")[:10] if by == "day" else str(e.get(by, "?"))
            g = groups.setdefault(key, {k: 0 for k in total})
            for bucket in (g, total):
                bucket["runs"] += 1
                bucket["calls"] += e.get("calls", 0)
                bucket["prompt_tokens"] += e.get("prompt_tokens", 0)
                bucket["completion_tokens"] += e.get("completion_tokens", 0)
                bucket["elapsed"] += e.get("elapsed", 0)
                bucket["load_s"] += e.get("load_s", 0) or 0
        return {"by": by, "days": days, "groups": dict(sorted(groups.items())), "total": total}
