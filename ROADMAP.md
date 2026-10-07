# Roadmap

Status of phases 1-4: done, see `task.md`. Items below are the next candidates, in the recommended
order. Estimates are in hours. Tick an item when it is done and note the date.

## 1. Live trial on a real project (1-2 h). Done 2026-10-07, findings in `task.md` (Phase 5)

In a fresh Claude Code session on another project:

- [x] `lm_models` and `lm_tasks` respond (MCP server `lmagent` connected, user scope).
- [x] `lm_search` on a question like "where is X handled", check that hits point to the right files.
- [x] `lm_delegate` with `task=summarize` on a big log or directory.
- [x] `lm_delegate` with `task=extract` returning JSON.
- [x] `lm_delegate` with `task=rewrite` on 2-3 files, first to `.lmagent/out/`, then `in_place`.
- [x] Same jobs through the `local-worker` subagent (`Agent` tool, `subagent_type: local-worker`), with the
      project directory named in the prompt so it can pass `cwd`.
- [x] Record for each: was the result usable as-is, how many tokens stayed local (`offloaded_tokens`),
      which prompts need tuning. Write findings into `task.md`.

## 2. Usage rule for Claude (0.5 h). Done 2026-10-07

- [x] Add a short rule to the global `~/.claude/CLAUDE.md`: when to delegate (files or logs over ~5k tokens,
      many files, mechanical bulk edits, translation, "where is X" questions) and when not to (small reads,
      anything needing judgement on the final code). Without it Claude will not pick the tools on its own.
- [ ] Optionally the same rule as a project `CLAUDE.md` snippet for repos where it matters most.

## 3. Tests (2 h)

- [ ] pytest for `chunker.split_text` (budget, overlap, giant lines), `index.split_lines` (line ranges).
- [ ] Planner (`Runner._plan`) with a fake client: Qwen 89k / 32k contexts, each task's `output_ratio`,
      worker reduction only when it buys a bigger chunk.
- [ ] `Index.update` incremental logic with a fake embedder: unchanged, changed, deleted, excluded files.
- [ ] `_parse_json` / `strip_fences`, config layering.

## 4. Answer on top of search (1-2 h)

- [ ] `lmagent search QUERY --ask "question"` and `ask` parameter on `lm_search`: top-k chunks go to the local
      LLM with the question, only the answer plus the file/line citations come back.
- [ ] Reuse the `ask` task; cap the context by the planner's budget.

## 5. Guardrails for `rewrite --in-place` (1-2 h). Done 2026-10-07

- [x] Keep a copy of the original under `.lmagent/backup/<timestamp>/<path>` before overwriting.
- [x] Return a unified diff summary (lines added/removed per file) instead of just the path.
- [x] Syntax check before writing: `py_compile` for `.py`, `json.loads` for `.json`, YAML parse for `.yaml`;
      on failure keep the original and report.
- [x] Fixes from the trial: keep the source line endings (LF was becoming CRLF), strip an echoed
      `<content>` wrapper, report unchanged files, summarize reduce prompt merges duplicates.

## 6. Decide the `text` role model (0.5-1 h)

- [ ] Run `translate` and `summarize` on the same inputs with Gemma 4 31B and Qwen, compare quality and time.
- [ ] If the difference is small, set `text` to Qwen too: then no model switching happens at all.

## Later, not scheduled

- Progress notifications for long MCP runs.
- Friendly error when the LM Studio server is down, with the command to start it.
- Per-project `lmagent.yaml` examples (e.g. smaller chunks for log-heavy repos).
