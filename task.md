# Task: lmagent (BigSmall)

System that manages local LM Studio models (Qwen, Gemma, Nemotron) and delegates simple but token-heavy
tasks to them, so the expensive cloud model (Claude Code) only sees compact results.

Repo: https://github.com/ozymandi/BigSmall

## Decisions (designer, 2026-10-07)

- Both interfaces: MCP server inside Claude Code and a CLI, on one shared Python core.
- Language: Python 3.11.
- Local models may write files to disk (`.lmagent/out/` by default, `--in-place` to overwrite sources).
- Task list approved: ask, summarize, translate, extract, classify, rewrite, explain_diff, generate.

## Environment constraints

- RTX 5090, 32 GB VRAM: only one ~20 GB model loaded at a time, switching costs 10-20 s.
  Hence `load.policy: prefer_loaded` as default.
- LM Studio server on `localhost:1234`, `lms` CLI at `~/.lmstudio/bin`.
- Models downloaded: qwen/qwen3.8-27b, google/gemma-4-31b, gemma-4-31b-unc, nvidia/nemotron-3-nano-omni,
  nomic embed.

## Status

### Phase 1: core + CLI. Done 2026-10-07
- client, config, chunker, 8 task templates, runner (auto/single/per_chunk/per_file/map_reduce), CLI.
- Verified live against Qwen: ask, summarize (dir of source), classify (JSON schema), rewrite (file output).

### Phase 2: MCP server. Done 2026-10-07
- `mcp_server.py` with lm_delegate, lm_summarize_files, lm_batch, lm_models, lm_tasks.
- Registered in Claude Code user scope.

### Phase 3: hardening. Done 2026-10-07
- Retries in the client: timeouts, connection errors, 5xx, and "model not loaded" (auto reload after TTL).
- `load.policy: smart` (new default): switch to the role model only when the input is at least
  `switch_min_tokens` (40k). Load time is measured and logged.
- `lmagent stats --days N --by day|task|model|cwd` report.
- Chunk planner rewritten from measurements:
  - model tokenizers count 1.1x (code) to 1.45x (logs) more than tiktoken -> `chunking.token_safety: 1.5`;
  - LM Studio's loaded context is one shared budget for all concurrent sequences (prompt + generated),
    max_tokens is only a cap -> workers and chunk size are derived from ctx, slots and the task's
    `output_ratio` (translate/rewrite 1.2, summarize 0.15, classify 0.05, default 0.3);
  - if a chunk still overflows, it is split in half and retried.
- `lms` subprocess output decoded as UTF-8 (progress bars crashed cp1252 decoding on Windows).
- Reasoning off: LM Studio ignores `chat_template_kwargs.enable_thinking`; top-level
  `reasoning_effort: "none"` works for both Qwen and Nemotron (measured: 34 -> 2 completion tokens).
  Without it Nemotron spent 20+ minutes reasoning over a 168k-token log.
- Verified: 45k-token log -> automatic switch Qwen -> Nemotron (22 s load), 2 workers, 5 chunks + reduce,
  42 s total, 68k tokens handled locally.

### Phase 4: search + subagent. Done 2026-10-07
- `lmagent/index.py`: incremental embedding index in `.lmagent/index/` (nomic, 400-token chunks with
  line ranges, only changed files re-embedded, deleted files dropped). 22 files / 64 chunks in 4 s.
- CLI `lmagent index`, `lmagent search QUERY -k N [--files]`; MCP `lm_index`, `lm_search` (refreshes
  the index before each query).
- Subagent `~/.claude/agents/local-worker.md` (Haiku, only lm_* tools + Glob). Not yet exercised from a
  live session: agents are loaded at session start, so it needs a new Claude Code session.
- Two bugs found while testing through MCP:
  - importing numpy lazily inside a tool call deadlocked the server (C extension load in the event-loop
    thread on Windows); the index module is now imported at server startup;
  - the `lms` CLI is gone: all load/unload goes through `POST /api/v1/models/load|unload`
    (keys `model`, `context_length`, `ttl_seconds`, `parallel`), discovery via `GET /api/v1/models`.
- Verified through MCP stdio: lm_search 3.4 s with embed-model load, lm_delegate 3.4 s.

## Next step

1. New Claude Code session on another project: try `lm_search` + `lm_delegate` directly and through the
   `local-worker` subagent; collect which tasks are worth delegating and where prompts need tuning.
2. Decide on the `bulk` role model (Nemotron takes 26 GB and 32k context; a 8-14B model or Qwen itself
   may serve better on this GPU).

## Open questions

- None at the moment.
