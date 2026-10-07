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

### Phase 3: hardening. Not started
- Token-saving report (`lmagent stats` exists, needs a per-day view).
- Retry on transient LM Studio errors, better handling when the server is busy.
- Smarter model switching (estimate load time vs. task size).

### Phase 4, optional
- Embedding-based file search (`embed` role is configured, `client.embed` exists).
- A `local-worker` subagent definition for Claude Code.

## Next step

Use the MCP tools from a real Claude Code session on another project and collect feedback on
which tasks are worth delegating and where the prompts need tuning.

## Open questions

- None at the moment.
