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

### Phase 5: live trial on PlasticityMCP (411 ts files). Done 2026-10-07
Run from the lmagent session itself with `cwd=E:\Projects\PlasticityMCP` (MCP in user scope, so no new
session was needed). Trial files were reverted with `git checkout` afterwards.

| Job | Input | Time | Usable as-is | Notes |
|---|---|---|---|---|
| `lm_models`, `lm_tasks` | - | <1 s | yes | Qwen loaded, ctx 89k; roles code/bulk=Qwen, text=Gemma |
| `lm_search` "where is the websocket connection opened" | index 558 files / 16,299 chunks, 68 MB | first build ~1 min | partly | all 6 hits were markdown (docs/, README, task.md); the real code (`client.ts`, `cdp.ts`, `tools/bridge.ts`) did not appear |
| `lm_search` "zod schema validating fillet radius" | - | 2 s | partly | `solids.ts` only at #4 and with the handler's line range, not `FilletArgs` |
| `summarize` task.md | 43.9k tok in, 5.0k out | 42.7 s, 5 calls, 4 chunks | yes | accurate; reduce step repeated 2 bullets (Status, Tool count) at the end |
| `extract` tools from curves.ts + solids.ts | 13.7k in, 3.3k out | 25.9 s, 3 calls | yes | 28/28 names correct, required params correct on spot check |
| `rewrite` JSDoc on 3 files -> `.lmagent/out/` | 2.5k in, 2.3k out | 11.6 s | no | comments good, but see bugs below |
| `rewrite` same, `in_place` | 2.5k in | 11 s | no | same bugs; reverted |
| `local-worker` subagent (Haiku): locate `native_launch`, explain the debug-port trick | 6 tool calls, 21.5k subagent tokens | 57 s | yes | correct file, functions and mechanism; line numbers off by 3-8 (e.g. `unlockMainProcess` cited 142-196, real 139-196); it skipped the required closing line "model used and offloaded tokens" |

Offloaded in the trial: about 77k input tokens (plus the embedding index). `lmagent stats` for the day: 165k.

Bugs and tuning found:
- `rewrite`: `Path.write_text` translates LF to CRLF on Windows, so every line of every file changed
  (`git diff` hid it because of `core.autocrlf=input`, the files on disk still changed). Must preserve the
  original line endings (`newline=""` plus detecting the source EOL).
- `rewrite`: one output came back wrapped in `<content>...</content>` (the template's own tag echoed);
  `strip_fences` does not remove it. Nondeterministic: the in-place run of the same job was clean.
- `rewrite`: the model ignored "exported only" and commented module-level consts in `smoke.ts`.
- `summarize` reduce prompt needs an explicit "merge duplicates, no repeated bullets".
- `lm_search`: prose files outrank code for "where is X" questions. Candidates: a `kind=code|docs|all` filter
  (by extension), an exclude list in `lmagent.yaml` (here `plasticity-fork/` is vendored and triples the
  index), maybe bigger chunks for code.
- Item 5 of the roadmap (guardrails) is confirmed necessary before `in_place` is used for real.

### Phase 6: rewrite guardrails (roadmap item 5). Done 2026-10-07
- `runner.py`: `clean_output` (fences + echoed `<content>`), `write_like` (keeps the source EOL and
  trailing newline), `diff_stat`, `check_syntax` (py/json/yaml), backup to `.lmagent/backup/<stamp>/`
  before in-place writes, unchanged files skipped, failed checks keep the original.
- Verified live on scratch files: CRLF file stayed CRLF, backup created, `cfg.json` (model returned
  non-JSON) reported `NOT written`, second run reported `unchanged`.
- `summarize` reduce prompt: "each fact appears once".

### Phase 7: answer on top of search (roadmap item 4). Done 2026-10-07
- `lmagent/answer.py`: top hits (up to `index.answer_k`, capped by the planner budget) + question -> one
  `ask` call; returns answer with `file:start-end` citations and the hit list without text.
- CLI `lmagent search Q --ask [QUESTION] --kind code|docs`; MCP `lm_search(kind=, ask=)`.
- `Index.search(kind=)` masks chunks by extension; `index.exclude` supports `dir/` and `path/*`.
- Verified on this repo: "how are retries handled" -> correct answer in 5.8 s, 5.1k tokens local,
  claims checked against `client.py`. The MCP process must be restarted for the new parameters.

### Phase 8: tests (roadmap item 3). Done 2026-10-07
- `tests/` with a fake LM Studio client (no server needed), 31 tests, 0.4 s: chunker, planner (89k/32k,
  output ratios, worker reduction), incremental index (unchanged/changed/deleted/excluded, kind filter),
  output cleaning, config layering, rewrite guardrails end to end. `pip install -e .[dev]`, `python -m pytest`.

### Phase 9: `text` role model (roadmap item 6). Done 2026-10-07
Same inputs for both models: translate README excerpt (870 tok) to Ukrainian, summarize task.md (2.4k tok).

| Job | Qwen 3.8 27B | Gemma 4 31B |
|---|---|---|
| translate | 12.4 s | 37.2 s (+13 s model switch) |
| summarize | 10.3 s | 13.2 s |
| translation quality | usable; 2-3 slips (one verb form, "embedded" rendered as "інтегруються") | slightly more natural, correct terms |
| summary quality | equal (more detail per phase) | equal (more compact) |

Decision (roadmap rule "if the difference is small, set text to Qwen"): `models.text` = Qwen. No model switch
happens any more in the default setup; Gemma stays downloaded and can be forced with `-m google/gemma-4-31b`
or `model=` in `lm_delegate` when translation quality matters more than speed.

Decided 2026-10-07: `load.context_length` default raised to 89344 (designer).

### Phase 10: "Later" items. Done 2026-10-07
- Progress: `Runner.run(on_progress=cb)` ticks after every model call with a growing total estimate; MCP
  tools that call the model are now `async` and run in a worker thread (sync tools blocked the whole
  server, so notifications could never get through) and forward ticks via `ctx.report_progress`; CLI
  shows `[done/total] message` on a TTY.
- `LMStudioDown`: connection errors (refused, connect timeout) raise at once with the start instructions
  instead of 2 retries with backoff and a stack of httpx text.
- `examples/` with three `lmagent.yaml` variants and a README.

### Phase 11: verification after session restart. Done 2026-10-07
- New Claude Code session: the MCP schema now exposes `lm_search(kind=, ask=)`, async tools respond.
- `lm_models`: all LLMs unloaded at start, only nomic embed loaded (TTL expired).
- `lm_search(kind=code, ask="how does the client retry / reload")` on this repo: Qwen loaded in 8 s,
  answer in 6 s, 4.9k tokens in / 0.5k out locally; 12 hits, all code/yaml (no markdown). Every claim
  (RETRYABLE_STATUS set, NOT_LOADED_MARKERS, retries 2 / delay 2 s doubled, `on_not_loaded -> Runner._reload`)
  checked against `client.py`, `runner.py`, `default_config.yaml`; line citations within a few lines of the real ones.

## Next step

Roadmap fully done and verified in a fresh session. Nothing scheduled.

## Open questions

- None at the moment.
